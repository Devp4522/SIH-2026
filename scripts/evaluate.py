"""
Accuracy assessment on the held-out split (default: test city, full 512x512 patches).

    python scripts/evaluate.py --ckpt runs/edsr_baseline/best.pt --tta
    python scripts/evaluate.py --ckpt runs/edsr_baseline/best.pt runs/edsr_gan/final.pt   # compare
    python scripts/evaluate.py --ckpt runs/edsr_baseline/best.pt --split val

Writes to <run_dir>/eval_<split>/:
    per_image.csv     every metric for every patch, for bicubic and each model
    summary.md        table with 95% bootstrap CIs, per-band PSNR, per-scene breakdown
    summary.json      same numbers, machine-readable
    calibration.csv   does the uncertainty map actually track the error? (with --tta)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import _common  # noqa: F401
from srm.config import _wrap, apply_overrides, to_plain
from srm.data import (BAND_NAMES, SRPatchDataset, build_degradation, build_normalizer,
                      make_splits, resolve_patch_paths)
from srm.metrics import bootstrap_ci, per_image_metrics
from srm.models import build_model
from srm.trainer import get_device, make_loader
from srm.uncertainty import consistency_map, predict


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = _wrap(ck["cfg"])
    model = build_model(cfg.model, int(cfg.data.scale)).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg, ck


def get_split_df(cfg, run_dir, split):
    p = os.path.join(run_dir, "splits", f"{split}.csv")
    if os.path.exists(p):   # use the exact split the model was trained with
        return resolve_patch_paths(pd.read_csv(p), cfg.data.data_root)
    tr, va, te = make_splits(cfg)
    return {"train": tr, "val": va, "test": te}[split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints to compare")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--tta", action="store_true", help="8x dihedral TTA: better accuracy + uncertainty map")
    ap.add_argument("--crop", type=int, default=0, help="0 = full 512 patches")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--offset", type=float, default=1000.0,
                    help="S2 L2A DN offset removed before SAM/ERGAS/NDVI (0 = old behaviour)")
    ap.add_argument("--set", nargs="*", default=[], help="e.g. data.data_root=/path/to/patches")
    a = ap.parse_args()

    device = get_device()
    models, names = [], []
    for p in a.ckpt:
        m, c, ck = load_model(p, device)
        models.append(m)
        names.append(f"{c.model.name}_{ck.get('stage', 'psnr')}:{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}")
    first_cfg = _wrap(torch.load(a.ckpt[0], map_location="cpu", weights_only=False)["cfg"])
    cfg = _wrap(apply_overrides(to_plain(first_cfg), a.set))
    run_dir = os.path.dirname(a.ckpt[0])
    out_dir = a.out or os.path.join(run_dir, f"eval_{a.split}")
    os.makedirs(out_dir, exist_ok=True)

    scale = int(cfg.data.scale)
    norm, deg = build_normalizer(cfg), build_degradation(cfg)
    df = get_split_df(cfg, run_dir, a.split)
    ds = SRPatchDataset(df, norm, deg, train=False, crop_size=a.crop or None)
    loader = make_loader(ds, a.batch_size, False, int(cfg.data.get("num_workers", 2)))
    print(f"[eval] {a.split}: {len(ds)} patches, cities {sorted(df.city.unique())}, TTA={a.tta}")

    rows = []
    calib = {"std": [], "abs_err": [], "consist": []}
    rng = np.random.default_rng(0)
    for batch in loader:
        lr, hr, mask = batch["lr"].to(device), batch["hr"].to(device), batch["mask"].to(device)
        meta = [{"city": c, "scene": s, "idx": int(i)} for c, s, i in
                zip(batch["city"], batch["scene"], batch["idx"])]

        preds = {"bicubic": F.interpolate(lr, scale_factor=scale, mode="bicubic",
                                          align_corners=False).clamp(0, 1)}
        for m, n in zip(models, names):
            sr, std = predict(m, lr, tta=a.tta)
            preds[n] = sr
            if std is not None and n == names[0]:
                err = (sr - hr).abs().mean(1, keepdim=True)
                u = std.mean(1, keepdim=True)
                cm = consistency_map(sr, lr, deg.sigma_eval, scale)
                sel = mask.bool()
                e, u_, c_ = err[sel].cpu().numpy(), u[sel].cpu().numpy(), cm[sel].cpu().numpy()
                k = rng.choice(e.size, size=min(20000, e.size), replace=False)
                calib["abs_err"].append(e[k]); calib["std"].append(u_[k]); calib["consist"].append(c_[k])

        for n, p in preds.items():
            for mi, ri in zip(meta, per_image_metrics(p, hr, mask, norm, scale, min_valid_frac=0.0, offset=a.offset)):
                rows.append({"method": n, **mi, **ri})

    res = pd.DataFrame(rows)
    res = res[res.valid_frac >= 0.05].reset_index(drop=True)  # skip near-empty (cloud) patches
    res.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)

    # ---------------- summary ----------------
    metrics = ["psnr", "ssim", "sam_deg", "ergas", "ndvi_mae"]
    better = {"psnr": "higher", "ssim": "higher", "sam_deg": "lower", "ergas": "lower", "ndvi_mae": "lower"}
    summary = {}
    for n, g in res.groupby("method", sort=False):
        summary[n] = {m: dict(zip(("mean", "ci_lo", "ci_hi"), bootstrap_ci(g[m]))) for m in metrics}
        summary[n]["per_band_psnr"] = {b: float(g[f"psnr_{b}"].mean()) for b in BAND_NAMES}
        summary[n]["n_patches"] = int(len(g))

    lines = [f"# Evaluation - {a.split} split ({', '.join(sorted(df.city.unique()))})", "",
             f"Patches: {summary['bicubic']['n_patches']} | TTA: {a.tta} | "
             f"scale x{scale} (synthetic 40 m -> 10 m, Wald protocol) | DN offset removed: {a.offset:g}", "",
             "| Method | " + " | ".join(f"{m} ({better[m]} better)" for m in metrics) + " |",
             "|---" * (len(metrics) + 1) + "|"]
    for n, s in summary.items():
        cells = [f"{s[m]['mean']:.4f} [{s[m]['ci_lo']:.4f}, {s[m]['ci_hi']:.4f}]" for m in metrics]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-band PSNR (dB)", "", "| Method | " + " | ".join(BAND_NAMES) + " |",
              "|---" * (len(BAND_NAMES) + 1) + "|"]
    for n, s in summary.items():
        lines.append(f"| {n} | " + " | ".join(f"{s['per_band_psnr'][b]:.3f}" for b in BAND_NAMES) + " |")

    # paired gain vs bicubic (same patches) with CI
    lines += ["", "## Paired gain over bicubic (same patches)", ""]
    bic = res[res.method == "bicubic"].set_index(["city", "scene", "idx"])
    for n in names:
        mdf = res[res.method == n].set_index(["city", "scene", "idx"])
        d = (mdf["psnr"] - bic.loc[mdf.index, "psnr"]).to_numpy()
        mean, lo, hi = bootstrap_ci(d)
        wins = float((d > 0).mean())
        summary[n]["psnr_gain_vs_bicubic"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "win_rate": wins}
        lines.append(f"- **{n}**: {mean:+.3f} dB [{lo:+.3f}, {hi:+.3f}], better than bicubic on "
                     f"{wins * 100:.1f}% of patches")

    lines += ["", "## Per-scene PSNR (dB)", ""]
    piv = res.pivot_table(index="scene", columns="method", values="psnr", aggfunc="mean")
    try:
        lines.append(piv.round(3).to_markdown())
    except ImportError:  # tabulate not installed
        lines += ["```", piv.round(3).to_string(), "```"]

    if calib["std"]:
        e, u, c = (np.concatenate(calib[k]) for k in ("abs_err", "std", "consist"))
        cdf = pd.DataFrame({"abs_err": e, "tta_std": u, "consistency": c})
        rho_u = cdf[["tta_std", "abs_err"]].corr(method="spearman").iloc[0, 1]
        rho_c = cdf[["consistency", "abs_err"]].corr(method="spearman").iloc[0, 1]
        cdf["std_decile"] = pd.qcut(cdf.tta_std.rank(method="first"), 10, labels=False)
        tab = cdf.groupby("std_decile").agg(mean_std=("tta_std", "mean"), mean_abs_err=("abs_err", "mean"),
                                            n=("abs_err", "size"))
        tab.to_csv(os.path.join(out_dir, "calibration.csv"))
        summary["uncertainty"] = {"spearman_tta_std_vs_abs_err": float(rho_u),
                                  "spearman_consistency_vs_abs_err": float(rho_c),
                                  "err_ratio_top_vs_bottom_decile":
                                      float(tab.mean_abs_err.iloc[-1] / max(tab.mean_abs_err.iloc[0], 1e-9))}
        lines += ["", "## Uncertainty vs actual error", "",
                  f"- Spearman rho (TTA std vs |error|): **{rho_u:.3f}**",
                  f"- Spearman rho (LR-consistency vs |error|): **{rho_c:.3f}**",
                  f"- Mean |error| in top-uncertainty decile is "
                  f"{summary['uncertainty']['err_ratio_top_vs_bottom_decile']:.1f}x the bottom decile",
                  "", "rho > 0.3 means the uncertainty map is a usable warning signal; ~0 means it is not."]

    lines += ["", "> Caveat: targets are 10 m Sentinel-2. These numbers prove the model recovers detail",
              "> lost between 40 m and 10 m. They do NOT prove <4 m accuracy; that needs real",
              "> high-resolution reference imagery (scripts/validate_reference.py)."]

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines))
    print(f"\n[eval] written to {out_dir}")


if __name__ == "__main__":
    main()
