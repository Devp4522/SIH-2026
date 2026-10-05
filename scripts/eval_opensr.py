"""
REAL high-resolution validation (Phase 2) on the opensr-test benchmark.

opensr-test (ESA OpenSR, IEEE GRSL 2024) pairs REAL Sentinel-2 L2A images with REAL
2.5 m reference imagery (x4), already co-registered and radiometrically harmonised:
    naip         62 images  USA, crops / forest / bare soil      (aerial, 2.5 m)
    spot          9 images  WorldStrat SPOT 6/7: urban / crops   (satellite, 2.5 m)
    spain_crops  28 images  crops / forest                       (aerial, 2.5 m)
    spain_urban  20 images  urban areas + roads                  (aerial, 2.5 m)
This is 10 m -> 2.5 m on real data: the direct test of the "<4 m" claim.

    python scripts/eval_opensr.py --ckpt runs/edsr_final/best.pt --tta
    # + official hallucination/omission metrics (needs: pip install opensr-test)
    python scripts/eval_opensr.py --ckpt runs/edsr_final/best.pt --tta --opensr-metrics

Reports, SR vs bicubic, per dataset:
    psnr / ssim         vs the real 2.5 m reference (per-image percentile-normalised)
    sam_deg             spectral angle vs reference
    hf_corr             correlation of HIGH-FREQUENCY detail with the reference.
                        >> bicubic means the model adds REAL detail, not just contrast.
    consistency_l1      |downsample(SR) - LR|: does the output still agree with what S2 measured?
    shift_px            geolocation shift of SR vs input (phase correlation, in 10 m pixels)
    uncertainty rho     does the TTA uncertainty map predict the real error?
    (--opensr-metrics)  official hallucination / omission / improvement rates (ND distance,
                        pixel level) - same setting as the published opensr-test ND table.
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import _common  # noqa: F401
from srm.config import _wrap
from srm.data import BAND_NAMES, RGB_IDX, blur_downsample, build_degradation, build_normalizer, gaussian_kernel1d
from srm.losses import spectral_angle, ssim_map
from srm.metrics import bootstrap_ci
from srm.models import build_model
from srm.trainer import get_device
from srm.uncertainty import guarded_predict

L2A_IDX = [1, 2, 3, 7]  # B02, B03, B04, B08 in opensr-test's 12-band L2A stack
DATASETS = ["naip", "spot", "spain_crops", "spain_urban"]
BORDER = 16  # HR pixels ignored at each edge (same as opensr-test default)


# ----------------------------------------------------------------------------- helpers
HF_URL = "https://huggingface.co/datasets/isp-uv-es/opensr-test/resolve/main/021/{0}/{0}.pkl"


def load_dataset(name, data_dir):
    """Download (once) and unpickle an opensr-test dataset. Only needs `requests`."""
    import pickle
    import requests
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{name}.pkl")
    if not os.path.exists(path):
        url = HF_URL.format(name)
        print(f"[download] {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total, done = int(r.headers.get("content-length", 0)), 0
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
        os.replace(path + ".part", path)
        print()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # pickle needs the package's own classes -> use the official loader
        import opensr_test
        return opensr_test.load(name, model_dir=data_dir)


def to_f32(a):
    return np.asarray(a, dtype=np.float32)


def detect_hr_band_order(lr, hr, n=8):
    """Find which HR channel matches B02/B03/B04/B08 by correlating the downsampled HR with LR."""
    idx = np.linspace(0, len(lr) - 1, min(n, len(lr))).astype(int)
    C = np.zeros((4, 4))
    for i in idx:
        l = torch.from_numpy(to_f32(lr[i]))
        h = torch.from_numpy(to_f32(hr[i]))
        h = F.adaptive_avg_pool2d(h[None], l.shape[-2:])[0]
        for a in range(4):
            for b in range(4):
                x, y = l[a].flatten(), h[b].flatten()
                x, y = x - x.mean(), y - y.mean()
                C[a, b] += float((x * y).sum() / (x.norm() * y.norm() + 1e-9))
    best = max(itertools.permutations(range(4)), key=lambda p: sum(C[a, p[a]] for a in range(4)))
    return list(best), C / len(idx)


def high_pass(x, sigma=2.0):
    """x: (C,H,W) torch. Detail layer = image minus Gaussian blur."""
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = k.numel() // 2
    c = x.shape[0]
    b = F.pad(x[None], (r, r, r, r), mode="reflect")
    b = F.conv2d(b, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    b = F.conv2d(b, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x - b[0]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def phase_shift(a, b):
    """Sub-pixel translation between 2-D arrays a and b (pixels), via phase correlation."""
    a, b = a - a.mean(), b - b.mean()
    Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R /= np.abs(R) + 1e-12
    r = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(r), r.shape)

    def refine(p, n, get):
        l, c, rr = get((p - 1) % n), get(p), get((p + 1) % n)
        d = l - 2 * c + rr
        off = 0.5 * (l - rr) / d if abs(d) > 1e-12 else 0.0
        v = p + off
        return v - n if v > n / 2 else v

    dy = refine(py, r.shape[0], lambda i: r[i, px])
    dx = refine(px, r.shape[1], lambda j: r[py, j])
    return float(np.hypot(dy, dx))


def image_metrics(pred, hr, lr_ref, scale):
    """pred, hr: (4,H,W) reflectance torch; lr_ref (4,h,w). Returns dict."""
    b = BORDER
    P, H = pred[:, b:-b, b:-b], hr[:, b:-b, b:-b]
    lo = torch.quantile(H.flatten(1), 0.01, dim=1)[:, None, None]
    hi = torch.quantile(H.flatten(1), 0.99, dim=1)[:, None, None]
    n = lambda t: ((t - lo) / (hi - lo).clamp(min=1e-6)).clamp(-0.5, 1.5)
    Pn, Hn = n(P), n(H)
    mse = ((Pn - Hn) ** 2).mean((1, 2))
    psnr = (10 * torch.log10(1 / mse.clamp(min=1e-10))).mean().item()
    ssim = ssim_map(Pn[None], Hn[None])[0].mean().item()
    sam = torch.rad2deg(spectral_angle((P + 0.01)[None], (H + 0.01)[None])).mean().item()
    hp_p, hp_h = high_pass(pred)[:, b:-b, b:-b], high_pass(hr)[:, b:-b, b:-b]
    hf = float(np.mean([corr(hp_p[i], hp_h[i]) for i in range(4)]))
    down = F.avg_pool2d(pred[None], scale)[0]
    cons = (down - lr_ref).abs().mean().item()
    shift = phase_shift(down.mean(0).numpy(), lr_ref.mean(0).numpy())
    return {"psnr": psnr, "ssim": ssim, "sam_deg": sam, "hf_corr": hf, "consistency_l1": cons, "shift_px": shift}


def rgb(x):  # reflectance (4,H,W) -> display
    im = x[RGB_IDX].permute(1, 2, 0).numpy()
    lo, hi = np.percentile(im, [2, 98])
    return np.clip((im - lo) / (hi - lo + 1e-6), 0, 1)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--data-dir", default="opensr_data", help="cache folder for the downloaded .pkl files")
    ap.add_argument("--offset", type=float, default=1000.0,
                    help="DN offset of YOUR training data (S2 baseline>=04.00 adds +1000). "
                         "opensr-test L2A has no offset, so it is added before the model.")
    ap.add_argument("--norm-range", nargs=2, type=float, default=[-0.25, 1.25],
                    help="clamp range in normalised units (wider than training [0,1] so dark/bright "
                         "scenes outside India's p1-p99 are not clipped)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-guard", action="store_true",
                    help="disable the out-of-range guard (bicubic fallback where the input is outside "
                         "the model's training range)")
    ap.add_argument("--guard-range", nargs=2, type=float, default=None,
                    help="default: (-inf, upper end of the checkpoint's training clip range)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--opensr-metrics", action="store_true")
    ap.add_argument("--figs", type=int, default=3, help="comparison figures per dataset")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = get_device()
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = _wrap(ck["cfg"])
    scale = int(cfg.data.scale)
    model = build_model(cfg.model, scale).to(device).eval()
    model.load_state_dict(ck["model"])
    norm = build_normalizer(cfg)
    deg = build_degradation(cfg)
    rlo, rhi = a.norm_range
    # default: guard only the upper end (saturation of bright roofs -> colour shift); dark values
    # just below the range are harmless and guarding them would hand large areas to bicubic
    guard = None if a.no_guard else tuple(a.guard_range or (-1e9, cfg.data.get("clip", [0.0, 1.0])[1]))
    gdesc = "off" if guard is None else (f"input > {guard[1]}" if guard[0] < -1e8 else f"outside {list(guard)}")
    print(f"[opensr] clip {a.norm_range} | out-of-range guard: {gdesc}")
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "eval_opensr")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    official = None
    if a.opensr_metrics:
        import opensr_test  # pip install opensr-test
        official = opensr_test.Metrics(device=str(device), agg_method="pixel", patch_size=1,
                                       correctness_distance="nd", rgb_bands=[2, 1, 0])

    rows, calib = [], []
    rng = np.random.default_rng(0)
    for name in a.datasets:
        ds = load_dataset(name, a.data_dir)
        lr_all, hr_all = ds["L2A"], ds["HRharm"]
        lr_all = [to_f32(x)[L2A_IDX] for x in lr_all]
        perm, C = detect_hr_band_order(lr_all, hr_all)
        print(f"\n[{name}] {len(lr_all)} images | HR channel for B02,B03,B04,B08 -> {perm} "
              f"(match corr {np.mean([C[i, perm[i]] for i in range(4)]):.2f})")

        lr_means = np.mean([x.reshape(4, -1).mean(1) for x in lr_all], 0)
        train_means = np.array([norm.lo[i, 0, 0] + norm.hi[i, 0, 0] for i in range(4)]) / 2
        print(f"  LR mean DN (no offset) {lr_means.round(0)} | with +{a.offset:.0f}: "
              f"{(lr_means + a.offset).round(0)} | training p1/p99 midpoint {train_means.round(0)}")

        n = len(lr_all) if not a.max_images else min(a.max_images, len(lr_all))
        clipped, guarded = [], []
        for i in range(n):
            lr_dn = lr_all[i]
            hr = torch.from_numpy(to_f32(hr_all[i])[perm]) / 10000.0
            lr_ref = torch.from_numpy(lr_dn) / 10000.0
            H, W = lr_dn.shape[1] * scale, lr_dn.shape[2] * scale
            hr = hr[:, :H, :W]

            x = norm.norm(lr_dn + a.offset, clip=False)
            clipped.append(float(((x < rlo) | (x > rhi)).mean()))
            x = torch.from_numpy(x)[None].to(device)
            sr_n, std, gm = guarded_predict(model, x, (rlo, rhi), guard, scale, tta=a.tta)
            guarded.append(float(gm.mean()))
            sr = ((norm.denorm(sr_n[0]).cpu() - a.offset) / 10000.0).float()
            bic = F.interpolate(lr_ref[None], scale_factor=scale, mode="bicubic", align_corners=False)[0]

            m_sr, m_bic = image_metrics(sr, hr, lr_ref, scale), image_metrics(bic, hr, lr_ref, scale)
            m_sr["hr_shift_px"] = m_bic["hr_shift_px"] = phase_shift(
                F.avg_pool2d(hr[None], scale)[0].mean(0).numpy(), lr_ref.mean(0).numpy())

            if official is not None:
                for tag, pred, m in (("sr", sr, m_sr), ("bicubic", bic, m_bic)):
                    try:
                        res = official.compute(lr=lr_ref.to(device), sr=pred.to(device), hr=hr.to(device))
                        m.update({f"os_{k}": float(v) for k, v in res.items()})
                    except Exception as e:  # never let the optional package kill the run
                        print(f"  [warn] opensr metrics failed on {name}#{i} ({tag}): {e}")

            for tag, m in (("bicubic", m_bic), ("sr", m_sr)):
                rows.append({"dataset": name, "idx": i, "method": tag, **m})

            if std is not None:
                u = std[0].mean(0).cpu()[BORDER:-BORDER, BORDER:-BORDER].flatten()
                e = (sr - hr).abs().mean(0)[BORDER:-BORDER, BORDER:-BORDER].flatten()
                k = rng.choice(u.numel(), size=min(5000, u.numel()), replace=False)
                calib.append(pd.DataFrame({"dataset": name, "u": u[k].numpy(), "err": e[k].numpy()}))

            if i < a.figs:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                up = F.interpolate(lr_ref[None], scale_factor=scale, mode="nearest")[0]
                panels = [("Sentinel-2 input (10 m)", rgb(up)), ("Bicubic", rgb(bic)),
                          ("EDSR SR (2.5 m)", rgb(sr)), ("Real reference (2.5 m)", rgb(hr))]
                fig, ax = plt.subplots(1, 5 if std is not None else 4, figsize=(25, 5.4))
                for j, (t, im) in enumerate(panels):
                    ax[j].imshow(im); ax[j].set_title(t)
                if std is not None:
                    ax[4].imshow(std[0].mean(0).cpu(), cmap="viridis"); ax[4].set_title("Uncertainty (TTA std)")
                for z in ax:
                    z.axis("off")
                fig.suptitle(f"{name} #{i} | detail corr with reference: bicubic {m_bic['hf_corr']:.3f} -> "
                             f"SR {m_sr['hf_corr']:.3f} | PSNR {m_bic['psnr']:.2f} -> {m_sr['psnr']:.2f} dB")
                fig.tight_layout()
                fig.savefig(os.path.join(out, "figures", f"{name}_{i:02d}.png"), dpi=90)
                plt.close(fig)
        print(f"  input pixels outside norm range: {np.mean(clipped) * 100:.2f}% | "
              f"output pixels handed to bicubic by the guard: {np.mean(guarded) * 100:.2f}%")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "per_image.csv"), index=False)

    # ---------------- summary ----------------
    metrics = ["psnr", "ssim", "sam_deg", "hf_corr", "consistency_l1", "shift_px"]
    os_cols = [c for c in res.columns if c.startswith("os_")]
    better = {"psnr": "↑", "ssim": "↑", "sam_deg": "↓", "hf_corr": "↑", "consistency_l1": "↓", "shift_px": "↓"}
    summary, lines = {}, ["# Real high-resolution validation (opensr-test, Sentinel-2 10 m -> 2.5 m)", "",
                          f"Checkpoint: `{a.ckpt}` | TTA: {a.tta} | out-of-range guard: {gdesc}", ""]
    groups = [(d, res[res.dataset == d]) for d in a.datasets] + [("ALL", res)]
    for d, g in groups:
        lines += [f"## {d} ({g.idx.nunique() if d != 'ALL' else len(g) // 2} images)", "",
                  "| Method | " + " | ".join(f"{m} {better[m]}" for m in metrics + []) +
                  "".join(f" | {c}" for c in os_cols) + " |",
                  "|---" * (len(metrics) + len(os_cols) + 1) + "|"]
        summary[d] = {}
        for meth in ("bicubic", "sr"):
            s = g[g.method == meth]
            summary[d][meth] = {m: dict(zip(("mean", "lo", "hi"), bootstrap_ci(s[m]))) for m in metrics + os_cols}
            cells = [f"{summary[d][meth][m]['mean']:.4f}" for m in metrics + os_cols]
            lines.append(f"| {meth} | " + " | ".join(cells) + " |")
        b = g[g.method == "bicubic"].set_index(["dataset", "idx"])
        s = g[g.method == "sr"].set_index(["dataset", "idx"])
        dp = (s.psnr - b.psnr).to_numpy()
        dh = (s.hf_corr - b.hf_corr).to_numpy()
        mp, lp, hp = bootstrap_ci(dp)
        mh, lh, hh = bootstrap_ci(dh)
        summary[d]["gain"] = {"psnr_db": [mp, lp, hp], "psnr_win_rate": float((dp > 0).mean()),
                              "hf_corr": [mh, lh, hh], "hf_win_rate": float((dh > 0).mean())}
        lines += ["", f"- PSNR gain over bicubic: **{mp:+.3f} dB** [{lp:+.3f}, {hp:+.3f}], "
                      f"wins on {(dp > 0).mean() * 100:.0f}% of images",
                  f"- Detail-correlation gain: **{mh:+.4f}** [{lh:+.4f}, {hh:+.4f}], "
                  f"wins on {(dh > 0).mean() * 100:.0f}% of images", ""]

    lines += ["Geolocation: `shift_px` = shift between the downsampled output and the S2 input, in 10 m "
              "pixels (`hr_shift_px` = the reference's own residual misregistration, for scale).", ""]
    if calib:
        cdf = pd.concat(calib)
        rho = cdf[["u", "err"]].corr(method="spearman").iloc[0, 1]
        per = cdf.groupby("dataset").apply(lambda g: g[["u", "err"]].corr(method="spearman").iloc[0, 1])
        summary["uncertainty_rho"] = {"all": float(rho), **{k: float(v) for k, v in per.items()}}
        lines += ["## Uncertainty vs REAL error", "", f"- Spearman rho, all datasets: **{rho:.3f}**"] + \
                 [f"- {k}: {v:.3f}" for k, v in per.items()] + [""]
    lines += ["> Domain note: references are USA / Spain / WorldStrat sites; the model was trained only on",
              "> Indian cities with synthetic 40->10 m pairs. This is a zero-shot, cross-domain, real-scale test."]

    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines))
    print(f"\n[opensr] written to {out}")


if __name__ == "__main__":
    main()
