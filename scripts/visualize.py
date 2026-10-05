"""
Figures for the report / demo.

    # side-by-side panels on test patches (true colour + NDVI + error + uncertainty)
    python scripts/visualize.py samples --ckpt runs/edsr_baseline/best.pt --n 6 --tta

    # training curves from log.csv
    python scripts/visualize.py curves --run runs/edsr_baseline

Figures go to <run_dir>/figures/.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import _common  # noqa: F401
from srm.config import _wrap, apply_overrides, to_plain
from srm.data import NIR, RED, RGB_IDX, SRPatchDataset, build_degradation, build_normalizer, resolve_patch_paths
from srm.models import build_model
from srm.trainer import get_device
from srm.uncertainty import predict


def rgb(x):  # (C,H,W) normalised -> (H,W,3) display, gentle gamma
    return np.clip(x[RGB_IDX].transpose(1, 2, 0), 0, 1) ** 0.8


def ndvi(x_dn):
    return (x_dn[NIR] - x_dn[RED]) / np.clip(x_dn[NIR] + x_dn[RED], 1e-6, None)


def samples(a):
    device = get_device()
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = _wrap(apply_overrides(to_plain(ck["cfg"]), a.set))
    scale = int(cfg.data.scale)
    model = build_model(cfg.model, scale).to(device).eval()
    model.load_state_dict(ck["model"])
    norm, deg = build_normalizer(cfg), build_degradation(cfg)
    run_dir = os.path.dirname(a.ckpt)
    df = resolve_patch_paths(pd.read_csv(os.path.join(run_dir, "splits", f"{a.split}.csv")), cfg.data.data_root)
    ds = SRPatchDataset(df, norm, deg, train=False, crop_size=a.crop)
    out = os.path.join(run_dir, "figures")
    os.makedirs(out, exist_ok=True)

    idxs = np.random.default_rng(a.seed).choice(len(ds), size=min(a.n, len(ds)), replace=False)
    for j, i in enumerate(idxs):
        it = ds[int(i)]
        lr, hr, mask = it["lr"][None].to(device), it["hr"][None], it["mask"][0].numpy()
        sr, std = predict(model, lr, tta=a.tta)
        bic = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)
        lr_n, sr_n, bic_n, hr_n = (t[0].cpu().numpy() for t in (lr, sr, bic, hr))
        up = np.kron(lr_n, np.ones((1, scale, scale)))
        err = np.abs(sr_n - hr_n).mean(0) * mask
        dn = lambda x: norm.denorm(x)

        panels = [("LR input (40 m eq.)", rgb(up)), ("Bicubic", rgb(bic_n)), ("SR (model)", rgb(sr_n)),
                  ("HR target (10 m)", rgb(hr_n))]
        fig, ax = plt.subplots(2, 4, figsize=(18, 9))
        for k, (t, im) in enumerate(panels):
            ax[0, k].imshow(im); ax[0, k].set_title(t)
        ax[1, 0].imshow(ndvi(dn(sr_n)), cmap="RdYlGn", vmin=-0.2, vmax=0.8); ax[1, 0].set_title("NDVI (SR)")
        ax[1, 1].imshow(ndvi(dn(hr_n)), cmap="RdYlGn", vmin=-0.2, vmax=0.8); ax[1, 1].set_title("NDVI (HR)")
        e = ax[1, 2].imshow(err, cmap="magma", vmax=np.percentile(err, 99) + 1e-6)
        ax[1, 2].set_title("|SR - HR| (band mean)"); fig.colorbar(e, ax=ax[1, 2], fraction=0.046)
        if std is not None:
            u = std[0].mean(0).cpu().numpy()
            ui = ax[1, 3].imshow(u, cmap="viridis", vmax=np.percentile(u, 99) + 1e-6)
            ax[1, 3].set_title("Uncertainty (TTA std)"); fig.colorbar(ui, ax=ax[1, 3], fraction=0.046)
        else:
            ax[1, 3].imshow(mask, cmap="gray"); ax[1, 3].set_title("valid mask (run with --tta for uncertainty)")
        for x in ax.ravel():
            x.axis("off")
        psnr = lambda p: -10 * np.log10(((p - hr_n) ** 2 * mask).sum() / (mask.sum() * p.shape[0]) + 1e-10)
        fig.suptitle(f"{it['city']} / {it['scene']} / patch {it['idx']}  |  PSNR bicubic {psnr(bic_n):.2f} dB"
                     f"  ->  SR {psnr(sr_n):.2f} dB")
        fig.tight_layout()
        p = os.path.join(out, f"sample_{j:02d}_{it['city']}_{it['idx']}.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        print("saved", p)


def curves(a):
    log = pd.read_csv(os.path.join(a.run, "log.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(log.epoch, log.train_total, label="train total loss"); ax[0].set_yscale("log")
    ax[0].set_title("Training loss"); ax[0].legend()
    ax[1].plot(log.epoch, log.val_psnr, label="val PSNR")
    for b in ["B02", "B03", "B04", "B08"]:
        if f"val_psnr_{b}" in log:
            ax[1].plot(log.epoch, log[f"val_psnr_{b}"], alpha=0.5, label=b)
    ax[1].set_title("Validation PSNR (dB)"); ax[1].legend()
    ax[2].plot(log.epoch, log.val_gain_over_bicubic_db, color="k")
    ax[2].axhline(0, ls="--", c="gray"); ax[2].set_title("Gain over bicubic (dB)")
    for x in ax:
        x.set_xlabel("epoch"); x.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(a.run, "figures"), exist_ok=True)
    p = os.path.join(a.run, "figures", "training_curves.png")
    fig.savefig(p, dpi=110)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("samples")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--crop", type=int, default=256)
    s.add_argument("--tta", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--set", nargs="*", default=[])
    c = sub.add_parser("curves")
    c.add_argument("--run", required=True)
    a = ap.parse_args()
    samples(a) if a.cmd == "samples" else curves(a)


if __name__ == "__main__":
    main()
