"""
Evaluation metrics, all masked to valid pixels and computed PER IMAGE (then averaged),
which is the standard SR protocol and avoids the batch-mean-of-means bias.

Normalised space ([0,1]):  PSNR, SSIM per band
Digital-number space:      SAM (deg), ERGAS, RMSE per band, NDVI MAE
    - SAM / ERGAS / NDVI depend on real band ratios, so they must be computed after
      de-normalisation, not on per-band min-max scaled values.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import BAND_NAMES, NIR, RED
from .losses import spectral_angle, ssim_map


@torch.no_grad()
def per_image_metrics(pred, hr, mask, normalizer, scale: int, min_valid_frac: float = 0.05,
                      offset: float = 0.0):
    """pred, hr: (B,C,H,W) normalised & clamped. mask: (B,1,H,W). Returns list[dict].
    offset: DN offset to remove before the physical metrics (SAM, ERGAS, NDVI). Sentinel-2 L2A
    from processing baseline >= 04.00 stores reflectance*10000 + 1000; without removing it the
    band ratios (and therefore NDVI / SAM / ERGAS) are biased. PSNR/SSIM are unaffected."""
    pred, hr, mask = pred.float(), hr.float(), mask.float()
    npx = mask.sum((1, 2, 3))
    frac = npx / mask[0].numel()
    n = npx.clamp(min=1)[:, None]

    mse = ((pred - hr) ** 2 * mask).sum((2, 3)) / n                  # (B,C)
    psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
    ssim = (ssim_map(pred, hr) * mask).sum((2, 3)) / n

    p_dn, h_dn = normalizer.denorm(pred) - offset, normalizer.denorm(hr) - offset
    rmse_dn = torch.sqrt(((p_dn - h_dn) ** 2 * mask).sum((2, 3)) / n)
    mean_dn = (h_dn * mask).sum((2, 3)) / n
    ergas = 100.0 / scale * torch.sqrt(((rmse_dn / mean_dn.clamp(min=1e-6)) ** 2).mean(1))
    sam = (torch.rad2deg(spectral_angle(p_dn, h_dn)) * mask).sum((1, 2, 3)) / n[:, 0]

    ndvi = lambda t: (t[:, NIR:NIR + 1] - t[:, RED:RED + 1]) / (t[:, NIR:NIR + 1] + t[:, RED:RED + 1]).clamp(min=1e-6)
    ndvi_mae = ((ndvi(p_dn) - ndvi(h_dn)).abs() * mask).sum((1, 2, 3)) / n[:, 0]

    rows = []
    for i in range(pred.shape[0]):
        if frac[i] < min_valid_frac:
            continue
        r = {"valid_frac": frac[i].item(),
             "psnr": psnr[i].mean().item(), "ssim": ssim[i].mean().item(),
             "sam_deg": sam[i].item(), "ergas": ergas[i].item(), "ndvi_mae": ndvi_mae[i].item()}
        for b, name in enumerate(BAND_NAMES):
            r[f"psnr_{name}"] = psnr[i, b].item()
            r[f"ssim_{name}"] = ssim[i, b].item()
            r[f"rmse_dn_{name}"] = rmse_dn[i, b].item()
        rows.append(r)
    return rows


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
