"""
Uncertainty & error components delivered with every SR output.

1. TTA spread (std over the 8 dihedral transforms of the input).
   A trained SR net should be equivariant to rotations/flips. Where its 8 answers
   disagree, the detail is being *invented* rather than constrained by the input.
   This is a cheap proxy for epistemic uncertainty; `evaluate.py` checks how well it
   actually tracks the true error (Spearman rho + binned calibration table).

2. LR-consistency error |blur_down(SR) - LR|.
   Physics check: re-degrading the SR image with the assumed sensor PSF must give
   back the observed input. Large values = SR is spectrally/radiometrically
   inconsistent with what the satellite actually measured.

3. Ensemble spread (optional): pass several checkpoints; std across models.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import blur_downsample, dihedral, dihedral_inverse


@torch.no_grad()
def predict(models, lr, tta: bool = False, clamp=(0.0, 1.0)):
    """models: nn.Module or list of them. Returns (mean, std or None), both (B,C,H,W)."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    outs = []
    for m in models:
        ks = range(8) if tta else [0]
        for k in ks:
            y = m(dihedral(lr, k))
            outs.append(dihedral_inverse(y, k).float())
    stack = torch.stack(outs)
    mean = stack.mean(0)
    if clamp is not None:
        mean = mean.clamp(*clamp)
    std = stack.std(0) if stack.shape[0] > 1 else None
    return mean, std


@torch.no_grad()
def consistency_map(sr, lr, sigma: float, scale: int):
    """Per-pixel (band-mean) |blur_down(SR) - LR|, upsampled (nearest) to SR size. (B,1,H,W)."""
    err = (blur_downsample(sr, sigma, scale) - lr).abs().mean(1, keepdim=True)
    return F.interpolate(err, scale_factor=scale, mode="nearest")


@torch.no_grad()
def ood_mask(x_raw, lo: float, hi: float, scale: int, dilate: int = 1):
    """Out-of-range guard. x_raw: (B,C,h,w) normalised input WITHOUT clipping.
    Returns (B,1,H,W) float mask at output resolution: 1 where any band of the input is
    outside the range the model was trained on [lo, hi] (dilated by `dilate` LR pixels).
    There the model has never seen data, so the caller falls back to bicubic
    ("the model abstains") instead of returning clipped, colour-shifted values."""
    bad = ((x_raw < lo) | (x_raw > hi)).any(1, keepdim=True).float()
    if dilate > 0:
        k = 2 * dilate + 1
        bad = F.max_pool2d(bad, k, stride=1, padding=dilate)
    return F.interpolate(bad, scale_factor=scale, mode="nearest")


@torch.no_grad()
def guarded_predict(models, x_raw, clip_range, guard_range, scale, tta=False, dilate=1):
    """Clip input to clip_range, run the model(s), then replace out-of-range pixels by bicubic
    of the UNclipped input. Returns (sr, std, mask) in normalised units."""
    lo, hi = clip_range
    sr, std = predict(models, x_raw.clamp(lo, hi), tta=tta, clamp=(lo, hi))
    if guard_range is None:
        return sr, std, torch.zeros_like(sr[:, :1])
    m = ood_mask(x_raw, guard_range[0], guard_range[1], scale, dilate)
    bic = F.interpolate(x_raw, scale_factor=scale, mode="bicubic", align_corners=False)
    sr = torch.where(m.bool().expand_as(sr), bic, sr)
    return sr, std, m
