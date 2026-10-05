# Evaluation - test split (Guwahati)

Patches: 192 | TTA: True | scale x4 (synthetic 40 m -> 10 m, Wald protocol) | DN offset removed: 1000

| Method | psnr (higher better) | ssim (higher better) | sam_deg (lower better) | ergas (lower better) | ndvi_mae (lower better) |
|---|---|---|---|---|---|
| bicubic | 31.6634 [31.1963, 32.1239] | 0.7726 [0.7600, 0.7844] | 1.5570 [1.4816, 1.6360] | 1.7196 [1.5985, 1.8417] | 0.0245 [0.0232, 0.0259] |
| edsr_psnr:edsr_guwahati/best.pt | 32.3781 [31.9253, 32.8255] | 0.8108 [0.8008, 0.8201] | 1.4355 [1.3687, 1.5058] | 1.5699 [1.4615, 1.6805] | 0.0223 [0.0211, 0.0234] |

## Per-band PSNR (dB)

| Method | B02 | B03 | B04 | B08 |
|---|---|---|---|---|
| bicubic | 34.478 | 33.128 | 33.006 | 26.042 |
| edsr_psnr:edsr_guwahati/best.pt | 35.084 | 33.809 | 33.882 | 26.737 |

## Paired gain over bicubic (same patches)

- **edsr_psnr:edsr_guwahati/best.pt**: +0.715 dB [+0.681, +0.752], better than bicubic on 100.0% of patches

## Per-scene PSNR (dB)

```
method              bicubic  edsr_psnr:edsr_guwahati/best.pt
scene                                                       
Scene_1_2025-01-10   29.199                           29.948
Scene_2_2025-02-09   32.759                           33.455
Scene_3_2026-02-19   26.893                           27.892
Scene_4_2026-03-08   32.135                           32.831
Scene_5_2026-03-11   37.223                           37.695
```

## Uncertainty vs actual error

- Spearman rho (TTA std vs |error|): **0.353**
- Spearman rho (LR-consistency vs |error|): **0.157**
- Mean |error| in top-uncertainty decile is 3.1x the bottom decile

rho > 0.3 means the uncertainty map is a usable warning signal; ~0 means it is not.

> Caveat: targets are 10 m Sentinel-2. These numbers prove the model recovers detail
> lost between 40 m and 10 m. They do NOT prove <4 m accuracy; that needs real
> high-resolution reference imagery (scripts/validate_reference.py).
