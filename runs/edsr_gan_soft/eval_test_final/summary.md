# Evaluation - test split (Nashik)

Patches: 307 | TTA: True | scale x4 (synthetic 40 m -> 10 m, Wald protocol)

| Method | psnr (higher better) | ssim (higher better) | sam_deg (lower better) | ergas (lower better) | ndvi_mae (lower better) |
|---|---|---|---|---|---|
| bicubic | 23.6464 [23.4376, 23.8475] | 0.6035 [0.5973, 0.6094] | 1.8832 [1.8245, 1.9431] | 1.8030 [1.7633, 1.8457] | 0.0322 [0.0312, 0.0333] |
| edsr_gan:edsr_gan_soft/final.pt | 24.1868 [23.9881, 24.3783] | 0.6589 [0.6537, 0.6640] | 1.7258 [1.6788, 1.7730] | 1.6842 [1.6499, 1.7213] | 0.0291 [0.0282, 0.0299] |

## Per-band PSNR (dB)

| Method | B02 | B03 | B04 | B08 |
|---|---|---|---|---|
| bicubic | 24.938 | 24.326 | 23.441 | 21.881 |
| edsr_gan:edsr_gan_soft/final.pt | 25.414 | 24.788 | 24.130 | 22.415 |

## Paired gain over bicubic (same patches)

- **edsr_gan:edsr_gan_soft/final.pt**: +0.540 dB [+0.505, +0.575], better than bicubic on 94.8% of patches

## Per-scene PSNR (dB)

```
method              bicubic  edsr_gan:edsr_gan_soft/final.pt
scene                                                       
Scene_1_2025-01-28   23.897                           24.558
Scene_2_2025-02-27   24.440                           24.847
Scene_3_2025-04-28   23.553                           23.992
Scene_4_2025-03-29   23.830                           24.349
Scene_5_2026-06-17   21.980                           22.745
```

## Uncertainty vs actual error

- Spearman rho (TTA std vs |error|): **0.438**
- Spearman rho (LR-consistency vs |error|): **0.288**
- Mean |error| in top-uncertainty decile is 4.0x the bottom decile

rho > 0.3 means the uncertainty map is a usable warning signal; ~0 means it is not.

> Caveat: targets are 10 m Sentinel-2. These numbers prove the model recovers detail
> lost between 40 m and 10 m. They do NOT prove <4 m accuracy; that needs real
> high-resolution reference imagery (scripts/validate_reference.py).
