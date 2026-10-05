# Evaluation - test split (Nashik)

Patches: 307 | TTA: True | scale x4 (synthetic 40 m -> 10 m, Wald protocol) | DN offset removed: 1000

| Method | psnr (higher better) | ssim (higher better) | sam_deg (lower better) | ergas (lower better) | ndvi_mae (lower better) |
|---|---|---|---|---|---|
| bicubic | 23.6464 [23.4376, 23.8475] | 0.6035 [0.5973, 0.6094] | 2.8494 [2.7659, 2.9337] | 3.4230 [3.3235, 3.5289] | 0.0485 [0.0469, 0.0500] |
| edsr_psnr:edsr_baseline/best.pt | 24.8951 [24.6850, 25.0948] | 0.7021 [0.6971, 0.7068] | 2.3731 [2.3129, 2.4360] | 2.9560 [2.8730, 3.0423] | 0.0393 [0.0382, 0.0405] |
| edsr_gan:edsr_gan_soft/best.pt | 24.6539 [24.4404, 24.8578] | 0.6878 [0.6826, 0.6928] | 2.4259 [2.3644, 2.4908] | 3.0462 [2.9593, 3.1357] | 0.0404 [0.0393, 0.0416] |
| edsr_gan:edsr_gan_soft/final.pt | 24.1868 [23.9881, 24.3783] | 0.6589 [0.6537, 0.6640] | 2.6185 [2.5554, 2.6864] | 3.1950 [3.1100, 3.2829] | 0.0436 [0.0424, 0.0448] |

## Per-band PSNR (dB)

| Method | B02 | B03 | B04 | B08 |
|---|---|---|---|---|
| bicubic | 24.938 | 24.326 | 23.441 | 21.881 |
| edsr_psnr:edsr_baseline/best.pt | 26.158 | 25.490 | 24.775 | 23.158 |
| edsr_gan:edsr_gan_soft/best.pt | 25.927 | 25.189 | 24.410 | 23.090 |
| edsr_gan:edsr_gan_soft/final.pt | 25.414 | 24.788 | 24.130 | 22.415 |

## Paired gain over bicubic (same patches)

- **edsr_psnr:edsr_baseline/best.pt**: +1.249 dB [+1.218, +1.279], better than bicubic on 100.0% of patches
- **edsr_gan:edsr_gan_soft/best.pt**: +1.007 dB [+0.977, +1.036], better than bicubic on 100.0% of patches
- **edsr_gan:edsr_gan_soft/final.pt**: +0.540 dB [+0.505, +0.575], better than bicubic on 94.8% of patches

## Per-scene PSNR (dB)

```
method              bicubic  edsr_gan:edsr_gan_soft/best.pt  edsr_gan:edsr_gan_soft/final.pt  edsr_psnr:edsr_baseline/best.pt
scene                                                                                                                        
Scene_1_2025-01-28   23.897                          24.993                           24.558                           25.191
Scene_2_2025-02-27   24.440                          25.348                           24.847                           25.539
Scene_3_2025-04-28   23.553                          24.481                           23.992                           24.757
Scene_4_2025-03-29   23.830                          24.916                           24.349                           25.124
Scene_5_2026-06-17   21.980                          22.955                           22.745                           23.337
```

## Uncertainty vs actual error

- Spearman rho (TTA std vs |error|): **0.415**
- Spearman rho (LR-consistency vs |error|): **0.133**
- Mean |error| in top-uncertainty decile is 3.9x the bottom decile

rho > 0.3 means the uncertainty map is a usable warning signal; ~0 means it is not.

> Caveat: targets are 10 m Sentinel-2. These numbers prove the model recovers detail
> lost between 40 m and 10 m. They do NOT prove <4 m accuracy; that needs real
> high-resolution reference imagery (scripts/validate_reference.py).
