# Evaluation - val split (Ahmedabad, Guwahati, Nashik, New_Delhi)

Patches: 238 | TTA: True | scale x4 (synthetic 40 m -> 10 m, Wald protocol) | DN offset removed: 1000

| Method | psnr (higher better) | ssim (higher better) | sam_deg (lower better) | ergas (lower better) | ndvi_mae (lower better) |
|---|---|---|---|---|---|
| bicubic | 23.7090 [23.4196, 24.0058] | 0.6263 [0.6196, 0.6333] | 2.6989 [2.6086, 2.7941] | 2.9652 [2.8445, 3.0846] | 0.0457 [0.0441, 0.0475] |
| edsr_psnr:edsr_final_v2/best.pt | 25.2487 [24.9539, 25.5568] | 0.7378 [0.7322, 0.7436] | 2.1523 [2.0818, 2.2222] | 2.4827 [2.3859, 2.5836] | 0.0353 [0.0341, 0.0366] |
| edsr_finetune_real:edsr_ft_real/best.pt | 24.3039 [24.0281, 24.5900] | 0.6981 [0.6926, 0.7041] | 2.5192 [2.4399, 2.5987] | 2.7432 [2.6386, 2.8491] | 0.0419 [0.0404, 0.0434] |
| edsr_finetune_real:edsr_ft_joint/best.pt | 25.1882 [24.8941, 25.4961] | 0.7348 [0.7292, 0.7405] | 2.1776 [2.1067, 2.2486] | 2.5000 [2.4024, 2.6019] | 0.0357 [0.0345, 0.0369] |

## Per-band PSNR (dB)

| Method | B02 | B03 | B04 | B08 |
|---|---|---|---|---|
| bicubic | 24.544 | 24.001 | 23.191 | 23.100 |
| edsr_psnr:edsr_final_v2/best.pt | 25.998 | 25.478 | 24.881 | 24.638 |
| edsr_finetune_real:edsr_ft_real/best.pt | 25.139 | 24.574 | 23.909 | 23.594 |
| edsr_finetune_real:edsr_ft_joint/best.pt | 25.935 | 25.411 | 24.813 | 24.594 |

## Paired gain over bicubic (same patches)

- **edsr_psnr:edsr_final_v2/best.pt**: +1.540 dB [+1.501, +1.584], better than bicubic on 100.0% of patches
- **edsr_finetune_real:edsr_ft_real/best.pt**: +0.595 dB [+0.562, +0.629], better than bicubic on 99.2% of patches
- **edsr_finetune_real:edsr_ft_joint/best.pt**: +1.479 dB [+1.441, +1.524], better than bicubic on 100.0% of patches

## Per-scene PSNR (dB)

```
method              bicubic  edsr_finetune_real:edsr_ft_joint/best.pt  edsr_finetune_real:edsr_ft_real/best.pt  edsr_psnr:edsr_final_v2/best.pt
scene                                                                                                                                          
Scene_1_2025-01-28   23.738                                    25.233                                   24.417                           25.321
Scene_1_2025-05-01   23.258                                    24.817                                   24.048                           24.879
Scene_1_2025-06-12   27.484                                    29.208                                   27.879                           29.257
Scene_2_2025-02-27   24.030                                    25.355                                   24.714                           25.413
Scene_2_2025-04-03   23.709                                    25.340                                   24.543                           25.401
Scene_2_2025-06-07   19.683                                    21.188                                   20.489                           21.240
Scene_3_2025-04-28   23.488                                    24.893                                   24.144                           24.959
Scene_3_2025-05-16   21.465                                    22.735                                   22.109                           22.778
Scene_3_2025-05-23   23.300                                    24.847                                   23.882                           24.904
Scene_3_2026-02-19   26.918                                    27.951                                   27.196                           28.004
Scene_4_2025-03-22   23.949                                    25.500                                   24.732                           25.562
Scene_4_2025-03-29   23.983                                    25.551                                   24.756                           25.618
Scene_4_2025-05-13   24.110                                    25.593                                   24.494                           25.674
Scene_5_2025-04-06   22.435                                    24.042                                   23.282                           24.100
Scene_5_2025-04-23   22.846                                    24.179                                   23.456                           24.243
Scene_5_2026-06-17   22.193                                    23.664                                   22.901                           23.722
```

## Uncertainty vs actual error

- Spearman rho (TTA std vs |error|): **0.436**
- Spearman rho (LR-consistency vs |error|): **0.200**
- Mean |error| in top-uncertainty decile is 4.0x the bottom decile

rho > 0.3 means the uncertainty map is a usable warning signal; ~0 means it is not.

> Caveat: targets are 10 m Sentinel-2. These numbers prove the model recovers detail
> lost between 40 m and 10 m. They do NOT prove <4 m accuracy; that needs real
> high-resolution reference imagery (scripts/validate_reference.py).
