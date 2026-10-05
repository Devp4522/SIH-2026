# Evaluation - val split (Ahmedabad, Guwahati, Nashik, New_Delhi)

Patches: 238 | TTA: True | scale x4 (synthetic 40 m -> 10 m, Wald protocol)

| Method | psnr (higher better) | ssim (higher better) | sam_deg (lower better) | ergas (lower better) | ndvi_mae (lower better) |
|---|---|---|---|---|---|
| bicubic | 23.7090 [23.4196, 24.0058] | 0.6263 [0.6196, 0.6333] | 1.8145 [1.7437, 1.8916] | 1.6657 [1.6083, 1.7224] | 0.0313 [0.0300, 0.0326] |
| edsr_finetune_real:edsr_ft_joint/best.pt | 25.1882 [24.8941, 25.4961] | 0.7348 [0.7292, 0.7405] | 1.4693 [1.4133, 1.5280] | 1.4054 [1.3558, 1.4534] | 0.0245 [0.0236, 0.0255] |

## Per-band PSNR (dB)

| Method | B02 | B03 | B04 | B08 |
|---|---|---|---|---|
| bicubic | 24.544 | 24.001 | 23.191 | 23.100 |
| edsr_finetune_real:edsr_ft_joint/best.pt | 25.935 | 25.411 | 24.813 | 24.594 |

## Paired gain over bicubic (same patches)

- **edsr_finetune_real:edsr_ft_joint/best.pt**: +1.479 dB [+1.441, +1.524], better than bicubic on 100.0% of patches

## Per-scene PSNR (dB)

```
method              bicubic  edsr_finetune_real:edsr_ft_joint/best.pt
scene                                                                
Scene_1_2025-01-28   23.738                                    25.233
Scene_1_2025-05-01   23.258                                    24.817
Scene_1_2025-06-12   27.484                                    29.208
Scene_2_2025-02-27   24.030                                    25.355
Scene_2_2025-04-03   23.709                                    25.340
Scene_2_2025-06-07   19.683                                    21.188
Scene_3_2025-04-28   23.488                                    24.893
Scene_3_2025-05-16   21.465                                    22.735
Scene_3_2025-05-23   23.300                                    24.847
Scene_3_2026-02-19   26.918                                    27.951
Scene_4_2025-03-22   23.949                                    25.500
Scene_4_2025-03-29   23.983                                    25.551
Scene_4_2025-05-13   24.110                                    25.593
Scene_5_2025-04-06   22.435                                    24.042
Scene_5_2025-04-23   22.846                                    24.179
Scene_5_2026-06-17   22.193                                    23.664
```

## Uncertainty vs actual error

- Spearman rho (TTA std vs |error|): **0.432**
- Spearman rho (LR-consistency vs |error|): **0.201**
- Mean |error| in top-uncertainty decile is 3.9x the bottom decile

rho > 0.3 means the uncertainty map is a usable warning signal; ~0 means it is not.

> Caveat: targets are 10 m Sentinel-2. These numbers prove the model recovers detail
> lost between 40 m and 10 m. They do NOT prove <4 m accuracy; that needs real
> high-resolution reference imagery (scripts/validate_reference.py).
