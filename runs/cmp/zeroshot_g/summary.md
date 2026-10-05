# Real high-resolution validation (opensr-test, Sentinel-2 10 m -> 2.5 m)

Checkpoint: `runs/edsr_final_v2/best.pt` | TTA: True | out-of-range guard: input > 1.5

## spot (9 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 17.4499 | 0.4003 | 1.7868 | 0.3402 | 0.0019 | 0.0007 |
| sr | 17.3672 | 0.4319 | 2.2261 | 0.3467 | 0.0018 | 0.0008 |

- PSNR gain over bicubic: **-0.083 dB** [-0.185, +0.018], wins on 33% of images
- Detail-correlation gain: **+0.0065** [-0.0019, +0.0151], wins on 67% of images

## spain_urban (20 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.4057 | 0.3238 | 3.8161 | 0.2357 | 0.0021 | 0.0007 |
| sr | 16.1285 | 0.3423 | 4.2191 | 0.2324 | 0.0026 | 0.0021 |

- PSNR gain over bicubic: **-0.277 dB** [-0.567, -0.066], wins on 45% of images
- Detail-correlation gain: **-0.0032** [-0.0169, +0.0105], wins on 45% of images

## ALL (29 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.7297 | 0.3475 | 3.1863 | 0.2681 | 0.0020 | 0.0007 |
| sr | 16.5129 | 0.3701 | 3.6006 | 0.2679 | 0.0024 | 0.0017 |

- PSNR gain over bicubic: **-0.217 dB** [-0.426, -0.060], wins on 41% of images
- Detail-correlation gain: **-0.0002** [-0.0103, +0.0094], wins on 52% of images

Geolocation: `shift_px` = shift between the downsampled output and the S2 input, in 10 m pixels (`hr_shift_px` = the reference's own residual misregistration, for scale).

## Uncertainty vs REAL error

- Spearman rho, all datasets: **0.496**
- spain_urban: 0.443
- spot: 0.572

> Domain note: references are USA / Spain / WorldStrat sites; the model was trained only on
> Indian cities with synthetic 40->10 m pairs. This is a zero-shot, cross-domain, real-scale test.
