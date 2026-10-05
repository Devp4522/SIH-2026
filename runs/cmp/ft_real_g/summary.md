# Real high-resolution validation (opensr-test, Sentinel-2 10 m -> 2.5 m)

Checkpoint: `runs/edsr_ft_real/best.pt` | TTA: True | out-of-range guard: input > 1.5

## spot (9 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 17.4499 | 0.4003 | 1.7868 | 0.3402 | 0.0019 | 0.0007 |
| sr | 17.8398 | 0.4699 | 1.7998 | 0.4038 | 0.0043 | 0.0016 |

- PSNR gain over bicubic: **+0.390 dB** [+0.186, +0.559], wins on 89% of images
- Detail-correlation gain: **+0.0636** [+0.0515, +0.0758], wins on 100% of images

## spain_urban (20 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.4057 | 0.3238 | 3.8161 | 0.2357 | 0.0021 | 0.0007 |
| sr | 16.5059 | 0.3623 | 3.7822 | 0.2626 | 0.0050 | 0.0024 |

- PSNR gain over bicubic: **+0.100 dB** [-0.209, +0.360], wins on 65% of images
- Detail-correlation gain: **+0.0270** [+0.0036, +0.0515], wins on 70% of images

## ALL (29 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.7297 | 0.3475 | 3.1863 | 0.2681 | 0.0020 | 0.0007 |
| sr | 16.9199 | 0.3957 | 3.1670 | 0.3064 | 0.0048 | 0.0022 |

- PSNR gain over bicubic: **+0.190 dB** [-0.036, +0.381], wins on 72% of images
- Detail-correlation gain: **+0.0383** [+0.0196, +0.0563], wins on 79% of images

Geolocation: `shift_px` = shift between the downsampled output and the S2 input, in 10 m pixels (`hr_shift_px` = the reference's own residual misregistration, for scale).

## Uncertainty vs REAL error

- Spearman rho, all datasets: **0.497**
- spain_urban: 0.438
- spot: 0.576

> Domain note: references are USA / Spain / WorldStrat sites; the model was trained only on
> Indian cities with synthetic 40->10 m pairs. This is a zero-shot, cross-domain, real-scale test.
