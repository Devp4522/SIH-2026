# Real high-resolution validation (opensr-test, Sentinel-2 10 m -> 2.5 m)

Checkpoint: `runs/edsr_ft_real/best.pt` | TTA: True

## spot (9 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 17.4499 | 0.4003 | 1.7868 | 0.3402 | 0.0019 | 0.0007 |
| sr | 17.8168 | 0.4706 | 1.8164 | 0.4068 | 0.0049 | 0.0028 |

- PSNR gain over bicubic: **+0.367 dB** [+0.099, +0.605], wins on 78% of images
- Detail-correlation gain: **+0.0666** [+0.0538, +0.0791], wins on 100% of images

## spain_urban (20 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.4057 | 0.3238 | 3.8161 | 0.2357 | 0.0021 | 0.0007 |
| sr | 16.4876 | 0.3623 | 3.8016 | 0.2619 | 0.0057 | 0.0032 |

- PSNR gain over bicubic: **+0.082 dB** [-0.253, +0.370], wins on 70% of images
- Detail-correlation gain: **+0.0262** [+0.0013, +0.0525], wins on 65% of images

## ALL (29 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.7297 | 0.3475 | 3.1863 | 0.2681 | 0.0020 | 0.0007 |
| sr | 16.9001 | 0.3959 | 3.1855 | 0.3068 | 0.0055 | 0.0031 |

- PSNR gain over bicubic: **+0.170 dB** [-0.068, +0.379], wins on 72% of images
- Detail-correlation gain: **+0.0387** [+0.0194, +0.0577], wins on 76% of images

Geolocation: `shift_px` = shift between the downsampled output and the S2 input, in 10 m pixels (`hr_shift_px` = the reference's own residual misregistration, for scale).

## Uncertainty vs REAL error

- Spearman rho, all datasets: **0.497**
- spain_urban: 0.438
- spot: 0.575

> Domain note: references are USA / Spain / WorldStrat sites; the model was trained only on
> Indian cities with synthetic 40->10 m pairs. This is a zero-shot, cross-domain, real-scale test.
