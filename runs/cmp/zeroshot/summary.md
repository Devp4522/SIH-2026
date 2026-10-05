# Real high-resolution validation (opensr-test, Sentinel-2 10 m -> 2.5 m)

Checkpoint: `runs/edsr_final_v2/best.pt` | TTA: True

## spot (9 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 17.4499 | 0.4003 | 1.7868 | 0.3402 | 0.0019 | 0.0007 |
| sr | 17.3279 | 0.4318 | 2.2594 | 0.3443 | 0.0021 | 0.0021 |

- PSNR gain over bicubic: **-0.122 dB** [-0.260, +0.013], wins on 33% of images
- Detail-correlation gain: **+0.0042** [-0.0097, +0.0156], wins on 67% of images

## spain_urban (20 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.4057 | 0.3238 | 3.8161 | 0.2357 | 0.0021 | 0.0007 |
| sr | 16.1016 | 0.3421 | 4.2465 | 0.2309 | 0.0031 | 0.0027 |

- PSNR gain over bicubic: **-0.304 dB** [-0.602, -0.077], wins on 40% of images
- Detail-correlation gain: **-0.0048** [-0.0194, +0.0100], wins on 45% of images

## ALL (29 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.7297 | 0.3475 | 3.1863 | 0.2681 | 0.0020 | 0.0007 |
| sr | 16.4822 | 0.3699 | 3.6298 | 0.2661 | 0.0028 | 0.0025 |

- PSNR gain over bicubic: **-0.248 dB** [-0.459, -0.088], wins on 38% of images
- Detail-correlation gain: **-0.0020** [-0.0133, +0.0085], wins on 52% of images

Geolocation: `shift_px` = shift between the downsampled output and the S2 input, in 10 m pixels (`hr_shift_px` = the reference's own residual misregistration, for scale).

## Uncertainty vs REAL error

- Spearman rho, all datasets: **0.496**
- spain_urban: 0.442
- spot: 0.573

> Domain note: references are USA / Spain / WorldStrat sites; the model was trained only on
> Indian cities with synthetic 40->10 m pairs. This is a zero-shot, cross-domain, real-scale test.
