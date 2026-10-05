# Real high-resolution validation (opensr-test, Sentinel-2 10 m -> 2.5 m)

Checkpoint: `runs/edsr_ft_joint/best.pt` | TTA: True | out-of-range guard: input > 1.5

## spot (9 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 17.4499 | 0.4003 | 1.7868 | 0.3402 | 0.0019 | 0.0007 |
| sr | 17.6579 | 0.4596 | 2.0143 | 0.3926 | 0.0034 | 0.0014 |

- PSNR gain over bicubic: **+0.208 dB** [-0.022, +0.413], wins on 78% of images
- Detail-correlation gain: **+0.0524** [+0.0371, +0.0666], wins on 100% of images

## spain_urban (20 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.4057 | 0.3238 | 3.8161 | 0.2357 | 0.0021 | 0.0007 |
| sr | 16.3175 | 0.3561 | 3.9954 | 0.2554 | 0.0034 | 0.0022 |

- PSNR gain over bicubic: **-0.088 dB** [-0.393, +0.163], wins on 55% of images
- Detail-correlation gain: **+0.0198** [-0.0006, +0.0407], wins on 70% of images

## ALL (29 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.7297 | 0.3475 | 3.1863 | 0.2681 | 0.0020 | 0.0007 |
| sr | 16.7335 | 0.3882 | 3.3806 | 0.2980 | 0.0034 | 0.0019 |

- PSNR gain over bicubic: **+0.004 dB** [-0.224, +0.196], wins on 62% of images
- Detail-correlation gain: **+0.0299** [+0.0134, +0.0455], wins on 79% of images

Geolocation: `shift_px` = shift between the downsampled output and the S2 input, in 10 m pixels (`hr_shift_px` = the reference's own residual misregistration, for scale).

## Uncertainty vs REAL error

- Spearman rho, all datasets: **0.500**
- spain_urban: 0.449
- spot: 0.585

> Domain note: references are USA / Spain / WorldStrat sites; the model was trained only on
> Indian cities with synthetic 40->10 m pairs. This is a zero-shot, cross-domain, real-scale test.
