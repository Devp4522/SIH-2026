# Real high-resolution validation (opensr-test, Sentinel-2 10 m -> 2.5 m)

Checkpoint: `runs/edsr_final/best.pt` | TTA: True

## naip (62 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 19.6129 | 0.4366 | 1.6726 | 0.3454 | 0.0010 | 0.0007 |
| sr | 19.6700 | 0.4687 | 1.7628 | 0.3690 | 0.0015 | 0.0015 |

- PSNR gain over bicubic: **+0.057 dB** [-0.043, +0.140], wins on 66% of images
- Detail-correlation gain: **+0.0235** [+0.0178, +0.0290], wins on 89% of images

## spot (9 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 17.4499 | 0.4003 | 1.7868 | 0.3402 | 0.0019 | 0.0007 |
| sr | 17.3107 | 0.4284 | 2.2283 | 0.3439 | 0.0029 | 0.0015 |

- PSNR gain over bicubic: **-0.139 dB** [-0.400, +0.036], wins on 33% of images
- Detail-correlation gain: **+0.0037** [-0.0128, +0.0160], wins on 78% of images

## spain_crops (28 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 17.6348 | 0.3356 | 2.1717 | 0.2105 | 0.0011 | 0.0005 |
| sr | 17.5020 | 0.3477 | 2.2814 | 0.2053 | 0.0015 | 0.0013 |

- PSNR gain over bicubic: **-0.133 dB** [-0.248, -0.023], wins on 39% of images
- Detail-correlation gain: **-0.0052** [-0.0187, +0.0065], wins on 46% of images

## spain_urban (20 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 16.4057 | 0.3238 | 3.8161 | 0.2357 | 0.0021 | 0.0007 |
| sr | 16.2582 | 0.3447 | 4.0220 | 0.2325 | 0.0031 | 0.0019 |

- PSNR gain over bicubic: **-0.147 dB** [-0.372, +0.028], wins on 40% of images
- Detail-correlation gain: **-0.0032** [-0.0166, +0.0106], wins on 45% of images

## ALL (119 images)

| Method | psnr ↑ | ssim ↑ | sam_deg ↓ | hf_corr ↑ | consistency_l1 ↓ | shift_px ↓ |
|---|---|---|---|---|---|---|
| bicubic | 18.4448 | 0.3911 | 2.1589 | 0.2948 | 0.0013 | 0.0007 |
| sr | 18.4080 | 0.4163 | 2.2997 | 0.3056 | 0.0018 | 0.0015 |

- PSNR gain over bicubic: **-0.037 dB** [-0.105, +0.028], wins on 53% of images
- Detail-correlation gain: **+0.0108** [+0.0054, +0.0161], wins on 71% of images

Geolocation: `shift_px` = shift between the downsampled output and the S2 input, in 10 m pixels (`hr_shift_px` = the reference's own residual misregistration, for scale).

## Uncertainty vs REAL error

- Spearman rho, all datasets: **0.493**
- naip: 0.439
- spain_crops: 0.364
- spain_urban: 0.476
- spot: 0.564

> Domain note: references are USA / Spain / WorldStrat sites; the model was trained only on
> Indian cities with synthetic 40->10 m pairs. This is a zero-shot, cross-domain, real-scale test.
