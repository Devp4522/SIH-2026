| Figure | Content | Source files |
|---|---|---|
| `fig01_model_comparison.png` | Final 3-model comparison: synthetic India vs real 2.5 m | `runs/edsr_final_v2/eval_val/summary.json`, `runs/cmp/zeroshot_g/summary.json`, `runs/cmp/ft_real_g/summary.json`, `runs/cmp/ft_joint_g/summary.json` |
| `fig02_unseen_cities.png` | Generalisation to held-out cities, overall + per scene | `runs/edsr_baseline/eval_test/summary.json`, `runs/edsr_guwahati/eval_test/summary.json` |
| `fig03_gan_tradeoff.png` | EDSR vs GAN on unseen Nashik: PSNR vs SAM | `runs/edsr_baseline/eval_test/summary.json` |
| `fig04_uncertainty_calibration.png` | Error vs uncertainty decile | `runs/edsr_baseline/eval_test/calibration.csv`, `runs/edsr_guwahati/eval_test/calibration.csv`, `runs/edsr_ft_joint/eval_val/calibration.csv` |
| `fig05_dataset.png` | Dataset composition and acquisition dates | `master_patch_index.csv` |
| `fig06_training.png` | Stage-1 convergence and stage-2 joint fine-tuning dynamics | `runs/edsr_final_v2/log.csv`, `runs/edsr_ft_joint/log.csv`, `runs/edsr_ft_real/log.csv` |
| `fig07_per_band.png` | Per-band PSNR, bicubic vs final model | `runs/edsr_final_v2/eval_val/summary.json` |
| `fig08_real_per_dataset.png` | Real-reference detail gain per dataset and model | `runs/cmp/zeroshot_g/summary.json`, `runs/cmp/ft_real_g/summary.json`, `runs/cmp/ft_joint_g/summary.json` |
| `fig09_demo_1.png` | Real 10 m -> 2.5 m inference: Nashik city | `demo/nashik_10m.tif`, `demo/nashik_final_2p5m.tif`, `demo/nashik_final_2p5m_uncertainty.tif` |
| `fig09_demo_2.png` | Real 10 m -> 2.5 m inference: Nashik farmland | `demo/nashik_farm_10m.tif`, `demo/nashik_farm_2p5m.tif`, `demo/nashik_farm_2p5m_uncertainty.tif` |
