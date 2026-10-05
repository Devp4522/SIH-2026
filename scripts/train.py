"""
Train a super-resolution model.

    # 1) ALWAYS dry-run first (tiny model, 2 epochs, ~50 patches): proves the pipeline end to end
    python scripts/train.py --config configs/edsr_baseline.yaml --dry-run

    # 2) Stage 1 - the real baseline
    python scripts/train.py --config configs/edsr_baseline.yaml

    # 3) Optional stage 2 - adversarial fine-tune
    python scripts/train.py --config configs/esrgan_finetune.yaml --stage gan

    # Override anything without editing YAML
    python scripts/train.py --config configs/edsr_baseline.yaml --set train.epochs=30 data.test_city=Guwahati
"""
import argparse

import _common  # noqa: F401  (adds project root to sys.path)
from srm.config import load_config
from srm.trainer import run_training

DRY_RUN = ["model.n_resblocks=2", "model.n_feats=16", "model.n_groups=1", "train.epochs=2",
           "train.batch_size=4", "train.val_batch_size=4", "train.warmup_steps=5",
           "train.resume=false", "data.crops_per_patch=1", "data.num_workers=0",
           "_dry_run_limit=48", "gan.epochs=1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["psnr", "gan"], default="psnr")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=10")
    a = ap.parse_args()

    overrides = list(a.set)
    if a.dry_run:
        overrides = DRY_RUN + [f"train.out_dir=runs/dry_run_{a.stage}"] + overrides
        if a.stage == "gan":
            overrides.append("gan.init_from=runs/dry_run_psnr/best.pt")
    cfg = load_config(a.config, overrides)
    run_training(cfg, a.stage)


if __name__ == "__main__":
    main()
