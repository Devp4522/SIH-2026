"""
Sanity-check the dataset before training. Run this first on any new machine.

    python scripts/check_data.py --config configs/edsr_baseline.yaml
"""
import argparse

import numpy as np

import _common  # noqa: F401
from srm.config import load_config
from srm.data import BAND_NAMES, SRPatchDataset, build_degradation, build_normalizer, make_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=20, help="patches to open for content checks")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)

    train_df, val_df, test_df = make_splits(cfg)   # also verifies every file exists
    print(f"\n[ok] all {len(train_df) + len(val_df) + len(test_df)} referenced patch files found")

    sample = train_df.sample(n=min(a.n, len(train_df)), random_state=0)
    nan_fracs, shapes = [], set()
    for p in sample["path"]:
        arr = np.load(p)
        shapes.add((arr.shape, str(arr.dtype)))
        nan_fracs.append(np.isnan(arr).mean())
    print(f"[ok] shapes/dtypes seen: {shapes}")
    print(f"[ok] NaN fraction in sample: mean {np.mean(nan_fracs):.4f}, max {np.max(nan_fracs):.4f}")
    bad = [s for s in shapes if s[0][0] != 4]
    if bad:
        raise SystemExit(f"[FAIL] expected 4 bands first, got {bad}")

    norm, deg = build_normalizer(cfg), build_degradation(cfg)
    ds = SRPatchDataset(train_df, norm, deg, train=True, crop_size=int(cfg.data.crop_size))
    item = ds[0]
    print(f"[ok] dataset item: lr {tuple(item['lr'].shape)}  hr {tuple(item['hr'].shape)}  "
          f"mask {tuple(item['mask'].shape)} (valid {item['mask'].mean():.3f})")
    for i, b in enumerate(BAND_NAMES):
        print(f"     {b}: hr range [{item['hr'][i].min():.3f}, {item['hr'][i].max():.3f}]  "
              f"lr mean {item['lr'][i].mean():.3f}")

    # leakage check: no train patch footprint may overlap a val patch of the same city
    if cfg.data.val_mode == "spatial":
        overlaps = 0
        for city, v in val_df.groupby("city"):
            t = train_df[train_df.city == city]
            for _, r in v.drop_duplicates(["row_start", "col_start"]).iterrows():
                s = r.patch_size
                ov = ((t.row_start < r.row_start + s) & (t.row_start + t.patch_size > r.row_start) &
                      (t.col_start < r.col_start + s) & (t.col_start + t.patch_size > r.col_start))
                overlaps += int(ov.sum())
        print(f"[{'ok' if overlaps == 0 else 'WARN'}] train/val footprint overlaps: {overlaps}")
    print("\nData looks good. Next: python scripts/train.py --config", a.config, "--dry-run")


if __name__ == "__main__":
    main()
