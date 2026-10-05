"""
Stage 2 (split) + Stage 3 (degrade) + Stage 4 (dataloader) of the SR pipeline.

Reads the master_index.csv produced by consolidate_patches.py, splits it
by city/scene (never randomly — see split_master_index), and exposes a
PyTorch Dataset that loads a HR patch, synthetically degrades it to build
the LR input on the fly, and returns normalized (lr, hr, valid_mask)
tensors.

Usage:
    from sr_dataset import split_master_index, SRPatchDataset, compute_band_stats

    train_df, val_df, test_df = split_master_index(
        "patches/master_index.csv",
        test_city="New_Delhi",          # entire city held out for test
        val_fraction_scenes=0.25,        # ~1 scene per remaining city -> val
        seed=42,
    )

    # Compute normalization stats from TRAIN ONLY to avoid leakage.
    band_stats = compute_band_stats(train_df)

    train_ds = SRPatchDataset(train_df, band_stats, scale=4, train=True)
    val_ds   = SRPatchDataset(val_df,   band_stats, scale=4, train=False)
    test_ds  = SRPatchDataset(test_df,  band_stats, scale=4, train=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
import json

BAND_NAMES = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR


# ------------------------------------------------------------------
# STAGE 3 — SPLIT BY SCENE / CITY (never randomly at the patch level)
# ------------------------------------------------------------------
def split_master_index(
    master_csv: str,
    test_city: str,
    val_fraction_scenes: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    - `test_city`: an entire city held out completely for test (checks
      generalization to a new region, not just interpolation).
    - Within the remaining cities, ~val_fraction_scenes of each city's
      scenes are held out for validation.
    - Everything else is train.
    """
    df = pd.read_csv(master_csv)
    rng = np.random.default_rng(seed)

    if test_city not in df["city"].unique():
        raise ValueError(f"test_city={test_city!r} not found. Available: {sorted(df['city'].unique())}")

    test_df = df[df["city"] == test_city].reset_index(drop=True)
    remaining = df[df["city"] != test_city]

    val_rows = []
    train_rows = []
    for city, city_df in remaining.groupby("city"):
        scenes = sorted(city_df["scene"].unique())
        n_val = max(1, round(len(scenes) * val_fraction_scenes))
        val_scenes = set(rng.choice(scenes, size=n_val, replace=False))

        val_rows.append(city_df[city_df["scene"].isin(val_scenes)])
        train_rows.append(city_df[~city_df["scene"].isin(val_scenes)])

    val_df = pd.concat(val_rows, ignore_index=True)
    train_df = pd.concat(train_rows, ignore_index=True)

    print(f"Train: {len(train_df)} patches | cities={sorted(train_df['city'].unique())}")
    print(f"Val  : {len(val_df)} patches | cities={sorted(val_df['city'].unique())}, "
          f"scenes held out: {val_df.groupby('city')['scene'].unique().to_dict()}")
    print(f"Test : {len(test_df)} patches | city={test_city} (fully unseen)")

    return train_df, val_df, test_df


# ------------------------------------------------------------------
# Per-band normalization stats — MUST be computed from train split only
# ------------------------------------------------------------------
def compute_band_stats(train_df: pd.DataFrame, sample_n: int = 200) -> dict:
    """Robust 2nd/98th percentile per band, estimated from a sample of
    training patches (reading all of them would be slow/unnecessary)."""
    path_col = "npy_path" if "npy_path" in train_df.columns else "patch_path"
    sample = train_df.sample(n=min(sample_n, len(train_df)), random_state=0)
    band_values = {b: [] for b in BAND_NAMES}

    for path in sample[path_col]:
        arr = np.load(path)  # (4, H, W)
        for i, b in enumerate(BAND_NAMES):
            vals = arr[i][~np.isnan(arr[i])]
            if vals.size:
                band_values[b].append(vals)

    stats = {}
    for b in BAND_NAMES:
        all_vals = np.concatenate(band_values[b])
        stats[b] = {
            "p2": float(np.percentile(all_vals, 2)),
            "p98": float(np.percentile(all_vals, 98)),
        }
    print("Band stats (from train split only):", stats)
    return stats


def save_band_stats(stats: dict, path: str = "band_stats.json"):
    """Persist band stats to disk so everyone downstream (e.g. your teammate
    training the model) normalizes with the exact same numbers instead of
    each recomputing their own from a random sample."""
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Band stats saved to {path}")


def load_band_stats(path: str = "band_stats.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# STAGE 3 — DEGRADATION (build LR from HR on the fly)
# ------------------------------------------------------------------
def degrade(hr: np.ndarray, scale: int = 4, train: bool = True) -> np.ndarray:
    """
    hr: (C, H, W) float32, may contain NaN.
    Returns lr: (C, H/scale, W/scale) float32, NaN-free (NaNs are filled
    with the per-band mean before blurring so cloud gaps don't smear).
    """
    c, h, w = hr.shape
    lr_bands = []

    # Randomize blur sigma / noise per-call when training, for generalization.
    sigma = np.random.uniform(0.6, 1.2) if train else 0.8
    noise_std = np.random.uniform(0.0, 0.01) if train else 0.0

    for i in range(c):
        band = hr[i]
        nan_mask = np.isnan(band)
        if nan_mask.any():
            fill_val = np.nanmean(band) if not np.all(nan_mask) else 0.0
            band = np.where(nan_mask, fill_val, band)

        blurred = gaussian_filter(band, sigma=sigma)

        # Downsample via area averaging (equivalent to bicubic/area for SR).
        t = torch.from_numpy(blurred).float()[None, None]  # (1,1,H,W)
        down = F.interpolate(t, scale_factor=1 / scale, mode="area")[0, 0].numpy()

        if noise_std > 0:
            down = down + np.random.normal(0, noise_std * np.nanstd(band), down.shape)

        lr_bands.append(down.astype(np.float32))

    return np.stack(lr_bands, axis=0)


# ------------------------------------------------------------------
# STAGE 4 — DATASET
# ------------------------------------------------------------------
class SRPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, band_stats: dict, scale: int = 4, train: bool = True,
                 crop_size: int | None = None):
        """
        crop_size: if set, crop the HR patch down to (crop_size, crop_size) before
        degrading — must be divisible by `scale`. Random crop when train=True,
        center crop otherwise. Use e.g. 256 for fast baseline iteration, None
        (or 512) once the pipeline is confirmed and you're ready for the full run.
        """
        self.df = df.reset_index(drop=True)
        self.band_stats = band_stats
        self.scale = scale
        self.train = train
        self.crop_size = crop_size

    def _crop(self, hr, mask):
        if self.crop_size is None:
            return hr, mask
        c, h, w = hr.shape
        cs = self.crop_size
        if h < cs or w < cs:
            return hr, mask
        if self.train:
            top = np.random.randint(0, h - cs + 1)
            left = np.random.randint(0, w - cs + 1)
        else:
            top, left = (h - cs) // 2, (w - cs) // 2
        return hr[:, top:top + cs, left:left + cs], mask[top:top + cs, left:left + cs]

    def __len__(self):
        return len(self.df)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        # Per-band min-max, using whichever percentile/range keys are present
        # in band_stats — supports p2/p98 (this file's own format), p1/p99
        # (e.g. a full-dataset stats file computed elsewhere), or min/max.
        out = np.empty_like(arr)
        for i, b in enumerate(BAND_NAMES):
            s = self.band_stats[b]
            if "p2" in s and "p98" in s:
                lo, hi = s["p2"], s["p98"]
            elif "p1" in s and "p99" in s:
                lo, hi = s["p1"], s["p99"]
            elif "min" in s and "max" in s:
                lo, hi = s["min"], s["max"]
            else:
                raise KeyError(
                    f"band_stats['{b}'] has none of the expected key pairs "
                    f"(p2/p98, p1/p99, min/max). Found keys: {list(s.keys())}"
                )
            out[i] = np.clip((arr[i] - lo) / max(hi - lo, 1e-6), 0, 1)
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path_col = "npy_path" if "npy_path" in self.df.columns else "patch_path"
        hr = np.load(row[path_col]).astype(np.float32)  # (4, 512, 512), may have NaN

        valid_mask_hr = (~np.isnan(hr)).all(axis=0)  # (H, W) — pixels usable in the loss

        hr, valid_mask_hr = self._crop(hr, valid_mask_hr)

        lr = degrade(hr, scale=self.scale, train=self.train)

        # Fill remaining HR NaNs (per-band mean) AFTER computing the mask,
        # so the network sees real numbers but the loss can still ignore
        # masked-out pixels.
        hr_filled = hr.copy()
        for i in range(hr.shape[0]):
            band = hr_filled[i]
            nan_mask = np.isnan(band)
            if nan_mask.any():
                band[nan_mask] = np.nanmean(band) if not np.all(nan_mask) else 0.0

        hr_norm = self._normalize(hr_filled)
        lr_norm = self._normalize(lr)

        return {
            "lr": torch.from_numpy(lr_norm).float(),
            "hr": torch.from_numpy(hr_norm).float(),
            "valid_mask": torch.from_numpy(valid_mask_hr).float(),  # multiply into loss
            "city": row["city"],
            "scene": row["scene"],
        }
