"""
Data pipeline: index loading -> leak-free split -> normalisation -> synthetic
degradation (HR 10 m -> LR 40 m, Wald protocol) -> PyTorch Dataset.

Key design decisions (read before changing anything):

1.  PATHS. master_patch_index.csv stores absolute Windows paths from the machine
    that made the patches (C:\\Users\\...\\patches\\<city>\\<scene>\\patch_XXXX.npy).
    `resolve_patch_paths` rewrites them to <data_root>/<city>/<scene>/patch_XXXX.npy,
    so the same CSV works on any machine / OS / Colab.

2.  SPLIT. The five scenes of a city are the SAME ground on different dates, so
    holding out a *scene* for validation still leaks geography (the model has seen
    those exact buildings/fields in the other four dates). Default val_mode is
    "spatial": a strip of each training city (all dates) is held out, and train
    patches that overlap the strip are dropped. The test city is held out entirely.

3.  NORMALISATION. Uses the shared normalization_stats.json (p1/p99 per band).
    Do not recompute it, or results stop being comparable across the team.

4.  DEGRADATION. Implemented once in torch (`blur_downsample`) and reused by the
    dataset, the LR-consistency loss, the uncertainty maps and inference, so every
    part of the system assumes exactly the same sensor model.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

BAND_NAMES = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
RGB_IDX = [2, 1, 0]                        # B04, B03, B02 for true-colour display
RED, NIR = 2, 3


# ----------------------------------------------------------------------------
# Index + paths
# ----------------------------------------------------------------------------
def resolve_patch_paths(df: pd.DataFrame, data_root: str) -> pd.DataFrame:
    """Rewrite patch_path to <data_root>/<city>/<scene>/<file>, OS-independent."""
    def fix(p: str) -> str:
        parts = [x for x in re.split(r"[\\/]+", str(p)) if x]
        if "patches" in parts:
            idx = len(parts) - 1 - parts[::-1].index("patches")
            rel = parts[idx + 1:]
        else:
            rel = parts[-3:]
        return os.path.join(data_root, *rel)

    df = df.copy()
    df["path"] = df["patch_path"].map(fix)
    return df


def load_index(index_csv: str, data_root: str, min_valid_pct: float = 0.0,
               check_exists: bool = True) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    df = resolve_patch_paths(df, data_root)
    if min_valid_pct > 0 and "valid_percentage" in df.columns:
        df = df[df["valid_percentage"] >= min_valid_pct]
    if check_exists:
        missing = ~df["path"].map(os.path.exists)
        if missing.any():
            ex = df.loc[missing, "path"].iloc[0]
            raise FileNotFoundError(
                f"{missing.sum()}/{len(df)} patch files not found. First missing: {ex}\n"
                f"data_root is {data_root!r} - it must be the folder that contains "
                f"Ahmedabad/, Guwahati/, Nashik/, New_Delhi/.")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------------
def split_index(df: pd.DataFrame, test_city: str, val_mode: str = "spatial",
                val_fraction: float = 0.15, seed: int = 42, verbose: bool = True):
    """Returns (train_df, val_df, test_df).

    val_mode="spatial": per training city, the last `val_fraction` of its extent
        along its longer axis becomes validation (all dates), and train patches whose
        footprint overlaps that strip are discarded. No pixel of val geography is
        ever seen during training.
    val_mode="scene": original behaviour (hold out ~val_fraction of each city's
        scenes). Faster to reason about but leaks geography -> optimistic val scores.
    """
    cities = sorted(df["city"].unique())
    no_test = test_city is None or str(test_city).strip().lower() in ("", "none", "null", "all")
    if no_test:  # final deployment model: every city contributes to train/val
        test_df = df.iloc[0:0].copy()
        rest = df
    else:
        if test_city not in cities:
            raise ValueError(f"test_city={test_city!r} not in {cities} (or use 'none' to train on all)")
        test_df = df[df["city"] == test_city].reset_index(drop=True)
        rest = df[df["city"] != test_city]
    rng = np.random.default_rng(seed)
    tr_parts, va_parts, dropped = [], [], 0

    for city, cdf in rest.groupby("city"):
        if val_mode == "scene":
            scenes = sorted(cdf["scene"].unique())
            n_val = max(1, round(len(scenes) * val_fraction))
            vs = set(rng.choice(scenes, size=n_val, replace=False))
            va_parts.append(cdf[cdf["scene"].isin(vs)])
            tr_parts.append(cdf[~cdf["scene"].isin(vs)])
            continue
        if val_mode != "spatial":
            raise ValueError(f"val_mode must be 'spatial' or 'scene', got {val_mode!r}")

        size = cdf["patch_size"].to_numpy()
        r0, c0 = cdf["row_start"].to_numpy(), cdf["col_start"].to_numpy()
        r_span, c_span = (r0 + size).max() - r0.min(), (c0 + size).max() - c0.min()
        start, length = (c0, c_span) if c_span >= r_span else (r0, r_span)
        axis_min = start.min()
        thr = axis_min + length * (1.0 - val_fraction)
        val_m = start >= thr
        train_m = (start + size) <= thr           # footprint fully left of the strip
        if val_m.sum() == 0:                        # tiny city: fall back to last location
            val_m = start == start.max()
            thr = start.max()
            train_m = (start + size) <= thr
        va_parts.append(cdf[val_m])
        tr_parts.append(cdf[train_m])
        dropped += int((~val_m & ~train_m).sum())

    train_df = pd.concat(tr_parts, ignore_index=True)
    val_df = pd.concat(va_parts, ignore_index=True)
    if verbose:
        print(f"[split] mode={val_mode} test_city={test_city}")
        print(f"  train: {len(train_df):5d} patches  {train_df.groupby('city').size().to_dict()}")
        print(f"  val  : {len(val_df):5d} patches  {val_df.groupby('city').size().to_dict()}")
        print(f"  test : {len(test_df):5d} patches  " +
              ("(none - all cities used for training; report earlier held-out results)" if no_test
               else f"(city {test_city}, never seen)"))
        if val_mode == "spatial":
            print(f"  dropped {dropped} boundary patches overlapping the val strip (prevents leakage)")
    return train_df, val_df, test_df


# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------
class BandNormalizer:
    """Per-band affine scaling to ~[0, 1] using shared percentile stats."""

    def __init__(self, stats: dict, clip=(0.0, 1.0)):
        lo, hi = [], []
        for b in BAND_NAMES:
            s = stats[b]
            for a, z in (("p1", "p99"), ("p2", "p98"), ("min", "max")):
                if a in s and z in s:
                    lo.append(float(s[a])); hi.append(float(s[z]))
                    break
            else:
                raise KeyError(f"stats[{b}] needs p1/p99, p2/p98 or min/max; has {list(s)}")
        self.lo = np.array(lo, np.float32)[:, None, None]
        self.hi = np.array(hi, np.float32)[:, None, None]
        self.clip = tuple(clip) if clip is not None else None

    @classmethod
    def from_json(cls, path, clip=(0.0, 1.0)):
        with open(path) as f:
            return cls(json.load(f), clip)

    def norm(self, x: np.ndarray, clip: bool = True) -> np.ndarray:
        y = (x - self.lo) / np.maximum(self.hi - self.lo, 1e-6)
        if clip and self.clip is not None:
            y = np.clip(y, *self.clip)
        return y.astype(np.float32)

    def denorm(self, y):
        """Back to digital numbers. Works for numpy (C,H,W) or torch (B,C,H,W)/(C,H,W)."""
        if isinstance(y, torch.Tensor):
            lo = torch.as_tensor(self.lo, device=y.device, dtype=y.dtype)
            hi = torch.as_tensor(self.hi, device=y.device, dtype=y.dtype)
            return y * (hi - lo) + lo
        return y * (self.hi - self.lo) + self.lo


# ----------------------------------------------------------------------------
# Degradation (single source of truth for the sensor model)
# ----------------------------------------------------------------------------
def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(math.ceil(4.0 * sigma)))  # same truncation as scipy (truncate=4)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur_downsample(x: torch.Tensor, sigma: float, scale: int) -> torch.Tensor:
    """Gaussian PSF blur (separable, reflect-padded) + area downsampling by `scale`.
    x: (B, C, H, W) or (C, H, W); H, W divisible by scale."""
    squeeze = x.dim() == 3
    if squeeze:
        x = x[None]
    b, c, h, w = x.shape
    if sigma > 0:
        k = gaussian_kernel1d(sigma, x.device, x.dtype)
        r = k.numel() // 2
        pad_mode = "reflect" if min(h, w) > r else "replicate"
        x = F.pad(x, (r, r, r, r), mode=pad_mode)
        x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    x = F.avg_pool2d(x, scale)
    return x[0] if squeeze else x


class Degradation:
    """HR -> LR. Random PSF width + noise during training (robustness to the unknown
    real sensor MTF); fixed, noiseless kernel for val/test so scores are deterministic."""

    def __init__(self, scale=4, sigma_range=(0.6, 1.2), sigma_eval=0.9, noise_range=(0.0, 0.01)):
        self.scale = int(scale)
        self.sigma_range = tuple(sigma_range)
        self.sigma_eval = float(sigma_eval)
        self.noise_range = tuple(noise_range)

    def __call__(self, hr: torch.Tensor, train: bool, gen: torch.Generator | None = None):
        if train:
            u = torch.rand(2, generator=gen)
            sigma = self.sigma_range[0] + (self.sigma_range[1] - self.sigma_range[0]) * u[0].item()
            noise = self.noise_range[0] + (self.noise_range[1] - self.noise_range[0]) * u[1].item()
        else:
            sigma, noise = self.sigma_eval, 0.0
        lr = blur_downsample(hr, sigma, self.scale)
        if noise > 0:
            band_std = hr.flatten(-2).std(-1)[..., None, None].clamp(min=1e-3)
            lr = lr + torch.randn(lr.shape, generator=gen) * noise * band_std
        return lr


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 rotations/flips of the square, applied to the last two dims."""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def dihedral_inverse(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.rot90(x, -(k % 4), dims=(-2, -1))
    if k >= 4:
        x = x.flip(-1)
    return x


class SRPatchDataset(Dataset):
    """
    Returns dict:
        lr         (C, h, w)   normalised, synthetic 40 m-equivalent input
        hr         (C, H, W)   normalised 10 m target (NaNs mean-filled)
        mask       (1, H, W)   1 = real pixel, 0 = cloud gap / no-data -> ignore in loss
        lr_mask    (1, h, w)   1 only where every covered HR pixel is valid
        idx, city, scene
    """

    def __init__(self, df, normalizer: BandNormalizer, degradation: Degradation,
                 train: bool, crop_size: int | None = 128, crops_per_patch: int = 1,
                 augment: bool = True, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.norm = normalizer
        self.deg = degradation
        self.scale = degradation.scale
        self.train = train
        self.crop_size = crop_size
        self.cpp = max(1, int(crops_per_patch)) if train else 1
        self.augment = augment and train
        self.seed = seed
        if crop_size is not None and crop_size % self.scale:
            raise ValueError(f"crop_size {crop_size} must be divisible by scale {self.scale}")

    def __len__(self):
        return len(self.df) * self.cpp

    def _generator(self, idx):
        g = torch.Generator()
        if self.train:
            # torch's global RNG is re-seeded per worker and per epoch by the DataLoader,
            # so this gives fresh, reproducible randomness every epoch.
            g.manual_seed(int(torch.randint(0, 2 ** 62, (1,)).item()))
        else:
            g.manual_seed(self.seed * 1_000_003 + idx)
        return g

    def __getitem__(self, i):
        idx = i % len(self.df)
        row = self.df.iloc[idx]
        g = self._generator(i)

        hr = np.load(row["path"]).astype(np.float32)          # (4, H, W) raw DN, may contain NaN
        nan = np.isnan(hr)
        mask = ~nan.any(0) & ~(np.nan_to_num(hr) == 0).all(0)  # invalid if any band NaN or all-zero

        # --- crop (HR size must be a multiple of scale) ---
        c, h, w = hr.shape
        cs = self.crop_size
        if cs is not None and h >= cs and w >= cs:
            if self.train:
                top = int(torch.randint(0, h - cs + 1, (1,), generator=g))
                left = int(torch.randint(0, w - cs + 1, (1,), generator=g))
            else:
                top, left = (h - cs) // 2, (w - cs) // 2
            hr, mask = hr[:, top:top + cs, left:left + cs], mask[top:top + cs, left:left + cs]
        else:
            H, W = (h // self.scale) * self.scale, (w // self.scale) * self.scale
            hr, mask = hr[:, :H, :W], mask[:H, :W]

        # --- fill gaps so the network sees real numbers; the mask keeps them out of the loss ---
        hr = hr.copy()
        for b in range(c):
            band = hr[b]
            bad = np.isnan(band) | ~mask
            if bad.any():
                good = band[~bad]
                band[bad] = good.mean() if good.size else float((self.norm.lo[b] + self.norm.hi[b]) / 2)

        # --- normalise (unclipped), degrade, then clip both ---
        hr_t = torch.from_numpy(self.norm.norm(hr, clip=False))
        mask_t = torch.from_numpy(mask.astype(np.float32))[None]

        if self.augment:
            k = int(torch.randint(0, 8, (1,), generator=g))
            hr_t, mask_t = dihedral(hr_t, k).contiguous(), dihedral(mask_t, k).contiguous()

        lr_t = self.deg(hr_t, train=self.train, gen=g)
        if self.norm.clip is not None:
            lo, hi = self.norm.clip
            hr_t, lr_t = hr_t.clamp(lo, hi), lr_t.clamp(lo, hi)
        lr_mask = -F.max_pool2d(-mask_t[None], self.scale)[0]  # min-pool

        return {"lr": lr_t.float(), "hr": hr_t.float(), "mask": mask_t, "lr_mask": lr_mask,
                "idx": idx, "city": str(row["city"]), "scene": str(row["scene"])}


# ----------------------------------------------------------------------------
# Convenience builder used by train / evaluate / visualize
# ----------------------------------------------------------------------------
def build_normalizer(cfg) -> BandNormalizer:
    return BandNormalizer.from_json(cfg.data.stats_json, clip=cfg.data.get("clip", [0.0, 1.0]))


def build_degradation(cfg) -> Degradation:
    d = cfg.data.degradation
    return Degradation(scale=int(cfg.data.scale), sigma_range=d.sigma_range,
                       sigma_eval=d.sigma_eval, noise_range=d.noise_range)


def make_splits(cfg, verbose=True):
    df = load_index(cfg.data.index_csv, cfg.data.data_root, cfg.data.get("min_valid_pct", 0.0))
    return split_index(df, cfg.data.test_city, cfg.data.val_mode, cfg.data.val_fraction,
                       int(cfg.train.seed), verbose=verbose)