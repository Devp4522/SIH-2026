"""
Consolidate per-city/per-scene metadata.csv files produced by the
Sentinel-2 preprocessing notebook (patches/<city>/<scene>/metadata.csv)
into a single master index CSV.

Assumes each metadata.csv has columns:
    patch_id, city, scene, row_start, col_start, patch_size, valid_percentage
and that the corresponding array is at:
    patches/<city>/<scene>/patch_{patch_id:04d}.npy

Usage:
    python consolidate_patches.py --root patches --out patches/master_index.csv
"""

import argparse
import glob
import os
import pandas as pd


def consolidate(root: str, out_path: str) -> pd.DataFrame:
    metadata_files = sorted(glob.glob(os.path.join(root, "*", "*", "metadata.csv")))

    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata.csv files found under {root}/<city>/<scene>/. "
            "Check --root points at the folder containing your city subfolders."
        )

    print(f"Found {len(metadata_files)} metadata.csv files under {root}")

    frames = []
    for meta_path in metadata_files:
        scene_dir = os.path.dirname(meta_path)
        df = pd.read_csv(meta_path)

        if df.empty:
            print(f"  ⚠ skipping empty metadata file: {meta_path}")
            continue

        # Build the absolute-ish path to each patch's .npy file.
        df["npy_path"] = df["patch_id"].apply(
            lambda pid: os.path.join(scene_dir, f"patch_{int(pid):04d}.npy")
        )

        # Sanity check the files actually exist before they end up in the
        # master index — better to catch a missing patch now than at
        # training time.
        missing = df.loc[~df["npy_path"].apply(os.path.exists)]
        if len(missing) > 0:
            print(f"  ⚠ {len(missing)} patch(es) listed in {meta_path} are missing on disk")
            df = df.drop(missing.index)

        frames.append(df)
        print(f"  ✓ {meta_path}: {len(df)} patches")

    master = pd.concat(frames, ignore_index=True)

    # Give every row a globally unique id — per-scene patch_id restarts at 0
    # in every folder, so it is NOT unique across the whole dataset.
    master.insert(0, "global_id", range(len(master)))

    master.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)
    print(f"Total patches       : {len(master)}")
    print(f"Cities               : {sorted(master['city'].unique().tolist())}")
    print(f"Scenes per city      :")
    print(master.groupby("city")["scene"].nunique())
    print(f"\nPatches per city:")
    print(master.groupby("city").size())
    print(f"\nMaster index saved to: {out_path}")

    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="patches", help="Folder containing <city>/<scene>/metadata.csv")
    parser.add_argument("--out", default="patches/master_index.csv", help="Output path for the master CSV")
    args = parser.parse_args()

    consolidate(args.root, args.out)
