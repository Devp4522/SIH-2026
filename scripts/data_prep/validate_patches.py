"""
Run this on your master_index.csv before doing anything else with the model.
Checks the things the external review flagged: reflectance scaling, NaN
percentage, band correspondence, and basic sanity per city.
"""
import numpy as np
import pandas as pd

def validate(master_csv: str, sample_per_city: int = 5):
    df = pd.read_csv(master_csv)
    issues = []

    # Your consolidation script may have named this column npy_path or
    # patch_path depending on which version generated the CSV — handle both.
    path_col = "npy_path" if "npy_path" in df.columns else "patch_path"
    if path_col not in df.columns:
        raise KeyError(
            f"Couldn't find a patch path column. Available columns: {df.columns.tolist()}"
        )

    for city, group in df.groupby("city"):
        sample = group.sample(n=min(sample_per_city, len(group)), random_state=0)
        for _, row in sample.iterrows():
            arr = np.load(row[path_col])  # (4, H, W)
            if arr.shape[0] != 4:
                issues.append(f"{row[path_col]}: expected 4 bands, got {arr.shape[0]}")

            nan_pct = 100 * np.isnan(arr).mean()
            if nan_pct > 15:
                issues.append(f"{row[path_col]}: {nan_pct:.1f}% NaN (higher than expected)")

            # Sentinel-2 L2A reflectance should be in [0, ~10000] pre-scaling,
            # or [0, 1] if you already divided by 10000. Catch whichever is broken.
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                issues.append(f"{row[path_col]}: entirely NaN")
                continue
            vmax = finite.max()
            if vmax > 20000:
                issues.append(f"{row[path_col]}: max value {vmax:.0f} — check reflectance scaling")
            if vmax <= 1.0 and finite.mean() < 0.001:
                issues.append(f"{row[path_col]}: suspiciously low values — check for accidental double-scaling")

            for i, band in enumerate(["B02", "B03", "B04", "B08"]):
                b = arr[i][np.isfinite(arr[i])]
                if b.size and (b.min() < 0):
                    issues.append(f"{row[path_col]} [{band}]: negative reflectance values present")

    print(f"Checked {sum(min(sample_per_city, len(g)) for _, g in df.groupby('city'))} sample patches "
          f"across {df['city'].nunique()} cities")
    if issues:
        print(f"\n{len(issues)} issue(s) found:")
        for i in issues:
            print(" -", i)
    else:
        print("No issues found in sampled patches.")

    return issues


if __name__ == "__main__":
    import sys
    validate(sys.argv[1] if len(sys.argv) > 1 else "patches/master_index.csv")
