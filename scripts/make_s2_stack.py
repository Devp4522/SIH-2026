"""
Turn a downloaded Sentinel-2 product (.SAFE folder, unzipped) into the 4-band GeoTIFF
that infer.py expects: B02, B03, B04, B08 at 10 m, raw digital numbers.

    # crop a 10 km x 10 km window around a point (recommended for the demo)
    python scripts/make_s2_stack.py --safe S2B_MSIL2A_20250328T....SAFE \
        --center 20.00 73.79 --size-km 10 --out demo/nashik_10m.tif

    # whole tile (10980 x 10980 px) - works, but inference output is ~1.9 GB
    python scripts/make_s2_stack.py --safe S2B_MSIL2A_....SAFE --out demo/tile_10m.tif

It also prints the band means next to normalization_stats.json. If they differ wildly
(e.g. 2x), the scene was processed differently from the training patches (L1C vs L2A,
or the +1000 offset of processing baseline >= 04.00) and the SR output will be off.
"""
import argparse
import glob
import json
import os

import numpy as np

import _common  # noqa: F401
from srm.data import BAND_NAMES


def find_band(safe, band):
    pats = [f"GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2",   # L2A
            f"GRANULE/*/IMG_DATA/*_{band}.jp2",             # L1C
            f"**/*{band}*.jp2", f"**/*{band}*.tif"]         # anything else
    for p in pats:
        hits = sorted(glob.glob(os.path.join(safe, p), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{band} not found under {safe}")


def main():
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window

    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", required=True, help="unzipped .SAFE folder (or any folder with the band files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"), help="crop centre (WGS84)")
    ap.add_argument("--size-km", type=float, default=10.0)
    ap.add_argument("--stats", default="normalization_stats.json")
    a = ap.parse_args()

    paths = [find_band(a.safe, b) for b in BAND_NAMES]
    for b, p in zip(BAND_NAMES, paths):
        print(f"  {b}: {os.path.basename(p)}")

    with rasterio.open(paths[0]) as ref:
        if a.center:
            xs, ys = warp_transform("EPSG:4326", ref.crs, [a.center[1]], [a.center[0]])
            row, col = ref.index(xs[0], ys[0])
            half = int(a.size_km * 1000 / abs(ref.transform.a) / 2)
            r0, c0 = max(0, row - half), max(0, col - half)
            r1, c1 = min(ref.height, row + half), min(ref.width, col + half)
            if r1 <= r0 or c1 <= c0:
                raise SystemExit("Centre point is outside this tile - pick the tile that covers it.")
            win = Window(c0, r0, c1 - c0, r1 - r0)
        else:
            win = Window(0, 0, ref.width, ref.height)
        prof = ref.profile.copy()
        prof.update(driver="GTiff", count=4, dtype="float32", width=int(win.width), height=int(win.height),
                    transform=ref.window_transform(win), nodata=0, compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with rasterio.open(a.out, "w", **prof) as dst:
        for i, p in enumerate(paths, start=1):
            with rasterio.open(p) as src:
                arr = src.read(1, window=win).astype(np.float32)
            dst.write(arr, i)
        dst.descriptions = tuple(BAND_NAMES)
    print(f"[stack] wrote {a.out}  ({int(win.height)} x {int(win.width)} px, 10 m)")

    if os.path.exists(a.stats):
        stats = json.load(open(a.stats))
        with rasterio.open(a.out) as s:
            data = s.read()
        valid = (data > 0).all(0)
        print("\n  band   scene mean   training mean   ratio")
        for i, b in enumerate(BAND_NAMES):
            m = float(data[i][valid].mean()) if valid.any() else float("nan")
            t = stats[b]["mean"]
            flag = "" if 0.6 < m / t < 1.6 else "   <-- CHECK processing level / offset"
            print(f"  {b}   {m:10.1f}   {t:13.1f}   {m / t:5.2f}{flag}")


if __name__ == "__main__":
    main()