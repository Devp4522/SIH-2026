"""
Phase 2 validation: compare the 2.5 m SR output against REAL high-resolution reference
imagery (e.g. PlanetScope 3 m, SkySat, aerial orthophoto, Cartosat) over the same area
and ideally within a few days of the Sentinel-2 acquisition.

This is the only way to make a claim about <4 m accuracy. The synthetic test set
(evaluate.py) only proves 40 m -> 10 m recovery.

    python scripts/validate_reference.py \
        --input  S2_scene_10m.tif        --bands 1,2,3,4 \
        --sr     out/scene_sr.tif \
        --reference planetscope_3m.tif   --ref-bands 1,2,3,4 \
        --out    out/reference_validation.json

What it does
  1. Reprojects the reference onto the SR grid (average resampling if it is finer).
  2. Bicubic-upsamples the 10 m input onto the same grid -> the baseline to beat.
  3. Radiometric matching: different sensors have different gains/offsets, so each
     reference band is linearly fitted to the (smooth) bicubic image. Fitting to the
     bicubic, not the SR, avoids rewarding the SR model for anything.
  4. Reports, for bicubic and SR vs reference:
       - PSNR / SSIM / correlation per band
       - HIGH-FREQUENCY correlation: correlation of the detail layers (image minus its
         blur). This is the metric that answers "did the SR add real detail, or just
         plausible-looking texture?"  SR > bicubic here = genuine detail recovered.
"""
import argparse
import json

import numpy as np
import torch

import _common  # noqa: F401
from srm.losses import ssim_map
from srm.data import BAND_NAMES, gaussian_kernel1d


def read_on_grid(path, bands, like, resampling):
    import rasterio
    from rasterio.warp import reproject
    with rasterio.open(like) as ref_grid:
        dst = np.full((len(bands), ref_grid.height, ref_grid.width), np.nan, np.float32)
        with rasterio.open(path) as src:
            for i, b in enumerate(bands):
                reproject(rasterio.band(src, b), dst[i], src_transform=src.transform, src_crs=src.crs,
                          src_nodata=src.nodata, dst_transform=ref_grid.transform, dst_crs=ref_grid.crs,
                          dst_nodata=np.nan, resampling=resampling)
    return dst


def high_pass(x, sigma=2.0):
    t = torch.from_numpy(np.nan_to_num(x))[None, None].float()
    k = gaussian_kernel1d(sigma)
    r = k.numel() // 2
    b = torch.nn.functional.pad(t, (r, r, r, r), mode="reflect")
    b = torch.nn.functional.conv2d(b, k.view(1, 1, 1, -1))
    b = torch.nn.functional.conv2d(b, k.view(1, 1, -1, 1))
    return (t - b)[0, 0].numpy()


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def main():
    import rasterio
    from rasterio.enums import Resampling
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="original 10 m Sentinel-2 GeoTIFF")
    ap.add_argument("--bands", default="1,2,3,4")
    ap.add_argument("--sr", required=True, help="SR GeoTIFF from infer.py (B02,B03,B04,B08)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ref-bands", default="1,2,3,4", help="reference band idx matching B02,B03,B04,B08")
    ap.add_argument("--out", default="reference_validation.json")
    a = ap.parse_args()

    with rasterio.open(a.sr) as s:
        sr = s.read().astype(np.float32)
        ref_res = abs(s.transform.a)
    with rasterio.open(a.reference) as r:
        finer = abs(r.transform.a) < ref_res
    ref = read_on_grid(a.reference, [int(b) for b in a.ref_bands.split(",")], a.sr,
                       Resampling.average if finer else Resampling.bilinear)
    bic = read_on_grid(a.input, [int(b) for b in a.bands.split(",")], a.sr, Resampling.cubic)

    valid = np.isfinite(sr).all(0) & np.isfinite(ref).all(0) & np.isfinite(bic).all(0)
    print(f"[ref] overlapping valid pixels: {valid.sum():,} ({valid.mean() * 100:.1f}% of SR grid)")
    if valid.sum() < 10000:
        raise SystemExit("Too little overlap - check CRS / extent / band order.")

    report = {"n_valid_pixels": int(valid.sum()), "bands": {}}
    for i, name in enumerate(BAND_NAMES):
        # radiometric matching: ref -> bicubic scale via least squares
        A = np.stack([ref[i][valid], np.ones(valid.sum())], 1)
        gain, off = np.linalg.lstsq(A, bic[i][valid], rcond=None)[0]
        refm = ref[i] * gain + off
        lo, hi = np.percentile(refm[valid], [1, 99])
        nz = lambda x: np.clip((np.where(valid, x, lo) - lo) / (hi - lo + 1e-6), 0, 1)
        R = nz(refm)
        hp_ref = high_pass(R)
        band = {"ref_gain": float(gain), "ref_offset": float(off)}
        for tag, img in (("bicubic", bic[i]), ("sr", sr[i])):
            X = nz(img)
            mse = float(((X - R) ** 2)[valid].mean())
            ss = ssim_map(torch.from_numpy(X)[None, None], torch.from_numpy(R)[None, None])[0, 0].numpy()
            band[tag] = {"psnr": float(-10 * np.log10(max(mse, 1e-10))),
                         "ssim": float(ss[valid].mean()),
                         "corr": corr(X[valid], R[valid]),
                         "highfreq_corr": corr(high_pass(X)[valid], hp_ref[valid])}
        report["bands"][name] = band
        b, s = band["bicubic"], band["sr"]
        print(f"{name}: PSNR bic {b['psnr']:.2f} -> SR {s['psnr']:.2f} | SSIM {b['ssim']:.3f} -> {s['ssim']:.3f}"
              f" | detail corr {b['highfreq_corr']:.3f} -> {s['highfreq_corr']:.3f}")

    gain_hf = np.mean([report["bands"][b]["sr"]["highfreq_corr"] - report["bands"][b]["bicubic"]["highfreq_corr"]
                       for b in BAND_NAMES])
    report["mean_highfreq_corr_gain"] = float(gain_hf)
    print(f"\nMean detail-correlation gain over bicubic: {gain_hf:+.3f} "
          f"({'SR recovers real detail' if gain_hf > 0.02 else 'no evidence of real detail beyond bicubic'})")
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ref] written {a.out}")


if __name__ == "__main__":
    main()
