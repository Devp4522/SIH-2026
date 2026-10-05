"""
Super-resolve a full Sentinel-2 scene (native 10 m) -> 2.5 m GeoTIFF, with uncertainty.

    python scripts/infer.py --ckpt runs/edsr_baseline/best.pt \
        --input S2_scene_B02_B03_B04_B08.tif --output out/scene_sr.tif --tta

    # input bands in a different order? give the 1-based band index of B02,B03,B04,B08:
    python scripts/infer.py ... --bands 2,3,4,8

    # also works on a single .npy patch (4,H,W) for quick checks
    python scripts/infer.py --ckpt runs/edsr_baseline/best.pt --input some_patch.npy --output sr.npy

Outputs
    <output>.tif              4 bands (B02,B03,B04,B08), digital numbers, pixel size / scale,
                              same CRS, transform rescaled -> drops straight into QGIS / ArcGIS
    <output>_uncertainty.tif  band 1: TTA std (normalised units, 0 = models agree)   [needs --tta]
                              band 2: LR-consistency error |blur_down(SR) - input|
                              band 3: 1 = input outside the training range, output is
                                      bicubic (model abstained) - e.g. very bright roofs
                              nodata = input nodata

Memory-safe: the scene is processed tile by tile with a halo (overlap) and written
window by window, so a full 10980 x 10980 tile works on a laptop GPU.

IMPORTANT (scale assumption): the model was trained on 40 m -> 10 m pairs and is applied
here to 10 m -> 2.5 m. This relies on the image statistics being similar across scales
(standard Wald-protocol assumption). Treat the 2.5 m output as model-inferred detail;
the uncertainty layers tell you where to trust it least.
"""
import argparse
import os

import numpy as np
import torch

import _common  # noqa: F401
from srm.config import _wrap, apply_overrides, to_plain
from srm.data import build_degradation, build_normalizer
from srm.models import build_model
from srm.trainer import get_device
from srm.uncertainty import consistency_map, guarded_predict


def load(ckpt, device, overrides):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = _wrap(apply_overrides(to_plain(ck["cfg"]), overrides))
    m = build_model(cfg.model, int(cfg.data.scale)).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, cfg


@torch.no_grad()
def sr_tile(models, arr_dn, valid, norm, deg, scale, tta, device, clip_range, guard_range):
    """arr_dn: (4,h,w) float32 DN; valid: (h,w) bool.
    Returns sr_dn (4,H,W) and unc (3,H,W): [tta_std, lr_consistency_error, out_of_range]."""
    x = arr_dn.copy()
    for b in range(x.shape[0]):
        good = x[b][valid]
        x[b][~valid] = good.mean() if good.size else (norm.lo[b, 0, 0] + norm.hi[b, 0, 0]) / 2
    x_raw = torch.from_numpy(norm.norm(x, clip=False))[None].to(device)
    sr, std, gm = guarded_predict(models, x_raw, clip_range, guard_range, scale, tta=tta)
    lr = x_raw.clamp(*clip_range)
    cons = consistency_map(sr.clamp(*clip_range), lr, deg.sigma_eval, scale)[0, 0]
    unc0 = std.mean(1)[0] if std is not None else torch.zeros_like(cons)
    sr_dn = norm.denorm(sr[0]).cpu().numpy().astype(np.float32)
    unc = torch.stack([unc0, cons, gm[0, 0]]).cpu().numpy().astype(np.float32)
    return sr_dn, unc


def iter_tiles(h, w, tile, halo):
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            r1, c1 = min(r0 + tile, h), min(c0 + tile, w)
            R0, C0 = max(0, r0 - halo), max(0, c0 - halo)
            R1, C1 = min(h, r1 + halo), min(w, c1 + halo)
            yield (r0, r1, c0, c1), (R0, R1, C0, C1)


def run_array(arr, models, norm, deg, scale, tta, device, tile, halo, clip_range, guard_range, nodata=None):
    c, h, w = arr.shape
    arr = arr.astype(np.float32)
    valid = ~np.isnan(arr).any(0)
    if nodata is not None:
        valid &= ~(arr == nodata).all(0)
    out = np.full((c, h * scale, w * scale), np.nan, np.float32)
    unc = np.full((3, h * scale, w * scale), np.nan, np.float32)
    for (r0, r1, c0, c1), (R0, R1, C0, C1) in iter_tiles(h, w, tile, halo):
        sub, v = arr[:, R0:R1, C0:C1], valid[R0:R1, C0:C1]
        if not v.any():
            continue
        s, u = sr_tile(models, sub, v, norm, deg, scale, tta, device, clip_range, guard_range)
        rs, cs = (r0 - R0) * scale, (c0 - C0) * scale
        H, W = (r1 - r0) * scale, (c1 - c0) * scale
        out[:, r0 * scale:r1 * scale, c0 * scale:c1 * scale] = s[:, rs:rs + H, cs:cs + W]
        unc[:, r0 * scale:r1 * scale, c0 * scale:c1 * scale] = u[:, rs:rs + H, cs:cs + W]
    vmask = np.kron(valid, np.ones((scale, scale), bool))
    out[:, ~vmask] = np.nan
    unc[:, ~vmask] = np.nan
    return out, unc


def run_geotiff(args, models, norm, deg, scale, device):
    import rasterio
    from rasterio.windows import Window
    from affine import Affine

    bands = [int(b) for b in args.bands.split(",")]
    with rasterio.open(args.input) as src:
        nodata = src.nodata if src.nodata is not None else 0
        prof = src.profile.copy()
        prof.update(driver="GTiff", width=src.width * scale, height=src.height * scale,
                    transform=src.transform * Affine.scale(1 / scale), count=4, dtype="float32",
                    nodata=np.nan, tiled=True, blockxsize=512, blockysize=512,
                    compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
        uprof = dict(prof, count=3)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        unc_path = os.path.splitext(args.output)[0] + "_uncertainty.tif"
        with rasterio.open(args.output, "w", **prof) as dst, rasterio.open(unc_path, "w", **uprof) as udst:
            dst.descriptions = ("B02", "B03", "B04", "B08")
            udst.descriptions = ("tta_std", "lr_consistency_error", "out_of_range_bicubic_fallback")
            tiles = list(iter_tiles(src.height, src.width, args.tile, args.halo))
            for i, ((r0, r1, c0, c1), (R0, R1, C0, C1)) in enumerate(tiles):
                arr = src.read(bands, window=Window(C0, R0, C1 - C0, R1 - R0)).astype(np.float32)
                v = ~np.isnan(arr).any(0) & ~(arr == nodata).all(0)
                ow = Window(c0 * scale, r0 * scale, (c1 - c0) * scale, (r1 - r0) * scale)
                if not v.any():
                    continue
                s, u = sr_tile(models, arr, v, norm, deg, scale, args.tta, device, args.clip_range, args.guard_range)
                rs, cs = (r0 - R0) * scale, (c0 - C0) * scale
                H, W = (r1 - r0) * scale, (c1 - c0) * scale
                vm = np.kron(v, np.ones((scale, scale), bool))[rs:rs + H, cs:cs + W]
                s, u = s[:, rs:rs + H, cs:cs + W], u[:, rs:rs + H, cs:cs + W]
                s[:, ~vm] = np.nan
                u[:, ~vm] = np.nan
                dst.write(s, window=ow)
                udst.write(u, window=ow)
                if i % 20 == 0:
                    print(f"  tile {i + 1}/{len(tiles)}")
    print(f"[infer] wrote {args.output} and {unc_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="several ckpts = ensemble")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bands", default="1,2,3,4", help="1-based GeoTIFF band indices of B02,B03,B04,B08")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tile", type=int, default=128, help="LR tile core size")
    ap.add_argument("--halo", type=int, default=16, help="LR overlap on each side (hides seams)")
    ap.add_argument("--no-guard", action="store_true",
                    help="disable the out-of-range guard (bicubic fallback outside the training range)")
    ap.add_argument("--set", nargs="*", default=[], help="e.g. data.stats_json=/path/stats.json")
    a = ap.parse_args()

    device = get_device()
    loaded = [load(p, device, a.set) for p in a.ckpt]
    models, cfg = [m for m, _ in loaded], loaded[0][1]
    norm, deg, scale = build_normalizer(cfg), build_degradation(cfg), int(cfg.data.scale)
    a.clip_range = tuple(cfg.data.get("clip", [0.0, 1.0]))
    a.guard_range = None if a.no_guard else (-1e9, a.clip_range[1])  # guard saturation only

    if a.input.lower().endswith(".npy"):
        sr, unc = run_array(np.load(a.input), models, norm, deg, scale, a.tta, device, a.tile, a.halo,
                            a.clip_range, a.guard_range)
        np.save(a.output, sr)
        np.save(os.path.splitext(a.output)[0] + "_uncertainty.npy", unc)
        print(f"[infer] {a.input} -> {a.output}  shape {sr.shape}")
    else:
        run_geotiff(a, models, norm, deg, scale, device)


if __name__ == "__main__":
    main()
