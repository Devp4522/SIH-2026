"""
PlotSphere SRM - live demo app.

    pip install streamlit
    streamlit run app.py

Two modes:
  1. Super-resolve a scene: upload a 4-band Sentinel-2 GeoTIFF (or pick a file in demo/), choose an
     area, run the model, inspect output / uncertainty / guard / NDVI, download georeferenced GeoTIFFs.
  2. Accuracy check: pick a held-out Indian patch, degrade it 10 m -> 40 m, super-resolve it and score
     the result against the real 10 m patch (PSNR, SSIM, SAM, NDVI error vs bicubic), live.
"""
import base64
import glob
import io
import json
import os
import sys
import time

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from infer import load, run_array  # noqa: E402
from srm.data import (BAND_NAMES, NIR, RED, SRPatchDataset, build_degradation,  # noqa: E402
                      build_normalizer, resolve_patch_paths)
from srm.metrics import per_image_metrics  # noqa: E402
from srm.trainer import get_device  # noqa: E402
from srm.uncertainty import guarded_predict  # noqa: E402

st.set_page_config(page_title="PlotSphere SRM - live demo", page_icon="🛰️", layout="wide")
TEAL = "#1F8A7E"


# ============================================================================ helpers
@st.cache_resource(show_spinner="Loading model...")
def get_model(ckpt):
    device = get_device()
    model, cfg = load(ckpt, device, [])
    norm, deg = build_normalizer(cfg), build_degradation(cfg)
    clip = tuple(cfg.data.get("clip", [0.0, 1.0]))
    return model, cfg, norm, deg, int(cfg.data.scale), clip, device


def list_ckpts():
    c = sorted(glob.glob("runs/*/best.pt") + glob.glob("runs/*/final.pt"))
    c = [p.replace("\\", "/") for p in c if "dry_run" not in p]
    pref = "runs/edsr_ft_joint/best.pt"
    return ([pref] if pref in c else []) + [p for p in c if p != pref]


def stretch_params(arr_dn):
    rgb = arr_dn[[2, 1, 0]]
    v = rgb[np.isfinite(rgb)]
    return np.percentile(v, [2, 98]) if v.size else (0, 1)


def to_rgb(arr_dn, lohi, gamma=0.9):
    lo, hi = lohi
    x = np.nan_to_num(arr_dn[[2, 1, 0]].transpose(1, 2, 0), nan=lo)
    return (np.clip((x - lo) / (hi - lo + 1e-6), 0, 1) ** gamma * 255).astype(np.uint8)


def colorize(a, cmap="magma", vmin=None, vmax=None):
    import matplotlib.cm as cm
    a = np.nan_to_num(a, nan=0.0)
    vmin = np.percentile(a, 1) if vmin is None else vmin
    vmax = np.percentile(a, 99) if vmax is None else vmax
    return (cm.get_cmap(cmap)(np.clip((a - vmin) / (vmax - vmin + 1e-9), 0, 1))[:, :, :3] * 255).astype(np.uint8)


def ndvi(arr_dn, offset):
    nir, red = arr_dn[NIR] - offset, arr_dn[RED] - offset
    return (nir - red) / np.clip(nir + red, 1e-6, None)


def b64jpg(a, q=88):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def upsample_nearest(a, k):
    return np.kron(a, np.ones((k, k, 1), dtype=a.dtype)) if a.ndim == 3 else np.kron(a, np.ones((k, k)))


def swipe(left, right, left_label, right_label, height=640):
    """Before/after slider as a tiny HTML component (no extra packages)."""
    h, w = left.shape[:2]
    html = f"""
    <div id="v" style="position:relative;width:100%;max-width:{height}px;aspect-ratio:{w}/{h};margin:auto;
         overflow:hidden;border-radius:8px;background:#000;touch-action:none;user-select:none">
      <img src="{b64jpg(left)}" style="position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated">
      <img id="a" src="{b64jpg(right)}" style="position:absolute;inset:0;width:100%;height:100%;clip-path:inset(0 0 0 50%)">
      <div id="d" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 0 1px #0006"></div>
      <span style="position:absolute;left:8px;bottom:8px;background:#000b;color:#fff;font:13px sans-serif;padding:3px 7px;border-radius:4px">{left_label}</span>
      <span style="position:absolute;right:8px;bottom:8px;background:#000b;color:#fff;font:13px sans-serif;padding:3px 7px;border-radius:4px">{right_label}</span>
    </div>
    <input id="r" type="range" min="0" max="100" value="50" aria-label="Comparison position"
           style="width:100%;max-width:{height}px;display:block;margin:10px auto 0;accent-color:{TEAL}">
    <script>
      const a=document.getElementById('a'),d=document.getElementById('d'),r=document.getElementById('r'),v=document.getElementById('v');
      const set=p=>{{a.style.clipPath=`inset(0 0 0 ${{p}}%)`;d.style.left=p+'%';r.value=p;}};
      r.oninput=()=>set(r.value); let drag=false;
      const mv=e=>{{const b=v.getBoundingClientRect();set(Math.max(0,Math.min(100,(e.clientX-b.left)/b.width*100)));}};
      v.onpointerdown=e=>{{drag=true;v.setPointerCapture(e.pointerId);mv(e);}}; v.onpointermove=e=>{{if(drag)mv(e);}}; v.onpointerup=()=>drag=false;
    </script>"""
    st.components.v1.html(html, height=int(height * h / w) + 60 if w >= h else height + 60)


def write_geotiff(arr, profile, transform, descriptions):
    from rasterio.io import MemoryFile
    prof = dict(profile)
    prof.update(driver="GTiff", count=arr.shape[0], height=arr.shape[1], width=arr.shape[2], dtype="float32",
                transform=transform, nodata=np.nan, compress="deflate", predictor=3)
    prof.pop("blockxsize", None); prof.pop("blockysize", None); prof.pop("tiled", None)
    with MemoryFile() as mf:
        with mf.open(**prof) as dst:
            dst.write(arr.astype(np.float32))
            dst.descriptions = tuple(descriptions)
        return mf.read()


# ============================================================================ sidebar
st.sidebar.title("PlotSphere SRM")
st.sidebar.caption("Sentinel-2 10 m → 2.5 m grid, with uncertainty")
ckpts = list_ckpts()
if not ckpts:
    st.error("No checkpoint found in runs/. Train a model or copy runs/edsr_ft_joint/best.pt into the project.")
    st.stop()
ckpt = st.sidebar.selectbox("Model", ckpts, help="runs/edsr_ft_joint/best.pt is the final model")
model, cfg, norm, deg, SCALE, CLIP, DEVICE = get_model(ckpt)
mode = st.sidebar.radio("Mode", ["Super-resolve a scene", "Accuracy check on a held-out patch"])
tta = st.sidebar.toggle("8× test-time augmentation (uncertainty)", value=True,
                        help="Runs 8 rotated/flipped passes; their spread is the uncertainty map. 8× slower.")
guard_on = st.sidebar.toggle("Out-of-range guard", value=True,
                             help="Pixels brighter than anything seen in training fall back to bicubic.")
GUARD = (-1e9, CLIP[1]) if guard_on else None
OFFSET = 1000.0
st.sidebar.caption(f"Device: {DEVICE} · clip range {CLIP}")

with open(cfg.data.stats_json) as f:
    STATS = json.load(f)
TRAIN_MEANS = np.array([STATS[b]["mean"] for b in BAND_NAMES])


# ============================================================================ mode 1: scene
def scene_mode():
    st.title("Super-resolve a Sentinel-2 scene")
    st.write("Input: a 4-band GeoTIFF with B02, B03, B04, B08 at 10 m (Level-2A digital numbers, as produced by "
             "`scripts/make_s2_stack.py`). Output: a 2.5 m GeoTIFF in the same CRS plus a 3-band uncertainty file.")
    import rasterio
    demos = sorted(p.replace("\\", "/") for p in glob.glob("demo/*10m*.tif"))
    src_choice = st.sidebar.radio("Input", (["Demo file"] if demos else []) + ["Upload GeoTIFF"])
    bands_txt = st.sidebar.text_input("Band numbers for B02,B03,B04,B08", "1,2,3,4")
    if src_choice == "Demo file":
        path = st.sidebar.selectbox("Demo file", demos)
        data = open(path, "rb").read()
        name = os.path.basename(path)
    else:
        up = st.sidebar.file_uploader("GeoTIFF", type=["tif", "tiff"])
        if up is None:
            st.info("Upload a GeoTIFF in the sidebar, or create one from a Sentinel-2 .SAFE folder with "
                    "`python scripts/make_s2_stack.py --safe <folder> --center <lat> <lon> --size-km 5 --out x.tif`.")
            return
        data, name = up.getvalue(), up.name

    from rasterio.io import MemoryFile
    try:
        bands = [int(b) for b in bands_txt.split(",")]
        with MemoryFile(data) as mf, mf.open() as src:
            full = src.read(bands).astype(np.float32)
            prof, transform, crs = src.profile, src.transform, src.crs
            nodata = src.nodata if src.nodata is not None else 0
            res = abs(transform.a)
    except Exception as e:
        st.error(f"Could not read {name}: {e}. Check that it is a GeoTIFF with at least 4 bands and that the band "
                 "numbers in the sidebar are correct.")
        return

    _, H, W = full.shape
    valid = np.isfinite(full).all(0) & ~(full == nodata).all(0)
    means = np.array([full[i][valid].mean() if valid.any() else np.nan for i in range(4)])
    ratio = means / TRAIN_MEANS

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input size", f"{W} × {H} px")
    c2.metric("Pixel size", f"{res:.1f} m")
    c3.metric("Coverage", f"{W * res / 1000:.1f} × {H * res / 1000:.1f} km")
    c4.metric("Valid pixels", f"{valid.mean() * 100:.1f}%")
    if abs(res - 10) > 0.5:
        st.warning(f"The input pixel size is {res:.1f} m. The model was trained on 10 m Sentinel-2 bands.")
    if np.any((ratio < 0.6) | (ratio > 1.6)):
        st.warning("Band brightness differs strongly from the training data "
                   f"(ratios {', '.join(f'{b} {r:.2f}' for b, r in zip(BAND_NAMES, ratio))}). "
                   "This usually means a different product level or a missing +1000 offset; results may be unreliable.")
    else:
        st.caption("Band brightness vs training data: " + ", ".join(f"{b} {r:.2f}×" for b, r in zip(BAND_NAMES, ratio)))

    # ---- area selection
    max_px = min(H, W, 1000)
    size = st.sidebar.slider("Area to process (pixels of 10 m)", 64, max_px, min(400, max_px), step=16,
                             help="400 px = 4 km. Larger areas take longer; 1000 px ≈ 1.5 min with 8× TTA on a laptop GPU.")
    cx = st.sidebar.slider("Area centre, x (%)", 0, 100, 50)
    cy = st.sidebar.slider("Area centre, y (%)", 0, 100, 50)
    c0 = int(np.clip(cx / 100 * W - size / 2, 0, W - size))
    r0 = int(np.clip(cy / 100 * H - size / 2, 0, H - size))
    lohi = stretch_params(full)
    ov = Image.fromarray(to_rgb(full, lohi))
    ImageDraw.Draw(ov).rectangle([c0, r0, c0 + size, r0 + size], outline=(255, 210, 0), width=max(2, W // 250))
    ov.thumbnail((700, 700))
    left, right = st.columns([1, 1])
    left.image(ov, caption=f"{name}: selected area in yellow ({size * res / 1000:.1f} km × {size * res / 1000:.1f} km)",
               use_container_width=True)
    with right:
        st.markdown("**Settings**")
        st.write(f"- Model: `{ckpt}`\n- Test-time augmentation: {'on (8 passes)' if tta else 'off'}\n"
                 f"- Out-of-range guard: {'on' if guard_on else 'off'}\n- Output: {size * SCALE} × {size * SCALE} px at "
                 f"{res / SCALE:.1f} m")
        go = st.button("Run super-resolution", type="primary", use_container_width=True)

    key = (name, len(data), ckpt, tta, guard_on, size, r0, c0, bands_txt)
    if go:
        crop = full[:, r0:r0 + size, c0:c0 + size]
        t0 = time.time()
        with st.spinner("Super-resolving..."):
            sr, unc = run_array(crop, [model], norm, deg, SCALE, tta, DEVICE, 128, 16, CLIP, GUARD, nodata=nodata)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
        st.session_state["scene"] = dict(key=key, crop=crop, sr=sr, unc=unc, secs=time.time() - t0, lohi=lohi,
                                         prof=prof, transform=transform * transform.translation(c0, r0), name=name)
    res_ = st.session_state.get("scene")
    if not res_ or res_["key"] != key:
        st.info("Choose an area and press **Run super-resolution**.")
        return
    show_scene_result(res_, res)


def show_scene_result(r, res):
    from affine import Affine
    crop, sr, unc, lohi = r["crop"], r["sr"], r["unc"], r["lohi"]
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runtime", f"{r['secs']:.1f} s")
    m2.metric("Output", f"{sr.shape[2]} × {sr.shape[1]} px")
    m3.metric("Guard fallback", f"{np.nanmean(unc[2]) * 100:.1f}% of pixels",
              help="Pixels brighter than the training range; these are bicubic, not model output.")
    m4.metric("Mean uncertainty", f"{np.nanmean(unc[0]):.4f}" if tta else "TTA off",
              help="Average spread of the 8 passes, in normalised units.")

    lr_up = upsample_nearest(to_rgb(crop, lohi), SCALE)
    sr_rgb = to_rgb(sr, lohi)
    n = sr_rgb.shape[0]
    view = st.radio("View", ["Whole area", "Zoom (1:1 pixels)"], horizontal=True)
    if view.startswith("Zoom") and n > 600:
        zc1, zc2 = st.columns(2)
        zx = zc1.slider("Zoom x (%)", 0, 100, 50)
        zy = zc2.slider("Zoom y (%)", 0, 100, 50)
        z = 600
        x0, y0 = int((n - z) * zx / 100), int((n - z) * zy / 100)
        sl = (slice(y0, y0 + z), slice(x0, x0 + z))
    else:
        sl = (slice(None), slice(None))

    def fit(a):
        im = Image.fromarray(a[sl])
        if im.size[0] > 1000:
            im = im.resize((1000, 1000), Image.LANCZOS)
        return np.array(im)

    t1, t2, t3, t4, t5 = st.tabs(["Compare", "Uncertainty", "Guard", "NDVI", "Download"])
    with t1:
        swipe(fit(lr_up), fit(sr_rgb), "Sentinel-2 input, 10 m", "Model output, 2.5 m grid")
        st.caption("Drag across the image. There is no 2.5 m reference for an uploaded scene, so this view is qualitative; "
                   "use the uncertainty tab to see where detail is inferred.")
    with t2:
        if not tta:
            st.info("Turn on test-time augmentation in the sidebar to get the uncertainty map.")
        else:
            swipe(fit(sr_rgb), fit(colorize(unc[0])), "Model output", "Uncertainty (brighter = less sure)")
            st.caption("Spread of 8 rotated/flipped predictions. On real 2.5 m references it correlates with the true "
                       "error at Spearman ρ ≈ 0.50.")
    with t3:
        g = unc[2] > 0.5
        gi = sr_rgb.copy()
        gi[g] = (0.35 * gi[g] + 0.65 * np.array([231, 111, 81])).astype(np.uint8)
        swipe(fit(sr_rgb), fit(gi), "Model output", "Guard fallback in orange")
        st.caption(f"{g.mean() * 100:.1f}% of pixels were brighter than the training range and use bicubic instead.")
    with t4:
        nd_lr = upsample_nearest(ndvi(crop, OFFSET), SCALE)
        nd_sr = ndvi(sr, OFFSET)
        swipe(fit(colorize(nd_lr, "RdYlGn", -0.1, 0.8)), fit(colorize(nd_sr, "RdYlGn", -0.1, 0.8)),
              "NDVI from 10 m input", "NDVI from output")
        st.caption(f"Mean NDVI: input {np.nanmean(nd_lr):.3f}, output {np.nanmean(nd_sr):.3f}. "
                   "The global bicubic skip keeps the large-scale radiometry of the input.")
    with t5:
        tr = r["transform"] * Affine.scale(1 / SCALE)
        base = os.path.splitext(r["name"])[0]
        st.download_button("Download 2.5 m GeoTIFF (B02, B03, B04, B08)",
                           write_geotiff(sr, r["prof"], tr, BAND_NAMES), f"{base}_2p5m.tif", "image/tiff",
                           use_container_width=True)
        st.download_button("Download uncertainty GeoTIFF (TTA std, LR-consistency, guard)",
                           write_geotiff(unc, r["prof"], tr, ["tta_std", "lr_consistency_error", "guard_fallback"]),
                           f"{base}_2p5m_uncertainty.tif", "image/tiff", use_container_width=True)
        b = io.BytesIO()
        Image.fromarray(np.concatenate([lr_up, sr_rgb], 1)).save(b, "PNG")
        st.download_button("Download side-by-side PNG", b.getvalue(), f"{base}_compare.png", "image/png",
                           use_container_width=True)
        st.caption("Both GeoTIFFs keep the input CRS with a 2.5 m transform, so they open directly in QGIS or ArcGIS.")


# ============================================================================ mode 2: accuracy check
@st.cache_data(show_spinner=False)
def val_patches(ckpt_path):
    import pandas as pd
    for p in (os.path.join(os.path.dirname(ckpt_path), "splits", "val.csv"), "runs/edsr_final_v2/splits/val.csv"):
        if os.path.exists(p):
            df = resolve_patch_paths(pd.read_csv(p), cfg.data.data_root)
            return df[df["path"].map(os.path.exists)].reset_index(drop=True)
    return None


def accuracy_mode():
    st.title("Accuracy check on a held-out patch")
    st.write("Pick a patch the model never trained on. The app blurs and downsamples the real 10 m patch to 40 m "
             "(the same sensor model used in training), super-resolves it back, and scores the result against the "
             "real 10 m patch. This is the synthetic benchmark, run live on one patch.")
    df = val_patches(ckpt)
    if df is None or not len(df):
        st.error("No validation split found. Expected runs/<model>/splits/val.csv and the patches/ folder.")
        return
    city = st.sidebar.selectbox("City", sorted(df.city.unique()))
    sub = df[df.city == city].reset_index(drop=True)
    i = st.sidebar.slider("Patch", 0, len(sub) - 1, 0)
    row = sub.iloc[i]
    st.caption(f"{row.city} · {row.scene} · {os.path.basename(row.path)} · spatially held-out validation strip")

    ds = SRPatchDataset(sub.iloc[[i]], norm, deg, train=False, crop_size=None)
    it = ds[0]
    lr, hr, mask = it["lr"][None].to(DEVICE), it["hr"][None].to(DEVICE), it["mask"][None].to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        sr, std, gm = guarded_predict([model], lr, CLIP, GUARD, SCALE, tta=tta)
        bic = F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(*CLIP)
    secs = time.time() - t0
    m_sr = per_image_metrics(sr.clamp(*CLIP), hr, mask, norm, SCALE, 0.0, OFFSET)[0]
    m_bi = per_image_metrics(bic, hr, mask, norm, SCALE, 0.0, OFFSET)[0]

    c = st.columns(5)
    c[0].metric("PSNR", f"{m_sr['psnr']:.2f} dB", f"{m_sr['psnr'] - m_bi['psnr']:+.2f} dB vs bicubic")
    c[1].metric("SSIM", f"{m_sr['ssim']:.3f}", f"{m_sr['ssim'] - m_bi['ssim']:+.3f}")
    c[2].metric("Spectral angle", f"{m_sr['sam_deg']:.2f}°", f"{m_sr['sam_deg'] - m_bi['sam_deg']:+.2f}°", delta_color="inverse")
    c[3].metric("NDVI error", f"{m_sr['ndvi_mae']:.4f}", f"{m_sr['ndvi_mae'] - m_bi['ndvi_mae']:+.4f}", delta_color="inverse")
    c[4].metric("Runtime", f"{secs:.2f} s", f"{'8× TTA' if tta else 'single pass'}", delta_color="off")

    dn = lambda t: norm.denorm(t[0].float().cpu().numpy())
    lr_dn, bic_dn, sr_dn, hr_dn = dn(lr), dn(bic), dn(sr), dn(hr)
    lohi = stretch_params(hr_dn)
    lr_rgb = upsample_nearest(to_rgb(lr_dn, lohi), SCALE)
    t1, t2, t3, t4 = st.tabs(["Model vs truth", "Bicubic vs model", "Error and uncertainty", "Four-panel"])
    with t1:
        swipe(to_rgb(sr_dn, lohi), to_rgb(hr_dn, lohi), "Model output", "Real 10 m patch")
    with t2:
        swipe(to_rgb(bic_dn, lohi), to_rgb(sr_dn, lohi), "Bicubic", "Model output")
    with t3:
        err = (sr - hr).abs().mean(1)[0].cpu().numpy() * mask[0, 0].cpu().numpy()
        if std is not None:
            u = std.mean(1)[0].cpu().numpy()
            swipe(colorize(err), colorize(u), "Actual error", "Predicted uncertainty")
            from scipy.stats import spearmanr
            k = np.random.default_rng(0).choice(err.size, 20000, replace=False)
            rho = spearmanr(err.ravel()[k], u.ravel()[k]).correlation
            st.caption(f"On this patch the uncertainty map correlates with the actual error at Spearman ρ = {rho:.2f}.")
        else:
            st.image(colorize(err), caption="Absolute error (turn on TTA for the uncertainty map)", use_container_width=True)
    with t4:
        cols = st.columns(4)
        for col, img, cap in zip(cols, [lr_rgb, to_rgb(bic_dn, lohi), to_rgb(sr_dn, lohi), to_rgb(hr_dn, lohi)],
                                 ["Input (40 m eq.)", f"Bicubic {m_bi['psnr']:.2f} dB", f"Model {m_sr['psnr']:.2f} dB",
                                  "Real 10 m"]):
            col.image(img, caption=cap, use_container_width=True)
    st.caption("Metrics are masked to valid pixels; SAM and NDVI error use reflectance after removing the +1000 "
               "offset, exactly as in the report.")


if mode.startswith("Super"):
    scene_mode()
else:
    accuracy_mode()
st.sidebar.divider()
st.sidebar.caption("Team PlotSphere · SIH · Contains modified Copernicus Sentinel data")
