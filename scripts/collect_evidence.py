"""
Collect every fact the report / PPT / README needs into ONE markdown file.

    python scripts/collect_evidence.py            # -> evidence.md in the project root

Read-only: it never modifies runs, data or checkpoints. Each section is wrapped so a
missing file only produces a "MISSING" line instead of stopping the script.
"""
import glob
import json
import os
import platform
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
OUT = []


def w(s=""):
    OUT.append(str(s))


def section(title):
    def deco(fn):
        def run():
            w(f"\n## {title}\n")
            try:
                fn()
            except Exception as e:  # keep going no matter what
                w(f"**ERROR in section:** `{e}`")
                w("```\n" + traceback.format_exc(limit=2) + "```")
        return run
    return deco


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def mb(n):
    return f"{n / 1e6:,.1f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


# --------------------------------------------------------------------------------------------
@section("1. Environment and hardware")
def env():
    w(f"- OS: {platform.platform()}")
    w(f"- Python: {sys.version.split()[0]}")
    for mod in ("torch", "torchvision", "numpy", "pandas", "scipy", "rasterio", "matplotlib", "yaml"):
        try:
            m = __import__(mod)
            w(f"- {mod}: {getattr(m, '__version__', '?')}")
        except Exception:
            w(f"- {mod}: NOT INSTALLED")
    import torch
    w(f"- CUDA available: {torch.cuda.is_available()} | torch CUDA build: {torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        w(f"- GPU: {p.name} | VRAM {p.total_memory / 1e9:.1f} GB")


@section("2. Dataset (master_patch_index.csv, metadata, normalisation)")
def dataset():
    import numpy as np
    import pandas as pd
    df = pd.read_csv("master_patch_index.csv")
    w(f"- rows: {len(df)} | columns: `{list(df.columns)}`")
    w(f"- cities: {sorted(df.city.unique())} | scenes: {df.scene.nunique()}")
    if "patch_size" in df:
        w(f"- patch_size values: {sorted(df.patch_size.unique().tolist())}")
    if "valid_percentage" in df:
        d = df.valid_percentage.describe()
        w(f"- valid_percentage: min {d['min']:.2f} | mean {d['mean']:.2f} | median {d['50%']:.2f}")
    w("\n**Patches per city / scene:**\n")
    t = df.groupby(["city", "scene"]).size().rename("patches").reset_index()
    w("```\n" + t.to_string(index=False) + "\n```")
    w("\n**Unique locations per city (row_start, col_start):**\n")
    if {"row_start", "col_start"} <= set(df.columns):
        u = df.groupby("city").apply(lambda g: g[["row_start", "col_start"]].drop_duplicates().shape[0])
        w("```\n" + u.to_string() + "\n```")
    for f in sorted(glob.glob("patches/*/all_scenes_metadata.csv")):
        m = pd.read_csv(f)
        w(f"\n**{f}** ({len(m)} rows) columns: `{list(m.columns)}`")
        w("```\n" + m.head(3).to_string() + "\n```")
    ex = sorted(glob.glob("patches/*/*/metadata.csv"))
    if ex:
        m = pd.read_csv(ex[0])
        w(f"\n**Example scene metadata {ex[0]}** columns: `{list(m.columns)}`")
        w("```\n" + m.head(3).to_string() + "\n```")
    for f in ("normalization_stats.json", "patches/normalization_stats.json"):
        if os.path.exists(f):
            w(f"\n**{f}:**\n```json\n{open(f).read()}\n```")
    npys = sorted(glob.glob("patches/*/*/patch_*.npy"))
    w(f"\n- .npy patch files on disk: {len(npys)} | total size {mb(size_of('patches'))}")
    if npys:
        a = np.load(npys[0], mmap_mode="r")
        w(f"- example patch {npys[0]}: shape {a.shape}, dtype {a.dtype}")


@section("3. Train / val / test splits per run")
def splits():
    import pandas as pd
    for d in sorted(glob.glob("runs/*/splits")):
        parts = []
        for s in ("train", "val", "test"):
            p = os.path.join(d, f"{s}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                by = df.groupby("city").size().to_dict() if len(df) else {}
                parts.append(f"{s}={len(df)} {by}")
        w(f"- **{d}**: " + " | ".join(parts))


@section("4. Config files")
def configs():
    for f in sorted(glob.glob("configs/*.yaml")) + sorted(glob.glob("runs/*/config.yaml")):
        w(f"\n**{f}**\n```yaml\n{open(f).read().strip()}\n```")


@section("5. Checkpoints (stage, epoch, params, embedded config)")
def ckpts():
    import torch
    for f in sorted(glob.glob("runs/*/*.pt")):
        sz = os.path.getsize(f)
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            w(f"- {f} ({mb(sz)}): could not load: {e}")
            continue
        n = sum(v.numel() for v in ck.get("model", {}).values()) if isinstance(ck.get("model"), dict) else "?"
        cfg = ck.get("cfg", {}) or {}
        d, m, t = cfg.get("data", {}), cfg.get("model", {}), cfg.get("train", {})
        pm = f"{n / 1e6:.3f}M" if isinstance(n, int) else "?"
        w(f"- **{f}** ({mb(sz)}) stage={ck.get('stage')} epoch={ck.get('epoch')} params={pm}")
        w(f"  - model: `{m}`")
        w(f"  - data: test_city={d.get('test_city')} clip={d.get('clip')} crop={d.get('crop_size')} "
          f"scale={d.get('scale')} val_mode={d.get('val_mode')} degradation={d.get('degradation')}")
        w(f"  - train: epochs={t.get('epochs')} bs={t.get('batch_size')} lr={t.get('lr')} loss={t.get('loss')}")
        if cfg.get("gan"):
            w(f"  - gan: `{cfg.get('gan')}`")
        if cfg.get("finetune"):
            w(f"  - finetune: `{cfg.get('finetune')}`")
        if ck.get("val"):
            w(f"  - stored val: `{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ck['val'].items()} }`")


@section("6. Training logs (convergence, time)")
def logs():
    import pandas as pd
    for f in sorted(glob.glob("runs/*/log.csv")):
        df = pd.read_csv(f)
        w(f"\n**{f}**: {len(df)} rows | columns `{list(df.columns)}`")
        if "time_s" in df:
            w(f"- epoch time: mean {df.time_s.mean():.1f}s | total {df.time_s.sum() / 3600:.2f} h")
        key = "val_psnr" if "val_psnr" in df else None
        if key:
            b = df.loc[df[key].idxmax()]
            w(f"- best {key}: {b[key]:.3f} at epoch {int(b['epoch'])}")
        cols = [c for c in df.columns if c in ("epoch", "train_total", "loss", "val_psnr", "val_ssim", "val_sam_deg",
                                                "val_gain_over_bicubic_db", "val_hf", "val_hf_bic",
                                                "val_psnr_bic", "india_gain_db", "time_s")]
        w("```\n" + pd.concat([df[cols].head(2), df[cols].tail(2)]).to_string(index=False) + "\n```")


@section("7. All evaluation summaries (verbatim)")
def evals():
    for f in sorted(glob.glob("runs/**/summary.md", recursive=True)):
        w(f"\n---\n### {f}\n")
        w(open(f, encoding="utf-8").read().strip())


@section("8. Calibration tables")
def calib():
    for f in sorted(glob.glob("runs/**/calibration.csv", recursive=True)):
        w(f"\n**{f}**\n```\n{open(f).read().strip()}\n```")


@section("9. Figure inventory")
def figs():
    for f in sorted(glob.glob("**/*.png", recursive=True)):
        if ".ipynb_checkpoints" in f or f.startswith("S2") or "HTML" in f:
            continue
        w(f"- {f} ({mb(os.path.getsize(f))})")


@section("10. Demo rasters")
def demo():
    import rasterio
    for f in sorted(glob.glob("demo/*.tif")):
        with rasterio.open(f) as s:
            w(f"- **{f}** ({mb(os.path.getsize(f))}): {s.width}x{s.height} px, {s.count} bands, "
              f"res {abs(s.transform.a):.2f} m, CRS {s.crs}, dtype {s.dtypes[0]}, bands {s.descriptions}")


@section("11. Repository hygiene (sizes, junk)")
def hygiene():
    items = [p for p in sorted(os.listdir(".")) if not p.startswith(".git")]
    for p in items:
        w(f"- {p}: {mb(size_of(p))}")
    for d in sorted(glob.glob("runs/*")):
        w(f"  - {d}: {mb(size_of(d))}")
    junk = sorted(set(glob.glob("**/.ipynb_checkpoints", recursive=True) + glob.glob("**/__pycache__", recursive=True)))
    w(f"\n- junk folders: {junk}")
    big = []
    for dp, _, fs in os.walk("."):
        if "patches" in dp or ".SAFE" in dp:
            continue
        for f in fs:
            p = os.path.join(dp, f)
            if os.path.getsize(p) > 50e6:
                big.append(f"{p} ({mb(os.path.getsize(p))})")
    w(f"- files > 50 MB (outside patches/ and .SAFE): {big}")
    w(f"- .gitignore present: {os.path.exists('.gitignore')}")


if __name__ == "__main__":
    w("# Evidence dump for SIH report / PPT / README\n")
    for fn in (env, dataset, splits, configs, ckpts, logs, evals, calib, figs, demo, hygiene):
        fn()
    with open("evidence.md", "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"wrote evidence.md ({len(OUT)} lines)")
