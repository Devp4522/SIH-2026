"""
Generate every data-driven figure for the report / PPT / README from files already in runs/ and demo/.

    python scripts/make_figures.py            # -> report_figures/*.png  (+ figures_manifest.md)

Nothing is re-computed from the model: all numbers come from the saved summary.json /
calibration.csv / log.csv / master_patch_index.csv, so figures always match the tables.
Each figure is independent - a missing input only skips that figure.
"""
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "report_figures"
os.makedirs(OUT, exist_ok=True)
MANIFEST = []

# ---- sources (edit here if a run folder has a different name) ----
SYN_INDIA = "runs/edsr_final_v2/eval_val/summary.json"         # v2 + ft_real + joint, India spatial hold-out
NASHIK = "runs/edsr_baseline/eval_test/summary.json"             # stage-1 + GAN, unseen city
GUWAHATI = "runs/edsr_guwahati/eval_test/summary.json"            # stage-1, unseen city
REAL = {"EDSR-Synthetic (v2)": "runs/cmp/zeroshot_g/summary.json",
        "EDSR-Real-FT": "runs/cmp/ft_real_g/summary.json",
        "EDSR-Joint (final)": "runs/cmp/ft_joint_g/summary.json"}
CALIB = {"Nashik (unseen city)": "runs/edsr_baseline/eval_test/calibration.csv",
         "Guwahati (unseen city)": "runs/edsr_guwahati/eval_test/calibration.csv",
         "India val, final model": "runs/edsr_ft_joint/eval_val/calibration.csv"}
LOG_STAGE1 = "runs/edsr_final_v2/log.csv"
LOG_JOINT = "runs/edsr_ft_joint/log.csv"
LOG_REAL = "runs/edsr_ft_real/log.csv"
LOG_GAN = {"GAN adv 5e-3": "runs/edsr_gan/log.csv", "GAN adv 1e-3": "runs/edsr_gan_soft/log.csv"}
DEMOS = [("demo/nashik_10m.tif", "demo/nashik_final_2p5m.tif", "demo/nashik_final_2p5m_uncertainty.tif", "Nashik city"),
         ("demo/nashik_farm_10m.tif", "demo/nashik_farm_2p5m.tif", "demo/nashik_farm_2p5m_uncertainty.tif",
          "Nashik farmland")]

C = {"bicubic": "#9e9e9e", "v2": "#4c72b0", "real": "#dd8452", "joint": "#2a9d8f", "gan": "#c44e52"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(p):
    with open(p) as f:
        return json.load(f)


def pick(summary, key):
    """find a method entry in an evaluate.py summary by substring."""
    for k, v in summary.items():
        if key in k:
            return v
    raise KeyError(f"{key} not in {list(summary)}")


def save(fig, name, desc, sources):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append(f"| `{name}` | {desc} | {', '.join(f'`{s}`' for s in sources)} |")
    print("saved", p)


def fig(fn):
    def run():
        try:
            fn()
        except Exception as e:
            print(f"[skip] {fn.__name__}: {e}")
            traceback.print_exc(limit=1)
    return run


def err(ci):  # (mean, lo, hi) -> yerr
    m, lo, hi = ci
    return [[m - lo], [hi - m]]


# ----------------------------------------------------------------------------- figures
@fig
def f1_model_comparison():
    s = load(SYN_INDIA)
    rows = [("EDSR-Synthetic (v2)", "edsr_final_v2", C["v2"]), ("EDSR-Real-FT", "edsr_ft_real", C["real"]),
            ("EDSR-Joint (final)", "edsr_ft_joint", C["joint"])]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, key, col) in enumerate(rows):
        g = pick(s, key)["psnr_gain_vs_bicubic"]
        ax[0].bar(i, g["mean"], color=col, yerr=err((g["mean"], g["ci_lo"], g["ci_hi"])), capsize=4)
        ax[0].text(i, g["ci_hi"] + 0.03, f"+{g['mean']:.2f}", ha="center")
        r = load(REAL[name])["ALL"]["gain"]["hf_corr"]
        ax[1].bar(i, r[0], color=col, yerr=err(r), capsize=4)
        ax[1].text(i, max(r[2], 0) + 0.002, f"{r[0]:+.3f}", ha="center")
    for a in ax:
        a.set_xticks(range(3), [r[0] for r in rows], rotation=12)
        a.axhline(0, color="k", lw=0.8)
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + (hi - lo) * 0.15)
    ax[0].set_title("Synthetic India (spatial hold-out, 238 patches)\nPSNR gain over bicubic (dB), 95% CI")
    ax[1].set_title("Real 2.5 m references, unseen (SPOT + Spain urban, 29 imgs)\ndetail-correlation gain over bicubic, 95% CI")
    save(fig, "fig01_model_comparison.png", "Final 3-model comparison: synthetic India vs real 2.5 m",
         [SYN_INDIA] + list(REAL.values()))


@fig
def f2_unseen_cities():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})
    for i, (city, p, key) in enumerate((("Nashik", NASHIK, "edsr_baseline"), ("Guwahati", GUWAHATI, "edsr_guwahati"))):
        g = pick(load(p), key)["psnr_gain_vs_bicubic"]
        ax[0].bar(i, g["mean"], color=C["v2"], yerr=err((g["mean"], g["ci_lo"], g["ci_hi"])), capsize=4)
        ax[0].text(i, g["ci_hi"] + 0.03, f"+{g['mean']:.2f} dB\nwin {g['win_rate'] * 100:.0f}%", ha="center")
    ax[0].set_xticks([0, 1], ["Nashik\n(semi-arid)", "Guwahati\n(humid)"])
    ax[0].set_title("Held-out city: PSNR gain over bicubic")
    ax[0].set_ylim(0, 1.6)
    for city, p, key, col, off in (("Nashik", "runs/edsr_baseline/eval_test/per_image.csv", "edsr_baseline", C["v2"], -0.2),
                                   ("Guwahati", "runs/edsr_guwahati/eval_test/per_image.csv", "edsr_guwahati", "#8172b2", 0.2)):
        df = pd.read_csv(p)
        b = df[df.method == "bicubic"].set_index(["scene", "idx"]).psnr
        m = df[df.method.str.contains(key)].set_index(["scene", "idx"]).psnr
        d = (m - b.loc[m.index]).groupby(level=0).mean()
        x = np.arange(len(d))
        ax[1].bar(x + off, d.values, width=0.4, color=col, label=city)
    ax[1].set_xticks(np.arange(5), [f"Scene {i + 1}" for i in range(5)])
    ax[1].set_title("Per-scene (date) gain over bicubic (dB)")
    ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.2)
    ax[1].legend(loc="upper center", ncol=2, frameon=False)
    save(fig, "fig02_unseen_cities.png", "Generalisation to held-out cities, overall + per scene",
         [NASHIK, GUWAHATI])


@fig
def f3_gan_tradeoff():
    s = load(NASHIK)
    pts = [("Bicubic", "bicubic", C["bicubic"]), ("EDSR (stage 1)", "edsr_baseline", C["v2"]),
           ("GAN, epoch 1", "gan_soft/best", "#e39a9b"), ("GAN, epoch 15", "gan_soft/final", C["gan"])]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, key, col in pts:
        v = s["bicubic"] if key == "bicubic" else pick(s, key)
        ax.scatter(v["sam_deg"]["mean"], v["psnr"]["mean"], s=90, color=col, zorder=3)
        ax.annotate(name, (v["sam_deg"]["mean"], v["psnr"]["mean"]), xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("SAM (deg)  <- better spectral fidelity")
    ax.set_ylabel("PSNR (dB)  better ->")
    ax.set_title("Nashik test: adversarial training trades\naccuracy for texture (perception-distortion)")
    ax.invert_xaxis()
    save(fig, "fig03_gan_tradeoff.png", "EDSR vs GAN on unseen Nashik: PSNR vs SAM", [NASHIK])


@fig
def f4_calibration():
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (name, p), col in zip(CALIB.items(), (C["v2"], "#8172b2", C["joint"])):
        d = pd.read_csv(p)
        ax.plot(d.std_decile + 1, d.mean_abs_err / d.mean_abs_err.iloc[0], "o-", color=col,
                label=f"{name} (top/bottom = {d.mean_abs_err.iloc[-1] / d.mean_abs_err.iloc[0]:.1f}x)")
    ax.set_xlabel("TTA-uncertainty decile (1 = most certain)")
    ax.set_ylabel("mean |error|, relative to decile 1")
    ax.set_title("Uncertainty is a usable warning signal:\nerror grows monotonically with predicted uncertainty")
    ax.legend(fontsize=8)
    save(fig, "fig04_uncertainty_calibration.png", "Error vs uncertainty decile", list(CALIB.values()))


@fig
def f5_dataset():
    df = pd.read_csv("master_patch_index.csv")
    df["date"] = pd.to_datetime(df.scene.str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv = df.groupby(["city", "scene"]).size().unstack(fill_value=0)
    cities = piv.index.tolist()
    bottom = np.zeros(len(cities))
    for j in range(piv.shape[1]):
        vals = np.array([sorted(piv.loc[c][piv.loc[c] > 0].values.tolist())[j] if (piv.loc[c] > 0).sum() > j else 0
                         for c in cities])
        ax[0].bar(cities, vals, bottom=bottom, color=plt.cm.viridis(j / 5), edgecolor="white")
        bottom += vals
    for i, c in enumerate(cities):
        ax[0].text(i, bottom[i] + 15, str(int(bottom[i])), ha="center")
    ax[0].set_title(f"{len(df)} patches of 512x512 (10 m), 5 dates per city")
    ax[0].set_ylabel("patches")
    ax[0].set_ylim(0, bottom.max() * 1.12)
    for i, c in enumerate(cities):
        d = df[df.city == c].drop_duplicates("scene")
        ax[1].scatter(d.date, [i] * len(d), s=60)
    ax[1].set_yticks(range(len(cities)), cities)
    ax[1].set_title("Acquisition dates (Sentinel-2 scenes)")
    plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right")
    save(fig, "fig05_dataset.png", "Dataset composition and acquisition dates", ["master_patch_index.csv"])


@fig
def f6_training():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    l1 = pd.read_csv(LOG_STAGE1)
    ax[0].plot(l1.epoch, l1.val_gain_over_bicubic_db, color=C["v2"])
    ax[0].set_title("Stage 1 (synthetic): val gain over bicubic (dB)")
    ax[0].set_xlabel("epoch")
    lj, lr_ = pd.read_csv(LOG_JOINT), pd.read_csv(LOG_REAL)
    ax[1].plot(lj.epoch, lj.val_hf - lj.val_hf_bic, color=C["joint"], label="Joint")
    ax[1].plot(lr_.epoch, lr_.val_hf - lr_.val_hf_bic, color=C["real"], label="Real-only")
    ax[1].set_title("Stage 2: real held-out detail-corr gain")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[2].plot(lj.epoch, lj.india_gain_db, color=C["joint"], label="Joint (tracked each epoch)")
    ax[2].axhline(lj.india_gain_db.iloc[0], ls="--", color=C["v2"], label="before fine-tuning")
    ax[2].set_ylim(min(0, lj.india_gain_db.min() * 1.2), max(0.1, lj.india_gain_db.max() * 1.3))
    ax[2].set_title("Stage 2: India synthetic gain retained (dB)")
    ax[2].legend()
    ax[2].set_xlabel("epoch")
    save(fig, "fig06_training.png", "Stage-1 convergence and stage-2 joint fine-tuning dynamics",
         [LOG_STAGE1, LOG_JOINT, LOG_REAL])


@fig
def f7_per_band():
    s = load(SYN_INDIA)
    bands = ["B02", "B03", "B04", "B08"]
    b = [s["bicubic"]["per_band_psnr"][k] for k in bands]
    j = [pick(s, "edsr_ft_joint")["per_band_psnr"][k] for k in bands]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - 0.2, b, 0.4, color=C["bicubic"], label="Bicubic")
    ax.bar(x + 0.2, j, 0.4, color=C["joint"], label="EDSR-Joint")
    for i in range(4):
        ax.text(i + 0.2, j[i] + 0.1, f"+{j[i] - b[i]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["B02 blue", "B03 green", "B04 red", "B08 NIR"])
    ax.set_ylim(min(b) - 2, max(j) + 1)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-band PSNR, India spatial hold-out")
    ax.legend()
    save(fig, "fig07_per_band.png", "Per-band PSNR, bicubic vs final model", [SYN_INDIA])


@fig
def f8_real_per_dataset():
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = (C["v2"], C["real"], C["joint"])
    for i, ((name, p), col) in enumerate(zip(REAL.items(), cols)):
        s = load(p)
        for k, ds in enumerate(("spot", "spain_urban")):
            g = s[ds]["gain"]["hf_corr"]
            ax.bar(k + (i - 1) * 0.27, g[0], 0.27, color=col, yerr=err(g), capsize=3, label=name if k == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0, 1], ["SPOT / WorldStrat (9)", "Spain urban aerial (20)"])
    ax.set_ylabel("detail-correlation gain vs bicubic")
    ax.set_title("Real 2.5 m references (never used in training), 95% CI")
    ax.legend(fontsize=8)
    save(fig, "fig08_real_per_dataset.png", "Real-reference detail gain per dataset and model", list(REAL.values()))


@fig
def f9_demos():
    import rasterio
    for i, (lrp, srp, up, title) in enumerate(DEMOS):
        if not os.path.exists(srp):
            print(f"[skip] demo {srp} missing")
            continue
        with rasterio.open(lrp) as a, rasterio.open(srp) as b, rasterio.open(up) as c:
            L, S, U = a.read([3, 2, 1]).astype(float), b.read([3, 2, 1]), c.read()
        h = L.shape[1]
        n = min(200, h)
        r0 = c0 = (h - n) // 2
        lo, hi = np.nanpercentile(L, [2, 98])
        s = lambda x: np.clip((np.nan_to_num(x).transpose(1, 2, 0) - lo) / (hi - lo), 0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6.3))
        ax[0].imshow(s(L[:, r0:r0 + n, c0:c0 + n]), interpolation="nearest")
        ax[0].set_title("Sentinel-2 input (10 m)")
        ax[1].imshow(s(S[:, r0 * 4:(r0 + n) * 4, c0 * 4:(c0 + n) * 4]))
        ax[1].set_title("EDSR-Joint output (2.5 m grid)")
        u = U[0, r0 * 4:(r0 + n) * 4, c0 * 4:(c0 + n) * 4]
        ax[2].imshow(u, cmap="viridis", vmax=np.nanpercentile(u, 99))
        ax[2].set_title("Uncertainty (TTA std)")
        if U.shape[0] > 2:
            g = U[2, r0 * 4:(r0 + n) * 4, c0 * 4:(c0 + n) * 4]
            ax[2].contour(g > 0.5, levels=[0.5], colors="red", linewidths=0.6)
            ax[2].set_title(f"Uncertainty (TTA std); red = guard fallback ({np.nanmean(U[2]) * 100:.1f}% of scene)")
        for x in ax:
            x.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.suptitle(f"{title}: central {n * 10 / 1000:.1f} km x {n * 10 / 1000:.1f} km", y=0.99, fontsize=13)
        save(fig, f"fig09_demo_{i + 1}.png", f"Real 10 m -> 2.5 m inference: {title}", [lrp, srp, up])


if __name__ == "__main__":
    for f in (f1_model_comparison, f2_unseen_cities, f3_gan_tradeoff, f4_calibration, f5_dataset,
              f6_training, f7_per_band, f8_real_per_dataset, f9_demos):
        f()
    with open(os.path.join(OUT, "figures_manifest.md"), "w") as fh:
        fh.write("| Figure | Content | Source files |\n|---|---|---|\n" + "\n".join(MANIFEST) + "\n")
    print(f"\n{len(MANIFEST)} figures -> {OUT}/  (see figures_manifest.md)")
