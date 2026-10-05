"""
Training for both stages.

Stage 1 ("psnr"):  generator only, content loss (pixel + SSIM + SAM + grad + LR-consistency).
                    This is the model you report accuracy numbers for.
Stage 2 ("gan"):   start from the stage-1 checkpoint, add a PatchGAN discriminator
                    (+ optional VGG perceptual loss). Sharper textures, usually LOWER PSNR,
                    and more hallucination risk -> always report it next to stage 1.

Outputs in cfg.train.out_dir:
    config.yaml, splits/{train,val,test}.csv, log.csv, best.pt, last.pt
"""
from __future__ import annotations

import csv
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import save_config, to_plain
from .data import (BAND_NAMES, SRPatchDataset, build_degradation, build_normalizer, make_splits)
from .losses import (ContentLoss, VGGPerceptual, relativistic_d_loss, relativistic_g_loss)
from .metrics import per_image_metrics
from .models import PatchDiscriminator, build_model, count_params


# ----------------------------------------------------------------------------- utils
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def warmup_cosine(optimizer, total_steps, warmup_steps, min_ratio=0.02):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def build_datasets(cfg, verbose=True):
    train_df, val_df, test_df = make_splits(cfg, verbose=verbose)
    dr = cfg.get("_dry_run_limit")
    if dr:
        train_df = train_df.sample(n=min(dr, len(train_df)), random_state=0)
        val_df = val_df.sample(n=min(max(8, dr // 4), len(val_df)), random_state=0)
        test_df = test_df.sample(n=min(max(8, dr // 4), len(test_df)), random_state=0)
    norm, deg = build_normalizer(cfg), build_degradation(cfg)
    d = cfg.data
    train_ds = SRPatchDataset(train_df, norm, deg, train=True, crop_size=int(d.crop_size),
                              crops_per_patch=int(d.get("crops_per_patch", 1)),
                              augment=bool(d.get("augment", True)))
    val_crop = d.get("val_crop_size", 256)
    val_ds = SRPatchDataset(val_df, norm, deg, train=False,
                            crop_size=None if val_crop in (None, "none", 0) else int(val_crop))
    return train_ds, val_ds, (train_df, val_df, test_df), norm, deg


@torch.no_grad()
def validate(model, loader, device, norm, scale, amp_dtype=None):
    model.eval()
    rows_sr, rows_bic = [], []
    for batch in loader:
        lr, hr, mask = batch["lr"].to(device), batch["hr"].to(device), batch["mask"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            sr = model(lr)
        sr = sr.float().clamp(0, 1)
        bic = torch.nn.functional.interpolate(lr, scale_factor=scale, mode="bicubic",
                                              align_corners=False).clamp(0, 1)
        rows_sr += per_image_metrics(sr, hr, mask, norm, scale)
        rows_bic += per_image_metrics(bic, hr, mask, norm, scale)
    agg = lambda rows, k: float(np.mean([r[k] for r in rows])) if rows else float("nan")
    keys = ["psnr", "ssim", "sam_deg", "ergas", "ndvi_mae"] + [f"psnr_{b}" for b in BAND_NAMES]
    return {k: agg(rows_sr, k) for k in keys}, {k: agg(rows_bic, k) for k in keys}


def _amp_dtype(cfg, device):
    if not cfg.train.get("amp", True) or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ----------------------------------------------------------------------------- main loop
def run_training(cfg, stage: str = "psnr"):
    assert stage in ("psnr", "gan")
    t = cfg.train
    set_seed(int(t.seed))
    device = get_device(t.get("device", "auto"))
    out_dir = t.out_dir
    os.makedirs(os.path.join(out_dir, "splits"), exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    train_ds, val_ds, (train_df, val_df, test_df), norm, deg = build_datasets(cfg)
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        d.to_csv(os.path.join(out_dir, "splits", f"{name}.csv"), index=False)

    nw = int(cfg.data.get("num_workers", 4))
    bs = int(t.batch_size)
    train_loader = make_loader(train_ds, bs, True, nw, drop_last=len(train_ds) >= bs)
    val_loader = make_loader(val_ds, int(t.get("val_batch_size", max(1, bs // 2))), False, nw)

    scale = int(cfg.data.scale)
    model = build_model(cfg.model, scale).to(device)
    print(f"[model] {cfg.model.name}  {count_params(model):.2f} M params  device={device}")

    gcfg = cfg.get("gan", {})
    disc = perceptual = None
    if stage == "gan":
        init = gcfg.get("init_from")
        if not init or not os.path.exists(init):
            raise FileNotFoundError(f"gan.init_from={init!r} - train stage 1 first")
        model.load_state_dict(torch.load(init, map_location=device, weights_only=False)["model"])
        print(f"[gan] generator initialised from {init}")
        disc = PatchDiscriminator(4, int(gcfg.get("d_base", 64))).to(device)
        if float(gcfg.get("perceptual_weight", 0)) > 0:
            perceptual = VGGPerceptual().to(device)

    epochs = int(gcfg.get("epochs", 30)) if stage == "gan" else int(t.epochs)
    lr_g = float(gcfg.get("lr_g", 1e-4)) if stage == "gan" else float(t.lr)
    opt_g = torch.optim.Adam(model.parameters(), lr=lr_g, betas=(0.9, 0.99),
                             weight_decay=float(t.get("weight_decay", 0)))
    steps = epochs * max(1, len(train_loader))
    sched_g = warmup_cosine(opt_g, steps, int(t.get("warmup_steps", 500)) if stage == "psnr" else 0)
    opt_d = sched_d = None
    if disc is not None:
        opt_d = torch.optim.Adam(disc.parameters(), lr=float(gcfg.get("lr_d", 1e-4)), betas=(0.9, 0.99))
        sched_d = warmup_cosine(opt_d, steps, 0)

    content = ContentLoss(t.loss, scale, deg.sigma_eval)
    amp_dtype = _amp_dtype(cfg, device) if stage == "psnr" else None  # GAN stage runs fp32 (stabler)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    clip = float(t.get("grad_clip", 1.0))

    # ---- resume ----
    start_epoch, best, since_best, step = 1, -float("inf"), 0, 0
    last_path = os.path.join(out_dir, "last.pt")
    if t.get("resume", True) and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("stage") == stage:
            model.load_state_dict(ck["model"]); opt_g.load_state_dict(ck["opt_g"])
            sched_g.load_state_dict(ck["sched_g"])
            if disc is not None and ck.get("disc"):
                disc.load_state_dict(ck["disc"]); opt_d.load_state_dict(ck["opt_d"])
                sched_d.load_state_dict(ck["sched_d"])
            start_epoch, best, since_best, step = ck["epoch"] + 1, ck["best"], ck["since_best"], ck["step"]
            print(f"[resume] from epoch {ck['epoch']} (best val PSNR {best:.3f})")

    # ---- bicubic reference on val (computed once) ----
    _, bic_val = validate(model, val_loader, device, norm, scale, amp_dtype)
    print(f"[val] bicubic baseline: PSNR {bic_val['psnr']:.3f} dB  SSIM {bic_val['ssim']:.4f}  "
          f"SAM {bic_val['sam_deg']:.3f} deg")

    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path) or start_epoch == 1
    log_f = open(log_path, "w" if new_log else "a", newline="")
    writer = None
    patience = int(t.get("patience", 12))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if disc is not None:
            disc.train()
        t0, run = time.time(), {}
        n_batches = 0
        for batch in train_loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            lr_mask = batch["lr_mask"].to(device, non_blocking=True)

            # ---------------- generator ----------------
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                sr = model(lr)
            loss_g, parts = content(sr, hr, mask, lr, lr_mask)
            if disc is not None:
                d_real = disc(hr).detach()
                d_fake = disc(sr.float())
                adv = relativistic_g_loss(d_real, d_fake)
                loss_g = loss_g + float(gcfg.get("adv_weight", 5e-3)) * adv
                parts["adv_g"] = adv.item()
                if perceptual is not None:
                    pl = perceptual(sr.float(), hr)
                    loss_g = loss_g + float(gcfg.perceptual_weight) * pl
                    parts["perceptual"] = pl.item()
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt_g)
            scaler.update()
            sched_g.step()

            # ---------------- discriminator ----------------
            if disc is not None:
                opt_d.zero_grad(set_to_none=True)
                d_real, d_fake = disc(hr), disc(sr.detach().float())
                loss_d = relativistic_d_loss(d_real, d_fake)
                loss_d.backward()
                opt_d.step(); sched_d.step()
                parts["d"] = loss_d.item()

            parts["total"] = loss_g.item()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            n_batches += 1
            step += 1
            if not math.isfinite(parts["total"]):
                raise RuntimeError(f"Non-finite loss at step {step}: {parts}. Lower lr or disable amp.")

        train_parts = {f"train_{k}": v / max(1, n_batches) for k, v in run.items()}
        val, _ = validate(model, val_loader, device, norm, scale, amp_dtype)
        score = val["psnr"]
        improved = score > best
        if improved:
            best, since_best = score, 0
        else:
            since_best += 1

        rec = {"epoch": epoch, "lr": opt_g.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1),
               **train_parts, **{f"val_{k}": v for k, v in val.items()},
               "val_gain_over_bicubic_db": val["psnr"] - bic_val["psnr"]}
        if writer is None:
            writer = csv.DictWriter(log_f, fieldnames=list(rec.keys()), extrasaction="ignore")
            if new_log:
                writer.writeheader()
        writer.writerow(rec); log_f.flush()

        print(f"[{stage}] ep {epoch:03d}/{epochs} | loss {train_parts.get('train_total', 0):.4f} | "
              f"val PSNR {val['psnr']:.3f} ({val['psnr'] - bic_val['psnr']:+.2f} vs bicubic) "
              f"SSIM {val['ssim']:.4f} SAM {val['sam_deg']:.3f} | {rec['time_s']}s"
              + ("  *best*" if improved else ""))

        ck = {"model": model.state_dict(), "cfg": to_plain(cfg), "stage": stage, "epoch": epoch,
              "best": best, "since_best": since_best, "step": step, "val": val,
              "opt_g": opt_g.state_dict(), "sched_g": sched_g.state_dict(),
              "disc": disc.state_dict() if disc is not None else None,
              "opt_d": opt_d.state_dict() if opt_d is not None else None,
              "sched_d": sched_d.state_dict() if sched_d is not None else None}
        torch.save(ck, last_path)
        if improved:
            torch.save({k: ck[k] for k in ("model", "cfg", "stage", "epoch", "val")},
                       os.path.join(out_dir, "best.pt"))

        if stage == "psnr" and since_best >= patience:
            print(f"[early stop] no val improvement for {patience} epochs")
            break

    log_f.close()
    if stage == "gan":  # for GAN, the final model (not max-PSNR) is the one you want to look at
        torch.save({k: ck[k] for k in ("model", "cfg", "stage", "epoch", "val")},
                   os.path.join(out_dir, "final.pt"))
    print(f"[done] best val PSNR {best:.3f} dB -> {out_dir}")
    return best
