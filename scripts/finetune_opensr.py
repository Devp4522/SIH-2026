"""
Fine-tune the synthetic-trained EDSR on REAL Sentinel-2 (10 m) -> 2.5 m reference pairs
(opensr-test), holding out whole datasets for an honest test.

    # train on NAIP + Spain crops (90 real pairs), never touch SPOT / Spain urban
    python scripts/finetune_opensr.py --ckpt runs/edsr_final_v2/best.pt

    # JOINT fine-tuning: half of every batch is real pairs, half synthetic Indian patches
    # (prevents forgetting the Indian landscapes; tracks both benchmarks every epoch)
    python scripts/finetune_opensr.py --ckpt runs/edsr_final_v2/best.pt --mix-synthetic 0.5 --out runs/edsr_ft_joint

    # then compare zero-shot vs fine-tuned on the UNSEEN datasets
    python scripts/eval_opensr.py --ckpt runs/edsr_final_v2/best.pt --datasets spot spain_urban --tta --out runs/cmp/zeroshot
    python scripts/eval_opensr.py --ckpt runs/edsr_ft_real/best.pt  --datasets spot spain_urban --tta --out runs/cmp/finetuned

Why this design
- Whole datasets are held out (not random images), so the test measures generalisation to new
  sensors/places, not memorisation. ~15% of the training images are held out for early stopping.
- Low learning rate + few epochs: 90 images is small; we adapt the synthetic prior, not replace it.
- Same losses as stage 1 (pixel + SSIM + SAM + edge + LR-consistency), so spectral consistency
  with the real Sentinel-2 input is still enforced.
"""
import argparse
import copy
import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import _common  # noqa: F401
from eval_opensr import L2A_IDX, corr, detect_hr_band_order, high_pass, load_dataset, to_f32
from srm.config import _wrap, to_plain
from srm.data import SRPatchDataset, build_degradation, build_normalizer, dihedral, make_splits
from srm.losses import ContentLoss
from srm.models import build_model, count_params
from srm.trainer import get_device, set_seed, validate as validate_synth, warmup_cosine


def load_pairs(names, data_dir, norm, offset, rng_range):
    """Returns list of (lr, hr) normalised float tensors, (4,h,w) and (4,4h,4w)."""
    lo, hi = rng_range
    pairs = []
    for name in names:
        ds = load_dataset(name, data_dir)
        lrs = [to_f32(x)[L2A_IDX] for x in ds["L2A"]]
        perm, _ = detect_hr_band_order(lrs, ds["HRharm"])
        for lr, hr in zip(lrs, ds["HRharm"]):
            hr = to_f32(hr)[perm]
            h, w = lr.shape[1] * 4, lr.shape[2] * 4
            hr = hr[:, :h, :w]
            ln = np.clip(norm.norm(lr + offset, clip=False), lo, hi)
            hn = np.clip(norm.norm(hr + offset, clip=False), lo, hi)
            pairs.append((torch.from_numpy(ln), torch.from_numpy(hn), name))
        print(f"  {name}: {len(lrs)} pairs (HR band order {perm})")
    return pairs


class RealPairs(Dataset):
    def __init__(self, pairs, crop_hr=128, crops_per_image=16, scale=4):
        self.pairs, self.c, self.k, self.s = pairs, crop_hr, crops_per_image, scale

    def __len__(self):
        return len(self.pairs) * self.k

    def __getitem__(self, i):
        lr, hr, _ = self.pairs[i % len(self.pairs)]
        s, cl = self.s, self.c // self.s
        h, w = lr.shape[1], lr.shape[2]
        t = int(torch.randint(0, h - cl + 1, (1,)))
        l_ = int(torch.randint(0, w - cl + 1, (1,)))
        lr_c = lr[:, t:t + cl, l_:l_ + cl]
        hr_c = hr[:, t * s:(t + cl) * s, l_ * s:(l_ + cl) * s]
        k = int(torch.randint(0, 8, (1,)))
        lr_c, hr_c = dihedral(lr_c, k).contiguous(), dihedral(hr_c, k).contiguous()
        return {"lr": lr_c, "hr": hr_c,
                "mask": torch.ones(1, *hr_c.shape[-2:]), "lr_mask": torch.ones(1, *lr_c.shape[-2:])}


@torch.no_grad()
def validate(model, pairs, device, scale, clamp):
    model.eval()
    res = {"psnr": [], "psnr_bic": [], "hf": [], "hf_bic": []}
    b = 16
    for lr, hr, _ in pairs:
        x = lr[None].to(device)
        sr = model(x)[0].float().clamp(*clamp).cpu()
        bic = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)[0].clamp(*clamp).cpu()
        H = hr[:, b:-b, b:-b]
        for tag, p in (("", sr), ("_bic", bic)):
            P = p[:, b:-b, b:-b]
            mse = ((P - H) ** 2).mean().clamp(min=1e-10)
            res["psnr" + tag].append(float(10 * torch.log10(1 / mse)))
            hp_p, hp_h = high_pass(p)[:, b:-b, b:-b], high_pass(hr)[:, b:-b, b:-b]
            res["hf" + tag].append(float(np.mean([corr(hp_p[i], hp_h[i]) for i in range(4)])))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="synthetic-trained checkpoint to start from")
    ap.add_argument("--train-datasets", nargs="+", default=["naip", "spain_crops"])
    ap.add_argument("--test-datasets", nargs="+", default=["spot", "spain_urban"],
                    help="only listed for safety: the script refuses to train on them")
    ap.add_argument("--data-dir", default="opensr_data")
    ap.add_argument("--offset", type=float, default=1000.0)
    ap.add_argument("--norm-range", nargs=2, type=float, default=None,
                    help="default: the checkpoint's own data.clip")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128, help="HR crop size")
    ap.add_argument("--crops-per-image", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--mix-synthetic", type=float, default=0.0,
                    help="fraction of every batch drawn from the synthetic Indian patches (0 = real only). "
                         "0.5 is a good start: keeps the Indian prior while learning the real degradation")
    ap.add_argument("--synth-workers", type=int, default=2)
    ap.add_argument("--india-val", type=int, default=64, help="Indian val patches scored each epoch (mix mode)")
    ap.add_argument("--select", choices=["psnr", "hf", "both"], default=None,
                    help="checkpoint selection. psnr/hf: real held-out images. both: real PSNR gain + Indian "
                         "PSNR gain (default when mixing), so neither benchmark is sacrificed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/edsr_ft_real")
    a = ap.parse_args()

    leak = set(a.train_datasets) & set(a.test_datasets)
    if leak:
        raise SystemExit(f"Refusing to train on test datasets: {sorted(leak)}")

    set_seed(a.seed)
    device = get_device()
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = _wrap(ck["cfg"])
    scale = int(cfg.data.scale)
    norm, deg = build_normalizer(cfg), build_degradation(cfg)
    rng_range = tuple(a.norm_range) if a.norm_range else tuple(cfg.data.get("clip", [0.0, 1.0]))
    model = build_model(cfg.model, scale).to(device)
    model.load_state_dict(ck["model"])
    print(f"[ft] start from {a.ckpt} ({count_params(model):.2f} M params), device={device}, "
          f"range={rng_range}")

    print("[ft] loading real pairs")
    pairs = load_pairs(a.train_datasets, a.data_dir, norm, a.offset, rng_range)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(pairs))
    n_val = max(2, int(round(len(pairs) * a.val_frac)))
    val_pairs = [pairs[i] for i in idx[:n_val]]
    tr_pairs = [pairs[i] for i in idx[n_val:]]
    print(f"[ft] train images {len(tr_pairs)} | val images {len(val_pairs)} | "
          f"never seen: {a.test_datasets}")

    mix = min(max(a.mix_synthetic, 0.0), 0.9)
    b_syn = int(round(a.batch_size * mix))
    b_real = a.batch_size - b_syn
    select = a.select or ("both" if mix > 0 else "psnr")
    loader = DataLoader(RealPairs(tr_pairs, a.crop, a.crops_per_image, scale), batch_size=b_real,
                        shuffle=True, num_workers=0, drop_last=True)
    synth_iter = india_loader = None
    if b_syn > 0:
        tr_df, va_df, _ = make_splits(cfg, verbose=False)
        if int(cfg.data.crop_size) != a.crop:
            raise SystemExit(f"--crop {a.crop} must equal the synthetic crop_size {cfg.data.crop_size}")
        sds = SRPatchDataset(tr_df, norm, deg, train=True, crop_size=a.crop, crops_per_patch=1)
        sl = DataLoader(sds, batch_size=b_syn, shuffle=True, num_workers=a.synth_workers, drop_last=True,
                        persistent_workers=a.synth_workers > 0)

        def forever(dl):
            while True:
                for bt in dl:
                    yield bt
        synth_iter = forever(sl)
        va_small = va_df.sample(n=min(a.india_val, len(va_df)), random_state=0)
        india_loader = DataLoader(SRPatchDataset(va_small, norm, deg, train=False, crop_size=256),
                                  batch_size=4, shuffle=False, num_workers=0)
        print(f"[ft] JOINT mode: each batch = {b_real} real + {b_syn} synthetic Indian | "
              f"synthetic pool {len(tr_df)} patches | Indian val {len(va_small)} patches | select={select}")

    def india_gain():
        if india_loader is None:
            return None
        sr_v, bic_v = validate_synth(model, india_loader, device, norm, scale, amp_dtype)
        return sr_v["psnr"] - bic_v["psnr"]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = warmup_cosine(opt, a.epochs * len(loader), warmup_steps=min(100, len(loader)))
    content = ContentLoss(cfg.train.loss, scale, deg.sigma_eval)
    amp = device.type == "cuda"
    amp_dtype = (torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16) if amp else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    os.makedirs(a.out, exist_ok=True)
    def score(v, ig):
        if select == "psnr":
            return v["psnr"]
        if select == "hf":
            return v["hf"]
        return (v["psnr"] - v["psnr_bic"]) + (ig if ig is not None else 0.0)

    v0 = validate(model, val_pairs, device, scale, rng_range)
    ig0 = india_gain()
    print(f"[ft] before: val PSNR {v0['psnr']:.3f} (bicubic {v0['psnr_bic']:.3f}) | "
          f"detail corr {v0['hf']:.4f} (bicubic {v0['hf_bic']:.4f})"
          + (f" | India gain {ig0:+.2f} dB" if ig0 is not None else ""))
    best, best_state, since = score(v0, ig0), copy.deepcopy(model.state_dict()), 0
    log = open(os.path.join(a.out, "log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "loss", "val_psnr", "val_psnr_bic", "val_hf", "val_hf_bic", "india_gain_db"])
    wr.writerow([0, "", v0["psnr"], v0["psnr_bic"], v0["hf"], v0["hf_bic"], ig0])

    for ep in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            if synth_iter is not None:
                sb = next(synth_iter)
                batch = {k: torch.cat([batch[k], sb[k]]) for k in ("lr", "hr", "mask", "lr_mask")}
            lr_, hr_ = batch["lr"].to(device), batch["hr"].to(device)
            m, lm = batch["mask"].to(device), batch["lr_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                sr = model(lr_)
            loss, _ = content(sr, hr_, m, lr_, lm)
            if not math.isfinite(loss.item()):
                raise RuntimeError("non-finite loss - lower --lr")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        v = validate(model, val_pairs, device, scale, rng_range)
        ig = india_gain()
        wr.writerow([ep, tot / nb, v["psnr"], v["psnr_bic"], v["hf"], v["hf_bic"], ig]); log.flush()
        sc = score(v, ig)
        improved = sc > best
        if improved:
            best, best_state, since = sc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        print(f"[ft] ep {ep:02d} | loss {tot / nb:.4f} | val PSNR {v['psnr']:.3f} "
              f"({v['psnr'] - v['psnr_bic']:+.2f} vs bic) | detail corr {v['hf']:.4f} "
              f"({v['hf'] - v['hf_bic']:+.4f} vs bic)"
              + (f" | India {ig:+.2f} dB vs bic" if ig is not None else "") + ("  *best*" if improved else ""))
        if since >= a.patience:
            print(f"[ft] early stop (no improvement for {a.patience} epochs)")
            break
    log.close()

    new_cfg = to_plain(cfg)
    new_cfg["data"]["clip"] = list(rng_range)
    new_cfg["finetune"] = {"init": a.ckpt, "train_datasets": a.train_datasets,
                           "test_datasets": a.test_datasets, "offset": a.offset, "lr": a.lr,
                           "mix_synthetic": mix, "select": select}
    torch.save({"model": best_state, "cfg": new_cfg, "stage": "finetune_real", "epoch": ep,
                "val": {select: best}}, os.path.join(a.out, "best.pt"))
    print(f"\n[ft] saved {a.out}/best.pt (best score [{select}] {best:.4f}, before {score(v0, ig0):.4f})")
    print("Now compare on the UNSEEN datasets:")
    t = " ".join(a.test_datasets)
    print(f"  python scripts/eval_opensr.py --ckpt {a.ckpt} --datasets {t} --tta "
          f"--norm-range {rng_range[0]} {rng_range[1]} --out runs/cmp/zeroshot")
    print(f"  python scripts/eval_opensr.py --ckpt {a.out}/best.pt --datasets {t} --tta "
          f"--norm-range {rng_range[0]} {rng_range[1]} --out runs/cmp/{os.path.basename(os.path.normpath(a.out))}")


if __name__ == "__main__":
    main()
