"""
Stage 5 — EDSR baseline model + training loop.

Usage:
    from sr_dataset import split_master_index, compute_band_stats, SRPatchDataset
    from edsr_train import EDSR, train

    train_df, val_df, test_df = split_master_index("patches/master_index.csv", test_city="New_Delhi")
    band_stats = compute_band_stats(train_df)

    train_ds = SRPatchDataset(train_df, band_stats, scale=4, train=True, crop_size=256)
    val_ds   = SRPatchDataset(val_df,   band_stats, scale=4, train=False, crop_size=256)
    test_ds  = SRPatchDataset(test_df,  band_stats, scale=4, train=False, crop_size=256)

    model = EDSR(scale=4, n_channels=4, n_resblocks=8, n_feats=64)
    train(model, train_ds, val_ds, test_ds, epochs=50, batch_size=8, out_dir="models")
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BAND_NAMES = ["B02", "B03", "B04", "B08"]


# ------------------------------------------------------------------
# MODEL — small EDSR (residual blocks, no batchnorm, PixelShuffle upsampling)
# ------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        return x + out * self.res_scale


class EDSR(nn.Module):
    def __init__(self, scale=4, n_channels=4, n_resblocks=8, n_feats=64):
        super().__init__()
        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # PixelShuffle upsampler, supports scale = 2 or 4 (chain of x2 steps)
        up_layers = []
        assert scale in (2, 4), "scale must be 2 or 4"
        for _ in range(1 if scale == 2 else 2):
            up_layers += [nn.Conv2d(n_feats, n_feats * 4, 3, padding=1), nn.PixelShuffle(2)]
        self.upsample = nn.Sequential(*up_layers)
        self.tail = nn.Conv2d(n_feats, n_channels, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body_tail(self.body(feat))
        feat = feat + res
        out = self.upsample(feat)
        return torch.clamp(self.tail(out), 0, 1)


# ------------------------------------------------------------------
# MASKED LOSS + METRICS — always ignore cloud-gap pixels (valid_mask)
# ------------------------------------------------------------------
def masked_l1_loss(pred, hr, valid_mask):
    # pred, hr: (B, C, H, W); valid_mask: (B, H, W)
    mask = valid_mask.unsqueeze(1)  # (B, 1, H, W), broadcasts over channels
    diff = (pred - hr).abs() * mask
    return diff.sum() / mask.sum().clamp(min=1) / pred.shape[1]


@torch.no_grad()
def masked_psnr_per_band(pred, hr, valid_mask):
    """Returns dict band_name -> psnr, averaged over the batch."""
    mask = valid_mask.unsqueeze(1)
    results = {}
    for i, band in enumerate(BAND_NAMES):
        p, h, m = pred[:, i:i + 1], hr[:, i:i + 1], mask
        mse = ((p - h) ** 2 * m).sum(dim=[1, 2, 3]) / m.sum(dim=[1, 2, 3]).clamp(min=1)
        psnr = -10 * torch.log10(mse.clamp(min=1e-10))
        results[band] = psnr.mean().item()
    return results


# ------------------------------------------------------------------
# TRAIN LOOP
# ------------------------------------------------------------------
def train(model, train_ds, val_ds, test_ds, epochs=50, batch_size=8, lr=1e-4,
          out_dir="models", device=None, patience=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    os.makedirs(out_dir, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_val_psnr = -float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            pred = model(lr_img)
            # pred is at HR resolution already (upsampled inside the model)
            loss = masked_l1_loss(pred, hr_img, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lr_img.size(0)
        train_loss /= len(train_ds)

        # ---- validation ----
        model.eval()
        val_psnrs = {b: [] for b in BAND_NAMES}
        with torch.no_grad():
            for batch in val_loader:
                lr_img = batch["lr"].to(device)
                hr_img = batch["hr"].to(device)
                mask = batch["valid_mask"].to(device)
                pred = model(lr_img)
                psnrs = masked_psnr_per_band(pred, hr_img, mask)
                for b in BAND_NAMES:
                    val_psnrs[b].append(psnrs[b])
        val_mean_per_band = {b: sum(v) / len(v) for b, v in val_psnrs.items()}
        val_mean = sum(val_mean_per_band.values()) / len(val_mean_per_band)

        scheduler.step(val_mean)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_PSNR_mean={val_mean:.2f} dB | "
              f"per-band={ {b: round(v,2) for b,v in val_mean_per_band.items()} }")

        if val_mean > best_val_psnr:
            best_val_psnr = val_mean
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
            torch.save(best_state, os.path.join(out_dir, "edsr_best.pt"))
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"No val improvement for {patience} epochs — stopping early.")
                break

    # ---- final test evaluation on the fully-held-out city, using best checkpoint ----
    model.load_state_dict(best_state)
    model.eval()
    test_psnrs = {b: [] for b in BAND_NAMES}
    with torch.no_grad():
        for batch in test_loader:
            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            mask = batch["valid_mask"].to(device)
            pred = model(lr_img)
            psnrs = masked_psnr_per_band(pred, hr_img, mask)
            for b in BAND_NAMES:
                test_psnrs[b].append(psnrs[b])
    test_mean_per_band = {b: sum(v) / len(v) for b, v in test_psnrs.items()}
    print("\n" + "=" * 60)
    print("FINAL TEST (held-out city) RESULTS")
    print("=" * 60)
    for b, v in test_mean_per_band.items():
        print(f"  {b}: {v:.2f} dB")
    print(f"  mean: {sum(test_mean_per_band.values())/len(test_mean_per_band):.2f} dB")

    return model, test_mean_per_band
