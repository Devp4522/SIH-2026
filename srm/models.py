"""
Generators
    EDSR  - residual CNN, no batch-norm, PixelShuffle upsampling. Strong, simple baseline.
    RCAN  - residual-in-residual with channel attention. Channel attention is useful
            here because it learns inter-band (spectral) weighting, e.g. NIR vs visible.
Both add the network output to a bicubic upsampling of the input (global residual):
the net only has to learn the missing high-frequency detail, which trains faster and
keeps the low-frequency radiometry (= spectral consistency) anchored to the input.

Discriminator
    PatchGAN with spectral norm, 4-band input, used only in the optional GAN stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def make_upsampler(scale: int, n_feats: int) -> nn.Sequential:
    if scale not in (2, 3, 4, 8):
        raise ValueError("scale must be 2, 3, 4 or 8")
    layers = []
    if scale == 3:
        layers += [nn.Conv2d(n_feats, n_feats * 9, 3, padding=1), nn.PixelShuffle(3)]
    else:
        for _ in range({2: 1, 4: 2, 8: 3}[scale]):
            layers += [nn.Conv2d(n_feats, n_feats * 4, 3, padding=1), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- EDSR
class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, scale=4, n_channels=4, n_resblocks=16, n_feats=64, res_scale=0.1,
                 global_residual=True):
        super().__init__()
        self.scale = scale
        self.global_residual = global_residual
        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)],
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.upsample = make_upsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, n_channels, 3, padding=1)

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)
        out = self.tail(self.upsample(f))
        if self.global_residual:
            out = out + F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return out  # NOT clamped: clamping kills gradients; clamp only at eval/inference


# ----------------------------------------------------------------------------- RCAN
class ChannelAttention(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(n_feats, max(4, n_feats // reduction), 1), nn.ReLU(True),
                                 nn.Conv2d(max(4, n_feats // reduction), n_feats, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class RCAB(nn.Module):
    def __init__(self, n_feats, reduction=16, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.ReLU(True),
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1),
                                  ChannelAttention(n_feats, reduction))
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(self, n_feats, n_blocks, reduction=16):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(n_feats, reduction) for _ in range(n_blocks)],
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class RCAN(nn.Module):
    def __init__(self, scale=4, n_channels=4, n_groups=5, n_resblocks=10, n_feats=64,
                 reduction=16, global_residual=True):
        super().__init__()
        self.scale = scale
        self.global_residual = global_residual
        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualGroup(n_feats, n_resblocks, reduction) for _ in range(n_groups)],
                                  nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.upsample = make_upsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, n_channels, 3, padding=1)

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)
        out = self.tail(self.upsample(f))
        if self.global_residual:
            out = out + F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return out


# ----------------------------------------------------------------------------- Discriminator
class PatchDiscriminator(nn.Module):
    """Spectral-norm PatchGAN. Outputs a logit map, so any input size works."""

    def __init__(self, n_channels=4, base=64, n_layers=4):
        super().__init__()
        layers = [spectral_norm(nn.Conv2d(n_channels, base, 3, 1, 1)), nn.LeakyReLU(0.2, True)]
        ch = base
        for i in range(n_layers):
            nxt = min(base * 2 ** (i + 1), 512)
            layers += [spectral_norm(nn.Conv2d(ch, nxt, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                       spectral_norm(nn.Conv2d(nxt, nxt, 3, 1, 1)), nn.LeakyReLU(0.2, True)]
            ch = nxt
        layers += [nn.Conv2d(ch, 1, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- factory
def build_model(mcfg, scale: int, n_channels: int = 4) -> nn.Module:
    name = mcfg.name.lower()
    if name == "edsr":
        return EDSR(scale, n_channels, int(mcfg.n_resblocks), int(mcfg.n_feats),
                    float(mcfg.get("res_scale", 0.1)), bool(mcfg.get("global_residual", True)))
    if name == "rcan":
        return RCAN(scale, n_channels, int(mcfg.get("n_groups", 5)), int(mcfg.n_resblocks),
                    int(mcfg.n_feats), int(mcfg.get("reduction", 16)),
                    bool(mcfg.get("global_residual", True)))
    raise ValueError(f"unknown model {mcfg.name!r} (edsr | rcan)")


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
