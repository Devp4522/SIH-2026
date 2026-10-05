"""
All losses take a validity mask (B,1,H,W) so cloud gaps / no-data never act as ground truth.

    pixel      Charbonnier (smooth L1)          - main fidelity term
    ssim       1 - SSIM                          - structure / edges
    sam        spectral angle                    - keeps band ratios (NDVI etc.) intact
    grad       Sobel-gradient L1                 - sharper edges (roads, field boundaries)
    lr_consist |blur_down(SR) - LR|              - physics prior: SR must re-degrade to the input
    adv        relativistic-average GAN          - optional stage 2 (texture)
    perceptual VGG19 features on the RGB bands   - optional stage 2
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RGB_IDX, blur_downsample


def masked_mean(x, mask):
    """x: (B,C,H,W), mask: (B,1,H,W)."""
    m = mask.expand_as(x)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def charbonnier(pred, hr, mask, eps=1e-3):
    return masked_mean(torch.sqrt((pred - hr) ** 2 + eps ** 2), mask)


def _gauss_window(size, sigma, channels, device, dtype):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim_map(x, y, data_range=1.0, size=11, sigma=1.5):
    c = x.shape[1]
    w = _gauss_window(size, sigma, c, x.device, x.dtype)
    p = size // 2
    pad = lambda t: F.pad(t, (p, p, p, p), mode="reflect")
    mu_x, mu_y = F.conv2d(pad(x), w, groups=c), F.conv2d(pad(y), w, groups=c)
    sxx = F.conv2d(pad(x * x), w, groups=c) - mu_x ** 2
    syy = F.conv2d(pad(y * y), w, groups=c) - mu_y ** 2
    sxy = F.conv2d(pad(x * y), w, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def ssim_loss(pred, hr, mask):
    return 1.0 - masked_mean(ssim_map(pred, hr), mask)


def spectral_angle(pred, hr, eps=1e-6):
    """Per-pixel angle (radians) between spectral vectors. (B,1,H,W)."""
    dot = (pred * hr).sum(1, keepdim=True)
    den = pred.norm(dim=1, keepdim=True) * hr.norm(dim=1, keepdim=True)
    return torch.acos((dot / (den + eps)).clamp(-1 + 1e-6, 1 - 1e-6))


def sam_loss(pred, hr, mask):
    # shift by +0.05 so near-zero (dark water) pixels don't make the angle unstable
    return masked_mean(spectral_angle(pred + 0.05, hr + 0.05), mask)


_SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0


def gradient_loss(pred, hr, mask):
    c = pred.shape[1]
    kx = _SOBEL.to(pred).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    ky = _SOBEL.t().to(pred).reshape(1, 1, 3, 3).expand(c, 1, 3, 3)
    g = lambda t, k: F.conv2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), k, groups=c)
    return masked_mean((g(pred, kx) - g(hr, kx)).abs() + (g(pred, ky) - g(hr, ky)).abs(), mask)


def lr_consistency_loss(pred, lr, lr_mask, sigma, scale):
    return masked_mean((blur_downsample(pred, sigma, scale) - lr).abs(), lr_mask)


class VGGPerceptual(nn.Module):
    """VGG19 conv5_4 features on the RGB bands. Downloads ImageNet weights on first use."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, hr):
        f = lambda t: self.features((t[:, RGB_IDX].clamp(0, 1) - self.mean) / self.std)
        return F.l1_loss(f(pred), f(hr))


class ContentLoss(nn.Module):
    """Weighted sum of the pixel-domain losses. Weights come from cfg.train.loss."""

    def __init__(self, weights: dict, scale: int, sigma_eval: float):
        super().__init__()
        self.w = {k: float(v) for k, v in weights.items()}
        self.scale, self.sigma = scale, sigma_eval

    def forward(self, pred, hr, mask, lr, lr_mask):
        pred = pred.float()
        parts = {}
        if self.w.get("pixel", 0):
            parts["pixel"] = charbonnier(pred, hr, mask)
        if self.w.get("ssim", 0):
            parts["ssim"] = ssim_loss(pred, hr, mask)
        if self.w.get("sam", 0):
            parts["sam"] = sam_loss(pred, hr, mask)
        if self.w.get("grad", 0):
            parts["grad"] = gradient_loss(pred, hr, mask)
        if self.w.get("lr_consistency", 0):
            parts["lr_consistency"] = lr_consistency_loss(pred, lr, lr_mask, self.sigma, self.scale)
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: v.item() for k, v in parts.items()}


# ----------------------------------------------------------------------------- GAN (ESRGAN-style)
def relativistic_d_loss(d_real, d_fake):
    return 0.5 * (F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.ones_like(d_real)) +
                  F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.zeros_like(d_fake)))


def relativistic_g_loss(d_real, d_fake):
    return 0.5 * (F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.zeros_like(d_real)) +
                  F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.ones_like(d_fake)))
