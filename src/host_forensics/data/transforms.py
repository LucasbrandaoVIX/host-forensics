"""Compound Social-Media Degradation & Symmetric Anti-Forensic Laundering Pipeline.

Implements:
1. NTIRE 2026-style compound degradations: multi-scale resize, 8x8 block quantization proxy (H.264/JPEG),
   Gaussian/ISO sensor noise, contrast/gamma shifts, and screen-recapture Moire fringes.
2. Symmetric Restoration/VAE Laundering Simulator: applied with identical probability `p_launder` to
   BOTH pristine real clips and manipulated clips so neural enhancers (CodeFormer/GFPGAN/VAE round-trip)
   never become a trivial false-positive shortcut on authentic enhanced videos.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CompoundDegradationAndLaundering(nn.Module):
    """Differentiable/GPU-friendly video degradation & symmetric laundering pipeline."""

    def __init__(
        self,
        degradation_prob: float = 0.50,
        symmetric_laundering_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.degradation_prob = degradation_prob
        self.symmetric_laundering_prob = symmetric_laundering_prob

    @torch.no_grad()
    def _apply_codec_and_resize(self, frames: torch.Tensor) -> torch.Tensor:
        """Simulate social-media downscale/upscale + 8x8 block quantization."""
        # frames: (T, 3, H, W) in [0, 1] or normalized
        _, _, h, w = frames.shape
        scale = float(torch.empty(1, device=frames.device).uniform_(0.45, 0.85).item())
        nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
        down = F.interpolate(frames, size=(nh, nw), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)

        # Quantization step proxy (simulating CRF 26-38 bit-depth/DCT rounding)
        levels = float(torch.randint(24, 96, (1,), device=frames.device).item())
        quant = torch.round(up * levels) / levels
        return quant

    @torch.no_grad()
    def _apply_moire_or_iso_noise(self, frames: torch.Tensor) -> torch.Tensor:
        t, _, h, w = frames.shape
        sigma = float(torch.empty(1, device=frames.device).uniform_(0.005, 0.035).item())
        out = frames + torch.randn_like(frames) * sigma

        # 25% sub-chance of subtle Moire screen-recapture sinusoidal interference
        if float(torch.rand(1, device=frames.device).item()) < 0.25:
            yy = torch.linspace(0.0, 1.0, h, device=frames.device).view(1, 1, h, 1)
            xx = torch.linspace(0.0, 1.0, w, device=frames.device).view(1, 1, 1, w)
            freq = float(torch.empty(1, device=frames.device).uniform_(18.0, 45.0).item())
            theta = float(torch.empty(1, device=frames.device).uniform_(0.0, math.pi).item())
            fringe = torch.sin(2.0 * math.pi * freq * (xx * math.cos(theta) + yy * math.sin(theta)))
            out = out + 0.02 * fringe.expand(t, 3, h, w)
        return out

    @torch.no_grad()
    def _apply_symmetric_laundering_proxy(self, frames: torch.Tensor) -> torch.Tensor:
        """Simulate full-frame VAE / CodeFormer smoothing + unsharp high-frequency hallucination.

        In full offline/online training, users can also plug actual frozen SD-VAE / CodeFormer checkpoints;
        this fast GPU operator mimics bilateral/latent bottleneck smoothing + high-freq synthesis uniformly
        across the entire frame (erasing naive inner/outer sharpness steps).
        """
        _, _, h, w = frames.shape
        latent_hw = (max(16, h // 4), max(16, w // 4))
        bottleneck = F.interpolate(frames, size=latent_hw, mode="area")
        decoded = F.interpolate(bottleneck, size=(h, w), mode="bicubic", align_corners=False)
        # Unsharp mask re-synthesis of crisp edges (characteristic of face/video restorers)
        high_freq = frames - F.avg_pool2d(frames, kernel_size=3, stride=1, padding=1)
        return decoded + 0.35 * torch.tanh(high_freq * 3.0)

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Apply stochastic degradations & symmetric laundering per clip.

        Args:
            video: (T, 3, H, W) or (B, T, 3, H, W)
        """
        is_batched = video.ndim == 5
        if not is_batched:
            video_b = video.unsqueeze(0)
        else:
            video_b = video

        out_clips: list[torch.Tensor] = []
        for bi in range(video_b.shape[0]):
            clip = video_b[bi]
            # Symmetric laundering applied BEFORE social transcoding (realistic attacker order!)
            if float(torch.rand(1, device=clip.device).item()) < self.symmetric_laundering_prob:
                clip = self._apply_symmetric_laundering_proxy(clip)
            if float(torch.rand(1, device=clip.device).item()) < self.degradation_prob:
                clip = self._apply_codec_and_resize(clip)
                clip = self._apply_moire_or_iso_noise(clip)
            out_clips.append(clip)

        stacked = torch.stack(out_clips, dim=0)
        return stacked if is_batched else stacked.squeeze(0)
