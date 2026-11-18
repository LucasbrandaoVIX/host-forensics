"""Motion-Noise Forensic Residual Stem (MN-Stem).

Captures low-level camera ISP/VAE quantization discontinuities (via zero-DC Bayar + SRM filters)
and optical-flow inpainting/stitching residuals (via local correlation cost-volume warping error),
directly addressing the 0.135 mIoU Object-Removal blind spot of purely semantic ViT/MLLM detectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from host_forensics.config import MNStemConfig


class BayarConv2d(nn.Module):
    """Constrained 5x5 zero-DC forensic convolution (Bayar & Stamm, IEEE TIFS).

    Enforces dynamically on every forward pass:
        K[:, :, center, center] = -1.0
        sum_{(u,v) != center} K[:, :, u, v] = +1.0
    so scene luminance/color content is suppressed and local resampling/VAE noise passes through.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 12, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("BayarConv2d requires an odd kernel_size >= 3.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Parameterize unconstrained off-center weights
        num_off_center = kernel_size * kernel_size - 1
        self.raw_off_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, num_off_center) * 0.1
        )

    def constrained_weight(self) -> torch.Tensor:
        # Normalize off-center weights per (out_channel, in_channel) via softmax so sum == 1/in_channels
        # ensuring sum over spatial != center across RGB sums cleanly to +1 against center -1/in_channels,
        # or per-channel zero-DC: sum_{(u,v)!=c} K[o, i, u, v] = +1 and K[o, i, c, c] = -1.
        off_norm = F.softmax(self.raw_off_weights, dim=-1)  # sums to +1.0 over off-center positions
        k2 = self.kernel_size * self.kernel_size
        center_idx = k2 // 2

        weight_flat = torch.empty(
            self.out_channels,
            self.in_channels,
            k2,
            device=self.raw_off_weights.device,
            dtype=self.raw_off_weights.dtype,
        )
        weight_flat[:, :, :center_idx] = off_norm[:, :, :center_idx]
        weight_flat[:, :, center_idx] = -1.0
        weight_flat[:, :, center_idx + 1 :] = off_norm[:, :, center_idx:]
        return weight_flat.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.constrained_weight()
        return F.conv2d(x, w.to(dtype=x.dtype), padding=self.padding)


class SRMFilterBank(nn.Module):
    """Fixed Spatial Rich Model (SRM) high-pass residual filters applied per RGB channel (9 output channels)."""

    def __init__(self) -> None:
        super().__init__()
        q1 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 4.0
        q2 = torch.tensor(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=torch.float32,
        ) / 12.0
        q3 = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ) / 2.0

        base = torch.stack([q1, q2, q3], dim=0).unsqueeze(1)  # (3, 1, 5, 5)
        # Depthwise across 3 RGB channels -> (9, 1, 5, 5) with groups=3
        kernels = base.repeat(3, 1, 1, 1)
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            F.conv2d(x, self.kernels.to(dtype=x.dtype), padding=2, groups=3),
            min=-3.0,
            max=3.0,
        )


class LocalCostVolumeMotionResidual(nn.Module):
    """Computes local neighborhood correlation cost-volume entropy and forward-backward motion residual.

    Unlike heavy external optical flow networks, this operates at 1/4 spatial resolution in <2 ms,
    highlighting regions where pixels were propagated by video inpainting (ProPainter/VideoPainter)
    or stitched across warped boundaries (LivePortrait/MuseTalk).
    """

    def __init__(self, feat_channels: int = 32, radius: int = 2, temperature: float = 0.1) -> None:
        super().__init__()
        self.radius = radius
        self.temperature = temperature
        self.num_neighbors = (2 * radius + 1) ** 2
        self.out_channels = feat_channels + 4  # residual vector + (fwd_err, bwd_err, fwd_entropy, bwd_entropy)

    def _directional_warp_residual(
        self, src: torch.Tensor, ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Warp `ref` toward `src` using local (2r+1)^2 patch inner products.

        Args:
            src: (N, C, H, W) L2-normalized features
            ref: (N, C, H, W) L2-normalized features
        Returns:
            diff_vec: (N, C, H, W), l2_err: (N, 1, H, W), entropy: (N, 1, H, W)
        """
        n, c, h, w = src.shape
        k = 2 * self.radius + 1
        # Unfold ref into local neighborhood patches: (N, C * k^2, H * W)
        ref_unfold = F.unfold(ref, kernel_size=k, padding=self.radius)
        ref_unfold = ref_unfold.view(n, c, self.num_neighbors, h, w)

        # Correlation logits: (N, k^2, H, W)
        corr = (src.unsqueeze(2) * ref_unfold).sum(dim=1) / max(self.temperature, 1e-4)
        attn = F.softmax(corr, dim=1)

        #Normalized correlation entropy in [0, 1]
        log_attn = F.log_softmax(corr, dim=1)
        entropy = -(attn * log_attn).sum(dim=1, keepdim=True) / torch.log(
            torch.tensor(float(self.num_neighbors), device=src.device, dtype=src.dtype)
        )

        # Reconstruct src from ref neighborhood: (N, C, H, W)
        warped = (attn.unsqueeze(1) * ref_unfold).sum(dim=2)
        diff_vec = src - warped
        l2_err = torch.norm(diff_vec.float(), p=2, dim=1, keepdim=True).to(dtype=src.dtype)
        return diff_vec, l2_err, entropy

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Args:
            feat: (B, T, C, H, W) shallow spatial features at 1/4 resolution.
        Returns:
            motion_res: (B, T, C + 4, H, W)
        """
        b, t, c, h, w = feat.shape
        if t == 1:
            zeros = torch.zeros(b, 1, 4, h, w, device=feat.device, dtype=feat.dtype)
            return torch.cat([torch.zeros_like(feat), zeros], dim=2)

        norm_f = F.normalize(feat.float(), p=2, dim=2).to(dtype=feat.dtype)
        prev_f = torch.cat([norm_f[:, :1], norm_f[:, :-1]], dim=1)
        next_f = torch.cat([norm_f[:, 1:], norm_f[:, -1:]], dim=1)

        src_flat = rearrange(norm_f, "b t c h w -> (b t) c h w")
        prev_flat = rearrange(prev_f, "b t c h w -> (b t) c h w")
        next_flat = rearrange(next_f, "b t c h w -> (b t) c h w")

        diff_bwd, err_bwd, ent_bwd = self._directional_warp_residual(src_flat, prev_flat)
        diff_fwd, err_fwd, ent_fwd = self._directional_warp_residual(src_flat, next_flat)

        bi_diff = 0.5 * (diff_bwd + diff_fwd)
        fused = torch.cat([bi_diff, err_fwd, err_bwd, ent_fwd, ent_bwd], dim=1)
        return rearrange(fused, "(b t) c h w -> b t c h w", b=b, t=t)


class MotionNoiseResidualStem(nn.Module):
    """Extracts patch-aligned forensic residual tokens R^{MN} and 1/4-res boundary skip features."""

    def __init__(self, cfg: MNStemConfig, patch_size: int = 16) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = patch_size
        if patch_size % 4 != 0:
            raise ValueError("patch_size must be divisible by 4 for MN-Stem skip alignment.")

        self.bayar = BayarConv2d(3, cfg.bayar_channels, kernel_size=5)
        self.srm = SRMFilterBank()
        noise_in_ch = cfg.bayar_channels + 9

        # Downsample noiseprint + shallow RGB stream to H/4 x W/4
        self.noise_down = nn.Sequential(
            nn.Conv2d(noise_in_ch, cfg.shallow_feat_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, cfg.shallow_feat_channels),
            nn.GELU(),
            nn.Conv2d(cfg.shallow_feat_channels, cfg.shallow_feat_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, cfg.shallow_feat_channels),
            nn.GELU(),
        )
        self.rgb_shallow = nn.Sequential(
            nn.Conv2d(3, cfg.shallow_feat_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, cfg.shallow_feat_channels),
            nn.GELU(),
            nn.Conv2d(cfg.shallow_feat_channels, cfg.shallow_feat_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, cfg.shallow_feat_channels),
            nn.GELU(),
        )

        self.motion_corr = LocalCostVolumeMotionResidual(
            feat_channels=cfg.shallow_feat_channels,
            radius=cfg.corr_radius,
            temperature=cfg.corr_temperature,
        )

        combined_qtr_ch = cfg.shallow_feat_channels + self.motion_corr.out_channels
        self.skip_proj = nn.Sequential(
            nn.Conv2d(combined_qtr_ch, cfg.out_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, cfg.out_dim // 2),
            nn.GELU(),
        )

        # Pool from H/4 -> H/patch_size (factor = patch_size // 4, typically 4 for patch_size=16)
        pool_stride = patch_size // 4
        self.to_patch_tokens = nn.Sequential(
            nn.Conv2d(
                combined_qtr_ch,
                cfg.out_dim,
                kernel_size=pool_stride,
                stride=pool_stride,
                bias=False,
            ),
            nn.GroupNorm(8, cfg.out_dim),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(cfg.out_dim)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            video: (B, T, 3, H, W)
        Returns:
            res_tokens: (B, T, P, out_dim) where P = (H/patch_size)*(W/patch_size)
            skip_qtr: (B*T, out_dim//2, H/4, W/4) high-res boundary skip feature map
        """
        b, t, _, h, w = video.shape
        frames = rearrange(video, "b t c h w -> (b t) c h w")

        noise_raw = torch.cat([self.bayar(frames), self.srm(frames)], dim=1)
        noise_qtr = self.noise_down(noise_raw)  # (B*T, C_s, H/4, W/4)

        rgb_qtr = rearrange(self.rgb_shallow(frames), "(b t) c h w -> b t c h w", b=b, t=t)
        motion_qtr = self.motion_corr(rgb_qtr)
        motion_qtr_flat = rearrange(motion_qtr, "b t c h w -> (b t) c h w")

        fused_qtr = torch.cat([noise_qtr, motion_qtr_flat], dim=1)
        skip_qtr = self.skip_proj(fused_qtr)

        patch_grid = self.to_patch_tokens(fused_qtr)  # (B*T, out_dim, H/p, W/p)
        tokens = rearrange(patch_grid, "(b t) d hp wp -> b t (hp wp) d", b=b, t=t)
        tokens = self.token_norm(tokens)
        return tokens, skip_qtr
