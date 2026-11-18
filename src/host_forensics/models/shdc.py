"""Self-Referential Host-Prototype Differential Conditioning (SHDC).

Dynamically isolates unmanipulated host anchor tokens inside the input video stream itself,
estimating uncontaminated camera/codec/restoration moments (mu_host, sigma_host) and applying
Host FiLM + Standardized Z-Score Differential Residual conditioning.

Mathematically cancels global channel-wise affine shifts from social-media transcoding (H.264/H.265)
and full-frame neural restoration, eliminating cross-domain score polarization (Cross-AUC / MID-FFD).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from host_forensics.config import SHDCConfig


@dataclass
class HostConditioningOutput:
    """Outputs of Self-Referential Host-Prototype Differential Conditioning."""

    tokens: torch.Tensor       # Same shape as input (B, ..., D)
    anchor_probs: torch.Tensor  # Per-token pristine likelihood alpha in (0, 1), shape (B, ...)
    mu_host: torch.Tensor      # (B, D) uncontaminated host mean
    sigma_host: torch.Tensor   # (B, D) uncontaminated host std


class SelfReferentialHostConditioning(nn.Module):
    """Anchor-weighted Host FiLM + Standardized Z-Residual conditioning module."""

    def __init__(self, dim: int, cfg: SHDCConfig) -> None:
        super().__init__()
        self.dim = dim
        self.beta = cfg.sharpening_beta
        self.eps = cfg.eps

        # Per-token Authenticity Anchor Gate (invariant to global per-sample channel mean/std shift
        # by normalizing across feature channels via LayerNorm before scoring!)
        self.anchor_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, cfg.anchor_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.anchor_hidden_dim, 1),
        )

        # Host FiLM generator mapping [mu_host; sigma_host] -> (gamma, beta)
        self.film_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # Standardized intra-video non-linear differential MLP (strictly affine-shift invariant,
        # and non-linear GELU breaks zero-sum cancellation under uniform initial pooling!)
        self.diff_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.out_gain = nn.Parameter(torch.ones(dim))

    def forward(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        strict_affine_invariance: bool = False,
    ) -> HostConditioningOutput:
        """Args:
            z: (B, ..., D) visual tubelet tokens (B, T, P, D) or audio tokens (B, T_a, D)
            valid_mask: optional boolean/float mask of shape (B, ...) for padded positions
            strict_affine_invariance: if True, omits raw z carryover so output is 100% invariant
                to global channel-wise affine transformations z' = a * z + b (for invariance verification).
        """
        orig_shape = z.shape
        b = orig_shape[0]
        d = orig_shape[-1]
        z_flat = z.reshape(b, -1, d)  # (B, M, D)
        m_tokens = z_flat.shape[1]

        # 1. Estimate per-token pristine anchor probability alpha in (0, 1)
        # Center via 50%-breakdown-point token median & scale by token std per channel d:
        # strictly invariant to z'_{b,m,d} = a_{b,d}*z_{b,m,d} + c_{b,d} (a_{b,d} > 0) AND uncontaminated by <50% RoI!
        z_fp32 = z_flat.float()
        tok_med = z_fp32.median(dim=1, keepdim=True).values
        tok_std = z_fp32.std(dim=1, unbiased=False, keepdim=True).clamp(min=self.eps)
        z_gate_in = ((z_fp32 - tok_med) / tok_std).to(dtype=z.dtype)
        alpha_logits = self.anchor_gate(z_gate_in).squeeze(-1)  # (B, M)
        alpha = torch.sigmoid(alpha_logits)

        if valid_mask is not None:
            vm = valid_mask.reshape(b, m_tokens).to(dtype=alpha.dtype)
            alpha_masked = alpha * vm
        else:
            alpha_masked = alpha

        # 2. Sharpened soft-anchor weights tilde_alpha summing to 1 over valid tokens
        weights = alpha_masked.float().pow(self.beta)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=self.eps)  # (B, M)

        # 3. Uncontaminated Host Prototype Moments (computed in FP32 for numerical stability)
        mu_host = torch.einsum("bm,bmd->bd", weights, z_fp32)  # (B, D)
        centered = z_fp32 - mu_host.unsqueeze(1)
        var_host = torch.einsum("bm,bmd->bd", weights, centered.pow(2))
        sigma_host = torch.sqrt(var_host + 1e-12).clamp(min=self.eps)  # (B, D)

        # 4. Standardized Intra-Video Differential Residual (strictly invariant to z' = a*z + b, a > 0)
        z_std = (centered / sigma_host.unsqueeze(1)).to(dtype=z.dtype)
        diff_term = (z_std + self.diff_mlp(z_std)) * self.out_gain.to(dtype=z.dtype)

        if strict_affine_invariance:
            out_flat = 10.0 * torch.tanh(diff_term / 10.0)
        else:
            host_cat = torch.cat([mu_host, sigma_host], dim=-1).to(dtype=z.dtype)
            gamma, shift = self.film_mlp(host_cat).chunk(2, dim=-1)  # (B, D) each
            film_term = z_flat * (1.0 + gamma.unsqueeze(1)) + shift.unsqueeze(1)
            out_flat = film_term + 10.0 * torch.tanh(diff_term / 10.0)

        return HostConditioningOutput(
            tokens=out_flat.reshape(orig_shape),
            anchor_probs=alpha.reshape(orig_shape[:-1]),
            mu_host=mu_host.to(dtype=z.dtype),
            sigma_host=sigma_host.to(dtype=z.dtype),
        )
