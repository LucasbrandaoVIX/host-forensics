"""Tri-Pool Auxiliary-Loss-Free Spatio-Temporal-Residual Mixture of Experts (Tri-STMoE).

Extends MoSTE (Nov 2026) from 2-pool cropped-face routing to 3 specialized forensic pools:
  - Pool S (Spatial): Intra-frame self-attention over patch tokens (swap seams, lighting, 3D geometry)
  - Pool T (Temporal): Axial temporal self-attention with 1D RoPE (reenactment jitter, rollout resets)
  - Pool R (Residual): Cross-attention into MN-Stem motion/noise tokens (object removal, VAE laundering)

Uses DeepSeek-V3 / MoSTE auxiliary-loss-free dynamic bias balancing so primary forensic gradients
never suffer interference from auxiliary load-balancing penalties.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from host_forensics.config import TriSTMoEConfig


class Rotary1D(nn.Module):
    """1D Rotary Position Embedding (RoPE) for axial temporal self-attention."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for Rotary1D.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            q, k: (N, H, L, D_h)
        """
        seq_len = q.shape[2]
        pos = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)  # (L, D_h/2)
        cos = freqs.cos()[None, None, :, :].to(dtype=q.dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=q.dtype)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., ::2], x[..., 1::2]
            rot1 = x1 * cos - x2 * sin
            rot2 = x1 * sin + x2 * cos
            return torch.stack([rot1, rot2], dim=-1).flatten(-2)

        return _apply(q), _apply(k)


class BiasBalancedTopKRouter(nn.Module):
    """Auxiliary-loss-free Top-k token router with non-differentiable load-balancing bias.

    Selection uses `logits + routing_bias`, whereas gating weights use clean `softmax(logits[top_k])`.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int, bias_update_rate: float = 1e-4) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"Invalid top_k={top_k} for num_experts={num_experts}.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("routing_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer(
            "last_expert_load",
            torch.full((num_experts,), 1.0 / num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (..., D) token tensor
        Returns:
            weights: (..., E) sparse nonnegative gating weights summing to 1.0 along last dim
        """
        logits = self.gate(x)  # (..., E)
        biased_logits = logits.float() + self.routing_bias.view(*([1] * (logits.ndim - 1)), -1)
        _, top_indices = torch.topk(biased_logits, k=self.top_k, dim=-1)  # (..., k)

        chosen_logits = torch.gather(logits, dim=-1, index=top_indices)
        chosen_weights = F.softmax(chosen_logits.float(), dim=-1).to(dtype=logits.dtype)

        weights = torch.zeros_like(logits)
        weights.scatter_(dim=-1, index=top_indices, src=chosen_weights)

        if self.training:
            with torch.no_grad():
                selected_mask = torch.zeros_like(logits, dtype=torch.float32)
                selected_mask.scatter_(dim=-1, index=top_indices, value=1.0)
                flat_mask = selected_mask.reshape(-1, self.num_experts)
                # Normalize so uniform usage equals 1 / num_experts
                load = flat_mask.mean(dim=0) / float(self.top_k)
                self.last_expert_load.copy_(load)

        return weights

    @torch.no_grad()
    def update_bias(self) -> None:
        """Adjust non-differentiable expert selection bias after each training step."""
        target = 1.0 / float(self.num_experts)
        delta = torch.sign(target - self.last_expert_load)
        self.routing_bias.add_(self.bias_update_rate * delta)


class SpatialAttentionExpert(nn.Module):
    """Intra-frame patch self-attention expert."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, D)
        b, t, p, d = x.shape
        h_in = rearrange(self.norm(x), "b t p d -> (b t) p d")
        qkv = self.qkv(h_in).chunk(3, dim=-1)
        q, k, v = [rearrange(tensor, "bt p (h dh) -> bt h p dh", h=self.num_heads) for tensor in qkv]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = self.proj(rearrange(out, "bt h p dh -> bt p (h dh)"))
        return rearrange(out, "(b t) p d -> b t p d", b=b, t=t, p=p, d=d)


class TemporalAttentionExpert(nn.Module):
    """Axial inter-frame temporal self-attention expert with 1D RoPE."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = Rotary1D(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, D) -> group identical patch locations across frames
        b, t, p, d = x.shape
        h_in = rearrange(self.norm(x), "b t p d -> (b p) t d")
        qkv = self.qkv(h_in).chunk(3, dim=-1)
        q, k, v = [rearrange(tensor, "bp t (h dh) -> bp h t dh", h=self.num_heads) for tensor in qkv]
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = self.proj(rearrange(out, "bp h t dh -> bp t (h dh)"))
        return rearrange(out, "(b p) t d -> b t p d", b=b, t=t, p=p, d=d)


class ResidualCrossAttentionExpert(nn.Module):
    """Cross-attention expert coupling semantic patch tokens to MN-Stem flow/noise residuals."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, res_mn: torch.Tensor) -> torch.Tensor:
        # x, res_mn: (B, T, P, D)
        b, t, p, d = x.shape
        q_in = rearrange(self.norm_q(x), "b t p d -> (b t) p d")
        kv_in = rearrange(self.norm_kv(x + res_mn), "b t p d -> (b t) p d")

        q = rearrange(self.q_proj(q_in), "bt p (h dh) -> bt h p dh", h=self.num_heads)
        k, v = [
            rearrange(tensor, "bt p (h dh) -> bt h p dh", h=self.num_heads)
            for tensor in self.kv_proj(kv_in).chunk(2, dim=-1)
        ]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = self.out_proj(rearrange(out, "bt h p dh -> bt p (h dh)"))
        return rearrange(out, "(b t) p d -> b t p d", b=b, t=t, p=p, d=d)


class TriSTMoEAdapter(nn.Module):
    """Single-stage Tri-Pool Spatio-Temporal-Residual MoE Adapter inserted into frozen ViT blocks."""

    def __init__(self, embed_dim: int, res_dim: int, cfg: TriSTMoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_mid = cfg.bottleneck_dim
        n_exp = cfg.num_experts_per_pool

        self.down_proj = nn.Linear(embed_dim, d_mid)
        self.res_align = nn.Linear(res_dim, d_mid, bias=False) if res_dim != d_mid else nn.Identity()

        self.router_s = BiasBalancedTopKRouter(d_mid, n_exp, cfg.top_k, cfg.bias_update_rate)
        self.router_t = BiasBalancedTopKRouter(d_mid, n_exp, cfg.top_k, cfg.bias_update_rate)
        self.router_r = BiasBalancedTopKRouter(d_mid, n_exp, cfg.top_k, cfg.bias_update_rate)

        self.experts_s = nn.ModuleList(
            [SpatialAttentionExpert(d_mid, cfg.num_heads, cfg.dropout) for _ in range(n_exp)]
        )
        self.experts_t = nn.ModuleList(
            [TemporalAttentionExpert(d_mid, cfg.num_heads, cfg.dropout) for _ in range(n_exp)]
        )
        self.experts_r = nn.ModuleList(
            [ResidualCrossAttentionExpert(d_mid, cfg.num_heads, cfg.dropout) for _ in range(n_exp)]
        )

        self.up_proj = nn.Linear(d_mid, embed_dim)
        # Near-identity zero-initialized residual up-projection for stable PEFT start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x_in: torch.Tensor, res_mn: torch.Tensor) -> torch.Tensor:
        """Args:
            x_in: (B, T, P, embed_dim)
            res_mn: (B, T, P, res_dim)
        Returns:
            x_out: (B, T, P, embed_dim)
        """
        x_mid = F.gelu(self.down_proj(x_in))  # (B, T, P, d_mid)
        r_mid = self.res_align(res_mn)

        ws = self.router_s(x_mid)  # (B, T, P, E)
        wt = self.router_t(x_mid)
        wr = self.router_r(x_mid + r_mid)

        ys = torch.zeros_like(x_mid)
        yt = torch.zeros_like(x_mid)
        yr = torch.zeros_like(x_mid)

        for idx, exp_s in enumerate(self.experts_s):
            ys = ys + ws[..., idx : idx + 1] * exp_s(x_mid)
        for idx, exp_t in enumerate(self.experts_t):
            yt = yt + wt[..., idx : idx + 1] * exp_t(x_mid)
        for idx, exp_r in enumerate(self.experts_r):
            yr = yr + wr[..., idx : idx + 1] * exp_r(x_mid, r_mid)

        fused = F.gelu(x_mid + ys + yt + yr)
        return x_in + self.up_proj(fused)

    @torch.no_grad()
    def step_biases(self) -> None:
        self.router_s.update_bias()
        self.router_t.update_bias()
        self.router_r.update_bias()

    def expert_load_summary(self) -> dict[str, list[float]]:
        return {
            "spatial": self.router_s.last_expert_load.detach().cpu().tolist(),
            "temporal": self.router_t.last_expert_load.detach().cpu().tolist(),
            "residual": self.router_r.last_expert_load.detach().cpu().tolist(),
        }
