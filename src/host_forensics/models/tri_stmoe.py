"""Tri-Pool Auxiliary-Loss-Free Spatio-Temporal-Residual Mixture of Experts (Tri-STMoE).

Extends MoSTE (2026) from 2-pool cropped-face routing to 3 specialized forensic pools:
  - Pool S (Spatial): Intra-frame self-attention over patch tokens (swap seams, lighting, 3D geometry)
  - Pool T (Temporal): Axial temporal self-attention with 1D RoPE (reenactment jitter, rollout resets)
  - Pool R (Residual): Cross-attention into MN-Stem motion/noise tokens (object removal, VAE laundering)

GPU Parallelization Design:
  - All `E` experts within each pool execute in parallel via stacked Tensor Core `einsum` projections
    and a single grouped FlashAttention-2 / cuDNN `F.scaled_dot_product_attention` launch, collapsing
    `12` sequential expert chains per block into `3` parallel pool launches.
  - `Rotary1D` caches `(cos, sin)` tables per `(seq_len, device, dtype)` to eliminate redundant trig ops.
  - `BiasBalancedTopKRouter` synchronizes empirical expert loads across ranks via `dist.all_reduce`
    whenever `torch.distributed` is initialized.
"""

from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from host_forensics.config import TriSTMoEConfig


class Rotary1D(nn.Module):
    """1D Rotary Position Embedding (RoPE) with cached `(cos, sin)` tables for axial temporal attention."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for Rotary1D.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_key: tuple[int, torch.device, torch.dtype] | None = None
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None

    def _get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, device, dtype)
        if self._cache_key != key or self._cos_cache is None or self._sin_cache is None:
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(pos, self.inv_freq.to(device=device))  # (L, D_h/2)
            self._cos_cache = freqs.cos()[None, None, :, :].to(dtype=dtype)
            self._sin_cache = freqs.sin()[None, None, :, :].to(dtype=dtype)
            self._cache_key = key
        return self._cos_cache, self._sin_cache

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            q, k: (N, H, L, D_h)
        """
        cos, sin = self._get_cos_sin(q.shape[2], q.device, q.dtype)

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
        self._accum_load = torch.zeros(num_experts, dtype=torch.float32)
        self._accum_count: int = 0

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
                load = flat_mask.mean(dim=0) / float(self.top_k)
                self.last_expert_load.copy_(load)
                if self._accum_load.device != load.device:
                    self._accum_load = self._accum_load.to(load.device)
                self._accum_load.add_(load)
                self._accum_count += 1

        return weights

    @torch.no_grad()
    def update_bias(self) -> None:
        """Adjust non-differentiable expert selection bias once per optimizer step (cross-rank synced)."""
        if self._accum_count > 0:
            load = self._accum_load / float(self._accum_count)
            self._accum_load.zero_()
            self._accum_count = 0
        else:
            load = self.last_expert_load.clone()

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(load, op=dist.ReduceOp.AVG)
        self.last_expert_load.copy_(load)

        target = 1.0 / float(self.num_experts)
        delta = torch.sign(target - self.last_expert_load)
        self.routing_bias.add_(self.bias_update_rate * delta)


class SpatialAttentionExpert(nn.Module):
    """Single intra-frame patch self-attention expert (retained for standalone modularity)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, p, d = x.shape
        h_in = rearrange(self.norm(x), "b t p d -> (b t) p d")
        qkv = self.qkv(h_in).chunk(3, dim=-1)
        q, k, v = [rearrange(tensor, "bt p (h dh) -> bt h p dh", h=self.num_heads) for tensor in qkv]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = self.proj(rearrange(out, "bt h p dh -> bt p (h dh)"))
        return rearrange(out, "(b t) p d -> b t p d", b=b, t=t, p=p, d=d)


class TemporalAttentionExpert(nn.Module):
    """Single axial inter-frame temporal self-attention expert with 1D RoPE."""

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
        b, t, p, d = x.shape
        h_in = rearrange(self.norm(x), "b t p d -> (b p) t d")
        qkv = self.qkv(h_in).chunk(3, dim=-1)
        q, k, v = [rearrange(tensor, "bp t (h dh) -> bp h t dh", h=self.num_heads) for tensor in qkv]
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = self.proj(rearrange(out, "bp h t dh -> bp t (h dh)"))
        return rearrange(out, "(b p) t d -> b t p d", b=b, t=t, p=p, d=d)


class ResidualCrossAttentionExpert(nn.Module):
    """Single cross-attention expert coupling semantic patch tokens to MN-Stem flow/noise residuals."""

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


def _init_stacked_linear(param: nn.Parameter) -> None:
    for idx in range(param.shape[0]):
        nn.init.kaiming_uniform_(param[idx], a=math.sqrt(5))


class BatchedSpatialExpertPool(nn.Module):
    """Executes all `E` intra-frame spatial experts in parallel via stacked GEMM + single grouped SDPA."""

    def __init__(self, num_experts: int, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.norm_bias = nn.Parameter(torch.zeros(num_experts, dim))
        self.w_qkv = nn.Parameter(torch.empty(num_experts, dim * 3, dim))
        self.w_proj = nn.Parameter(torch.empty(num_experts, dim, dim))
        _init_stacked_linear(self.w_qkv)
        _init_stacked_linear(self.w_proj)

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, D), routing_weights: (B, T, P, E)
        b, t, p, d = x.shape
        x_norm_base = F.layer_norm(x, (d,))  # (B, T, P, D)
        # Broadcast per-expert affine scale/shift: (B*T, E, P, D)
        x_bt = rearrange(x_norm_base, "b t p d -> (b t) 1 p d")
        h = x_bt * self.norm_weight.view(1, self.num_experts, 1, d) + self.norm_bias.view(
            1, self.num_experts, 1, d
        )

        qkv = torch.einsum("nepd,eod->nepo", h, self.w_qkv)  # (B*T, E, P, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "n e p (h dh) -> (n e) h p dh", h=self.num_heads)
        k = rearrange(k, "n e p (h dh) -> (n e) h p dh", h=self.num_heads)
        v = rearrange(v, "n e p (h dh) -> (n e) h p dh", h=self.num_heads)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn_merged = rearrange(attn, "(n e) h p dh -> n e p (h dh)", e=self.num_experts)
        out = torch.einsum("nepd,eod->nepo", attn_merged, self.w_proj)  # (B*T, E, P, D)
        out_bt = rearrange(out, "(b t) e p d -> b t p e d", b=b, t=t)
        return torch.einsum("btpe,btped->btpd", routing_weights, out_bt)


class BatchedTemporalExpertPool(nn.Module):
    """Executes all `E` axial 1D-RoPE temporal experts in parallel via stacked GEMM + single SDPA."""

    def __init__(self, num_experts: int, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.rope = Rotary1D(self.head_dim)

        self.norm_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.norm_bias = nn.Parameter(torch.zeros(num_experts, dim))
        self.w_qkv = nn.Parameter(torch.empty(num_experts, dim * 3, dim))
        self.w_proj = nn.Parameter(torch.empty(num_experts, dim, dim))
        _init_stacked_linear(self.w_qkv)
        _init_stacked_linear(self.w_proj)

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, D), routing_weights: (B, T, P, E)
        b, t, p, d = x.shape
        x_norm_base = F.layer_norm(x, (d,))
        x_bp = rearrange(x_norm_base, "b t p d -> (b p) 1 t d")
        h = x_bp * self.norm_weight.view(1, self.num_experts, 1, d) + self.norm_bias.view(
            1, self.num_experts, 1, d
        )

        qkv = torch.einsum("netd,eod->neto", h, self.w_qkv)  # (B*P, E, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "n e t (h dh) -> (n e) h t dh", h=self.num_heads)
        k = rearrange(k, "n e t (h dh) -> (n e) h t dh", h=self.num_heads)
        v = rearrange(v, "n e t (h dh) -> (n e) h t dh", h=self.num_heads)

        q, k = self.rope(q, k)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn_merged = rearrange(attn, "(n e) h t dh -> n e t (h dh)", e=self.num_experts)
        out = torch.einsum("netd,eod->neto", attn_merged, self.w_proj)  # (B*P, E, T, D)
        out_bt = rearrange(out, "(b p) e t d -> b t p e d", b=b, p=p)
        return torch.einsum("btpe,btped->btpd", routing_weights, out_bt)


class BatchedResidualExpertPool(nn.Module):
    """Executes all `E` residual cross-attention experts in parallel via stacked GEMM + single SDPA."""

    def __init__(self, num_experts: int, dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.norm_q_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.norm_q_bias = nn.Parameter(torch.zeros(num_experts, dim))
        self.norm_kv_weight = nn.Parameter(torch.ones(num_experts, dim))
        self.norm_kv_bias = nn.Parameter(torch.zeros(num_experts, dim))

        self.w_q = nn.Parameter(torch.empty(num_experts, dim, dim))
        self.w_kv = nn.Parameter(torch.empty(num_experts, dim * 2, dim))
        self.w_out = nn.Parameter(torch.empty(num_experts, dim, dim))
        _init_stacked_linear(self.w_q)
        _init_stacked_linear(self.w_kv)
        _init_stacked_linear(self.w_out)

    def forward(
        self, x: torch.Tensor, res_mn: torch.Tensor, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        b, t, p, d = x.shape
        q_base = rearrange(F.layer_norm(x, (d,)), "b t p d -> (b t) 1 p d")
        kv_base = rearrange(F.layer_norm(x + res_mn, (d,)), "b t p d -> (b t) 1 p d")

        q_in = q_base * self.norm_q_weight.view(1, self.num_experts, 1, d) + self.norm_q_bias.view(
            1, self.num_experts, 1, d
        )
        kv_in = kv_base * self.norm_kv_weight.view(1, self.num_experts, 1, d) + self.norm_kv_bias.view(
            1, self.num_experts, 1, d
        )

        q = torch.einsum("nepd,eod->nepo", q_in, self.w_q)
        kv = torch.einsum("nepd,eod->nepo", kv_in, self.w_kv)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "n e p (h dh) -> (n e) h p dh", h=self.num_heads)
        k = rearrange(k, "n e p (h dh) -> (n e) h p dh", h=self.num_heads)
        v = rearrange(v, "n e p (h dh) -> (n e) h p dh", h=self.num_heads)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn_merged = rearrange(attn, "(n e) h p dh -> n e p (h dh)", e=self.num_experts)
        out = torch.einsum("nepd,eod->nepo", attn_merged, self.w_out)
        out_bt = rearrange(out, "(b t) e p d -> b t p e d", b=b, t=t)
        return torch.einsum("btpe,btped->btpd", routing_weights, out_bt)


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

        self.pool_s = BatchedSpatialExpertPool(n_exp, d_mid, cfg.num_heads, cfg.dropout)
        self.pool_t = BatchedTemporalExpertPool(n_exp, d_mid, cfg.num_heads, cfg.dropout)
        self.pool_r = BatchedResidualExpertPool(n_exp, d_mid, cfg.num_heads, cfg.dropout)

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

        ys = self.pool_s(x_mid, ws)
        yt = self.pool_t(x_mid, wt)
        yr = self.pool_r(x_mid, r_mid, wr)

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
