"""Frozen Perceptual Foundation Backbones & Orthogonal SVD-LoRA Adaptation.

Implements:
1. `OrthogonalSVDLinear`: Principal-subspace-preserving low-rank residual adaptation (Effort, ICML'25):
       W = W_0 + (I - U_k U_k^T) B A (I - V_k V_k^T)
   guaranteeing that learned forensic updates lie strictly in the orthogonal complement of the
   pretrained foundation model's principal semantic directions.
2. `MultiPriorVisualBackbone`: Wraps HuggingFace DINOv3 / CLIP ViTs (or an offline-safe ViT stand-in
   for CI/testing), taps intermediate patch tokens across depths, and executes `TriSTMoEAdapter`
   at designated transformer blocks.
3. `FrozenAudioBackbone`: Wraps WavLM-Large / Qwen3-Omni audio encoders (or an offline-safe stand-in)
   and extracts paired intermediate ("detail") and top ("interface") tokens at 0.25s temporal stride.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from host_forensics.config import BackboneConfig, TriSTMoEConfig
from host_forensics.models.tri_stmoe import TriSTMoEAdapter


class OrthogonalSVDLinear(nn.Module):
    """Wraps a frozen `nn.Linear` with principal-subspace-orthogonal low-rank adaptation."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        principal_rank: int = 64,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        out_features, in_features = base_linear.weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale

        # Freeze pretrained weight & bias
        self.weight0 = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias0: nn.Parameter | None = nn.Parameter(
                base_linear.bias.detach().clone(), requires_grad=False
            )
        else:
            self.register_parameter("bias0", None)

        # Compute truncated principal singular subspaces U_k, V_k of W_0 once at initialization
        kp = min(principal_rank, min(out_features, in_features) - 1)
        with torch.no_grad():
            u, _, vh = torch.linalg.svd(self.weight0.float(), full_matrices=False)
            u_k = u[:, :kp].contiguous()   # (d_out, kp)
            v_k = vh[:kp, :].T.contiguous()  # (d_in, kp)

        self.register_buffer("u_k", u_k, persistent=True)
        self.register_buffer("v_k", v_k, persistent=True)

        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))

    def delta_weight_orthogonal(self) -> torch.Tensor:
        """Compute Delta W_perp = (I - U_k U_k^T) B A (I - V_k V_k^T)."""
        a_fp32 = self.lora_a.float()
        b_fp32 = self.lora_b.float()
        # Project rows of A orthogonal to V_k: A_perp = A - (A V_k) V_k^T
        a_perp = a_fp32 - (a_fp32 @ self.v_k) @ self.v_k.T
        # Project cols of B orthogonal to U_k: B_perp = B - U_k (U_k^T B)
        b_perp = b_fp32 - self.u_k @ (self.u_k.T @ b_fp32)
        return (b_perp @ a_perp) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight0.to(dtype=x.dtype), self.bias0.to(dtype=x.dtype) if self.bias0 is not None else None)
        dw = self.delta_weight_orthogonal().to(dtype=x.dtype)
        return base_out + F.linear(x, dw)


class _StandInViTBlock(nn.Module):
    """Frozen Vision Transformer block with optional OrthogonalSVDLinear Q/V adaptation."""

    def __init__(self, dim: int, num_heads: int, use_ortho_lora: bool, lora_rank: int, principal_rank: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        q_lin = nn.Linear(dim, dim)
        k_lin = nn.Linear(dim, dim)
        v_lin = nn.Linear(dim, dim)
        out_lin = nn.Linear(dim, dim)

        for p in [q_lin, k_lin, v_lin, out_lin]:
            for param in p.parameters():
                param.requires_grad_(False)

        if use_ortho_lora:
            self.q_proj: nn.Module = OrthogonalSVDLinear(q_lin, rank=lora_rank, principal_rank=principal_rank)
            self.v_proj: nn.Module = OrthogonalSVDLinear(v_lin, rank=lora_rank, principal_rank=principal_rank)
        else:
            self.q_proj = q_lin
            self.v_proj = v_lin
        self.k_proj = k_lin
        self.out_proj = out_lin

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, P, D)
        h = self.norm1(x)
        q = rearrange(self.q_proj(h), "n p (h d) -> n h p d", h=self.num_heads)
        k = rearrange(self.k_proj(h), "n p (h d) -> n h p d", h=self.num_heads)
        v = rearrange(self.v_proj(h), "n p (h d) -> n h p d", h=self.num_heads)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out_proj(rearrange(attn, "n h p d -> n p (h d)"))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiPriorVisualBackbone(nn.Module):
    """Frozen multi-layer ViT backbone (DINOv3 / CLIP or stand-in) coupled with Tri-STMoE adapters."""

    def __init__(
        self,
        cfg: BackboneConfig,
        moe_cfg: TriSTMoEConfig,
        mn_res_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.visual_embed_dim
        self.patch_size = cfg.patch_size
        self.tap_layers = set(cfg.tap_layers)
        self.adapted_layers = set(cfg.adapted_layers)

        # Build Tri-STMoE adapters keyed by 1-indexed transformer block number
        self.moe_adapters = nn.ModuleDict(
            {
                str(layer_idx): TriSTMoEAdapter(cfg.visual_embed_dim, mn_res_dim, moe_cfg)
                for layer_idx in cfg.adapted_layers
            }
        )

        # Learnable multi-depth layer fusion gates across tap_layers
        self.tap_layer_list = sorted(cfg.tap_layers)
        self.layer_weights = nn.Parameter(torch.ones(len(self.tap_layer_list), dtype=torch.float32))
        self.skip_proj = nn.Linear(cfg.visual_embed_dim, cfg.visual_embed_dim // 2)

        self._hf_model: nn.Module | None = None
        if cfg.visual_backend == "hf":
            self._init_hf_backbone(cfg)
        else:
            self._init_standin_backbone(cfg)

    def _init_hf_backbone(self, cfg: BackboneConfig) -> None:
        from transformers import AutoModel  # Optional dependency (`pip install host-forensics[hub]`)

        hf_model = AutoModel.from_pretrained(cfg.visual_name)
        for param in hf_model.parameters():
            param.requires_grad_(False)

        if cfg.use_ortho_lora:
            # Wrap query/value linear projections inside encoder blocks with OrthogonalSVDLinear
            for name, module in list(hf_model.named_modules()):
                if any(tag in name.lower() for tag in ("q_proj", "query", "v_proj", "value")) and isinstance(
                    module, nn.Linear
                ):
                    parent_path, attr = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = hf_model.get_submodule(parent_path) if parent_path else hf_model
                    setattr(
                        parent,
                        attr,
                        OrthogonalSVDLinear(
                            module,
                            rank=cfg.lora_rank,
                            principal_rank=cfg.principal_subspace_rank,
                        ),
                    )
        self._hf_model = hf_model

    def _init_standin_backbone(self, cfg: BackboneConfig) -> None:
        self.patch_embed = nn.Conv2d(
            3, cfg.visual_embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        for p in self.patch_embed.parameters():
            p.requires_grad_(False)

        num_heads = max(4, cfg.visual_embed_dim // 64)
        self.blocks = nn.ModuleList(
            [
                _StandInViTBlock(
                    dim=cfg.visual_embed_dim,
                    num_heads=num_heads,
                    use_ortho_lora=cfg.use_ortho_lora,
                    lora_rank=cfg.lora_rank,
                    principal_rank=cfg.principal_subspace_rank,
                )
                for _ in range(cfg.visual_num_layers)
            ]
        )

    def forward(
        self, video: torch.Tensor, res_mn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            video: (B, T, 3, H, W)
            res_mn: (B, T, P, D_mn) residual tokens from MN-Stem
        Returns:
            fused_tokens: (B, T, P, D) multi-layer adapted patch tokens
            shallow_skip: (B, T, P, D//2) early-layer geometric skip tokens for mask decoding
        """
        b, t, _, h, w = video.shape
        p_expected = (h // self.patch_size) * (w // self.patch_size)

        if self._hf_model is not None:
            frames = rearrange(video, "b t c h w -> (b t) c h w")
            outputs = self._hf_model(pixel_values=frames, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (B*T, seq_len, D)
            tapped: list[torch.Tensor] = []
            for layer_idx in range(1, len(hidden_states)):
                hs = hidden_states[layer_idx]
                # Strip CLS + optional DINOv3 register tokens to retain trailing P spatial patch tokens
                patch_hs = hs[:, -p_expected:, :]
                tokens_btpd = rearrange(patch_hs, "(b t) p d -> b t p d", b=b, t=t)
                if str(layer_idx) in self.moe_adapters:
                    tokens_btpd = self.moe_adapters[str(layer_idx)](tokens_btpd, res_mn)
                if layer_idx in self.tap_layers:
                    tapped.append(tokens_btpd)
        else:
            frames = rearrange(video, "b t c h w -> (b t) c h w")
            x = rearrange(self.patch_embed(frames), "bt d hp wp -> bt (hp wp) d")
            tapped = []
            for idx, blk in enumerate(self.blocks, start=1):
                x = blk(x)
                if str(idx) in self.moe_adapters:
                    x_btpd = rearrange(x, "(b t) p d -> b t p d", b=b, t=t)
                    x_btpd = self.moe_adapters[str(idx)](x_btpd, res_mn)
                    x = rearrange(x_btpd, "b t p d -> (b t) p d")
                if idx in self.tap_layers:
                    tapped.append(rearrange(x, "(b t) p d -> b t p d", b=b, t=t))

        gate = F.softmax(self.layer_weights, dim=0)
        fused = sum(w_i * t_i for w_i, t_i in zip(gate, tapped))
        shallow_skip = self.skip_proj(tapped[0])
        return fused, shallow_skip

    @torch.no_grad()
    def step_moe_biases(self) -> None:
        for adapter in self.moe_adapters.values():
            adapter.step_biases()  # type: ignore[operator]


class FrozenAudioBackbone(nn.Module):
    """Extracts frozen intermediate ("detail") and top ("interface") audio tokens at 0.25s stride.

    Follows AV-FABLE / NFSR (Oct 2026): reading intermediate perceptual + top interface features
    recovers >90 AUROC forensic signals that zero-shot AV-LLM text interfaces lose.
    """

    def __init__(self, cfg: BackboneConfig, out_dim: int = 256) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim
        self.samples_per_step = int(round(cfg.sample_rate * cfg.audio_stride_sec))

        # Frozen stand-in acoustic front-end (can be swapped for HF WavLMModel when `visual_backend=="hf"`)
        self.frozen_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2, bias=False),
            nn.GELU(),
            nn.Conv1d(256, cfg.audio_embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GELU(),
        )
        self.frozen_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.audio_embed_dim,
                    nhead=8,
                    dim_feedforward=cfg.audio_embed_dim * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(4)
            ]
        )
        for p in self.frozen_conv.parameters():
            p.requires_grad_(False)
        for p in self.frozen_blocks.parameters():
            p.requires_grad_(False)

        # Trainable NFSR-style gated detail+interface readout projection -> out_dim
        self.detail_proj = nn.Linear(cfg.audio_embed_dim, out_dim)
        self.interface_proj = nn.Linear(cfg.audio_embed_dim, out_dim)
        self.depth_gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, waveform: torch.Tensor, target_steps: int | None = None) -> torch.Tensor:
        """Args:
            waveform: (B, L_a) raw mono waveform at `cfg.sample_rate` (16 kHz)
            target_steps: optional exact number of 0.25s positions N = ceil(T_sec / 0.25)
        Returns:
            audio_tokens: (B, N_a, out_dim)
        """
        if waveform.ndim == 2:
            wav_in = waveform.unsqueeze(1)  # (B, 1, L_a)
        else:
            wav_in = waveform

        n_steps = target_steps or max(1, int(round(wav_in.shape[-1] / float(self.samples_per_step))))

        with torch.no_grad():
            feat = self.frozen_conv(wav_in)  # (B, D_a, L')
            feat_pooled = F.adaptive_avg_pool1d(feat, output_size=n_steps)  # (B, D_a, N_a)
            h = feat_pooled.transpose(1, 2)  # (B, N_a, D_a)
            mid_feat = h
            for idx, blk in enumerate(self.frozen_blocks):
                h = blk(h)
                if idx == 1:
                    mid_feat = h
            top_feat = h

        d_tok = self.detail_proj(mid_feat.to(dtype=self.detail_proj.weight.dtype))
        u_tok = self.interface_proj(top_feat.to(dtype=self.interface_proj.weight.dtype))
        g = self.depth_gate(torch.cat([d_tok, u_tok], dim=-1))
        fused = g * d_tok + (1.0 - g) * u_tok
        return self.norm(fused)
