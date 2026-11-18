"""Frozen Perceptual Foundation Backbones & Orthogonal SVD-LoRA Adaptation.

Implements:
1. `OrthogonalSVDLinear`: Principal-subspace-preserving low-rank residual adaptation (Effort, ICML'25):
       W = W_0 + (I - U_k U_k^T) B A (I - V_k V_k^T)
   guaranteeing that learned forensic updates lie strictly in the orthogonal complement of the
   pretrained foundation model's principal semantic directions (with `orthogonal=False` supporting
   apples-to-apples standard unconstrained LoRA `W = W_0 + B A` for Ablation A1).
2. `MultiPriorVisualBackbone`: Wraps HuggingFace DINOv3 / DINOv2 / CLIP / ViT models (or an
   offline-safe ViT stand-in for CI/testing), injects `TriSTMoEAdapter` **in-stream** via
   transformer block forward hooks (preserving CLS + register tokens), and taps intermediate
   patch tokens across depths without storing unused hidden states.
3. `FrozenAudioBackbone`: Wraps HuggingFace `WavLMModel` (`microsoft/wavlm-large`, or an
   offline-safe stand-in) and extracts paired intermediate ("detail") and top ("interface")
   tokens aligned at 0.25s temporal stride.
"""

from __future__ import annotations

from collections import OrderedDict
import inspect
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from host_forensics.config import BackboneConfig, TriSTMoEConfig
from host_forensics.models.tri_stmoe import TriSTMoEAdapter

_WAVLM_FROZEN_CACHE: OrderedDict[tuple[str, int, bytes], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()


def _discover_hf_encoder_layers(hf_model: nn.Module, expected_layers: int) -> nn.ModuleList:
    """Locate the transformer block `nn.ModuleList` across HF DINOv3, DINOv2, ViT, CLIP, and SigLIP."""
    candidate_paths = (
        "encoder.layer",               # Dinov2Model, ViTModel
        "encoder.layers",              # CLIPVisionTransformer, SiglipVisionTransformer
        "vision_model.encoder.layers", # CLIPVisionModel, SiglipVisionModel
        "layer",                       # DINOv3ViTModel
        "layers",                      # Generic / timm-wrapped HF towers
        "blocks",                      # timm VisionTransformer
    )
    for path in candidate_paths:
        try:
            sub = hf_model.get_submodule(path)
            if isinstance(sub, nn.ModuleList) and len(sub) > 0:
                return sub
        except AttributeError:
            continue

    # Fallback: locate the ModuleList matching `expected_layers` (or largest >= 2)
    best: nn.ModuleList | None = None
    for _, module in hf_model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            if len(module) == expected_layers:
                return module
            if best is None or len(module) > len(best):
                best = module

    if best is not None:
        return best
    raise ValueError(
        f"Could not locate transformer block ModuleList inside {type(hf_model).__name__}."
    )


class OrthogonalSVDLinear(nn.Module):
    """Wraps a frozen `nn.Linear` with principal-subspace-orthogonal (or standard) low-rank adaptation."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        principal_rank: int = 64,
        scale: float = 1.0,
        orthogonal: bool = True,
    ) -> None:
        super().__init__()
        out_features, in_features = base_linear.weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale
        self.orthogonal = orthogonal

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
            u_k = u[:, :kp].contiguous()     # (d_out, kp)
            v_k = vh[:kp, :].T.contiguous()  # (d_in, kp)

        self.register_buffer("u_k", u_k, persistent=True)
        self.register_buffer("v_k", v_k, persistent=True)

        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self._merged_eval_weight: torch.Tensor | None = None

    def train(self, mode: bool = True) -> "OrthogonalSVDLinear":
        super().train(mode)
        if mode:
            self._merged_eval_weight = None
        return self

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._merged_eval_weight = None
        super()._load_from_state_dict(*args, **kwargs)

    def orthogonal_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rank-r factor pair `(a_eff, b_eff)` of shapes `(r, d_in)` and `(d_out, r)`."""
        a_fp32 = self.lora_a.float()
        b_fp32 = self.lora_b.float()
        if not self.orthogonal:
            return a_fp32, b_fp32 * self.scale
        a_perp = a_fp32 - (a_fp32 @ self.v_k) @ self.v_k.T
        b_perp = (b_fp32 - self.u_k @ (self.u_k.T @ b_fp32)) * self.scale
        return a_perp, b_perp

    def delta_weight_orthogonal(self) -> torch.Tensor:
        """Compute full Delta W_perp = b_eff @ a_eff (used for inspection and one-shot eval merging)."""
        a_eff, b_eff = self.orthogonal_factors()
        return b_eff @ a_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias0.to(dtype=x.dtype) if self.bias0 is not None else None
        if not self.training:
            if (
                self._merged_eval_weight is None
                or self._merged_eval_weight.device != x.device
                or self._merged_eval_weight.dtype != x.dtype
            ):
                with torch.no_grad():
                    merged_fp32 = self.weight0.float() + self.delta_weight_orthogonal()
                    self._merged_eval_weight = merged_fp32.to(device=x.device, dtype=x.dtype)
            return F.linear(x, self._merged_eval_weight, bias)

        # Factorized O(N * r * (d_in + d_out)) training path (32x fewer FLOPs than dense d_out x d_in)
        a_eff, b_eff = self.orthogonal_factors()
        base_out = F.linear(x, self.weight0.to(dtype=x.dtype), bias)
        lora_out = F.linear(F.linear(x, a_eff.to(dtype=x.dtype)), b_eff.to(dtype=x.dtype))
        return base_out + lora_out


class _StandInViTBlock(nn.Module):
    """Frozen Vision Transformer block with OrthogonalSVDLinear (or standard LoRA) Q/V adaptation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_ortho_lora: bool,
        lora_rank: int,
        principal_rank: int,
    ) -> None:
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

        self.q_proj: nn.Module = OrthogonalSVDLinear(
            q_lin, rank=lora_rank, principal_rank=principal_rank, orthogonal=use_ortho_lora
        )
        self.v_proj: nn.Module = OrthogonalSVDLinear(
            v_lin, rank=lora_rank, principal_rank=principal_rank, orthogonal=use_ortho_lora
        )
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
    """Frozen multi-layer ViT backbone (DINOv3 / CLIP or stand-in) coupled with in-stream Tri-STMoE adapters."""

    def __init__(
        self,
        cfg: BackboneConfig,
        moe_cfg: TriSTMoEConfig,
        mn_res_dim: int,
        hf_model: nn.Module | None = None,
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
        self._hf_accepts_interp_pos: bool = False
        self._hook_handles: list[Any] = []
        self._active_forward_ctx: dict[str, Any] | None = None

        if hf_model is not None or cfg.visual_backend == "hf":
            self._init_hf_backbone(cfg, preloaded_model=hf_model)
        else:
            self._init_standin_backbone(cfg)

    def _init_hf_backbone(
        self, cfg: BackboneConfig, preloaded_model: nn.Module | None = None
    ) -> None:
        if preloaded_model is None:
            from transformers import AutoModel  # Optional dependency (`pip install host-forensics[hub]`)

            hf_model = AutoModel.from_pretrained(cfg.visual_name)
        else:
            hf_model = preloaded_model

        for param in hf_model.parameters():
            param.requires_grad_(False)
        hf_model.eval()

        encoder_layers = _discover_hf_encoder_layers(hf_model, expected_layers=cfg.visual_num_layers)

        # Wrap query/value linear projections strictly inside encoder blocks
        target_leaf_names = {"q_proj", "query", "v_proj", "value", "qkv"}
        for block in encoder_layers:
            for name, module in list(block.named_modules()):
                leaf = name.rsplit(".", 1)[-1] if "." in name else name
                if leaf in target_leaf_names and isinstance(module, nn.Linear):
                    parent_path, attr = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = block.get_submodule(parent_path) if parent_path else block
                    setattr(
                        parent,
                        attr,
                        OrthogonalSVDLinear(
                            module,
                            rank=cfg.lora_rank,
                            principal_rank=cfg.principal_subspace_rank,
                            orthogonal=cfg.use_ortho_lora,
                        ),
                    )

        # Register in-stream forward hooks on adapted and tapped transformer blocks (1-indexed)
        hook_targets = self.adapted_layers | self.tap_layers
        for layer_idx, block in enumerate(encoder_layers, start=1):
            if layer_idx in hook_targets:
                handle = block.register_forward_hook(self._make_block_hook(layer_idx))
                self._hook_handles.append(handle)

        try:
            sig = inspect.signature(hf_model.forward)
            self._hf_accepts_interp_pos = "interpolate_pos_encoding" in sig.parameters
        except (TypeError, ValueError):
            self._hf_accepts_interp_pos = False

        self._hf_model = hf_model

    def _make_block_hook(self, layer_idx: int) -> Any:
        def _hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            ctx = self._active_forward_ctx
            if ctx is None:
                return output

            is_tuple = isinstance(output, tuple)
            hs: torch.Tensor = output[0] if is_tuple else output
            b: int = ctx["b"]
            t: int = ctx["t"]
            p_expected: int = ctx["p_expected"]
            res_mn: torch.Tensor = ctx["res_mn"]

            # Separate optional prefix tokens (CLS + DINOv2/v3 register tokens) from P spatial patches
            prefix = hs[:, :-p_expected, :]
            patch_hs = hs[:, -p_expected:, :]
            tokens_btpd = rearrange(patch_hs, "(b t) p d -> b t p d", b=b, t=t)

            if str(layer_idx) in self.moe_adapters:
                tokens_btpd = self.moe_adapters[str(layer_idx)](tokens_btpd, res_mn)
                adapted_flat = rearrange(tokens_btpd, "b t p d -> (b t) p d").to(dtype=hs.dtype)
                hs = torch.cat([prefix, adapted_flat], dim=1)

            if layer_idx in self.tap_layers:
                ctx["tapped"][layer_idx] = tokens_btpd

            if is_tuple:
                return (hs, *output[1:])
            return hs

        return _hook

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

    def train(self, mode: bool = True) -> "MultiPriorVisualBackbone":
        super().train(mode)
        # Keep frozen pretrained ViT deterministic while letting OrthogonalSVDLinear track train/eval
        if self._hf_model is not None:
            self._hf_model.eval()
            for mod in self._hf_model.modules():
                if isinstance(mod, OrthogonalSVDLinear):
                    mod.train(mode)
        return self

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
            ctx: dict[str, Any] = {
                "b": b,
                "t": t,
                "p_expected": p_expected,
                "res_mn": res_mn,
                "tapped": {},
            }
            self._active_forward_ctx = ctx
            try:
                hf_kwargs: dict[str, Any] = {"pixel_values": frames}
                if self._hf_accepts_interp_pos:
                    hf_kwargs["interpolate_pos_encoding"] = True
                self._hf_model(**hf_kwargs)
                tapped = [ctx["tapped"][idx] for idx in self.tap_layer_list]
            finally:
                self._active_forward_ctx = None
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

    Follows AV-FABLE / NFSR (2026): reading intermediate perceptual + top interface features from
    frozen WavLM-Large recovers >90 AUROC forensic signals that zero-shot AV-LLM text prompts lose.
    """

    def __init__(
        self,
        cfg: BackboneConfig,
        out_dim: int = 256,
        hf_audio_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim
        self.samples_per_step = int(round(cfg.sample_rate * cfg.audio_stride_sec))

        self._hf_audio_model: nn.Module | None = None
        audio_feat_dim = cfg.audio_embed_dim

        if hf_audio_model is not None or (cfg.visual_backend == "hf" and bool(cfg.audio_name)):
            if hf_audio_model is None:
                from transformers import AutoModel  # Optional dependency (`pip install host-forensics[hub]`)

                hf_aud = AutoModel.from_pretrained(cfg.audio_name)
            else:
                hf_aud = hf_audio_model

            for p in hf_aud.parameters():
                p.requires_grad_(False)
            hf_aud.eval()
            audio_feat_dim = int(getattr(getattr(hf_aud, "config", None), "hidden_size", cfg.audio_embed_dim))
            self._hf_audio_model = hf_aud
        else:
            # Offline-safe frozen acoustic stand-in for CI and unit tests
            self.frozen_conv = nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=3, bias=False),
                nn.GELU(),
                nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2, bias=False),
                nn.GELU(),
                nn.Conv1d(256, audio_feat_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GELU(),
            )
            self.frozen_blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=audio_feat_dim,
                        nhead=8,
                        dim_feedforward=audio_feat_dim * 2,
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
        self.detail_proj = nn.Linear(audio_feat_dim, out_dim)
        self.interface_proj = nn.Linear(audio_feat_dim, out_dim)
        self.depth_gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def train(self, mode: bool = True) -> "FrozenAudioBackbone":
        super().train(mode)
        # Keep frozen WavLM in eval() so layerdrop / feature dropout stays disabled during training
        if self._hf_audio_model is not None:
            self._hf_audio_model.eval()
        return self

    def forward(self, waveform: torch.Tensor, target_steps: int | None = None) -> torch.Tensor:
        """Args:
            waveform: (B, L_a) or (B, 1, L_a) raw mono waveform at `cfg.sample_rate` (16 kHz)
            target_steps: optional exact number of 0.25s positions N = ceil(T_sec / 0.25)
        Returns:
            audio_tokens: (B, N_a, out_dim)
        """
        wav_2d = waveform.squeeze(1) if waveform.ndim == 3 else waveform
        n_steps = target_steps or max(1, int(round(wav_2d.shape[-1] / float(self.samples_per_step))))

        if self._hf_audio_model is not None:
            with torch.no_grad():
                mid_list: list[torch.Tensor | None] = [None] * wav_2d.shape[0]
                top_list: list[torch.Tensor | None] = [None] * wav_2d.shape[0]
                uncached_indices: list[int] = []
                uncached_keys: list[tuple[str, int, bytes]] = []
                model_id = str(getattr(self.cfg, "audio_name", "wavlm"))

                for bi in range(wav_2d.shape[0]):
                    # Fast 64-sample strided fingerprint + L1 energy for exact waveform lookup
                    w_cpu = wav_2d[bi].detach().float().cpu()
                    stride = max(1, w_cpu.numel() // 64)
                    sig = torch.cat([w_cpu[::stride][:64], w_cpu.abs().mean().view(1)]).numpy().tobytes()
                    ckey = (model_id, n_steps, sig)
                    cached = _WAVLM_FROZEN_CACHE.get(ckey)
                    if cached is not None:
                        _WAVLM_FROZEN_CACHE.move_to_end(ckey)
                        mid_list[bi] = cached[0].to(device=wav_2d.device)
                        top_list[bi] = cached[1].to(device=wav_2d.device)
                    else:
                        uncached_indices.append(bi)
                        uncached_keys.append(ckey)

                if uncached_indices:
                    sub_wav = wav_2d[uncached_indices].float()
                    wav_norm = (sub_wav - sub_wav.mean(dim=-1, keepdim=True)) / (
                        sub_wav.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
                    )
                    aud_dtype = next(self._hf_audio_model.parameters()).dtype
                    outputs = self._hf_audio_model(
                        input_values=wav_norm.to(dtype=aud_dtype),
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states
                    mid_idx = len(hidden_states) // 2
                    mid_sub = F.adaptive_avg_pool1d(
                        hidden_states[mid_idx].transpose(1, 2), output_size=n_steps
                    ).transpose(1, 2)
                    top_sub = F.adaptive_avg_pool1d(
                        hidden_states[-1].transpose(1, 2), output_size=n_steps
                    ).transpose(1, 2)

                    for local_j, (bi, ckey) in enumerate(zip(uncached_indices, uncached_keys)):
                        m_j = mid_sub[local_j]
                        t_j = top_sub[local_j]
                        mid_list[bi] = m_j
                        top_list[bi] = t_j
                        _WAVLM_FROZEN_CACHE[ckey] = (m_j.detach().float().cpu(), t_j.detach().float().cpu())
                        if len(_WAVLM_FROZEN_CACHE) > 768:
                            _WAVLM_FROZEN_CACHE.popitem(last=False)

                mid_feat = torch.stack(mid_list, dim=0)  # type: ignore[arg-type]
                top_feat = torch.stack(top_list, dim=0)  # type: ignore[arg-type]
        else:
            wav_in = wav_2d.unsqueeze(1)  # (B, 1, L_a)
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

        proj_dtype = self.detail_proj.weight.dtype
        d_tok = self.detail_proj(mid_feat.to(dtype=proj_dtype))
        u_tok = self.interface_proj(top_feat.to(dtype=proj_dtype))
        g = self.depth_gate(torch.cat([d_tok, u_tok], dim=-1))
        fused = g * d_tok + (1.0 - g) * u_tok
        return self.norm(fused)

