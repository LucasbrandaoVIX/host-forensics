"""Typed configuration schemas for HoST-MoE architecture, heterogeneous training, and RTX 5090 optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Tuple
import yaml


@dataclass
class BackboneConfig:
    """Configuration for frozen perceptual encoders and Orthogonal SVD-LoRA adaptation."""

    visual_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    visual_backend: str = "standin"  # "hf" for HuggingFace DINOv3/CLIP, "standin" for offline-safe ViT
    img_size: int = 384
    patch_size: int = 16
    visual_embed_dim: int = 1024
    visual_num_layers: int = 24
    tap_layers: Tuple[int, ...] = (6, 12, 18, 24)
    adapted_layers: Tuple[int, ...] = (4, 8, 12, 16, 20, 24)

    # Orthogonal SVD-LoRA (Effort-style principal subspace preservation)
    use_ortho_lora: bool = True
    lora_rank: int = 16
    principal_subspace_rank: int = 64

    # Optional frozen Audio encoder (WavLM-Large / Qwen3-Omni audio tower)
    audio_name: str = "microsoft/wavlm-large"
    audio_embed_dim: int = 1024
    audio_stride_sec: float = 0.25
    sample_rate: int = 16000


@dataclass
class MNStemConfig:
    """Configuration for the Motion-Noise Forensic Residual Stem (MN-Stem)."""

    bayar_channels: int = 12
    srm_channels: int = 9
    shallow_feat_channels: int = 32
    corr_radius: int = 2  # (2r+1)x(2r+1) = 5x5 local correlation neighborhood
    corr_temperature: float = 0.1
    out_dim: int = 128


@dataclass
class TriSTMoEConfig:
    """Configuration for Tri-Pool Auxiliary-Loss-Free Spatio-Temporal-Residual MoE."""

    bottleneck_dim: int = 128
    num_experts_per_pool: int = 4
    top_k: int = 2
    num_heads: int = 4
    dropout: float = 0.05
    bias_update_rate: float = 1e-4  # DeepSeek-V3 / MoSTE auxiliary-loss-free bias step


@dataclass
class SHDCConfig:
    """Configuration for Self-Referential Host-Prototype Differential Conditioning."""

    anchor_hidden_dim: int = 256
    sharpening_beta: float = 2.0
    eps: float = 1e-5


@dataclass
class ReadoutConfig:
    """Configuration for Modality-Decoupled Native Readouts (MDNR), KP-Verify, and Mask Head."""

    hidden_dim: int = 256
    pyramid_levels: int = 4
    pyramid_kernel_size: int = 3
    soft_nms_sigma: float = 0.5
    max_proposals: int = 20
    mask_base_channels: int = 128
    default_far_budget: float = 0.05


@dataclass
class LossConfig:
    """Weights for the heterogeneous multi-task forensic training objective."""

    lambda_cls: float = 1.0
    lambda_polarization: float = 0.25
    lambda_anchor: float = 0.50
    lambda_tfl_focal: float = 1.0
    lambda_tfl_diou: float = 1.0
    lambda_mask_bce: float = 1.0
    lambda_mask_dice: float = 1.0
    lambda_mask_boundary: float = 0.5
    mil_top_host_ratio: float = 0.25
    mil_bottom_fake_ratio: float = 0.10
    mil_margin: float = 0.60


@dataclass
class TrainConfig:
    """RTX 5090 (32 GB GDDR7) BF16 training & symmetric laundering configuration."""

    precision: str = "bf16"
    batch_size: int = 4
    grad_accum_steps: int = 4
    num_frames: int = 16
    clip_duration_sec: float = 4.0
    lr: float = 3e-4
    weight_decay: float = 5e-4
    epochs_stage1: int = 10
    epochs_stage2_highres: int = 4
    highres_img_size: int = 448
    modality_dropout_prob: float = 0.35
    compound_degradation_prob: float = 0.50
    symmetric_laundering_prob: float = 0.15  # Applied equally to real & fake!
    grad_clip_norm: float = 1.0


@dataclass
class HoSTConfig:
    """Top-level configuration container for HoST-MoE."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    mn_stem: MNStemConfig = field(default_factory=MNStemConfig)
    moe: TriSTMoEConfig = field(default_factory=TriSTMoEConfig)
    shdc: SHDCConfig = field(default_factory=SHDCConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HoSTConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        def _merge(dc_cls: Any, sub: dict[str, Any] | None) -> Any:
            if not sub:
                return dc_cls()
            converted = {}
            for k, v in sub.items():
                if isinstance(v, list) and k in {"tap_layers", "adapted_layers"}:
                    converted[k] = tuple(v)
                else:
                    converted[k] = v
            return dc_cls(**converted)

        return cls(
            backbone=_merge(BackboneConfig, raw.get("backbone")),
            mn_stem=_merge(MNStemConfig, raw.get("mn_stem")),
            moe=_merge(TriSTMoEConfig, raw.get("moe")),
            shdc=_merge(SHDCConfig, raw.get("shdc")),
            readout=_merge(ReadoutConfig, raw.get("readout")),
            loss=_merge(LossConfig, raw.get("loss")),
            train=_merge(TrainConfig, raw.get("train")),
        )
