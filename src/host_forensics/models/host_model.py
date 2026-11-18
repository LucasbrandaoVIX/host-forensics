"""Unified HoST-MoE Forensics Model & Pristine Quantile Calibrator."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from host_forensics.config import HoSTConfig
from host_forensics.models.backbones import FrozenAudioBackbone, MultiPriorVisualBackbone
from host_forensics.models.heads import (
    KinematicProsodicVerifier,
    MultiScaleTemporalPyramidDecoder,
    SpatialTubeletMaskDecoder,
    TemporalProposal,
)
from host_forensics.models.mn_stem import MotionNoiseResidualStem
from host_forensics.models.shdc import HostConditioningOutput, SelfReferentialHostConditioning


ATTRIBUTION_CLASSES = ("clean", "visual_only", "audio_only", "both_track")


@dataclass
class HoSTOutput:
    """Structured forward/inference output of HoST-MoE."""

    video_logit: torch.Tensor                        # (B,) raw manipulation logit
    visual_track_logit: torch.Tensor                 # (B,) visual manipulation logit
    audio_track_logit: torch.Tensor                  # (B,) audio manipulation logit (masked when silent)
    mask_logits: torch.Tensor                        # (B, T, H, W) dense spatial-temporal mask logits
    visual_shdc: HostConditioningOutput              # includes anchor_probs (B, T, P) and mu/sigma_host
    audio_shdc: HostConditioningOutput | None        # includes audio anchor_probs (B, T_a) if audio active
    visual_pyramid: dict[str, object]                # multi-scale temporal logits/bounds & proposals
    audio_pyramid: dict[str, object] | None          # multi-scale audio logits/bounds & proposals
    has_audio: torch.Tensor                          # (B,) boolean tensor
    calibrated_prob: torch.Tensor | None = None      # (B,) calibrated probability in [0, 1]
    attribution_pred: list[str] | None = None        # 4-way track attribution labels


class PristineQuantileCalibrator(nn.Module):
    """Calibrates visual and audio decision thresholds once on pristine source-validation videos.

    Eliminates per-dataset test-time threshold tuning and maintains a bounded false-alarm rate (FAR)
    under domain shift (following AV-FABLE & MID-FFD 2026 best practices).
    """

    def __init__(self, far_budget: float = 0.05) -> None:
        super().__init__()
        self.far_budget = far_budget
        self.register_buffer("visual_thresh", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("audio_thresh", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("visual_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("audio_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(
        self,
        pristine_visual_logits: torch.Tensor,
        pristine_audio_logits: torch.Tensor | None = None,
    ) -> None:
        q = 1.0 - self.far_budget
        v_clean = pristine_visual_logits.detach().float().flatten()
        if v_clean.numel() == 0:
            raise ValueError("pristine_visual_logits cannot be empty for calibration.")
        self.visual_thresh.copy_(torch.quantile(v_clean, q))
        self.visual_scale.copy_(v_clean.std(unbiased=False).clamp(min=0.25))

        if pristine_audio_logits is not None and pristine_audio_logits.numel() > 0:
            a_clean = pristine_audio_logits.detach().float().flatten()
            self.audio_thresh.copy_(torch.quantile(a_clean, q))
            self.audio_scale.copy_(a_clean.std(unbiased=False).clamp(min=0.25))

        self.is_calibrated.fill_(True)

    def calibrate_decisions(
        self,
        visual_logit: torch.Tensor,
        audio_logit: torch.Tensor,
        has_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str]]:
        """Return calibrated manipulation probability and 4-way track attribution."""
        v_shifted = (visual_logit.float() - self.visual_thresh) / self.visual_scale
        v_prob = torch.sigmoid(v_shifted)
        v_flag = visual_logit.float() > self.visual_thresh

        a_shifted = (audio_logit.float() - self.audio_thresh) / self.audio_scale
        a_prob = torch.sigmoid(a_shifted)
        a_flag = (audio_logit.float() > self.audio_thresh) & has_audio.bool()

        joint_prob = torch.where(has_audio.bool(), torch.maximum(v_prob, a_prob), v_prob)

        labels: list[str] = []
        for vf, af in zip(v_flag.tolist(), a_flag.tolist()):
            if vf and af:
                labels.append("both_track")
            elif vf:
                labels.append("visual_only")
            elif af:
                labels.append("audio_only")
            else:
                labels.append("clean")

        return joint_prob, labels


class HoSTForensicsModel(nn.Module):
    """Unified Host-Differential Spatio-Temporal MoE Detector & Localizer."""

    def __init__(self, cfg: HoSTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HoSTConfig()

        # 1. Motion-Noise Forensic Residual Stem (MN-Stem)
        self.mn_stem = MotionNoiseResidualStem(
            self.cfg.mn_stem, patch_size=self.cfg.backbone.patch_size
        )

        # 2. Frozen Multi-Prior Visual Backbone + Orthogonal SVD-LoRA + Tri-STMoE Adapters
        self.visual_backbone = MultiPriorVisualBackbone(
            self.cfg.backbone,
            self.cfg.moe,
            mn_res_dim=self.cfg.mn_stem.out_dim,
        )

        # 3. Frozen Audio Backbone (gracefully skipped when silent)
        self.audio_backbone = FrozenAudioBackbone(
            self.cfg.backbone, out_dim=self.cfg.readout.hidden_dim
        )

        # 4. Self-Referential Host-Prototype Differential Conditioning (SHDC)
        self.visual_shdc = SelfReferentialHostConditioning(
            dim=self.cfg.backbone.visual_embed_dim, cfg=self.cfg.shdc
        )
        self.audio_shdc = SelfReferentialHostConditioning(
            dim=self.cfg.readout.hidden_dim, cfg=self.cfg.shdc
        )

        # 5. Directional Kinematic-Prosodic Verifier & Null-Audio Gate
        self.kp_verifier = KinematicProsodicVerifier(
            visual_dim=self.cfg.backbone.visual_embed_dim,
            audio_dim=self.cfg.readout.hidden_dim,
            out_dim=self.cfg.readout.hidden_dim,
            dropout_prob=self.cfg.train.modality_dropout_prob,
        )

        # Attention pooling from spatial patches -> frame-level sequence for Visual Pyramid
        self.spatial_anomaly_pool = nn.Linear(self.cfg.backbone.visual_embed_dim, 1)

        # 6. Modality-Separated Multi-Scale Temporal Pyramid Decoders (D_v and D_a)
        self.visual_pyramid = MultiScaleTemporalPyramidDecoder(
            in_dim=self.cfg.backbone.visual_embed_dim + self.cfg.readout.hidden_dim,
            cfg=self.cfg.readout,
            modality_name="visual",
        )
        self.audio_pyramid = MultiScaleTemporalPyramidDecoder(
            in_dim=self.cfg.readout.hidden_dim,
            cfg=self.cfg.readout,
            modality_name="audio",
        )

        # 7. Spatial-Temporal Tubelet Mask Head (D_mask)
        self.mask_decoder = SpatialTubeletMaskDecoder(
            embed_dim=self.cfg.backbone.visual_embed_dim,
            vit_skip_dim=self.cfg.backbone.visual_embed_dim // 2,
            mn_skip_dim=self.cfg.mn_stem.out_dim // 2,
            base_ch=self.cfg.readout.mask_base_channels,
            patch_size=self.cfg.backbone.patch_size,
        )

        # 8. Pristine-Validation Quantile Calibrator
        self.calibrator = PristineQuantileCalibrator(
            far_budget=self.cfg.readout.default_far_budget
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor | None = None,
        has_audio: torch.Tensor | None = None,
        duration_sec: float | None = None,
    ) -> HoSTOutput:
        """Args:
            video: (B, T, 3, H, W) normalized RGB frames
            audio: optional (B, L_a) mono waveform at 16 kHz
            has_audio: optional (B,) boolean/int indicator; inferred automatically if omitted
            duration_sec: clip duration in seconds (defaults to `cfg.train.clip_duration_sec`)
        """
        b, t, _, h, w = video.shape
        clip_dur = duration_sec if duration_sec is not None else self.cfg.train.clip_duration_sec

        if has_audio is None:
            has_aud_bool = torch.full(
                (b,), fill_value=(audio is not None), device=video.device, dtype=torch.bool
            )
        else:
            has_aud_bool = has_audio.to(device=video.device, dtype=torch.bool)

        # Step 1: Extract Motion-Noise Forensic Residuals + Multi-Layer Adapted Visual Tokens
        res_mn, mn_skip_qtr = self.mn_stem(video)
        vis_tokens, vit_skip_tokens = self.visual_backbone(video, res_mn)

        # Step 2: Visual Self-Referential Host-Prototype Differential Conditioning
        vis_shdc_out = self.visual_shdc(vis_tokens)

        # Step 3: Optional Audio Extraction & Audio SHDC
        aud_shdc_out: HostConditioningOutput | None = None
        aud_pyr_out: dict[str, object] | None = None
        if audio is not None and has_aud_bool.any():
            n_aud_steps = max(2, int(round(clip_dur / self.cfg.backbone.audio_stride_sec)))
            aud_tokens = self.audio_backbone(audio, target_steps=n_aud_steps)
            aud_shdc_out = self.audio_shdc(aud_tokens)
            aud_pyr_out = self.audio_pyramid(aud_shdc_out.tokens, duration_sec=clip_dur)
            raw_aud_logit = aud_pyr_out["track_logit"]  # type: ignore[assignment]
            assert isinstance(raw_aud_logit, torch.Tensor)
            audio_track_logit = torch.where(
                has_aud_bool, raw_aud_logit, torch.full_like(raw_aud_logit, -20.0)
            )
        else:
            audio_track_logit = torch.full((b,), -20.0, device=video.device, dtype=torch.float32)

        # Step 4: Directional Kinematic-Prosodic Verification (with Silent-Prior fallback)
        kp_ctx = self.kp_verifier(
            visual_tokens=vis_shdc_out.tokens,
            anchor_probs=vis_shdc_out.anchor_probs,
            audio_tokens=aud_shdc_out.tokens if aud_shdc_out is not None else None,
            has_audio=has_aud_bool,
        )

        # Pool spatial tokens per frame emphasizing anomalous patches (1 - anchor_prob + learned attn)
        patch_scores = self.spatial_anomaly_pool(vis_shdc_out.tokens).squeeze(-1) + (
            1.0 - vis_shdc_out.anchor_probs
        ).to(dtype=vis_shdc_out.tokens.dtype)
        patch_weights = torch.softmax(patch_scores, dim=-1)  # (B, T, P)
        frame_vis_seq = torch.einsum("btp,btpd->btd", patch_weights, vis_shdc_out.tokens)

        # Step 5: Visual Temporal Pyramid & Spatial Tubelet Mask Decoding
        vis_pyr_in = torch.cat([frame_vis_seq, kp_ctx], dim=-1)
        vis_pyr_out = self.visual_pyramid(vis_pyr_in, duration_sec=clip_dur)
        visual_track_logit = vis_pyr_out["track_logit"]  # type: ignore[assignment]
        assert isinstance(visual_track_logit, torch.Tensor)

        mask_logits = self.mask_decoder(
            shdc_tokens=vis_shdc_out.tokens,
            vit_skip_tokens=vit_skip_tokens,
            mn_skip_qtr=mn_skip_qtr,
            out_hw=(h, w),
        )

        # Video-level manipulation logit: soft-max / max over active tracks
        video_logit = torch.where(
            has_aud_bool,
            torch.logaddexp(visual_track_logit, audio_track_logit),
            visual_track_logit,
        )

        cal_prob, attr_labels = self.calibrator.calibrate_decisions(
            visual_track_logit, audio_track_logit, has_aud_bool
        )

        return HoSTOutput(
            video_logit=video_logit,
            visual_track_logit=visual_track_logit,
            audio_track_logit=audio_track_logit,
            mask_logits=mask_logits,
            visual_shdc=vis_shdc_out,
            audio_shdc=aud_shdc_out,
            visual_pyramid=vis_pyr_out,
            audio_pyramid=aud_pyr_out,
            has_audio=has_aud_bool,
            calibrated_prob=cal_prob,
            attribution_pred=attr_labels,
        )

    @torch.no_grad()
    def step_moe_biases(self) -> None:
        """Update auxiliary-loss-free routing biases across all adapted ViT blocks."""
        self.visual_backbone.step_moe_biases()

    def parameter_summary(self) -> dict[str, int | float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(100.0 * trainable / max(total, 1), 2),
        }

    def _inference_amp_context(self, device: torch.device) -> object:
        from contextlib import nullcontext

        if device.type == "cuda":
            if self.cfg.train.precision == "fp16":
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.cfg.train.precision in {"bf16", "fp32"}:
                # Default to BF16 Tensor Core + FlashAttention-2 acceleration on Ampere/Ada/Blackwell GPUs
                if self.cfg.train.precision == "bf16":
                    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def compile_forensic_submodules(self, mode: str = "default") -> None:
        """Apply `torch.compile(..., dynamic=True)` to trainable forensic adapters and decoders."""
        if not hasattr(torch, "compile"):
            return
        self.mn_stem = torch.compile(self.mn_stem, mode=mode, dynamic=True)  # type: ignore[assignment]
        self.visual_shdc = torch.compile(self.visual_shdc, mode=mode, dynamic=True)  # type: ignore[assignment]
        self.audio_shdc = torch.compile(self.audio_shdc, mode=mode, dynamic=True)  # type: ignore[assignment]
        self.kp_verifier = torch.compile(self.kp_verifier, mode=mode, dynamic=True)  # type: ignore[assignment]
        self.visual_pyramid = torch.compile(self.visual_pyramid, mode=mode, dynamic=True)  # type: ignore[assignment]
        self.audio_pyramid = torch.compile(self.audio_pyramid, mode=mode, dynamic=True)  # type: ignore[assignment]
        self.mask_decoder = torch.compile(self.mask_decoder, mode=mode, dynamic=True)  # type: ignore[assignment]
        for key in list(self.visual_backbone.moe_adapters.keys()):
            self.visual_backbone.moe_adapters[key] = torch.compile(  # type: ignore[assignment]
                self.visual_backbone.moe_adapters[key], mode=mode, dynamic=True
            )

    @torch.inference_mode()
    def predict_structured(
        self,
        video: torch.Tensor,
        audio: torch.Tensor | None = None,
        has_audio: torch.Tensor | None = None,
        duration_sec: float = 4.0,
    ) -> list[dict[str, object]]:
        """Run fast BF16/inference_mode forward pass and format JSON records per clip."""
        was_training = self.training
        self.eval()
        with self._inference_amp_context(video.device):  # type: ignore[attr-defined]
            out = self.forward(video, audio=audio, has_audio=has_audio, duration_sec=duration_sec)
        if was_training:
            self.train()

        vis_props: list[list[TemporalProposal]] = out.visual_pyramid["proposals"]  # type: ignore[assignment]
        aud_props: list[list[TemporalProposal]] | None = (
            out.audio_pyramid["proposals"] if out.audio_pyramid is not None else None  # type: ignore[assignment]
        )
        mask_probs = torch.sigmoid(out.mask_logits.float())

        results: list[dict[str, object]] = []
        for i in range(video.shape[0]):
            prob = float(out.calibrated_prob[i].item()) if out.calibrated_prob is not None else float(torch.sigmoid(out.video_logit[i]).item())
            attr = out.attribution_pred[i] if out.attribution_pred is not None else "unknown"
            results.append(
                {
                    "manipulated_probability": round(prob, 4),
                    "decision": "MANIPULATED" if attr != "clean" else "AUTHENTIC",
                    "attribution": attr,
                    "has_audio": bool(out.has_audio[i].item()),
                    "visual_intervals": [
                        [round(p.start_sec, 3), round(p.end_sec, 3), round(p.score, 4)]
                        for p in vis_props[i][:5]
                    ],
                    "audio_intervals": (
                        [
                            [round(p.start_sec, 3), round(p.end_sec, 3), round(p.score, 4)]
                            for p in aud_props[i][:5]
                        ]
                        if (aud_props is not None and bool(out.has_audio[i].item()))
                        else []
                    ),
                    "peak_mask_area_ratio": round(float((mask_probs[i] > 0.5).float().mean().item()), 4),
                }
            )
        return results
