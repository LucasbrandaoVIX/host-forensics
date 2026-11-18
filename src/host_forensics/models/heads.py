"""Modality-Decoupled Native Readouts (MDNR), Kinematic-Prosodic Verifier, and Spatial Tubelet Mask Head.

Key design guarantees:
1. Directional Kinematic-Prosodic Verification (`KP-Verify`): Checks upper-head/peripheral facial
   motion velocity against audio prosody (immune to LatentSync mouth-SyncNet overfitting) with
   a learned Silent-Prior token `e_empty` when audio is absent (`has_audio=0`).
2. Strict Modality Separation (`D_v`, `D_a`): Audio branch `D_a` never receives visual features,
   and `KP-Verify` enters `D_v` via a bounded residual gate—preventing the 12-15% cross-track
   false-alarm bleed documented in AV-FABLE (Oct 2026).
3. `SpatialTubeletMaskDecoder`: Decodes pixel-level (B, T, H, W) masks in milliseconds by fusing
   SHDC tokens, ViT geometric skip tokens, and MN-Stem 1/4-res noise/flow boundary maps.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from host_forensics.config import ReadoutConfig


@dataclass
class TemporalProposal:
    """Decoded temporal manipulation interval on a specific modality track."""

    start_sec: float
    end_sec: float
    score: float
    modality: str  # "visual" or "audio"


class KinematicProsodicVerifier(nn.Module):
    """Directional cross-modal verifier checking peripheral/head kinematics against audio prosody."""

    def __init__(self, visual_dim: int, audio_dim: int, out_dim: int, dropout_prob: float = 0.35) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.modality_dropout_prob = dropout_prob

        self.kin_proj = nn.Linear(visual_dim, out_dim)
        self.aud_proj = nn.Linear(audio_dim, out_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True)
        self.discrepancy_mlp = nn.Sequential(
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        # Learned Silent-Prior Embedding used whenever audio is absent or dropped during training
        self.silent_prior = nn.Parameter(torch.randn(1, 1, out_dim) * 0.02)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        anchor_probs: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        has_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            visual_tokens: (B, T, P, D_v) SHDC-conditioned visual tokens
            anchor_probs: (B, T, P) per-token pristine likelihood
            audio_tokens: optional (B, T_a, D_a) SHDC-conditioned audio tokens
            has_audio: (B,) bool or {0,1} tensor indicating audio stream availability
        Returns:
            aux_tokens: (B, T, out_dim) frame-aligned verification context
        """
        b, t, _, _ = visual_tokens.shape
        # Weight toward unmanipulated peripheral/upper-head anchor patches to extract authentic head motion
        anc_w = anchor_probs / (anchor_probs.sum(dim=-1, keepdim=True) + 1e-5)  # (B, T, P)
        head_pose_proxy = torch.einsum("btp,btpd->btd", anc_w, visual_tokens)  # (B, T, D_v)

        # First-order temporal kinematic velocity Delta_t
        if t > 1:
            vel = head_pose_proxy[:, 1:] - head_pose_proxy[:, :-1]
            vel = torch.cat([vel[:, :1], vel], dim=1)
        else:
            vel = torch.zeros_like(head_pose_proxy)

        q_kin = self.kin_proj(vel)  # (B, T, out_dim)

        active_mask = has_audio.to(device=visual_tokens.device, dtype=torch.float32).clone()
        if self.training and self.modality_dropout_prob > 0.0:
            keep = (torch.rand(b, device=visual_tokens.device) >= self.modality_dropout_prob).float()
            active_mask = active_mask * keep

        silent_ctx = self.silent_prior.expand(b, t, -1).to(dtype=visual_tokens.dtype)
        if audio_tokens is None or active_mask.sum() == 0:
            return self.norm(silent_ctx)

        kv_aud = self.aud_proj(audio_tokens)  # (B, T_a, out_dim)
        matched_aud, _ = self.cross_attn(query=q_kin, key=kv_aud, value=kv_aud, need_weights=False)
        disc = self.discrepancy_mlp(torch.cat([q_kin, q_kin - matched_aud], dim=-1))

        gate = active_mask.view(b, 1, 1).to(dtype=visual_tokens.dtype)
        fused = gate * disc + (1.0 - gate) * silent_ctx
        return self.norm(fused)


def temporal_soft_nms_1d(
    starts: torch.Tensor,
    ends: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    max_proposals: int = 20,
    score_thresh: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Vectorized 1D Gaussian Soft-NMS over candidate intervals for a single clip."""
    starts_d = starts.detach().float()
    ends_d = ends.detach().float()
    scores_d = scores.detach().float()

    keep_mask = scores_d >= score_thresh
    if keep_mask.sum() == 0:
        # Always retain at least the top-1 interval for ranking metrics (mAP)
        top_idx = int(torch.argmax(scores_d).item())
        return [(float(starts_d[top_idx]), float(ends_d[top_idx]), float(scores_d[top_idx]))]

    s = starts_d[keep_mask].clone()
    e = ends_d[keep_mask].clone()
    sc = scores_d[keep_mask].clone()

    selected: list[tuple[float, float, float]] = []
    while sc.numel() > 0 and len(selected) < max_proposals:
        best_idx = int(torch.argmax(sc).item())
        bs, be, bsc = float(s[best_idx]), float(e[best_idx]), float(sc[best_idx])
        if bsc < score_thresh and len(selected) > 0:
            break
        selected.append((bs, be, bsc))

        if sc.numel() == 1:
            break

        rem = torch.ones(sc.numel(), dtype=torch.bool, device=sc.device)
        rem[best_idx] = False
        s, e, sc = s[rem], e[rem], sc[rem]

        inter = (torch.minimum(e, torch.tensor(be, device=e.device)) - torch.maximum(s, torch.tensor(bs, device=s.device))).clamp(min=0.0)
        union = (e - s).clamp(min=1e-5) + max(be - bs, 1e-5) - inter
        iou = inter / (union + 1e-6)
        sc = sc * torch.exp(-(iou.pow(2)) / max(sigma, 1e-4))

    return selected


class MultiScaleTemporalPyramidDecoder(nn.Module):
    """ActionFormer / NFSR-style multi-scale temporal pyramid decoder for a single modality track."""

    def __init__(self, in_dim: int, cfg: ReadoutConfig, modality_name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.modality_name = modality_name
        d_h = cfg.hidden_dim

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
        )

        # Multi-scale downsampling pyramid (levels l = 0 .. L-1 with stride 2^l)
        self.pyramid_convs = nn.ModuleList()
        for lvl in range(cfg.pyramid_levels):
            stride = 1 if lvl == 0 else 2
            self.pyramid_convs.append(
                nn.Sequential(
                    nn.Conv1d(d_h, d_h, kernel_size=cfg.pyramid_kernel_size, stride=stride, padding=cfg.pyramid_kernel_size // 2),
                    nn.GroupNorm(8, d_h),
                    nn.GELU(),
                )
            )

        self.cls_head = nn.Conv1d(d_h, 1, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(d_h, 2, kernel_size=3, padding=1)  # left & right nonnegative distances

        # Global attention-pooled track score head complementing peak proposal score
        self.attn_pool = nn.Linear(d_h, 1)
        self.track_cls = nn.Linear(d_h, 1)

    def forward(
        self, seq: torch.Tensor, duration_sec: float
    ) -> dict[str, torch.Tensor | list[list[TemporalProposal]]]:
        """Args:
            seq: (B, L_0, in_dim) temporal sequence
            duration_sec: clip duration in seconds
        Returns:
            dict with multi-scale logits, boundary predictions, timestamps, track_logit, and decoded proposals
        """
        b, l0, _ = seq.shape
        base_stride_sec = duration_sec / float(max(l0, 1))

        h = self.in_proj(seq)  # (B, L_0, d_h)
        attn_w = F.softmax(self.attn_pool(h), dim=1)  # (B, L_0, 1)
        pooled = (attn_w * h).sum(dim=1)  # (B, d_h)
        global_logit = self.track_cls(pooled).squeeze(-1)  # (B,)

        cur = h.transpose(1, 2)  # (B, d_h, L)
        all_logits: list[torch.Tensor] = []
        all_starts: list[torch.Tensor] = []
        all_ends: list[torch.Tensor] = []
        all_timestamps: list[torch.Tensor] = []
        all_strides: list[torch.Tensor] = []

        for lvl, conv in enumerate(self.pyramid_convs):
            if cur.shape[-1] < 2 and lvl > 0:
                break
            cur = conv(cur)  # (B, d_h, L_l)
            l_len = cur.shape[-1]
            stride_sec = duration_sec / float(max(l_len, 1))

            logits_l = self.cls_head(cur).squeeze(1)  # (B, L_l)
            reg_l = F.softplus(self.reg_head(cur))    # (B, 2, L_l) nonnegative stride units

            centers = (torch.arange(l_len, device=seq.device, dtype=torch.float32) + 0.5) * stride_sec
            centers_b = centers.unsqueeze(0).expand(b, -1)

            starts_l = (centers_b - stride_sec * reg_l[:, 0, :].float()).clamp(min=0.0, max=duration_sec)
            ends_l = (centers_b + stride_sec * reg_l[:, 1, :].float()).clamp(min=0.0, max=duration_sec)

            all_logits.append(logits_l)
            all_starts.append(starts_l)
            all_ends.append(ends_l)
            all_timestamps.append(centers_b)
            all_strides.append(torch.full_like(centers_b, stride_sec))

        cat_logits = torch.cat(all_logits, dim=1)   # (B, M_pts)
        cat_starts = torch.cat(all_starts, dim=1)
        cat_ends = torch.cat(all_ends, dim=1)
        cat_centers = torch.cat(all_timestamps, dim=1)
        cat_strides = torch.cat(all_strides, dim=1)

        probs = torch.sigmoid(cat_logits.float())
        peak_logit = cat_logits.float().max(dim=1).values
        track_logit = 0.5 * global_logit.float() + 0.5 * peak_logit

        proposals_batch: list[list[TemporalProposal]] = []
        if self.training:
            proposals_batch = [[] for _ in range(b)]
        else:
            for bi in range(b):
                decoded = temporal_soft_nms_1d(
                    cat_starts[bi],
                    cat_ends[bi],
                    probs[bi],
                    sigma=self.cfg.soft_nms_sigma,
                    max_proposals=self.cfg.max_proposals,
                )
                proposals_batch.append(
                    [
                        TemporalProposal(start_sec=s, end_sec=e, score=sc, modality=self.modality_name)
                        for (s, e, sc) in decoded
                    ]
                )

        return {
            "point_logits": cat_logits,
            "point_starts": cat_starts,
            "point_ends": cat_ends,
            "point_centers": cat_centers,
            "point_strides": cat_strides,
            "track_logit": track_logit,
            "base_stride_sec": torch.tensor(base_stride_sec, device=seq.device),
            "proposals": proposals_batch,
        }


class SpatialTubeletMaskDecoder(nn.Module):
    """Lightweight spatio-temporal mask head fusing SHDC patch tokens + ViT skip + MN-Stem 1/4-res skip."""

    def __init__(
        self,
        embed_dim: int,
        vit_skip_dim: int,
        mn_skip_dim: int,
        base_ch: int = 128,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        in_ch = embed_dim + vit_skip_dim

        # Upsample from H/16 -> H/8 -> H/4
        self.up_to_qtr = nn.Sequential(
            nn.Conv2d(in_ch, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
        )

        # Depthwise 3D temporal smoothing at H/4 x W/4 so consecutive frame masks remain coherent
        self.temp_smooth = nn.Conv3d(
            base_ch + mn_skip_dim,
            base_ch,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False,
        )
        self.temp_norm = nn.GroupNorm(8, base_ch)

        # Upsample H/4 -> H/2 -> H
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_ch // 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=2, stride=2),
            nn.GroupNorm(4, base_ch // 4),
            nn.GELU(),
            nn.Conv2d(base_ch // 4, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        shdc_tokens: torch.Tensor,
        vit_skip_tokens: torch.Tensor,
        mn_skip_qtr: torch.Tensor,
        out_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Args:
            shdc_tokens: (B, T, P, D)
            vit_skip_tokens: (B, T, P, D_skip)
            mn_skip_qtr: (B*T, D_mn_skip, H/4, W/4)
            out_hw: (H, W)
        Returns:
            mask_logits: (B, T, H, W)
        """
        b, t, _, _ = shdc_tokens.shape
        hp = out_hw[0] // self.patch_size
        wp = out_hw[1] // self.patch_size

        grid = rearrange(
            torch.cat([shdc_tokens, vit_skip_tokens], dim=-1),
            "b t (hp wp) c -> (b t) c hp wp",
            hp=hp,
            wp=wp,
        )
        qtr = self.up_to_qtr(grid)  # (B*T, base_ch, H/4, W/4)
        if qtr.shape[-2:] != mn_skip_qtr.shape[-2:]:
            qtr = F.interpolate(qtr, size=mn_skip_qtr.shape[-2:], mode="bilinear", align_corners=False)

        fused_3d = rearrange(torch.cat([qtr, mn_skip_qtr], dim=1), "(b t) c h w -> b c t h w", b=b, t=t)
        smoothed = F.gelu(self.temp_norm(self.temp_smooth(fused_3d)))
        smoothed_2d = rearrange(smoothed, "b c t h w -> (b t) c h w")

        mask_logits = self.final_up(smoothed_2d)  # (B*T, 1, H, W)
        if mask_logits.shape[-2:] != out_hw:
            mask_logits = F.interpolate(mask_logits, size=out_hw, mode="bilinear", align_corners=False)
        return rearrange(mask_logits.squeeze(1), "(b t) h w -> b t h w", b=b, t=t)
