"""Heterogeneous Multi-Task Forensic Objective for HoST-MoE.

Supports mixed training batches across datasets with partial/missing annotations:
  - Binary video/track labels only (e.g., DF40, Celeb-DF++, TalkingHeadBench)
  - Pixel manipulation masks (e.g., FF++, ManiVid-38K, VideoSham)
  - Modality-specific temporal intervals [start_sec, end_sec] (e.g., AV-Deepfake1M++, AV-FABLE, LAV-DF)
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from host_forensics.config import LossConfig
from host_forensics.models.host_model import HoSTOutput


@dataclass
class BatchTargets:
    """Ground-truth container with explicit task-availability masks."""

    video_label: torch.Tensor                                  # (B,) float {0.0=real, 1.0=manipulated}
    visual_label: torch.Tensor | None = None                   # (B,) float {0, 1}; defaults to video_label if silent
    audio_label: torch.Tensor | None = None                    # (B,) float {0, 1}
    gt_masks: torch.Tensor | None = None                       # (B, T, H, W) float in [0, 1]
    has_mask_gt: torch.Tensor | None = None                    # (B,) bool/float mask availability
    visual_intervals: list[list[tuple[float, float]]] | None = None  # per-sample list of (start_s, end_s)
    audio_intervals: list[list[tuple[float, float]]] | None = None   # per-sample list of (start_s, end_s)
    has_visual_tfl_gt: torch.Tensor | None = None              # (B,) bool/float
    has_audio_tfl_gt: torch.Tensor | None = None               # (B,) bool/float


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Element-wise sigmoid focal loss."""
    logits_f = logits.float()
    targets_f = targets.float()
    probs = torch.sigmoid(logits_f)
    ce = F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none")
    p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)
    modulating = (1.0 - p_t).pow(gamma)
    alpha_t = alpha * targets_f + (1.0 - alpha) * (1.0 - targets_f)
    return alpha_t * modulating * ce


def diou_loss_1d(
    pred_starts: torch.Tensor,
    pred_ends: torch.Tensor,
    gt_starts: torch.Tensor,
    gt_ends: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """1D Distance-IoU loss for temporal interval boundary regression."""
    ps, pe = pred_starts.float(), pred_ends.float()
    gs, ge = gt_starts.float(), gt_ends.float()

    inter = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp(min=0.0)
    pred_len = (pe - ps).clamp(min=eps)
    gt_len = (ge - gs).clamp(min=eps)
    union = pred_len + gt_len - inter + eps
    iou = inter / union

    pred_center = 0.5 * (ps + pe)
    gt_center = 0.5 * (gs + ge)
    enclose_len = (torch.maximum(pe, ge) - torch.minimum(ps, gs)).clamp(min=eps)
    center_penalty = (pred_center - gt_center).pow(2) / enclose_len.pow(2)
    return 1.0 - iou + center_penalty


class HeterogeneousForensicLoss(nn.Module):
    """Unified multi-task loss handling video labels, host de-polarization, anchors, intervals, and masks."""

    def __init__(self, cfg: LossConfig | None = None, patch_size: int = 16) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.patch_size = patch_size

        lap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", lap, persistent=False)

    def _compute_polarization_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Penalize logit dispersion across authentic host videos from mixed domains in the batch."""
        real_mask = labels < 0.5
        if real_mask.sum() >= 2:
            real_logits = logits.float()[real_mask]
            return real_logits.var(unbiased=False)
        return logits.new_zeros((), dtype=torch.float32)

    def _compute_anchor_loss(
        self,
        anchor_probs: torch.Tensor,
        video_labels: torch.Tensor,
        gt_masks: torch.Tensor | None,
        has_mask_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise SHDC Authenticity Anchor Gate via dense inverse masks or weak MIL margin."""
        # anchor_probs: (B, T, P) in (0, 1)
        b, t, p = anchor_probs.shape
        anc_f = anchor_probs.float().clamp(1e-5, 1.0 - 1e-5)
        losses: list[torch.Tensor] = []

        if gt_masks is not None:
            hp = gt_masks.shape[-2] // self.patch_size
            wp = gt_masks.shape[-1] // self.patch_size
            gt_flat = rearrange(gt_masks.float(), "b t h w -> (b t) 1 h w")
            gt_patch = F.adaptive_avg_pool2d(gt_flat, output_size=(hp, wp))
            gt_pristine = 1.0 - rearrange(gt_patch, "(b t) 1 hp wp -> b t (hp wp)", b=b, t=t)
        else:
            gt_pristine = None

        for i in range(b):
            if gt_pristine is not None and bool(has_mask_gt[i].item()):
                losses.append(F.binary_cross_entropy(anc_f[i], gt_pristine[i]))
            else:
                flat_i = anc_f[i].reshape(-1)
                if video_labels[i].item() < 0.5:
                    # Pristine video: almost all spatial-temporal tokens are authentic host
                    losses.append(F.binary_cross_entropy(flat_i, torch.ones_like(flat_i)))
                else:
                    # Manipulated video without mask: MIL Top-k (host background) vs Bottom-k (forged RoI)
                    n_tok = flat_i.numel()
                    k_host = max(1, int(round(n_tok * self.cfg.mil_top_host_ratio)))
                    k_fake = max(1, int(round(n_tok * self.cfg.mil_bottom_fake_ratio)))
                    top_host = torch.topk(flat_i, k=k_host, largest=True).values.mean()
                    bot_fake = torch.topk(flat_i, k=k_fake, largest=False).values.mean()

                    host_keep = F.relu(0.85 - top_host)
                    sep_margin = F.relu(self.cfg.mil_margin - (top_host - bot_fake))
                    losses.append(host_keep + sep_margin)

        return torch.stack(losses).mean()

    def _compute_pyramid_tfl_loss(
        self,
        pyramid_out: dict[str, object],
        gt_intervals: list[list[tuple[float, float]]],
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Focal + 1D-DIoU loss for a single modality's temporal pyramid."""
        logits: torch.Tensor = pyramid_out["point_logits"]   # type: ignore[assignment]
        p_starts: torch.Tensor = pyramid_out["point_starts"] # type: ignore[assignment]
        p_ends: torch.Tensor = pyramid_out["point_ends"]     # type: ignore[assignment]
        centers: torch.Tensor = pyramid_out["point_centers"] # type: ignore[assignment]

        focal_list: list[torch.Tensor] = []
        diou_list: list[torch.Tensor] = []

        for bi in range(logits.shape[0]):
            if not bool(active_mask[bi].item()):
                continue

            c_i = centers[bi]  # (M_pts,)
            intervals_i = gt_intervals[bi]
            cls_target = torch.zeros_like(c_i)
            matched_gs = torch.zeros_like(c_i)
            matched_ge = torch.zeros_like(c_i)
            best_len = torch.full_like(c_i, float("inf"))

            for (gs, ge) in intervals_i:
                if ge <= gs:
                    continue
                inside = (c_i >= gs) & (c_i <= ge)
                seg_len = ge - gs
                update = inside & (seg_len < best_len)
                cls_target = torch.where(inside, torch.ones_like(cls_target), cls_target)
                matched_gs = torch.where(update, torch.full_like(matched_gs, gs), matched_gs)
                matched_ge = torch.where(update, torch.full_like(matched_ge, ge), matched_ge)
                best_len = torch.where(update, torch.full_like(best_len, seg_len), best_len)

            focal_i = sigmoid_focal_loss(logits[bi], cls_target)
            pos_mask = cls_target > 0.5
            num_pos = pos_mask.sum().clamp(min=1).float()
            focal_list.append(focal_i.sum() / num_pos)

            if pos_mask.any():
                diou_i = diou_loss_1d(
                    p_starts[bi][pos_mask],
                    p_ends[bi][pos_mask],
                    matched_gs[pos_mask],
                    matched_ge[pos_mask],
                )
                diou_list.append(diou_i.mean())

        zero = logits.new_zeros((), dtype=torch.float32)
        loss_focal = torch.stack(focal_list).mean() if focal_list else zero
        loss_diou = torch.stack(diou_list).mean() if diou_list else zero
        return loss_focal, loss_diou

    def _compute_mask_loss(
        self,
        mask_logits: torch.Tensor,
        gt_masks: torch.Tensor | None,
        has_mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pixel BCE, Soft Dice, and Laplacian Boundary loss on samples with mask supervision."""
        zero = mask_logits.new_zeros((), dtype=torch.float32)
        if gt_masks is None or not bool(has_mask_gt.bool().any().item()):
            return zero, zero, zero

        valid = has_mask_gt.bool()
        pred_v = mask_logits[valid].float()  # (B_v, T, H, W)
        gt_v = gt_masks[valid].float()

        fg_ratio = gt_v.mean().detach().clamp(min=0.02, max=0.50)
        pos_w = ((1.0 - fg_ratio) / fg_ratio).clamp(min=1.0, max=12.0)
        loss_bce = F.binary_cross_entropy_with_logits(pred_v, gt_v, pos_weight=pos_w)

        probs = torch.sigmoid(pred_v)
        probs_flat = probs.reshape(probs.shape[0], -1)
        gt_flat = gt_v.reshape(gt_v.shape[0], -1)
        inter = (probs_flat * gt_flat).sum(dim=1)
        dice = (2.0 * inter + 1.0) / (probs_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1.0)
        loss_dice = (1.0 - dice).mean()

        # 2D Laplacian edge consistency
        p_2d = rearrange(probs, "b t h w -> (b t) 1 h w")
        g_2d = rearrange(gt_v, "b t h w -> (b t) 1 h w")
        p_edge = F.conv2d(p_2d, self.laplacian_kernel, padding=1).abs()
        g_edge = F.conv2d(g_2d, self.laplacian_kernel, padding=1).abs()
        loss_bnd = F.l1_loss(p_edge, g_edge)

        return loss_bce, loss_dice, loss_bnd

    def forward(self, out: HoSTOutput, targets: BatchTargets) -> dict[str, torch.Tensor]:
        device = out.video_logit.device
        b = out.video_logit.shape[0]
        vid_y = targets.video_label.to(device=device, dtype=torch.float32)

        # 1. Video & Track Classification Loss
        loss_cls = F.binary_cross_entropy_with_logits(out.video_logit.float(), vid_y)
        if targets.visual_label is not None:
            vis_y = targets.visual_label.to(device=device, dtype=torch.float32)
            loss_cls = loss_cls + 0.5 * F.binary_cross_entropy_with_logits(
                out.visual_track_logit.float(), vis_y
            )
        if targets.audio_label is not None and out.has_audio.any():
            aud_y = targets.audio_label.to(device=device, dtype=torch.float32)
            aud_mask = out.has_audio.bool()
            loss_cls = loss_cls + 0.5 * F.binary_cross_entropy_with_logits(
                out.audio_track_logit[aud_mask].float(), aud_y[aud_mask]
            )

        # 2. Host-Polarization Regularizer
        loss_pol = self._compute_polarization_loss(out.video_logit, vid_y)

        # 3. SHDC Authenticity Anchor Supervision
        has_mask = (
            targets.has_mask_gt.to(device=device, dtype=torch.bool)
            if targets.has_mask_gt is not None
            else torch.zeros(b, device=device, dtype=torch.bool)
        )
        loss_anc = self._compute_anchor_loss(
            out.visual_shdc.anchor_probs, vid_y, targets.gt_masks, has_mask
        )

        # 4. Modality-Separated Temporal Forgery Localization Losses
        zero = out.video_logit.new_zeros((), dtype=torch.float32)
        if targets.visual_intervals is not None and targets.has_visual_tfl_gt is not None:
            v_focal, v_diou = self._compute_pyramid_tfl_loss(
                out.visual_pyramid,
                targets.visual_intervals,
                targets.has_visual_tfl_gt.to(device=device, dtype=torch.bool),
            )
        else:
            v_focal, v_diou = zero, zero

        if (
            out.audio_pyramid is not None
            and targets.audio_intervals is not None
            and targets.has_audio_tfl_gt is not None
        ):
            a_active = targets.has_audio_tfl_gt.to(device=device, dtype=torch.bool) & out.has_audio
            a_focal, a_diou = self._compute_pyramid_tfl_loss(
                out.audio_pyramid, targets.audio_intervals, a_active
            )
        else:
            a_focal, a_diou = zero, zero

        loss_tfl = self.cfg.lambda_tfl_focal * (v_focal + a_focal) + self.cfg.lambda_tfl_diou * (
            v_diou + a_diou
        )

        # 5. Spatial Forgery Mask Losses
        m_bce, m_dice, m_bnd = self._compute_mask_loss(out.mask_logits, targets.gt_masks, has_mask)
        loss_mask = (
            self.cfg.lambda_mask_bce * m_bce
            + self.cfg.lambda_mask_dice * m_dice
            + self.cfg.lambda_mask_boundary * m_bnd
        )

        total = (
            self.cfg.lambda_cls * loss_cls
            + self.cfg.lambda_polarization * loss_pol
            + self.cfg.lambda_anchor * loss_anc
            + loss_tfl
            + loss_mask
        )

        return {
            "loss_total": total,
            "loss_cls": loss_cls,
            "loss_polarization": loss_pol,
            "loss_anchor": loss_anc,
            "loss_tfl": loss_tfl,
            "loss_mask": loss_mask,
        }
