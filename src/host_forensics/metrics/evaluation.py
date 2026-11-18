"""Reviewer-Proof 2026-2027 Evaluation Metrics for AI-Manipulated Video Forensics.

Includes:
1. Standard `AUROC` & `EER`.
2. `Cross-AUC` with 1D Wasserstein Polarization Penalty (Nguyen et al., June 2026, "When AUC Misleads").
3. `MID-FFD` Fixed-Threshold Domain-Unspecified Accuracy & Realized FAR (Cheng et al., CVPR 2026).
4. `Native-Track AV-mAP@[0.30:0.05:0.70]` & `Event-IoU` (Huang et al., Oct 2026, "AV-FABLE"),
   preserving modality identity so visual predictions never match audio events.
5. `Spatial Grounding mIoU` & `J&F` (Kang et al., Sept 2026, "ManiVidBench").
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import wasserstein_distance


def compute_auroc(labels: Sequence[int | float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Exact rank-based Area Under the ROC Curve (AUROC) in [0, 1]."""
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    pos = y > 0.5
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # Handle ties via average rank
    sorted_s = s[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_eer(labels: Sequence[int | float] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Compute Equal Error Rate (EER) where False Acceptance Rate == False Rejection Rate."""
    y = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    desc = np.argsort(-s)
    y_sorted = y[desc]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    fnr = 1.0 - (tps / float(n_pos))
    fpr = fps / float(n_neg)
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float(0.5 * (fpr[idx] + fnr[idx]))


def compute_polarization_cross_auc(
    domain_data: dict[str, tuple[Sequence[int | float] | np.ndarray, Sequence[float] | np.ndarray]],
) -> dict[str, float]:
    """Compute Polarization-Aware Cross-AUC across multiple unseen domains.

    Quantifies whether a detector achieves high per-dataset AUC yet shifts its score range across
    domains (score polarization), using pairwise 1D Wasserstein distance on [0, 1]-normalized scores.
    """
    if not domain_data:
        raise ValueError("domain_data must contain at least one domain.")

    per_domain_aucs: list[float] = []
    real_dists: list[np.ndarray] = []
    fake_dists: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    pooled_s: list[np.ndarray] = []

    for _, (labels, probs) in domain_data.items():
        y = np.asarray(labels, dtype=np.float64).ravel()
        p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 0.0, 1.0)
        per_domain_aucs.append(compute_auroc(y, p))
        pooled_y.append(y)
        pooled_s.append(p)
        if (y <= 0.5).any():
            real_dists.append(p[y <= 0.5])
        if (y > 0.5).any():
            fake_dists.append(p[y > 0.5])

    mean_within_auc = float(np.mean(per_domain_aucs))
    pooled_auc = compute_auroc(np.concatenate(pooled_y), np.concatenate(pooled_s))

    def _pairwise_w1(dists: list[np.ndarray]) -> float:
        if len(dists) < 2:
            return 0.0
        vals = [
            float(wasserstein_distance(dists[i], dists[j]))
            for i in range(len(dists))
            for j in range(i + 1, len(dists))
        ]
        return float(np.mean(vals))

    w1_real = _pairwise_w1(real_dists)
    w1_fake = _pairwise_w1(fake_dists)
    polarization = 0.5 * (w1_real + w1_fake)
    cross_auc = float(max(0.0, mean_within_auc * (1.0 - polarization)))

    return {
        "mean_within_auc": mean_within_auc,
        "pooled_mixed_auc": pooled_auc,
        "polarization_w1": polarization,
        "cross_auc": cross_auc,
    }


def compute_mid_ffd_metrics(
    labels: Sequence[int | float] | np.ndarray,
    calibrated_probs: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Multi-In-Domain / Unspecified-Domain single-threshold Accuracy, F1, and Realized FAR."""
    y = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    p = np.asarray(calibrated_probs, dtype=np.float64).ravel() >= threshold

    acc = float((y == p).mean()) if y.size > 0 else 0.0
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    n_real = int((~y).sum())

    f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
    realized_far = float(fp / max(n_real, 1))
    return {
        "mid_acc": acc,
        "mid_f1": float(f1),
        "realized_far": realized_far,
    }


def _interval_iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(1e-6, (a[1] - a[0]) + (b[1] - b[0]) - inter)
    return inter / union


def _single_track_ap(
    preds_per_video: list[list[tuple[float, float, float]]],
    gts_per_video: list[list[tuple[float, float]]],
    iou_thresh: float,
) -> float:
    """Compute Average Precision at `iou_thresh` across all videos (including pristine negatives)."""
    total_gt = sum(len(g) for g in gts_per_video)
    if total_gt == 0:
        return 0.0

    flat_preds: list[tuple[float, int, float, float]] = []
    for vid_idx, plist in enumerate(preds_per_video):
        for (s, e, score) in plist:
            flat_preds.append((score, vid_idx, s, e))
    if not flat_preds:
        return 0.0

    flat_preds.sort(key=lambda item: item[0], reverse=True)
    matched = [[False] * len(g) for g in gts_per_video]

    tp = np.zeros(len(flat_preds), dtype=np.float64)
    fp = np.zeros(len(flat_preds), dtype=np.float64)

    for rank_idx, (_, vid_idx, ps, pe) in enumerate(flat_preds):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_seg in enumerate(gts_per_video[vid_idx]):
            iou = _interval_iou_1d((ps, pe), gt_seg)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_thresh and best_gt_idx >= 0 and not matched[vid_idx][best_gt_idx]:
            matched[vid_idx][best_gt_idx] = True
            tp[rank_idx] = 1.0
        else:
            fp[rank_idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    # All-point interpolated AP
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_native_track_av_map(
    visual_preds: list[list[tuple[float, float, float]]],
    audio_preds: list[list[tuple[float, float, float]]],
    visual_gts: list[list[tuple[float, float]]],
    audio_gts: list[list[tuple[float, float]]],
    iou_thresholds: Sequence[float] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
) -> dict[str, float]:
    """AV-FABLE Native-Track Localization Protocol: preserves modality identity across tracks."""
    v_aps = [_single_track_ap(visual_preds, visual_gts, t) for t in iou_thresholds]
    a_aps = [_single_track_ap(audio_preds, audio_gts, t) for t in iou_thresholds]

    v_map = float(np.mean(v_aps))
    a_map = float(np.mean(a_aps))

    has_v_events = sum(len(g) for g in visual_gts) > 0
    has_a_events = sum(len(g) for g in audio_gts) > 0
    active_maps = ([v_map] if has_v_events else []) + ([a_map] if has_a_events else [])
    av_map = float(np.mean(active_maps)) if active_maps else 0.0

    def _event_iou(
        preds_list: list[list[tuple[float, float, float]]],
        gts_list: list[list[tuple[float, float]]],
    ) -> float:
        ious: list[float] = []
        for plist, glist in zip(preds_list, gts_list):
            for gt_seg in glist:
                best = max((_interval_iou_1d((ps, pe), gt_seg) for (ps, pe, _) in plist), default=0.0)
                ious.append(best)
        return float(np.mean(ious)) if ious else 0.0

    return {
        "v_map": v_map,
        "a_map": a_map,
        "av_map": av_map,
        "v_event_iou": _event_iou(visual_preds, visual_gts),
        "a_event_iou": _event_iou(audio_preds, audio_gts),
    }


def compute_spatial_grounding_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute ManiVidBench / DAVIS spatial video grounding metrics: mIoU (J), Contour F, and J&F.

    Args:
        pred_masks: (N, T, H, W) float probabilities in [0, 1]
        gt_masks: (N, T, H, W) binary ground truth in {0, 1}
    """
    pred_bin = np.asarray(pred_masks) >= threshold
    gt_bin = np.asarray(gt_masks) >= 0.5

    j_scores: list[float] = []
    f_scores: list[float] = []

    for n_idx in range(pred_bin.shape[0]):
        p_vid = pred_bin[n_idx]
        g_vid = gt_bin[n_idx]

        inter = np.logical_and(p_vid, g_vid).sum()
        union = np.logical_or(p_vid, g_vid).sum()
        j_val = 1.0 if union == 0 else float(inter / union)
        j_scores.append(j_val)

        # Frame-averaged boundary F-score
        frame_f: list[float] = []
        for t_idx in range(p_vid.shape[0]):
            pf = p_vid[t_idx]
            gf = g_vid[t_idx]
            p_bnd = np.logical_xor(pf, binary_dilation(pf, iterations=1))
            g_bnd = np.logical_xor(gf, binary_dilation(gf, iterations=1))
            if not p_bnd.any() and not g_bnd.any():
                frame_f.append(1.0)
                continue
            if not p_bnd.any() or not g_bnd.any():
                frame_f.append(0.0)
                continue
            p_dil = binary_dilation(p_bnd, iterations=2)
            g_dil = binary_dilation(g_bnd, iterations=2)
            prec = float(np.logical_and(p_bnd, g_dil).sum() / max(p_bnd.sum(), 1))
            rec = float(np.logical_and(g_bnd, p_dil).sum() / max(g_bnd.sum(), 1))
            f_val = (2.0 * prec * rec) / max(prec + rec, 1e-8)
            frame_f.append(f_val)
        f_scores.append(float(np.mean(frame_f)))

    miou = float(np.mean(j_scores))
    f_bound = float(np.mean(f_scores))
    return {
        "miou": miou,
        "boundary_f": f_bound,
        "j_and_f": 0.5 * (miou + f_bound),
    }
