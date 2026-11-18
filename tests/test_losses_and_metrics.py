"""Unit tests for Heterogeneous Forensic Loss, Pristine Calibrator, 2026 Metrics, and Trainer."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from host_forensics.cli import _build_compact_smoke_config
from host_forensics.data.datasets import (
    SyntheticManipulatedVideoDataset,
    UnifiedForensicManifestDataset,
    forensic_collate_fn,
)
from host_forensics.engine.trainer import HoSTTrainer
from host_forensics.losses.forensic_losses import BatchTargets, HeterogeneousForensicLoss
from host_forensics.metrics.evaluation import (
    compute_auroc,
    compute_eer,
    compute_mid_ffd_metrics,
    compute_native_track_av_map,
    compute_polarization_cross_auc,
    compute_spatial_grounding_metrics,
)
from host_forensics.models.host_model import HoSTForensicsModel, PristineQuantileCalibrator


def test_heterogeneous_loss_mixed_annotations() -> None:
    torch.manual_seed(10)
    cfg = _build_compact_smoke_config()
    model = HoSTForensicsModel(cfg)
    criterion = HeterogeneousForensicLoss(cfg.loss, patch_size=cfg.backbone.patch_size)

    # Batch of 3 clips:
    #   0: Pristine clip with dense zero mask + intervals
    #   1: Manipulated clip with ONLY video-level label (no mask, no intervals -> triggers MIL anchor loss!)
    #   2: Manipulated clip with dense mask + visual & audio intervals
    video = torch.randn(3, 4, 3, 128, 128)
    audio = torch.randn(3, 32000)
    has_audio = torch.tensor([True, False, True])

    gt_masks = torch.zeros(3, 4, 128, 128)
    gt_masks[2, 1:3, 32:96, 32:96] = 1.0

    targets = BatchTargets(
        video_label=torch.tensor([0.0, 1.0, 1.0]),
        visual_label=torch.tensor([0.0, 1.0, 1.0]),
        audio_label=torch.tensor([0.0, 0.0, 1.0]),
        gt_masks=gt_masks,
        has_mask_gt=torch.tensor([True, False, True]),
        visual_intervals=[[], [], [(0.5, 1.5)]],
        audio_intervals=[[], [], [(0.25, 1.25)]],
        has_visual_tfl_gt=torch.tensor([True, False, True]),
        has_audio_tfl_gt=torch.tensor([True, False, True]),
    )

    out = model(video=video, audio=audio, has_audio=has_audio, duration_sec=2.0)
    losses = criterion(out, targets)

    for key in ("loss_total", "loss_cls", "loss_polarization", "loss_anchor", "loss_tfl", "loss_mask"):
        assert key in losses
        assert torch.isfinite(losses[key])

    losses["loss_total"].backward()
    assert losses["loss_anchor"].item() > 0.0
    assert losses["loss_mask"].item() > 0.0


def test_pristine_quantile_calibrator_far_bound_and_4way_attribution() -> None:
    torch.manual_seed(11)
    cal = PristineQuantileCalibrator(far_budget=0.05)

    pristine_v = torch.randn(1000) * 0.8 - 1.0
    pristine_a = torch.randn(1000) * 0.6 - 0.5
    cal.fit(pristine_v, pristine_a)
    assert bool(cal.is_calibrated.item()) is True

    # Empirical FAR on pristine samples must adhere closely to 5% budget
    _, clean_labels = cal.calibrate_decisions(
        pristine_v, torch.full_like(pristine_v, -10.0), torch.zeros(1000, dtype=torch.bool)
    )
    empirical_far = sum(lbl != "clean" for lbl in clean_labels) / 1000.0
    assert abs(empirical_far - 0.05) <= 0.01

    # Verify 4-way track attribution ("clean", "visual_only", "audio_only", "both_track")
    v_test = torch.tensor([-3.0, 5.0, -3.0, 5.0])
    a_test = torch.tensor([-3.0, -3.0, 5.0, 5.0])
    has_aud = torch.tensor([True, True, True, True])
    probs, attrs = cal.calibrate_decisions(v_test, a_test, has_aud)
    assert attrs == ["clean", "visual_only", "audio_only", "both_track"]
    assert probs[0].item() < 0.5
    assert all(p > 0.8 for p in probs[1:].tolist())


def test_polarization_cross_auc_penalizes_domain_shift() -> None:
    # Calibrated detector across two unseen domains -> high Cross-AUC
    calibrated_domains = {
        "domain_A": ([0, 0, 1, 1], [0.10, 0.15, 0.85, 0.90]),
        "domain_B": ([0, 0, 1, 1], [0.12, 0.18, 0.82, 0.88]),
    }
    res_cal = compute_polarization_cross_auc(calibrated_domains)
    assert res_cal["mean_within_auc"] == 1.0
    assert res_cal["cross_auc"] > 0.95

    # Polarized detector: 1.0 AUC inside each domain individually, but Domain A lives in [0.01..0.08]
    # while Domain B lives in [0.91..0.99] -> catastrophic real-world threshold failure!
    polarized_domains = {
        "domain_A": ([0, 0, 1, 1], [0.01, 0.02, 0.06, 0.08]),
        "domain_B": ([0, 0, 1, 1], [0.91, 0.93, 0.97, 0.99]),
    }
    res_pol = compute_polarization_cross_auc(polarized_domains)
    assert res_pol["mean_within_auc"] == 1.0
    assert res_pol["polarization_w1"] > 0.80
    assert res_pol["cross_auc"] < 0.20

    # Check AUROC, EER, and MID-FFD helpers
    assert compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert compute_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0.0
    mid = compute_mid_ffd_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert mid["mid_acc"] == 1.0 and mid["realized_far"] == 0.0


def test_native_track_av_map_rejects_cross_modality_bleed() -> None:
    # Ground truth has an AUDIO-only splice on [1.0, 2.5], visual stream is pristine
    visual_gts = [[], []]
    audio_gts = [[(1.0, 2.5)], []]

    # Detector A wrongly predicts a visual interval on [1.0, 2.5] and misses audio -> must get 0.0 A-mAP
    bleed_metrics = compute_native_track_av_map(
        visual_preds=[[(1.0, 2.5, 0.95)], []],
        audio_preds=[[], []],
        visual_gts=visual_gts,
        audio_gts=audio_gts,
    )
    assert bleed_metrics["a_map"] == 0.0
    assert bleed_metrics["av_map"] == 0.0

    # Detector B predicts on the native audio track -> achieves 1.0 A-mAP
    exact_metrics = compute_native_track_av_map(
        visual_preds=[[], []],
        audio_preds=[[(1.0, 2.5, 0.95)], []],
        visual_gts=visual_gts,
        audio_gts=audio_gts,
    )
    assert exact_metrics["a_map"] == 1.0
    assert exact_metrics["av_map"] == 1.0


def test_spatial_grounding_and_manifest_trainer_pipeline(tmp_path: Path) -> None:
    gt = np.zeros((2, 3, 32, 32), dtype=np.float32)
    gt[0, :, 8:24, 8:24] = 1.0
    pred_perfect = gt.copy()
    seg_metrics = compute_spatial_grounding_metrics(pred_perfect, gt)
    assert seg_metrics["miou"] == 1.0
    assert seg_metrics["j_and_f"] == 1.0

    # Build a tiny on-disk JSONL manifest + tensor files to verify UnifiedForensicManifestDataset & HoSTTrainer
    cfg = _build_compact_smoke_config()
    synth = SyntheticManipulatedVideoDataset(
        length=4,
        num_frames=cfg.train.num_frames,
        img_size=cfg.backbone.img_size,
        clip_duration_sec=cfg.train.clip_duration_sec,
    )
    manifest_file = tmp_path / "mini_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for i in range(len(synth)):
            item = synth[i]
            v_pt = tmp_path / f"vid_{i}.pt"
            a_pt = tmp_path / f"aud_{i}.pt"
            m_pt = tmp_path / f"msk_{i}.pt"
            torch.save(item["video"], v_pt)
            torch.save(item["audio"], a_pt)
            torch.save(item["gt_mask"], m_pt)
            rec = {
                "video_id": f"clip_{i}",
                "domain": item["domain"],
                "video_tensor_pt": str(v_pt),
                "audio_tensor_pt": str(a_pt) if item["has_audio"] else "",
                "has_audio": item["has_audio"],
                "duration_sec": item["duration_sec"],
                "video_label": item["video_label"],
                "visual_label": item["visual_label"],
                "audio_label": item["audio_label"],
                "mask_tensor_pt": str(m_pt),
                "visual_intervals": [list(iv) for iv in item["visual_intervals"]],
                "audio_intervals": [list(iv) for iv in item["audio_intervals"]],
            }
            f.write(json.dumps(rec) + "\n")

    ds = UnifiedForensicManifestDataset(
        manifest_path=manifest_file,
        num_frames=cfg.train.num_frames,
        img_size=cfg.backbone.img_size,
        clip_duration_sec=cfg.train.clip_duration_sec,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=forensic_collate_fn)

    model = HoSTForensicsModel(cfg)
    trainer = HoSTTrainer(model=model, cfg=cfg, device="cpu")
    epoch_stats = trainer.train_one_epoch(loader)
    assert "loss_total" in epoch_stats and np.isfinite(epoch_stats["loss_total"])

    cal_stats = trainer.calibrate_on_pristine(loader)
    assert "visual_thresh" in cal_stats
    assert bool(model.calibrator.is_calibrated.item()) is True
