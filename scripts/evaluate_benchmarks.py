#!/usr/bin/env python3
"""Zero-Shot Multi-Domain Benchmark Evaluator & LaTeX Booktabs Exporter.

Evaluates a single calibrated HoST-MoE checkpoint across one or more unseen benchmark manifests
WITHOUT per-dataset threshold tuning, reporting:
  - Per-benchmark `AUROC` and `EER`
  - Multi-domain `Cross-AUC` & `Polarization W1` (When AUC Misleads, June 2026)
  - Fixed-threshold `MID-FFD` Accuracy, F1, and Realized FAR (CVPR 2026)
  - `AV-FABLE` Native-Track `V-mAP`, `A-mAP`, `AV-mAP`, and `Event-IoU` (Oct 2026)
  - `ManiVidBench` Spatial Grounding `mIoU` & `J&F` (Sept 2026)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from host_forensics.config import HoSTConfig
from host_forensics.data.datasets import UnifiedForensicManifestDataset, build_optimized_dataloader
from host_forensics.engine.trainer import configure_cuda_hardware_acceleration
from host_forensics.metrics.evaluation import (
    compute_auroc,
    compute_eer,
    compute_mid_ffd_metrics,
    compute_native_track_av_map,
    compute_polarization_cross_auc,
    compute_spatial_grounding_metrics,
)
from host_forensics.models.heads import TemporalProposal
from host_forensics.models.host_model import HoSTForensicsModel


@torch.inference_mode()
def evaluate_single_manifest(
    model: HoSTForensicsModel,
    manifest_path: str | Path,
    cfg: HoSTConfig,
    device: torch.device,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    ds = UnifiedForensicManifestDataset(
        manifest_path=manifest_path,
        num_frames=cfg.train.num_frames,
        img_size=cfg.backbone.img_size,
        sample_rate=cfg.backbone.sample_rate,
        clip_duration_sec=cfg.train.clip_duration_sec,
    )
    loader = build_optimized_dataloader(
        dataset=ds,
        batch_size=batch_size or cfg.train.eval_batch_size,
        shuffle=False,
        device=device,
        num_workers=cfg.train.num_workers if num_workers is None else num_workers,
        pin_memory=cfg.train.pin_memory,
        prefetch_factor=cfg.train.prefetch_factor,
    )

    all_labels: list[float] = []
    all_probs: list[float] = []
    vis_preds_all: list[list[tuple[float, float, float]]] = []
    aud_preds_all: list[list[tuple[float, float, float]]] = []
    vis_gts_all: list[list[tuple[float, float]]] = []
    aud_gts_all: list[list[tuple[float, float]]] = []
    has_tfl = False

    pred_masks_list: list[np.ndarray] = []
    gt_masks_list: list[np.ndarray] = []

    model.eval()
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        has_audio = batch["has_audio"].to(device, non_blocking=True)
        targets = batch["targets"]

        with model._inference_amp_context(device):  # type: ignore[attr-defined]
            out = model(video=video, audio=audio, has_audio=has_audio, duration_sec=batch["duration_sec"])
        probs = (
            out.calibrated_prob.detach().float().cpu().tolist()
            if out.calibrated_prob is not None
            else torch.sigmoid(out.video_logit.float()).detach().cpu().tolist()
        )
        all_labels.extend(targets.video_label.tolist())
        all_probs.extend(probs)

        if targets.has_visual_tfl_gt is not None and bool(targets.has_visual_tfl_gt.any().item()):
            has_tfl = True
            v_props: list[list[TemporalProposal]] = out.visual_pyramid["proposals"]  # type: ignore[assignment]
            a_props: list[list[TemporalProposal]] | None = (
                out.audio_pyramid["proposals"] if out.audio_pyramid is not None else None  # type: ignore[assignment]
            )
            for i in range(video.shape[0]):
                vis_preds_all.append([(p.start_sec, p.end_sec, p.score) for p in v_props[i]])
                if a_props is not None and bool(has_audio[i].item()):
                    aud_preds_all.append([(p.start_sec, p.end_sec, p.score) for p in a_props[i]])
                else:
                    aud_preds_all.append([])
                vis_gts_all.append(targets.visual_intervals[i] if targets.visual_intervals else [])
                aud_gts_all.append(targets.audio_intervals[i] if targets.audio_intervals else [])

        if targets.has_mask_gt is not None and targets.gt_masks is not None:
            valid_m = targets.has_mask_gt.bool()
            if bool(valid_m.any().item()):
                pm = torch.sigmoid(out.mask_logits[valid_m].float()).detach().cpu().numpy()
                gm = targets.gt_masks[valid_m].detach().cpu().numpy()
                pred_masks_list.append(pm)
                gt_masks_list.append(gm)

    res: dict[str, Any] = {
        "num_clips": len(all_labels),
        "auroc": round(100.0 * compute_auroc(all_labels, all_probs), 2),
        "eer": round(100.0 * compute_eer(all_labels, all_probs), 2),
        "labels": all_labels,
        "probs": all_probs,
    }
    mid = compute_mid_ffd_metrics(all_labels, all_probs, threshold=0.5)
    res["mid_acc"] = round(100.0 * mid["mid_acc"], 2)
    res["realized_far"] = round(100.0 * mid["realized_far"], 2)

    if has_tfl:
        tfl = compute_native_track_av_map(vis_preds_all, aud_preds_all, vis_gts_all, aud_gts_all)
        res["tfl"] = {k: round(100.0 * v, 2) for k, v in tfl.items()}

    if pred_masks_list:
        seg = compute_spatial_grounding_metrics(
            np.concatenate(pred_masks_list, axis=0),
            np.concatenate(gt_masks_list, axis=0),
        )
        res["spatial"] = {k: round(v, 4) for k, v in seg.items()}

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot multi-benchmark evaluation & LaTeX table exporter")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to calibrated checkpoint (.pt)")
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="Named manifest pair `Name=/path/to/manifest.jsonl` (repeatable)",
    )
    parser.add_argument("--output-json", type=str, default="eval_results.json")
    parser.add_argument("--latex-out", type=str, default=None, help="Optional destination .tex row snippet")
    parser.add_argument("--method-name", type=str, default="HoST-MoE (Ours)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on forensic submodules")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = HoSTConfig.from_yaml(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_cuda_hardware_acceleration(device)

    model = HoSTForensicsModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    if args.compile or cfg.train.compile_model:
        model.compile_forensic_submodules()

    per_bench: dict[str, dict[str, Any]] = {}
    domain_pairs: dict[str, tuple[list[float], list[float]]] = {}
    pooled_labels: list[float] = []
    pooled_probs: list[float] = []

    for spec in args.benchmark:
        if "=" not in spec:
            raise ValueError(f"Expected Name=/path/to/manifest.jsonl, got {spec}")
        b_name, m_path = spec.split("=", 1)
        b_res = evaluate_single_manifest(
            model,
            m_path,
            cfg,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        domain_pairs[b_name] = (b_res.pop("labels"), b_res.pop("probs"))
        pooled_labels.extend(domain_pairs[b_name][0])
        pooled_probs.extend(domain_pairs[b_name][1])
        per_bench[b_name] = b_res

    cross = compute_polarization_cross_auc(domain_pairs)
    pooled_mid = compute_mid_ffd_metrics(pooled_labels, pooled_probs, threshold=0.5)

    summary = {
        "method": args.method_name,
        "benchmarks": per_bench,
        "cross_domain_calibration": {
            "mean_within_auc": round(100.0 * cross["mean_within_auc"], 2),
            "pooled_mixed_auc": round(100.0 * cross["pooled_mixed_auc"], 2),
            "polarization_w1": round(cross["polarization_w1"], 4),
            "cross_auc": round(100.0 * cross["cross_auc"], 2),
            "pooled_mid_acc": round(100.0 * pooled_mid["mid_acc"], 2),
            "pooled_realized_far": round(100.0 * pooled_mid["realized_far"], 2),
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.latex_out:
        Path(args.latex_out).parent.mkdir(parents=True, exist_ok=True)
        auc_cells = " & ".join(f"{b['auroc']:.2f}" for b in per_bench.values())
        c_auc = summary["cross_domain_calibration"]["cross_auc"]
        m_acc = summary["cross_domain_calibration"]["pooled_mid_acc"]
        r_far = summary["cross_domain_calibration"]["pooled_realized_far"]
        tex_line = f"\\textbf{{{args.method_name}}} & {auc_cells} & \\textbf{{{c_auc:.2f}}} & \\textbf{{{m_acc:.2f}}} & {r_far:.2f}\\% \\\\\n"
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write(tex_line)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
