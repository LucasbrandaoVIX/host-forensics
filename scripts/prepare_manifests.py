#!/usr/bin/env python3
"""Multi-Benchmark Raw Video -> Unified JSONL Manifest & Tensor Preprocessor.

Supports:
  - `generic`: `<root>/real/*.mp4` and `<root>/fake/*.mp4` (+ optional `<root>/masks/<stem>.*`)
  - `df40`: DF40 / DeepfakeBench folder tree (`real` vs generator subdirectories)
  - `manivid`: ManiVid-38K / ManiVidBench (`videos/`, `masks/`, `metadata.json` with task/intervals)
  - `av_deepfake1m`: AV-Deepfake1M / 1M++ (`metadata.json` with `fake_segments` / `visual_fake_segments` / `audio_fake_segments`)
  - `av_fable`: AV-FABLE pristine twins + native visual/audio interval annotations (`annotations.jsonl` or `.json`)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any
import torch

from host_forensics.data.video_io import extract_mask_sequence_ffmpeg, extract_video_and_audio_ffmpeg


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def _discover_raw_items(dataset_type: str, root: Path, domain_prefix: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if dataset_type in {"generic", "df40"}:
        for vid_path in sorted(root.rglob("*")):
            if vid_path.suffix.lower() not in VIDEO_EXTS or "masks" in vid_path.parts:
                continue
            parts_lower = [p.lower() for p in vid_path.relative_to(root).parts]
            is_real = any(tag in parts_lower for tag in ("real", "original", "youtube-real", "celeb-real", "pristine"))
            sub_domain = parts_lower[0] if len(parts_lower) > 1 else "default"
            mask_candidates = list((root / "masks").glob(f"{vid_path.stem}.*")) if (root / "masks").exists() else []
            items.append(
                {
                    "video_id": f"{domain_prefix}_{vid_path.stem}",
                    "domain": f"{domain_prefix}_{sub_domain}",
                    "video_path": str(vid_path),
                    "mask_path": str(mask_candidates[0]) if mask_candidates else "",
                    "video_label": 0 if is_real else 1,
                    "visual_label": 0 if is_real else 1,
                    "audio_label": 0,
                    "visual_intervals": None,
                    "audio_intervals": None,
                }
            )
        return items

    meta_json = root / "metadata.json"
    if not meta_json.exists():
        raise FileNotFoundError(f"Expected {meta_json} for dataset_type={dataset_type}")
    with open(meta_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    for idx, rec in enumerate(records):
        vid_rel = rec["video_path"]
        vid_full = root / vid_rel
        mask_rel = rec.get("mask_path", "")
        mask_full = str(root / mask_rel) if mask_rel else ""

        v_iv = rec.get("visual_intervals", rec.get("visual_fake_segments"))
        a_iv = rec.get("audio_intervals", rec.get("audio_fake_segments"))
        v_lbl = int(rec.get("visual_label", 1 if v_iv else 0))
        a_lbl = int(rec.get("audio_label", 1 if a_iv else 0))
        vid_lbl = int(rec.get("video_label", max(v_lbl, a_lbl)))

        items.append(
            {
                "video_id": str(rec.get("video_id", f"{domain_prefix}_{idx:06d}")),
                "domain": str(rec.get("domain", domain_prefix)),
                "video_path": str(vid_full),
                "mask_path": mask_full,
                "video_label": vid_lbl,
                "visual_label": v_lbl,
                "audio_label": a_lbl,
                "visual_intervals": v_iv,
                "audio_intervals": a_iv,
            }
        )
    return items


def _process_single_item(task: tuple[dict[str, Any], str, int, int, int, float]) -> dict[str, Any]:
    item, out_tensor_dir, num_frames, img_size, sample_rate, max_dur = task
    out_dir = Path(out_tensor_dir)
    vid_id = item["video_id"]

    decoded = extract_video_and_audio_ffmpeg(
        video_path=item["video_path"],
        num_frames=num_frames,
        img_size=img_size,
        sample_rate=sample_rate,
        max_duration_sec=max_dur,
    )

    v_pt = out_dir / f"{vid_id}_rgb.pt"
    torch.save(decoded["video"], v_pt)

    a_pt_str = ""
    if decoded["has_audio"]:
        a_pt = out_dir / f"{vid_id}_aud.pt"
        torch.save(decoded["audio"], a_pt)
        a_pt_str = str(a_pt)

    m_pt_str = ""
    if item["mask_path"] and Path(item["mask_path"]).exists():
        mask = extract_mask_sequence_ffmpeg(
            mask_path=item["mask_path"],
            num_frames=num_frames,
            img_size=img_size,
            clip_duration_sec=decoded["duration_sec"],
        )
        m_pt = out_dir / f"{vid_id}_mask.pt"
        torch.save(mask, m_pt)
        m_pt_str = str(m_pt)
    elif item["video_label"] == 0:
        # Pristine clip has exact all-zero spatial manipulation mask
        mask = torch.zeros(num_frames, img_size, img_size, dtype=torch.float32)
        m_pt = out_dir / f"{vid_id}_mask.pt"
        torch.save(mask, m_pt)
        m_pt_str = str(m_pt)

    manifest_rec: dict[str, Any] = {
        "video_id": vid_id,
        "domain": item["domain"],
        "video_tensor_pt": str(v_pt),
        "audio_tensor_pt": a_pt_str,
        "has_audio": bool(decoded["has_audio"]),
        "duration_sec": round(float(decoded["duration_sec"]), 4),
        "video_label": int(item["video_label"]),
        "visual_label": int(item["visual_label"]),
        "audio_label": int(item["audio_label"]),
        "mask_tensor_pt": m_pt_str,
    }
    if item["visual_intervals"] is not None:
        manifest_rec["visual_intervals"] = item["visual_intervals"]
    if item["audio_intervals"] is not None:
        manifest_rec["audio_intervals"] = item["audio_intervals"]
    return manifest_rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HoST-MoE JSONL manifests from raw video datasets")
    parser.add_argument("--dataset-type", choices=["generic", "df40", "manivid", "av_deepfake1m", "av_fable"], default="generic")
    parser.add_argument("--input-root", type=str, required=True, help="Root folder of raw videos")
    parser.add_argument("--output-manifest", type=str, required=True, help="Output .jsonl manifest path")
    parser.add_argument("--tensor-dir", type=str, required=True, help="Directory to store extracted .pt tensors")
    parser.add_argument("--domain-prefix", type=str, default="bench")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.input_root)
    tensor_dir = Path(args.tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    raw_items = _discover_raw_items(args.dataset_type, root, args.domain_prefix)
    tasks = [
        (it, str(tensor_dir), args.num_frames, args.img_size, args.sample_rate, args.max_duration_sec)
        for it in raw_items
    ]

    records: list[dict[str, Any]] = []
    if args.workers <= 1:
        for t in tasks:
            records.append(_process_single_item(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(_process_single_item, tasks):
                records.append(rec)

    with open(out_manifest, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(json.dumps({"processed_clips": len(records), "manifest": str(out_manifest)}, indent=2))


if __name__ == "__main__":
    main()
