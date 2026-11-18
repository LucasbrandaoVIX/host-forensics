"""Unified JSONL Manifest Dataset & Synthetic Smoke-Test Manipulation Dataset.

Manifest schema (one JSON object per line, compatible with DeepfakeBench/DF40, ManiVid-38K,
AV-Deepfake1M++, and AV-FABLE pristine-twin exports):
```json
{
  "video_id": "sample_0001",
  "domain": "df40_dreamid_v",
  "video_tensor_pt": "/path/to/frames.pt",
  "audio_tensor_pt": "/path/to/audio.pt",
  "has_audio": true,
  "duration_sec": 4.0,
  "video_label": 1,
  "visual_label": 1,
  "audio_label": 0,
  "mask_tensor_pt": "/path/to/mask.pt",
  "visual_intervals": [[0.8, 3.2]],
  "audio_intervals": []
}
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

from host_forensics.data.transforms import CompoundDegradationAndLaundering
from host_forensics.losses.forensic_losses import BatchTargets


class UnifiedForensicManifestDataset(Dataset[dict[str, Any]]):
    """Loads heterogeneous multi-modal manipulation records from a JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 16,
        img_size: int = 384,
        sample_rate: int = 16000,
        clip_duration_sec: float = 4.0,
        augment: CompoundDegradationAndLaundering | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.num_frames = num_frames
        self.img_size = img_size
        self.audio_len = int(round(sample_rate * clip_duration_sec))
        self.clip_duration_sec = clip_duration_sec
        self.augment = augment

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.records: list[dict[str, Any]] = [
                json.loads(line) for line in f if line.strip()
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        video: torch.Tensor = torch.load(rec["video_tensor_pt"], map_location="cpu", weights_only=True)
        if self.augment is not None:
            video = self.augment(video)

        has_audio = bool(rec.get("has_audio", False)) and bool(rec.get("audio_tensor_pt"))
        if has_audio:
            audio = torch.load(rec["audio_tensor_pt"], map_location="cpu", weights_only=True)
        else:
            audio = torch.zeros(self.audio_len, dtype=torch.float32)

        has_mask = bool(rec.get("mask_tensor_pt"))
        if has_mask:
            gt_mask = torch.load(rec["mask_tensor_pt"], map_location="cpu", weights_only=True).float()
        else:
            gt_mask = torch.zeros(video.shape[0], video.shape[-2], video.shape[-1], dtype=torch.float32)

        vis_iv = [tuple(map(float, seg)) for seg in rec.get("visual_intervals", [])]
        aud_iv = [tuple(map(float, seg)) for seg in rec.get("audio_intervals", [])]

        return {
            "video": video.float(),
            "audio": audio.float(),
            "has_audio": has_audio,
            "duration_sec": float(rec.get("duration_sec", self.clip_duration_sec)),
            "video_label": float(rec["video_label"]),
            "visual_label": float(rec.get("visual_label", rec["video_label"])),
            "audio_label": float(rec.get("audio_label", 0.0)),
            "gt_mask": gt_mask,
            "has_mask_gt": has_mask,
            "visual_intervals": vis_iv,
            "audio_intervals": aud_iv,
            "has_visual_tfl_gt": "visual_intervals" in rec,
            "has_audio_tfl_gt": "audio_intervals" in rec,
            "domain": str(rec.get("domain", "default")),
        }


class SyntheticManipulatedVideoDataset(Dataset[dict[str, Any]]):
    """Procedurally generates authentic-host vs. locally manipulated clips for smoke-training & CI.

    Simulates four realistic regimes:
      0: Pristine twin (unmanipulated camera-noise-correlated video + clean harmonic audio)
      1: Localized spatial-temporal splice / face-swap proxy (box region with altered VAE/freq & motion)
      2: Audio-only speech splice (pristine visual stream + localized audio segment replacement)
      3: Silent visual object-removal / inpainting proxy (`has_audio=False`, smoothed flow patch)
    """

    def __init__(
        self,
        length: int = 32,
        num_frames: int = 8,
        img_size: int = 128,
        clip_duration_sec: float = 4.0,
        sample_rate: int = 16000,
        augment: CompoundDegradationAndLaundering | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.length = length
        self.num_frames = num_frames
        self.img_size = img_size
        self.clip_duration_sec = clip_duration_sec
        self.audio_len = int(round(sample_rate * clip_duration_sec))
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        mode = idx % 4

        t, h, w = self.num_frames, self.img_size, self.img_size
        # Host video with consistent camera ISP noise signature + smooth global pan
        isp_pattern = torch.randn(1, 3, h, w, generator=gen) * 0.08
        base_scene = torch.randn(1, 3, h // 8, w // 8, generator=gen)
        base_scene = torch.nn.functional.interpolate(base_scene, size=(h, w), mode="bilinear", align_corners=False)
        video = base_scene.repeat(t, 1, 1, 1) + isp_pattern + torch.randn(t, 3, h, w, generator=gen) * 0.02

        t_axis = torch.linspace(0.0, self.clip_duration_sec, self.audio_len)
        audio = 0.2 * torch.sin(2.0 * 3.14159 * 220.0 * t_axis) + torch.randn(self.audio_len, generator=gen) * 0.02

        gt_mask = torch.zeros(t, h, w, dtype=torch.float32)
        vis_intervals: list[tuple[float, float]] = []
        aud_intervals: list[tuple[float, float]] = []
        has_audio = mode != 3
        has_mask_gt = mode in {0, 1, 3}

        if mode == 0:
            video_label, vis_label, aud_label = 0.0, 0.0, 0.0
        elif mode == 1:
            # Localized spatial-temporal manipulation on frames [t//4 : 3*t//4]
            t0, t1 = max(0, t // 4), max(1, (3 * t) // 4)
            y0, y1 = h // 4, (3 * h) // 4
            x0, x1 = w // 4, (3 * w) // 4
            forged_patch = torch.randn(t1 - t0, 3, y1 - y0, x1 - x0, generator=gen) * 0.35
            video[t0:t1, :, y0:y1, x0:x1] = forged_patch
            gt_mask[t0:t1, y0:y1, x0:x1] = 1.0
            s_sec = (t0 / float(t)) * self.clip_duration_sec
            e_sec = (t1 / float(t)) * self.clip_duration_sec
            vis_intervals.append((s_sec, e_sec))
            video_label, vis_label, aud_label = 1.0, 1.0, 0.0
        elif mode == 2:
            # Audio-only splice on [1.0s, 2.5s]
            a0 = int(0.25 * self.audio_len)
            a1 = int(0.625 * self.audio_len)
            audio[a0:a1] = torch.randn(a1 - a0, generator=gen) * 0.45
            aud_intervals.append((0.25 * self.clip_duration_sec, 0.625 * self.clip_duration_sec))
            video_label, vis_label, aud_label = 1.0, 0.0, 1.0
            has_mask_gt = True  # All-zero visual mask is exact ground truth for audio-only splice!
        else:
            # Mode 3: Silent localized object-removal / over-smoothed flow region
            y0, y1 = h // 3, (2 * h) // 3
            x0, x1 = w // 3, (2 * w) // 3
            smoothed = torch.nn.functional.avg_pool2d(video[:, :, y0:y1, x0:x1], kernel_size=5, stride=1, padding=2)
            video[:, :, y0:y1, x0:x1] = smoothed
            gt_mask[:, y0:y1, x0:x1] = 1.0
            vis_intervals.append((0.0, self.clip_duration_sec))
            audio.zero_()
            video_label, vis_label, aud_label = 1.0, 1.0, 0.0

        if self.augment is not None:
            video = self.augment(video)

        return {
            "video": video,
            "audio": audio,
            "has_audio": has_audio,
            "duration_sec": self.clip_duration_sec,
            "video_label": video_label,
            "visual_label": vis_label,
            "audio_label": aud_label,
            "gt_mask": gt_mask,
            "has_mask_gt": has_mask_gt,
            "visual_intervals": vis_intervals,
            "audio_intervals": aud_intervals,
            "has_visual_tfl_gt": True,
            "has_audio_tfl_gt": has_audio,
            "domain": f"synth_mode_{mode}",
        }


def forensic_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate heterogeneous video/audio/mask/interval samples into a unified batch."""
    videos = torch.stack([item["video"] for item in batch], dim=0)
    audios = torch.stack([item["audio"] for item in batch], dim=0)
    has_audio = torch.tensor([bool(item["has_audio"]) for item in batch], dtype=torch.bool)
    gt_masks = torch.stack([item["gt_mask"] for item in batch], dim=0)

    targets = BatchTargets(
        video_label=torch.tensor([item["video_label"] for item in batch], dtype=torch.float32),
        visual_label=torch.tensor([item["visual_label"] for item in batch], dtype=torch.float32),
        audio_label=torch.tensor([item["audio_label"] for item in batch], dtype=torch.float32),
        gt_masks=gt_masks,
        has_mask_gt=torch.tensor([bool(item["has_mask_gt"]) for item in batch], dtype=torch.bool),
        visual_intervals=[item["visual_intervals"] for item in batch],
        audio_intervals=[item["audio_intervals"] for item in batch],
        has_visual_tfl_gt=torch.tensor(
            [bool(item["has_visual_tfl_gt"]) for item in batch], dtype=torch.bool
        ),
        has_audio_tfl_gt=torch.tensor(
            [bool(item["has_audio_tfl_gt"]) for item in batch], dtype=torch.bool
        ),
    )

    return {
        "video": videos,
        "audio": audios,
        "has_audio": has_audio,
        "duration_sec": float(batch[0]["duration_sec"]),
        "targets": targets,
        "domains": [item["domain"] for item in batch],
    }
