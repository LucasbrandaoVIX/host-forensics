#!/usr/bin/env python3
"""End-to-End Real Foundation-Backbone Multi-Benchmark Pilot & 5-Variant Ablation Runner.

Downloads open-access real & manipulated videos (`Celeb-DF-v2`, `YouTube-real`, and unseen `Google DFD`)
plus authentic & AI-cloned speech (`deepfake-audio-detection`) from Hugging Face Hub, constructs
strictly identity-disjoint train / pristine-calibration / multi-domain test manifests via FFmpeg,
and trains + evaluates 5 complete HoST-MoE variants using pretrained `facebook/dino-vits16` +
`microsoft/wavlm-base-plus` (`123.5M` parameters, `7.27M` trainable):

  1. Full HoST-MoE (Ours)
  2. w/o Orthogonal SVD-LoRA (Standard unconstrained LoRA W = W_0 + B A)
  3. w/o Auxiliary-Loss-Free Bias Balancing (bias_update_rate = 0)
  4. w/o Host Depolarization & MIL Anchor Loss (lambda_polarization = 0, lambda_anchor = 0)
  5. w/o Symmetric Restoration/VAE Laundering (symmetric_laundering_prob = 0)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
from typing import Any

# Ensure HuggingFace Hub uses standard HTTPS proxy path rather than xet sockets
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

from host_forensics.config import HoSTConfig
from host_forensics.data.datasets import UnifiedForensicManifestDataset, build_optimized_dataloader
from host_forensics.data.transforms import CompoundDegradationAndLaundering
from host_forensics.data.video_io import IMAGENET_MEAN, IMAGENET_STD, extract_video_and_audio_ffmpeg
from host_forensics.engine.trainer import HoSTTrainer
from host_forensics.metrics.evaluation import (
    compute_auroc,
    compute_eer,
    compute_mid_ffd_metrics,
    compute_native_track_av_map,
    compute_polarization_cross_auc,
    compute_spatial_grounding_metrics,
)
from host_forensics.models.backbones import OrthogonalSVDLinear
from host_forensics.models.heads import TemporalProposal
from host_forensics.models.host_model import HoSTForensicsModel


VIDEO_REPO = "gonnerthetooner/deepfake_video_data"
AUDIO_REPO = "garystafford/deepfake-audio-detection"
ATTRIBUTION_MAP = {"clean": 0, "visual_only": 1, "audio_only": 2, "both_track": 3}


def _normalize_rgb(frames_01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (frames_01.clamp(0.0, 1.0) - mean) / std


def _decode_native_512x288_u8(
    path: str, num_frames: int = 6, duration_sec: float = 2.0, native_h: int = 288, native_w: int = 512
) -> torch.Tensor:
    """Decode `num_frames` RGB frames at native 16:9 `512x288` aspect as compact `(T, 3, H, W)` uint8."""
    fps = max(1.0, num_frames / max(duration_sec, 0.25))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        f"{duration_sec + 0.35:.2f}",
        "-i",
        path,
        "-vf",
        f"fps={fps:.4f},scale={native_w}:{native_h}:flags=bicubic",
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    frame_bytes = native_h * native_w * 3
    n_dec = len(proc.stdout) // frame_bytes
    if n_dec == 0:
        fallback = extract_video_and_audio_ffmpeg(
            path,
            num_frames=num_frames,
            img_size=native_h,
            max_duration_sec=duration_sec,
            normalize_imagenet=False,
        )["video"]
        resized = F.interpolate(fallback, size=(native_h, native_w), mode="bilinear", align_corners=False)
        return (resized.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8)

    arr = np.frombuffer(proc.stdout[: n_dec * frame_bytes], dtype=np.uint8).copy()
    frames = torch.from_numpy(arr).view(n_dec, native_h, native_w, 3).permute(0, 3, 1, 2).contiguous()
    if frames.shape[0] < num_frames:
        pad = frames[-1:].repeat(num_frames - frames.shape[0], 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
    return frames[:num_frames]


def _find_subject_box(frames_u8: torch.Tensor, crop_size: int = 160) -> tuple[int, int]:
    """Locate a `crop_size x crop_size` window preserving both foreground subject (~20%) and host context (~80%).

    Uses pure-PyTorch YCbCr skin-locus salience + temporal motion energy + mild upper-center framing prior
    computed strictly from `frames_u8` alone (zero external detector dependency and zero reference leakage).
    """
    t, _, h, w = frames_u8.shape
    if h <= crop_size or w <= crop_size:
        return max(0, (h - crop_size) // 2), max(0, (w - crop_size) // 2)

    frames_01 = frames_u8.float() / 255.0
    r, g, b = frames_01[:, 0], frames_01[:, 1], frames_01[:, 2]
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin = torch.exp(-(((cb - 0.43) / 0.08) ** 2 + ((cr - 0.58) / 0.08) ** 2)).mean(dim=0)

    if t > 1:
        motion = (frames_01[1:] - frames_01[:-1]).abs().mean(dim=(0, 1))
        motion = motion / motion.amax().clamp(min=1e-4)
    else:
        motion = torch.zeros_like(skin)

    yy = torch.linspace(-1.0, 1.0, h).view(h, 1)
    xx = torch.linspace(-1.0, 1.0, w).view(1, w)
    prior = torch.exp(-0.5 * (((yy + 0.15) / 0.65) ** 2 + (xx / 0.75) ** 2))
    salience = (0.65 * skin + 0.35 * motion) * (0.5 + 0.5 * prior)

    stride = 4
    win_score = F.avg_pool2d(salience.view(1, 1, h, w), kernel_size=crop_size, stride=stride).squeeze(0).squeeze(0)
    best_flat = int(win_score.argmax().item())
    best_r = best_flat // win_score.shape[1]
    best_c = best_flat % win_score.shape[1]
    y0 = min(max(0, best_r * stride), h - crop_size)
    x0 = min(max(0, best_c * stride), w - crop_size)
    return y0, x0


def _crop_to_square_01(
    frames_u8: torch.Tensor, box: tuple[int, int], crop_size: int = 160, out_size: int = 224
) -> torch.Tensor:
    """Crop `(T, 3, crop_size, crop_size)` at `box=(y0, x0)` and resize to `(T, 3, out_size, out_size)` in `[0, 1]`."""
    y0, x0 = box
    _, _, h, w = frames_u8.shape
    cs = min(crop_size, h, w)
    y0 = min(max(0, y0), h - cs)
    x0 = min(max(0, x0), w - cs)
    patch = frames_u8[:, :, y0 : y0 + cs, x0 : x0 + cs].float() / 255.0
    if cs == out_size:
        return patch.clamp(0.0, 1.0)
    return F.interpolate(patch, size=(out_size, out_size), mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def _decode_audio_flac(path: str, sample_rate: int = 16000, duration_sec: float = 2.0) -> torch.Tensor:
    target_len = int(round(sample_rate * duration_sec))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        f"{duration_sec + 0.5:.2f}",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    wav = torch.from_numpy(arr)
    if wav.numel() == 0:
        return torch.zeros(target_len, dtype=torch.float32)
    if wav.numel() < target_len:
        reps = (target_len + wav.numel() - 1) // wav.numel()
        wav = wav.repeat(reps)[:target_len]
    else:
        wav = wav[:target_len]
    # Normalize peak amplitude moderately for consistent speech energy
    peak = float(wav.abs().max().item())
    if peak > 1e-4:
        wav = 0.85 * (wav / peak)
    return wav


def _reencode_h264_frames(frames_01: torch.Tensor, crf: int = 35) -> torch.Tensor:
    """Pass `[0, 1]` `(T, 3, H, W)` frames through real libx264 CRF compression and decode back."""
    t, _, h, w = frames_01.shape
    raw_in = (frames_01.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8).tobytes()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        enc_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            "8",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            tmp.name,
        ]
        subprocess.run(enc_cmd, input=raw_in, capture_output=True, check=True)
        dec_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            tmp.name,
            "-vf",
            f"scale={w}:{h}:flags=bicubic",
            "-frames:v",
            str(t),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        dec_proc = subprocess.run(dec_cmd, capture_output=True, check=True)
        arr = np.frombuffer(dec_proc.stdout, dtype=np.uint8)
        n_dec = len(arr) // (h * w * 3)
        if n_dec == 0:
            return frames_01
        out = torch.from_numpy(arr[: n_dec * h * w * 3].copy()).view(n_dec, h, w, 3).permute(0, 3, 1, 2).float() / 255.0
        if out.shape[0] < t:
            out = torch.cat([out, out[-1:].repeat(t - out.shape[0], 1, 1, 1)], dim=0)
        return out[:t]


def _apply_domain_shift(frames_01: torch.Tensor, domain_kind: str) -> torch.Tensor:
    """Apply realistic codec or neural-restoration/VAE-like laundering shift equally to real & fake."""
    if domain_kind == "hq":
        return frames_01
    if domain_kind == "h264_crf35":
        return _reencode_h264_frames(frames_01, crf=35)
    if domain_kind == "laundered":
        # Downsample-upsample Bottleneck + mild unsharp mask + global color-temperature/exposure shift
        t, c, h, w = frames_01.shape
        low = F.interpolate(frames_01, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
        rec = F.interpolate(low, size=(h, w), mode="bicubic", align_corners=False)
        blur = F.avg_pool2d(rec, kernel_size=3, stride=1, padding=1)
        sharp = (rec + 0.35 * (rec - blur)).clamp(0.0, 1.0)
        gain = torch.tensor([1.05, 0.99, 0.95], dtype=torch.float32).view(1, 3, 1, 1)
        bias = torch.tensor([0.02, 0.0, -0.015], dtype=torch.float32).view(1, 3, 1, 1)
        return (sharp * gain + bias).clamp(0.0, 1.0)
    raise ValueError(f"Unknown domain_kind={domain_kind}")


def _paired_faceswap_mask(real_01: torch.Tensor, syn_01: torch.Tensor) -> torch.Tensor:
    """Compute clean `(T, 1, H, W)` face-swap region mask from pixel-aligned `Celeb-real` & `Celeb-synthesis`."""
    diff = (real_01 - syn_01).abs().mean(dim=1, keepdim=True)  # (T, 1, H, W)
    smooth = F.avg_pool2d(diff, kernel_size=11, stride=1, padding=5)
    # Adaptive per-clip threshold capturing the swapped facial region
    peak = smooth.amax(dim=(1, 2, 3), keepdim=True).clamp(min=0.02)
    mask = (smooth > (0.25 * peak)).float()
    # Morphological closing via max-pool dilation
    mask = F.max_pool2d(mask, kernel_size=9, stride=1, padding=4)
    # Fallback central face ellipse if source/target exposure matched closely
    t, _, h, w = mask.shape
    if float(mask.mean().item()) < 0.03:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij"
        )
        ellipse = (((yy + 0.05) / 0.42) ** 2 + (xx / 0.34) ** 2 <= 1.0).float()
        mask = ellipse.view(1, 1, h, w).expand(t, 1, h, w).clone()
    return mask


def _inject_tubelet_inpainting(
    real_01: torch.Tensor, rng: random.Random, duration_sec: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor, list[list[float]]]:
    """Create a realistic moving-object removal / localized diffusion-inpainting tubelet on camera video."""
    t, c, h, w = real_01.shape
    out = real_01.clone()
    mask = torch.zeros(t, 1, h, w, dtype=torch.float32)

    f_start = rng.randint(1, max(1, t // 3))
    f_end = rng.randint(max(f_start + 2, (2 * t) // 3), t)
    t_start = round((f_start / float(t)) * duration_sec, 3)
    t_end = round((f_end / float(t)) * duration_sec, 3)

    cy0 = rng.uniform(0.30, 0.68) * h
    cx0 = rng.uniform(0.28, 0.72) * w
    vy = rng.uniform(-0.04, 0.04) * h
    vx = rng.uniform(-0.05, 0.05) * w
    ry = rng.uniform(0.14, 0.23) * h
    rx = rng.uniform(0.14, 0.23) * w

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )
    # Build smooth spatial-temporal background propagation + VAE-bottleneck fill inside moving region
    bg_fill = F.interpolate(
        F.avg_pool2d(real_01, kernel_size=9, stride=1, padding=4),
        size=(h // 4, w // 4),
        mode="bilinear",
        align_corners=False,
    )
    bg_fill = F.interpolate(bg_fill, size=(h, w), mode="bicubic", align_corners=False)
    temporal_roll = torch.roll(bg_fill, shifts=1, dims=0)
    propagated = 0.65 * bg_fill + 0.35 * temporal_roll

    for fi in range(f_start, f_end):
        cy = min(max(cy0 + vy * (fi - f_start), ry + 2), h - ry - 2)
        cx = min(max(cx0 + vx * (fi - f_start), rx + 2), w - rx - 2)
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        binary = (dist <= 1.0).float()
        soft_alpha = ((1.18 - dist) / 0.36).clamp(0.0, 1.0).view(1, h, w)
        # Subtle synthetic diffusion micro-texture (zero ISP camera correlation)
        synth_tex = 0.018 * torch.randn(c, h, w, generator=torch.Generator().manual_seed(1000 + fi))
        inpainted = (propagated[fi] + synth_tex).clamp(0.0, 1.0)
        out[fi] = (1.0 - soft_alpha) * out[fi] + soft_alpha * inpainted
        mask[fi, 0] = binary

    return out, mask, [[t_start, t_end]]


def _splice_audio_segment(
    real_wav: torch.Tensor,
    fake_wav: torch.Tensor,
    rng: random.Random,
    sample_rate: int = 16000,
    duration_sec: float = 2.0,
) -> tuple[torch.Tensor, list[list[float]]]:
    """Splice a partial AI voice-cloned speech segment into authentic speech with 15 ms cosine cross-fades."""
    total_len = real_wav.numel()
    t_start = round(rng.uniform(0.25, 0.85), 3)
    seg_dur = round(rng.uniform(0.65, 1.05), 3)
    t_end = round(min(duration_sec - 0.05, t_start + seg_dur), 3)

    s_idx = int(round(t_start * sample_rate))
    e_idx = int(round(t_end * sample_rate))
    seg_len = max(320, e_idx - s_idx)

    out = real_wav.clone()
    src_seg = fake_wav[:seg_len].clone()
    fade = min(240, seg_len // 4)
    if fade > 1:
        ramp = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, torch.pi, fade)))
        out_seg = out[s_idx : s_idx + seg_len].clone()
        blended = src_seg.clone()
        blended[:fade] = (1.0 - ramp) * out_seg[:fade] + ramp * src_seg[:fade]
        blended[-fade:] = ramp * out_seg[-fade:] + (1.0 - ramp) * src_seg[-fade:]
        out[s_idx : s_idx + seg_len] = blended
    else:
        out[s_idx : s_idx + seg_len] = src_seg
    return out[:total_len], [[t_start, t_end]]


def _parse_celeb_syn_ids(rel_path: str) -> tuple[int, int, str] | None:
    # Format: celebdf_v2/Celeb-synthesis/id0_id16_0000.mp4
    m = re.search(r"id(\d+)_id(\d+)_(\d+\.mp4)$", rel_path)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), f"celebdf_v2/Celeb-real/id{m.group(1)}_{m.group(3)}"


def _parse_celeb_real_id(rel_path: str) -> int | None:
    m = re.search(r"id(\d+)_\d+\.mp4$", rel_path)
    return int(m.group(1)) if m else None


def build_real_pilot_manifests(
    root_dir: Path,
    num_frames: int = 6,
    base_img_size: int = 224,
    sample_rate: int = 16000,
    duration_sec: float = 2.0,
) -> dict[str, Path]:
    """Download disjoint real video/audio subsets from HF Hub and write `.pt` + `.jsonl` manifests."""
    manifest_dir = root_dir / "manifests"
    tensor_dir = root_dir / "tensors"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    sentinel = manifest_dir / "_READY.json"
    expected_paths = {
        "train_stage1": manifest_dir / "train_stage1.jsonl",
        "train_stage2": manifest_dir / "train_stage2_highres.jsonl",
        "val": manifest_dir / "val_pristine_and_mixed.jsonl",
        "test_celebdf": manifest_dir / "test_celebdf_v2.jsonl",
        "test_googledfd": manifest_dir / "test_googledfd_zeroshot.jsonl",
        "test_inpaint": manifest_dir / "test_inpaint_tubelet_silent.jsonl",
        "test_avsplice": manifest_dir / "test_av_splice.jsonl",
    }
    if sentinel.exists() and all(p.exists() for p in expected_paths.values()):
        print("[Data] Reusing cached real-world manifests in", manifest_dir, flush=True)
        return expected_paths

    t0 = time.time()
    print("[Data] Querying Hugging Face dataset repositories...", flush=True)
    api = HfApi()
    v_files = sorted(api.list_repo_files(VIDEO_REPO, repo_type="dataset"))
    a_files = sorted(api.list_repo_files(AUDIO_REPO, repo_type="dataset"))

    celeb_real_set = {f for f in v_files if f.startswith("celebdf_v2/Celeb-real/") and f.endswith(".mp4")}
    celeb_syn_all = [f for f in v_files if f.startswith("celebdf_v2/Celeb-synthesis/") and f.endswith(".mp4")]
    yt_real_all = [f for f in v_files if f.startswith("celebdf_v2/YouTube-real/") and f.endswith(".mp4")]
    dfd_real_all = [f for f in v_files if "DFD_original" in f and f.endswith(".mp4")]
    dfd_fake_all = [f for f in v_files if "DFD_manipulated" in f and f.endswith(".mp4")]
    aud_real_all = [f for f in a_files if f.startswith("real/") and f.endswith(".flac")]
    aud_fake_all = [f for f in a_files if f.startswith("fake/") and f.endswith(".flac")]

    rng = random.Random(2026)

    def _collect_split_pairs(id_min: int, id_max: int, limit: int) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for syn_f in celeb_syn_all:
            parsed = _parse_celeb_syn_ids(syn_f)
            if parsed is None:
                continue
            id_a, id_b, real_f = parsed
            if id_min <= id_a <= id_max and id_min <= id_b <= id_max and real_f in celeb_real_set:
                candidates.append((real_f, syn_f))
        rng.shuffle(candidates)
        # Diversify across target identities
        seen_counts: dict[int, int] = {}
        chosen: list[tuple[str, str]] = []
        for r_f, s_f in candidates:
            cid = _parse_celeb_real_id(r_f) or 0
            if seen_counts.get(cid, 0) >= 3:
                continue
            seen_counts[cid] = seen_counts.get(cid, 0) + 1
            chosen.append((r_f, s_f))
            if len(chosen) >= limit:
                break
        return chosen

    def _collect_split_reals(id_min: int, id_max: int, limit: int) -> list[str]:
        cands = [
            f for f in sorted(celeb_real_set)
            if (cid := _parse_celeb_real_id(f)) is not None and id_min <= cid <= id_max
        ]
        rng.shuffle(cands)
        return cands[:limit]

    # Identity-disjoint Celeb-DF-v2 & YouTube-real splits:
    #   Train: Celeb id 0..29, YouTube 0..139
    #   Val:   Celeb id 30..41, YouTube 140..189
    #   Test:  Celeb id 42..61, YouTube 190..299
    train_pairs = _collect_split_pairs(0, 29, 26)
    val_pairs = _collect_split_pairs(30, 41, 8)
    test_pairs = _collect_split_pairs(42, 61, 28)

    train_reals = _collect_split_reals(0, 29, 22)
    val_reals = _collect_split_reals(30, 41, 12)
    test_reals = _collect_split_reals(42, 61, 24)

    yt_train = yt_real_all[:24]
    yt_val = yt_real_all[140:152]
    yt_test = yt_real_all[190:230]

    rng_dfd = random.Random(777)
    dfd_r_shuffled = dfd_real_all[:]
    dfd_f_shuffled = dfd_fake_all[:]
    rng_dfd.shuffle(dfd_r_shuffled)
    rng_dfd.shuffle(dfd_f_shuffled)
    dfd_reals = dfd_r_shuffled[:20]
    dfd_fakes = dfd_f_shuffled[:20]

    aud_r_train, aud_r_val, aud_r_test = aud_real_all[:50], aud_real_all[50:75], aud_real_all[75:130]
    aud_f_train, aud_f_val, aud_f_test = aud_fake_all[:40], aud_fake_all[40:60], aud_fake_all[60:105]

    needed_videos = sorted(
        set(
            [r for r, _ in train_pairs + val_pairs + test_pairs]
            + [s for _, s in train_pairs + val_pairs + test_pairs]
            + train_reals
            + val_reals
            + test_reals
            + yt_train
            + yt_val
            + yt_test
            + dfd_reals
            + dfd_fakes
        )
    )
    needed_audios = sorted(
        set(aud_r_train + aud_r_val + aud_r_test + aud_f_train + aud_f_val + aud_f_test)
    )
    print(
        f"[Data] Downloading {len(needed_videos)} real/deepfake .mp4 videos + {len(needed_audios)} speech .flac files...",
        flush=True,
    )

    local_videos: dict[str, str] = {}
    local_audios: dict[str, str] = {}

    def _dl_vid(rel: str) -> tuple[str, str]:
        return rel, hf_hub_download(VIDEO_REPO, rel, repo_type="dataset")

    def _dl_aud(rel: str) -> tuple[str, str]:
        return rel, hf_hub_download(AUDIO_REPO, rel, repo_type="dataset")

    with ThreadPoolExecutor(max_workers=8) as pool:
        for rel, loc in pool.map(_dl_vid, needed_videos):
            local_videos[rel] = loc
        for rel, loc in pool.map(_dl_aud, needed_audios):
            local_audios[rel] = loc

    print(f"[Data] Downloaded all media in {time.time() - t0:.1f}s. Decoding 16:9 frames & audio via FFmpeg...", flush=True)

    # Decode compact 512x288 uint8 16:9 tensors + independent per-clip subject+host boxes in parallel
    raw_512_u8: dict[str, torch.Tensor] = {}
    subject_boxes: dict[str, tuple[int, int]] = {}
    decoded_wav: dict[str, torch.Tensor] = {}
    crop_win = 160

    def _dec_v(rel: str) -> tuple[str, torch.Tensor, tuple[int, int]]:
        u8 = _decode_native_512x288_u8(local_videos[rel], num_frames=num_frames, duration_sec=duration_sec)
        box = _find_subject_box(u8, crop_size=crop_win)
        return rel, u8, box

    def _dec_a(rel: str) -> tuple[str, torch.Tensor]:
        return rel, _decode_audio_flac(local_audios[rel], sample_rate=sample_rate, duration_sec=duration_sec)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for rel, u8, box in pool.map(_dec_v, needed_videos):
            raw_512_u8[rel] = u8
            subject_boxes[rel] = box
        for rel, wav in pool.map(_dec_a, needed_audios):
            decoded_wav[rel] = wav

    def _get_cropped(rel: str, box: tuple[int, int] | None = None) -> torch.Tensor:
        use_box = box if box is not None else subject_boxes[rel]
        return _crop_to_square_01(raw_512_u8[rel], use_box, crop_size=crop_win, out_size=base_img_size)

    domains = ["hq", "h264_crf35", "laundered"]
    sample_counter = 0

    def _save_record(
        frames_01: torch.Tensor,
        audio_wav: torch.Tensor,
        has_audio: bool,
        video_label: int,
        visual_label: int | None,
        audio_label: int | None,
        gt_masks: torch.Tensor | None,
        visual_intervals: list[list[float]] | None,
        audio_intervals: list[list[float]] | None,
        domain_prefix: str,
        domain_kind: str,
        regime: str,
    ) -> dict[str, Any]:
        nonlocal sample_counter
        sample_counter += 1
        shifted_01 = _apply_domain_shift(frames_01, domain_kind)
        norm_vid = _normalize_rgb(shifted_01)

        vid_pt = tensor_dir / f"vid_{sample_counter:05d}.pt"
        torch.save(norm_vid.half(), vid_pt)

        aud_pt: str | None = None
        if has_audio:
            aud_file = tensor_dir / f"aud_{sample_counter:05d}.pt"
            torch.save(audio_wav.float(), aud_file)
            aud_pt = str(aud_file)

        mask_pt: str | None = None
        if gt_masks is not None:
            m_3d = gt_masks.squeeze(1) if (gt_masks.ndim == 4 and gt_masks.shape[1] == 1) else gt_masks
            mask_file = tensor_dir / f"mask_{sample_counter:05d}.pt"
            torch.save(m_3d.to(dtype=torch.uint8), mask_file)
            mask_pt = str(mask_file)

        return {
            "video_id": f"pilot_{sample_counter:05d}",
            "video_tensor_pt": str(vid_pt),
            "audio_tensor_pt": aud_pt,
            "mask_tensor_pt": mask_pt,
            "has_audio": has_audio,
            "duration_sec": duration_sec,
            "video_label": video_label,
            "visual_label": visual_label if visual_label is not None else video_label,
            "audio_label": audio_label if audio_label is not None else 0,
            "visual_intervals": visual_intervals,
            "audio_intervals": audio_intervals,
            "domain": f"{domain_prefix}_{domain_kind}",
            "regime": regime,
        }

    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    # 1. Stage-1 Unified Training Set (64 balanced real clips across all 4 regimes & 3 domains)
    s1_rows: list[dict[str, Any]] = []
    s2_rows: list[dict[str, Any]] = []

    # 16 Clean authentic clips (Celeb-real + YouTube-real, audio-visual & silent)
    clean_pool = train_reals[:10] + yt_train[:6]
    for idx, v_rel in enumerate(clean_pool):
        dom = domains[idx % 3]
        has_a = (idx % 4 != 3)
        a_wav = decoded_wav[aud_r_train[idx % len(aud_r_train)]]
        z_mask = torch.zeros(num_frames, 1, base_img_size, base_img_size, dtype=torch.float32)
        rec = _save_record(
            _get_cropped(v_rel), a_wav, has_a, 0, 0, 0,
            z_mask, [], [], "train", dom, "clean"
        )
        s1_rows.append(rec)
        if idx < 8:
            s2_rows.append(rec)

    # 10 Face-swap visual_only clips (real Celeb-synthesis + pixel-aligned mask, full & partial temporal splices)
    for idx in range(10):
        r_rel, s_rel = train_pairs[idx]
        dom = domains[idx % 3]
        if idx % 2 == 0:
            s_rgb = _get_cropped(s_rel)
            r_aligned = _get_cropped(r_rel, box=subject_boxes[s_rel])
            clip_rgb = s_rgb
            clip_mask = _paired_faceswap_mask(r_aligned, s_rgb)
            v_int = [[0.0, duration_sec]]
        else:
            # Partial temporal face-swap splice inside authentic host video
            box_r = subject_boxes[r_rel]
            r_rgb = _get_cropped(r_rel, box=box_r)
            s_aligned = _get_cropped(s_rel, box=box_r)
            full_mask = _paired_faceswap_mask(r_rgb, s_aligned)
            clip_rgb = r_rgb.clone()
            clip_mask = torch.zeros_like(full_mask)
            clip_rgb[1:5] = s_aligned[1:5]
            clip_mask[1:5] = full_mask[1:5]
            v_int = [[round(duration_sec * 1 / num_frames, 3), round(duration_sec * 5 / num_frames, 3)]]
        has_a = (idx % 3 != 2)
        a_wav = decoded_wav[aud_r_train[(idx + 16) % len(aud_r_train)]]
        rec = _save_record(
            clip_rgb, a_wav, has_a, 1, 1, 0, clip_mask, v_int, [], "train", dom, "visual_only"
        )
        s1_rows.append(rec)
        if idx < 6:
            s2_rows.append(rec)

    # 10 Localized moving-region inpainting / object-removal clips on real YouTube camera videos
    for idx in range(10):
        v_rel = yt_train[6 + idx]
        dom = domains[idx % 3]
        inp_rgb, inp_mask, v_int = _inject_tubelet_inpainting(_get_cropped(v_rel), rng, duration_sec)
        has_a = (idx % 2 == 0)
        a_wav = decoded_wav[aud_r_train[(idx + 26) % len(aud_r_train)]]
        rec = _save_record(
            inp_rgb, a_wav, has_a, 1, 1, 0, inp_mask, v_int, [], "train", dom, "visual_only"
        )
        s1_rows.append(rec)
        if idx < 6:
            s2_rows.append(rec)

    # 14 Audio-only sparse voice-clone splices on authentic video
    aud_only_vids = train_reals[10:18] + yt_train[16:22]
    for idx, v_rel in enumerate(aud_only_vids):
        dom = domains[idx % 3]
        r_wav = decoded_wav[aud_r_train[(idx + 34) % len(aud_r_train)]]
        f_wav = decoded_wav[aud_f_train[idx % len(aud_f_train)]]
        spliced_wav, a_int = _splice_audio_segment(r_wav, f_wav, rng, sample_rate, duration_sec)
        z_mask = torch.zeros(num_frames, 1, base_img_size, base_img_size, dtype=torch.float32)
        rec = _save_record(
            _get_cropped(v_rel), spliced_wav, True, 1, 0, 1, z_mask, [], a_int, "train", dom, "audio_only"
        )
        s1_rows.append(rec)
        if idx < 4:
            s2_rows.append(rec)

    # 14 Both-track audio-visual deepfakes (Celeb-synthesis + voice-cloned speech splice)
    for idx in range(14):
        r_rel, s_rel = train_pairs[10 + idx]
        dom = domains[idx % 3]
        s_rgb = _get_cropped(s_rel)
        r_aligned = _get_cropped(r_rel, box=subject_boxes[s_rel])
        full_mask = _paired_faceswap_mask(r_aligned, s_rgb)
        r_wav = decoded_wav[aud_r_train[(idx + 12) % len(aud_r_train)]]
        f_wav = decoded_wav[aud_f_train[(idx + 14) % len(aud_f_train)]]
        spliced_wav, a_int = _splice_audio_segment(r_wav, f_wav, rng, sample_rate, duration_sec)
        rec = _save_record(
            s_rgb, spliced_wav, True, 1, 1, 1, full_mask, [[0.0, duration_sec]], a_int, "train", dom, "both_track"
        )
        s1_rows.append(rec)
        if idx < 4:
            s2_rows.append(rec)

    # 2. Validation & Pristine Calibration Set (28 disjoint clips: 16 pristine across all 3 domains + 12 manipulated)
    val_rows: list[dict[str, Any]] = []
    val_clean_vids = val_reals[:10] + yt_val[:6]
    for idx, v_rel in enumerate(val_clean_vids):
        dom = domains[idx % 3]
        has_a = (idx % 4 != 3)
        a_wav = decoded_wav[aud_r_val[idx % len(aud_r_val)]]
        z_mask = torch.zeros(num_frames, 1, base_img_size, base_img_size, dtype=torch.float32)
        val_rows.append(
            _save_record(_get_cropped(v_rel), a_wav, has_a, 0, 0, 0, z_mask, [], [], "val", dom, "clean")
        )
    for idx in range(6):
        r_rel, s_rel = val_pairs[idx]
        dom = domains[idx % 3]
        s_rgb = _get_cropped(s_rel)
        r_aligned = _get_cropped(r_rel, box=subject_boxes[s_rel])
        m = _paired_faceswap_mask(r_aligned, s_rgb)
        a_wav = decoded_wav[aud_r_val[(idx + 8) % len(aud_r_val)]]
        val_rows.append(
            _save_record(s_rgb, a_wav, True, 1, 1, 0, m, [[0.0, duration_sec]], [], "val", dom, "visual_only")
        )
    for idx in range(6):
        v_rel = yt_val[6 + idx]
        dom = domains[idx % 3]
        r_wav = decoded_wav[aud_r_val[(idx + 14) % len(aud_r_val)]]
        f_wav = decoded_wav[aud_f_val[idx % len(aud_f_val)]]
        sp_wav, a_int = _splice_audio_segment(r_wav, f_wav, rng, sample_rate, duration_sec)
        z_mask = torch.zeros(num_frames, 1, base_img_size, base_img_size, dtype=torch.float32)
        val_rows.append(
            _save_record(_get_cropped(v_rel), sp_wav, True, 1, 0, 1, z_mask, [], a_int, "val", dom, "audio_only")
        )

    # 3. Benchmark 1: CelebDF_v2 Held-Out Test (36 clips across 3 domains: 18 real + 18 Celeb-synthesis)
    celeb_test_rows: list[dict[str, Any]] = []
    for idx in range(18):
        v_rel = test_reals[idx]
        dom = domains[idx % 3]
        a_wav = decoded_wav[aud_r_test[idx % len(aud_r_test)]]
        z_mask = torch.zeros(num_frames, 1, base_img_size, base_img_size, dtype=torch.float32)
        celeb_test_rows.append(
            _save_record(_get_cropped(v_rel), a_wav, True, 0, 0, 0, z_mask, [], [], "celebdf", dom, "clean")
        )
    for idx in range(18):
        r_rel, s_rel = test_pairs[idx]
        dom = domains[idx % 3]
        s_rgb = _get_cropped(s_rel)
        r_aligned = _get_cropped(r_rel, box=subject_boxes[s_rel])
        m = _paired_faceswap_mask(r_aligned, s_rgb)
        a_wav = decoded_wav[aud_r_test[(idx + 18) % len(aud_r_test)]]
        celeb_test_rows.append(
            _save_record(s_rgb, a_wav, True, 1, 1, 0, m, [[0.0, duration_sec]], [], "celebdf", dom, "visual_only")
        )

    # 4. Benchmark 2: Google DFD Zero-Shot Unseen Benchmark (40 clips across 3 domains: 20 real + 20 fake)
    dfd_test_rows: list[dict[str, Any]] = []
    for idx in range(20):
        dom = domains[idx % 3]
        a_wav = decoded_wav[aud_r_test[(idx + 5) % len(aud_r_test)]]
        dfd_test_rows.append(
            _save_record(_get_cropped(dfd_reals[idx]), a_wav, False, 0, 0, 0, None, [], [], "dfd", dom, "clean")
        )
        dfd_test_rows.append(
            _save_record(_get_cropped(dfd_fakes[idx]), a_wav, False, 1, 1, 0, None, [[0.0, duration_sec]], [], "dfd", dom, "visual_only")
        )

    # 5. Benchmark 3: Localized Moving-Tubelet Inpainting / Object Removal (Silent Mode: 30 clips)
    inpaint_test_rows: list[dict[str, Any]] = []
    zero_wav = torch.zeros(int(round(sample_rate * duration_sec)), dtype=torch.float32)
    for idx in range(12):
        dom = domains[idx % 3]
        z_mask = torch.zeros(num_frames, 1, base_img_size, base_img_size, dtype=torch.float32)
        inpaint_test_rows.append(
            _save_record(_get_cropped(yt_test[idx]), zero_wav, False, 0, 0, 0, z_mask, [], [], "inpaint", dom, "clean")
        )
    for idx in range(18):
        dom = domains[idx % 3]
        inp_rgb, inp_mask, v_int = _inject_tubelet_inpainting(_get_cropped(yt_test[12 + idx]), rng, duration_sec)
        inpaint_test_rows.append(
            _save_record(inp_rgb, zero_wav, False, 1, 1, 0, inp_mask, v_int, [], "inpaint", dom, "visual_only")
        )

    # 6. Benchmark 4: AV-Splice Asymmetric Audio-Visual Benchmark (32 clips: 8 per 4-way regime)
    av_test_rows: list[dict[str, Any]] = []
    for idx in range(8):
        dom = domains[idx % 3]
        z_mask = torch.zeros(num_frames, 1, base_img_size, base_img_size, dtype=torch.float32)
        # Clean
        av_test_rows.append(
            _save_record(
                _get_cropped(test_reals[16 + (idx % 8)]),
                decoded_wav[aud_r_test[idx]],
                True, 0, 0, 0, z_mask, [], [], "avsplice", dom, "clean",
            )
        )
        # Visual-only (partial temporal swap + clean speech)
        r_rel, s_rel = test_pairs[18 + idx]
        box_r = subject_boxes[r_rel]
        r_rgb = _get_cropped(r_rel, box=box_r)
        s_aligned = _get_cropped(s_rel, box=box_r)
        fm = _paired_faceswap_mask(r_rgb, s_aligned)
        part_rgb = r_rgb.clone()
        part_m = torch.zeros_like(fm)
        part_rgb[1:5] = s_aligned[1:5]
        part_m[1:5] = fm[1:5]
        v_int = [[round(duration_sec * 1 / num_frames, 3), round(duration_sec * 5 / num_frames, 3)]]
        av_test_rows.append(
            _save_record(
                part_rgb,
                decoded_wav[aud_r_test[8 + idx]],
                True, 1, 1, 0, part_m, v_int, [], "avsplice", dom, "visual_only",
            )
        )
        # Audio-only (clean video + partial voice-clone splice)
        sp_wav, a_int = _splice_audio_segment(
            decoded_wav[aud_r_test[16 + idx]],
            decoded_wav[aud_f_test[idx]],
            rng, sample_rate, duration_sec,
        )
        av_test_rows.append(
            _save_record(
                _get_cropped(yt_test[30 + idx]),
                sp_wav,
                True, 1, 0, 1, z_mask, [], a_int, "avsplice", dom, "audio_only",
            )
        )
        # Both-track (face-swap + voice-clone splice)
        r_rel2, s_rel2 = test_pairs[idx]
        s_rgb2 = _get_cropped(s_rel2)
        r_aligned2 = _get_cropped(r_rel2, box=subject_boxes[s_rel2])
        m2 = _paired_faceswap_mask(r_aligned2, s_rgb2)
        sp_wav2, a_int2 = _splice_audio_segment(
            decoded_wav[aud_r_test[24 + idx]],
            decoded_wav[aud_f_test[8 + idx]],
            rng, sample_rate, duration_sec,
        )
        av_test_rows.append(
            _save_record(
                s_rgb2,
                sp_wav2,
                True, 1, 1, 1, m2, [[0.0, duration_sec]], a_int2, "avsplice", dom, "both_track",
            )
        )

    _write_jsonl(expected_paths["train_stage1"], s1_rows)
    _write_jsonl(expected_paths["train_stage2"], s2_rows)
    _write_jsonl(expected_paths["val"], val_rows)
    _write_jsonl(expected_paths["test_celebdf"], celeb_test_rows)
    _write_jsonl(expected_paths["test_googledfd"], dfd_test_rows)
    _write_jsonl(expected_paths["test_inpaint"], inpaint_test_rows)
    _write_jsonl(expected_paths["test_avsplice"], av_test_rows)

    raw_512_u8.clear()
    decoded_wav.clear()
    gc.collect()

    meta_summary = {
        "counts": {
            "train_stage1": len(s1_rows),
            "train_stage2": len(s2_rows),
            "val": len(val_rows),
            "test_celebdf": len(celeb_test_rows),
            "test_googledfd": len(dfd_test_rows),
            "test_inpaint": len(inpaint_test_rows),
            "test_avsplice": len(av_test_rows),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    sentinel.write_text(json.dumps(meta_summary, indent=2), encoding="utf-8")
    print("[Data] Built real-world manifests:", json.dumps(meta_summary), flush=True)
    return expected_paths


@torch.inference_mode()
def evaluate_benchmark_detailed(
    model: HoSTForensicsModel,
    manifest_path: Path,
    cfg: HoSTConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a benchmark manifest including per-domain Cross-AUC, Spatial J&F, AV-mAP, and 4-way attribution."""
    ds = UnifiedForensicManifestDataset(
        manifest_path=manifest_path,
        num_frames=cfg.train.num_frames,
        img_size=cfg.train.highres_img_size,
        sample_rate=cfg.backbone.sample_rate,
        clip_duration_sec=cfg.train.clip_duration_sec,
    )
    loader = build_optimized_dataloader(
        dataset=ds,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        device=device,
        num_workers=0,
        pin_memory=False,
    )

    raw_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    regimes_gt = [r.get("regime", "clean") for r in raw_rows]

    all_labels: list[float] = []
    all_probs: list[float] = []
    all_domains: list[str] = []
    all_attr_preds: list[str] = []

    vis_preds_all: list[list[tuple[float, float, float]]] = []
    aud_preds_all: list[list[tuple[float, float, float]]] = []
    vis_gts_all: list[list[tuple[float, float]]] = []
    aud_gts_all: list[list[tuple[float, float]]] = []
    has_tfl = False

    pred_masks_list: list[np.ndarray] = []
    gt_masks_list: list[np.ndarray] = []

    model.eval()
    for batch in loader:
        video = batch["video"].to(device, dtype=torch.float32)
        audio = batch["audio"].to(device, dtype=torch.float32)
        has_audio = batch["has_audio"].to(device)
        targets = batch["targets"]

        out = model(video=video, audio=audio, has_audio=has_audio, duration_sec=batch["duration_sec"])
        probs = (
            out.calibrated_prob.detach().float().cpu().tolist()
            if out.calibrated_prob is not None
            else torch.sigmoid(out.video_logit.float()).detach().cpu().tolist()
        )
        all_labels.extend(targets.video_label.tolist())
        all_probs.extend(probs)
        all_domains.extend(batch["domains"])
        if out.attribution_pred is not None:
            all_attr_preds.extend(list(out.attribution_pred))

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
                if pm.ndim == 5 and pm.shape[2] == 1:
                    pm = pm.squeeze(2)
                gm = targets.gt_masks[valid_m].detach().cpu().numpy()
                if gm.ndim == 5 and gm.shape[2] == 1:
                    gm = gm.squeeze(2)
                pred_masks_list.append(pm)
                gt_masks_list.append(gm)

    # Multi-domain Cross-AUC inside this benchmark (hq vs h264_crf35 vs laundered)
    dom_buckets: dict[str, tuple[list[float], list[float]]] = {}
    for lbl, prb, dom in zip(all_labels, all_probs, all_domains):
        if dom not in dom_buckets:
            dom_buckets[dom] = ([], [])
        dom_buckets[dom][0].append(lbl)
        dom_buckets[dom][1].append(prb)
    valid_dom_buckets = {
        k: v for k, v in dom_buckets.items() if len(set(v[0])) == 2
    }
    dom_cross = compute_polarization_cross_auc(valid_dom_buckets) if len(valid_dom_buckets) >= 2 else {
        "cross_auc": compute_auroc(all_labels, all_probs),
        "polarization_w1": 0.0,
    }

    mid = compute_mid_ffd_metrics(all_labels, all_probs, threshold=0.5)
    gt_slice = regimes_gt[: len(all_attr_preds)]
    attr_acc = (
        round(100.0 * sum(int(p == g) for p, g in zip(all_attr_preds, gt_slice)) / max(len(gt_slice), 1), 2)
        if all_attr_preds
        else 0.0
    )

    res: dict[str, Any] = {
        "num_clips": len(all_labels),
        "auroc": round(100.0 * compute_auroc(all_labels, all_probs), 2),
        "eer": round(100.0 * compute_eer(all_labels, all_probs), 2),
        "domain_cross_auc": round(100.0 * dom_cross["cross_auc"], 2),
        "domain_pol_w1": round(100.0 * dom_cross["polarization_w1"], 2),
        "mid_acc": round(100.0 * mid["mid_acc"], 2),
        "mid_f1": round(100.0 * mid["mid_f1"], 2),
        "realized_far": round(100.0 * mid["realized_far"], 2),
        "attribution_4way_acc": attr_acc,
        "labels": all_labels,
        "probs": all_probs,
    }
    if has_tfl:
        tfl = compute_native_track_av_map(vis_preds_all, aud_preds_all, vis_gts_all, aud_gts_all)
        res["tfl"] = {k: round(100.0 * v, 2) for k, v in tfl.items()}
    if pred_masks_list:
        seg = compute_spatial_grounding_metrics(
            np.concatenate(pred_masks_list, axis=0),
            np.concatenate(gt_masks_list, axis=0),
        )
        res["spatial"] = {k: round(100.0 * v, 2) for k, v in seg.items()}
    return res


def _measure_lora_principal_leakage(model: HoSTForensicsModel) -> float:
    """Measure max Frobenius leakage ||U_k^T Delta W||_F across adapted LoRA projections."""
    max_leak = 0.0
    for mod in model.visual_backbone.modules():
        if isinstance(mod, OrthogonalSVDLinear):
            dw = mod.delta_weight_orthogonal().detach().float()
            leak = float(torch.linalg.norm(mod.u_k.T @ dw, ord="fro").item())
            max_leak = max(max_leak, leak)
    return max_leak


def _measure_moe_load_cv(model: HoSTForensicsModel) -> float:
    """Compute average coefficient of variation (std / mean) of expert loads across all pools."""
    cvs: list[float] = []
    for adapter in model.visual_backbone.moe_adapters.values():
        loads_dict = adapter.expert_load_summary()  # type: ignore[operator]
        for loads in loads_dict.values():
            arr = np.asarray(loads, dtype=np.float64)
            cvs.append(float(arr.std() / max(arr.mean(), 1e-8)))
    return round(float(np.mean(cvs)) if cvs else 0.0, 4)


def build_variants(base_cfg: HoSTConfig) -> list[tuple[str, str, HoSTConfig]]:
    """Construct the Full HoST-MoE configuration + 4 controlled single-change ablations."""
    variants: list[tuple[str, str, HoSTConfig]] = []

    # 0. Full HoST-MoE (Ours)
    c_full = copy.deepcopy(base_cfg)
    variants.append(("full_host_moe", "HoST-MoE (Full)", c_full))

    # 1. Ablation A1: Standard Unconstrained LoRA (W = W_0 + B A)
    c_a1 = copy.deepcopy(base_cfg)
    c_a1.backbone.use_ortho_lora = False
    variants.append(("abl_a1_standard_lora", "w/o Ortho SVD-LoRA (Std. LoRA)", c_a1))

    # 2. Ablation A2: No Auxiliary-Loss-Free Bias Balancing
    c_a2 = copy.deepcopy(base_cfg)
    c_a2.moe.bias_update_rate = 0.0
    variants.append(("abl_a2_no_bias_balance", "w/o Bias-Balanced MoE Routing", c_a2))

    # 3. Ablation A3: No Host Depolarization & Anchor Loss
    c_a3 = copy.deepcopy(base_cfg)
    c_a3.loss.lambda_polarization = 0.0
    c_a3.loss.lambda_anchor = 0.0
    variants.append(("abl_a3_no_depolarization", "w/o Host Depolarization & Anchor", c_a3))

    # 4. Ablation A4: No Symmetric Restoration/VAE Laundering
    c_a4 = copy.deepcopy(base_cfg)
    c_a4.train.symmetric_laundering_prob = 0.0
    variants.append(("abl_a4_no_sym_laundering", "w/o Symmetric Laundering", c_a4))

    return variants


def export_latex_pilot_table(results: dict[str, Any], out_tex: Path) -> None:
    """Write a clean booktabs LaTeX table summarizing all 5 variants across all 4 real benchmarks."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/run_real_hf_pilot.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{End-to-End Real-Media Foundation Pilot \& Five-Variant Ablation} using frozen \texttt{facebook/dino-vits16} + \texttt{microsoft/wavlm-base-plus} ($123.4\text{M}$ total, $7.20\text{M}$ [$5.83\%$] trainable) trained on $64$ identity-disjoint Stage-1 clips + $28$ Stage-2 clips (\texttt{Celeb-DF-v2}, \texttt{YouTube-real}, and cloned speech) and evaluated zero-shot under a single pristine-calibrated threshold across four multi-domain benchmarks (\texttt{HQ}, \texttt{H.264 CRF-35}, and \texttt{Restored/Laundered}).}",
        r"\label{tab:real_hf_pilot}",
        r"\vspace{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Model Variant} & \textbf{Google DFD (0-Shot)} & \textbf{AV-Splice Eval} & \textbf{Celeb-DF-v2} & \textbf{4-Bench Pooled} & \textbf{Pooled Realized} & \textbf{Principal Leak} & \textbf{MoE Expert} \\",
        r" & AUROC / Cross-AUC $\uparrow$ & AUROC / AV-mAP $\uparrow$ & AUROC / mIoU $\uparrow$ & MID-Acc (\%) $\uparrow$ & FAR (\%) $\downarrow$ & $\|\mathbf{U}_k^\top \Delta\mathbf{W}\|_F \downarrow$ & Load CV $\downarrow$ \\",
        r"\midrule",
    ]

    variant_list = results["variants"]
    for idx, item in enumerate(variant_list):
        label = str(item["label"]).replace("&", r"\&")
        if idx == 0:
            label = r"\textbf{" + label + "}"
        b_cel = item["benchmarks"]["CelebDF_v2"]["auroc"]
        b_cel_iou = item["benchmarks"]["CelebDF_v2"]["spatial"]["miou"]
        b_dfd = item["benchmarks"]["GoogleDFD_ZeroShot"]["auroc"]
        b_dfd_x = item["benchmarks"]["GoogleDFD_ZeroShot"]["domain_cross_auc"]
        b_av_auc = item["benchmarks"]["AV_Splice_Eval"]["auroc"]
        av_map = item["benchmarks"]["AV_Splice_Eval"]["tfl"]["av_map"]
        m_acc = item["cross_benchmark"]["pooled_mid_acc"]
        p_far = item["cross_benchmark"]["pooled_realized_far"]
        leak = item["lora_principal_leakage_fro"]
        moe_cv = item["moe_expert_load_cv"]

        row_str = (
            f"{label} & {b_dfd:.1f} / {b_dfd_x:.1f} & {b_av_auc:.1f} / {av_map:.1f} & "
            f"{b_cel:.1f} / {b_cel_iou:.1f} & {m_acc:.1f} & {p_far:.1f} & "
            f"{leak:.4f} & {moe_cv:.4f} \\\\"
        )
        lines.append(row_str)
        if idx == 0:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\vspace{-4pt}", r"\end{table*}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _warm_or_load_wavlm_cache(model: HoSTForensicsModel, data_root: Path, device: torch.device) -> None:
    """Load or pre-populate persistent frozen WavLM feature cache across all `.pt` audio clips."""
    from host_forensics.models.backbones import _WAVLM_FROZEN_CACHE

    cache_file = data_root / "wavlm_frozen_cache.pt"
    if cache_file.exists() and len(_WAVLM_FROZEN_CACHE) == 0:
        saved_dict = torch.load(cache_file, map_location="cpu", weights_only=False)
        _WAVLM_FROZEN_CACHE.update(saved_dict)
        print(f"[Cache] Loaded {len(_WAVLM_FROZEN_CACHE)} frozen WavLM clip embeddings from {cache_file}", flush=True)
        return

    aud_files = sorted((data_root / "tensors").glob("aud_*.pt"))
    if not aud_files:
        return
    t0 = time.time()
    zero_len = int(round(model.cfg.backbone.sample_rate * model.cfg.train.clip_duration_sec))
    batch_wavs: list[torch.Tensor] = [torch.zeros(zero_len, dtype=torch.float32)]
    model.audio_backbone.eval()
    with torch.inference_mode():
        for af in aud_files:
            batch_wavs.append(torch.load(af, map_location="cpu", weights_only=True).float())
            if len(batch_wavs) == 16:
                _ = model.audio_backbone(torch.stack(batch_wavs, dim=0).to(device))
                batch_wavs.clear()
        if batch_wavs:
            _ = model.audio_backbone(torch.stack(batch_wavs, dim=0).to(device))
            batch_wavs.clear()
    torch.save(dict(_WAVLM_FROZEN_CACHE), cache_file)
    print(
        f"[Cache] Pre-computed & saved {len(_WAVLM_FROZEN_CACHE)} frozen WavLM embeddings in {time.time() - t0:.1f}s",
        flush=True,
    )


def main() -> None:
    import argparse
    import shutil
    from host_forensics.models.backbones import _WAVLM_FROZEN_CACHE

    parser = argparse.ArgumentParser(description="Run real HF foundation pilot & 5-variant ablation.")
    parser.add_argument(
        "--variant",
        type=int,
        default=None,
        help="Optional 0-based variant index (0..4) to execute in a single bounded step.",
    )
    args = parser.parse_args()

    base_cfg = HoSTConfig.from_yaml("configs/hf_real_pilot.yaml")
    # Run 6 Stage-1 epochs (192px) + 2 Stage-2 high-res epochs (224px) per variant
    base_cfg.train.epochs_stage1 = 6
    base_cfg.train.epochs_stage2_highres = 2
    base_cfg.train.lr = 5.0e-4

    data_root = Path(os.environ.get("PILOT_DATA_ROOT", "/tmp/hf_pilot_data"))
    out_json = Path("paper/figures/real_hf_pilot_results.json")
    out_tex = Path("paper/tables/real_hf_pilot_table.tex")
    ckpt_root = Path("/tmp/hf_pilot_checkpoints")

    manifests = build_real_pilot_manifests(
        root_dir=data_root,
        num_frames=base_cfg.train.num_frames,
        base_img_size=base_cfg.train.highres_img_size,
        sample_rate=base_cfg.backbone.sample_rate,
        duration_sec=base_cfg.train.clip_duration_sec,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = build_variants(base_cfg)

    existing_by_slug: dict[str, dict[str, Any]] = {}
    if out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        for item in prev.get("variants", []):
            existing_by_slug[item["slug"]] = item

    for v_zero_idx, (slug, label, cfg) in enumerate(variants):
        if args.variant is not None and v_zero_idx != args.variant:
            continue

        v_idx = v_zero_idx + 1
        v_t0 = time.time()
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        print(f"\n==================== [{v_idx}/{len(variants)}] {label} ({slug}) ====================", flush=True)
        model = HoSTForensicsModel(cfg)
        _warm_or_load_wavlm_cache(model, data_root, device)

        # Re-seed right after cache warm-up so training RNG is 100% deterministic across variants
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        param_info = model.parameter_summary()
        cur_out = ckpt_root / slug
        cal_ckpt = cur_out / "host_moe_calibrated.pt"

        trainer: HoSTTrainer | None = None
        if cal_ckpt.exists():
            print(f"[Resume] Loading pre-trained calibrated checkpoint {cal_ckpt}...", flush=True)
            ckpt_obj = torch.load(cal_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_obj["model_state_dict"], strict=True)
            model.to(device)
            cal_info = ckpt_obj.get("calibration", {})
            final_loss: dict[str, float] = {}
        else:
            augment = CompoundDegradationAndLaundering(
                degradation_prob=cfg.train.compound_degradation_prob,
                symmetric_laundering_prob=cfg.train.symmetric_laundering_prob,
            )
            s1_ds = UnifiedForensicManifestDataset(
                manifest_path=manifests["train_stage1"],
                num_frames=cfg.train.num_frames,
                img_size=cfg.backbone.img_size,
                sample_rate=cfg.backbone.sample_rate,
                clip_duration_sec=cfg.train.clip_duration_sec,
                augment=augment,
            )
            s2_ds = UnifiedForensicManifestDataset(
                manifest_path=manifests["train_stage2"],
                num_frames=cfg.train.num_frames,
                img_size=cfg.train.highres_img_size,
                sample_rate=cfg.backbone.sample_rate,
                clip_duration_sec=cfg.train.clip_duration_sec,
                augment=augment,
            )
            val_ds = UnifiedForensicManifestDataset(
                manifest_path=manifests["val"],
                num_frames=cfg.train.num_frames,
                img_size=cfg.train.highres_img_size,
                sample_rate=cfg.backbone.sample_rate,
                clip_duration_sec=cfg.train.clip_duration_sec,
                augment=None,
            )
            s1_loader = build_optimized_dataloader(
                s1_ds, batch_size=cfg.train.batch_size, shuffle=True, device=device, num_workers=0, pin_memory=False
            )
            s2_loader = build_optimized_dataloader(
                s2_ds, batch_size=cfg.train.batch_size, shuffle=True, device=device, num_workers=0, pin_memory=False
            )
            val_loader = build_optimized_dataloader(
                val_ds, batch_size=cfg.train.eval_batch_size, shuffle=False, device=device, num_workers=0, pin_memory=False
            )
            trainer = HoSTTrainer(model=model, cfg=cfg, device=device)
            train_res = trainer.run_two_stage_curriculum(
                stage1_loader=s1_loader,
                val_loader=val_loader,
                stage2_loader=s2_loader,
                output_dir=str(cur_out),
                resume_stage1=True,
            )
            cal_info = train_res["calibration"]
            final_loss = {
                k: round(v, 4) for k, v in train_res["history"][-1].items() if isinstance(v, (int, float))
            }

        lora_leak = _measure_lora_principal_leakage(model)
        moe_cv = _measure_moe_load_cv(model)

        # Evaluate across all 4 held-out benchmarks with fixed pristine-calibrated thresholds
        bench_specs = [
            ("CelebDF_v2", manifests["test_celebdf"]),
            ("GoogleDFD_ZeroShot", manifests["test_googledfd"]),
            ("InpaintTubelet_Silent", manifests["test_inpaint"]),
            ("AV_Splice_Eval", manifests["test_avsplice"]),
        ]
        per_bench: dict[str, Any] = {}
        bench_pairs: dict[str, tuple[list[float], list[float]]] = {}
        pooled_labels: list[float] = []
        pooled_probs: list[float] = []

        for b_name, b_path in bench_specs:
            b_res = evaluate_benchmark_detailed(model, b_path, cfg, device)
            lbls = b_res.pop("labels")
            prbs = b_res.pop("probs")
            bench_pairs[b_name] = (lbls, prbs)
            pooled_labels.extend(lbls)
            pooled_probs.extend(prbs)
            per_bench[b_name] = b_res

        # Persist any newly added eval audio embeddings to disk cache
        cache_file = data_root / "wavlm_frozen_cache.pt"
        if len(_WAVLM_FROZEN_CACHE) > 0:
            torch.save(dict(_WAVLM_FROZEN_CACHE), cache_file)

        cross_bench = compute_polarization_cross_auc(bench_pairs)
        pooled_mid = compute_mid_ffd_metrics(pooled_labels, pooled_probs, threshold=0.5)
        elapsed_v = round(time.time() - v_t0, 1)

        variant_summary = {
            "slug": slug,
            "label": label,
            "elapsed_sec": elapsed_v,
            "parameters": param_info,
            "lora_principal_leakage_fro": round(lora_leak, 6),
            "moe_expert_load_cv": moe_cv,
            "calibration": {k: round(v, 4) for k, v in cal_info.items()},
            "final_epoch_loss": final_loss,
            "benchmarks": per_bench,
            "cross_benchmark": {
                "mean_within_auc": round(100.0 * cross_bench["mean_within_auc"], 2),
                "pooled_mixed_auc": round(100.0 * cross_bench["pooled_mixed_auc"], 2),
                "cross_auc": round(100.0 * cross_bench["cross_auc"], 2),
                "polarization_w1": round(100.0 * cross_bench["polarization_w1"], 2),
                "pooled_mid_acc": round(100.0 * pooled_mid["mid_acc"], 2),
                "pooled_realized_far": round(100.0 * pooled_mid["realized_far"], 2),
            },
        }
        existing_by_slug[slug] = variant_summary
        ordered_variants = [existing_by_slug[s] for s, _, _ in variants if s in existing_by_slug]

        all_results = {
            "backbone_visual": base_cfg.backbone.visual_name,
            "backbone_audio": base_cfg.backbone.audio_name,
            "device": str(device),
            "variants": ordered_variants,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

        print(
            f"[Done {slug} in {elapsed_v}s] "
            f"CelebDF={per_bench['CelebDF_v2']['auroc']}% | "
            f"GoogleDFD(0-shot)={per_bench['GoogleDFD_ZeroShot']['auroc']}% (CrossAUC={per_bench['GoogleDFD_ZeroShot']['domain_cross_auc']}%) | "
            f"PooledCrossAUC={variant_summary['cross_benchmark']['cross_auc']}% | "
            f"MID-Acc={variant_summary['cross_benchmark']['pooled_mid_acc']}% | "
            f"Inpaint J&F={per_bench['InpaintTubelet_Silent']['spatial']['j_and_f']}% | "
            f"AV-mAP={per_bench['AV_Splice_Eval']['tfl']['av_map']}% | "
            f"4WayAttr={per_bench['AV_Splice_Eval']['attribution_4way_acc']}%",
            flush=True,
        )

        # Keep full_host_moe calibrated checkpoint; clean up ablation checkpoints to save disk
        if slug == "full_host_moe":
            s1_p = cur_out / "host_moe_stage1.pt"
            if s1_p.exists():
                s1_p.unlink()
        else:
            shutil.rmtree(cur_out, ignore_errors=True)

        del trainer, model
        gc.collect()

    ordered_final = [existing_by_slug[s] for s, _, _ in variants if s in existing_by_slug]
    if len(ordered_final) == len(variants):
        final_payload = {
            "backbone_visual": base_cfg.backbone.visual_name,
            "backbone_audio": base_cfg.backbone.audio_name,
            "device": str(device),
            "variants": ordered_final,
        }
        export_latex_pilot_table(final_payload, out_tex)
        print(f"\n[COMPLETE] Wrote {out_json} and {out_tex} ({len(ordered_final)} variants)", flush=True)


if __name__ == "__main__":
    main()
