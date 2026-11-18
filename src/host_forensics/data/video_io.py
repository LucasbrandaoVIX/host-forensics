"""Zero-extra-dependency FFmpeg/FFprobe video, 16 kHz audio, and spatial mask decoding utilities.

Enables direct ingestion of raw `.mp4`/`.mkv`/`.webm`/`.mov` files (H.264, H.265/HEVC, AV1, VP9)
for dataset manifest preparation and direct CLI inference (`host-detect infer --video clip.mp4`).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def probe_media_info(video_path: str | Path) -> dict[str, Any]:
    """Inspect container duration, resolution, frame count, and audio stream presence via `ffprobe`."""
    path_str = str(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        path_str,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout or "{}")

    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    if not has_video:
        raise ValueError(f"No video stream found in {video_path}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    v_stream = next(s for s in streams if s.get("codec_type") == "video")
    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))

    raw_dur = info.get("format", {}).get("duration")
    duration_sec = float(raw_dur) if raw_dur is not None else 4.0

    return {
        "duration_sec": max(duration_sec, 0.1),
        "has_audio": has_audio,
        "width": width,
        "height": height,
    }


def extract_video_and_audio_ffmpeg(
    video_path: str | Path,
    num_frames: int = 16,
    img_size: int = 384,
    sample_rate: int = 16000,
    max_duration_sec: float | None = 4.0,
    normalize_imagenet: bool = True,
) -> dict[str, Any]:
    """Decode uniformly sampled RGB frames `(T, 3, H, W)` and 16 kHz mono audio `(L_a,)` via FFmpeg."""
    meta = probe_media_info(video_path)
    src_dur = float(meta["duration_sec"])
    clip_dur = min(src_dur, max_duration_sec) if max_duration_sec is not None else src_dur
    clip_dur = max(clip_dur, 0.25)

    fps_target = num_frames / clip_dur
    vf_filter = f"fps={fps_target:.6f},scale={img_size}:{img_size}:flags=bicubic"

    v_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        f"{clip_dur:.4f}",
        "-i",
        str(video_path),
        "-vf",
        vf_filter,
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    v_proc = subprocess.run(v_cmd, capture_output=True, check=True)
    frame_bytes = img_size * img_size * 3
    raw_arr = np.frombuffer(v_proc.stdout, dtype=np.uint8)
    decoded_frames = len(raw_arr) // frame_bytes
    if decoded_frames == 0:
        raise RuntimeError(f"FFmpeg decoded 0 frames from {video_path}")

    raw_arr = raw_arr[: decoded_frames * frame_bytes].reshape(decoded_frames, img_size, img_size, 3)
    video = torch.from_numpy(raw_arr.copy()).permute(0, 3, 1, 2).float() / 255.0

    # Pad by repeating final frame if short container produced < num_frames
    if video.shape[0] < num_frames:
        pad_n = num_frames - video.shape[0]
        video = torch.cat([video, video[-1:].repeat(pad_n, 1, 1, 1)], dim=0)
    elif video.shape[0] > num_frames:
        video = video[:num_frames]

    if normalize_imagenet:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        video = (video - mean) / std

    target_audio_len = int(round(sample_rate * clip_dur))
    has_audio = bool(meta["has_audio"])
    if has_audio:
        a_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-t",
            f"{clip_dur:.4f}",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        a_proc = subprocess.run(a_cmd, capture_output=True, check=False)
        if a_proc.returncode == 0 and len(a_proc.stdout) >= 4:
            aud_np = np.frombuffer(a_proc.stdout, dtype=np.float32).copy()
            audio = torch.from_numpy(aud_np)
            if audio.numel() < target_audio_len:
                audio = F.pad(audio, (0, target_audio_len - audio.numel()))
            else:
                audio = audio[:target_audio_len]
        else:
            has_audio = False
            audio = torch.zeros(target_audio_len, dtype=torch.float32)
    else:
        audio = torch.zeros(target_audio_len, dtype=torch.float32)

    return {
        "video": video,
        "audio": audio,
        "has_audio": has_audio,
        "duration_sec": clip_dur,
    }


def extract_mask_sequence_ffmpeg(
    mask_path: str | Path,
    num_frames: int = 16,
    img_size: int = 384,
    clip_duration_sec: float = 4.0,
) -> torch.Tensor:
    """Load a ground-truth spatial manipulation mask from `.pt`, `.npy`, video, or PNG directory."""
    p = Path(mask_path)
    if p.suffix == ".pt":
        m = torch.load(p, map_location="cpu", weights_only=True).float()
        if m.ndim == 2:
            m = m.unsqueeze(0).repeat(num_frames, 1, 1)
        if m.shape[-2:] != (img_size, img_size):
            m = F.interpolate(m.unsqueeze(1), size=(img_size, img_size), mode="nearest").squeeze(1)
        if m.shape[0] != num_frames:
            idx = torch.linspace(0, m.shape[0] - 1, num_frames).round().long()
            m = m[idx]
        return (m > 0.5).float()

    if p.suffix == ".npy":
        m = torch.from_numpy(np.load(p)).float()
        if m.ndim == 2:
            m = m.unsqueeze(0).repeat(num_frames, 1, 1)
        if m.shape[-2:] != (img_size, img_size):
            m = F.interpolate(m.unsqueeze(1), size=(img_size, img_size), mode="nearest").squeeze(1)
        if m.shape[0] != num_frames:
            idx = torch.linspace(0, m.shape[0] - 1, num_frames).round().long()
            m = m[idx]
        return (m > 0.5).float()

    # Decode mask video or image via FFmpeg single-channel gray8
    fps_target = num_frames / max(clip_duration_sec, 0.25)
    vf_filter = f"fps={fps_target:.6f},scale={img_size}:{img_size}:flags=neighbor"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        f"{clip_duration_sec:.4f}",
        "-i",
        str(p),
        "-vf",
        vf_filter,
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    frame_bytes = img_size * img_size
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    n_dec = max(1, len(arr) // frame_bytes)
    arr = arr[: n_dec * frame_bytes].reshape(n_dec, img_size, img_size)
    mask = (torch.from_numpy(arr.copy()).float() / 255.0 > 0.5).float()
    if mask.shape[0] < num_frames:
        mask = torch.cat([mask, mask[-1:].repeat(num_frames - mask.shape[0], 1, 1)], dim=0)
    return mask[:num_frames]


def synthesize_test_mp4_ffmpeg(
    out_path: str | Path,
    duration_sec: float = 2.0,
    fps: int = 8,
    size: int = 128,
    with_audio: bool = True,
) -> Path:
    """Create a valid H.264/MPEG-4 `.mp4` clip via `ffmpeg` lavfi for testing & CLI verification."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}x{size}:rate={fps}:duration={duration_sec}",
    ]
    if with_audio:
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=16000:duration={duration_sec}",
                "-c:a",
                "aac",
            ]
        )
    cmd.extend(["-pix_fmt", "yuv420p", str(out_p)])
    subprocess.run(cmd, check=True)
    return out_p
