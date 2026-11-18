"""End-to-end integration test covering real FFmpeg .mp4 decoding, manifest preparation,
two-stage curriculum training, zero-shot benchmark evaluation, and LaTeX row export."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from host_forensics.cli import _build_compact_smoke_config, main as cli_main
from host_forensics.data.video_io import extract_video_and_audio_ffmpeg, synthesize_test_mp4_ffmpeg


def test_real_mp4_manifest_train_eval_and_infer_pipeline(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_bench"
    real_mp4 = synthesize_test_mp4_ffmpeg(raw_root / "real" / "pristine_01.mp4", duration_sec=1.0, fps=4, size=128, with_audio=True)
    fake_mp4 = synthesize_test_mp4_ffmpeg(raw_root / "fake" / "forged_01.mp4", duration_sec=1.0, fps=4, size=128, with_audio=False)

    # 1. Verify direct FFmpeg extraction on both audio-visual and silent .mp4 containers
    dec_real = extract_video_and_audio_ffmpeg(real_mp4, num_frames=4, img_size=128, max_duration_sec=1.0)
    dec_fake = extract_video_and_audio_ffmpeg(fake_mp4, num_frames=4, img_size=128, max_duration_sec=1.0)
    assert dec_real["video"].shape == (4, 3, 128, 128)
    assert dec_real["has_audio"] is True
    assert dec_fake["has_audio"] is False

    # 2. Run scripts/prepare_manifests.py on raw_root
    repo_root = Path(__file__).resolve().parents[1]
    manifest_jsonl = tmp_path / "manifests" / "mini_bench.jsonl"
    tensor_dir = tmp_path / "tensors"
    prep_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_manifests.py"),
        "--dataset-type",
        "generic",
        "--input-root",
        str(raw_root),
        "--output-manifest",
        str(manifest_jsonl),
        "--tensor-dir",
        str(tensor_dir),
        "--num-frames",
        "4",
        "--img-size",
        "128",
        "--max-duration-sec",
        "1.0",
        "--workers",
        "1",
    ]
    subprocess.run(prep_cmd, check=True)
    assert manifest_jsonl.exists()

    # 3. Run two-stage curriculum (`host-detect train` with Stage-1 + Stage-2)
    cfg = _build_compact_smoke_config()
    cfg.train.epochs_stage1 = 1
    cfg.train.epochs_stage2_highres = 1
    cfg.train.highres_img_size = 128
    cfg.train.clip_duration_sec = 1.0
    cfg_path = tmp_path / "mini_cfg.yaml"
    cfg.to_yaml(cfg_path)

    ckpt_dir = tmp_path / "ckpts"
    rc_train = cli_main(
        [
            "train",
            "--config",
            str(cfg_path),
            "--train-manifest",
            str(manifest_jsonl),
            "--stage2-manifest",
            str(manifest_jsonl),
            "--val-manifest",
            str(manifest_jsonl),
            "--output-dir",
            str(ckpt_dir),
            "--device",
            "cpu",
        ]
    )
    assert rc_train == 0
    cal_ckpt = ckpt_dir / "host_moe_calibrated.pt"
    assert cal_ckpt.exists()

    # 4. Run direct CLI inference on raw .mp4 file
    rc_infer = cli_main(
        [
            "infer",
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(cal_ckpt),
            "--video",
            str(real_mp4),
            "--device",
            "cpu",
        ]
    )
    assert rc_infer == 0

    # 5. Run scripts/evaluate_benchmarks.py and verify JSON + LaTeX booktabs export
    eval_json = tmp_path / "eval_out.json"
    latex_tex = tmp_path / "row.tex"
    eval_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate_benchmarks.py"),
        "--config",
        str(cfg_path),
        "--checkpoint",
        str(cal_ckpt),
        "--benchmark",
        f"MiniBench={manifest_jsonl}",
        "--output-json",
        str(eval_json),
        "--latex-out",
        str(latex_tex),
        "--device",
        "cpu",
    ]
    subprocess.run(eval_cmd, check=True)
    summary = json.loads(eval_json.read_text(encoding="utf-8"))
    assert "MiniBench" in summary["benchmarks"]
    assert "cross_auc" in summary["cross_domain_calibration"]
    assert "HoST-MoE" in latex_tex.read_text(encoding="utf-8")
