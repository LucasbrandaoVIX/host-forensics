"""Command-line interface (`host-detect`) for HoST-MoE training, calibration, inspection, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

from host_forensics.config import BackboneConfig, HoSTConfig, MNStemConfig, ReadoutConfig, TrainConfig, TriSTMoEConfig
from host_forensics.data.datasets import (
    SyntheticManipulatedVideoDataset,
    UnifiedForensicManifestDataset,
    forensic_collate_fn,
)
from host_forensics.data.transforms import CompoundDegradationAndLaundering
from host_forensics.engine.trainer import HoSTTrainer
from host_forensics.models.host_model import HoSTForensicsModel


def _build_compact_smoke_config() -> HoSTConfig:
    """Compact stand-in configuration for instant CPU/CI smoke verification."""
    return HoSTConfig(
        backbone=BackboneConfig(
            visual_name="standin-vit-compact",
            visual_backend="standin",
            img_size=128,
            patch_size=16,
            visual_embed_dim=128,
            visual_num_layers=4,
            tap_layers=(1, 2, 3, 4),
            adapted_layers=(2, 4),
            use_ortho_lora=True,
            lora_rank=8,
            principal_subspace_rank=16,
            audio_embed_dim=128,
            audio_stride_sec=0.25,
            sample_rate=16000,
        ),
        mn_stem=MNStemConfig(
            bayar_channels=8,
            srm_channels=9,
            shallow_feat_channels=24,
            corr_radius=1,
            out_dim=64,
        ),
        moe=TriSTMoEConfig(
            bottleneck_dim=64,
            num_experts_per_pool=4,
            top_k=2,
            num_heads=4,
            dropout=0.0,
            bias_update_rate=1e-3,
        ),
        readout=ReadoutConfig(
            hidden_dim=128,
            pyramid_levels=3,
            mask_base_channels=64,
            default_far_budget=0.05,
        ),
        train=TrainConfig(
            precision="fp32",
            batch_size=4,
            grad_accum_steps=1,
            num_frames=4,
            clip_duration_sec=2.0,
            lr=1e-3,
            epochs_stage1=1,
            epochs_stage2_highres=0,
        ),
    )


def _load_or_default_config(config_path: str | None, compact_default: bool = False) -> HoSTConfig:
    if config_path:
        return HoSTConfig.from_yaml(config_path)
    if compact_default:
        return _build_compact_smoke_config()
    return HoSTConfig()


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _load_or_default_config(args.config, compact_default=args.compact)
    model = HoSTForensicsModel(cfg)
    summary = model.parameter_summary()

    breakdown: dict[str, dict[str, int]] = {}
    for name, module in model.named_children():
        tot = sum(p.numel() for p in module.parameters())
        trn = sum(p.numel() for p in module.parameters() if p.requires_grad)
        breakdown[name] = {"total": tot, "trainable": trn}

    payload = {
        "visual_backbone": cfg.backbone.visual_name,
        "visual_backend": cfg.backbone.visual_backend,
        "img_size": cfg.backbone.img_size,
        "adapted_layers": list(cfg.backbone.adapted_layers),
        "summary": summary,
        "modules": breakdown,
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_dump_config(args: argparse.Namespace) -> int:
    cfg = _build_compact_smoke_config() if args.compact else HoSTConfig()
    cfg.to_yaml(args.output)
    print(f"Wrote configuration to {args.output}")
    return 0


def cmd_smoke_train(args: argparse.Namespace) -> int:
    cfg = _load_or_default_config(args.config, compact_default=True)
    torch.manual_seed(42)

    augment = CompoundDegradationAndLaundering(
        degradation_prob=cfg.train.compound_degradation_prob,
        symmetric_laundering_prob=cfg.train.symmetric_laundering_prob,
    )
    train_ds = SyntheticManipulatedVideoDataset(
        length=args.num_samples,
        num_frames=cfg.train.num_frames,
        img_size=cfg.backbone.img_size,
        clip_duration_sec=cfg.train.clip_duration_sec,
        sample_rate=cfg.backbone.sample_rate,
        augment=augment,
        seed=42,
    )
    val_ds = SyntheticManipulatedVideoDataset(
        length=max(8, args.num_samples // 2),
        num_frames=cfg.train.num_frames,
        img_size=cfg.backbone.img_size,
        clip_duration_sec=cfg.train.clip_duration_sec,
        sample_rate=cfg.backbone.sample_rate,
        augment=None,
        seed=999,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=forensic_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=forensic_collate_fn,
    )

    model = HoSTForensicsModel(cfg)
    trainer = HoSTTrainer(model=model, cfg=cfg, device=args.device)

    metrics_history: list[dict[str, float]] = []
    for _ in range(args.epochs):
        epoch_losses = trainer.train_one_epoch(train_loader)
        metrics_history.append({k: round(v, 5) for k, v in epoch_losses.items()})

    cal_info = trainer.calibrate_on_pristine(val_loader)

    # Run structured inference on a 4-sample batch covering all 4 synthetic regimes
    demo_batch = next(iter(val_loader))
    preds = model.predict_structured(
        video=demo_batch["video"].to(trainer.device),
        audio=demo_batch["audio"].to(trainer.device),
        has_audio=demo_batch["has_audio"].to(trainer.device),
        duration_sec=demo_batch["duration_sec"],
    )

    first_adapter_key = sorted(model.visual_backbone.moe_adapters.keys())[0]
    moe_loads = model.visual_backbone.moe_adapters[first_adapter_key].expert_load_summary()  # type: ignore[operator]

    report = {
        "status": "ok",
        "device": str(trainer.device),
        "parameter_summary": model.parameter_summary(),
        "train_epochs": metrics_history,
        "pristine_calibration": {k: round(v, 5) for k, v in cal_info.items()},
        "sample_moe_load_layer_" + first_adapter_key: {
            pool: [round(x, 4) for x in loads] for pool, loads in moe_loads.items()
        },
        "demo_predictions": preds[:4],
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load_or_default_config(args.config, compact_default=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    augment = CompoundDegradationAndLaundering(
        degradation_prob=cfg.train.compound_degradation_prob,
        symmetric_laundering_prob=cfg.train.symmetric_laundering_prob,
    )
    train_ds = UnifiedForensicManifestDataset(
        manifest_path=args.train_manifest,
        num_frames=cfg.train.num_frames,
        img_size=cfg.backbone.img_size,
        sample_rate=cfg.backbone.sample_rate,
        clip_duration_sec=cfg.train.clip_duration_sec,
        augment=augment,
    )
    val_ds = UnifiedForensicManifestDataset(
        manifest_path=args.val_manifest,
        num_frames=cfg.train.num_frames,
        img_size=cfg.backbone.img_size,
        sample_rate=cfg.backbone.sample_rate,
        clip_duration_sec=cfg.train.clip_duration_sec,
        augment=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=forensic_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=forensic_collate_fn,
    )

    stage2_loader: DataLoader[dict[str, object]] | None = None
    if args.stage2_manifest:
        stage2_ds = UnifiedForensicManifestDataset(
            manifest_path=args.stage2_manifest,
            num_frames=cfg.train.num_frames,
            img_size=cfg.train.highres_img_size,
            sample_rate=cfg.backbone.sample_rate,
            clip_duration_sec=cfg.train.clip_duration_sec,
            augment=augment,
        )
        stage2_loader = DataLoader(
            stage2_ds,
            batch_size=max(1, cfg.train.batch_size // 2),
            shuffle=True,
            collate_fn=forensic_collate_fn,
        )

    if args.epochs is not None:
        cfg.train.epochs_stage1 = args.epochs

    model = HoSTForensicsModel(cfg)
    trainer = HoSTTrainer(model=model, cfg=cfg, device=args.device)
    result = trainer.run_two_stage_curriculum(
        stage1_loader=train_loader,
        val_loader=val_loader,
        stage2_loader=stage2_loader,
        output_dir=str(out_dir),
    )
    print(
        json.dumps(
            {
                "saved_checkpoint": result["checkpoint"],
                "epochs_completed": len(result["history"]),
                "calibration": result["calibration"],
            },
            indent=2,
        )
    )
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from host_forensics.data.video_io import extract_video_and_audio_ffmpeg

    cfg = _load_or_default_config(args.config, compact_default=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HoSTForensicsModel(cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

    audio: torch.Tensor | None = None
    has_audio: torch.Tensor | None = None
    clip_dur = args.duration_sec or cfg.train.clip_duration_sec

    if args.video:
        decoded = extract_video_and_audio_ffmpeg(
            video_path=args.video,
            num_frames=cfg.train.num_frames,
            img_size=cfg.backbone.img_size,
            sample_rate=cfg.backbone.sample_rate,
            max_duration_sec=args.duration_sec,
        )
        video = decoded["video"].unsqueeze(0).to(device)
        clip_dur = float(decoded["duration_sec"])
        if not args.silent and bool(decoded["has_audio"]):
            audio = decoded["audio"].unsqueeze(0).to(device)
            has_audio = torch.ones(1, dtype=torch.bool, device=device)
        else:
            has_audio = torch.zeros(1, dtype=torch.bool, device=device)
    elif args.video_pt:
        video = torch.load(args.video_pt, map_location=device, weights_only=True).float()
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if args.silent:
            has_audio = torch.zeros(video.shape[0], dtype=torch.bool, device=device)
        elif args.audio_pt:
            audio = torch.load(args.audio_pt, map_location=device, weights_only=True).float()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            has_audio = torch.ones(video.shape[0], dtype=torch.bool, device=device)
    else:
        ds = SyntheticManipulatedVideoDataset(
            length=2,
            num_frames=cfg.train.num_frames,
            img_size=cfg.backbone.img_size,
            clip_duration_sec=cfg.train.clip_duration_sec,
            sample_rate=cfg.backbone.sample_rate,
        )
        sample = ds[1]
        video = sample["video"].unsqueeze(0).to(device)
        if args.silent:
            has_audio = torch.zeros(1, dtype=torch.bool, device=device)

    preds = model.predict_structured(
        video=video,
        audio=audio,
        has_audio=has_audio,
        duration_sec=clip_dur,
    )
    print(json.dumps(preds, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="host-detect",
        description="HoST-MoE: Unified AI-Manipulated Video Detection & Spatio-Temporal Localization CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Print parameter counts, trainable ratio, and module breakdown")
    p_info.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p_info.add_argument("--compact", action="store_true", help="Use compact smoke-test config")
    p_info.set_defaults(func=cmd_info)

    p_dump = sub.add_parser("dump-config", help="Write default YAML config to disk")
    p_dump.add_argument("--output", type=str, required=True, help="Destination YAML file path")
    p_dump.add_argument("--compact", action="store_true", help="Write compact CI/CPU config instead of RTX 5090 full config")
    p_dump.set_defaults(func=cmd_dump_config)

    p_smoke = sub.add_parser(
        "smoke-train",
        help="Run end-to-end synthetic training, MoE bias balancing, pristine calibration, and structured inference",
    )
    p_smoke.add_argument("--config", type=str, default=None, help="Optional YAML config (defaults to compact config)")
    p_smoke.add_argument("--epochs", type=int, default=2, help="Number of smoke training epochs")
    p_smoke.add_argument("--num-samples", type=int, default=12, help="Number of synthetic clips")
    p_smoke.add_argument("--device", type=str, default=None, help="Torch device (e.g. cpu, cuda)")
    p_smoke.set_defaults(func=cmd_smoke_train)

    p_train = sub.add_parser("train", help="Train HoST-MoE on JSONL manifests and calibrate on pristine validation clips")
    p_train.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p_train.add_argument("--train-manifest", type=str, required=True, help="Path to Stage-1 training JSONL manifest")
    p_train.add_argument("--val-manifest", type=str, required=True, help="Path to validation JSONL manifest")
    p_train.add_argument("--stage2-manifest", type=str, default=None, help="Optional high-res Stage-2 mask/TFL manifest")
    p_train.add_argument("--output-dir", type=str, default="checkpoints", help="Checkpoint output directory")
    p_train.add_argument("--epochs", type=int, default=None, help="Override stage-1 epoch count")
    p_train.add_argument("--device", type=str, default=None, help="Torch device")
    p_train.set_defaults(func=cmd_train)

    p_infer = sub.add_parser("infer", help="Run calibrated manipulation detection & localization on a clip")
    p_infer.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p_infer.add_argument("--checkpoint", type=str, default=None, help="Path to calibrated checkpoint (.pt)")
    p_infer.add_argument("--video", type=str, default=None, help="Path to raw video file (.mp4/.mkv/.webm/.mov)")
    p_infer.add_argument("--video-pt", type=str, default=None, help="Path to pre-extracted (T, 3, H, W) tensor (.pt)")
    p_infer.add_argument("--audio-pt", type=str, default=None, help="Path to pre-extracted (L_a,) 16kHz audio tensor (.pt)")
    p_infer.add_argument("--silent", action="store_true", help="Force graceful visual-only Silent-Prior fallback")
    p_infer.add_argument("--duration-sec", type=float, default=None, help="Clip duration in seconds")
    p_infer.add_argument("--device", type=str, default=None, help="Torch device")
    p_infer.set_defaults(func=cmd_infer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
