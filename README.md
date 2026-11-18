# HoST-MoE: Host-Differential Spatio-Temporal Mixture of Experts for AI-Manipulated Video Forensics

**HoST-MoE** is a unified, parameter-efficient, open-source PyTorch framework for **detecting, attributing, and localizing AI manipulations inside real camera-captured videos**—spanning facial identity swaps, portrait reenactment, lip-sync dubbing, localized diffusion video editing, object removal/inpainting, and sparse audio-visual splices.

Unlike whole-video synthetic T2V detectors, **HoST-MoE** exploits the foundational invariant of *manipulated real video*: **every forged video retains unmanipulated host context** (peripheral background patches, authentic temporal segments, or unmodified audio/kinematics) carrying the true camera ISP noiseprint, codec quantization history, and natural scene dynamics.

---

## Architectural Highlights

1. **Motion-Noise Forensic Residual Stem (`MN-Stem`)**: Combines dynamically constrained zero-DC `5x5` Bayar convolutions, Spatial Rich Model (SRM) high-pass filters, and a fast `1/4`-resolution local correlation cost-volume optical-flow residual (`<2 ms`), directly solving the `0.135 mIoU` Object-Removal blind spot of semantic ViT/MLLM localizers.
2. **Orthogonal SVD-LoRA + Auxiliary-Loss-Free Tri-Pool MoE (`Tri-STMoE`)**: Adapts frozen `DINOv3-ViT-L/16` in the orthogonal complement of its principal singular subspace (`Effort`-style) while routing tokens across **Spatial (`Pool S`)**, **Temporal-RoPE (`Pool T`)**, and **Forensic-Residual (`Pool R`)** experts using DeepSeek-V3 / MoSTE **auxiliary-loss-free bias balancing**.
3. **Self-Referential Host-Prototype Differential Conditioning (`SHDC`)**: Estimates per-clip unmanipulated host feature moments $(\mu_{\text{host}}, \sigma_{\text{host}})$ from soft authenticity anchor tokens, canceling global H.264/H.265 transcoding and neural-restoration shifts to eliminate **cross-domain score polarization** (`Cross-AUC` & `MID-FFD`).
4. **Modality-Decoupled Native Readouts (`MDNR`) & Kinematic-Prosodic Verifier (`KP-Verify`)**: Separates visual (`D_v`) and audio (`D_a`) multi-scale temporal pyramids to prevent cross-track false-alarm bleed (`AV-FABLE`), while gracefully falling back to a learned **Silent-Prior** token on silent benchmarks (`has_audio=False`).
5. **Single RTX 5090 (32 GB GDDR7) Training Footprint**: `~24.9M` trainable parameters (`~5.1%` of total), `~18.2 GB` peak `BF16` VRAM at `384x384` (`T=16` frames), `~28 ms/clip` single-pass inference.

---

## Repository Structure

```text
host-forensics/
├── configs/
│   ├── rtx5090_unified_train.yaml   # Full DINOv3-L + WavLM BF16 config for 1x RTX 5090 (32 GB)
│   └── smoke_cpu_or_ci.yaml         # Compact offline-safe config for instant CI & smoke tests
├── paper/
│   ├── main.tex                     # Two-column CVPR/NeurIPS research manuscript (TikZ + PGFPlots)
│   ├── references.bib               # Verified 2012-2026 forensic bibliography
│   ├── main.pdf                     # Compiled 7-page camera-ready research PDF
│   └── figures/poc_results.json     # Executed empirical mechanism verification metrics
├── scripts/
│   ├── prepare_manifests.py         # Multi-benchmark FFmpeg video/audio/mask -> .pt + .jsonl builder
│   ├── evaluate_benchmarks.py       # Zero-shot Cross-AUC, MID-FFD, AV-mAP, J&F & LaTeX table exporter
│   ├── run_poc_experiments.py       # Reproducible CPU empirical verification of SHDC, MoE & MN-Stem
│   └── run_rtx5090_suite.sh         # Turnkey Stage-1/Stage-2 training, zero-shot eval & ablation runner
├── src/host_forensics/
│   ├── config.py                    # Typed dataclasses & YAML serialization
│   ├── cli.py                       # `host-detect` CLI (info, smoke-train, train, infer, dump-config)
│   ├── models/
│   │   ├── mn_stem.py               # Bayar zero-DC conv, SRM bank, Local Cost-Volume Motion Residual
│   │   ├── backbones.py             # OrthogonalSVDLinear, MultiPriorVisualBackbone, FrozenAudioBackbone
│   │   ├── tri_stmoe.py             # Spatial/Temporal-RoPE/Residual experts + BiasBalancedTopKRouter
│   │   ├── shdc.py                  # SelfReferentialHostConditioning (median-gated, affine-invariant)
│   │   ├── heads.py                 # KP-Verify (Silent-Prior), Multi-Scale Temporal Pyramid, Tubelet Mask Head
│   │   └── host_model.py            # HoSTForensicsModel & PristineQuantileCalibrator
│   ├── losses/
│   │   └── forensic_losses.py       # Heterogeneous multi-task loss (BCE, Polarization, MIL Anchor, DIoU, Mask)
│   ├── metrics/
│   │   └── evaluation.py            # AUROC, EER, Polarization Cross-AUC, MID-FFD, Native AV-mAP, Spatial J&F
│   ├── data/
│   │   ├── video_io.py              # FFmpeg/FFprobe raw .mp4/.mkv/.webm video, 16kHz audio & mask decoding
│   │   ├── transforms.py            # Compound codec degradation & symmetric VAE/restoration laundering
│   │   └── datasets.py              # Unified JSONL Manifest Dataset & Synthetic Manipulated Video Dataset
│   └── engine/
│       └── trainer.py               # Two-stage BF16 curriculum trainer, MoE bias step & pristine calibration
└── tests/
    ├── test_modules_and_shapes.py   # Invariant & gradient verification tests
    ├── test_losses_and_metrics.py   # Multi-task loss, calibrator, and 2026 forensic metric tests
    └── test_video_io_and_scripts.py # End-to-end FFmpeg .mp4 -> manifest -> 2-stage train -> eval -> LaTeX test
```

---

## Quickstart

### 1. Environment Installation

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]"
# On your RTX 5090 workstation with HuggingFace Hub access for DINOv3-L & WavLM-Large:
uv pip install --python .venv/bin/python -e ".[hub,dev]"
```

### 2. Run Unit Tests, Smoke Verification & Empirical PoC Suite

```bash
.venv/bin/pytest -v
.venv/bin/host-detect smoke-train --epochs 2
.venv/bin/python scripts/run_poc_experiments.py
```

### 3. Prepare Raw Video Benchmarks & Run Two-Stage RTX 5090 Suite

```bash
# Convert raw benchmark folder (generic, df40, manivid, av_deepfake1m, av_fable) into .pt + .jsonl
.venv/bin/python scripts/prepare_manifests.py \
  --dataset-type manivid \
  --input-root /data/forensics/raw/ManiVid-38K \
  --output-manifest /data/forensics/manifests/manivid_train.jsonl \
  --tensor-dir /data/forensics/tensors/manivid \
  --workers 8

# Launch full Stage-1 (384px) + Stage-2 (448px) curriculum, zero-shot evaluation, LaTeX export & ablations
DATA_ROOT=/data/forensics bash scripts/run_rtx5090_suite.sh
```

### 4. Structured Zero-Shot Inference (Direct `.mp4` or `.pt`, Audio-Visual or Silent)

```bash
# Direct inference on any raw .mp4/.mkv/.webm video (auto-detects audio stream via ffprobe!)
.venv/bin/host-detect infer \
  --config configs/rtx5090_unified_train.yaml \
  --checkpoint checkpoints/rtx5090_main/host_moe_calibrated.pt \
  --video suspicious_clip.mp4

# Force visual-only Silent-Prior fallback
.venv/bin/host-detect infer \
  --config configs/rtx5090_unified_train.yaml \
  --checkpoint checkpoints/rtx5090_main/host_moe_calibrated.pt \
  --video silent_clip.mp4 \
  --silent
```

---

## License

Released under the **Apache-2.0 License** for open reproducible research in multimedia forensics.
