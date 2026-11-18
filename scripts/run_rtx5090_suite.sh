#!/usr/bin/env bash
# Turnkey Single-RTX-5090 (32 GB GDDR7) Training, Zero-Shot Evaluation & Ablation Runner
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/forensics}"
OUT_ROOT="${OUT_ROOT:-checkpoints/rtx5090_main}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HOST_CLI="${HOST_CLI:-.venv/bin/host-detect}"

echo "[1/3] Launching HoST-MoE Stage-1 (384px) + Stage-2 (448px) Unified Training & Pristine Calibration..."
"${HOST_CLI}" train \
  --config configs/rtx5090_unified_train.yaml \
  --train-manifest "${DATA_ROOT}/manifests/unified_stage1_train.jsonl" \
  --stage2-manifest "${DATA_ROOT}/manifests/highres_stage2_mask_tfl.jsonl" \
  --val-manifest "${DATA_ROOT}/manifests/pristine_and_val.jsonl" \
  --output-dir "${OUT_ROOT}" \
  --device cuda

echo "[2/3] Evaluating Zero-Shot Generalization, Cross-AUC Polarization, AV-mAP, and Spatial J&F..."
"${PYTHON_BIN}" scripts/evaluate_benchmarks.py \
  --config configs/rtx5090_unified_train.yaml \
  --checkpoint "${OUT_ROOT}/host_moe_calibrated.pt" \
  --benchmark "DF40=${DATA_ROOT}/manifests/test_df40_unseen.jsonl" \
  --benchmark "CelebDFpp=${DATA_ROOT}/manifests/test_celebdf_pp.jsonl" \
  --benchmark "TalkingHead=${DATA_ROOT}/manifests/test_talkingheadbench.jsonl" \
  --benchmark "ManiVidBench=${DATA_ROOT}/manifests/test_manividbench.jsonl" \
  --benchmark "AVFABLE=${DATA_ROOT}/manifests/test_av_fable.jsonl" \
  --benchmark "NTIRE2026=${DATA_ROOT}/manifests/test_ntire2026_wild.jsonl" \
  --output-json "${OUT_ROOT}/zero_shot_eval.json" \
  --latex-out "paper/tables/rtx5090_main_row.tex"

echo "[3/3] Generating & Running Ablation Configs (Table 4)..."
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from host_forensics.config import HoSTConfig

base = HoSTConfig.from_yaml("configs/rtx5090_unified_train.yaml")
abl_dir = Path("configs/ablations")
abl_dir.mkdir(parents=True, exist_ok=True)

# 1. Standard LoRA instead of Orthogonal SVD-LoRA
c1 = HoSTConfig.from_yaml("configs/rtx5090_unified_train.yaml")
c1.backbone.use_ortho_lora = False
c1.to_yaml(abl_dir / "abl_no_ortho_lora.yaml")

# 2. Zero bias update rate (no auxiliary-loss-free load balancing)
c2 = HoSTConfig.from_yaml("configs/rtx5090_unified_train.yaml")
c2.moe.bias_update_rate = 0.0
c2.to_yaml(abl_dir / "abl_no_bias_balance.yaml")

# 3. Disable Polarization Regularizer & Anchor Loss
c3 = HoSTConfig.from_yaml("configs/rtx5090_unified_train.yaml")
c3.loss.lambda_polarization = 0.0
c3.loss.lambda_anchor = 0.0
c3.to_yaml(abl_dir / "abl_no_host_depolarization.yaml")

# 4. Asymmetric/Zero Laundering Simulation
c4 = HoSTConfig.from_yaml("configs/rtx5090_unified_train.yaml")
c4.train.symmetric_laundering_prob = 0.0
c4.to_yaml(abl_dir / "abl_no_symmetric_laundering.yaml")
print("Generated 4 ablation configs in configs/ablations/")
PY

echo "Done! Main metrics written to ${OUT_ROOT}/zero_shot_eval.json and paper/tables/rtx5090_main_row.tex."
