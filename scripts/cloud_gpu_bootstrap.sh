#!/usr/bin/env bash
# Turnkey Cloud / Bare-Metal GPU Provisioning & Full-Scale HoST-MoE Execution Script
# Compatible with RunPod, Lambda Labs, Vast.ai, AWS EC2, and local RTX 5090 / H100 workstations.
#
# Usage:
#   HF_TOKEN=hf_xxx DATA_ROOT=/workspace/forensics bash scripts/cloud_gpu_bootstrap.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/LucasbrandaoVIX/host-forensics.git}"
WORK_DIR="${WORK_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-/workspace/forensics}"
OUT_ROOT="${OUT_ROOT:-checkpoints/rtx5090_main}"
USE_COMPILE="${USE_COMPILE:-1}"
PREP_WORKERS="${PREP_WORKERS:-16}"

echo "=================================================================="
echo " HoST-MoE Cloud GPU Bootstrap & Full-Scale Benchmark Runner"
echo "=================================================================="

# 1. Hardware & System Tooling Check
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. Attach at least 1 NVIDIA GPU (RTX 5090 / 4090 / A100 / H100)." >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
echo "[INFO] Detected ${GPU_COUNT} GPU(s) (using NPROC_PER_NODE=${NPROC_PER_NODE}):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[INFO] Installing ffmpeg..."
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    apt-get update -qq && apt-get install -y -qq ffmpeg git curl
  else
    echo "[ERROR] ffmpeg is required on PATH. Install ffmpeg and re-run." >&2
    exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] Installing uv package manager..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# 2. Clone Repository if Running Standalone via curl | bash
if [[ ! -f "${WORK_DIR}/pyproject.toml" ]]; then
  echo "[INFO] Cloning ${REPO_URL} into ${WORK_DIR}/host-forensics..."
  git clone "${REPO_URL}" "${WORK_DIR}/host-forensics"
  WORK_DIR="${WORK_DIR}/host-forensics"
fi

# 3. Python 3.11 Virtualenv & HuggingFace Dependencies
echo "[INFO] Provisioning Python environment in ${WORK_DIR}/.venv..."
uv venv "${WORK_DIR}/.venv" --python 3.11 --allow-existing
uv pip install --python "${WORK_DIR}/.venv/bin/python" -e "${WORK_DIR}[hub,dev]" huggingface_hub

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  echo "[INFO] HF_TOKEN detected in environment (enabled for gated DINOv3-ViT-L/16 weights)."
else
  echo "[WARN] HF_TOKEN is not set. Meta's facebook/dinov3-vitl16-pretrain-lvd1689m requires accepting the"
  echo "       license on Hugging Face Hub and exporting HF_TOKEN=hf_..."
fi

# 4. Pre-Flight Unit Test Verification
echo "[INFO] Running pre-flight verification suite..."
"${WORK_DIR}/.venv/bin/pytest" "${WORK_DIR}/tests" -q

# 5. Optional Automated Raw Dataset -> Manifest Conversion
mkdir -p "${DATA_ROOT}/manifests" "${DATA_ROOT}/tensors"
declare -A RAW_DATASETS=(
  ["df40"]="DF40"
  ["manivid"]="ManiVid-38K"
  ["av_deepfake1m"]="AV-Deepfake1M-PlusPlus"
  ["av_fable"]="AV-FABLE"
)

for dtype in "${!RAW_DATASETS[@]}"; do
  folder="${DATA_ROOT}/raw/${RAW_DATASETS[$dtype]}"
  manifest="${DATA_ROOT}/manifests/${dtype}_prepared.jsonl"
  if [[ -d "${folder}" && ! -f "${manifest}" ]]; then
    echo "[INFO] Preparing manifest for ${RAW_DATASETS[$dtype]} (${folder})..."
    "${WORK_DIR}/.venv/bin/python" "${WORK_DIR}/scripts/prepare_manifests.py" \
      --dataset-type "${dtype}" \
      --input-root "${folder}" \
      --output-manifest "${manifest}" \
      --tensor-dir "${DATA_ROOT}/tensors/${dtype}" \
      --workers "${PREP_WORKERS}"
  fi
done

# 6. Execute Full Stage-1 + Stage-2 Curriculum, Zero-Shot Evaluation & Ablations
if [[ ! -f "${DATA_ROOT}/manifests/unified_stage1_train.jsonl" ]]; then
  echo "=================================================================="
  echo "[READY] Environment, GPUs, and HoST-MoE package verified."
  echo "Place or symlink your prepared JSONL manifests under ${DATA_ROOT}/manifests/:"
  echo "  - unified_stage1_train.jsonl"
  echo "  - highres_stage2_mask_tfl.jsonl"
  echo "  - pristine_and_val.jsonl"
  echo "  - test_df40_unseen.jsonl, test_celebdf_pp.jsonl, test_talkingheadbench.jsonl,"
  echo "    test_manividbench.jsonl, test_av_fable.jsonl, test_ntire2026_wild.jsonl"
  echo "Then re-run:"
  echo "  DATA_ROOT=${DATA_ROOT} NPROC_PER_NODE=${NPROC_PER_NODE} USE_COMPILE=${USE_COMPILE} bash scripts/run_rtx5090_suite.sh"
  echo "=================================================================="
  exit 0
fi

DATA_ROOT="${DATA_ROOT}" \
OUT_ROOT="${OUT_ROOT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
USE_COMPILE="${USE_COMPILE}" \
bash "${WORK_DIR}/scripts/run_rtx5090_suite.sh"

# 7. Optional LaTeX Manuscript Recompilation
if command -v pdflatex >/dev/null 2>&1 && [[ -f "${WORK_DIR}/paper/main.tex" ]]; then
  echo "[INFO] Recompiling paper/main.pdf with updated experimental tables..."
  pdflatex -interaction=nonstopmode -output-directory="${WORK_DIR}/paper" "${WORK_DIR}/paper/main.tex" >/dev/null
  bibtex "${WORK_DIR}/paper/main" >/dev/null || true
  pdflatex -interaction=nonstopmode -output-directory="${WORK_DIR}/paper" "${WORK_DIR}/paper/main.tex" >/dev/null
  pdflatex -interaction=nonstopmode -output-directory="${WORK_DIR}/paper" "${WORK_DIR}/paper/main.tex" >/dev/null
fi

echo "=================================================================="
echo "[COMPLETE] Calibrated checkpoint: ${OUT_ROOT}/host_moe_calibrated.pt"
echo "[COMPLETE] Zero-shot evaluation:  ${OUT_ROOT}/zero_shot_eval.json"
echo "=================================================================="
