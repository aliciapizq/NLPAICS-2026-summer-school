#!/usr/bin/env bash
# Setup for this lesson. The root ./setup.sh runs this INSIDE the lesson's
# own virtualenv — plain `pip install` lands in this lesson's environment.
# The venv inherits the base image's torch/CUDA via --system-site-packages,
# so do NOT reinstall torch unless you need a specific version.
#
# Fill in everything your practical needs: pinned pip installs, dataset
# downloads, model caching. Must run non-interactively; safe to re-run.
set -euo pipefail

# Example:
# pip install --quiet scikit-learn==1.5.2 datasets==3.2.0


# ---------------------------------------------------------------------------
# 1. Install exact library versions
# ---------------------------------------------------------------------------
echo "[setup] Installing dependencies..."
# Blackwell GPUs (sm_120, e.g. RTX PRO 6000) require CUDA 13.x kernels.
# cu126 tops out at sm_90; use cu130 for Blackwell support.

pip install \
    transformers==5.11.0 \
    accelerate==1.14.0 \
    pandas==3.0.3 \
    notebook==7.5.7 \
    ipywidgets==8.1.8 \
    krippendorff==0.6

# ---------------------------------------------------------------------------
# 2. Pre-download models (cached in ~/.cache/huggingface/hub by default)
# ---------------------------------------------------------------------------
GENERATOR_MODEL="Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL="AtlaAI/Selene-1-Mini-Llama-3.1-8B"

echo "[setup] Downloading generator model: ${GENERATOR_MODEL}"
HF_HOME="${HF_HOME:-}" python - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("${GENERATOR_MODEL}")
AutoModelForCausalLM.from_pretrained("${GENERATOR_MODEL}")
print("  Generator model cached.")
EOF

echo "[setup] Downloading judge model: ${JUDGE_MODEL}"
HF_HOME="${HF_HOME:-}" python - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("${JUDGE_MODEL}")
AutoModelForCausalLM.from_pretrained("${JUDGE_MODEL}")
print("  Judge model cached.")
EOF

echo "" 