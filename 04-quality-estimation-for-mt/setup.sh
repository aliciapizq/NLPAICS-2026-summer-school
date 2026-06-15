#!/usr/bin/env bash
# Setup for this lesson. The root ./setup.sh runs this INSIDE the lesson's
# own virtualenv — plain `pip install` lands in this lesson's environment.
# The venv inherits the base image's torch/CUDA via --system-site-packages,
# so do NOT reinstall torch unless you need a specific version.
#
# Fill in everything your practical needs: pinned pip installs, dataset
# downloads, model caching. Must run non-interactively; safe to re-run.
set -euo pipefail

# This notebook is 100% PyTorch (BLEU, BERTScore, COMET, Qwen). Keep transformers
# off TensorFlow so it never imports the TF that Colab/base images ship with.
export USE_TF=0
export USE_FLAX=0
# Quieter, non-interactive downloads.
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

echo ">>> Installing pinned Python dependencies ..."
# NOTE on pins: unbabel-comet requires transformers<5.0 AND huggingface-hub<1.0
# (and numpy<2.0), so we cannot use the latest 5.x/1.x. transformers 4.46.3 is
# the sweet spot: it satisfies COMET and still supports Qwen2.5. torch is NOT
# listed — it is inherited from the base image via --system-site-packages.
pip install --quiet \
  "unbabel-comet==2.2.7" \
  "transformers==4.46.3" \
  "accelerate==1.1.1" \
  "huggingface-hub==0.26.5" \
  "sentencepiece==0.2.0" \
  "sacremoses==0.1.1" \
  "nltk==3.9.1" \
  "bert-score==0.3.13" \
  "numpy==1.26.4"

