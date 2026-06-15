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

pip install numpy pandas matplotlib seaborn nltk scipy spacy
python -m spacy download en_core_web_sm
