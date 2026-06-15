#!/usr/bin/env bash
# Lesson 07, Automated Hyperparameter Optimization & Model Selection for NLP Pipelines
#
# Installs the AutoGOAL AutoML stack (migrated to Python 3.10) together with its
# scikit-learn and NLTK contribs, entirely via pip. The root ./setup.sh runs this
# INSIDE this lesson's own virtualenv, which on the vast.ai PyTorch (Vast) images is
# created from /venv/main (Python 3.10). Plain `pip install` therefore lands in this
# lesson's environment only. Must run non-interactively and be safe to re-run.
#
# Why pinned / why 3.10: AutoGOAL's scientific stack is pinned to the exact versions
# validated for the session; AutoGOAL needs Python 3.9/3.10 (scipy/scikit-learn/gensim
# ship cp310 wheels at these versions, so the whole install is wheels-only, no compiler
# needed). torch from /venv/main stays importable alongside numpy 1.26 (verified).
set -euo pipefail

# --- 0. Guard the interpreter version -------------------------------------------------
py="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$py" in
  3.9|3.10) ;;
  *) echo "[07] AutoGOAL needs Python 3.9 or 3.10, but this environment is Python $py." >&2
     echo "[07] Use a vastai/pytorch '...-py310' image (its /venv/main is Python 3.10)." >&2
     exit 1;;
esac
echo "[07] Python $py at $(python -c 'import sys; print(sys.executable)')"

# --- 1. Pinned scientific stack (the exact versions validated for the class) ----------
# Installed first so AutoGOAL resolves against these; all have cp310 wheels.
pip install --quiet \
  "numpy==1.26.4" \
  "scipy==1.12.0" \
  "scikit-learn==1.0.2" \
  "gensim==4.3.3" \
  "nltk==3.9.4" \
  "python-crfsuite==0.9.12" \
  "sklearn-crfsuite==0.3.6"

# --- 2. AutoGOAL AutoML + sklearn/nltk contribs, via pip from GitHub ------------------
# Installed from the Python-3.10 branches (core v1.0.4, contribs v0.8.5). All four are
# listed explicitly so pip resolves the cross-dependencies against these git checkouts,
# NOT the older PyPI `autogoal` (which pins scipy 1.6 and breaks on Python 3.10).
# Requires git (the vast.ai PyTorch image provides it). For a frozen ref, replace the
# branch names with the commit SHAs (core 0aae518, contribs 39d2711).
CORE="git+https://github.com/gia-uh/autogoal.git@v1.0.4"
CONTRIB="git+https://github.com/autogoal/autogoal-contrib.git@v0.8.5"
pip install --quiet \
  "${CORE}#subdirectory=autogoal" \
  "${CONTRIB}#subdirectory=autogoal_contrib" \
  "${CONTRIB}#subdirectory=autogoal_sklearn" \
  "${CONTRIB}#subdirectory=autogoal_nltk"

# --- 2b. Sentence-Transformers, the "LLM bridge" block in Part 3 ----------------------
# A small pretrained transformer (all-MiniLM-L6-v2) is wrapped as a custom AutoGOAL block
# so the search can pick a transformer embedding over TF-IDF. torch comes from the base
# image (/venv/main), we do NOT reinstall it. numpy is pinned on the command line so the
# transformers/tokenizers pull cannot drag in numpy 2.x and break AutoGOAL's scientific
# stack. (Harmless pip note: AutoGOAL's unused CLI wants an older `typer`; the import is
# unaffected.)
pip install --quiet "numpy==1.26.4" "sentence-transformers==5.5.1" "matplotlib==3.9.2"

# --- 3. NLTK data used by the AutoGOAL nltk wrappers (tokenizers/stemmers/stopwords) ---
python - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    nltk.download(pkg, quiet=True)
print("[07] NLTK data ready")
PY

# --- 4. Pre-cache the dataset + the embedding model so the session never waits ---------
# These downloads are retried: the 20 Newsgroups mirror occasionally returns a transient
# HTTP error, and conference wifi is conference wifi. If a download keeps failing, just
# re-run ./setup.sh 07 once the network settles.
python - <<'PY'
import time

def retry(fn, what, tries=5, wait=6):
    for i in range(1, tries + 1):
        try:
            fn(); return
        except Exception as e:
            print(f"[07] {what}: attempt {i}/{tries} failed ({type(e).__name__}: {e})")
            if i < tries:
                time.sleep(wait)
    raise SystemExit(f"[07] {what}: gave up after {tries} attempts. "
                     f"Re-run ./setup.sh 07 once the network is reachable.")

# 20 Newsgroups (4 topics): sklearn caches the full archive under ~/scikit_learn_data
from sklearn.datasets import fetch_20newsgroups
cats = ['sci.med', 'sci.space', 'rec.autos', 'talk.politics.misc']
def get_news():
    for sub in ("train", "test"):
        fetch_20newsgroups(subset=sub, categories=cats, remove=("headers", "footers", "quotes"))
retry(get_news, "20 Newsgroups download")
print("[07] 20 Newsgroups cached")

# The Part-3 embedding model (about 90 MB on first use), cached now so the session is offline
from sentence_transformers import SentenceTransformer
retry(lambda: SentenceTransformer("all-MiniLM-L6-v2", device="cpu"), "MiniLM model download")
print("[07] all-MiniLM-L6-v2 embedding model cached")
PY

echo "[07] setup complete."
