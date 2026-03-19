#!/usr/bin/env bash
set -e

echo "== Activating environment =="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shid_env

echo "== Step 1: Fetch + Clean =="
python pipeline/01_fetch_clean.py

echo "== Step 2: Train + Score =="
python pipeline/02_train_score.py

echo "== Step 3: Build FAISS index =="
python pipeline/03_build_index.py

echo "✅ Pipeline finished successfully"
