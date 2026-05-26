#!/usr/bin/env bash
# =============================================================================
# PeakVI Probe — run script
# Edit the dataset block below, then execute:
#   bash run_probe.sh
# or to run in background:
#   nohup bash run_probe.sh > run_probe.log 2>&1 &
# =============================================================================
set -euo pipefail

# ─── Dataset configuration ──────────────────────────────────────────────────
#  Pick one dataset block and comment out the rest.

# Dataset: 5w_GSE196830_atac (top-12k stratified, 29 classes)
DATASET_ID="5w_GSE196830_atac"

# Dataset: 10w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="10w_GSE196830_atac"

# Dataset: 20w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="20w_GSE196830_atac"

# Dataset: 40w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="40w_GSE196830_atac"

# Dataset: GSE96583_atac (noncoding33, 8 classes)
# DATASET_ID="GSE96583_atac"

# ─── Run configuration ──────────────────────────────────────────────────────
RUN_NAME="probe"
WANDB_PROJECT="peakvi-probe"
N_LATENT=20          # PeakVI default
N_HIDDEN=512
N_LAYERS=2
BATCH_SIZE_TRAIN=512   # stable default for multi-GPU sweeps on 32GB V100s
MAX_EPOCHS=500         # 500 epochs: sufficient for convergence + early stopping exits early
EARLY_STOPPING="--early_stopping"  # monitor val ELBO; use "" to disable
EARLY_STOPPING_PATIENCE=24  # tightened from scvi-tools default of 50 — cuts wasted tail epochs after the plateau
N_JOBS=12
MAX_ITER=2000
SAVE_EMBEDDINGS="--save_embeddings"   # set to "--save_embeddings" to also save .npy for visualize.py

PYTHON="/root/project/chaohan/.conda/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_probe"

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

$PYTHON probe.py \
    --dataset_id        "${DATASET_ID}" \
    --output_dir        "${OUTPUT_DIR}" \
    --run_name          "${RUN_NAME}" \
    --wandb_project     "${WANDB_PROJECT}" \
    --n_latent          "${N_LATENT}" \
    --n_hidden          "${N_HIDDEN}" \
    --n_layers          "${N_LAYERS}" \
    --batch_size_train  "${BATCH_SIZE_TRAIN}" \
    --max_epochs        "${MAX_EPOCHS}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --n_jobs            "${N_JOBS}" \
    --max_iter          "${MAX_ITER}" \
    ${EARLY_STOPPING} \
    ${SAVE_EMBEDDINGS}
