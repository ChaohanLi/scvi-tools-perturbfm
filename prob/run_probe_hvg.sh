#!/usr/bin/env bash
# =============================================================================
# scVI Probe (HVG-filtered) — run script
# Selects top HVG genes on the training split (seurat_v3 flavor, raw counts)
# before scVI training.  All other protocol details are identical to probe.py.
#
# Usage:
#   bash run_probe_hvg.sh
#   nohup bash run_probe_hvg.sh > run_probe_hvg.log 2>&1 &
# =============================================================================
set -euo pipefail

# ─── Dataset configuration ──────────────────────────────────────────────────
# Uncomment ONE block.

# Dataset: 5w_GSE196830 (raw counts, 29 classes)
# DATASET_ID="5w_GSE196830"
# GENE_SPACE="hgnc"

# Dataset: GSE96583 (raw counts, 8 classes)
# DATASET_ID="GSE96583"
# GENE_SPACE="ensembl"

# Dataset: 10w_GSE196830 (raw counts, 29 classes)
# DATASET_ID="10w_GSE196830"
# GENE_SPACE="hgnc"

# Dataset: 20w_GSE196830 (raw counts, 29 classes)
DATASET_ID="20w_GSE196830"
GENE_SPACE="hgnc"

# Dataset: 40w_GSE196830 (raw counts, 29 classes)
# DATASET_ID="40w_GSE196830"
# GENE_SPACE="hgnc"

# ─── Run configuration ──────────────────────────────────────────────────────
RUN_NAME="probe_hvg"
WANDB_PROJECT="scvi-probe-hvg"
SYMBOL_MAP="/lichaohan/readData/gene_id_to_symbol.tsv"
N_TOP_GENES=5000          # HVG count; set to 0 to disable HVG selection
N_LATENT=30
N_HIDDEN=128
N_LAYERS=2
GENE_LIKELIHOOD="nb"
BATCH_SIZE_TRAIN=1024  # up from scVI's 128 default for better GPU utilization; VAE/Adam + epoch-based KL warmup is robust to this jump
MAX_EPOCHS=500            # 500 epochs upper bound; early stopping exits early on val ELBO plateau
EARLY_STOPPING="--early_stopping"  # stop early on val ELBO plateau; use "" to disable
EARLY_STOPPING_PATIENCE=24         # tightened from scvi-tools default of 45 — cuts wasted tail epochs after the plateau
N_JOBS=16
MAX_ITER=2000
SAVE_EMBEDDINGS="--save_embeddings"   # remove to skip saving .npy files

PYTHON="/lichaohan/miniconda3/envs/scvi/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_probe"

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

$PYTHON probe_hvg.py \
    --dataset_id        "${DATASET_ID}" \
    --gene_space        "${GENE_SPACE}" \
    --symbol_map        "${SYMBOL_MAP}" \
    --n_top_genes       "${N_TOP_GENES}" \
    --output_dir        "${OUTPUT_DIR}" \
    --run_name          "${RUN_NAME}" \
    --wandb_project     "${WANDB_PROJECT}" \
    --n_latent          "${N_LATENT}" \
    --n_hidden          "${N_HIDDEN}" \
    --n_layers          "${N_LAYERS}" \
    --gene_likelihood   "${GENE_LIKELIHOOD}" \
    --batch_size_train  "${BATCH_SIZE_TRAIN}" \
    --max_epochs        "${MAX_EPOCHS}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --n_jobs            "${N_JOBS}" \
    --max_iter          "${MAX_ITER}" \
    ${EARLY_STOPPING} \
    ${SAVE_EMBEDDINGS}
