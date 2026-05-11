#!/usr/bin/env bash
# =============================================================================
# scVI Probe — run script
# Edit the variables below, then execute:
#   bash run_probe.sh
# or to run in background:
#   nohup bash run_probe.sh > run_probe.log 2>&1 &
#
# Note: scVI with gene_likelihood="nb" (default) works best with raw integer
#       counts. All new datasets (GSE196830, GSE96583) are raw counts. The
#       original 5w_symbol dataset is log1p normalized; NB still converges
#       on it, but results may differ from raw-count benchmarks.
# =============================================================================
set -euo pipefail

# ─── Dataset configuration ──────────────────────────────────────────────────
#  Pick one dataset block and comment out the rest, or override all variables.

# Dataset: 5w_symbol (original, log1p normalized, 29 classes)
# H5AD="/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad"
# DATASET_ID="5w_symbol"
# GENE_SPACE="ensembl"    # var_names already HGNC; set "ensembl" to keep all 30k genes as-is

# Dataset: 5w_GSE196830 (raw counts, 29 classes) — recommended for NB
# H5AD="/lichaohan/readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad"
# DATASET_ID="5w_GSE196830"
# GENE_SPACE="hgnc"    # or "hgnc" to map Ensembl→HGNC and drop unmapped genes

# Dataset: GSE96583 (raw counts, 8 classes) — recommended for NB
# H5AD="/lichaohan/readData/GSE96583_PBMC/GSE96583_merged_dedup.h5ad"
# DATASET_ID="GSE96583"
# GENE_SPACE="hgnc"    # or "hgnc"

# Dataset: 10w_GSE196830 (raw counts, 29 classes) — recommended for NB
H5AD="/lichaohan/readData/10w_PBMC_GSE196830/10w_allcelltype.h5ad"
DATASET_ID="10w_GSE196830"
GENE_SPACE="hgnc"    # or "hgnc"

#Dataset: 20w_GSE196830 (raw counts, 29 classes) — recommended for NB
# H5AD="/lichaohan/readData/20w_PBMC_GSE196830/20w_allcelltype.h5ad"
# DATASET_ID="20w_GSE196830"
# GENE_SPACE="hgnc"    # or "hgnc"

# ─── Run configuration ──────────────────────────────────────────────────────
RUN_NAME="probe"                       # wandb run name prefix (dataset_id + gene_space appended automatically)
WANDB_PROJECT="scvi-probe"
SYMBOL_MAP="/lichaohan/readData/gene_id_to_symbol.tsv"  # used only when GENE_SPACE=hgnc
N_LATENT=30
N_HIDDEN=128
N_LAYERS=2
GENE_LIKELIHOOD="nb"                   # nb (recommended), zinb, or normal
BATCH_SIZE_TRAIN=128
N_JOBS=16
MAX_ITER=2000
SAVE_EMBEDDINGS=""                     # set to "--save_embeddings" to also save embeddings_val.npy / labels_val.npy (needed for visualize.py)

PYTHON="/lichaohan/miniconda3/envs/scvi/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_probe"   # output root; run_name appended automatically

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

$PYTHON probe.py \
    --h5ad              "${H5AD}" \
    --dataset_id        "${DATASET_ID}" \
    --gene_space        "${GENE_SPACE}" \
    --symbol_map        "${SYMBOL_MAP}" \
    --output_dir        "${OUTPUT_DIR}" \
    --run_name          "${RUN_NAME}" \
    --wandb_project     "${WANDB_PROJECT}" \
    --n_latent          "${N_LATENT}" \
    --n_hidden          "${N_HIDDEN}" \
    --n_layers          "${N_LAYERS}" \
    --gene_likelihood   "${GENE_LIKELIHOOD}" \
    --batch_size_train  "${BATCH_SIZE_TRAIN}" \
    --n_jobs            "${N_JOBS}" \
    --max_iter          "${MAX_ITER}" \
    ${SAVE_EMBEDDINGS}
