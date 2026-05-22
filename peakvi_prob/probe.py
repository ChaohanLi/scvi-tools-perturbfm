"""
Cell-type evaluation for PeakVI embeddings.

Protocol (identical to scVI / scFoundation / scGPT probe baselines):
  - Same stratified 80/20 train/val split (seed=42)
  - Train PeakVI **unsupervised** on train cells only (no cell-type labels)
  - Extract latent z (give_mean=True) from val cells
  - 5-fold StratifiedKFold on val embeddings:
      each fold: StandardScaler -> optional PCA -> LinearSVC(dual=False)
  - Report mean CV train/test accuracy, macro-F1, balanced-accuracy

Input data:
  PeakVI operates on cell × peak binary/count accessibility matrices from
  scATAC-seq.  Each dataset is provided as a counts parquet (cells × sites)
  plus a site_to_gene TSV that maps site column indices to genomic coordinates
  and gene labels.  The parquet is loaded directly as a cell × peak AnnData;
  no gene-activity aggregation is performed (PeakVI works on raw peak space).

PeakVI reference: Ashuach et al. 2022 (Nature Methods).
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import scanpy as sc
import torch
import wandb
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import scvi
from lightning.pytorch.callbacks import Callback
from scvi.model import PEAKVI

_PROB_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry must have exactly one of:
#   "h5ad"   — pre-built cell×peak AnnData (obs must contain "cell_type")
#   "parquet" — raw accessibility parquet (cells × sites) from the readData/
#               pipeline; requires "site_tsv" and "label_csv" companions.
#
# "n_class" is only used for the assertion check; it is inferred from the data
# automatically when the --dataset_id shortcut is used.
DATASET_REGISTRY = {
    # ── scATAC parquet datasets (GSE196830 + GSE96583) ──────────────────
    "5w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/5w_PBMC_GSE196830/counts_top12k_stratified_allchr.parquet",
        "site_tsv":  "/lichaohan/readData/site_to_gene_index_stratified_top12k_bl.tsv",
        "label_csv": "/lichaohan/readData/5w_PBMC_GSE196830/filtered_5w_all_cells.csv",
        "n_class":   29,
    },
    "10w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/10w_PBMC_GSE196830/stratified_noncoding33_counts.parquet",
        "site_tsv":  "/lichaohan/readData/10w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_10w_nochr5.tsv",
        "label_csv": "/lichaohan/readData/10w_PBMC_GSE196830/filtered_10w_all_celltype.csv",
        "n_class":   29,
    },
    "20w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/20w_PBMC_GSE196830/stratified_noncoding33_counts.parquet",
        "site_tsv":  "/lichaohan/readData/20w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_20w_nochr5_18.tsv",
        "label_csv": "/lichaohan/readData/20w_PBMC_GSE196830/filtered_20w_all_celltype.csv",
        "n_class":   29,
    },
    "40w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/40w_PBMC_GSE196830/stratified_noncoding33_40w_counts.parquet",
        "site_tsv":  "/lichaohan/readData/40w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_40w.tsv",
        "label_csv": "/lichaohan/readData/40w_PBMC_GSE196830/filtered_40w_all_celltype.csv",
        "n_class":   29,
    },
    "80w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/80w_PBMC_GSE196830/stratified_noncoding33_80w_counts.parquet",
        "site_tsv":  "/lichaohan/readData/80w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_80w.tsv",
        "label_csv": "/lichaohan/readData/80w_PBMC_GSE196830/filtered_80w_all_celltype.csv",
        "n_class":   29,
    },
    "120w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/120w_PBMC_GSE196830/stratified_noncoding33_120w_counts.parquet",
        "site_tsv":  "/lichaohan/readData/120w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_120w.tsv",
        "label_csv": "/lichaohan/readData/120w_PBMC_GSE196830/filtered_120w_all_celltype.csv",
        "n_class":   29,
    },
    "GSE96583_atac": {
        "parquet":   "/lichaohan/readData/GSE96583_PBMC/stratified_noncoding33_counts.parquet",
        "site_tsv":  "/lichaohan/readData/GSE96583_PBMC/site_map.tsv",
        "label_csv": "/lichaohan/readData/GSE96583_PBMC/filtered_stratified_noncoding33_celltype.csv",
        "n_class":   8,
    },
}


# ---------------------------------------------------------------------------
# Build AnnData from parquet (cell × raw-peak counts)
# ---------------------------------------------------------------------------
def parquet_to_adata(parquet_path: str, site_tsv_path: str, label_csv_path: str):
    """Load cell × peak accessibility matrix directly — no gene aggregation.

    Columns str(1) … str(N_sites) become peak features; rows are cells.
    Cell-type labels are joined from label_csv on cell_barcode.
    The resulting AnnData.X is a CSR integer matrix suitable for PeakVI.
    """
    import pandas as pd
    import scipy.sparse as sp
    import anndata as ad

    print(f"  Loading site TSV: {site_tsv_path}")
    site_df = pd.read_csv(site_tsv_path, sep="\t")
    n_sites = len(site_df)
    # Build human-readable peak names  "chr1:1014251"
    if "chrom" in site_df.columns and "col_idx_0based" in site_df.columns:
        peak_names = (
            site_df["chrom"].astype(str)
            + ":"
            + site_df["col_idx_0based"].astype(str)
        ).tolist()
    else:
        peak_names = [str(i) for i in range(n_sites)]
    print(f"  {n_sites} peaks")

    print(f"  Loading parquet: {parquet_path}")
    import pyarrow.parquet as pq
    data_cols = [str(i + 1) for i in range(n_sites)]
    cols_to_read = ["cell_barcode"] + data_cols

    pf = pq.ParquetFile(parquet_path)
    # Verify column availability (guard against mismatched site count)
    parquet_col_set = set(pf.schema_arrow.names)
    missing = [c for c in data_cols[:5] if c not in parquet_col_set]
    if missing:
        raise ValueError(
            f"Parquet does not contain expected site columns (e.g. {missing}). "
            f"Check that site_tsv and parquet correspond to the same dataset."
        )

    # Chunked read: convert 2 000 rows at a time to avoid a dense
    # n_cells × n_peaks intermediate (e.g. 200 k × 228 k × 4 B ≈ 183 GB).
    # Each 2 000-row batch peaks at ~1.7 GB before going sparse.
    barcodes_list: list = []
    chunks: list = []
    n_batches = 0
    for batch in pf.iter_batches(batch_size=2000, columns=cols_to_read):
        df_b = batch.to_pandas()
        barcodes_list.append(df_b["cell_barcode"].values)
        chunks.append(sp.csr_matrix(df_b[data_cols].values, dtype=np.float32))
        del df_b
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"    … {n_batches * 2000:,} rows loaded", flush=True)

    barcodes = np.concatenate(barcodes_list)
    X = sp.vstack(chunks, format="csr")
    del chunks, barcodes_list
    print(f"  Parquet loaded: {X.shape[0]:,} cells × {X.shape[1]:,} peaks  "
          f"(nnz={X.nnz:,}, density={X.nnz/X.shape[0]/X.shape[1]:.4f})")

    print(f"  Loading cell-type labels: {label_csv_path}")
    lbl_df = pd.read_csv(label_csv_path)
    # Support both "cell_type" and "celltype" column names
    ct_col = "cell_type" if "cell_type" in lbl_df.columns else lbl_df.columns[1]
    bc_to_ct = dict(zip(lbl_df["cell_barcode"].values, lbl_df[ct_col].values))
    cell_types = [bc_to_ct[bc] for bc in barcodes]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_type": cell_types},
            index=pd.Index(barcodes, name=""),
        ),
        var=pd.DataFrame(index=pd.Index(peak_names, name="")),
    )
    print(
        f"  AnnData built: {adata.shape}, "
        f"{adata.obs['cell_type'].nunique()} cell types"
    )
    return adata


# ---------------------------------------------------------------------------
# Exact replica of the stratified split used across all probe baselines
# ---------------------------------------------------------------------------
def _stratified_train_val_split(
    barcodes,
    labels,
    train_size: float = 0.8,
    random_state: int = 42,
):
    barcodes = np.asarray(barcodes)
    labels   = np.asarray(labels)

    rng = np.random.default_rng(int(random_state))
    train_parts, val_parts = [], []

    for lab in np.unique(labels):
        idx  = np.flatnonzero(labels == lab)
        perm = rng.permutation(idx)
        n_tr = int(np.floor(len(idx) * train_size))
        if len(idx) >= 2:
            n_tr = min(max(n_tr, 1), len(idx) - 1)
        else:
            n_tr = len(idx)
        train_parts.append(perm[:n_tr])
        val_parts.append(perm[n_tr:])

    train_idx = np.concatenate(train_parts)
    val_idx   = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return barcodes[train_idx], barcodes[val_idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train PeakVI and evaluate a LinearSVC probe on val embeddings"
    )
    p.add_argument("--dataset_id", type=str, default="10w_GSE196830_atac",
                   help="Registry key (e.g. 10w_GSE196830_atac, 20w_GSE196830_atac, "
                        "GSE96583_atac).  Overrides --parquet / --site_tsv / --label_csv.")
    # Manual path overrides (ignored when --dataset_id resolves from registry)
    p.add_argument("--parquet",   type=str, default=None,
                   help="Path to cell×site counts parquet")
    p.add_argument("--site_tsv",  type=str, default=None,
                   help="Path to site-to-gene mapping TSV")
    p.add_argument("--label_csv", type=str, default=None,
                   help="Path to cell barcode + cell_type CSV")
    # Split / seed
    p.add_argument("--train_size", type=float, default=0.8)
    p.add_argument("--seed",       type=int,   default=42)
    # PeakVI hyperparameters
    p.add_argument("--n_latent",   type=int,   default=20,
                   help="PeakVI latent dimension (default 20, paper default)")
    p.add_argument("--n_hidden",   type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=2)
    p.add_argument("--max_epochs", type=int,   default=500,
                   help="PeakVI training epochs (default 500; sufficient for convergence "
                        "on large datasets — built-in heuristic gives only 25-50 for "
                        "20w-40w, far from converged). Early stopping will terminate early.")
    p.add_argument("--batch_size_train", type=int, default=1024,
                   help="Mini-batch size for PeakVI training (default 1024; up from "
                        "scvi-tools' 128 default for better GPU utilization on large "
                        "ATAC datasets — VAE/Adam with epoch-based KL warmup is robust)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override PeakVI learning rate (default None → use framework "
                        "default 1e-4). Bump if you go beyond batch=2048.")
    p.add_argument("--early_stopping",   default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Enable PeakVI early stopping on val ELBO (default True). "
                        "Use --no_early_stopping to disable.")
    p.add_argument("--early_stopping_patience", type=int, default=24,
                   help="Early-stopping patience in epochs (default 24; was 50 in "
                        "scvi-tools default — tightened to cut wasted tail epochs).")
    # LinearSVC probe hyperparameters (identical to other baselines)
    p.add_argument("--cv_folds",   type=int,   default=5)
    p.add_argument("--max_samples", type=int,  default=5000)
    p.add_argument("--pca_dim",    type=int,   default=None,
                   help="PCA before SVC. Default None (n_latent=20 is already compact).")
    p.add_argument("--max_iter",   type=int,   default=2000)
    p.add_argument("--n_jobs",     type=int,   default=16)
    # Output
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_PROB_DIR, "outputs_probe"))
    p.add_argument("--run_name",   type=str,   default=None)
    p.add_argument("--save_embeddings", action="store_true")
    # Weights & Biases
    p.add_argument("--wandb_project", type=str, default="peakvi-probe")
    p.add_argument("--no_wandb",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LinearSVC probe (identical to scVI / scFoundation / scGPT probes)
# ---------------------------------------------------------------------------
def build_probe(train_embeddings, args):
    steps = [("scaler", StandardScaler())]
    if args.pca_dim is not None:
        pca_dim = min(
            int(args.pca_dim),
            train_embeddings.shape[0],
            train_embeddings.shape[1],
        )
        if pca_dim >= 1 and pca_dim < train_embeddings.shape[1]:
            steps.append(("pca", PCA(n_components=pca_dim, random_state=args.seed)))
    steps.append(("svc", LinearSVC(
        random_state=args.seed,
        dual=False,
        max_iter=args.max_iter,
    )))
    return Pipeline(steps)


def compute_metrics(labels, preds):
    return {
        "accuracy":          float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1":          float(f1_score(labels, preds, average="macro",
                                            zero_division=0)),
        "weighted_f1":       float(f1_score(labels, preds, average="weighted",
                                            zero_division=0)),
        "n_samples":         int(len(labels)),
    }


def run_svc_cv(embeddings, labels, args):
    unique, counts = np.unique(labels, return_counts=True)
    keep_classes   = unique[counts >= args.cv_folds]

    if len(keep_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes with >= {args.cv_folds} samples; "
            f"got {len(keep_classes)}."
        )

    dropped_info = [
        {"class_id": int(c), "count": int(n)}
        for c, n in zip(unique, counts)
        if n < args.cv_folds
    ]
    if len(keep_classes) != len(unique):
        mask       = np.isin(labels, keep_classes)
        embeddings = embeddings[mask]
        labels     = labels[mask]
        labels     = np.searchsorted(keep_classes, labels)

    splitter = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.seed,
    )

    n_fold_jobs = min(args.cv_folds, args.n_jobs)
    n_jobs_ovr  = max(1, args.n_jobs // n_fold_jobs)
    print(f"Parallelism: {n_fold_jobs} fold workers × {n_jobs_ovr} OvR cores "
          f"= {n_fold_jobs * n_jobs_ovr} cores used (of {args.n_jobs})",
          flush=True)

    splits = list(splitter.split(embeddings, labels))

    def _run_fold(fold_idx, train_idx, test_idx):
        x_train = embeddings[train_idx]
        y_train = labels[train_idx]
        x_test  = embeddings[test_idx]
        y_test  = labels[test_idx]

        if args.max_samples and len(x_train) > args.max_samples:
            sampled_idx = np.random.choice(len(x_train), args.max_samples, replace=False)
            x_train_fit = x_train[sampled_idx]
            y_train_fit = y_train[sampled_idx]
        else:
            x_train_fit = x_train
            y_train_fit = y_train

        probe = build_probe(x_train_fit, args)
        print(f"  Fold {fold_idx}/{args.cv_folds}: fitting SVC on "
              f"{len(x_train_fit)} samples...", flush=True)
        probe.fit(x_train_fit, y_train_fit)

        train_preds = probe.predict(x_train)
        test_preds  = probe.predict(x_test)
        print(f"  Fold {fold_idx}/{args.cv_folds}: done.", flush=True)
        n_kept = len(keep_classes)
        return {
            "fold":           fold_idx,
            "train_size":     int(len(x_train)),
            "train_fit_size": int(len(x_train_fit)),
            "test_size":      int(len(x_test)),
            "train":          compute_metrics(y_train, train_preds),
            "test":           compute_metrics(y_test,  test_preds),
            "probe_steps":    list(probe.named_steps.keys()),
            "test_per_class": {
                "f1":        f1_score(y_test, test_preds, average=None,
                                      labels=np.arange(n_kept), zero_division=0).tolist(),
                "precision": precision_score(y_test, test_preds, average=None,
                                             labels=np.arange(n_kept), zero_division=0).tolist(),
                "recall":    recall_score(y_test, test_preds, average=None,
                                          labels=np.arange(n_kept), zero_division=0).tolist(),
                "support":   [int((y_test == c).sum()) for c in range(n_kept)],
            },
        }

    fold_metrics = Parallel(n_jobs=n_fold_jobs, backend="loky")(
        delayed(_run_fold)(fold_idx, train_idx, test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1)
    )

    mean_metrics = {}
    for split in ["train", "test"]:
        mean_metrics[split] = {
            key: float(np.mean([f[split][key] for f in fold_metrics]))
            for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        }

    n_kept = len(keep_classes)
    per_class_cv = []
    for c in range(n_kept):
        fold_f1s   = [fm["test_per_class"]["f1"][c]        for fm in fold_metrics]
        fold_precs = [fm["test_per_class"]["precision"][c] for fm in fold_metrics]
        fold_recs  = [fm["test_per_class"]["recall"][c]    for fm in fold_metrics]
        fold_sups  = [fm["test_per_class"]["support"][c]   for fm in fold_metrics]
        per_class_cv.append({
            "class_idx":      int(c),
            "mean_f1":        float(np.mean(fold_f1s)),
            "std_f1":         float(np.std(fold_f1s)),
            "mean_precision": float(np.mean(fold_precs)),
            "std_precision":  float(np.std(fold_precs)),
            "mean_recall":    float(np.mean(fold_recs)),
            "std_recall":     float(np.std(fold_recs)),
            "mean_support":   float(np.mean(fold_sups)),
        })

    return {
        "fold_metrics":           fold_metrics,
        "mean_metrics":           mean_metrics,
        "per_class_cv":           per_class_cv,
        "kept_classes":           [int(x) for x in keep_classes.tolist()],
        "dropped_classes":        dropped_info,
        "n_samples_after_filter": int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Fast single-split probe for per-epoch checkpoint selection
# ---------------------------------------------------------------------------
def _quick_probe_f1(z_val: np.ndarray, y_val: np.ndarray, args) -> float:
    """Stratified 70/30 single-split macro-F1.  O(seconds) per epoch."""
    from sklearn.model_selection import StratifiedShuffleSplit

    unique, counts = np.unique(y_val, return_counts=True)
    keep = unique[counts >= 2]
    if len(keep) < 2:
        return 0.0
    if len(keep) < len(unique):
        mask = np.isin(y_val, keep)
        z_val = z_val[mask]
        y_val = y_val[mask]
        y_val = np.searchsorted(keep, y_val)

    if len(z_val) > 6000:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(z_val), 6000, replace=False)
        z_val = z_val[idx]
        y_val = y_val[idx]

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(z_val, y_val))
    except ValueError:
        return 0.0

    z_tr, z_te = z_val[tr_idx], z_val[te_idx]
    y_tr, y_te = y_val[tr_idx], y_val[te_idx]

    if len(z_tr) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(z_tr), args.max_samples, replace=False)
        z_tr = z_tr[idx]
        y_tr = y_tr[idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    LinearSVC(dual=False, max_iter=args.max_iter,
                             random_state=args.seed)),
    ])
    try:
        pipe.fit(z_tr, y_tr)
        return float(f1_score(pipe.predict(z_te), y_te,
                              average="macro", zero_division=0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-epoch probe callback (Lightning)
# ---------------------------------------------------------------------------
class PerEpochProbeCallback(Callback):
    """Logs ELBO + quick downstream probe F1 after each validation epoch.

    Saves model state to ``best_ckpt_dir`` whenever a new best quick-F1 is
    achieved.  After training, load that checkpoint for the official full-CV
    evaluation.
    """

    def __init__(
        self,
        peakvi_model,
        adata_val,
        y_val: np.ndarray,
        args,
        best_ckpt_dir: str,
        use_wandb: bool,
    ):
        super().__init__()
        self.peakvi_model  = peakvi_model
        self.adata_val     = adata_val
        self.y_val         = y_val
        self.args          = args
        self.best_ckpt_dir = best_ckpt_dir
        self.use_wandb     = use_wandb

        self.best_f1    = -1.0
        self.best_epoch = -1
        self.elbo_history: list = []  # (epoch, elbo_train, elbo_val)
        self.f1_history:   list = []  # (epoch, quick_f1)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-based

        elbo_train = float(trainer.callback_metrics.get("elbo_train",
                                                         float("nan")))
        elbo_val   = float(trainer.callback_metrics.get("elbo_validation",
                                                         float("nan")))
        self.elbo_history.append((epoch, elbo_train, elbo_val))

        # _check_if_trained raises unless is_trained_ is True;
        # set it temporarily — the module weights are valid mid-training.
        _was_trained = self.peakvi_model.is_trained_
        self.peakvi_model.is_trained_ = True
        try:
            with torch.no_grad():
                z_val = self.peakvi_model.get_latent_representation(
                    self.adata_val, give_mean=True
                )
        finally:
            self.peakvi_model.is_trained_ = _was_trained

        f1 = _quick_probe_f1(z_val, self.y_val, self.args)
        self.f1_history.append((epoch, f1))

        if self.use_wandb:
            wandb.log({
                "epoch":                epoch,
                "train/elbo_train":     elbo_train,
                "train/elbo_val":       elbo_val,
                "epoch_probe/quick_f1": f1,
            })

        print(
            f"  [Epoch {epoch:3d}] elbo_train={elbo_train:.4f}  "
            f"elbo_val={elbo_val:.4f}  quick_f1={f1:.4f}",
            flush=True,
        )

        if f1 > self.best_f1:
            self.best_f1    = f1
            self.best_epoch = epoch
            # Force is_trained_=True before save so the serialized attr_dict
            # passes _check_if_trained on reload; restore in finally so the
            # rest of the training loop is unaffected.
            _was_trained = self.peakvi_model.is_trained_
            self.peakvi_model.is_trained_ = True
            try:
                self.peakvi_model.save(self.best_ckpt_dir, overwrite=True)
            finally:
                self.peakvi_model.is_trained_ = _was_trained
            print(
                f"    → New best downstream F1={f1:.4f} at epoch {epoch} "
                f"— checkpoint saved.",
                flush=True,
            )


def save_fold_metrics(path, cv_result):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "train_size", "train_fit_size", "test_size",
            "train_accuracy", "test_accuracy",
            "train_macro_f1", "test_macro_f1",
        ])
        for fold in cv_result["fold_metrics"]:
            writer.writerow([
                fold["fold"],
                fold["train_size"],
                fold["train_fit_size"],
                fold["test_size"],
                fold["train"]["accuracy"],
                fold["test"]["accuracy"],
                fold["train"]["macro_f1"],
                fold["test"]["macro_f1"],
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Auto-resolve paths from registry when dataset_id is given
    if args.dataset_id in DATASET_REGISTRY:
        cfg = DATASET_REGISTRY[args.dataset_id]
        args.parquet   = cfg["parquet"]
        args.site_tsv  = cfg["site_tsv"]
        args.label_csv = cfg["label_csv"]

    if args.parquet is None or args.site_tsv is None or args.label_csv is None:
        raise ValueError(
            "Provide --dataset_id (from registry) or all of "
            "--parquet, --site_tsv, --label_csv."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    scvi.settings.seed = args.seed

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or time.strftime("probe_%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{args.dataset_id}"
    out_dir  = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device:           {device}")
    print(f"Output directory: {out_dir}")

    # ── Weights & Biases init ─────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\nLoading ATAC data (cell × peak) ...")
    adata = parquet_to_adata(args.parquet, args.site_tsv, args.label_csv)
    print(f"  Shape: {adata.shape}  obs columns: {list(adata.obs.columns)}")

    # ── Build label encoding ──────────────────────────────────────────
    cell_types = adata.obs["cell_type"].values
    classes    = sorted(set(cell_types))
    type2idx   = {c: i for i, c in enumerate(classes)}
    labels     = np.array([type2idx[c] for c in cell_types], dtype=np.int64)
    n_class    = len(classes)
    print(f"  Classes: {n_class}")

    # ── Stratified 80/20 split — identical to all probe baselines ─────
    barcodes         = np.array(adata.obs_names)
    train_bc, val_bc = _stratified_train_val_split(
        barcodes, labels,
        train_size=args.train_size,
        random_state=args.seed,
    )
    bc2idx    = {bc: i for i, bc in enumerate(barcodes)}
    train_idx = np.array([bc2idx[bc] for bc in train_bc])
    val_idx   = np.array([bc2idx[bc] for bc in val_bc])

    adata_train = adata[train_idx].copy()
    adata_val   = adata[val_idx].copy()
    del adata  # free full matrix — train+val copies are sufficient; saves RAM for large datasets
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val")

    y_val   = labels[val_idx]
    y_train = labels[train_idx]
    y_all   = labels   # original order preserved by train_idx/val_idx

    # ── Set up PeakVI ─────────────────────────────────────────────────
    scvi.settings.num_threads = 4
    PEAKVI.setup_anndata(adata_train)

    model = PEAKVI(
        adata_train,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers_encoder=args.n_layers,
        n_layers_decoder=args.n_layers,
    )
    print(
        f"\nPeakVI model: n_latent={args.n_latent}, n_hidden={args.n_hidden}, "
        f"n_layers={args.n_layers}"
    )
    print(f"  Parameters: {sum(p.numel() for p in model.module.parameters()):,}")

    # ── Per-epoch probe callback ──────────────────────────────────────
    best_ckpt_dir = os.path.join(out_dir, "peakvi_model_best_f1")
    probe_callback = PerEpochProbeCallback(
        peakvi_model  = model,
        adata_val     = adata_val,
        y_val         = y_val,
        args          = args,
        best_ckpt_dir = best_ckpt_dir,
        use_wandb     = not args.no_wandb,
    )

    # ── Train ─────────────────────────────────────────────────────────
    scvi.settings.verbosity = 20  # WARNING — suppress lightning verbosity
    print(f"\nTraining PeakVI (unsupervised) — max_epochs={args.max_epochs}, "
          f"early_stopping={args.early_stopping}...", flush=True)
    train_kwargs = dict(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size_train,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        callbacks=[probe_callback],
    )
    if args.lr is not None:
        train_kwargs["lr"] = args.lr
    model.train(**train_kwargs)
    trained_epochs = model.history["elbo_train"].shape[0]
    final_elbo     = float(model.history["elbo_train"].iloc[-1])
    print(f"  Trained {trained_epochs} epochs | final train ELBO: {final_elbo:.4f}")

    # ── Save final (early-stopped) model ─────────────────────────────
    final_ckpt_dir = os.path.join(out_dir, "peakvi_model_final")
    model.save(final_ckpt_dir, overwrite=True)
    print(f"  Final model saved → {final_ckpt_dir}")
    print(f"  Best downstream F1={probe_callback.best_f1:.4f} "
          f"at epoch {probe_callback.best_epoch} → {best_ckpt_dir}")

    # ── Load best checkpoint, extract embeddings, run full probe ─────
    print(f"\nLoading best-downstream checkpoint (epoch {probe_callback.best_epoch})...")
    best_model = PEAKVI.load(best_ckpt_dir, adata=adata_train)
    # Older checkpoints from this script were saved with is_trained_=False
    # (see callback fix); force it on after load so inference works.
    best_model.is_trained_ = True

    print("Extracting validation embeddings (best checkpoint)...")
    z_val   = best_model.get_latent_representation(adata_val,   give_mean=True)
    print(f"  {z_val.shape[1]} dims  |  {len(y_val)} val samples")

    print("Extracting training embeddings (best checkpoint)...")
    z_train = best_model.get_latent_representation(adata_train, give_mean=True)
    print(f"  {len(y_train)} train samples")

    # Reconstruct full-dataset embeddings from train+val (avoids keeping adata in RAM).
    # Concatenate in [train_idx, val_idx] order; reorder to original cell order afterwards.
    _z_concat = np.vstack([z_train, z_val])
    _y_concat = np.concatenate([y_train, y_val])
    _orig_order = np.argsort(np.concatenate([train_idx, val_idx]))
    z_all = _z_concat[_orig_order]
    y_all = _y_concat[_orig_order]
    del _z_concat, _y_concat, _orig_order
    print(f"  {len(y_all)} total samples (reconstructed from train+val)")

    # ── Full 5-fold SVC on best checkpoint (primary result) ──────────
    print("\nRunning full 5-fold SVC probe on best-downstream checkpoint...")
    cv_result = run_svc_cv(z_val, y_val, args)

    # ── Full 5-fold SVC on final checkpoint (comparison) ─────────────
    print("\nRunning full 5-fold SVC probe on final (early-stopped) checkpoint...")
    z_val_final     = model.get_latent_representation(adata_val, give_mean=True)
    cv_result_final = run_svc_cv(z_val_final, y_val, args)

    # ── Save results ───────────────────────────────────────────────────
    result = {
        # Primary result: best-downstream checkpoint
        "metrics":                cv_result["mean_metrics"],
        "fold_metrics":           cv_result["fold_metrics"],
        "per_class_cv":           cv_result["per_class_cv"],
        "embedding_dim":          int(z_val.shape[1]),
        "class_names":            classes,
        "type2idx":               type2idx,
        "kept_classes":           cv_result["kept_classes"],
        "dropped_classes":        cv_result["dropped_classes"],
        "n_samples_after_filter": cv_result["n_samples_after_filter"],
        # Final (early-stopped) checkpoint for comparison
        "final_metrics":          cv_result_final["mean_metrics"],
        # Training metadata
        "best_downstream_epoch":  probe_callback.best_epoch,
        "best_downstream_f1":     probe_callback.best_f1,
        "peakvi_trained_epochs":  trained_epochs,
        "peakvi_final_elbo":      final_elbo,
        "elbo_history":           probe_callback.elbo_history,
        "f1_history":             probe_callback.f1_history,
        "args":                   vars(args),
        "protocol":               "peakvi_train_val_embeddings_5fold_svc_cv_best_ckpt",
    }
    with open(os.path.join(out_dir, "probe_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(classes, f, indent=2)
    save_fold_metrics(os.path.join(out_dir, "probe_fold_metrics.csv"), cv_result)

    # Best-downstream and final models were already saved during training

    if args.save_embeddings:
        np.save(os.path.join(out_dir, "embeddings_val.npy"),   z_val)
        np.save(os.path.join(out_dir, "labels_val.npy"),       y_val)
        np.save(os.path.join(out_dir, "embeddings_train.npy"), z_train)
        np.save(os.path.join(out_dir, "labels_train.npy"),     y_train)
        np.save(os.path.join(out_dir, "embeddings_all.npy"),   z_all)
        np.save(os.path.join(out_dir, "labels_all.npy"),       y_all)

    # ── Print summary ──────────────────────────────────────────────────
    print("\n── Best-downstream checkpoint ──")
    for split in ["train", "test"]:
        m = cv_result["mean_metrics"][split]
        print(
            f"cv {split:>5}: acc={m['accuracy']:.4f} "
            f"bal_acc={m['balanced_accuracy']:.4f} "
            f"macro_f1={m['macro_f1']:.4f} "
            f"weighted_f1={m['weighted_f1']:.4f}"
        )
    print(f"  (epoch {probe_callback.best_epoch} of {trained_epochs} total)")
    print("\n── Final (early-stopped) checkpoint ──")
    for split in ["train", "test"]:
        m = cv_result_final["mean_metrics"][split]
        print(
            f"cv {split:>5}: acc={m['accuracy']:.4f} "
            f"bal_acc={m['balanced_accuracy']:.4f} "
            f"macro_f1={m['macro_f1']:.4f} "
            f"weighted_f1={m['weighted_f1']:.4f}"
        )
    if cv_result["dropped_classes"]:
        print(f"Dropped classes with < {args.cv_folds} samples: "
              f"{cv_result['dropped_classes']}")
    print(f"\nSaved to: {out_dir}")

    # ── Weights & Biases logging ───────────────────────────────────────
    if not args.no_wandb:
        mean     = cv_result["mean_metrics"]       # best checkpoint
        mean_fin = cv_result_final["mean_metrics"] # final checkpoint
        wandb.log({
            # ── Best-downstream checkpoint (primary metric) ──────────
            "cv_train/accuracy":          mean["train"]["accuracy"],
            "cv_train/balanced_accuracy": mean["train"]["balanced_accuracy"],
            "cv_train/macro_f1":          mean["train"]["macro_f1"],
            "cv_train/weighted_f1":       mean["train"]["weighted_f1"],
            "cv_test/accuracy":           mean["test"]["accuracy"],
            "cv_test/balanced_accuracy":  mean["test"]["balanced_accuracy"],
            "cv_test/macro_f1":           mean["test"]["macro_f1"],
            "cv_test/weighted_f1":        mean["test"]["weighted_f1"],
            # ── Final (early-stopped) checkpoint (comparison) ────────
            "final/cv_test/accuracy":           mean_fin["test"]["accuracy"],
            "final/cv_test/balanced_accuracy":  mean_fin["test"]["balanced_accuracy"],
            "final/cv_test/macro_f1":           mean_fin["test"]["macro_f1"],
            "final/cv_test/weighted_f1":        mean_fin["test"]["weighted_f1"],
            # ── Training metadata ────────────────────────────────────
            "embedding_dim":              int(z_val.shape[1]),
            "n_val_samples":              int(len(y_val)),
            "n_classes_used":             len(cv_result["kept_classes"]),
            "n_classes_dropped":          len(cv_result["dropped_classes"]),
            "best_downstream_epoch":      probe_callback.best_epoch,
            "best_downstream_f1":         probe_callback.best_f1,
            "peakvi_trained_epochs":      trained_epochs,
            "peakvi_final_elbo":          final_elbo,
        })
        fold_table = wandb.Table(
            columns=["fold", "train_size", "test_size",
                     "train_acc", "test_acc", "train_macro_f1", "test_macro_f1"]
        )
        for fold in cv_result["fold_metrics"]:
            fold_table.add_data(
                fold["fold"],
                fold["train_size"],
                fold["test_size"],
                fold["train"]["accuracy"],
                fold["test"]["accuracy"],
                fold["train"]["macro_f1"],
                fold["test"]["macro_f1"],
            )
        wandb.log({"fold_metrics": fold_table})

        per_class_table = wandb.Table(
            columns=["class_name", "mean_f1", "std_f1",
                     "mean_recall", "mean_precision", "mean_support"]
        )
        kept = cv_result["kept_classes"]
        for entry in cv_result["per_class_cv"]:
            orig = kept[entry["class_idx"]]
            name = classes[orig] if orig < len(classes) else str(orig)
            per_class_table.add_data(
                name,
                round(entry["mean_f1"],        4),
                round(entry["std_f1"],         4),
                round(entry["mean_recall"],    4),
                round(entry["mean_precision"], 4),
                round(entry["mean_support"],   1),
            )
        wandb.log({"per_class_metrics": per_class_table})

        # ELBO history table (training curve)
        elbo_table = wandb.Table(columns=["epoch", "elbo_train", "elbo_val"])
        for ep, et, ev in probe_callback.elbo_history:
            elbo_table.add_data(ep, round(et, 4), round(ev, 4))
        wandb.log({"elbo_history": elbo_table})

        # Per-epoch quick-probe F1 table
        f1_table = wandb.Table(columns=["epoch", "quick_f1"])
        for ep, f1 in probe_callback.f1_history:
            f1_table.add_data(ep, round(f1, 4))
        wandb.log({"epoch_f1_history": f1_table})

        wandb.finish()


if __name__ == "__main__":
    main()
