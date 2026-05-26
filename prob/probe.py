"""
Cell-type evaluation for scVI embeddings.

Protocol (matches scFoundation / scGPT probe baselines):
  - Same stratified 80/20 train/val split (seed=42)
  - Train scVI **unsupervised** on train cells only (no cell-type labels)
  - Extract latent z (give_mean=True) from val cells
  - 5-fold StratifiedKFold on val embeddings:
      each fold: StandardScaler -> optional PCA -> LinearSVC(dual=False)
  - Report mean CV train/test accuracy, macro-F1, balanced-accuracy

Data note: adata.X is log1p-normalized (no raw counts available).
  scVI uses gene_likelihood="nb" (default) for numerical stability — the NB
  likelihood is robust even on normalized data and avoids the scale-collapse
  instability that "normal" likelihood suffers after ~150 epochs.
scVI reference: Lopez et al. 2018 (Nature Methods).
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
from scvi.model import SCVI

_PROB_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Build AnnData from scATAC-seq parquet by aggregating sites → gene activity
# ---------------------------------------------------------------------------
def parquet_to_adata(parquet_path: str, site_to_gene_path: str, label_h5ad_path: str):
    """Aggregate 240 000 chromatin-accessibility sites into gene activity scores.

    Mapping convention: mapping TSV row i (0-based) corresponds to parquet
    data column str(i+1).  Sites are summed per gene_index to produce raw
    integer gene activity counts suitable for scVI (NB likelihood).

    Cell-type labels are taken from label_h5ad_path (barcodes must match).
    """
    import pandas as pd
    import scipy.sparse as sp
    import anndata as ad
    import scanpy as sc

    print(f"  Loading site-to-gene mapping: {site_to_gene_path}")
    mapping = pd.read_csv(site_to_gene_path, sep="\t")
    gene_indices = mapping["gene_index"].values          # (240000,) int
    unique_gi, inv = np.unique(gene_indices, return_inverse=True)
    n_genes = len(unique_gi)
    gi_to_label = mapping.groupby("gene_index")["gene_label"].first()
    gene_names = [str(gi_to_label[gi]) for gi in unique_gi]
    print(f"  {len(mapping)} sites → {n_genes} genes")

    print(f"  Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    barcodes = df["cell_barcode"].values
    data_cols = [str(i + 1) for i in range(len(mapping))]
    df_data = df[data_cols]                              # (n_cells, 240000)

    print(f"  Aggregating sites → genes (groupby sum)…")
    # Rename columns to gene_index values, then groupby-sum on axis=1
    df_data = df_data.copy()
    df_data.columns = gene_indices
    # Transpose → groupby rows → transpose back:  (240000, n_cells).groupby → (n_genes, n_cells) → (n_cells, n_genes)
    X_gene = df_data.T.groupby(level=0).sum().T         # (n_cells, n_genes)
    X_sparse = sp.csr_matrix(X_gene.values.astype(np.float32))

    print(f"  Loading cell-type labels: {label_h5ad_path}")
    adata_ref = sc.read_h5ad(label_h5ad_path)
    bc_to_ct = dict(zip(adata_ref.obs_names, adata_ref.obs["cell_type"]))
    cell_types = [bc_to_ct[bc] for bc in barcodes]

    adata = ad.AnnData(
        X=X_sparse,
        obs=pd.DataFrame({"cell_type": cell_types}, index=pd.Index(barcodes, name="")),
        var=pd.DataFrame(index=pd.Index(gene_names, name="")),
    )
    print(f"  AnnData built: {adata.shape}, {adata.obs['cell_type'].nunique()} cell types")
    return adata


# ---------------------------------------------------------------------------
# Exact replica of the stratified split used in scFoundation / scGPT baselines
# to guarantee the same train/val cells across all models.
# ---------------------------------------------------------------------------
def _stratified_train_val_split(
    barcodes: np.ndarray,
    labels: np.ndarray,
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
# Dataset registry — maps dataset_id → (h5ad path, gene_space)
# Sweep agents only need to pass --dataset_id; other dataset-specific fields
# are resolved automatically from this table.
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "5w_symbol":     {
        "h5ad":       "/root/project/chaohan/readData/5w_allcelltype_anno_symbol.h5ad",
        "gene_space": "ensembl",
    },
    "5w_GSE196830":  {
        "h5ad":       "/root/project/chaohan/readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad",
        "gene_space": "hgnc",
    },
    "GSE96583":      {
        "h5ad":       "/root/project/chaohan/readData/GSE96583_PBMC/GSE96583_merged_dedup.h5ad",
        "gene_space": "ensembl",   # var_names are already HGNC symbols
    },
    "10w_GSE196830": {
        "h5ad":       "/root/project/chaohan/readData/10w_PBMC_GSE196830/10w_allcelltype.h5ad",
        "gene_space": "hgnc",
    },
    "20w_GSE196830": {
        "h5ad":       "/root/project/chaohan/readData/20w_PBMC_GSE196830/20w_allcelltype.h5ad",
        "gene_space": "hgnc",
    },
    "40w_GSE196830": {
        "h5ad":       "/root/project/chaohan/readData/40w_PBMC_GSE196830/GSE196830_40w_subset.h5ad",
        "gene_space": "hgnc",
    },
    "80w_GSE196830": {
        "h5ad":       "/root/project/chaohan/readData/80w_PBMC_GSE196830/GSE196830_80w_subset.h5ad",
        "gene_space": "hgnc",
    },
    "120w_GSE196830": {
        "h5ad":       "/root/project/chaohan/readData/120w_PBMC_GSE196830/GSE196830_120w_subset.h5ad",
        "gene_space": "hgnc",
    },
    # scATAC-seq gene-activity baseline: 240k sites aggregated to gene counts
    "5w_GSE196830_atac": {
        "parquet":      "/root/project/chaohan/readData/5w_PBMC_GSE196830/counts_top12k_stratified_allchr.parquet",
        "site_to_gene": "/root/project/chaohan/readData/5w_PBMC_GSE196830/site_to_gene_index_stratified_top12k_bl.tsv",
        "label_h5ad":   "/root/project/chaohan/readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad",
        # var_names are already HGNC symbols after aggregation → skip Ensembl filter
        "gene_space":   "ensembl",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train scVI and evaluate a LinearSVC probe on val embeddings"
    )
    p.add_argument("--h5ad", type=str,
                   default="/root/project/chaohan/readData/5w_allcelltype_anno_symbol.h5ad")
    p.add_argument("--dataset_id", type=str, default="5w_symbol",
                   help="Short dataset tag appended to run_name (e.g. 5w_symbol, "
                        "5w_GSE196830, GSE96583, 10w_GSE196830, 20w_GSE196830, 40w_GSE196830)")
    p.add_argument("--train_size", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    # scVI model hyperparameters
    p.add_argument("--n_latent", type=int, default=30,
                   help="scVI latent dimension (default 30, common benchmark standard)")
    p.add_argument("--n_hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--gene_likelihood", type=str, default="nb",
                   choices=["nb", "zinb", "normal"],
                   help="scVI gene likelihood (default nb; numerically stable for log1p data)")
    p.add_argument("--max_epochs", type=int, default=500,
                   help="scVI training epochs upper bound (default 500; early stopping "
                        "exits early when val ELBO plateaus).")
    p.add_argument("--batch_size_train", type=int, default=512,
                   help="Mini-batch size for scVI training (default 512; up from "
                        "scVI's 128 default for better GPU utilization on large "
                        "scRNA datasets — VAE/Adam with epoch-based KL warmup is robust)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override scVI learning rate (default None → use framework "
                        "default 1e-3, passed via plan_kwargs). Bump if you go "
                        "beyond batch=2048.")
    p.add_argument("--early_stopping", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Enable scVI early stopping on val ELBO (default True). "
                        "Use --no_early_stopping to disable.")
    p.add_argument("--early_stopping_patience", type=int, default=24,
                   help="Early-stopping patience in epochs (default 24; was 45 in "
                        "scvi-tools default — tightened to cut wasted tail epochs).")
    # Gene space selection
    p.add_argument("--gene_space", type=str, default="ensembl",
                   choices=["ensembl", "hgnc"],
                   help="'ensembl': use var_names as-is; "
                        "'hgnc': map Ensembl IDs → HGNC symbols via symbol_map, "
                        "drop genes without a valid HGNC entry")
    p.add_argument("--symbol_map", type=str,
                   default="/root/project/chaohan/readData/gene_id_to_symbol.tsv",
                   help="TSV (gene_id \t gene_symbol) for Ensembl→HGNC mapping. "
                        "Only used when --gene_space hgnc")
    # LinearSVC probe hyperparameters
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--pca_dim", type=int, default=None,
                   help="PCA before SVC. Default None = skip (n_latent=30 is already compact).")
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--n_jobs", type=int, default=12)
    # Output
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_PROB_DIR, "outputs_probe"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_embeddings", action="store_true")
    # Weights & Biases
    p.add_argument("--wandb_project", type=str, default="scvi-probe")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LinearSVC probe (identical to scFoundation / scGPT probes)
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
            steps.append(("pca", PCA(n_components=pca_dim,
                                     random_state=args.seed)))
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
            sampled_idx = np.random.choice(
                len(x_train), args.max_samples, replace=False
            )
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

    # Aggregate per-class metrics across folds (mean ± std)
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
    """Stratified 70/30 single-split macro-F1.  Runs in O(seconds) per epoch;
    used only to decide which checkpoint has best downstream performance."""
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

    # Cap at 6 000 to keep inference fast on large val sets
    if len(z_val) > 6000:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(z_val), 6000, replace=False)
        z_val = z_val[idx]
        y_val = y_val[idx]

        # The cap can leave rare classes with a single sample, which makes
        # StratifiedShuffleSplit fail. Re-filter after subsampling.
        unique, counts = np.unique(y_val, return_counts=True)
        keep = unique[counts >= 2]
        if len(keep) < 2:
            return 0.0
        if len(keep) < len(unique):
            mask = np.isin(y_val, keep)
            z_val = z_val[mask]
            y_val = y_val[mask]
            y_val = np.searchsorted(keep, y_val)

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(z_val, y_val))
    except ValueError:
        return 0.0

    z_tr, z_te = z_val[tr_idx], z_val[te_idx]
    y_tr, y_te = y_val[tr_idx], y_val[te_idx]

    if args.max_samples and len(z_tr) > args.max_samples:
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
        return float(f1_score(y_te, pipe.predict(z_te),
                              average="macro", zero_division=0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-epoch probe callback (Lightning)
# ---------------------------------------------------------------------------
class PerEpochProbeCallback(Callback):
    """Runs a quick downstream SVC probe after each validation epoch.

    Reads per-epoch ELBO from Lightning callback_metrics and logs it to wandb.
    After each validation epoch, extracts val embeddings from the scvi-tools
    model and runs _quick_probe_f1.  If the F1 improves, saves the current
    model state as the best-downstream checkpoint.

    Attributes
    ----------
    best_f1 : float
        Best quick macro-F1 seen so far.
    best_epoch : int
        1-based epoch at which best_f1 was achieved.
    elbo_history : list of (epoch, elbo_train, elbo_val)
    f1_history   : list of (epoch, quick_f1)
    """

    def __init__(
        self,
        scvi_model,
        adata_val,
        y_val: np.ndarray,
        args,
        best_ckpt_dir: str,
        use_wandb: bool,
    ):
        super().__init__()
        self.scvi_model    = scvi_model
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
        # Lightning current_epoch is 0-based; use 1-based for human readability
        epoch = trainer.current_epoch + 1

        elbo_train = float(trainer.callback_metrics.get("elbo_train",
                                                         float("nan")))
        elbo_val   = float(trainer.callback_metrics.get("elbo_validation",
                                                         float("nan")))
        self.elbo_history.append((epoch, elbo_train, elbo_val))

        # _check_if_trained raises unless is_trained_ is True;
        # set it temporarily — the module weights are valid mid-training.
        _was_trained = self.scvi_model.is_trained_
        self.scvi_model.is_trained_ = True
        try:
            with torch.no_grad():
                z_val = self.scvi_model.get_latent_representation(
                    self.adata_val, give_mean=True
                )
        finally:
            self.scvi_model.is_trained_ = _was_trained

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
            _was_trained = self.scvi_model.is_trained_
            self.scvi_model.is_trained_ = True
            try:
                self.scvi_model.save(self.best_ckpt_dir, overwrite=True)
            finally:
                self.scvi_model.is_trained_ = _was_trained
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
    # Auto-resolve dataset-specific fields from registry (enables wandb sweep).
    # gene_space from registry is only used as fallback when not explicitly passed.
    _explicit = set()
    import sys as _sys
    for _tok in _sys.argv[1:]:
        if _tok.startswith("--gene_space"):
            _explicit.add("gene_space")
    # Sweep variants can be written as <dataset_id>__<gene_space>, e.g.
    # 20w_GSE196830__hgnc. This avoids invalid W&B cartesian products.
    if "__" in args.dataset_id:
        base_dataset_id, variant_gene_space = args.dataset_id.rsplit("__", 1)
        if variant_gene_space in {"ensembl", "hgnc"}:
            args.dataset_id = base_dataset_id
            args.gene_space = variant_gene_space
            _explicit.add("gene_space")

    if args.dataset_id in DATASET_REGISTRY:
        cfg = DATASET_REGISTRY[args.dataset_id]
        if "h5ad" in cfg:
            args.h5ad = cfg["h5ad"]
        if "gene_space" not in _explicit:
            args.gene_space = cfg["gene_space"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    scvi.settings.seed = args.seed

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or time.strftime("probe_%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{args.dataset_id}_{args.gene_space}"
    out_dir  = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")

    # ── Weights & Biases init ─────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # ── Load data ─────────────────────────────────────────────────────
    _cfg = DATASET_REGISTRY.get(args.dataset_id, {})
    if "parquet" in _cfg:
        print(f"Loading from parquet (ATAC gene-activity): {_cfg['parquet']}")
        adata = parquet_to_adata(_cfg["parquet"], _cfg["site_to_gene"], _cfg["label_h5ad"])
    else:
        print(f"Loading h5ad: {args.h5ad}")
        adata = sc.read_h5ad(args.h5ad)
    print(f"  Shape: {adata.shape}  obs columns: {list(adata.obs.columns)}")

    # ── HGNC gene filtering (optional) ───────────────────────────────
    if args.gene_space == "hgnc":
        import pandas as pd
        sym_df = pd.read_csv(args.symbol_map, sep="\t", index_col=0)
        # Keep only genes whose Ensembl ID has a valid HGNC symbol
        keep_mask = np.array([g in sym_df.index for g in adata.var_names])
        n_before = adata.n_vars
        adata = adata[:, keep_mask].copy()
        # Rename var_names to HGNC symbols
        adata.var_names = pd.Index(
            [sym_df.loc[g, "gene_symbol"] for g in adata.var_names]
        )
        print(f"  HGNC filter: {keep_mask.sum()}/{n_before} genes kept, "
              f"renamed to HGNC symbols")

    # ── Build label encoding ──────────────────────────────────────────
    cell_types  = adata.obs["cell_type"].values
    classes     = sorted(set(cell_types))
    type2idx    = {c: i for i, c in enumerate(classes)}
    labels      = np.array([type2idx[c] for c in cell_types], dtype=np.int64)
    n_class     = len(classes)

    # ── Stratified 80/20 split — identical to scFoundation/scGPT split ─
    barcodes           = np.array(adata.obs_names)
    train_bc, val_bc   = _stratified_train_val_split(
        barcodes, labels,
        train_size=args.train_size,
        random_state=args.seed,
    )
    bc2idx    = {bc: i for i, bc in enumerate(barcodes)}
    train_idx = np.array([bc2idx[bc] for bc in train_bc])
    val_idx   = np.array([bc2idx[bc] for bc in val_bc])
    y_val     = labels[val_idx]

    adata_train = adata[train_idx].copy()
    adata_val   = adata[val_idx].copy()
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val")
    print(f"  Classes: {n_class}")

    # ── Set up scVI ───────────────────────────────────────────────────
    # Data matrices are raw integer counts in the current registry; gene_likelihood="nb" is appropriate.
    # Training is fully unsupervised: no cell-type labels used.
    scvi.settings.num_threads = 4
    SCVI.setup_anndata(adata_train)

    model = SCVI(
        adata_train,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        gene_likelihood=args.gene_likelihood,
    )
    print(f"\nscVI model: n_latent={args.n_latent}, n_hidden={args.n_hidden}, "
          f"n_layers={args.n_layers}, gene_likelihood={args.gene_likelihood}")
    print(f"  Parameters: {sum(p.numel() for p in model.module.parameters()):,}")

    # ── Per-epoch probe callback ──────────────────────────────────────
    best_ckpt_dir = os.path.join(out_dir, "scvi_model_best_f1")
    probe_callback = PerEpochProbeCallback(
        scvi_model    = model,
        adata_val     = adata_val,
        y_val         = y_val,
        args          = args,
        best_ckpt_dir = best_ckpt_dir,
        use_wandb     = not args.no_wandb,
    )

    # ── Train ─────────────────────────────────────────────────────────
    # Suppress scVI's verbose lightning output
    scvi.settings.verbosity = 20  # WARNING level
    print(f"\nTraining scVI (unsupervised) — max_epochs={args.max_epochs}, "
          f"early_stopping={args.early_stopping}...", flush=True)
    train_kwargs = dict(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size_train,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        callbacks=[probe_callback],
    )
    if args.lr is not None:
        # scVI takes lr via plan_kwargs (TrainingPlan.__init__), not as a
        # direct train() arg.
        train_kwargs["plan_kwargs"] = {"lr": args.lr}
    model.train(**train_kwargs)
    trained_epochs = model.history["elbo_train"].shape[0]
    final_elbo     = float(model.history["elbo_train"].iloc[-1])
    print(f"  Trained {trained_epochs} epochs | final train ELBO: {final_elbo:.4f}")

    # ── Save final (early-stopped) model ─────────────────────────────
    final_ckpt_dir = os.path.join(out_dir, "scvi_model_final")
    model.save(final_ckpt_dir, overwrite=True)
    print(f"  Final model saved → {final_ckpt_dir}")
    print(f"  Best downstream F1={probe_callback.best_f1:.4f} "
          f"at epoch {probe_callback.best_epoch} → {best_ckpt_dir}")

    # ── Load best checkpoint, extract embeddings, run full probe ─────
    print(f"\nLoading best-downstream checkpoint (epoch {probe_callback.best_epoch})...")
    best_model = SCVI.load(best_ckpt_dir, adata=adata_train)
    # Older checkpoints from this script were saved with is_trained_=False
    # (see callback fix); force it on after load so inference works.
    best_model.is_trained_ = True

    print("Extracting validation embeddings (best checkpoint)...")
    z_val   = best_model.get_latent_representation(adata_val,   give_mean=True)
    print(f"  {z_val.shape[1]} dims  |  {len(y_val)} val samples")

    print("Extracting training embeddings (best checkpoint)...")
    z_train = best_model.get_latent_representation(adata_train, give_mean=True)
    y_train = labels[train_idx]
    print(f"  {len(y_train)} train samples")

    print("Extracting full-dataset embeddings (best checkpoint)...")
    z_all   = best_model.get_latent_representation(adata,       give_mean=True)
    y_all   = labels
    print(f"  {len(y_all)} total samples")

    # ── Full 5-fold SVC on best checkpoint (primary result) ──────────
    print("\nRunning full 5-fold SVC probe on best-downstream checkpoint...")
    cv_result = run_svc_cv(z_val, y_val, args)

    # ── Full 5-fold SVC on final checkpoint (comparison) ─────────────
    print("\nRunning full 5-fold SVC probe on final (early-stopped) checkpoint...")
    z_val_final     = model.get_latent_representation(adata_val, give_mean=True)
    cv_result_final = run_svc_cv(z_val_final, y_val, args)

    primary_checkpoint = "best_downstream"
    if probe_callback.best_f1 <= 0.0:
        print(
            "Quick probe never found a positive-F1 checkpoint; "
            "using final checkpoint as primary result.",
            flush=True,
        )
        primary_checkpoint = "final_fallback"
        cv_result = cv_result_final
        z_val = z_val_final
        z_train = model.get_latent_representation(adata_train, give_mean=True)
        z_all = model.get_latent_representation(adata, give_mean=True)

    # ── Save results ──────────────────────────────────────────────────
    result = {
        # Primary result: best-downstream checkpoint, or final fallback when quick probe fails
        "primary_checkpoint":     primary_checkpoint,
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
        "scvi_trained_epochs":    trained_epochs,
        "scvi_final_elbo":        final_elbo,
        "elbo_history":           probe_callback.elbo_history,
        "f1_history":             probe_callback.f1_history,
        "args":                   vars(args),
        "protocol":               "scvi_train_val_embeddings_5fold_svc_cv_best_or_final_fallback",
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

    # ── Print summary ─────────────────────────────────────────────────
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

    # ── Weights & Biases logging ──────────────────────────────────────
    if not args.no_wandb:
        mean      = cv_result["mean_metrics"]       # best checkpoint
        mean_fin  = cv_result_final["mean_metrics"] # final checkpoint
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
            "scvi_trained_epochs":        trained_epochs,
            "scvi_final_elbo":            final_elbo,
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
        # Per-class accuracy table (mean ± std across folds, test split)
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
