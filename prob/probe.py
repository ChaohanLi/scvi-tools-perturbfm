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
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import scvi
from scvi.model import SCVI

_PROB_DIR = os.path.dirname(os.path.abspath(__file__))


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
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train scVI and evaluate a LinearSVC probe on val embeddings"
    )
    p.add_argument("--h5ad", type=str,
                   default="/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad")
    p.add_argument("--dataset_id", type=str, default="5w_symbol",
                   help="Short dataset tag appended to run_name (e.g. 5w_symbol, "
                        "5w_GSE196830, GSE96583, 10w_GSE196830, 20w_GSE196830)")
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
    p.add_argument("--max_epochs", type=int, default=None,
                   help="scVI training epochs. None uses the built-in heuristic.")
    p.add_argument("--batch_size_train", type=int, default=128,
                   help="Mini-batch size for scVI training")
    p.add_argument("--early_stopping", action="store_true",
                   help="Enable scVI early stopping on ELBO")
    # Gene space selection
    p.add_argument("--gene_space", type=str, default="ensembl",
                   choices=["ensembl", "hgnc"],
                   help="'ensembl': use var_names as-is; "
                        "'hgnc': map Ensembl IDs → HGNC symbols via symbol_map, "
                        "drop genes without a valid HGNC entry")
    p.add_argument("--symbol_map", type=str,
                   default="/lichaohan/readData/gene_id_to_symbol.tsv",
                   help="TSV (gene_id \t gene_symbol) for Ensembl→HGNC mapping. "
                        "Only used when --gene_space hgnc")
    # LinearSVC probe hyperparameters
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--pca_dim", type=int, default=None,
                   help="PCA before SVC. Default None = skip (n_latent=30 is already compact).")
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--n_jobs", type=int, default=16)
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
        return {
            "fold":           fold_idx,
            "train_size":     int(len(x_train)),
            "train_fit_size": int(len(x_train_fit)),
            "test_size":      int(len(x_test)),
            "train":          compute_metrics(y_train, train_preds),
            "test":           compute_metrics(y_test,  test_preds),
            "probe_steps":    list(probe.named_steps.keys()),
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

    return {
        "fold_metrics":           fold_metrics,
        "mean_metrics":           mean_metrics,
        "kept_classes":           [int(x) for x in keep_classes.tolist()],
        "dropped_classes":        dropped_info,
        "n_samples_after_filter": int(len(labels)),
    }


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

    adata_train = adata[train_idx].copy()
    adata_val   = adata[val_idx].copy()
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val")
    print(f"  Classes: {n_class}")

    # ── Set up scVI ───────────────────────────────────────────────────
    # Data is log1p-normalized → use gene_likelihood="normal" (Gaussian).
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

    # ── Train ─────────────────────────────────────────────────────────
    # Suppress scVI's verbose lightning output
    scvi.settings.verbosity = 20  # WARNING level
    print("\nTraining scVI (unsupervised)...", flush=True)
    model.train(
        max_epochs=args.max_epochs,    # None → heuristic (~200 for 40K cells)
        batch_size=args.batch_size_train,
        early_stopping=args.early_stopping,
    )
    trained_epochs = model.history["elbo_train"].shape[0]
    final_elbo     = float(model.history["elbo_train"].iloc[-1])
    print(f"  Trained {trained_epochs} epochs | final train ELBO: {final_elbo:.4f}")

    # ── Extract val embeddings ────────────────────────────────────────
    print("Extracting validation embeddings...")
    z_val = model.get_latent_representation(adata_val, give_mean=True)
    # z_val shape: (n_val, n_latent)
    y_val = labels[val_idx]
    print(f"Embedding shape: {z_val.shape[1]} dims")
    print(f"Validation samples: {len(y_val)}")

    # ── LinearSVC probe ───────────────────────────────────────────────
    cv_result = run_svc_cv(z_val, y_val, args)

    # ── Save results ──────────────────────────────────────────────────
    result = {
        "metrics":                cv_result["mean_metrics"],
        "fold_metrics":           cv_result["fold_metrics"],
        "embedding_dim":          int(z_val.shape[1]),
        "class_names":            classes,
        "type2idx":               type2idx,
        "kept_classes":           cv_result["kept_classes"],
        "dropped_classes":        cv_result["dropped_classes"],
        "n_samples_after_filter": cv_result["n_samples_after_filter"],
        "scvi_trained_epochs":    trained_epochs,
        "scvi_final_elbo":        final_elbo,
        "args":                   vars(args),
        "protocol":               "scvi_train_val_embeddings_5fold_svc_cv",
    }
    with open(os.path.join(out_dir, "probe_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(classes, f, indent=2)
    save_fold_metrics(os.path.join(out_dir, "probe_fold_metrics.csv"), cv_result)

    # Save trained scVI model for future use
    model.save(os.path.join(out_dir, "scvi_model"), overwrite=True)

    if args.save_embeddings:
        np.save(os.path.join(out_dir, "embeddings_val.npy"), z_val)
        np.save(os.path.join(out_dir, "labels_val.npy"),     y_val)

    # ── Print summary ─────────────────────────────────────────────────
    print("\nProbe metrics")
    for split in ["train", "test"]:
        m = cv_result["mean_metrics"][split]
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
        mean = cv_result["mean_metrics"]
        wandb.log({
            "cv_train/accuracy":          mean["train"]["accuracy"],
            "cv_train/balanced_accuracy": mean["train"]["balanced_accuracy"],
            "cv_train/macro_f1":          mean["train"]["macro_f1"],
            "cv_train/weighted_f1":       mean["train"]["weighted_f1"],
            "cv_test/accuracy":           mean["test"]["accuracy"],
            "cv_test/balanced_accuracy":  mean["test"]["balanced_accuracy"],
            "cv_test/macro_f1":           mean["test"]["macro_f1"],
            "cv_test/weighted_f1":        mean["test"]["weighted_f1"],
            "embedding_dim":              int(z_val.shape[1]),
            "n_val_samples":              int(len(y_val)),
            "n_classes_used":             len(cv_result["kept_classes"]),
            "n_classes_dropped":          len(cv_result["dropped_classes"]),
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
        wandb.finish()


if __name__ == "__main__":
    main()
