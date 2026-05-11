# scVI Probe

Unsupervised **scVI** (VAE, n_latent=30) trained on train cells, then a **LinearSVC** probe evaluated via 5-fold cross-validation on val embeddings.

---

## Supported Datasets

| `--dataset_id`   | h5ad path                                                 | Cells | Classes | Notes               |
|------------------|-----------------------------------------------------------|-------|---------|---------------------|
| `5w_symbol`      | `readData/5w_allcelltype_anno_symbol.h5ad`               | 50 k  | 29      | log1p normalized (NB still converges) |
| `5w_GSE196830`   | `readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad`         | 50 k  | 29      | raw counts — ideal for NB |
| `GSE96583`       | `readData/GSE96583_PBMC/GSE96583_merged_dedup.h5ad`      | 41 k  | 8       | raw counts — ideal for NB |
| `10w_GSE196830`  | `readData/10w_PBMC_GSE196830/10w_allcelltype.h5ad`       | 100 k | 29      | raw counts — ideal for NB |
| `20w_GSE196830`  | `readData/20w_PBMC_GSE196830/20w_allcelltype.h5ad`       | 200 k | 29      | raw counts — ideal for NB |

> **No `--preprocess` needed for scVI.** `gene_likelihood="nb"` (default) is numerically stable for both raw counts and log1p data.

---

## Quick Start

### Interactive (foreground)

```bash
bash run_probe.sh
```

Edit `run_probe.sh` to select the dataset (uncomment the block you want).

### Background (nohup)

```bash
nohup bash run_probe.sh > run_probe.log 2>&1 &
tail -f run_probe.log
```

### Manual CLI

```bash
python probe.py \
    --h5ad /lichaohan/readData/5w_allcelltype_anno_symbol.h5ad \
    --dataset_id 5w_symbol \
    --run_name my_run \
    --wandb_project scvi-probe

# Raw count dataset (preferred for NB)
python probe.py \
    --h5ad /lichaohan/readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad \
    --dataset_id 5w_GSE196830 \
    --run_name my_run \
    --wandb_project scvi-probe
```

---

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--h5ad` | `5w_allcelltype_anno_symbol.h5ad` | Input h5ad path |
| `--dataset_id` | `5w_symbol` | Tag appended to wandb run name and output dir |
| `--gene_space` | `ensembl` | `ensembl` keeps all genes as-is; `hgnc` maps Ensembl→HGNC via `--symbol_map` |
| `--symbol_map` | `None` | Path to Ensembl→HGNC TSV (`gene_id_to_symbol.tsv`); required when `--gene_space hgnc` |
| `--n_latent` | `30` | scVI latent dimension |
| `--n_hidden` | `128` | scVI encoder/decoder hidden size |
| `--n_layers` | `2` | Number of scVI layers |
| `--gene_likelihood` | `nb` | `nb` (recommended), `zinb`, or `normal` |
| `--max_epochs` | auto | scVI training epochs (None = built-in heuristic) |
| `--batch_size_train` | `128` | Mini-batch size for scVI training |
| `--early_stopping` | off | Enable scVI ELBO early stopping |
| `--run_name` | auto timestamp | wandb / output folder name prefix |
| `--wandb_project` | `scvi-probe` | wandb project name |
| `--n_jobs` | `16` | CPU cores for parallel fold evaluation |
| `--no_wandb` | off | Disable wandb logging |
| `--save_embeddings` | off | Save `embeddings_val.npy` and `labels_val.npy` in output dir (required for `visualize.py`) |

---

## Protocol

```
scVI (unsupervised, no labels used during training)
  └─ trained on 80% train cells
  └─ get_latent_representation(val_cells, give_mean=True) → (n_val, 30)

val embeddings (30-dim)
  └─ 5-fold StratifiedKFold (shuffle, seed=42)
       └─ StandardScaler → LinearSVC(dual=False, max_iter=2000)
            (no PCA: n_latent=30 is already compact)
            folds are run in parallel (--n_jobs controls core count)
            if >5000 train samples per fold → subsampled to 5000
```

---

## Output

Results are saved to `outputs_probe/<run_name>_<dataset_id>_<gene_space>/`:

| File | Description |
|------|-------------|
| `probe_metrics.json` | Scalar means, per-fold list, `per_class_cv`, kept/dropped classes, scVI info |
| `probe_fold_metrics.csv` | Per-fold train/test accuracy and F1 in CSV |
| `class_names.json` | Ordered list of cell-type label strings |
| `scvi_model/` | Saved scVI model weights |
| `embeddings_val.npy` | Val-set latent vectors — only written with `--save_embeddings` |
| `labels_val.npy` | Val-set integer labels — only written with `--save_embeddings` |

### wandb

**Scalar metrics** (`cv_train/*`, `cv_test/*`): `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`, `embedding_dim`, `scvi_trained_epochs`, `scvi_final_elbo`.

**`fold_metrics` Table**: one row per fold — fold index, train/test size, train/test accuracy and macro-F1.

**`per_class_metrics` Table**: one row per kept class — `class_name`, `mean_f1 ± std_f1`, `mean_recall` (= per-class accuracy), `mean_precision`, `mean_support` (avg test samples across folds).
