# Compact Fashion Vision System

End-to-end computer vision pipeline built on FashionMNIST across three stages: backbone comparison (Part A), metric learning for retrieval (Part B), and model compression (Part C).

---

## Project Structure

```
├── checkpoints/
│   ├── part_a/                # StandardCNN and SeparableCNN classification checkpoints
│   ├── part_b/                # Metric learning checkpoint
│   └── part_c/                # Pruned checkpoints and distilled student checkpoint
│
├── data/                      # Dataset files from FashionMNIST, not included in the repo, automatically downloaded by `data.py`, DO NOT MODIFY
│
├── results/                   # All output plots and logs
│   ├── train_classifier_log.txt
│   ├── train_metric_log.txt
│   ├── distillation_log.txt
│   ├── pruning_log.txt
│   ├── retrieval_grid.png
│   ├── embedding_tsne.png
│   └── pruning_curve.png
│
├── src/                       # Source code for all parts
│   ├── config.py              # Provided fixed experiment constants (seed, sizes, model names) (do not modify)
│   ├── data.py                # Provided shared data-loading utilities (do not modify)
│   ├── models.py              # Model definitions: StandardCNN, SeparableCNN, CompactSeparableCNN
│   │
│   ├── losses.py              # ContrastiveLoss and TripletLoss definitions.
│   ├── loaders.py             # Checkpoint loading utilities (load_backbone, load_teacher, embed_loader)
│   ├── utils.py               # Terminal formatting and logging helpers (My own utility functions I use across multiple projects)
│   │ 
│   ├── train_classifier.py        # Part A — train and compare CNN vs SeparableCNN
│   ├── train_metric.py            # Part B — fine-tune backbone for retrieval
│   ├── visualize_retrieval.py     # Part B — nearest-neighbour retrieval grid
│   ├── visualize_embeddings.py    # Part B — t-SNE embedding space plot
│   ├── prune_and_distill.py       # Part C — pruning and knowledge distillation
│   └── run_project.py             # Optional end-to-end runner for all parts
│
├── requirements.txt             # Python dependencies
├── report.md                    # Project report
├── result_table.md              # Results table (also included in the report)
└── README.md                 
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.1`, `torchvision>=0.16`, `scikit-learn>=1.3`, `numpy>=1.24`, `matplotlib`, `Pillow`

---

## Running the Project

### Option 1 — End-to-end runner (recommended)

Runs all three parts in sequence, automatically passing the best Part A backbone into Parts B and C:

```bash
python run_project.py
```

**Available options:**

| Flag | Default | Description                                                |
|---|---|------------------------------------------------------------|
| `--parts A B C` | `A B C` | Which parts to run                                         |
| `--backbone` | auto from Part A | Override backbone for Parts B/C (`cnn` or `separable_cnn`) |
| `--loss` | `triplet` | Metric learning loss (`triplet` or `contrastive`)          |
| `--n_samples` | `2000` | Max samples passed to t-SNE in Part B                                    |
| `--n_queries` | `12` | Number of query rows in the retrieval grid                 |
| `--top_k` | `10` | Number of nearest neighbours in the retrieval grid         |
| `--skip_pruning` | off | Skip pruning in Part C                                     |
| `--skip_distill` | off | Skip distillation in Part C                                |

**Examples:**

```bash
# Run everything with defaults
python run_project.py

# Run only Parts B and C using a pre-trained Part A backbone
python run_project.py --parts B C --backbone separable_cnn

# Run Part C only, skipping distillation
python run_project.py --parts C --backbone separable_cnn --skip_distill

# Run with contrastive loss instead of triplet
python run_project.py --loss contrastive
```

---

### Option 2 — Run each part individually

#### Part A — Classification

```bash
python train_classifier.py
```

Trains both `StandardCNN` and `SeparableCNN` on a fixed 12,000-sample subset, evaluates on 2,000 samples, and prints a comparison table. Saves checkpoints to `checkpoints/part_a/`.

#### Part B — Metric Learning

```bash
python train_metric.py # Defaults to `separable_cnn` backbone and `triplet` loss
python train_metric.py --backbone cnn --loss contrastive
```

Loads the Part A checkpoint, replaces the classification head with a 64-dim embedding head, and fine-tunes with triplet or contrastive loss. Saves checkpoint to `checkpoints/part_b/`.

```bash
# Visualise nearest-neighbour retrieval grid
python visualize_retrieval.py # Defaults to `separable_cnn` backbone, 12 query rows, and top-10 neighbours
python visualize_retrieval.py --backbone separable_cnn --n_queries 5 --top_k 5

# Visualise embedding space with t-SNE
python visualize_embeddings.py # Defaults to `separable_cnn` backbone and 2000 samples for t-SNE
python visualize_embeddings.py --backbone cnn --n_samples 3000
```


#### Part C — Compression

```bash
# Run both pruning and distillation
python prune_and_distill.py # Defaults to `separable_cnn` backbone and both pruning and distillation

# Run only pruning
python prune_and_distill.py --skip_distill

# Run only distillation
python prune_and_distill.py --skip_pruning
```

Saves pruned checkpoints to `checkpoints/part_c/pruned_<sparsity>.pt` and the distilled student to `checkpoints/part_c/student_distilled.pt`.

---

## Fixed Settings

The following must **not** be changed to maintain comparability across students:

| Setting | Value |
|---|---|
| `SEED` | `42` |
| `PART_A_MODELS` | `("cnn", "separable_cnn")` |
| `COMPRESSION_STUDENT_MODEL` | `"compact_separable_cnn"` |
| `CLASSIFICATION_TRAIN_SIZE` | `12000` |
| `CLASSIFICATION_VAL_SIZE` | `2000` |
| `METRIC_TRAIN_SIZE` | `10000` |
| `METRIC_EVAL_SIZE` | `2000` |

The following are safe to change:

| Setting | Value |
|---|---|
| `CLASSIFICATION_BATCH_SIZE` | `64` |
| `METRIC_BATCH_SIZE` | `64` |

- Learning rates and number of epochs
- Choice of triplet vs contrastive loss

---

## Outputs

| File | Generated by | Description |
|---|---|---|
| `checkpoints/part_a/*.pt` | `train_classifier.py` | Best classification checkpoint per model |
| `checkpoints/part_b/*_metric.pt` | `train_metric.py` | Best retrieval checkpoint |
| `checkpoints/part_c/pruned_*.pt` | `prune_and_distill.py` | Pruned model at each sparsity level |
| `checkpoints/part_c/student_distilled.pt` | `prune_and_distill.py` | Distilled compact student |
| `results/retrieval_grid.png` | `visualize_retrieval.py` | Query + top-K neighbour grid |
| `results/embedding_tsne.png` | `visualize_embeddings.py` | t-SNE embedding space plot |
| `results/pruning_curve.png` | `prune_and_distill.py` | Recall@1 vs sparsity curve |
| `results/train_classifier_log.txt` | `train_classifier.py` | Full Part A training log |
| `results/train_metric_log.txt` | `train_metric.py` | Full Part B training log |

---

## Submission Deadline

**June 5th.** Submit Python source files, the report, a result table, trained checkpoints, and generated plots.