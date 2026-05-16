## Results Table

### Part A — Classification

| Model | Accuracy | Macro F1 | Parameters | FLOPs |
|---|---|---|---|---|
| StandardCNN | 0.8165 | 0.8104 | 93,962 | 14,904,832 |
| SeparableCNN | **0.8850** | **0.8844** | **31,338** | **3,947,648** |

### Part B — Metric Learning (SeparableCNN, TripletLoss, margin=0.5)

| Metric | Value |
|---|---|
| Best Recall@1 | **0.8410** |
| Epochs | 40 (best at epoch 28) |
| Embedding dim | 64 |

### Part C-1 — Pruning (Global Magnitude, SeparableCNN)

| Sparsity | Non-zero Params | Recall@1 | Retention |
|---|---|---|---|
| 0% (baseline) | 38,302 | 0.8410 | 100.0% |
| 20% | 30,878 | 0.8345 | 99.2% |
| 40% | 23,454 | 0.8150 | 96.9% |
| 50% | 19,742 | 0.7930 | 94.3% |
| 60% | 16,030 | 0.7870 | 93.6% |
| 70% | 12,318 | 0.7455 | 88.6% |
| 80% | 8,606 | 0.6955 | 82.7% |
| 85% | 6,750 | 0.3560 | 42.3% |
| 90% | 4,894 | 0.1020 | 12.1% |

### Part C-2 — Knowledge Distillation

| | Teacher (SeparableCNN) | Student (CompactSeparableCNN) |
|---|---|---|
| Recall@1 | 0.8410 | **0.7855** |
| Parameters | 38,304 | 12,528 |
| FLOPs (approx.) | 3,947,648 | ~1,300,000 |
| Compression | 1.0× | **3.1×** |
| Retention | — | **93.4%** |