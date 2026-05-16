"""
Part B — Metric Learning for Retrieval

Loads the best Part A checkpoint, replaces the classification head with an
embedding head, and fine-tunes with triplet loss.

Usage:
    python train_metric.py --backbone separable_cnn
    python train_metric.py --backbone cnn

The script evaluates Recall@1 on the eval split after every epoch and saves
the best model to:
    checkpoints/part_b/<backbone>_metric.pt
"""


import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import SEED
from data import get_metric_dataloader, get_retrieval_eval_dataloader
from losses import TripletLoss, ContrastiveLoss
from loaders import load_backbone
from utils import cprint, format_section_header, Logger
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
LR = 5e-5
BACK_BONE_LR = 1e-5  # or None to use LR defaults
HEAD_LR = 1e-4  # or None to use LR defaults
EPOCHS = 40
WARMUP_EPOCHS = 10
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints/part_b"


@torch.no_grad()
def compute_recall_at_1(model: nn.Module, loader: DataLoader) -> float:
    """
    Compute Recall@1: for each query embedding, check whether the nearest
    neighbour (excluding itself) shares the same class.

    Parameters
    ----------
    model : nn.Module
        The metric learning model used to compute embeddings for the evaluation dataset.
    loader : DataLoader
        A DataLoader providing batches of images and their corresponding labels from the evaluation dataset.

    Returns
    -------
    float
        The Recall@1 metric
    """

    # Set model to evaluation mode and collect all embeddings and labels from the loader
    model.eval()
    all_embeddings, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        emb = model(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    # Concatenate all embeddings and labels into single tensors
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    # L2-normalise for cosine similarity via dot product
    embeddings = nn.functional.normalize(embeddings, dim=1)

    # Pairwise distance matrix — process in chunks to avoid OOM
    chunk_size = 256
    hits = 0
    n = embeddings.size(0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]
        dists = torch.cdist(chunk, embeddings)

        # Mask self-distance
        for local_i, global_i in enumerate(range(start, end)):
            dists[local_i, global_i] = float("inf")

        nn_idx = dists.argmin(dim=1)
        correct = labels[start:end] == labels[nn_idx]
        hits += correct.sum().item()

    return hits / n


def train_metric(backbone_name: str, loss_type: str = "triplet") -> None:
    """
    Train a metric learning model with the specified backbone architecture.

    Parameters
    ----------
    backbone_name : str
        The name of the backbone architecture to use for training (e.g., 'cnn' or 'separable_cnn').
    loss_type : str
        The type of loss to use for training (default: 'triplet'). Options include 'triplet' for Triplet Loss and 'contrastive' for Contrastive Loss.
    """

    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    # Load the backbone model with Part A weights and replace the head for metric learning
    model = load_backbone(backbone_name, EMBEDDING_DIM)

    # Freeze everything except the head for warmup
    for name, p in model.named_parameters():
        if "head" not in name:
            p.requires_grad_(False)

    # Initialize the loss criterion based on the specified loss type (triplet or contrastive)
    if loss_type == "contrastive":
        criterion = ContrastiveLoss(margin=1.0)
    else:
        criterion = TripletLoss(margin=0.5)

    # Set up the optimizer (Adam) with diff learning rates if specified, otherwise use the same LR for all parameters
    if BACK_BONE_LR is not None or HEAD_LR is not None:
        print(f"Using custom learning rates — backbone: {BACK_BONE_LR or LR}, head: {HEAD_LR or LR}")
        backbone_params = list(model.stem.parameters()) + list(model.features.parameters())
        head_params = list(model.head.parameters())
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": BACK_BONE_LR or LR},
            {"params": head_params, "lr": HEAD_LR or LR}
        ])
    else:
        print(f"Using learning rates for Adam optimizer: {LR} for all parameters")
        optimizer = optim.Adam(model.parameters(), lr=LR)

    # Set up learning rate scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Get the training and evaluation data loaders for metric learning
    train_loader = get_metric_dataloader(loss_name=loss_type)
    eval_loader = get_retrieval_eval_dataloader()

    # Print training header with model name, embedding dim, and loss type
    metric_learning_section = f"Part B — Metric learning: {backbone_name}\n    Embedding dim : {EMBEDDING_DIM}\n    Loss          : {criterion}"
    cprint(format_section_header(metric_learning_section, align='left', width=55)[:-1], color="yellow")

    best_recall = 0.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):

        # Unfreeze the full model after warmup epochs or immediately if WARMUP_EPOCHS is set to 0
        if epoch == WARMUP_EPOCHS + 1:
            for p in model.parameters():
                p.requires_grad_(True)
            if WARMUP_EPOCHS > 0:
                print("  [warmup done] backbone unfrozen")

        # Set model to training mode
        model.train()

        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            # Zero the gradients before backpropagation
            optimizer.zero_grad()

            # handle both loss types in the same loop by unpacking the batch accordingly and computing the loss based on the specified loss type
            if loss_type == "contrastive":
                image_a, image_b, pair_label = batch
                image_a = image_a.to(DEVICE)
                image_b = image_b.to(DEVICE)
                pair_label = pair_label.to(DEVICE)

                z1 = F.normalize(model(image_a), dim=1)
                z2 = F.normalize(model(image_b), dim=1)
                loss = criterion(z1, z2, pair_label)

            else:  # triplet
                anchors_img, positives_img, negatives_img = batch
                anchors_img = anchors_img.to(DEVICE)
                positives_img = positives_img.to(DEVICE)
                negatives_img = negatives_img.to(DEVICE)

                anchors = F.normalize(model(anchors_img), dim=1)
                positives = F.normalize(model(positives_img), dim=1)
                negatives = F.normalize(model(negatives_img), dim=1)
                loss = criterion(anchors, positives, negatives)

            # Backpropagate the loss and update model parameters
            loss.backward()

            # Perform an optimization step to update the model parameters based on the computed gradients
            optimizer.step()

            # Accumulate the total loss for reporting after the epoch
            total_loss += loss.item()
            n_batches += 1

        # Update the learning rate scheduler at the end of the epoch
        scheduler.step()

        # Compute Recall@1 on the evaluation set after the epoch and print the results
        recall = compute_recall_at_1(model, eval_loader)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:02d}/{EPOCHS} | loss {avg_loss:.4f} | Recall@1 {recall:.4f}")

        # Update best model if Recall@1 has improved
        if recall > best_recall:
            best_recall = recall
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save the best model checkpoint with Recall@1 in the filename and metadata
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{backbone_name}_metric.pt")
    torch.save(
        {
            "model_state": best_state,
            "model_name": backbone_name,
            "embedding_dim": EMBEDDING_DIM,
            "recall_at_1": best_recall,
        },
        ckpt_path,
    )

    # Print summary of training results and checkpoint information
    cprint(f"\n  Best Recall@1: {best_recall:.4f}")
    cprint(f"  Saved checkpoint → {ckpt_path}")


def main() -> None:
    """
    Main function to parse command-line arguments and initiate metric learning training.
    """
    original_terminal = sys.stdout
    logger_instance = Logger("results/train_metric_log.txt")
    sys.stdout = logger_instance

    # Set up argument parser to allow selection of backbone architecture for training
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default="separable_cnn", # This was the better-performing backbone in Part A.
        choices=["cnn", "separable_cnn"],
        help="Which Part A backbone to use (default: separable_cnn)",
    )
    parser.add_argument("--loss", default="triplet", choices=["triplet", "contrastive"],
                        help="Metric learning loss to use (default: triplet)")
    args = parser.parse_args()

    # Start the metric learning training process with the specified backbone architecture
    train_metric(args.backbone)

    sys.stdout = original_terminal
    logger_instance.log.close()


if __name__ == "__main__":
    main()