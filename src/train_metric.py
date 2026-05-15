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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import SEED
from data import get_metric_loaders
from losses import TripletLoss, build_triplets
from models import build_model
from utils import cprint, format_section_header

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
LR = 3e-4
EPOCHS = 20
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PART_A_CHECKPOINT_DIR = "checkpoints/part_a"
CHECKPOINT_DIR = "checkpoints/part_b"


def load_backbone(backbone_name: str) -> nn.Module:
    """
    Load Part A weights and swap the head for an embedding head.

    Parameters
    ----------
    backbone_name : str
        The name of the backbone architecture to load (e.g., 'cnn' or 'separable_cnn').

    Returns
    -------
    nn.Module
        A PyTorch model initialized with the Part A weights for the specified backbone,
        but with the classification head replaced by an embedding head suitable for metric learning.
        The model is moved to the specified DEVICE (GPU if available, otherwise CPU).
    """

    # Define path to the checkpoint generated in Part A
    ckpt_path = os.path.join(PART_A_CHECKPOINT_DIR, f"{backbone_name}.pt")

    # Ensure the required backbone exists before proceeding
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Part A checkpoint not found at {ckpt_path}. Run train_classifier.py first.")

    # Load the checkpoint onto the active device (CPU or GPU)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Build model with embedding head instead of classification head
    model = build_model(backbone_name, embedding_dim=EMBEDDING_DIM, input_channels=1)

    # Load the backbone weights from Part A, ignoring the head layers
    state = ckpt["model_state"]
    compatible = {k: v for k, v in state.items() if "head" not in k}
    missing, unexpected = model.load_state_dict(compatible, strict=False)

    # Print summary of loaded weights
    print(f"Loaded Part A weights for '{backbone_name}'")
    if missing:
        print(f"Randomly initialized: {missing}")
    return model.to(DEVICE)


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


def train_metric(backbone_name: str) -> None:
    """
    Train a metric learning model with the specified backbone architecture.

    Parameters
    ----------
    backbone_name : str
        The name of the backbone architecture to use for training (e.g., 'cnn' or 'separable_cnn').
    """

    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    # Load the backbone model with Part A weights and replace the head for metric learning
    model = load_backbone(backbone_name)

    # Initialize the triplet loss criterion, optimizer, and learning rate scheduler
    criterion = TripletLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Get the training and evaluation data loaders for metric learning
    train_loader, eval_loader = get_metric_loaders()

    # Print training header with model name, parameter count, and FLOP count
    metric_learning_section = f"Part B — Metric learning: {backbone_name}\n    Embedding dim : {EMBEDDING_DIM}\n    Loss          : {TripletLoss (margin=0.3)}"
    cprint(format_section_header(metric_learning_section, align='left', width=55)[:-1], color="yellow")

    best_recall = 0.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        # Set model to training mode
        model.train()

        total_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Zero the gradients before backpropagation
            optimizer.zero_grad()

            # Compute embeddings for the current batch of images
            embeddings = model(images)

            # Build triplets (anchor, positive, negative) from the embeddings and labels
            triplets = build_triplets(embeddings, labels)
            if triplets is None:
                continue
            anchors, positives, negatives = triplets

            # Compute the triplet loss for the current batch
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

    # Set up argument parser to allow selection of backbone architecture for training
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default="separable_cnn", # This was the better-performing backbone in Part A.
        choices=["cnn", "separable_cnn"],
        help="Which Part A backbone to use (default: separable_cnn)",
    )
    args = parser.parse_args()

    # Start the metric learning training process with the specified backbone architecture
    train_metric(args.backbone)


if __name__ == "__main__":
    main()