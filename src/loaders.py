import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR_A = "checkpoints/part_a"
CHECKPOINT_DIR_B = "checkpoints/part_b"


def load_backbone(backbone_name: str, embedding_dim: int) -> nn.Module:
    """
    Load Part A weights and swap the head for an embedding head.

    Parameters
    ----------
    backbone_name : str
        The name of the backbone architecture to load (e.g., 'cnn' or 'separable_cnn').
    embedding_dim : int
        The dimensionality of the embedding space for the new head.

    Returns
    -------
    nn.Module
        A PyTorch model initialized with the Part A weights for the specified backbone,
        but with the classification head replaced by an embedding head suitable for metric learning.
        The model is moved to the specified DEVICE (GPU if available, otherwise CPU).
    """

    # Define path to the checkpoint generated in Part A
    ckpt_path = os.path.join(CHECKPOINT_DIR_A, f"{backbone_name}.pt")

    # Ensure the required backbone exists before proceeding
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Part A checkpoint not found at {ckpt_path}. Run train_classifier.py first.")

    # Load the checkpoint onto the active device (CPU or GPU)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Build model with embedding head instead of classification head
    model = build_model(backbone_name, embedding_dim=embedding_dim, input_channels=1)

    # Load the backbone weights from Part A, ignoring the head layers
    state = ckpt["model_state"]
    compatible = {k: v for k, v in state.items() if "head" not in k}
    missing, unexpected = model.load_state_dict(compatible, strict=False)

    # Print summary of loaded weights
    print(f"Loaded Part A weights for '{backbone_name}'")
    if missing:
        print(f"Randomly initialized: {missing}")
    return model.to(DEVICE)


def load_metric_model(backbone_name: str, print_recall: bool = True) -> nn.Module:
    """
    Load the trained metric model from Part B.

    Parameters
    ----------
    backbone_name : str
        The backbone architecture name (e.g., "cnn" or "separable_cnn").
    print_recall : bool
        Whether to print the Recall@1 achieved at training time (default: True).

    Returns
    -------
    nn.Module
        The loaded metric model, set to eval mode and moved to DEVICE.
    """

    # Load the checkpoint of the trained metric model
    ckpt_path = os.path.join(CHECKPOINT_DIR_B, f"{backbone_name}_metric.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Part B checkpoint not found at {ckpt_path}. Run train_metric.py first.")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Build the model architecture, load the state dict move to DEVICE, and set to eval mode
    embedding_dim = ckpt["embedding_dim"]
    model = build_model(backbone_name, embedding_dim=embedding_dim, input_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    if print_recall:
        print(f"  Recall@1 at training time: {ckpt.get('recall_at_1', 'N/A'):.4f}")
    return model


@torch.no_grad()
def embed_loader(model: nn.Module, loader: DataLoader):
    """
    Embed all images from the loader using the model and collect their labels and original images.

    Parameters
    ----------
    model : nn.Module
        The metric model used to compute embeddings.
    loader : DataLoader
        The DataLoader providing batches of (images, labels).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - embeddings: A NumPy array of shape (N, D) with the computed embeddings.
        - labels: A NumPy array of shape (N,) with the corresponding class labels.
    """
    all_emb, all_lbl, all_img = [], [], []
    for images, labels in loader:
        emb = model(images.to(DEVICE))
        all_emb.append(emb.cpu())
        all_lbl.append(labels)
        all_img.append(images)

    return (
        torch.cat(all_emb),  # (N, D)
        torch.cat(all_lbl),  # (N,)
        torch.cat(all_img),  # (N, 1, H, W)
    )


def load_teacher(backbone_name: str, embedding_dim: int) -> nn.Module:
    """
    Load the teacher model from Part B for evaluation.

    Parameters
    ----------
    backbone_name : str
        The backbone architecture name (e.g., "cnn" or "separable_cnn").
    embedding_dim : int
        The dimensionality of the embedding space for the model.

    Returns
    -------
    nn.Module
        The loaded teacher model, set to eval mode and moved to DEVICE.
    """

    # Load the checkpoint of the trained metric model to use as a teacher
    ckpt_path = os.path.join(CHECKPOINT_DIR_B, f"{backbone_name}_metric.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Part B checkpoint not found at {ckpt_path}. "
            "Run train_metric.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Build the model architecture, load the state dict, move to DEVICE, and set to eval mode
    model = build_model(backbone_name, embedding_dim=embedding_dim, input_channels=1)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    print(f"  Loaded teacher: {ckpt_path}")
    print(f"  Teacher Recall@1 (training time): {ckpt.get('recall_at_1', 'N/A'):.4f}")
    return model

