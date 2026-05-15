"""
Part B — Retrieval Visualisation

For a set of query images, finds the top-K nearest neighbours in the eval
embedding space and saves a grid image showing:

    [query | nn1 | nn2 | ... | nnK]

One row per query.  Neighbours with the correct class are highlighted with a
green border; incorrect ones with red.

Usage:
    python visualize_retrieval.py --backbone separable_cnn
    python visualize_retrieval.py --backbone separable_cnn --n_queries 10 --top_k 5

Output:
    results/retrieval_grid.png
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image

from data import get_metric_loaders
from loaders import load_metric_model, embed_loader
from utils import colorize, cprint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results"

# FashionMNIST classes in order
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def add_border(img: torch.Tensor, color: tuple[float, float, float], width: int = 3) -> torch.Tensor:
    """
    Add a coloured border to a single image tensor. The input image is expected to be (C, H, W) with C=1 or C=3. The border is added in-place.

    Parameters
    ----------
    img : torch.Tensor
        The input image tensor of shape (C, H, W).
    color : tuple[float, float, float]
        The RGB color of the border, with values in [0.0, 1. 0].
    width : int
        The width of the border in pixels (default: 3).

    Returns
    -------
    torch.Tensor
        The image tensor with the added border, of shape (3, H, W).
    """
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    r, g, b = color

    # Add border by setting the pixel values in the border region to the specified color
    img[:, :width, :] = torch.tensor([r, g, b]).view(3, 1, 1)
    img[:, -width:, :] = torch.tensor([r, g, b]).view(3, 1, 1)
    img[:, :, :width] = torch.tensor([r, g, b]).view(3, 1, 1)
    img[:, :, -width:] = torch.tensor([r, g, b]).view(3, 1, 1)
    return img


def unnormalise(img: torch.Tensor) -> torch.Tensor:
    """
    Unnormalise an image tensor that was normalised with mean=0.2860 and std=0.3530.
    The input img is expected to be in the range [0, 1] after normalisation, and the output will be clamped to [0, 1].

    Parameters
    ----------
    img : torch.Tensor
        The normalised image tensor of shape (C, H, W) with values in [0, 1].

    Returns
    -------
    torch.Tensor
        The unnormalised image tensor of shape (C, H, W) with values cl
    """
    mean, std = 0.2860, 0.3530
    return torch.clamp(img * std + mean, 0.0, 1.0)


def build_retrieval_grid(query_imgs: torch.Tensor, query_labels: torch.Tensor, db_imgs: torch.Tensor,
                         db_labels: torch.Tensor, db_embs: torch.Tensor, query_embs: torch.Tensor,
                         top_k: int = 10) -> torch.Tensor:
    """
    Build a grid of query images and their top-K nearest neighbours from the database, with coloured borders indicating correctness.

    Parameters
    ----------
    query_imgs : torch.Tensor
        Tensor of shape (Q, C, H, W) containing the query images.
    query_labels : torch.Tensor
        Tensor of shape (Q,) containing the class labels for the query images.
    db_imgs : torch.Tensor
        Tensor of shape (N, C, H, W) containing the database images.
    db_labels : torch.Tensor
        Tensor of shape (N,) containing the class labels for the database images.
    db_embs : torch.Tensor
        Tensor of shape (N, D) containing the embeddings for the database images.
    query_embs : torch.Tensor
        Tensor of shape (Q, D) containing the embeddings for the query images.
    top_k : int
        The number of nearest neighbours to retrieve for each query (default: 10).

    Returns
    -------
    torch.Tensor
        A tensor of shape (Q*(top_k+1), 3, H, W) containing the query images and their top-K neighbours with borders, ready to be arranged in a grid
    """

    # Compute cosine similarities between query embeddings and database embeddings
    q_norm = F.normalize(query_embs, dim=1)
    db_norm = F.normalize(db_embs, dim=1)
    sims = q_norm @ db_norm.T

    rows = []
    for q_idx in range(query_imgs.size(0)):
        sim_row = sims[q_idx].clone()

        # Exclude the query image itself from the nearest neighbours by setting its similarity to -inf
        sim_row[sim_row.argmax()] = -float("inf")

        # Get the indices of the top-K nearest neighbours for the current query
        top_indices = sim_row.topk(top_k).indices

        # Query image — white border
        q_img = unnormalise(query_imgs[q_idx])
        q_img = add_border(q_img.clone(), (1.0, 1.0, 1.0), width=4)
        rows.append(q_img)

        # Nearest neighbours — green border if correct class, red if wrong
        for nn_idx in top_indices:
            nn_img = unnormalise(db_imgs[nn_idx])
            correct = db_labels[nn_idx].item() == query_labels[q_idx].item()
            colour = (0.0, 0.9, 0.0) if correct else (0.9, 0.0, 0.0)
            rows.append(add_border(nn_img.clone(), colour, width=4))

    return torch.stack(rows)  # (Q*(top_k+1), 3, H, W)


def upscale_grid(grid_np: np.ndarray, scale_factor: int = 4) -> Image:
    """
    Upscale the final grid using nearest-neighbor interpolation to keep pixel edges crisp for the report.

    Parameters
    ----------
    grid_np : np.ndarray
        The input grid as a NumPy array of shape (Hq * (top_k+1), Wq, 3) with dtype uint8.
    scale_factor : int
        The factor by which to upscale the grid (default: 4).

    Returns
    -------
    Image
        The upscaled grid as a PIL Image object.
    """
    img = Image.fromarray(grid_np)
    w, h = img.size

    # Use Resampling.NEAREST to avoid blurring the FashionMNIST pixels
    return img.resize((w * scale_factor, h * scale_factor), resample=Image.NEAREST)


def visualize_retrieval(backbone_name: str, n_queries: int = 12, top_k: int = 10) -> None:
    """
    Visualise the retrieval results by building a grid of query images and their top-K nearest neighbours, with coloured borders indicating correctness.

    Parameters
    ----------
    backbone_name : str
        The name of the backbone architecture used for the metric model (e.g., "cnn" or "separable_cnn").
    n_queries : int
        The number of query images (rows) to include in the grid (default: 12).
    top_k : int
        The number of nearest neighbours to show for each query (default: 10).
    """

    # Load the trained metric model and the eval DataLoader and embed the eval set to get embeddings and labels
    model = load_metric_model(backbone_name)
    _, eval_loader = get_metric_loaders()

    print("  Embedding eval set …")
    db_embs, db_labels, db_imgs = embed_loader(model, eval_loader)

    # Normalise database embeddings for cosine similarity
    db_embs = F.normalize(db_embs, dim=1)

    # Pick one query per class so the grid is representative
    chosen_indices = []
    seen_classes = set()
    perm = torch.randperm(len(db_labels))
    for idx in perm:
        cls = db_labels[idx].item()

        # Only add the first occurrence of each class to the chosen indices until we have n_queries
        if cls not in seen_classes:
            chosen_indices.append(idx.item())
            seen_classes.add(cls)

        # Stop once we have enough queries (e.g., 8) to fill the grid, even if there are more classes
        if len(chosen_indices) == n_queries:
            break

    # Extract the query images, labels, and embeddings for the chosen indices
    query_imgs = db_imgs[chosen_indices]
    query_labels = db_labels[chosen_indices]
    query_embs = db_embs[chosen_indices]

    # Build the retrieval grid tensor containing the query images and their top-K nearest neighbours with coloured borders
    print(f"  Building retrieval grid ({n_queries} queries, top-{top_k}) …")
    grid_tensor = build_retrieval_grid(
        query_imgs, query_labels,
        db_imgs, db_labels, db_embs,
        query_embs, top_k=top_k,
    )
    grid = make_grid(grid_tensor, nrow=top_k + 1, padding=2, pad_value=0.5)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Save the grid image to the output directory with upscaling for high resolution in the report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "retrieval_grid.png")

    print(f"  Upscaling and saving grid...")
    final_image = upscale_grid(grid_np, scale_factor=3)
    final_image.save(out_path)

    cprint(f"  Saved retrieval grid → {out_path}")
    print("  (" + colorize('green border = correct class') + ', ' + colorize('red border = wrong class', 'red') + ")")


def main() -> None:
    """
    Main function to parse command-line arguments and run the retrieval visualisation.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="separable_cnn",
                        choices=["cnn", "separable_cnn"])
    parser.add_argument("--n_queries", type=int, default=12,
                        help="Number of query rows in the grid (default: 12)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of nearest neighbours to show (default: 10)")
    args = parser.parse_args()

    # Run the retrieval visualisation with the specified arguments
    visualize_retrieval(args.backbone, args.n_queries, args.top_k)


if __name__ == "__main__":
    main()
