"""
Part B — Embedding Space Visualisation

Projects the eval-set embeddings down to 2-D with t-SNE and saves a
scatter plot coloured by FashionMNIST class.

Usage:
    python visualize_embeddings.py --backbone separable_cnn
    python visualize_embeddings.py --backbone separable_cnn --n_samples 1000

Output:
    results/embedding_tsne.png
"""

import argparse
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data import get_metric_loaders
from config import SEED
from loaders import load_metric_model, embed_loader
from utils import cprint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results"

# FashionMNIST classes in order
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# One visually distinct colour per class
CLASS_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def run_tsne(embeddings: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
    """
    Run t-SNE on the given embeddings to reduce them to n_components dimensions.

    Parameters
    ----------
    embeddings : np.ndarray
        The input embeddings of shape (N, D) to be projected.
    n_components : int, optional
        The number of dimensions for the t-SNE output (default is 2).
    perplexity : float, optional
        The perplexity parameter for t-SNE (default is 30.0).

    Returns
    -------
    np.ndarray
        The t-SNE coordinates of shape (N, n_components).
    """
    print(f"  Running t-SNE on {embeddings.shape[0]} points …")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(coords: np.ndarray, labels: np.ndarray, backbone_name: str, out_path: str) -> None:
    """
    Plot the t-SNE coordinates with points coloured by class.

    Parameters
    ----------
    coords : np.ndarray
        The t-SNE coordinates of shape (N, 2).
    labels : np.ndarray
        The class labels of shape (N,).
    backbone_name : str
        The name of the backbone architecture (used in the plot title).
    out_path : str
        The file path where the plot image will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each class separately for better colour control
    for cls_idx in range(10):
        mask = labels == cls_idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=CLASS_COLOURS[cls_idx],
            s=6,
            alpha=0.6,
            linewidths=0,
            label=CLASSES[cls_idx],
        )

    # Create custom legend patches for the classes
    legend_patches = [
        mpatches.Patch(color=CLASS_COLOURS[i], label=CLASSES[i]) for i in range(10)
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        markerscale=2,
        framealpha=0.8,
    )
    ax.set_title(f"t-SNE of {backbone_name} embeddings (Part B)", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    cprint(f"Saved t-SNE plot → {out_path}")


def visualize_embeddings(backbone_name: str, n_samples: int = 2000) -> None:
    """
    Main function to visualize embeddings using t-SNE.

    Parameters
    ----------
    backbone_name : str
        The name of the backbone architecture (e.g., "cnn" or "separable_cnn").
    n_samples : int, optional
        The maximum number of samples to pass to t-SNE (default is 2000).
    """

    # Load the trained metric model and the eval DataLoader and embed the eval set to get embeddings and labels
    model = load_metric_model(backbone_name, print_recall=False)
    _, eval_loader = get_metric_loaders()

    print("  Embedding eval set …")
    embeddings, labels, _ = embed_loader(model, eval_loader)

    # L2-normalise before t-SNE (embeddings live on the unit hypersphere)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    # Subsample if needed (t-SNE is O(N^2))
    if len(embeddings) > n_samples:
        # Use a fixed random seed for reproducibility
        rng = np.random.default_rng(SEED)

        # Randomly choose n_samples indices without replacement
        idx = rng.choice(len(embeddings), size=n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        print(f"  Using {n_samples} random samples for t-SNE")

    # Run t-SNE to get 2-D coordinates and plot them coloured by class
    coords = run_tsne(embeddings)

    # Create the output directory if it doesn't exist and save the t-SNE plot to a file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "embedding_tsne.png")
    plot_tsne(coords, labels, backbone_name, out_path)


def main() -> None:
    """
    Main function to parse command-line arguments and run the embedding visualisation.
    """

    # Set up an argument parser to allow users to specify the backbone architecture and the number of samples for t-SNE.
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="separable_cnn",
                        choices=["cnn", "separable_cnn"])
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Max samples passed to t-SNE (default: 2000)")
    args = parser.parse_args()

    # Call the visualize_embeddings function with the specified backbone and number of samples to generate the t-SNE plot.
    visualize_embeddings(args.backbone, args.n_samples)


if __name__ == "__main__":
    main()
