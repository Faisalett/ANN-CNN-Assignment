"""
End-to-end project runner.

Executes all three parts in sequence:

    Part A  →  train_classifier.py
    Part B  →  train_metric.py  +  visualize_retrieval.py  +  visualize_embeddings.py
    Part C  →  prune_and_distill.py

The backbone selected at the end of Part A is passed automatically into
Parts B and C.

Usage:
    python run_project.py                         # run everything
    python run_project.py --parts A B             # run only Parts A and B
    python run_project.py --backbone separable_cnn --parts B C  # skip Part A
    python run_project.py --parts C --backbone separable_cnn    # Part C only

Options:
    --parts         Subset of parts to run (A, B, C). Default: all three.
    --backbone      Override the backbone used for Parts B and C.
                    If omitted and Part A ran, the best Part A backbone is
                    selected automatically by accuracy.
    --n_queries     Number of query rows in the retrieval grid (default: 8).
    --top_k         Number of nearest neighbours in the retrieval grid (default: 5).
    --skip_pruning  Skip pruning in Part C.
    --skip_distill  Skip distillation in Part C.
"""

import argparse
import os
import sys

from utils import cprint, format_section_header, format_subsection_header


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_part_a_backbone() -> str:
    """
    Inspect Part A checkpoint directory and return the backbone name whose
    checkpoint has the highest stored val accuracy.  Falls back to
    'separable_cnn' if no checkpoints exist.
    """
    import torch

    ckpt_dir = "checkpoints/part_a"
    if not os.path.isdir(ckpt_dir):
        print("  [run_project] No Part A checkpoints found; defaulting to separable_cnn.")
        return "separable_cnn"

    best_name, best_acc = "separable_cnn", -1.0
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"):
            continue
        ckpt = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu")
        # train_classifier saves model_name but not accuracy directly;
        # we infer from filename.
        name = fname.replace(".pt", "")
        # Try to load accuracy from a companion results file if present
        acc_file = os.path.join(ckpt_dir, f"{name}_acc.txt")
        if os.path.exists(acc_file):
            with open(acc_file) as f:
                acc = float(f.read().strip())
            if acc > best_acc:
                best_acc, best_name = acc, name
        else:
            # No accuracy file — pick separable_cnn as the default recommended
            # backbone per the assignment spec (fewer params, comparable acc).
            if name == "separable_cnn":
                best_name = name

    print(f"  [run_project] Selected Part A backbone: {best_name}")
    return best_name


def _run_part_a() -> str:
    """Train both classifiers and return the name of the best backbone."""
    cprint(format_section_header("PART A — Classification"), color="cyan")

    # Import and run directly (avoids subprocess overhead and keeps the same
    # Python environment / installed packages).
    import train_classifier
    train_classifier.main()

    return _best_part_a_backbone()


def _run_part_b(backbone: str, n_samples : int, n_queries: int, top_k: int, loss : str) -> None:
    cprint(format_section_header(f"PART B — Metric Learning (backbone: {backbone})"), color="cyan")

    cprint(format_subsection_header("Training metric model"), color="cyan")
    import train_metric
    # Patch sys.argv so argparse inside train_metric picks up our backbone
    _argv_ctx(["train_metric.py", "--backbone", backbone,  "--loss", loss], train_metric.main)

    import visualize_retrieval
    cprint(format_subsection_header("Visualize retrieval results"), color="cyan")
    _argv_ctx(
        ["visualize_retrieval.py", "--backbone", backbone,
         "--n_queries", str(n_queries), "--top_k", str(top_k)],
        visualize_retrieval.main,
    )

    cprint(format_subsection_header("Visualize embedding space"), color="cyan")
    import visualize_embeddings
    _argv_ctx(
        ["visualize_embeddings.py", "--backbone", backbone, "--n_samples", str(n_samples)],
        visualize_embeddings.main,
    )


def _run_part_c(backbone: str, skip_pruning: bool, skip_distill: bool) -> None:
    cprint(format_section_header(f"PART C — Compression (backbone: {backbone}, Pruning: {not skip_pruning}, Distilling: {not skip_distill})"), color="cyan")

    import prune_and_distill
    argv = ["prune_and_distill.py", "--backbone", backbone]
    if skip_pruning:
        argv.append("--skip_pruning")
    if skip_distill:
        argv.append("--skip_distill")
    _argv_ctx(argv, prune_and_distill.main)


def _argv_ctx(argv: list[str], fn) -> None:
    """Temporarily replace sys.argv and call fn()."""
    old = sys.argv
    try:
        sys.argv = argv
        fn()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end project runner.")
    parser.add_argument(
        "--parts", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"],
        help="Which parts to run (default: A B C)",
    )
    parser.add_argument(
        "--backbone", default=None,
        choices=["cnn", "separable_cnn"],
        help="Backbone for Parts B/C. Auto-selected from Part A if omitted.",
    )
    parser.add_argument("--loss", default="triplet", choices=["triplet", "contrastive"], help="Metric learning loss to use in Part B (default: triplet)")
    parser.add_argument( "--n_samples", type=int, default=2000, help="Max samples passed to t-SNE in Part B (default: 2000)")
    parser.add_argument("--n_queries", type=int, default=12, help="Number of query rows in the retrieval grid (default: 12)")
    parser.add_argument("--top_k",     type=int, default=10, help="Number of nearest neighbours in the retrieval grid (default: 10)")
    parser.add_argument("--skip_pruning", action="store_true", help="Skip pruning in Part C")
    parser.add_argument("--skip_distill", action="store_true", help="Skip distillation in Part C")
    args = parser.parse_args()

    parts = set(args.parts)
    backbone = args.backbone

    if "A" in parts:
        selected = _run_part_a()
        if backbone is None:
            backbone = selected
    else:
        if backbone is None:
            backbone = _best_part_a_backbone()

    if "B" in parts:
        _run_part_b(backbone, args.n_samples, args.n_queries, args.top_k, args.loss)

    if "C" in parts:
        _run_part_c(backbone, args.skip_pruning, args.skip_distill)

    cprint(format_section_header(f" ALl requested parts complete.\n   Backbone used for B/C : {backbone}\n   Outputs               : results/\n   Checkpoints           : checkpoints/", align='left'), color="green")


if __name__ == "__main__":
    main()