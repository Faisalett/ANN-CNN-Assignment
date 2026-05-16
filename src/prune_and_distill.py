"""
Part C — Compression: Pruning + Knowledge Distillation

Two independent subsections, both run by default:

  1. Pruning
     Applies iterative global unstructured magnitude pruning to the Part B
     retrieval model at increasing sparsity levels and reports Recall@1 at
     each level.  Uses torch.nn.utils.prune (no extra dependencies).

  2. Knowledge Distillation
     Distils the Part B teacher into the fixed CompactSeparableCNN student
     using an embedding-alignment loss (MSE between L2-normalised embeddings).
     Evaluates Recall@1 of the student before and after distillation.

Usage:
    python prune_and_distill.py --backbone separable_cnn
    python prune_and_distill.py --backbone separable_cnn --skip_pruning
    python prune_and_distill.py --backbone separable_cnn --skip_distill

Output:
    checkpoints/part_c/pruned_<sparsity>.pt   (one per pruning level)
    checkpoints/part_c/student_distilled.pt
    results/pruning_curve.png
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt

from config import COMPRESSION_STUDENT_MODEL, SEED
from data import get_metric_dataloader, get_retrieval_eval_dataloader
from models import build_model
from utils import format_subsection_header, cprint, Logger
from train_metric import compute_recall_at_1
from loaders import load_teacher

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints/part_c"
OUTPUT_DIR = "results"

# Embedding dimensions (must match train_metric.py)
EMBEDDING_DIM = 64
STUDENT_EMBEDDING_DIM = 64

# ---------------------------------------------------------------------------
# Distillation and Pruning hyperparameters (tune as needed for better results)
# ---------------------------------------------------------------------------

# Distillation
DISTILL_LR = 3e-4
DISTILL_EPOCHS = 20
DISTILL_TEMPERATURE = 4.0   # softens teacher embeddings
DISTILL_ALPHA = 0.5         # weight between embedding MSE and triplet loss

# Pruning
PRUNE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def sparsity_of(model: nn.Module) -> float:
    """
    Compute the actual sparsity level of the model (fraction of zero weights in Conv2d and Linear layers).

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which to compute the sparsity level. The function iterates through

    Returns
    -------
    float
        The actual sparsity level of the model
    """
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / max(total, 1)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which to count the parameters.

    Returns
    -------
    int
        The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable non-zero parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which to count the non-zero parameters.

    Returns
    -------
    int
        The total number of trainable non-zero parameters in the model.
    """
    return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Part C-1: Pruning
# ---------------------------------------------------------------------------
def apply_global_magnitude_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Apply global unstructured magnitude pruning to all Conv2d and Linear layers in the model.

    Parameters
    ----------
    model : nn.Module
        The model to be pruned. This model will be modified in-place.
    amount : float
        The target sparsity level (fraction of parameters to prune, e.g., 0.5 for 50% pruning).

    Returns
    -------
    nn.Module
        The pruned model with the specified sparsity level. Note that the pruning is made permanent by removing the re-parameterisation hooks, so the returned model is ready for evaluation or saving.
    """

    # Identify all parameters to prune (weights of Conv2d and Linear layers)
    parameters_to_prune = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]

    # Apply global unstructured pruning based on L1 magnitude
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make pruning permanent (remove re-parameterisation hooks)
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")
    return model


def run_pruning(backbone_name: str, eval_loader) -> list[dict]:
    """
    Run iterative global unstructured magnitude pruning on the Part B teacher model at increasing sparsity levels, evaluate Recall@1 at each level, and save results.

    Parameters
    ----------
    backbone_name : str
        The backbone architecture name (e.g., "cnn" or "separable_cnn") to identify which Part B teacher model to load for pruning evaluation.
    eval_loader : DataLoader
        The DataLoader for the evaluation set, used to compute Recall@1 for the teacher model before pruning and for each pruned model to evaluate the impact of pruning on retrieval performance.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the results for each pruning level, including target sparsity, actual sparsity, Recall@1, number of parameters, and drop in Recall@1 from the baseline.
    """
    cprint(format_subsection_header(f"Pruning backbone: {backbone_name}"), color="cyan")
    results = []

    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load the teacher model from Part B to use as the baseline for pruning evaluation
    teacher = load_teacher(backbone_name, EMBEDDING_DIM)

    # Baseline Recall@1 before pruning
    baseline_recall = compute_recall_at_1(teacher, eval_loader)
    print(f"\n  Sparsity   Actual   Params     Recall@1   Drop")
    print(f"  {'-'*52}")

    # Iterate over the defined pruning levels, apply pruning to the model, evaluate Recall@1, and save results
    for target_sparsity in PRUNE_LEVELS:

        # Create a deep copy of the teacher model to apply pruning without affecting the original teacher model used for evaluation
        model = copy.deepcopy(teacher)
        if target_sparsity > 0.0:
            apply_global_magnitude_pruning(model, amount=target_sparsity)

        # Calculate the actual sparsity level of the pruned model
        actual_sparsity = sparsity_of(model)

        # Compute Recall@1 for the pruned model and calculate the drop from the baseline
        recall = compute_recall_at_1(model, eval_loader)
        drop = baseline_recall - recall
        params = count_nonzero_parameters(model)

        # Format the label for the current pruning level and print the results in a tabular format
        label = f"{int(target_sparsity * 100):3d}%"
        print(f"  {label}       {actual_sparsity:.2f}    {params:>9,}    {recall:.4f}     {drop:+.4f}")

        # Save the pruned model checkpoint with metadata about the pruning level and performance
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"pruned_{int(target_sparsity*100):03d}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": backbone_name,
                "embedding_dim": EMBEDDING_DIM,
                "target_sparsity": target_sparsity,
                "actual_sparsity": actual_sparsity,
                "recall_at_1": recall,
            },
            ckpt_path,
        )

        # Append the results for the current pruning level to the results list for later plotting of the pruning curve
        results.append(
            {
                "target_sparsity": target_sparsity,
                "actual_sparsity": actual_sparsity,
                "recall": recall,
                "params": params,
                "drop": drop,
            }
        )

    # After iterating through all pruning levels, plot the pruning curve showing Recall@1 vs actual sparsity and save the plot to the output directory
    _plot_pruning_curve(results, baseline_recall)
    return results


def _plot_pruning_curve(results: list[dict], baseline: float) -> None:
    """
    Plot the pruning curve showing Recall@1 vs actual sparsity levels for the pruned models, with a reference line for the baseline Recall@1 before pruning.

    Parameters
    ----------
    results : list[dict]
        A list of dictionaries containing the results for each pruning level, including actual sparsity and Recall
    baseline : float
        The baseline Recall@1 value before pruning, used to plot a horizontal reference line on the
        pruning curve to visually compare the performance of pruned models against the original unpruned model.
    """

    # Extract actual sparsity levels and corresponding Recall@1 values from the results to plot the pruning curve
    sparsities = [r["actual_sparsity"] * 100 for r in results]
    recalls = [r["recall"] for r in results]

    # Create a line plot of Recall@1 vs actual sparsity using Matplotlib, with a horizontal dashed line indicating the baseline Recall@1 before pruning for reference.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sparsities, recalls, marker="o", linewidth=2, color="#4363d8")
    ax.axhline(baseline, linestyle="--", color="grey", label=f"Baseline {baseline:.3f}")
    ax.set_xlabel("Actual sparsity (%)")
    ax.set_ylabel("Recall@1")
    ax.set_title("Pruning curve: Recall@1 vs sparsity")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()

    # Save the pruning curve plot to the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "pruning_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    cprint(f"\n  Pruning curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Part C-2: Knowledge Distillation
# ---------------------------------------------------------------------------
def build_student() -> nn.Module:
    """
    CompactSeparableCNN with an embedding head.

    Returns
    -------
    nn.Module
         The student model architecture, initialized with random weights and moved to DEVICE.
    """
    model = build_model(
        COMPRESSION_STUDENT_MODEL,
        embedding_dim=STUDENT_EMBEDDING_DIM,
        input_channels=1,
    )
    return model.to(DEVICE)


def distillation_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Embedding-alignment distillation loss.
    Aligns L2-normalised student embeddings to L2-normalised teacher embeddings scaled by temperature (encourages the student to match the angular structure of the teacher's space).

    Parameters
    ----------
    student_emb : torch.Tensor
        The output embeddings from the student model for a batch of images, of shape (batch_size, embedding_dim).
    teacher_emb : torch.Tensor
        The output embeddings from the teacher model for the same batch of images, of shape (batch_size, embedding_dim).
    temperature : float
        The temperature scaling factor to soften the teacher embeddings. Higher values encourage the student to focus more on matching the teacher's embedding structure rather than just the nearest neighbors.

    Returns
    -------
    torch.Tensor
        The computed distillation loss (MSE between the normalized student and teacher embeddings).
    """
    s = F.normalize(student_emb / temperature, dim=1)
    t = F.normalize(teacher_emb / temperature, dim=1)
    return F.mse_loss(s, t)


def run_distillation(backbone_name: str, eval_loader, train_loader) -> None:
    """
    Run knowledge distillation from the Part B teacher to the CompactSeparableCNN student.

    Parameters
    ----------
    backbone_name : str
        The backbone architecture name (e.g., "cnn" or "separable_cnn") to identify which Part B teacher model to load for distillation.
    eval_loader : DataLoader
        The DataLoader for the evaluation set, used to compute Recall@1 before and after distillation.
    train_loader : DataLoader
        The DataLoader for the training set, used to provide batches of images for distillation training of the student model.
    """
    cprint(format_subsection_header(f"Knowledge Distillation - Teacher : {backbone_name}, Student : {COMPRESSION_STUDENT_MODEL}"), color="cyan")

    # Load teacher model and set to eval mode
    teacher = load_teacher(backbone_name, EMBEDDING_DIM)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Build student model
    student = build_student()

    # Set up optimizer and learning rate scheduler for distillation training
    optimizer = optim.Adam(student.parameters(), lr=DISTILL_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DISTILL_EPOCHS)

    # Print model sizes and compression ratio before distillation
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = count_parameters(student)
    print(f"  Teacher params : {teacher_params:,}")
    print(f"  Student params : {student_params:,}")
    compression_ratio = teacher_params / student_params

    # Recall@1 before distillation
    recall_before = compute_recall_at_1(student, eval_loader)
    print(f"\n  Student Recall@1 (before distillation): {recall_before:.4f}")
    print(f"  {'─'*45}")

    best_recall = 0.0
    best_state = None
    for epoch in range(1, DISTILL_EPOCHS + 1):
        # Set student to training mode for distillation
        student.train()

        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch[0].to(DEVICE)  # Same for both triplet and contrastive loaders

            # Set gradients to zero before backpropagation for the current batch
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_emb = teacher(images)

            # Compute student embeddings for the current batch of images
            student_emb = student(images)

            # Compute the distillation loss between the student and teacher embeddings
            loss = distillation_loss(student_emb, teacher_emb, DISTILL_TEMPERATURE)

            # Backpropagate the loss and update the student model parameters
            loss.backward()

            # Perform an optimization step to update the student model parameters based on the computed gradients
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Update the learning rate scheduler at the end of the epoch
        scheduler.step()

        recall = compute_recall_at_1(student, eval_loader)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:02d}/{DISTILL_EPOCHS} | distill_loss {avg_loss:.5f} | Recall@1 {recall:.4f}")

        # Update best student model if Recall@1 has improved
        if recall > best_recall:
            best_recall = recall
            best_state = {k: v.clone() for k, v in student.state_dict().items()}

    # Final summary
    student.load_state_dict(best_state)
    recall_after = compute_recall_at_1(student, eval_loader)
    teacher_recall = compute_recall_at_1(teacher, eval_loader)

    print(f"\n  {'─'*45}")
    print(f"  Teacher Recall@1              : {teacher_recall:.4f}")
    print(f"  Student Recall@1 (before)     : {recall_before:.4f}")
    print(f"  Student Recall@1 (after)      : {recall_after:.4f}")
    retention = recall_after / max(teacher_recall, 1e-8) * 100
    print(f"  Retrieval retention           : {retention:.1f}%")
    print(f"  Compression ratio (params)    : {compression_ratio:.1f}×")
    print(f"  {'─'*45}")

    # Save the distilled student checkpoint with metadata
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "student_distilled.pt")
    torch.save(
        {
            "model_state": best_state,
            "model_name": COMPRESSION_STUDENT_MODEL,
            "embedding_dim": STUDENT_EMBEDDING_DIM,
            "recall_at_1": recall_after,
            "teacher_recall_at_1": teacher_recall,
        },
        ckpt_path,
    )
    cprint(f"\n  Saved distilled student → {ckpt_path}")


def main() -> None:
    """
    Main function to parse command-line arguments and run the pruning and distillation subsections.
    """

    original_terminal = sys.stdout

    # Set up argument parser to allow users to specify the backbone architecture and whether to skip pruning or distillation
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="separable_cnn",
                        choices=["cnn", "separable_cnn"])
    parser.add_argument("--skip_pruning", action="store_true",
                        help="Skip the pruning subsection")
    parser.add_argument("--skip_distill", action="store_true",
                        help="Skip the distillation subsection")
    parser.add_argument("--loss", default="triplet", choices=["triplet", "contrastive"],
                        help="Metric learning loss to use for distillation (default: triplet)")
    args = parser.parse_args()

    # Set random seed for reproducibility and load the evaluation and training data loaders
    torch.manual_seed(SEED)
    train_loader = get_metric_dataloader(loss_name=args.loss)
    eval_loader = get_retrieval_eval_dataloader()

    # Run pruning
    if not args.skip_pruning:
        logger_instance = Logger("results/pruning_log.txt")
        sys.stdout = logger_instance
        run_pruning(args.backbone, eval_loader)
        sys.stdout = original_terminal
        logger_instance.log.close()

    # Run distillation
    if not args.skip_distill:
        logger_instance = Logger("results/distillation_log.txt")
        sys.stdout = logger_instance
        run_distillation(args.backbone, eval_loader, train_loader)
        sys.stdout = original_terminal
        logger_instance.log.close()


if __name__ == "__main__":
    main()
