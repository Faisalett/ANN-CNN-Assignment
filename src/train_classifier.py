"""
Part A — Classification

Trains StandardCNN and SeparableCNN on FashionMNIST and reports:
  - accuracy
  - macro F1
  - parameter count
  - FLOPs for a single forward pass

Usage:
    python train_classifier.py

The script saves one checkpoint per model under checkpoints/part_a/:
    checkpoints/part_a/cnn.pt
    checkpoints/part_a/separable_cnn.pt

It also prints a summary table so you can decide which backbone to carry into Part B.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import PART_A_MODELS, SEED
from data import get_classification_loaders
from models import build_model
from utils import cprint, format_section_header, format_underline, format_bold, Logger


# ---------------------------------------------------------------------------
# Hyper-parameters (safe to change without breaking comparability)
# ---------------------------------------------------------------------------
LR = 1e-3
EPOCHS = 15
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints/part_a"


def count_parameters(model: nn.Module) -> int:
    """
    Return the total number of trainable parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which to count trainable parameters.

    Returns
    -------
    int
        The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_channels: int = 1, img_size: int = 28) -> int:
    """
    Return the total number of FLOPs for a single forward pass through the model.
    This function first attempts to use torch.profiler for an accurate FLOP count.
    If torch.profiler fails (e.g., due to unsupported operations), it falls back
    to a manual calculation that handles common layers like Conv2d and Linear.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which to count FLOPs.
    input_channels : int, optional
        The number of input channels (default is 1 for grayscale images).
    img_size : int, optional
        The height and width of the input image (default is 28 for FashionMNIST).
    """
    model.eval()

    # Create a dummy input matching the FashionMNIST dimensions
    dummy = torch.randn(1, input_channels, img_size, img_size).to(DEVICE)

    try:
        # Attempt to use the PyTorch Profiler for a high-fidelity FLOP count
        from torch.profiler import profile, ProfilerActivity
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [
                    ProfilerActivity.CPU],
                record_shapes=True,
                with_flops=True  # Required to enable FLOP estimation in the profiler. Doesn't work without this flag.
        ) as prof:
            with torch.no_grad():
                model(dummy)

        # sum up FLOPs from all operations recorded by the profiler
        total = sum(e.flops for e in prof.key_averages() if hasattr(e, "flops") and e.flops is not None)
        if total > 0:
            return int(total)

    except Exception:
        # Fallback if the profiler is unavailable or fails for specific layers
        print("  Warning: torch.profiler failed to compute FLOPs, falling back to manual calculation.")
        pass

    # Manual Fallback
    total_flops = 0
    h, w = img_size, img_size

    # This loop estimates computational cost for standard architecture components
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Calculate output feature map dimensions based on padding, kernel size, and stride
            kh, kw = module.kernel_size
            sh, sw = module.stride
            ph, pw = module.padding

            h = (h + 2 * ph - kh) // sh + 1
            w = (w + 2 * pw - kw) // sw + 1

            # Compute FLOPs using the formula: 2 * (K^2 * Cin / groups) * Cout * H_out * W_out
            # The '2' accounts for the Multiply-Accumulate (MAC) operation
            layer_flops = 2 * (kh * kw * module.in_channels // module.groups) * module.out_channels * h * w
            total_flops += layer_flops

        elif isinstance(module, nn.Linear):
            # Compute FLOPs for fully connected layers: 2 * Input_Features * Output_Features
            total_flops += 2 * module.in_features * module.out_features

    return int(total_flops)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """
    Train the model for one epoch and return the average training loss.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    loader : DataLoader
        The DataLoader providing the training data.
    criterion : nn.Module
        The loss function to use for training.
    optimizer : optim.Optimizer
        The optimizer to use for updating the model parameters.

    Returns
    -------
    float
        The average training loss over the epoch.
    """

    # Set model to training mode
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Clear existing gradients from the previous step to avoid accumulation
        optimizer.zero_grad()

        # Forward pass: compute model output (logits) for the current batch
        logits = model(images)

        # Calculate loss (the distance between predictions and actual labels)
        loss = criterion(logits, labels)

        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()

        # Update model parameters based on computed gradients
        optimizer.step()

        # Accumulate loss weighted by the current batch size
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """
    Evaluate the model on the given DataLoader and return accuracy and macro F1 score.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    loader : DataLoader
        The DataLoader providing the evaluation data.

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - accuracy: The overall accuracy of the model on the evaluation set.
        - macro F1: The macro-averaged F1 score across all classes.
    """

    # Set the model to evaluation mode
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)

        # Get predictions by finding the index of the maximum logit value
        preds = model(images).argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(labels)

    # Flatten collected batches into single arrays for metric calculation
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Calculate accuracy and macro F1 score
    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return float(accuracy), float(macro_f1)


def train_model(name: str, train_loader: DataLoader, val_loader: DataLoader) -> dict:
    """
    Train a single model and return its results.

    Parameters
    ----------
    name : str
        The name of the model architecture to train ("cnn", "separable_cnn" or "compact_separable_cnn").
    train_loader : DataLoader
        The DataLoader providing the training data.
    val_loader : DataLoader
        The DataLoader providing the validation data.

    Returns
    -------
    dict
        A dictionary containing:
        - "name": The name of the model architecture.
        - "accuracy": The final accuracy of the model on the validation set after training.
        - "macro_f1": The final macro F1 score of the model on the validation set after training.
        - "parameters": The total number of trainable parameters in the model.
        - "flops": The total number of FLOPs for a single forward pass through the model.
        - "checkpoint": The file path to the saved checkpoint of the trained model.
    """

    # Ensure reproducibility by setting the random seed before model initialization
    torch.manual_seed(SEED)

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    model = build_model(name, num_classes=10, input_channels=1).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Print training header with model name, parameter count, and FLOP count
    train_section = f"Training: {name}\n  Parameters : {count_parameters(model):,}\n  FLOPs      : {count_flops(model):,}"
    cprint(format_section_header(train_section, align='left')[:-1], color="yellow")

    best_acc = 0.0
    best_state = None

    # Main training loop over epochs
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Execute training and evaluation for the current epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, val_f1 = evaluate(model, val_loader)

        # Update learning rate based on the current epoch
        scheduler.step()

        # Print epoch results
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/{EPOCHS} | loss {train_loss:.4f} | "f"acc {val_acc:.4f} | F1 {val_f1:.4f} | {elapsed:.1f}s")

        # Track the best model state based on validation accuracy to avoid using overfitted weights
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Reload best weights
    model.load_state_dict(best_state)

    # Save the checkpoint for best model state to disk for later use in Part B and reproducibility
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
    torch.save({"model_state": best_state, "model_name": name}, ckpt_path)
    cprint(f"  Saved checkpoint → {ckpt_path}")

    # Final evaluation with best weights
    final_acc, final_f1 = evaluate(model, val_loader)
    flops = count_flops(model)
    params = count_parameters(model)

    return {
        "name": name,
        "accuracy": final_acc,
        "macro_f1": final_f1,
        "parameters": params,
        "flops": flops,
        "checkpoint": ckpt_path,
    }


def print_results_table(results: list[dict]) -> None:
    """
    Print a formatted table of results for all models and a summary of wins by each metric.

    Parameters
    ----------
    results : list[dict]
        A list of dictionaries, each containing the results for a model
        with keys: "name", "accuracy", "macro_f1", "parameters", "flops", and "checkpoint".
    """

    # Print a formatted table of results for all models
    print("\n" + "=" * 75)
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Params':>12} {'FLOPs':>12}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['name']:<25} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} "
            f"{r['parameters']:>12,} {r['flops']:>12,}"
        )
    print("=" * 75)

    # Print a summary of wins by each metric to help with decision-making for Part B
    print('\n' + format_bold(format_underline("Summary of wins:")))
    print(f'Wins by Accuracy: {max(results, key=lambda r: r["accuracy"])["name"]}')
    print(f'Wins by Macro F1: {max(results, key=lambda r: r["macro_f1"])["name"]}')
    print(f'Wins by Params: {min(results, key=lambda r: r["parameters"])["name"]}')
    print(f'Wins by FLOPs: {min(results, key=lambda r: r["flops"])["name"]}')

    # Print recommended backbone for Part B based on the highest accuracy
    best = max(results, key=lambda r: r["accuracy"])
    cprint(f"\nRecommended backbone for Part B: {best['name']}")
    cprint(f"  → checkpoint: {best['checkpoint']}")


def main() -> None:
    """
    Main function to train both classifiers and print results.
    """
    original_terminal = sys.stdout
    logger_instance = Logger("results/train_classifier_log.txt")
    sys.stdout = logger_instance

    # Get data loaders for training and validation sets
    train_loader, val_loader = get_classification_loaders()
    results = []

    # Train each model specified in PART_A_MODELS and collect results for comparison in a summary table.
    for name in PART_A_MODELS:
        result = train_model(name, train_loader, val_loader)
        results.append(result)
    print_results_table(results)

    sys.stdout = original_terminal
    logger_instance.log.close()


if __name__ == "__main__":
    main()