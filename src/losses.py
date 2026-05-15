"""
Metric learning losses for Part B.

Provides:
- ContrastiveLoss  (Hadsell et al., CVPR 2006)
- TripletLoss      (Schroff et al., CVPR 2015)
"""

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for siamese-style training.

    Given a pair of embeddings and a binary label (1 = same class, 0 = different),
    pulls same-class pairs together and pushes different-class pairs apart up to
    a margin.

    Loss = (1 - y) * 0.5 * d^2
         + y       * 0.5 * max(0, margin - d)^2

    where d = ||z1 - z2||_2.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1:    (B, D) embeddings for the first item in each pair.
            z2:    (B, D) embeddings for the second item in each pair.
            label: (B,)   1 if the pair is from different classes, 0 if same.
                          (Convention follows the original paper.)
        """
        dist = F.pairwise_distance(z1, z2)
        same_loss = (1 - label).float() * 0.5 * dist.pow(2)
        diff_loss = label.float() * 0.5 * F.relu(self.margin - dist).pow(2)
        return (same_loss + diff_loss).mean()


class TripletLoss(nn.Module):
    """
    Triplet loss with semi-hard negative mining.

    For each anchor, a positive (same class) and negative (different class)
    are selected. The loss encourages:

        d(anchor, positive) + margin < d(anchor, negative)

    Loss = mean(max(0, d_pos - d_neg + margin))
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchors:   (B, D)
            positives: (B, D) — same class as anchor
            negatives: (B, D) — different class from anchor
        """
        d_pos = F.pairwise_distance(anchors, positives)
        d_neg = F.pairwise_distance(anchors, negatives)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


def build_pairs(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build contrastive pairs from a batch.

    Returns (z1, z2, pair_labels) where pair_labels=1 means different class.
    Uses all O(N^2) pairs — practical for typical batch sizes up to ~256.
    """
    n = embeddings.size(0)
    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=embeddings.device)
    z1 = embeddings[idx_i]
    z2 = embeddings[idx_j]
    pair_labels = (labels[idx_i] != labels[idx_j]).long()
    return z1, z2, pair_labels


def build_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] or None:
    """
    Batch-hard mining (Hermans et al., 2017):
    For each anchor, pick the hardest positive (furthest same-class)
    and hardest negative (closest different-class) in the batch.
    """
    if labels.unique().numel() < 2:
        return None

    # L2-normalise before computing distances
    emb = F.normalize(embeddings, dim=1)
    dist = torch.cdist(emb, emb)  # (B, B)

    same = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
    diff = ~same

    # Hardest positive: furthest among same-class (excluding self)
    same.fill_diagonal_(False)
    dist_pos = dist.clone()
    dist_pos[~same] = -float("inf")
    hardest_pos = dist_pos.argmax(dim=1)

    # Hardest negative: closest among different-class
    dist_neg = dist.clone()
    dist_neg[~diff] = float("inf")
    hardest_neg = dist_neg.argmin(dim=1)

    # Drop anchors that have no valid positive or negative
    valid = same.any(dim=1) & diff.any(dim=1)
    if not valid.any():
        return None

    idx = valid.nonzero(as_tuple=True)[0]
    return (
        embeddings[idx],
        embeddings[hardest_pos[idx]],
        embeddings[hardest_neg[idx]],
    )