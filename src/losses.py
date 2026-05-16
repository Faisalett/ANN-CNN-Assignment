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
