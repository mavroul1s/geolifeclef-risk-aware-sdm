from __future__ import annotations

import torch
from torch.nn import functional as F


def asymmetric_loss(logits: torch.Tensor, targets: torch.Tensor, gamma_negative: float = 4.0, gamma_positive: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = targets * torch.log(probabilities.clamp_min(1e-8))
    negative = (1 - targets) * torch.log((1 - probabilities).clamp_min(1e-8))
    weights = targets * (1 - probabilities).pow(gamma_positive) + (1 - targets) * probabilities.pow(gamma_negative)
    return -(weights * (positive + negative)).mean()


def class_balanced_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

