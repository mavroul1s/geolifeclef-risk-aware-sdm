from __future__ import annotations

import torch
from torch.nn import functional as F


def asymmetric_loss(logits: torch.Tensor, targets: torch.Tensor, gamma_negative: float = 4.0, gamma_positive: float = 0.0, clip: float = 0.05) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = targets * torch.log(probabilities.clamp_min(1e-8))
    negative_probabilities = (probabilities - clip).clamp_min(0.0)
    negative = (1 - targets) * torch.log((1 - negative_probabilities).clamp_min(1e-8))
    weights = targets * (1 - probabilities).pow(gamma_positive) + (1 - targets) * negative_probabilities.pow(gamma_negative)
    return -(weights * (positive + negative)).mean()


def class_balanced_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
