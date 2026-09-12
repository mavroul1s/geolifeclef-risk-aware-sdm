"""Randomly initialized competition PO expert with weak-background supervision."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width * 2, width),
            nn.Dropout(0.1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class POExpert(nn.Module):
    """Compact environmental/location MLP; no external initialization or weights.

    Inputs are already aligned and normalized environmental/location vectors.
    Every PA species keeps an output column even when absent from PO training.
    The same architecture is used for the zero-PO control.
    """

    def __init__(self, num_labels: int, input_dim: int, width: int = 256):
        super().__init__()
        if any(not isinstance(value, int) or value < 1 for value in (num_labels, input_dim, width)):
            raise ValueError("num_labels, input_dim and width must be positive integers")
        self.num_labels, self.input_dim, self.width = num_labels, input_dim, width
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            _ResidualBlock(width),
            _ResidualBlock(width),
            nn.LayerNorm(width),
        )
        self.classifier = nn.Linear(width, num_labels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"Expected a [batch, {self.input_dim}] feature matrix")
        return self.classifier(self.encoder(values))


def masked_po_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    unobserved_weight: float = 0.05,
) -> torch.Tensor:
    """Average positive and weak-background log loss separately per pseudo-survey.

    Targets are binary deduplicated presence indicators, not occurrence counts.
    Each pseudo-survey contributes mean positive softplus(-logit), plus
    ``unobserved_weight`` times mean unobserved softplus(logit). Unobserved
    species are explicitly weak background, not confirmed absences. This is a
    down-weighted presence/background objective, not an unbiased PU-risk claim.

    Empty-positive rows are rejected: they provide no observed PO evidence and
    must be removed during preparation. An all-positive row contributes only
    its positive term. Compute the loss in float32 for stable mixed precision.
    Sampling/publisher weights belong to pseudo-survey construction, not binary
    target values; duplicated raw occurrences must be deduplicated upstream.
    """
    if logits.ndim != 2 or targets.shape != logits.shape or not logits.numel():
        raise ValueError("Expected matching nonempty [batch, species] logits and targets")
    if not torch.is_floating_point(logits):
        raise ValueError("PO logits must be floating point")
    if logits.device != targets.device:
        raise ValueError("PO logits and targets must be on the same device")
    if not math.isfinite(unobserved_weight) or not 0 <= unobserved_weight <= 1:
        raise ValueError("unobserved_weight must be finite and between zero and one")
    if not torch.isfinite(logits).all():
        raise ValueError("PO logits must be finite")
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("PO targets must be finite binary presence indicators")
    positive_mask = targets.float()
    positive_count = positive_mask.sum(dim=1)
    if (positive_count == 0).any():
        raise ValueError("Every PO pseudo-survey must contain at least one positive species")
    values = logits.float()
    unobserved_mask = 1 - positive_mask
    positive_loss = (F.softplus(-values) * positive_mask).sum(dim=1) / positive_count
    background_loss = (F.softplus(values) * unobserved_mask).sum(dim=1)
    background_loss = background_loss / unobserved_mask.sum(dim=1).clamp_min(1)
    return (positive_loss + unobserved_weight * background_loss).mean()
