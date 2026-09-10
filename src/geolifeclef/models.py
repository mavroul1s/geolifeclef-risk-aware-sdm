from __future__ import annotations

import torch
from torch import nn


class TemporalCNN(nn.Module):
    """Small seasonal temporal CNN: [batch, time, variables] -> logits."""
    def __init__(self, input_features: int, num_labels: int, channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(input_features, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.GELU(), nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.BatchNorm1d(channels), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout))
        self.classifier = nn.Linear(channels, num_labels)

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.features(sequence.transpose(1, 2))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(sequence))


class StaticMLP(nn.Module):
    def __init__(self, input_features: int, embedding_dim: int = 64, dropout: float = 0.15):
        super().__init__(); self.network = nn.Sequential(nn.Linear(input_features, embedding_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim), nn.GELU())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GatedFusion(nn.Module):
    """Lightweight, ablation-friendly fusion of encoded modality vectors."""
    def __init__(self, modality_dims: dict[str, int], num_labels: int, fusion_dim: int = 64):
        super().__init__(); self.names = tuple(modality_dims); self.projectors = nn.ModuleDict({name: nn.Linear(size, fusion_dim) for name, size in modality_dims.items()}); self.gate = nn.Sequential(nn.Linear(fusion_dim * len(self.names), len(self.names)), nn.Sigmoid()); self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = [self.projectors[name](embeddings[name]) for name in self.names]
        weights = self.gate(torch.cat(projected, dim=-1)).unsqueeze(-1)
        fused = (torch.stack(projected, 1) * weights).sum(1) / weights.sum(1).clamp_min(1e-6)
        return self.classifier(fused)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

