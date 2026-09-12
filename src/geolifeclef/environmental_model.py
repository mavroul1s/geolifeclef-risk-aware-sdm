"""Competition-only environmental residual network with multimodal specialists."""
from __future__ import annotations

import torch
from torch import nn

from geolifeclef.models import SentinelEncoder


class VectorResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class EnvironmentalChallenger(nn.Module):
    """Keep variable/month identity; condition species scores on actual environment.

    Random initialization only. This is an experimental challenger, not a
    reproduction of the winning system or a verified state-of-the-art model.
    """
    def __init__(self, num_labels: int, environment_features: int, static_features: int = 21, width: int = 512):
        super().__init__()
        self.environment = nn.Sequential(nn.Linear(environment_features + static_features, width), nn.GELU(), VectorResidualBlock(width), VectorResidualBlock(width), nn.LayerNorm(width))
        self.temporal = nn.Sequential(nn.Linear(84 * 6 + 228 * 4, width), nn.GELU(), VectorResidualBlock(width), VectorResidualBlock(width), nn.LayerNorm(width))
        self.sentinel = SentinelEncoder(192)
        self.fusion = nn.Sequential(nn.Linear(width * 2 + 192, width), nn.GELU(), VectorResidualBlock(width), nn.LayerNorm(width))
        self.head = nn.Linear(width, num_labels)
        self.environment_head = nn.Linear(width, num_labels)
        self.temporal_head = nn.Linear(width, num_labels)
        self.specialist_weight = nn.Parameter(torch.tensor([-1.0, -1.0]))

    def forward_with_aux(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        environment = self.environment(torch.cat([batch["environment"], batch["static"]], dim=1))
        temporal = self.temporal(torch.cat([batch["landsat"].flatten(1), batch["climate"].flatten(1)], dim=1))
        image = self.sentinel(batch["sentinel"])
        environment_logits, temporal_logits = self.environment_head(environment), self.temporal_head(temporal)
        logits = self.head(self.fusion(torch.cat([environment, temporal, image], dim=1)))
        weights = self.specialist_weight.sigmoid()
        logits = logits + weights[0] * environment_logits + weights[1] * temporal_logits
        return logits, (environment_logits, temporal_logits)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_with_aux(batch)[0]
