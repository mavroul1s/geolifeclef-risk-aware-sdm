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


class ResidualTemporalEncoder(nn.Module):
    """Dilated temporal encoder for compact environmental time-series cubes."""

    def __init__(self, input_features: int, width: int = 96, output_dim: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_features, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        width, width, 3, padding=dilation, dilation=dilation, groups=width
                    ),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Conv1d(width, width, 1),
                    nn.Dropout(0.1),
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.projection = nn.Sequential(nn.Linear(width * 2, output_dim), nn.LayerNorm(output_dim))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        values = self.stem(sequence.transpose(1, 2))
        for block in self.blocks:
            values = values + block(values)
        pooled = torch.cat((values.mean(-1), values.amax(-1)), dim=-1)
        return self.projection(pooled)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, 1),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(self.norm(self.depthwise(image)))
        return image + self.scale * update


class SentinelEncoder(nn.Module):
    """Small ConvNeXt-style encoder sized for 4x32x32 Sentinel patches."""

    def __init__(self, output_dim: int = 192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            nn.Conv2d(64, 128, 2, stride=2),
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
            nn.Conv2d(128, output_dim, 2, stride=2),
            ConvNeXtBlock(output_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


class CompetitiveFusionSDM(nn.Module):
    """Compute-aware multimodal SDM with attention across environmental modalities."""

    def __init__(
        self,
        num_labels: int,
        static_features: int,
        model_dim: int = 192,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.landsat = ResidualTemporalEncoder(6, output_dim=model_dim)
        self.climate = ResidualTemporalEncoder(4, output_dim=model_dim)
        self.sentinel = SentinelEncoder(output_dim=model_dim)
        self.static = nn.Sequential(
            nn.Linear(static_features, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.modality_embeddings = nn.Parameter(torch.randn(1, 4, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=6,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.gate = nn.Sequential(nn.Linear(model_dim, 1), nn.Sigmoid())
        self.landsat_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_labels),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_labels),
        )
        # Start close to a stable Landsat predictor; learn multimodal corrections gradually.
        self.fusion_logit = nn.Parameter(torch.tensor(-2.0))

    def forward_features(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        landsat_token = self.landsat(batch["landsat"])
        tokens = torch.stack(
            (
                landsat_token,
                self.climate(batch["climate"]),
                self.sentinel(batch["sentinel"]),
                self.static(batch["static"]),
            ),
            dim=1,
        )
        tokens = self.fusion(tokens + self.modality_embeddings)
        weights = self.gate(tokens)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1e-6)
        logits = self.landsat_head(landsat_token) + torch.sigmoid(self.fusion_logit) * self.head(
            pooled
        )
        return logits, pooled

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_features(batch)[0]


class RareAwareCompetitiveFusionSDM(nn.Module):
    """Single multimodal model with rare-taxon and survey-cardinality specialists."""

    def __init__(
        self,
        num_labels: int,
        static_features: int,
        rare_indices: torch.Tensor,
        model_dim: int = 192,
        dropout: float = 0.15,
    ):
        super().__init__()
        rare_indices = torch.as_tensor(rare_indices, dtype=torch.long)
        if rare_indices.ndim != 1 or len(rare_indices) == 0:
            raise ValueError("rare_indices must be a non-empty one-dimensional tensor")
        if int(rare_indices.min()) < 0 or int(rare_indices.max()) >= num_labels:
            raise ValueError("rare_indices contain a label outside num_labels")
        self.backbone = CompetitiveFusionSDM(
            num_labels,
            static_features=static_features,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.register_buffer("rare_indices", rare_indices)
        self.rare_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, len(rare_indices)),
        )
        self.rare_residual_logit = nn.Parameter(torch.tensor(-1.0))
        self.cardinality_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

    def forward_with_aux(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, pooled = self.backbone.forward_features(batch)
        rare_logits = self.rare_head(pooled)
        rare_delta = torch.zeros_like(logits).index_copy(1, self.rare_indices, rare_logits)
        logits = logits + torch.sigmoid(self.rare_residual_logit) * rare_delta
        log_cardinality = self.cardinality_head(pooled).squeeze(-1)
        return logits, rare_logits, log_cardinality

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_with_aux(batch)[0]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
