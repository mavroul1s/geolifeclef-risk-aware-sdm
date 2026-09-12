import torch

from geolifeclef.models import (
    CompetitiveFusionSDM,
    GatedFusion,
    RareAwareCompetitiveFusionSDM,
    TemporalCNN,
)


def test_model_forward_shapes():
    assert TemporalCNN(5, 7, channels=8)(torch.randn(3, 12, 5)).shape == (3, 7)
    fusion = GatedFusion({"landsat": 8, "static": 4}, 7, fusion_dim=6)
    assert fusion({"landsat": torch.randn(3, 8), "static": torch.randn(3, 4)}).shape == (3, 7)


def test_competitive_fusion_shape():
    model = CompetitiveFusionSDM(num_labels=7, static_features=13, model_dim=48)
    batch = {
        "landsat": torch.randn(2, 84, 6),
        "climate": torch.randn(2, 228, 4),
        "sentinel": torch.randn(2, 4, 32, 32),
        "static": torch.randn(2, 13),
    }
    assert model(batch).shape == (2, 7)


def test_rare_aware_model_returns_class_and_cardinality_outputs():
    model = RareAwareCompetitiveFusionSDM(
        num_labels=7,
        static_features=13,
        rare_indices=torch.tensor([1, 5]),
        model_dim=48,
    )
    batch = {
        "landsat": torch.randn(2, 84, 6),
        "climate": torch.randn(2, 228, 4),
        "sentinel": torch.randn(2, 4, 32, 32),
        "static": torch.randn(2, 13),
    }
    logits, rare_logits, log_cardinality = model.forward_with_aux(batch)
    assert logits.shape == (2, 7)
    assert rare_logits.shape == (2, 2)
    assert log_cardinality.shape == (2,)
