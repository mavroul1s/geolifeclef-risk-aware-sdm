import torch

from geolifeclef.models import CompetitiveFusionSDM, GatedFusion, TemporalCNN


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
