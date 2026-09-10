import torch

from geolifeclef.models import GatedFusion, TemporalCNN


def test_model_forward_shapes():
    assert TemporalCNN(5, 7, channels=8)(torch.randn(3, 12, 5)).shape == (3, 7)
    fusion = GatedFusion({"landsat": 8, "static": 4}, 7, fusion_dim=6)
    assert fusion({"landsat": torch.randn(3, 8), "static": torch.randn(3, 4)}).shape == (3, 7)

