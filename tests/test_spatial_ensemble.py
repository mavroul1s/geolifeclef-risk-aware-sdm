import numpy as np

from scripts.train_spatial_competition import sample_f1_from_predictions


def test_probability_ensemble_can_correct_complementary_errors():
    targets = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    first = np.array([[0.9, 0.6], [0.3, 0.6]], dtype=np.float32)
    second = np.array([[0.7, 0.2], [0.5, 0.8]], dtype=np.float32)
    ensemble = (first + second) / 2
    predictions = ensemble >= 0.55
    assert sample_f1_from_predictions(targets, predictions) == 1.0
