import numpy as np

from scripts.train_spatial_competition import (
    apply_prediction_policy,
    fit_prediction_policy,
    sample_f1_from_predictions,
    threshold_min_k_predictions,
)


def test_sample_f1_and_policy_use_per_survey_scoring():
    targets = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    probabilities = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.7]], dtype=np.float32)
    policy = fit_prediction_policy(targets, probabilities)
    predictions = apply_prediction_policy(probabilities, policy)
    assert sample_f1_from_predictions(targets, predictions) == 1.0


def test_threshold_policy_enforces_minimum_and_maximum_cardinality():
    probabilities = np.linspace(0, 1, 100, dtype=np.float32)[None, :]
    low = threshold_min_k_predictions(probabilities, threshold=0.99, minimum_k=8)
    high = threshold_min_k_predictions(probabilities, threshold=0.01, minimum_k=8, maximum_k=20)
    assert low.sum() == 8
    assert high.sum() == 20
