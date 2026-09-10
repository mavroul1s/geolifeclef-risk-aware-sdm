import numpy as np

from geolifeclef.metrics import fit_adaptive_thresholds, multilabel_f1


def test_metrics_and_adaptive_thresholds():
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    probabilities = np.array([[.9, .1], [.1, .9], [.8, .2], [.2, .8]])
    assert multilabel_f1(targets, probabilities) == 1.0
    assert fit_adaptive_thresholds(targets, probabilities, minimum_positives=1).shape == (2,)

