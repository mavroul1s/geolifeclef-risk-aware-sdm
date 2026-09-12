import numpy as np

from scripts.train_sota_single import (
    adaptive_cardinalities,
    rank_blend_predictions,
    sample_f1_from_lists,
)


def test_adaptive_cardinality_is_scaled_and_clipped():
    log_cardinality = np.log1p(np.array([1.0, 10.0, 100.0]))
    assert adaptive_cardinalities(log_cardinality, 1.2).tolist() == [4, 12, 50]


def test_rank_blend_can_promote_a_spatial_species_outside_neural_labels():
    probabilities = np.array([[0.9, 0.8, 0.1]], dtype=np.float32)
    predictions = rank_blend_predictions(
        probabilities,
        np.array([10, 20, 30]),
        np.array([2]),
        po_candidates=[{99: 1.0}],
        ood_mask=np.array([True]),
        po_weight_ood=1.0,
    )
    assert 99 in predictions[0]
    assert len(predictions[0]) == 2


def test_sample_f1_from_lists_uses_per_survey_average():
    truth = [{1}, {2, 3}]
    predictions = [[1], [2, 3]]
    assert sample_f1_from_lists(truth, predictions) == 1.0
