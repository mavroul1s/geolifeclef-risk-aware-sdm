import numpy as np
import pandas as pd
import pytest

from geolifeclef.decision_reliability import (
    balanced_country_weights, choose_threshold, decision_counts, decision_frame,
    decision_ranks, grouped_risk_prediction)
from geolifeclef.reliability import r1_accept


def test_tied_scores_have_declared_species_order():
    assert decision_ranks([[.2, .8, .8, .1]], 3).tolist() == [[1, 2, 0]]
    with pytest.raises(ValueError):
        decision_ranks([[np.nan]])


def test_decoder_counts_and_threshold_fit_on_known_calibration():
    v = np.array([[.9, .8, .1], [.4, .3, .2]])
    assert decision_counts(v, v.sum(1), 'calibrated_threshold', .5, minimum=1).tolist() == [2, 1]
    hits = np.array([[1, 1, 0], [1, 0, 0]])
    t, records = choose_threshold(v, hits, [2, 1], [.1, .5], minimum=1)
    assert t == .5 and records[1]['sample_F1'] == 1


def test_decision_metrics_keep_rare_occurrences_and_set_confidence_separate():
    p = np.array([[.9, .8, .1], [.4, .3, .2]])
    y = np.array([[1, 1, 0], [1, 0, 1]])
    f = decision_frame(p, y, pd.DataFrame({'surveyId': [1, 2]}), [50, 10, 2], 'top18')
    assert f.predicted_richness.tolist() == [3, 3]
    assert f.sample_F1.tolist() == pytest.approx([.8, .8])
    assert f.rare_truth.tolist() == [1, 1]
    assert f.rare_hits.tolist() == [1, 1]
    assert f.mean_set_confidence.tolist() == pytest.approx([.6, .3])


def test_inverse_sqrt_weights_reduce_large_country_dominance():
    w = balanced_country_weights(['a']*9+['b'])
    assert w[-1]/w[0] == pytest.approx(3)
    assert w.mean() == pytest.approx(1)


@pytest.mark.parametrize('variant', ['global_lean', 'country_balanced_lean', 'country_blend', 'bioregion_blend'])
def test_regional_prediction_does_not_read_evaluation_outcomes(variant):
    c = pd.DataFrame({'x': np.arange(80), 'loss': np.linspace(0, 1, 80),
                      'country': ['a']*60+['b']*20, 'region': ['r']*80})
    e = c.iloc[[3, 35, 70]].copy()
    parameters = {'max_iter': 5, 'max_leaf_nodes': 3, 'min_samples_leaf': 5, 'random_state': 1}
    a, record = grouped_risk_prediction(c, e.drop(columns='loss'), ['x'], parameters, variant, minimum_local=10)
    e['loss'] = [1e9, -1e9, np.nan]
    b, _ = grouped_risk_prediction(c, e, ['x'], parameters, variant, minimum_local=10)
    np.testing.assert_array_equal(a, b)
    assert record['calibration_surveys'] == 80


def test_group_constant_risk_offsets_cannot_change_within_group_quota_ranking():
    risk = np.linspace(0, 1, 20)
    country = np.array(['a']*10+['b']*10)
    group = np.array([f'{c}|area={i%2}' for i, c in enumerate(country)])
    a = r1_accept(risk, np.arange(20), .8, group)
    b = r1_accept(risk+np.where(country == 'a', 7, -5), np.arange(20), .8, group)
    np.testing.assert_array_equal(a, b)
