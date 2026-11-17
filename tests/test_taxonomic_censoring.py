import numpy as np
import pytest

from geolifeclef.taxonomic_censoring import genus_contract, censor_genera, paired_spatial_interval
from geolifeclef.neon_assessment import score_fixed_sets
from geolifeclef.partial_set_bounds import fixed_set_f1_bounds


def example():
    y = np.array([[1, 0, 1, 0, -1], [0, 0, 0, 0, -1], [1, 1, 0, 0, -1]])
    return y, y >= 0, np.array([0, 0, 1, 1, 2]), np.ones(3), np.array([10, 20, 30])


def test_zero_dose_is_identity_and_zero_positive_cell_stays_known():
    y, e, g, w, ids = example()
    known, positive, hidden = censor_genera(y, e, g, w, ids, 0, 12)
    assert np.array_equal(known, e)
    assert np.array_equal(positive, [2, 0, 1]) and not hidden.any()
    known, _, hidden = censor_genera(y, e, g, w, ids, .2, 12)
    assert np.array_equal(hidden, [1, 0, 1])
    assert np.array_equal(known[1], e[1])


def test_entire_genera_are_hidden_including_negative_labels_without_restoring_positives():
    y, e, g, w, ids = example()
    known, _, _ = censor_genera(y, e, g, w, ids, 1, 12)
    assert not known[0].any() and not known[2, :2].any()
    assert known[2, 2:4].all() and not known[:, 4].any()
    assert not np.any((y > 0)&known)


def test_nested_doses_row_order_invariance_and_seed_reproducibility():
    y, e, g, w, ids = example()
    low, _, _ = censor_genera(y, e, g, w, ids, .2, 12)
    high, _, _ = censor_genera(y, e, g, w, ids, .8, 12)
    assert not np.any(high&~low)
    perm = [2, 0, 1]
    reordered, _, _ = censor_genera(y[perm], e[perm], g, w, ids[perm], .2, 12)
    assert np.array_equal(low[perm], reordered)
    assert np.array_equal(low, censor_genera(y, e, g, w, ids, .2, 12)[0])


def test_rarity_weights_use_genus_training_prevalence_with_universe_filter():
    names, codes, weights = genus_contract(['A a', 'A b', 'B c'], ['A', 'A', 'B'],
        [1, 100, .25], [True, False, True])
    assert names.tolist() == ['A', 'B']
    assert codes.tolist() == [0, 0, 1] and np.allclose(weights, [1, 2])
    with pytest.raises(ValueError):
        genus_contract(['A a'], ['B'], [1], [True])


def test_weighted_mask_selects_high_weight_genus_more_often():
    y = np.tile([1, 0, 1, 0], (600, 1)); e = np.ones_like(y, bool)
    known, _, _ = censor_genera(y, e, [0, 0, 1, 1], [1, 100], np.arange(600), .5, 7)
    assert (~known[:, 2]).mean() > .95


def test_known_fraction_can_remain_high_while_positive_records_and_scores_change():
    y = np.zeros((1, 100), int); y[0, :2] = 1
    known = np.ones_like(y, bool); known[0, 0] = False
    truth = np.where(known, y, -1); rank = np.array([[0, 2]]); count = np.array([2])
    score = score_fixed_sets(truth, known, rank, count)['F1'][0]
    assert known.mean() == .99 and score == 0
    lower, upper = fixed_set_f1_bounds(truth, known, np.ones_like(known), rank, count)
    assert lower[0] <= .5 <= upper[0]  # Oracle F1 = 2/(2+2).


def test_no_missingness_scores_and_bounds_equal_oracle_including_empty_empty():
    y = np.array([[1, 0, 1], [0, 0, 0]]); e = np.ones_like(y, bool)
    rank = np.array([[0, 1, 2], [0, 1, 2]]); count = np.array([1, 0])
    score = score_fixed_sets(y, e, rank, count)['F1']
    low, high = fixed_set_f1_bounds(y, e, e, rank, count)
    assert np.allclose(score, [2/3, 1]) and np.array_equal(low, high)
    assert np.allclose(low, score)


def test_spatial_pairing_cancels_identical_policies_and_preserves_constant_bias():
    q = np.array([.1, .2, .3, .4]); blocks = ['a', 'a', 'b', 'b']; a = [True, False, True, False]
    result = paired_spatial_interval(q, blocks, a, a, repetitions=100)
    assert result['point'] == 0 and result['interval'] == [0, 0]
    result = paired_spatial_interval(np.repeat(-.03, 4), blocks, repetitions=100)
    assert np.allclose(result['interval'], [-.03, -.03])


@pytest.mark.parametrize('dose', [-.1, 1.1])
def test_invalid_dose_rejected(dose):
    y, e, g, w, ids = example()
    with pytest.raises(ValueError):
        censor_genera(y, e, g, w, ids, dose, 12)


def test_incomplete_oracle_and_nonpositive_weights_rejected():
    y, e, g, w, ids = example()
    with pytest.raises(ValueError):
        censor_genera(y, np.ones_like(e), g, w, ids, .2, 12)
    with pytest.raises(ValueError):
        censor_genera(y, e, g, [1, 0, 1], ids, .2, 12)
