import numpy as np
import pytest

from geolifeclef.reliability_analysis import selection_gain_decomposition


def test_selecting_only_an_easier_group_has_pure_composition_gain():
    d = selection_gain_decomposition([.9, .9, .1, .1], [1, 1, 0, 0], ['easy', 'easy', 'hard', 'hard'])
    assert d['gain'] == pytest.approx(.4)
    assert d['composition'] == pytest.approx(.4)
    assert d['within'] == 0
    assert next(g for g in d['groups'] if g['group'] == 'hard')['accepted_mean'] is None


def test_equal_group_coverage_can_have_pure_within_group_gain():
    d = selection_gain_decomposition([.9, .7, .3, .1], [1, 0, 1, 0], ['a', 'a', 'b', 'b'])
    assert d['composition'] == pytest.approx(0)
    assert d['within'] == pytest.approx(.1)
    assert d['gain'] == pytest.approx(.1)


def test_decomposition_reconstructs_every_nonempty_selection_of_a_small_cohort():
    scores = np.array([.1, .3, .9, .4, .6])
    for bits in range(1, 32):
        selected = np.array([bool(bits & (1 << i)) for i in range(5)])
        d = selection_gain_decomposition(scores, selected, ['a', 'a', 'b', 'b', 'c'])
        assert d['composition']+d['within'] == pytest.approx(scores[selected].mean()-scores.mean())


def test_empty_acceptance_is_reported_as_undefined():
    with pytest.raises(ValueError):
        selection_gain_decomposition([.1, .2], [0, 0], ['a', 'b'])
