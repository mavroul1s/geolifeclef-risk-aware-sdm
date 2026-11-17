import numpy as np
import pytest

from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.reliability import r1_accept


def test_cached_quotas_match_original_selector_with_ties_and_permutation():
    ids = np.arange(31)
    groups = np.array(['a']*12+['b']*13+['c']*6)
    risk = np.round(np.linspace(1, 0, 31), 1)
    selector = QuotaSelection(ids, .8, groups)
    np.testing.assert_array_equal(selector.accept(risk), r1_accept(risk, ids, .8, groups))
    order = np.random.default_rng(19).permutation(len(ids))
    permuted = QuotaSelection(ids[order], .8, groups[order]).accept(risk[order])
    np.testing.assert_array_equal(np.sort(ids[order][permuted]), np.sort(ids[selector.accept(risk)]))


def test_proxy_constraint_changes_selection_and_preserves_each_group_budget():
    selector = QuotaSelection(np.arange(8), .5, ['a']*4+['b']*4)
    risk = np.array([0., .1, .8, .9]*2)
    mass = np.array([0., 0., 2., 3.]*2)
    chosen, record = selector.preserve_mass(risk, mass)
    assert chosen.sum() == 4 and chosen[:4].sum() == 2
    assert mass[chosen].sum() >= .5*mass.sum()
    assert record['feasible'] and record['lambda'] > 0


def test_already_feasible_and_zero_mass_leave_base_decisions_unchanged():
    selector = QuotaSelection(np.arange(6), .5, ['a']*6)
    risk = np.arange(6)
    for mass in [np.ones(6), np.zeros(6)]:
        result, info = selector.preserve_mass(risk, mass)
        np.testing.assert_array_equal(result, selector.accept(risk))
        assert info['lambda'] == 0


def test_infeasible_group_rounding_is_reported_without_claiming_a_guarantee():
    # One group must receive zero budget; choose its identity from the fixed rule.
    selector = QuotaSelection(np.arange(3), .34, ['a', 'b', 'c'])
    selected = selector.accept(np.zeros(3))
    mass = (~selected).astype(float)
    chosen, info = selector.preserve_mass(np.zeros(3), mass, target=1)
    assert not info['feasible'] and mass[chosen].sum() == 0


def test_invalid_mass_does_not_silently_affect_selection():
    selector = QuotaSelection(np.arange(2), .5, ['a', 'a'])
    with pytest.raises(ValueError):
        selector.preserve_mass([0, 1], [-1, 2])
