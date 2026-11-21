import itertools
import numpy as np
import pytest
from geolifeclef.taxonomic_queries import PublicKeys, QueryState, acquisition_order
from geolifeclef.linkage_verification import record_keys, comparison_members, IdentityOracle, resolve_group
from geolifeclef.adaptive_verification import (ScopeCredit, priorities, choose_batch,
    mismatch_forecast, equal_support_weight)


def problem():
    keys = PublicKeys(np.array([0, 0, 1]), (np.array([0, 1]), np.array([1, 2]), np.array([0, 2])))
    e = np.ones((3, 4), bool); genera = np.array([0, 0, 1, 1]); pred = np.array([[1, 0, 1, 0]]*3, bool)
    state = QueryState(e, np.zeros_like(e), genera, pred, record_keys(keys))
    offsets = np.array([0, 2, 4, 6]); w = np.array([.5, -.5, 0.])
    support = comparison_members(e, state.known, genera, w, keys)
    scope = np.concatenate([np.isin(c, s) for c, s in zip(keys.cells, support)])
    return keys, offsets, state, support, w, ScopeCredit(state, keys, offsets, scope)


def test_matched_width_and_paired_credit_differ_only_by_constant():
    keys, offsets, state, support, w, graph = problem()
    public = QueryState(state.eligible, state.positive, state.genera, state.prediction, keys)
    _, paired = acquisition_order(public, w, 'paired_width', 9)
    np.testing.assert_allclose(paired, graph.initial_credit*equal_support_weight(w), atol=1e-15)


def test_equivalence_requires_equal_absolute_nonzero_weights():
    assert equal_support_weight([.25, -.25, 0]) == .25
    with pytest.raises(ValueError): equal_support_weight([.25, -.5, 0])
    with pytest.raises(ValueError): equal_support_weight([0, 0])


def test_audit_forecast_uses_observed_counts_and_stays_in_unit_interval():
    assert mismatch_forecast(0, 0, 0) == .05
    assert mismatch_forecast(10, 10, 100) < .05
    assert mismatch_forecast(10, 30, 100) > .05
    with pytest.raises(ValueError): mismatch_forecast(10, 9, 100)
    with pytest.raises(ValueError): mismatch_forecast(10, 111, 100)


def test_cost_normalization_changes_order_for_equal_width_credit():
    keys, offsets, state, support, w, graph = problem(); graph.initial_credit[:] = 1.
    score, expected, _ = priorities(graph, state, 'static_scope_width_per_cost', .5, 0, 0, 0)
    assert graph.members.tolist() == [2, 1, 1]
    assert expected[0] > expected[1] and score[0] < score[1]


def test_fixed_controls_do_not_change_with_later_audit_forecasts():
    keys, offsets, state, support, w, graph = problem()
    a = priorities(graph, state, 'static_scope_width_per_cost', .1, 0, 0, 0)[0]
    b = priorities(graph, state, 'static_scope_width_per_cost', .1, 30, 35, 100)[0]
    np.testing.assert_array_equal(a, b)


def test_fixed_forecast_ablation_separates_learning_from_width_updates():
    keys, offsets, state, support, w, graph = problem()
    fixed, cost_fixed, p_fixed = priorities(graph, state, 'adaptive_scope_width_per_cost_fixed', .1, 1, 2, 2)
    learned, cost_learned, p_learned = priorities(graph, state, 'adaptive_scope_width_per_cost', .1, 1, 2, 2)
    assert p_fixed == .05 and p_learned > p_fixed
    assert cost_learned[0] > cost_fixed[0] and learned[0] < fixed[0]


def test_cached_public_counts_match_reconstruction_after_all_tiny_answers():
    for answers in itertools.product([-1, 0, 1], repeat=4):
        keys, offsets, state, support, w, graph = problem(); oracle = IdentityOracle([*answers, -1, -1])
        for group in [0, 1]:
            rows = resolve_group(state, group, keys, offsets, oracle, 'audit_support', support)
            graph.observe(state, rows, int(keys.genus[group]))
            for j, (cell, genus) in enumerate(zip(graph.pair_cells, graph.pair_genus)):
                assert graph.remaining[j] == state.remaining[int(cell), int(genus)]
                assert graph.unknown[j] == int((state.eligible[cell, state.classes[int(genus)]] & ~state.known[cell, state.classes[int(genus)]]).sum())
            assert np.isfinite(graph.credit(state)).all()


def test_adaptive_credit_responds_to_revealed_information_without_private_answers():
    keys, offsets, state, support, w, graph = problem(); before = graph.credit(state)
    oracle = IdentityOracle([0, 0, 1, 1, 2, 2])
    rows = resolve_group(state, 0, keys, offsets, oracle, 'audit_support', support); graph.observe(state, rows, 0)
    assert not np.array_equal(before, graph.credit(state))
    assert 'answers' not in graph.__dict__ and '_answers' not in graph.__dict__


def test_batch_uses_common_ties_and_excludes_queried_or_unscoped_groups():
    order = choose_batch([1., 1., 10., 20.], [1, 1, 0, 1], [False, False, False, True], [.8, .2, .1, .3], 32)
    np.testing.assert_array_equal(order, [1, 0])
    with pytest.raises(ValueError): choose_batch([1.], [1], [False], [.2], 0)
