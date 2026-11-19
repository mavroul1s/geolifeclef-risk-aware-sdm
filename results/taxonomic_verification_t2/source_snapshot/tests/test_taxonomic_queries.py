import itertools
import numpy as np
import pytest

from geolifeclef.taxonomic_queries import (PublicKeys, QueryState, acquisition_order,
    contrast_interval, simulate_keys)


def f1(y, pred):
    den = y.sum(1)+pred.sum(1)
    return np.divide(2*(y & pred).sum(1), den, out=np.ones(len(y)), where=den > 0)


def test_outside_target_resolution_cannot_imply_a_modeled_positive():
    keys = PublicKeys(np.array([0]), (np.array([0]),))
    s = QueryState(np.ones((1, 2), bool), np.zeros((1, 2), bool), [0, 0],
                   np.array([[True, False]]), keys)
    assert not s.known.any()
    s.query(0, -1)
    assert s.known.all() and not s.positive.any()
    np.testing.assert_array_equal(s.bounds()[0], [0.])


def test_one_answer_never_reveals_other_absences_while_records_remain():
    keys = PublicKeys(np.array([0, 0]), (np.array([0]), np.array([0])))
    s = QueryState(np.ones((1, 3), bool), np.zeros((1, 3), bool), [0]*3,
                   np.array([[True, False, False]]), keys)
    s.query(0, 0)
    np.testing.assert_array_equal(s.known, [[True, False, False]])
    s.query(1, -1)
    assert s.known.all() and s.positive.sum() == 1


def test_one_shared_query_updates_every_linked_cell_and_respects_native_masks():
    eligible = np.array([[True, True], [True, False]])
    keys = PublicKeys(np.array([0]), (np.array([0, 1]),))
    s = QueryState(eligible, np.zeros_like(eligible), [0, 0], np.zeros_like(eligible), keys)
    s.query(0, 1)
    np.testing.assert_array_equal(s.positive, [[False, True], [False, False]])
    np.testing.assert_array_equal(s.known, eligible)


def test_known_positive_overrides_overlapping_unknown_record():
    keys = PublicKeys(np.array([0]), (np.array([0]),))
    s = QueryState(np.ones((1, 2), bool), np.array([[True, False]]), [0, 0],
                   np.array([[True, False]]), keys)
    assert s.known[0, 0] and not s.known[0, 1]
    s.query(0, 0)
    assert s.known.all() and s.positive_count[0] == 1


def test_invalid_or_duplicate_queries_fail():
    s = QueryState(np.ones((1, 2), bool), np.zeros((1, 2), bool), [0, 1],
                   np.zeros((1, 2), bool), PublicKeys(np.array([0]), (np.array([0]),)))
    with pytest.raises(ValueError): s.query(0, 1)
    with pytest.raises(ValueError): s.query(-1, 0)
    with pytest.raises(ValueError): s.query(1, 0)
    s.query(0, 0)
    with pytest.raises(ValueError): s.query(0, 0)
    with pytest.raises(ValueError): acquisition_order(s, np.array([1.]), 'uniform', 8)


def test_exhaustive_small_oracles_and_query_orders_preserve_bounds():
    genus = np.array([0, 0, 1, 1])
    eligible = np.array([[True, True, True, False], [True, True, False, True]])
    pred = np.array([[True, False, True, False], [False, True, False, False]])
    weights = np.array([1., -1.])
    for bits in itertools.product([False, True], repeat=4):
        full = np.tile(bits, (2, 1)); full &= eligible
        initial, problems = simulate_keys(full, np.ones_like(eligible), np.ones(4, bool), genus, ['a', 'a'], 27, 1.)
        for keys, answers in problems.values():
            for order in [np.arange(len(answers)), np.arange(len(answers))[::-1]]:
                state = QueryState(eligible, initial & eligible, genus, pred, keys)
                oracle = f1(full, pred); old_width = np.inf
                for key in [None, *order]:
                    if key is not None: state.query(key, int(answers[key]))
                    lo, hi = state.bounds(); assert np.all(lo <= oracle+1e-12) and np.all(oracle <= hi+1e-12)
                    bounds = contrast_interval(lo, hi, weights)
                    assert bounds[0]-1e-12 <= weights@oracle <= bounds[1]+1e-12
                    assert bounds[1]-bounds[0] <= old_width+1e-12
                    old_width = bounds[1]-bounds[0]
                np.testing.assert_allclose(state.bounds()[0], oracle)


def test_shared_and_local_units_start_from_identical_censored_records():
    full = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    native = np.ones_like(full, bool); universe = np.array([True, False, True])
    initial, problems = simulate_keys(full, native, universe, [0, 0, 1], ['a']*3, 55, 1.)
    states = [QueryState(native & universe, initial, [0, 0, 1], np.zeros_like(native), p[0]) for p in problems.values()]
    np.testing.assert_array_equal(states[0].known, states[1].known)
    assert len(problems['block_shared'][1]) == 3 and len(problems['cell_local'][1]) == 6
    for state, (_, answers) in zip(states, problems.values()):
        for key, answer in enumerate(answers): state.query(key, int(answer))
        np.testing.assert_array_equal(state.positive, (full > 0) & universe)
        np.testing.assert_array_equal(state.known, native & universe)


def test_policies_cannot_receive_oracle_species_answers():
    keys = PublicKeys(np.array([0, 0, 0]), (np.array([0, 1]), np.array([1]), np.array([2])))
    state = QueryState(np.ones((3, 2), bool), np.zeros((3, 2), bool), [0, 0],
                       np.array([[True, False]]*3), keys)
    weights = np.array([1., -1., 0.])
    for policy in ['uniform_support', 'incidence_support', 'paired_width']:
        order, _ = acquisition_order(state, weights, policy, 8)
        assert set(order) == {0, 1}
    order, _ = acquisition_order(state, weights, 'incidence_support', 8)
    assert order[0] == 0
    assert set(state.keys.__dataclass_fields__) == {'genus', 'cells'}
    a, _ = acquisition_order(state, weights, 'paired_width', 8)
    b, _ = acquisition_order(state, weights, 'paired_width', 8)
    np.testing.assert_array_equal(a, b)


def test_invalid_or_incomplete_inventory_is_rejected():
    with pytest.raises(ValueError):
        simulate_keys([[-1, 0]], [[True, True]], [True, True], [0, 0], ['a'], 1)


def test_bounds_attain_exhaustive_binary_completion_endpoints_including_empty_sets():
    for predicted in [[], [0], [0, 2]]:
        pred = np.zeros((1, 4), bool); pred[0, predicted] = True
        for positives in [[], [0], [2]]:
            pos = np.zeros_like(pred); pos[0, positives] = True
            state = QueryState(np.ones_like(pred), pos, [0, 0, 1, 1], pred,
                PublicKeys(np.array([0, 1]), (np.array([0]), np.array([0]))))
            unknown = np.flatnonzero(~state.known[0]); values = []
            for bits in itertools.product([False, True], repeat=len(unknown)):
                full = pos.copy(); full[0, unknown] = bits; values.append(f1(full, pred)[0])
            lo, hi = state.bounds()
            np.testing.assert_allclose([lo[0], hi[0]], [min(values), max(values)])
