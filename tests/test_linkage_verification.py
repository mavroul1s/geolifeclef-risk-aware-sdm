from collections import Counter
import itertools
import numpy as np
import pytest

from geolifeclef.taxonomic_queries import PublicKeys, QueryState
from geolifeclef.linkage_verification import (IdentityOracle, comparison_members,
    corrupt_memberships, record_keys, resolve_group, verification_cost)


def toy():
    # Two public groups, one heterogeneous target-label member in each.
    keys = PublicKeys(np.array([0, 0]), (np.array([0, 1]), np.array([2, 3])))
    answers = np.array([0, 1, 1, 0]); offsets = np.array([0, 2, 4])
    eligible = np.ones((4, 2), bool); prediction = np.tile([True, False], (4, 1))
    state = QueryState(eligible, np.zeros_like(eligible), [0, 0], prediction, record_keys(keys))
    support = comparison_members(eligible, state.known, [0, 0], np.array([1., 0., -1., 0.]), keys)
    return keys, answers, offsets, state, support


def test_unchecked_propagation_can_create_false_positives_and_absences():
    keys, answers, offsets, state, support = toy(); oracle = IdentityOracle(answers)
    resolve_group(state, 0, keys, offsets, oracle, 'unchecked_all', support)
    assert state.positive[1, 0] and not state.positive[1, 1]
    assert state.known[1].all() and oracle.resolutions == 1 and oracle.audits == 0


def test_full_audit_resolves_mismatch_and_charges_each_operation():
    keys, answers, offsets, state, support = toy(); oracle = IdentityOracle(answers)
    for group in range(2): resolve_group(state, group, keys, offsets, oracle, 'audit_all', support)
    np.testing.assert_array_equal(state.positive, np.eye(2, dtype=bool)[answers])
    assert state.known.all() and oracle.resolutions == 4 and oracle.audits == 2
    assert verification_cost(oracle.resolutions, oracle.audits, .5) == 5.


def test_support_audit_leaves_other_records_unknown():
    keys, answers, offsets, state, support = toy(); oracle = IdentityOracle(answers)
    for group in range(2): resolve_group(state, group, keys, offsets, oracle, 'audit_support', support)
    assert state.known[[0, 2]].all() and not state.known[[1, 3]].any()
    assert not state.queried[[1, 3]].any() and oracle.resolutions == 2 and oracle.audits == 0


def test_outside_target_follower_requires_resolution_after_mismatch():
    keys, _, offsets, state, support = toy(); oracle = IdentityOracle(np.array([0, -1, 1, 0]))
    resolve_group(state, 0, keys, offsets, oracle, 'audit_all', support)
    assert state.positive[0, 0] and not state.positive[1].any() and state.known[1].all()
    assert oracle.resolutions == 2 and oracle.audits == 1


def test_matching_followers_cost_audits_without_extra_identity_resolutions():
    keys, _, offsets, state, support = toy(); oracle = IdentityOracle(np.array([0, 0, 1, 1]))
    resolve_group(state, 0, keys, offsets, oracle, 'audit_all', support)
    assert oracle.resolutions == 1 and oracle.audits == 1
    with pytest.raises(ValueError): resolve_group(state, 0, keys, offsets, oracle, 'audit_all', support)
    assert oracle.resolutions == 1 and oracle.audits == 1


def test_overlapping_unresolved_records_do_not_reveal_absences_early():
    keys = PublicKeys(np.array([0, 0]), (np.array([0, 1]), np.array([0, 2])))
    eligible = np.ones((3, 3), bool)
    state = QueryState(eligible, np.zeros_like(eligible), [0]*3, np.zeros_like(eligible), record_keys(keys))
    support = (keys.cells[0], keys.cells[1]); oracle = IdentityOracle([0, 0, 1, 1])
    resolve_group(state, 0, keys, np.array([0, 2, 4]), oracle, 'audit_all', support)
    np.testing.assert_array_equal(state.known[0], [True, False, False])
    resolve_group(state, 1, keys, np.array([0, 2, 4]), oracle, 'audit_all', support)
    assert state.known.all() and state.positive[0, :2].all()


def test_corruption_preserves_inventory_genus_counts_and_unique_cells():
    keys = PublicKeys(np.array([0]*4), tuple(np.arange(i*10, (i+1)*10) for i in range(4)))
    key_answers = np.array([0, 1, -1, 2]); blocks = np.array(['a']*40)
    old = Counter((int(c), 0, int(a)) for members, a in zip(keys.cells, key_answers) for c in members)
    prefix = []
    for rate in [0., .1, .2]:
        moved, answers, offsets, receipt = corrupt_memberships(keys, key_answers, blocks, rate, 52, .2)
        new = Counter((int(c), int(g), int(a)) for k, g in enumerate(moved.genus)
            for c, a in zip(moved.cells[k], answers[offsets[k]:offsets[k+1]]))
        assert old == new and all(len(set(c)) == len(c) for c in moved.cells)
        assert receipt['realized_swaps'] == receipt['requested_swaps']
        prefix.append(np.concatenate(moved.cells) != np.concatenate(keys.cells))
    assert np.all(prefix[0] <= prefix[1]) and np.all(prefix[1] <= prefix[2])


def test_error_generation_never_crosses_genus_or_block():
    keys = PublicKeys(np.array([0, 1, 0, 1]), tuple(np.arange(i*4, (i+1)*4) for i in range(4)))
    _, _, _, receipt = corrupt_memberships(keys, [0, 1, 2, 3], ['a']*8+['b']*8, .2, 3, .2)
    assert receipt['realized_swaps'] == 0 and receipt['same_block_genus_category_pairs'] == 0


def test_empty_scope_and_invalid_costs_fail():
    keys, answers, offsets, state, _ = toy(); oracle = IdentityOracle(answers)
    with pytest.raises(ValueError): resolve_group(state, 0, keys, offsets, oracle, 'audit_support', (np.array([], int),)*2)
    assert oracle.resolutions == 0
    with pytest.raises(ValueError): verification_cost(1, 2, -.1)


def test_guarded_bounds_contain_all_tiny_target_oracles_at_every_group_prefix():
    for values in itertools.product([-1, 0, 1], repeat=4):
        for mode in ['audit_all', 'audit_support']:
            for order in [(0, 1), (1, 0)]:
                keys, _, offsets, state, support = toy(); oracle = IdentityOracle(values)
                truth = np.zeros((4, 2), bool)
                for row, answer in enumerate(values):
                    if answer >= 0: truth[row, answer] = True
                den = truth.sum(1)+state.prediction.sum(1)
                quality = np.divide(2*(truth & state.prediction).sum(1), den, out=np.ones(4), where=den > 0)
                for group in [None, *order]:
                    if group is not None: resolve_group(state, group, keys, offsets, oracle, mode, support)
                    lo, hi = state.bounds()
                    assert np.all(lo <= quality+1e-12) and np.all(quality <= hi+1e-12)
                    assert not np.any(state.known & (state.positive != truth))
