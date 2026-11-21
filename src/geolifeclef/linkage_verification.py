"""Synthetic linkage stress and explicitly charged target-label verification.

The audit is a perfect binary oracle for compatibility of modeled target labels,
including a single outside-target category. It is not a validated taxonomist,
botanical identity test, or free field information.
"""
from collections import defaultdict
from itertools import combinations
import hashlib
import numpy as np

from geolifeclef.taxonomic_queries import PublicKeys


def corrupt_memberships(keys, key_answers, blocks, rate, seed, maximum_rate=.05):
    """Swap record memberships within block/genus, preserving record truth.

    Only different target-label categories and groups with >=2 members can swap.
    Each position changes at most once; every group retains an original member.
    No group may acquire duplicate cells. A common swap sequence makes error
    doses nested. Rates describe deliberately informative stress, not field data.
    """
    answers = np.asarray(key_answers, int); blocks = np.asarray(blocks, str)
    if not 0 <= rate <= maximum_rate <= 1 or len(answers) != len(keys.cells):
        raise ValueError('Invalid error rate or key alignment')
    offsets = np.r_[0, np.cumsum([len(c) for c in keys.cells])]
    cells = np.concatenate(keys.cells) if len(keys.cells) else np.array([], int)
    n = len(cells); requested_swaps = int(np.floor(rate*n/2))
    maximum_swaps = int(np.floor(maximum_rate*n/2))
    strata = defaultdict(list)
    for key, (genus, members) in enumerate(zip(keys.genus, keys.cells)):
        if not len(members): raise ValueError('Empty concept key')
        if len(set(blocks[members])) != 1: raise ValueError('A source key crosses blocks')
        if len(members) >= 2: strata[blocks[members[0]], int(genus)].append(key)
    pairs = [(a, b) for values in strata.values() for a, b in combinations(values, 2) if answers[a] != answers[b]]
    order = np.arange(n); used = np.zeros(n, bool)
    group_used = np.zeros(len(answers), int)
    group_cells = [set(map(int, c)) for c in keys.cells]
    rng = np.random.default_rng(seed); swaps = []; attempts = 0
    while len(swaps) < maximum_swaps and pairs and attempts < max(1000, 100*maximum_swaps):
        attempts += 1; a, b = pairs[int(rng.integers(len(pairs)))]
        if group_used[a] >= len(keys.cells[a])-1 or group_used[b] >= len(keys.cells[b])-1: continue
        available_a = np.flatnonzero(~used[offsets[a]:offsets[a+1]])+offsets[a]
        available_b = np.flatnonzero(~used[offsets[b]:offsets[b+1]])+offsets[b]
        if not len(available_a) or not len(available_b): continue
        ia, ib = int(rng.choice(available_a)), int(rng.choice(available_b))
        ca, cb = int(cells[ia]), int(cells[ib])
        if ca == cb or cb in group_cells[a] or ca in group_cells[b]: continue
        group_cells[a].remove(ca); group_cells[a].add(cb)
        group_cells[b].remove(cb); group_cells[b].add(ca)
        used[ia] = used[ib] = True; group_used[a] += 1; group_used[b] += 1
        swaps.append((ia, ib))
    # Use the requested prefix of the common maximum-dose construction.
    for ia, ib in swaps[:requested_swaps]: order[ia], order[ib] = order[ib], order[ia]
    flat_answers = np.repeat(answers, np.diff(offsets))[order]
    moved_cells = cells[order]
    moved = PublicKeys(np.asarray(keys.genus, int).copy(),
        tuple(moved_cells[offsets[k]:offsets[k+1]].copy() for k in range(len(answers))))
    realized = min(requested_swaps, len(swaps))
    return moved, flat_answers, offsets, {
        'records': n, 'requested_rate': rate, 'requested_swaps': requested_swaps,
        'realized_swaps': realized, 'realized_rate': 2*realized/max(n, 1),
        'maximum_feasible_swaps_found': len(swaps), 'attempts': attempts,
        'same_block_genus_category_pairs': len(pairs),
        'record_permutation_sha256': hashlib.sha256(np.asarray(order, '<i8').tobytes()).hexdigest()}


def record_keys(keys):
    genus = np.repeat(keys.genus, [len(c) for c in keys.cells])
    cells = tuple(np.array([int(c)], int) for group in keys.cells for c in group)
    return PublicKeys(genus, cells)


def comparison_members(eligible, known, genera, weights, keys):
    """Public, fixed scope; never consult private identities or oracle losses."""
    classes = {g: np.flatnonzero(np.asarray(genera) == g) for g in np.unique(genera)}
    result = []
    for genus, cells in zip(keys.genus, keys.cells):
        result.append(np.array([int(cell) for cell in cells if weights[cell] != 0 and
            np.any(eligible[cell, classes[genus]] & ~known[cell, classes[genus]])], int))
    return tuple(result)


class IdentityOracle:
    def __init__(self, private_answers):
        self._answers = np.asarray(private_answers, int)
        self.resolutions = 0; self.audits = 0

    def resolve(self, record):
        self.resolutions += 1
        return int(self._answers[record])

    def compatible(self, representative, record):
        self.audits += 1
        return bool(self._answers[representative] == self._answers[record])


def resolve_group(state, group, keys, offsets, oracle, mode, support_members):
    """Atomic group action. Unverified followers remain explicitly unresolved."""
    if mode not in {'unchecked_all', 'audit_all', 'audit_support'}:
        raise ValueError('Unknown verification mode')
    cells = keys.cells[group]
    if mode == 'audit_support':
        selected = np.flatnonzero(np.isin(cells, support_members[group]))
    else: selected = np.arange(len(cells))
    if not len(selected): raise ValueError('An empty comparison group must be skipped without charge')
    if np.any(state.queried[offsets[group]+selected]):
        raise ValueError('A resolved member cannot be charged twice')
    representative = int(selected[np.argmin(cells[selected])])
    ref = int(offsets[group]+representative)
    answer = oracle.resolve(ref)
    for position in selected:
        record = int(offsets[group]+position)
        resolved = answer
        if mode != 'unchecked_all' and record != ref and not oracle.compatible(ref, record):
            resolved = oracle.resolve(record)
        state.query(record, resolved)
    return cells[selected]


def verification_cost(resolutions, audits, audit_price):
    if audit_price < 0 or resolutions < 0 or audits < 0: raise ValueError('Nonnegative costs required')
    return float(resolutions+audit_price*audits)
