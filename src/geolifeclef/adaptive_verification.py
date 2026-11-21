"""Public-information adaptive controls for a fixed verification contract.

Equal-sized acceptance sets make absolute nonzero paired weights constant.
Paired width credit is then ordinary width credit on the same scope, up to a
constant. Canonical unscaled priorities avoid artificial floating-point ties.
"""
import numpy as np

POLICIES = ('static_scope_width', 'static_scope_width_per_cost', 'adaptive_scope_width',
            'adaptive_scope_width_per_cost_fixed', 'adaptive_scope_width_per_cost', 'adaptive_incidence_per_cost')


def equal_support_weight(weights):
    values = np.abs(np.asarray(weights, float)); nonzero = values[values > 0]
    if not len(nonzero) or not np.allclose(nonzero, nonzero[0], rtol=1e-13, atol=0):
        raise ValueError('Constant absolute nonzero weights required for this equivalence')
    return float(nonzero[0])


def mismatch_forecast(groups, resolutions, audits, alpha=1., beta=19.):
    mismatches = resolutions-groups
    if groups < 0 or audits < 0 or mismatches < 0 or mismatches > audits or alpha <= 0 or beta <= 0:
        raise ValueError('Invalid observed audit counts or prior')
    return float((alpha+mismatches)/(alpha+beta+audits))


class ScopeCredit:
    """Cache observable cell/genus counts, not private identities."""
    def __init__(self, state, keys, offsets, scope):
        self.keys = keys; cells = np.concatenate(keys.cells)
        groups = np.repeat(np.arange(len(keys.genus)), np.diff(offsets)); scope = np.asarray(scope, bool)
        if scope.shape != cells.shape: raise ValueError('Aligned fixed scope required')
        self.member_group = groups[scope]; self.member_cells = cells[scope]
        pairs, self.member_pair = np.unique(np.column_stack((self.member_cells,
            keys.genus[self.member_group])), axis=0, return_inverse=True)
        self.pair_cells = pairs[:, 0]; self.pair_genus = pairs[:, 1]
        self.pair_index = {tuple(map(int, pair)): j for j, pair in enumerate(pairs)}
        self.classes = tuple(state.classes[int(g)][state.eligible[int(c), state.classes[int(g)]]] for c, g in pairs)
        self.unknown = np.array([int((~state.known[c, indices]).sum()) for c, indices in zip(self.pair_cells, self.classes)])
        self.remaining = np.array([state.remaining[int(c), int(g)] for c, g in pairs], int)
        self.members = np.bincount(self.member_group, minlength=len(keys.genus))
        self.initial_credit = self.credit(state)

    def credit(self, state):
        low, high = state.bounds(); unknown = state.unknown_pred+state.unknown_other
        denominator = np.maximum(unknown[self.pair_cells], 1)*np.maximum(self.remaining, 1)
        value = (high-low)[self.pair_cells]*self.unknown/denominator
        return np.bincount(self.member_group, weights=value[self.member_pair], minlength=len(self.keys.genus))

    def incidence(self, state):
        low, high = state.bounds()
        active = (self.unknown > 0) & ((high-low)[self.pair_cells] > 0)
        return np.bincount(self.member_group, weights=active[self.member_pair], minlength=len(self.keys.genus))

    def observe(self, state, rows, genus):
        for cell in rows:
            pair = self.pair_index[int(cell), int(genus)]
            self.unknown[pair] = int((~state.known[cell, self.classes[pair]]).sum())
            self.remaining[pair] = state.remaining[int(cell), int(genus)]


def priorities(graph, state, policy, price, groups, resolutions, audits, alpha=1., beta=19.):
    if policy not in POLICIES or price < 0: raise ValueError('Invalid policy or price')
    if policy.startswith('static_'):
        score = graph.initial_credit.copy(); probability = mismatch_forecast(0, 0, 0, alpha, beta)
    else:
        score = graph.incidence(state) if policy == 'adaptive_incidence_per_cost' else graph.credit(state)
        probability = mismatch_forecast(groups, resolutions, audits, alpha, beta)
        if policy == 'adaptive_scope_width_per_cost_fixed': probability = mismatch_forecast(0, 0, 0, alpha, beta)
    expected_cost = 1+np.maximum(graph.members-1, 0)*(price+probability)
    if '_per_cost' in policy: score = score/expected_cost
    return score, expected_cost, probability


def choose_batch(score, members, queried, tie, batch_size):
    score, tie = np.asarray(score, float), np.asarray(tie, float)
    queried, members = np.asarray(queried, bool), np.asarray(members, int)
    if not (score.shape == tie.shape == queried.shape == members.shape) or not np.isfinite(score).all():
        raise ValueError('Aligned finite public priorities required')
    if batch_size <= 0: raise ValueError('Positive batch size required')
    order = np.lexsort((tie, -score)); order = order[(members[order] > 0) & ~queried[order]]
    return order[:batch_size]
