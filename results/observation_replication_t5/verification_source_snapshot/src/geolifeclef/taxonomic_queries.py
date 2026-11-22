"""Synthetic shared-concept verification with conservative F1 outer bounds.

Public query policies receive incidence, genus, predictions and revealed labels.
Private species identities are kept separately and supplied only at query time.
The complete recorded-inventory and reliable shared-identity assumptions are
simulation contracts, not established properties of field inventories.
"""
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class PublicKeys:
    genus: np.ndarray
    cells: tuple


def simulate_keys(full_truth, native, universe, genera, blocks, seed, fraction=.5):
    """Two cost units, with identical censored records and initial information.

    Hide Bernoulli-selected block/species concepts, then either link all their
    cells or issue a separate anonymous key at each cell. IDs are permuted;
    neither sorted species indices nor private answers are policy inputs.
    """
    truth, native = np.asarray(full_truth), np.asarray(native, bool)
    universe, genera, blocks = np.asarray(universe, bool), np.asarray(genera, int), np.asarray(blocks, str)
    if truth.shape != native.shape or universe.shape != (truth.shape[1],) or genera.shape != universe.shape:
        raise ValueError('Misaligned label space')
    if blocks.shape != (len(truth),) or not np.isin(truth[native], [0, 1]).all():
        raise ValueError('Complete recorded native labels required')
    if not 0 <= fraction <= 1:
        raise ValueError('Invalid censoring fraction')
    rows, species = np.nonzero((truth > 0) & native)
    _, block_codes = np.unique(blocks, return_inverse=True)
    pairs = np.column_stack((block_codes[rows], species))
    concepts, inverse = np.unique(pairs, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    hidden = rng.random(len(concepts)) < fraction
    initial = np.zeros_like(native)
    keep = ~hidden[inverse] & universe[species]
    initial[rows[keep], species[keep]] = True
    records = [[] for _ in concepts]
    for row, key in zip(rows[hidden[inverse]], inverse[hidden[inverse]]):
        records[key].append(int(row))
    shared = [(int(concepts[k, 1]), np.array(records[k], int)) for k in np.flatnonzero(hidden)]
    local = [(s, np.array([row], int)) for s, cells in shared for row in cells]
    output = {}
    for kind, entries in [('block_shared', shared), ('cell_local', local)]:
        order = rng.permutation(len(entries))
        answers = np.array([entries[i][0] if universe[entries[i][0]] else -1 for i in order], int)
        codes = np.array([genera[entries[i][0]] for i in order], int)
        incidence = tuple(entries[i][1] for i in order)
        output[kind] = (PublicKeys(codes, incidence), answers)
    return initial, output


class QueryState:
    def __init__(self, eligible, initial_positive, genera, prediction, keys):
        self.eligible = np.asarray(eligible, bool)
        self.positive = np.asarray(initial_positive, bool).copy()
        self.genera = np.asarray(genera, int)
        self.prediction = np.asarray(prediction, bool)
        if self.positive.shape != self.eligible.shape or self.prediction.shape != self.eligible.shape:
            raise ValueError('Aligned cell/class arrays required')
        if np.any(self.positive & ~self.eligible) or np.any(self.prediction & ~self.eligible):
            raise ValueError('Positive/prediction outside target')
        if len(keys.genus) != len(keys.cells) or self.genera.shape != (self.eligible.shape[1],):
            raise ValueError('Aligned public key metadata required')
        self.keys = keys
        self.classes = {g: np.flatnonzero(self.genera == g) for g in np.unique(self.genera)}
        self.remaining = Counter()
        for genus, cells in zip(keys.genus, keys.cells):
            if len(cells) != len(np.unique(cells)) or np.any(cells < 0) or np.any(cells >= len(self.eligible)):
                raise ValueError('Invalid key incidence')
            for cell in cells:
                self.remaining[int(cell), int(genus)] += 1
        self.known = self.eligible.copy()
        for cell, genus in self.remaining:
            self.known[cell, self.classes.get(genus, [])] = False
        self.known |= self.positive
        self.queried = np.zeros(len(keys.genus), bool)
        self.positive_count = self.positive.sum(1)
        self.hits = (self.positive & self.prediction).sum(1)
        self.predicted_count = self.prediction.sum(1)
        self.unknown_pred = (self.eligible & ~self.known & self.prediction).sum(1)
        self.unknown_other = (self.eligible & ~self.known & ~self.prediction).sum(1)

    def bounds(self, rows=None):
        rows = slice(None) if rows is None else rows
        low_den = self.predicted_count[rows] + self.positive_count[rows] + self.unknown_other[rows]
        high_den = self.predicted_count[rows] + self.positive_count[rows] + self.unknown_pred[rows]
        low = np.divide(2*self.hits[rows], low_den, out=np.ones_like(low_den, float), where=low_den > 0)
        high = np.divide(2*(self.hits[rows]+self.unknown_pred[rows]), high_den,
                         out=np.ones_like(high_den, float), where=high_den > 0)
        return low, high

    def query(self, key, answer):
        """Resolve one shared key; -1 is a recorded outside-target species."""
        if key < 0 or key >= len(self.queried):
            raise ValueError('Unknown query key')
        if self.queried[key]:
            raise ValueError('A key cannot be charged or resolved twice')
        genus = int(self.keys.genus[key])
        if answer != -1 and (answer < 0 or answer >= len(self.genera) or self.genera[answer] != genus):
            raise ValueError('Answer outside the public genus')
        self.queried[key] = True
        for cell in self.keys.cells[key]:
            if answer != -1 and self.eligible[cell, answer] and not self.positive[cell, answer]:
                self.positive[cell, answer] = True
                self.positive_count[cell] += 1
                self.hits[cell] += int(self.prediction[cell, answer])
                if not self.known[cell, answer]:
                    if self.prediction[cell, answer]: self.unknown_pred[cell] -= 1
                    else: self.unknown_other[cell] -= 1
                    self.known[cell, answer] = True
            self.remaining[int(cell), genus] -= 1
            if self.remaining[int(cell), genus] == 0:
                candidates = self.classes.get(genus, np.array([], int))
                reveal = candidates[self.eligible[cell, candidates] & ~self.known[cell, candidates]]
                n_pred = int(self.prediction[cell, reveal].sum())
                self.unknown_pred[cell] -= n_pred
                self.unknown_other[cell] -= len(reveal)-n_pred
                self.known[cell, reveal] = True
        return self.keys.cells[key]


def acquisition_order(state, weights, policy, seed):
    """Fixed batch priorities, without private species answers or oracle losses.

    Width credit distributes each cell's initial width by the unknown classes
    in a genus and divides by its unresolved record count. It is a heuristic,
    not an expected information-gain estimator or a reproduction of VMA/LURE.
    """
    if state.queried.any():
        raise ValueError('Fixed batch priorities require the initial observation state')
    weights = np.asarray(weights, float)
    if weights.shape != (len(state.eligible),) or not np.isfinite(weights).all():
        raise ValueError('Aligned finite contrast weights required')
    policies = {'uniform', 'uniform_support', 'incidence_support', 'global_width', 'paired_width'}
    if policy not in policies:
        raise ValueError('Unknown acquisition policy')
    lower, upper = state.bounds()
    width = upper-lower
    total_unknown = state.unknown_pred + state.unknown_other
    contributions = {}
    for (cell, genus), count in state.remaining.items():
        classes = state.classes.get(genus, np.array([], int))
        unknown = int((state.eligible[cell, classes] & ~state.known[cell, classes]).sum())
        credit = width[cell]*unknown/max(total_unknown[cell], 1)/count
        contributions[cell, genus] = (credit, bool(unknown and weights[cell] != 0))
    scores = np.zeros(len(state.keys.genus))
    support = np.zeros(len(scores), bool)
    for key, (genus, cells) in enumerate(zip(state.keys.genus, state.keys.cells)):
        for cell in cells:
            credit, relevant = contributions[int(cell), int(genus)]
            support[key] |= relevant
            if policy == 'global_width': scores[key] += credit/len(weights)
            elif policy == 'paired_width': scores[key] += credit*abs(weights[cell])
            elif policy == 'incidence_support': scores[key] += int(relevant)
    random = np.random.default_rng(seed).random(len(scores))
    order = np.lexsort((random, -scores))
    if policy in {'uniform_support', 'incidence_support', 'paired_width'}:
        order = order[support[order]]
    return order, scores


def contrast_interval(lower, upper, weights):
    lo = np.where(weights >= 0, weights*lower, weights*upper).sum()
    hi = np.where(weights >= 0, weights*upper, weights*lower).sum()
    return float(lo), float(hi)
