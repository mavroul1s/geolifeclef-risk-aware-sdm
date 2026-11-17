import itertools
import numpy as np
import pytest

from geolifeclef.genus_completion_bounds import positive_genus_f1_bounds


@pytest.mark.parametrize('values,codes', [
    ([-1, -1, -1, -1], [0, 0, 1, 1]),
    ([-1, -1, 1, 0], [0, 0, 1, 1]),
    ([-1, -1, -1, 0], [0, 0, 1, 2]),
    ([0, 1, 0, 0], [0, 0, 1, 1]),
])
def test_exact_endpoints_equal_exhaustive_completions_with_positive_genera(values, codes):
    y = np.array([values]); g = np.array(codes); known = y >= 0; eligible = np.ones_like(known)
    missing = np.flatnonzero(~known[0]); hidden = np.unique(g[missing])
    for permutation in itertools.permutations(range(4)):
        rank = np.array([permutation])
        for count in range(5):
            low, high = positive_genus_f1_bounds(y, known, eligible, g, rank, np.array([count]))
            possible = []
            for fill in itertools.product([0, 1], repeat=len(missing)):
                complete = y[0].copy(); complete[missing] = fill
                if any(not complete[g == genus].any() for genus in hidden):
                    continue
                den = count+complete.sum()
                possible.append(2*complete[list(permutation[:count])].sum()/den if den else 1.)
            assert np.isclose(low[0], min(possible)) and np.isclose(high[0], max(possible))


def test_positive_unknown_genus_rules_out_apparent_empty_inventory():
    low, high = positive_genus_f1_bounds([[-1, -1]], [[False, False]], [[True, True]],
                                        [0, 0], [[0, 1]], [0])
    assert low[0] == high[0] == 0


def test_partially_censored_genus_rejected():
    with pytest.raises(ValueError):
        positive_genus_f1_bounds([[-1, 0]], [[False, True]], [[True, True]], [0, 0], [[0, 1]], [1])
