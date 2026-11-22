"""Exhaustive binary completions for the shared-inventory contrast."""
from itertools import product

import numpy as np
import pytest

from geolifeclef.paired_completion import shared_f1_difference_bounds


@pytest.mark.parametrize("empty_value", [0, 1])
def test_paired_bounds_match_exhaustive_binary_completions(empty_value):
    # Exhaust every three-class pair, positive/negative/unknown observation and
    # eligibility configuration, including empty predictions and inventories.
    eligible_rows, known_rows, known_truth, a_rows, b_rows = [], [], [], [], []
    extrema = []
    for labels in product(range(4), repeat=3):
        eligible = np.array(labels) > 0
        known = np.array(labels) >= 2
        positive = np.array(labels) == 3
        unknown = eligible & ~known
        sets = [np.array(value, bool) for value in product([False, True], repeat=3)
                if not np.any(np.array(value) & ~eligible)]
        for a, b in product(sets, repeat=2):
            values = []
            for hidden in product([False, True], repeat=int(unknown.sum())):
                truth = positive.copy()
                truth[unknown] = hidden
                score = []
                for prediction in (a, b):
                    denominator = int(prediction.sum() + truth.sum())
                    score.append(2*(prediction & truth).sum()/denominator if denominator else empty_value)
                values.append(score[0] - score[1])
            eligible_rows.append(eligible)
            known_rows.append(known)
            known_truth.append(positive)
            a_rows.append(a)
            b_rows.append(b)
            extrema.append([min(values), max(values)])
    got = shared_f1_difference_bounds(a_rows, b_rows, known_truth, known_rows, eligible_rows, empty_value)
    np.testing.assert_allclose(np.stack(got, axis=1), extrema, atol=1e-15, rtol=0)
