"""Paired F1 difference bounds with a single completion shared by both models."""
import numpy as np


def shared_f1_difference_bounds(prediction_a, prediction_b, truth, known, eligible, empty_value=1):
    """Return cellwise extrema over binary completions of eligible unknown labels."""
    a, b, truth, known, eligible = (np.asarray(value, dtype=bool) for value in
                                    (prediction_a, prediction_b, truth, known, eligible))
    if a.ndim != 2 or any(value.shape != a.shape for value in (b, truth, known, eligible)):
        raise ValueError("Aligned cell-by-species masks required")
    if np.any((a | b | known | truth) & ~eligible) or empty_value not in (0, 1):
        raise ValueError("Invalid eligible masks or empty convention")

    observed = truth & known
    unknown = eligible & ~known
    k = observed.sum(axis=1)
    size_a, size_b = a.sum(axis=1), b.sum(axis=1)
    base_a, base_b = (observed & a).sum(axis=1), (observed & b).sum(axis=1)
    totals = unknown.sum(axis=1)
    first_only = (unknown & a & ~b).sum(axis=1)
    second_only = (unknown & ~a & b).sum(axis=1)
    both = (unknown & a & b).sum(axis=1)
    neither = totals - first_only - second_only - both

    # For positive denominators, the A-only coefficient is largest and B-only
    # smallest. The shared-species coefficient may fall on either side of zero.
    both_above_zero = size_a <= size_b
    before_high_both = np.where(both_above_zero, 0, neither)
    before_low_both = np.where(both_above_zero, neither, 0)

    def value(a_hits, b_hits, t):
        den_a, den_b = size_a + k + t, size_b + k + t
        fa = np.divide(2*a_hits, den_a, out=np.full(len(a), float(empty_value)), where=den_a > 0)
        fb = np.divide(2*b_hits, den_b, out=np.full(len(b), float(empty_value)), where=den_b > 0)
        return fa - fb

    lower = value(base_a, base_b, 0)
    upper = lower.copy()
    for t in range(1, int(totals.max(initial=0))+1):
        active = totals >= t
        take_a, take_b = np.minimum(first_only, t), np.minimum(second_only, t)
        high_left, low_left = t - take_a, t - take_b
        high_both = np.clip(high_left - before_high_both, 0, both)
        low_both = np.clip(low_left - before_low_both, 0, both)
        high_b = np.clip(high_left - both - neither, 0, second_only)
        low_a = np.clip(low_left - both - neither, 0, first_only)
        high = value(base_a + take_a + high_both, base_b + high_b + high_both, t)
        low = value(base_a + low_a + low_both, base_b + take_b + low_both, t)
        lower[active] = np.minimum(lower[active], low[active])
        upper[active] = np.maximum(upper[active], high[active])
    return lower, upper
