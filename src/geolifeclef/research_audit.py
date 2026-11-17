"""Retrospective diagnostics, never prediction policies fitted on assessment labels."""
from __future__ import annotations

import numpy as np


def prefix_f1(ranked: np.ndarray, truths: list[np.ndarray]) -> np.ndarray:
    """F1 of every nonempty prefix; the last axis enumerates set sizes 1..K.

    Accept [survey, expert, rank] integer columns. Truths must be unique columns.
    These are label-using diagnostics and must never be used at inference time.
    """
    ranked = np.asarray(ranked)
    if ranked.ndim != 3 or len(ranked) != len(truths) or ranked.shape[-1] == 0:
        raise ValueError("Expected nonempty [survey, expert, rank] and one truth set per survey")
    if not np.issubdtype(ranked.dtype, np.integer) or np.any(ranked < 0):
        raise ValueError("Rankings must contain nonnegative integer columns")
    if np.any(np.diff(np.sort(ranked, axis=-1), axis=-1) == 0):
        raise ValueError("A ranking contains duplicate species")
    hits = np.empty(ranked.shape, dtype=bool)
    sizes = []
    for i, truth in enumerate(truths):
        truth = np.asarray(truth)
        if truth.ndim != 1 or len(np.unique(truth)) != len(truth):
            raise ValueError("Truths must be unique one-dimensional label sets")
        hits[i] = np.isin(ranked[i], truth)
        sizes.append(len(truth))
    k = np.arange(1, ranked.shape[-1] + 1)
    return 2 * hits.cumsum(axis=-1) / (np.asarray(sizes)[:, None, None] + k)


def set_size_ceiling(truth_size, available_true, minimum=1, maximum=None):
    """Oracle F1 for a predictor limited to a given label support and size range.

    Assumes enough distinct output labels to fill the selected size. `available_true`
    counts truth labels a support-limited predictor is allowed to output. This is
    not an accuracy forecast and does not apply to unrestricted label outputs.
    """
    m, a = np.broadcast_arrays(np.asarray(truth_size), np.asarray(available_true))
    if (minimum < 1 or (maximum is not None and maximum < minimum)
            or np.any(m < 1) or np.any(a < 0) or np.any(a > m)):
        raise ValueError("Invalid truth/support counts or set-size range")
    k = np.maximum(a, minimum)
    if maximum is not None:
        k = np.minimum(k, maximum)
    return 2 * np.minimum(a, k) / (m + k)


def cluster_bootstrap(values, clusters, *, repetitions=2000, seed=20261113):
    """Resample whole clusters, estimating the survey-weighted mean each time.

    Conditional on the saved models/data. Does not correct adaptive selection,
    dependence between clusters, or mismatched target geography.
    """
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if (values.ndim != 1 or clusters.shape != values.shape or len(values) == 0
            or not np.isfinite(values).all() or repetitions < 1):
        raise ValueError("Expected aligned finite values and clusters")
    names, inverse = np.unique(clusters.astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 256):
        draw = rng.integers(len(names), size=(min(256, repetitions-start), len(names)))
        estimates[start:start+len(draw)] = sums[draw].sum(1) / counts[draw].sum(1)
    return {"mean": float(values.mean()), "clusters": len(names),
            "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "repetitions": repetitions, "seed": seed,
            "interpretation": "retrospective descriptive interval; no adaptive-selection correction"}
