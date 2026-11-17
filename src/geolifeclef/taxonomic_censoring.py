"""Controlled species-identity censoring; predictors never consume these masks."""
from __future__ import annotations

import numpy as np


def genus_contract(names, genera, prevalence, universe):
    names, genera = np.asarray(names, str), np.asarray(genera, str)
    prevalence, universe = np.asarray(prevalence, float), np.asarray(universe, bool)
    if not (names.shape == genera.shape == prevalence.shape == universe.shape):
        raise ValueError('Aligned class metadata required')
    if any(not g or name.split()[0] != g for name, g in zip(names, genera)):
        raise ValueError('Canonical genus identity mismatch')
    if not np.isfinite(prevalence).all() or np.any(prevalence < 0):
        raise ValueError('Invalid training prevalence')
    genus_names, codes = np.unique(genera, return_inverse=True)
    weights = np.ones(len(genus_names))
    for code in range(len(genus_names)):
        values = prevalence[(codes == code) & universe]
        if len(values):
            weights[code] = 1 / np.sqrt(max(float(np.median(values)), 1e-6))
    return genus_names, codes, weights


def censor_genera(truth, eligible, genus_codes, weights, cell_ids, dose, seed):
    """Return known-label mask; hide entire positive genera with nested doses.

    Positives determine the synthetic missingness, never prediction/selection.
    Per-cell RNG makes masks invariant to input row order. Zero dose is identity.
    """
    y, eligible = np.asarray(truth), np.asarray(eligible, bool)
    codes, weights, ids = np.asarray(genus_codes, int), np.asarray(weights, float), np.asarray(cell_ids, int)
    if y.shape != eligible.shape or y.ndim != 2 or codes.shape != (y.shape[1],) or ids.shape != (len(y),):
        raise ValueError('Censoring identity mismatch')
    if not 0 <= dose <= 1 or np.any(y[eligible] < 0) or np.any((y > 0) & ~eligible):
        raise ValueError('Complete eligible binary labels required')
    if np.any(~np.isin(y[eligible], [0, 1])) or np.any(codes < 0) or np.any(codes >= len(weights)):
        raise ValueError('Invalid binary/genus contract')
    if not np.isfinite(weights).all() or np.any(weights <= 0) or np.any(ids < 0):
        raise ValueError('Positive finite weights and nonnegative cell IDs required')
    known = eligible.copy()
    positive_genera = np.zeros(len(y), int)
    hidden_genera = np.zeros(len(y), int)
    for row, cell_id in enumerate(ids):
        candidates = np.unique(codes[y[row] > 0])
        positive_genera[row] = len(candidates)
        n = int(np.ceil(dose * len(candidates)))
        hidden_genera[row] = n
        if not n:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(cell_id)]))
        uniforms = np.maximum(rng.random(len(candidates)), np.finfo(float).tiny)
        order = np.argsort(-np.log(uniforms) / weights[candidates], kind='stable')
        hidden = candidates[order[:n]]
        known[row] &= ~np.isin(codes, hidden)
    return known, positive_genera, hidden_genera


def paired_spatial_interval(values, blocks, accepted_a=None, accepted_b=None,
                            repetitions=4000, confidence=.975, seed=20261117):
    """Paired mean/selection-contrast interval, conditional on fixed policies."""
    values = np.asarray(values, float)
    names, inverse = np.unique(np.asarray(blocks, str), return_inverse=True)
    if values.shape != inverse.shape or not np.isfinite(values).all():
        raise ValueError('Finite aligned cell values required')
    a = np.ones(len(values), bool) if accepted_a is None else np.asarray(accepted_a, bool)
    b = np.zeros(len(values), bool) if accepted_b is None else np.asarray(accepted_b, bool)
    if a.shape != values.shape or b.shape != values.shape or not a.any():
        raise ValueError('Aligned nonempty acceptance required')
    columns = [a, values*a, b, values*b]
    totals = np.array([np.bincount(inverse, weights=c, minlength=len(names)) for c in columns])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(names), size=(repetitions, len(names)))
    boot = totals[:, draws].sum(2)
    valid = boot[0] > 0
    contrast = values[a].mean()
    result = boot[1] / np.maximum(boot[0], 1)
    if b.any():
        valid &= boot[2] > 0
        result -= boot[3] / np.maximum(boot[2], 1)
        contrast -= values[b].mean()
    tail = (1-confidence)/2
    return {'point': float(contrast), 'interval': np.quantile(result[valid], [tail, 1-tail]).tolist(),
            'clusters': len(names), 'valid_draws': int(valid.sum()), 'confidence': confidence}
