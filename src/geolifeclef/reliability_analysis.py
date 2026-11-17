"""Descriptive analysis of selection gains; separate from frozen GPU code."""
from __future__ import annotations

import numpy as np


def selection_gain_decomposition(scores, accepted, groups):
    """Exact composition/within-stratum identity for an observed cohort.

    gain = sum_g (q_g - pi_g) * full_mean_g
         + sum_g q_g * (accepted_mean_g - full_mean_g)

    pi is full-cohort group share; q is accepted-cohort share. A zero-retention
    group's within term is zero, without inventing an accepted mean. This is a
    reweighting identity, not causal identification or a new uncertainty theorem.
    """
    scores, accepted, groups = np.asarray(scores, float), np.asarray(accepted, bool), np.asarray(groups).astype(str)
    if scores.ndim != 1 or accepted.shape != scores.shape or groups.shape != scores.shape:
        raise ValueError('Aligned one-dimensional inputs required')
    if not len(scores) or not accepted.any() or not np.isfinite(scores).all():
        raise ValueError('Finite scores and a nonempty accepted cohort required')
    composition, within, details = 0., 0., []
    for name in np.unique(groups):
        member = groups == name
        chosen = member & accepted
        pi, q = member.mean(), chosen.sum()/accepted.sum()
        mean = scores[member].mean()
        selected_mean = scores[chosen].mean() if chosen.any() else None
        between = (q-pi)*mean
        inside = q*(selected_mean-mean) if chosen.any() else 0.
        composition += between
        within += inside
        details.append({'group': str(name), 'full_n': int(member.sum()), 'accepted_n': int(chosen.sum()),
            'full_mean': float(mean), 'accepted_mean': float(selected_mean) if chosen.any() else None,
            'composition_contribution': float(between), 'within_contribution': float(inside)})
    observed = float(scores[accepted].mean()-scores.mean())
    if not np.isclose(composition+within, observed, atol=1e-12):
        raise ArithmeticError('Selection decomposition failed to reconstruct the gain')
    return {'gain': observed, 'composition': float(composition), 'within': float(within),
            'full_mean': float(scores.mean()), 'accepted_mean': float(scores[accepted].mean()),
            'groups': details, 'interpretation': 'descriptive composition identity; not causal attribution'}
