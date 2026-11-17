"""Fixed-quota selection with a label-free predicted ecological-mass constraint.

This is a Lagrangian search heuristic, not a new optimization theorem or a
guarantee about observed species. Realized ecological service must be evaluated.
"""
from __future__ import annotations

import numpy as np

from geolifeclef.reliability import r1_accept, r1_hash


class QuotaSelection:
    def __init__(self, ids, coverage, groups):
        ids, groups = np.asarray(ids), np.asarray(groups).astype(str)
        if groups.shape != ids.shape or ids.ndim != 1:
            raise ValueError('Aligned one-dimensional IDs and groups required')
        self.ids, self.coverage = ids, coverage
        self.tie = np.array([r1_hash(f'20261113:{i}') for i in ids], np.uint64)
        allocation = r1_accept(np.zeros(len(ids)), ids, coverage, groups)
        self.groups = [(np.flatnonzero(groups == name), int(allocation[groups == name].sum()))
                       for name in np.unique(groups)]

    def accept(self, risk):
        risk = np.asarray(risk, float)
        if risk.shape != self.ids.shape or not np.isfinite(risk).all():
            raise ValueError('Aligned finite risk required')
        chosen = np.zeros(len(risk), bool)
        for take, quota in self.groups:
            chosen[take[np.lexsort((self.tie[take], risk[take]))[:quota]]] = True
        return chosen

    def preserve_mass(self, risk, predicted_mass, target=None, steps=50):
        """Choose using predictions only; targets/outcomes are never arguments."""
        risk, mass = np.asarray(risk, float), np.asarray(predicted_mass, float)
        if mass.shape != self.ids.shape or not np.isfinite(mass).all() or np.any(mass < 0):
            raise ValueError('Aligned nonnegative ecological-mass estimates required')
        target = self.coverage if target is None else target
        if not 0 <= target <= 1:
            raise ValueError('Invalid mass retention target')
        baseline = self.accept(risk)
        total = float(mass.sum())
        if total == 0:
            return baseline, {'feasible': True, 'lambda': 0., 'predicted_mass_retention': None,
                              'zero_predicted_mass': True}
        required = target*total
        achieved = lambda mask: float(mass[mask].sum())
        if achieved(baseline) >= required-1e-12:
            return baseline, {'feasible': True, 'lambda': 0., 'predicted_mass_retention': achieved(baseline)/total}
        maximum = self.accept(-mass)
        if achieved(maximum) < required-1e-12:
            return maximum, {'feasible': False, 'lambda': None,
                             'predicted_mass_retention': achieved(maximum)/total,
                             'reason': 'fixed group quotas make the proxy target infeasible'}
        normalized = mass/max(float(mass.mean()), 1e-12)
        lo, hi = 0., 1.
        chosen = self.accept(risk-hi*normalized)
        for _ in range(60):
            if achieved(chosen) >= required-1e-12:
                break
            lo, hi = hi, hi*2
            chosen = self.accept(risk-hi*normalized)
        else:
            return maximum, {'feasible': True, 'lambda': None,
                             'predicted_mass_retention': achieved(maximum)/total,
                             'reason': 'search saturated; maximum proxy-mass ordering used'}
        for _ in range(steps):
            mid = (lo+hi)/2
            proposal = self.accept(risk-mid*normalized)
            if achieved(proposal) >= required-1e-12:
                hi, chosen = mid, proposal
            else:
                lo = mid
        return chosen, {'feasible': True, 'lambda': hi,
                        'predicted_mass_retention': achieved(chosen)/total}


def paired_ecological_comparison(scores, rare_truth, candidate, baseline, blocks,
                                repetitions=2000, confidence=.95):
    """Joint fixed-policy spatial intervals for conditional F1 and occurrence service."""
    scores, rare = np.asarray(scores, float), np.asarray(rare_truth, float)
    a, b = np.asarray(candidate, bool), np.asarray(baseline, bool)
    _, inverse = np.unique(np.asarray(blocks).astype(str), return_inverse=True)
    n = int(inverse.max()+1)
    columns = [a, b, scores*a, scores*b, rare*a, rare*b, rare]
    values = np.stack([np.bincount(inverse, weights=v, minlength=n) for v in columns])
    rng = np.random.default_rng(20261116)
    draws = rng.integers(n, size=(repetitions, n))
    boot = values[:, draws].sum(2)
    valid = (boot[0] > 0) & (boot[1] > 0) & (boot[6] > 0)
    delta_f1 = boot[2, valid]/boot[0, valid]-boot[3, valid]/boot[1, valid]
    delta_rare = (boot[4, valid]-boot[5, valid])/boot[6, valid]
    tail = (1-confidence)/2
    return {'F1_gain': float(scores[a].mean()-scores[b].mean()),
        'rare_coverage_gain': float((rare[a].sum()-rare[b].sum())/rare.sum()) if rare.sum() else None,
        'F1_interval': np.quantile(delta_f1, [tail, 1-tail]).tolist() if valid.any() else None,
        'rare_coverage_interval': np.quantile(delta_rare, [tail, 1-tail]).tolist() if valid.any() else None,
        'confidence': confidence, 'clusters': n, 'valid_draws': int(valid.sum()),
        'interpretation': 'spatial cluster interval conditional on the fitted predictions and selection rules'}
