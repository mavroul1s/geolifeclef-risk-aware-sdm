"""Model-uncertainty and selection helpers for the frozen R1 study."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def r1_hash(text):
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:15], 16)


def r1_binary_entropy(probability):
    p = np.clip(np.asarray(probability, np.float32), 1e-7, 1-1e-7)
    return -(p*np.log(p)+(1-p)*np.log1p(-p))


def r1_signals(mean, second_moment, average_entropy, member_sets, k=18):
    """Label-free signals from ensemble moments and each member's top-k set."""
    p = np.asarray(mean, np.float32)
    k = min(k, p.shape[1])
    ranks = np.argsort(-p, axis=1, kind='stable')[:, :k]
    top = np.take_along_axis(p, ranks, axis=1)
    variance = np.maximum(np.asarray(second_moment, np.float32)-p*p, 0)
    entropy = r1_binary_entropy(p)
    js = np.maximum(entropy.mean(1)-np.asarray(average_entropy), 0)
    disagreement = np.zeros(len(p), np.float32)
    pairs = 0
    for a in range(member_sets.shape[1]):
        for b in range(a+1, member_sets.shape[1]):
            intersection = (member_sets[:, a, :, None] == member_sets[:, b, None, :]).any(2).sum(1)
            disagreement += 1-intersection/np.maximum(2*k-intersection, 1)
            pairs += 1
    result = {'probability_mass': p.sum(1), 'max_probability': p.max(1),
        'mean_top18_confidence': top.mean(1),
        'expected_F1_surrogate': 2*top.sum(1)/(k+p.sum(1)),
        'top18_entropy': r1_binary_entropy(top).mean(1),
        'all_label_entropy': entropy.mean(1),
        'ensemble_variance': variance.mean(1),
        'top18_variance': np.take_along_axis(variance, ranks, axis=1).mean(1),
        'ensemble_mutual_information': js,
        'set_disagreement': disagreement/max(pairs, 1)}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite confidence signal')
    return result, ranks


def r1_area_bins(area, training_area, quantiles=(.2, .4, .6, .8)):
    source = pd.to_numeric(pd.Series(training_area), errors='coerce').to_numpy(float)
    source = source[np.isfinite(source) & (source >= 0)]
    edges = np.unique(np.quantile(np.log1p(source), quantiles)) if len(source) else np.array([])
    values = pd.to_numeric(pd.Series(area), errors='coerce').to_numpy(float)
    valid = np.isfinite(values) & (values >= 0)
    bins = np.full(len(values), 'missing', dtype='U16')
    bins[valid] = np.searchsorted(edges, np.log1p(values[valid]), side='right').astype(str)
    return bins, edges


def r1_accept(risk, ids, coverage, groups=None):
    """Exact total budget with label-free, largest-remainder group quotas."""
    risk, ids = np.asarray(risk, float), np.asarray(ids)
    if risk.ndim != 1 or ids.shape != risk.shape or not np.isfinite(risk).all() or not 0 < coverage <= 1:
        raise ValueError('Invalid selection inputs')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate survey IDs in a selection pool')
    count = int(np.ceil(len(risk)*coverage))
    tie = np.array([r1_hash(f'20261113:{i}') for i in ids], dtype=np.uint64)
    accepted = np.zeros(len(risk), bool)
    if groups is None:
        accepted[np.lexsort((tie, risk))[:count]] = True
        return accepted
    groups = np.asarray(groups).astype(str)
    if groups.shape != risk.shape:
        raise ValueError('Group dimensions disagree')
    names, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    exact = sizes*count/len(risk)
    quotas = np.floor(exact).astype(int)
    remaining = count-quotas.sum()
    group_tie = np.array([r1_hash(name) for name in names], dtype=np.uint64)
    allocations = np.lexsort((group_tie, -(exact-quotas)))[:remaining]
    quotas[allocations] += 1
    for group, quota in enumerate(quotas):
        take = np.flatnonzero(inverse == group)
        accepted[take[np.lexsort((tie[take], risk[take]))[:quota]]] = True
    assert accepted.sum() == count
    return accepted


R1_MODEL_FEATURES = ['probability_mass', 'max_probability', 'mean_top18_confidence',
    'expected_F1_surrogate', 'top18_entropy', 'all_label_entropy', 'ensemble_variance',
    'top18_variance', 'ensemble_mutual_information', 'set_disagreement']
R1_METADATA_FEATURES = ['log_area', 'year', 'log_geo_uncertainty']


def r1_fit_selectors(calibration, assessment, config):
    """Fit only to calibration.loss; assessment need not contain a label column."""
    models = {'area_only': ['log_area'], 'metadata_only': R1_METADATA_FEATURES,
        'metadata_distance': R1_METADATA_FEATURES+['log_pa_distance'],
        'model_only_risk': R1_MODEL_FEATURES,
        'metadata_model_risk': R1_METADATA_FEATURES+R1_MODEL_FEATURES,
        'metadata_distance_model_risk': R1_METADATA_FEATURES+['log_pa_distance']+R1_MODEL_FEATURES,
        'metadata_support_risk': R1_METADATA_FEATURES+['log_pa_distance', 'environment_novelty'],
        'full_risk': R1_METADATA_FEATURES+['log_pa_distance', 'environment_novelty']+R1_MODEL_FEATURES}
    if not np.isfinite(calibration.loss).all():
        raise ValueError('Calibration losses must be finite')
    risks, fitting = {}, {}
    for name, features in models.items():
        model = HistGradientBoostingRegressor(**config)
        model.fit(calibration[features].to_numpy(float), calibration.loss.to_numpy(float))
        risks[name] = model.predict(assessment[features].to_numpy(float))
        fitting[name] = {'features': features, 'calibration_surveys': len(calibration)}
    risks['random'] = np.asarray([r1_hash(f'20261113:{i}') for i in assessment.surveyId], float)
    risks['distance_only'] = assessment.log_pa_distance.to_numpy()
    risks['mean_top18_confidence'] = -assessment.mean_top18_confidence.to_numpy()
    for name in ['top18_entropy', 'all_label_entropy', 'ensemble_variance',
                 'ensemble_mutual_information', 'set_disagreement', 'environment_novelty']:
        risks[name] = assessment[name].to_numpy()
    return risks, fitting


def r1_selection_tables(frame, risks, coverages, policies):
    """Evaluation only: losses never influence selection order or quotas."""
    curves, countries, selected_primary = [], [], {}
    for policy in policies:
        groups = None if policy == 'unconstrained' else frame.country.astype(str).to_numpy()
        if policy == 'country_area':
            groups = (frame.country.astype(str)+'|area='+frame.area_bin.astype(str)).to_numpy()
        for method, risk in risks.items():
            for coverage in coverages:
                accepted = r1_accept(risk, frame.surveyId.to_numpy(), coverage, groups)
                chosen = frame.loc[accepted]
                if policy == 'country_area' and coverage == .8:
                    selected_primary[method] = accepted
                for group, mask in [('all', np.ones(len(chosen), bool)),
                    ('core', chosen.country.isin(['Denmark', 'Netherlands'])),
                    ('outside_core', ~chosen.country.isin(['Denmark', 'Netherlands']))]:
                    s = chosen.loc[mask]
                    curves.append({'policy': policy, 'method': method, 'coverage': coverage,
                        'group': group, 'retained_surveys': len(s),
                        'sample_F1': float(s.sample_F1.mean()) if len(s) else None,
                        'rare_recall': float(s.rare_hits.sum()/s.rare_truth.sum()) if s.rare_truth.sum() else None})
                for country, original in frame.groupby('country'):
                    s = chosen[chosen.country == country]
                    countries.append({'policy': policy, 'method': method, 'coverage': coverage,
                        'country': country, 'available_surveys': len(original), 'retained_surveys': len(s),
                        'country_retention': len(s)/len(original),
                        'sample_F1': float(s.sample_F1.mean()) if len(s) else None})
    return curves, countries, selected_primary


def r1_paired_selected_bootstrap(scores, candidate, baseline, blocks, repetitions=2000):
    scores = np.asarray(scores, float)
    candidate, baseline = np.asarray(candidate, bool), np.asarray(baseline, bool)
    _, inverse = np.unique(np.asarray(blocks).astype(str), return_inverse=True)
    ca = np.bincount(inverse, weights=candidate)
    cb = np.bincount(inverse, weights=baseline)
    sa = np.bincount(inverse, weights=scores*candidate)
    sb = np.bincount(inverse, weights=scores*baseline)
    rng = np.random.default_rng(20261113)
    gains = []
    for _ in range(repetitions):
        draw = rng.integers(len(ca), size=len(ca))
        if ca[draw].sum() and cb[draw].sum():
            gains.append(sa[draw].sum()/ca[draw].sum()-sb[draw].sum()/cb[draw].sum())
    return {'gain': float(scores[candidate].mean()-scores[baseline].mean()),
        'ci95': np.quantile(gains, [.025, .975]).tolist(), 'clusters': len(ca),
        'valid_repetitions': len(gains), 'repetitions': repetitions,
        'interpretation': 'conditional descriptive spatial interval on historically consumed data'}
