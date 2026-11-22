"""Outcome-separated calibration, decision features and utility for E1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

META_FEATURES = ['log_area', 'year', 'log_geo_uncertainty', 'log_calibration_distance', 'eligible_labels']
RISK_FEATURES = META_FEATURES+['probability_mass', 'max_probability', 'mean_set_confidence',
    'SDC', 'predicted_richness', 'predicted_rare_mass', 'empty_probability']


def fit_calibration(probability, targets, eligible, bounds):
    p, y = np.asarray(probability, float), np.asarray(targets)
    if p.shape != y.shape or p.shape != eligible.shape or np.any(y[eligible] < 0):
        raise ValueError('Calibration target/mask mismatch')
    z = np.log(np.clip(p[eligible], 1e-7, 1-1e-7)/(1-np.clip(p[eligible], 1e-7, 1-1e-7)))
    truth = y[eligible].astype(float)
    edges = np.linspace(-16, 16, 513)
    bucket = np.clip(np.searchsorted(edges, z)-1, 0, 511)
    count = np.bincount(bucket, minlength=512).astype(float)
    positive = np.bincount(bucket, weights=truth, minlength=512)
    mean = np.bincount(bucket, weights=z, minlength=512)/np.maximum(count, 1)
    def objective(theta):
        logits = theta[0]*mean+theta[1]
        residual = count*expit(logits)-positive
        loss = np.sum(count*np.logaddexp(0, logits)-positive*logits)/count.sum()
        gradient = np.array([np.dot(residual, mean), residual.sum()])/count.sum()
        return loss, gradient
    fit = minimize(objective, [1., 0.], jac=True, method='L-BFGS-B',
                   bounds=[tuple(bounds['slope']), tuple(bounds['intercept'])], options={'maxiter': 300})
    if not fit.success:
        raise RuntimeError('Probability calibration failed: '+str(fit.message))
    return {'slope': float(fit.x[0]), 'intercept': float(fit.x[1]),
            'eligible_pairs': int(eligible.sum()), 'fit_rows': len(p), 'binned_nll': float(fit.fun)}


def apply_calibration(probability, calibration, eligible):
    p = np.asarray(probability, np.float32)
    if not np.isfinite(p[eligible]).all() or np.any((p[eligible] < 0) | (p[eligible] > 1)):
        raise ValueError('Invalid eligible model scores')
    clean = np.clip(p, 1e-7, 1-1e-7)
    result = expit(calibration['slope']*np.log(clean/(1-clean))+calibration['intercept']).astype(np.float32)
    return np.where(eligible, result, 0)


def prediction_context(probability, eligible, maximum=80):
    p = np.asarray(probability, np.float32)
    rank = np.argsort(-p, axis=1, kind='stable')[:, :min(maximum, p.shape[1])]
    values = np.take_along_axis(p, rank, axis=1)
    mass = p.sum(1)
    empty = np.exp(np.log1p(-np.clip(p, 0, 1-1e-7)).sum(1))
    return {'rank': rank, 'values': values, 'mass': mass, 'empty': empty,
            'eligible_count': eligible.sum(1), 'probability': p}


def decode_counts(context, decoder, threshold=None):
    values, mass = context['values'], context['mass']
    allowed = np.minimum(context['eligible_count'], values.shape[1])
    if decoder == 'top18':
        return np.minimum(allowed, 18).astype(int)
    if decoder == 'threshold_0_80':
        if threshold is None:
            raise ValueError('Threshold must be chosen on calibration data')
        return np.minimum((values >= threshold).sum(1), allowed).astype(int)
    if decoder not in ['expected_F1_8_40', 'expected_F1_0_80']:
        raise ValueError('Unknown decoder')
    sizes = np.arange(1, values.shape[1]+1)
    score = 2*np.cumsum(values, axis=1)/(sizes[None, :]+mass[:, None]+1e-12)
    valid = sizes[None, :] <= allowed[:, None]
    if decoder == 'expected_F1_8_40':
        valid &= (sizes[None, :] >= np.minimum(8, allowed)[:, None]) & (sizes[None, :] <= 40)
    score = np.where(valid, score, -1)
    count = score.argmax(1)+1
    if decoder == 'expected_F1_0_80':
        count[context['empty'] >= score.max(1)] = 0
    count[allowed == 0] = 0
    return count.astype(int)


def decision_features(context, counts, metadata, rare):
    values, rank = context['values'], context['rank']
    chosen = np.arange(values.shape[1])[None, :] < counts[:, None]
    positive_mass = (values*chosen).sum(1)
    empty = counts == 0
    out = metadata.reset_index(drop=True).copy()
    out['eligible_labels'] = context['eligible_count']
    out['probability_mass'] = context['mass']
    out['max_probability'] = values.max(1)
    out['empty_probability'] = context['empty']
    out['predicted_richness'] = counts
    out['mean_set_confidence'] = positive_mass/np.maximum(counts, 1)
    out.loc[empty, 'mean_set_confidence'] = context['empty'][empty]
    out['SDC'] = 2*positive_mass/(counts+context['mass']+1e-12)
    out.loc[empty, 'SDC'] = context['empty'][empty]
    clipped = np.clip(values, 1e-7, 1-1e-7)
    entropy = -clipped*np.log(clipped)-(1-clipped)*np.log1p(-clipped)
    out['set_entropy'] = (entropy*chosen).sum(1)/np.maximum(counts, 1)
    p0 = np.clip(context['empty'][empty], 1e-7, 1-1e-7)
    out.loc[empty, 'set_entropy'] = -p0*np.log(p0)-(1-p0)*np.log1p(-p0)
    out['predicted_rare_mass'] = context['probability'][:, rare].sum(1)
    return out


def decision_outcomes(targets, eligible, ranks, counts, rare):
    y = np.asarray(targets)
    if y.shape != eligible.shape or np.any((y > 0) & ~eligible) or np.any(y[eligible] < 0):
        raise ValueError('Published native-range target alignment failed')
    positive = y > 0
    selected = np.arange(ranks.shape[1])[None, :] < counts[:, None]
    hit = np.take_along_axis(positive, ranks, axis=1)*selected
    truth = positive.sum(1)
    denominator = counts+truth
    f1zero = 2*hit.sum(1)/np.maximum(denominator, 1)
    f1 = np.where(denominator == 0, 1., f1zero)
    return pd.DataFrame({'sample_F1': f1, 'sample_F1_zero_empty': f1zero, 'loss': 1-f1,
        'true_richness': truth, 'hits': hit.sum(1), 'rare_truth': positive[:, rare].sum(1),
        'rare_hits': (hit*rare[ranks]).sum(1), 'rare_predictions': (selected*rare[ranks]).sum(1)})


def choose_threshold(context, targets, eligible, rare, grid):
    results = []
    for threshold in grid:
        counts = decode_counts(context, 'threshold_0_80', threshold)
        outcome = decision_outcomes(targets, eligible, context['rank'], counts, rare)
        results.append({'threshold': threshold, 'sample_F1': float(outcome.sample_F1.mean())})
    chosen = results[int(np.argmax([r['sample_F1'] for r in results]))]['threshold']
    return chosen, results


def guarded_target_rows(dataset, cohort, role, phase, assessment_admitted=False):
    """Restrict materialized target values to the declared analysis role."""
    if role not in ['probability_calibration', 'risk_calibration', 'assessment']:
        raise ValueError('Unknown role')
    if role == 'assessment' and (phase != 'assess' or not assessment_admitted):
        raise PermissionError('Assessment targets remain closed until every policy is frozen')
    if phase == 'assess' and role != 'assessment':
        raise PermissionError('Assessment phase cannot refit calibration policies')
    rows = cohort.loc[cohort.role.eq(role), 'prediction_row'].to_numpy(int)
    return read_rows(dataset, rows)


def read_rows(dataset, rows):
    """Efficient row gathering; never materialize unrequested row values."""
    rows = np.asarray(rows, int)
    chunk_rows = dataset.chunks[0] if getattr(dataset, 'chunks', None) else 331
    result = np.empty((len(rows), *dataset.shape[1:]), dataset.dtype)
    for block in np.unique(rows//chunk_rows):
        positions = np.flatnonzero(rows//chunk_rows == block)
        order = np.argsort(rows[positions])
        result[positions[order]] = dataset[rows[positions[order]]]
    return result


def joint_ecological_comparison(scores, rare_truth, candidate, baseline, blocks,
                               repetitions=2000, confidence=.975):
    """Average fixed-model contrasts, sharing spatial resamples across models."""
    scores = np.asarray(scores, float)
    a, b = np.asarray(candidate, bool), np.asarray(baseline, bool)
    rare = np.broadcast_to(np.asarray(rare_truth, float), scores.shape)
    if scores.ndim != 2 or a.shape != scores.shape or b.shape != scores.shape:
        raise ValueError('Models by shared cells required')
    _, inverse = np.unique(np.asarray(blocks).astype(str), return_inverse=True)
    n = int(inverse.max()+1)
    columns = [a, b, scores*a, scores*b, rare*a, rare*b, rare]
    sums = np.array([[np.bincount(inverse, weights=v, minlength=n) for v in col] for col in columns])
    draws = np.random.default_rng(20261116).integers(n, size=(repetitions, n))
    boot = sums[:, :, draws].sum(3)
    valid = ((boot[0] > 0) & (boot[1] > 0) & (boot[6] > 0)).all(0)
    selected_boot = boot[:, :, valid]
    df = (selected_boot[2]/selected_boot[0]-selected_boot[3]/selected_boot[1]).mean(0)
    dr = ((selected_boot[4]-selected_boot[5])/selected_boot[6]).mean(0)
    tail = (1-confidence)/2
    point_f = [(s[x].mean()-s[y].mean()) for s, x, y in zip(scores, a, b)]
    point_r = [(r[x].sum()-r[y].sum())/r.sum() if r.sum() else np.nan for r, x, y in zip(rare, a, b)]
    return {'F1_gain': float(np.mean(point_f)), 'rare_coverage_gain': float(np.mean(point_r)),
        'F1_interval': np.quantile(df, [tail, 1-tail]).tolist() if valid.any() else None,
        'rare_coverage_interval': np.quantile(dr, [tail, 1-tail]).tolist() if valid.any() else None,
        'confidence': confidence, 'clusters': n, 'models': scores.shape[0],
        'valid_draws': int(valid.sum()), 'model_F1_gains': point_f,
        'model_rare_coverage_gains': point_r,
        'interpretation': 'fixed model average, shared spatial block resampling; not independent model trials'}
