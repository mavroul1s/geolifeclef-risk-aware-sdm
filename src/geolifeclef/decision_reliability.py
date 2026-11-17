"""Decision-aligned diagnostics, separate from the frozen R1 experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd


def decision_ranks(probability, maximum=40):
    p = np.asarray(probability, np.float32)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite probability matrix required')
    return np.argsort(-p, axis=1, kind='stable')[:, :min(maximum, p.shape[1])]


def decision_counts(values, mass, decoder, threshold=None, minimum=8):
    values, mass = np.asarray(values), np.asarray(mass)
    maximum = values.shape[1]
    minimum = min(minimum, maximum)
    if decoder == 'top18':
        return np.full(len(values), min(18, maximum), int)
    if decoder == 'calibrated_threshold':
        if threshold is None or not 0 <= threshold <= 1:
            raise ValueError('A calibrated threshold is required')
        return np.clip((values >= threshold).sum(1), minimum, maximum)
    if decoder == 'expected_F1_surrogate':
        sizes = np.arange(1, maximum+1)
        objective = 2*np.cumsum(values, axis=1)/(sizes[None, :]+mass[:, None]+1e-8)
        return objective[:, minimum-1:].argmax(1)+minimum
    raise ValueError('Unknown decoder')


def choose_threshold(values, truth_hits, true_richness, grid, minimum=8):
    """Only probability-calibration outcomes belong in this function."""
    records = []
    for threshold in grid:
        count = decision_counts(values, np.zeros(len(values)), 'calibrated_threshold', threshold, minimum)
        hits = np.cumsum(truth_hits, axis=1)[np.arange(len(count)), count-1]
        score = 2*hits/(count+np.asarray(true_richness))
        records.append({'threshold': threshold, 'sample_F1': float(score.mean())})
    return records[int(np.argmax([r['sample_F1'] for r in records]))]['threshold'], records


def decision_frame(probability, labels, metadata, frequency, decoder, threshold=None):
    """Per-survey outcomes and label-free signals for one frozen decision rule."""
    p, y = np.asarray(probability, np.float32), np.asarray(labels, np.uint8)
    if p.shape != y.shape or len(metadata) != len(p):
        raise ValueError('Prediction, truth and metadata dimensions differ')
    rank = decision_ranks(p)
    values = np.take_along_axis(p, rank, axis=1)
    mass = p.sum(1)
    count = decision_counts(values, mass, decoder, threshold)
    selected = np.arange(rank.shape[1])[None, :] < count[:, None]
    hits = np.take_along_axis(y, rank, axis=1)*selected
    rare = (np.asarray(frequency) >= 1) & (np.asarray(frequency) <= 25)
    out = metadata.reset_index(drop=True).copy()
    out['decoder'] = decoder
    out['predicted_richness'], out['true_richness'] = count, y.sum(1)
    out['hits'] = hits.sum(1)
    out['sample_F1'] = 2*out.hits/(count+out.true_richness)
    out['loss'] = 1-out.sample_F1
    out['rare_truth'] = y[:, rare].sum(1)
    out['rare_hits'] = (hits*rare[rank]).sum(1)
    out['rare_predictions'] = (selected*rare[rank]).sum(1)
    out['probability_mass'], out['max_probability'] = mass, p.max(1)
    out['mean_set_confidence'] = (values*selected).sum(1)/count
    out['expected_F1_surrogate'] = 2*(values*selected).sum(1)/(count+mass)
    top = np.clip(values, 1e-7, 1-1e-7)
    out['set_entropy'] = ((-top*np.log(top)-(1-top)*np.log1p(-top))*selected).sum(1)/count
    out['brier'] = np.square(p-y).mean(1)
    return out


def balanced_country_weights(countries, exponent=.5):
    labels = pd.Series(countries).astype(str)
    size = labels.map(labels.value_counts()).to_numpy(float)
    weights = size**(-exponent)
    return weights/weights.mean()


def grouped_risk_prediction(calibration, evaluation, features, parameters, variant,
                            minimum_local=100, pseudocount=200):
    """Targets are read from calibration only; evaluation requires no outcomes."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    valid = ['global_lean', 'country_balanced_lean', 'country_blend', 'bioregion_blend']
    if variant not in valid:
        raise ValueError('Unknown regional variant')
    x, y = calibration[features].to_numpy(float), calibration.loss.to_numpy(float)
    model = HistGradientBoostingRegressor(**parameters)
    weight = balanced_country_weights(calibration.country) if variant == 'country_balanced_lean' else None
    model.fit(x, y, sample_weight=weight)
    risk = model.predict(evaluation[features].to_numpy(float))
    fitted = []
    if variant.endswith('_blend'):
        column = 'country' if variant == 'country_blend' else 'region'
        for name, group in calibration.groupby(column):
            mask = evaluation[column].astype(str).to_numpy() == str(name)
            if len(group) < minimum_local or not mask.any():
                continue
            local = HistGradientBoostingRegressor(**parameters)
            local.fit(group[features].to_numpy(float), group.loss.to_numpy(float))
            alpha = len(group)/(len(group)+pseudocount)
            risk[mask] = (1-alpha)*risk[mask]+alpha*local.predict(evaluation.loc[mask, features].to_numpy(float))
            fitted.append({'group': str(name), 'calibration_surveys': len(group), 'local_weight': alpha})
    return risk, {'variant': variant, 'calibration_surveys': len(calibration), 'local_models': fitted}
