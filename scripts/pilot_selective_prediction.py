"""Exploratory selective prediction for the cached matched-v32 recipe.

Fit error predictors on calibration views only. Evaluate their rankings on the
saved spatial assessment. All these labels were historically used in development;
the fit/evaluation separation here does not make them independent new evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import BallTree

from scripts.v30_notebook_core import make_splits

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261113


def metadata_features(rows, distance):
    area = np.log1p(pd.to_numeric(rows.areaInM2, errors='coerce').clip(lower=0))
    uncertainty = np.log1p(pd.to_numeric(rows.geoUncertaintyInM, errors='coerce').clip(lower=0))
    return np.column_stack([area, rows.year, uncertainty, np.log1p(distance)])


def main():
    started = time.monotonic()
    output = ROOT/'results/research_audit_2026_11_13'
    rows = pd.read_csv(ROOT/'data/raw/GLC25_PA_metadata_train.csv').drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    splits, manifests = make_splits(rows)
    folder = ROOT/'results/v34_kaggle_output'
    manifest = json.loads((folder/'v34_manifest.json').read_text())
    if [m['id_hashes'] for m in manifests] != [m['id_hashes'] for m in manifest['splits']]:
        raise ValueError('Spatial split identity changed')
    archive = np.load(folder/'calibration_top128_v34.npz', allow_pickle=False)
    cached = pd.read_csv(folder/'regression_per_survey_v34.csv')
    coords = np.deg2rad(rows[['lat', 'lon']].to_numpy())
    bundles, fits = [], []
    for fold, split in enumerate(splits):
        take = np.flatnonzero(archive['fold'] == fold)
        cal_index = pd.Index(rows.surveyId).get_indexer(archive['survey_id'][take])
        if not np.isin(cal_index, split['calibration']).all():
            raise ValueError('Archive calibration role mismatch')
        if np.intersect1d(cal_index, split['assessment']).size:
            raise ValueError('Risk calibration overlaps assessment')
        baseline_f1 = []
        for i in take:
            a, b = archive['true_offsets'][i:i+2]
            truth = set(archive['true_species_columns'][a:b].astype(int))
            predicted = archive['reference_columns_padded65535'][i]
            predicted = set(predicted[predicted != 65535].astype(int))
            baseline_f1.append(2*len(predicted & truth)/(len(predicted)+len(truth)))
        assessment = cached[cached.fold == fold].copy()
        assessment_index = pd.Index(rows.surveyId).get_indexer(assessment.surveyId)
        if set(assessment_index) != set(split['assessment']):
            raise ValueError('Assessment coverage mismatch')
        tree = BallTree(coords[split['training']], metric='haversine')
        cal_distance = tree.query(coords[cal_index])[0][:, 0]*6371.0088
        assessment_distance = tree.query(coords[assessment_index])[0][:, 0]*6371.0088
        x_cal = metadata_features(rows.iloc[cal_index], cal_distance)
        x_assessment = metadata_features(rows.iloc[assessment_index], assessment_distance)
        for name, columns in [('area_only', [0]), ('metadata_only', [0, 1, 2]),
                              ('metadata_distance', [0, 1, 2, 3])]:
            model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7,
                min_samples_leaf=50, l2_regularization=5, learning_rate=.05,
                early_stopping=False, random_state=SEED)
            model.fit(x_cal[:, columns], 1-np.asarray(baseline_f1))
            assessment[f'risk_{name}'] = model.predict(x_assessment[:, columns])
        assessment['risk_distance_only'] = assessment_distance
        assessment['risk_random'] = [int(hashlib.sha256(f'{SEED}:{i}'.encode()).hexdigest()[:12], 16) for i in assessment.surveyId]
        assessment['loss'] = 1-assessment.matched_v32_f1
        bundles.append(assessment)
        fits.append({'fold': fold, 'calibration_views': len(take),
            'calibration_assessment_overlap': 0, 'split_id_hashes': manifests[fold]['id_hashes']})
    frame = pd.concat(bundles).sort_values('surveyId').reset_index(drop=True)
    if frame.surveyId.duplicated().any():
        raise ValueError('Repeated assessment IDs')
    records, countries, areas = [], [], []
    methods = ['random', 'distance_only', 'area_only', 'metadata_only', 'metadata_distance',
               'distance_only_country_balanced', 'metadata_distance_country_balanced']
    for method in methods:
        # Label-independent hashed tie-breaking; assessment labels never select order.
        source = method.removesuffix('_country_balanced')
        order = np.lexsort((frame.risk_random.to_numpy(), frame[f'risk_{source}'].to_numpy()))
        if method.endswith('_country_balanced'):
            # A deliberately simple coverage baseline, not a new risk guarantee:
            # rank within country, then interleave equally by country percentile.
            ordered = frame.iloc[order]
            percentiles = (ordered.groupby('country').cumcount()+.5)/ordered.groupby('country').country.transform('size')
            order = order[np.lexsort((ordered.risk_random.to_numpy(), percentiles.to_numpy()))]
        losses = frame.loss.to_numpy()[order]
        areas.append({'method': method, 'AURC': float(np.mean(losses.cumsum()/np.arange(1, len(losses)+1)))})
        for coverage in [1., .9, .8, .6, .4, .2]:
            selected = frame.iloc[order[:int(np.ceil(coverage*len(frame)))]]
            for group, mask in [('all', np.ones(len(selected), bool)),
                ('core', selected.country.isin(['Denmark', 'Netherlands'])),
                ('outside_core', ~selected.country.isin(['Denmark', 'Netherlands']))]:
                s = selected.loc[mask]
                records.append({'method': method, 'overall_retained_fraction': coverage,
                    'group': group, 'surveys': len(s), 'mean_loss_1_minus_F1': float(s.loss.mean()),
                    'mean_sample_F1': float(1-s.loss.mean())})
            for country, full in frame.groupby('country'):
                s = selected[selected.country == country]
                countries.append({'method': method, 'overall_retained_fraction': coverage,
                    'country': country, 'available_surveys': len(full), 'retained_surveys': len(s),
                    'country_retained_fraction': len(s)/len(full),
                    'mean_sample_F1': float(1-s.loss.mean()) if len(s) else None})
    pd.DataFrame(records).to_csv(output/'selective_pilot.csv', index=False)
    pd.DataFrame(countries).to_csv(output/'selective_country_coverage.csv', index=False)
    report = {'status': 'completed_local_CPU_pilot', 'fresh_assessment': False,
        'scope': 'matched-v32 recipe refit from v34; reused assessment, not production or test predictions',
        'loss_definition': '1 minus per-survey sample F1', 'assessment_surveys': len(frame),
        'risk_model_fits': 6, 'seed': SEED, 'fit_records': fits,
        'features': ['log1p plot area', 'year', 'log1p geographic uncertainty', 'log1p distance to same-fold PA training'],
        'test_labels_used': False, 'neural_training': False, 'hyperparameter_search': False,
        'area_under_risk_coverage': areas, 'results': records,
        'seconds': time.monotonic()-started,
        'limitations': ['Retrospective data have influenced past method development.',
            'Risk-coverage ordering uses assessment inputs, not assessment labels; it is not a frozen deployment threshold.',
            'Pooled risk ranking may benefit from geography and survey protocol confounding.',
            'Country-balanced ordering is a simple within-country percentile baseline with finite-sample rounding, not a novel algorithm or calibrated risk guarantee.',
            'Selective F1 is not full-coverage Kaggle F1; country coverage must be inspected.',
            'No distribution-free risk or conformal guarantee is established.']}
    (output/'selective_pilot.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(pd.DataFrame(records).query("group == 'all' and overall_retained_fraction in [1., .8, .4]").to_string(index=False))
    print(json.dumps({'AURC': areas, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
