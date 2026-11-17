"""Training-only metadata count heads with archived, frozen neural rankings.

An exploratory mechanism check on reused v30 calibration views, not independent
assessment. No neural network training, new Kaggle score, or test-label access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from geolifeclef.research_audit import prefix_f1
from scripts.v30_notebook_core import make_splits

ROOT = Path(__file__).resolve().parents[1]


def features(frame, geo):
    area = np.log1p(pd.to_numeric(frame.areaInM2, errors='coerce').clip(lower=0))
    uncertainty = np.log1p(pd.to_numeric(frame.geoUncertaintyInM, errors='coerce').clip(lower=0))
    values = [area.to_numpy(), pd.to_numeric(frame.year, errors='coerce').to_numpy(), uncertainty.to_numpy()]
    if geo:
        values.extend([frame.lat.to_numpy(), frame.lon.to_numpy()])
    return np.column_stack(values).astype(float)


def main():
    started = time.monotonic()
    output = ROOT/'results/research_audit_2026_11_13'
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT/'data/raw/GLC25_PA_metadata_train.csv')
    rows = raw.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    counts = raw[['surveyId', 'speciesId']].drop_duplicates().groupby('surveyId').size().reindex(rows.surveyId).to_numpy()
    splits, manifests = make_splits(rows)
    path = ROOT/'results/v30_kaggle_output/calibration_top64_v30.npz'
    archive = np.load(path, allow_pickle=False)
    offsets = archive['true_offsets']
    truths = [archive['true_species_columns'][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    scores = prefix_f1(archive['ranked_species_columns'], truths)
    m = np.diff(offsets)
    row_index = pd.Index(rows.surveyId).get_indexer(archive['survey_id'])
    if (row_index < 0).any() or not np.array_equal(counts[row_index], m):
        raise ValueError('Archived survey truths disagree with official training metadata')
    count_predictions, fit_records = {}, []
    for geo in [False, True]:
        prediction = np.zeros(len(row_index), float)
        for fold, split in enumerate(splits):
            views = np.flatnonzero(archive['fold'] == fold)
            take = row_index[views]
            if not np.isin(take, split['calibration']).all() or np.intersect1d(take, split['training']).size:
                raise ValueError('Count-model training overlaps archived calibration')
            model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                min_samples_leaf=50, l2_regularization=5, learning_rate=.05,
                early_stopping=False, random_state=20261113)
            begin = time.monotonic()
            model.fit(features(rows.iloc[split['training']], geo), np.log1p(counts[split['training']]))
            prediction[views] = np.expm1(model.predict(features(rows.iloc[take], geo)))
            fit_records.append({'geo': geo, 'fold': fold, 'training_surveys': len(split['training']),
                'training_id_sha256': manifests[fold]['id_hashes']['training'],
                'training_calibration_overlap': 0, 'seconds': time.monotonic()-begin})
        count_predictions['metadata_geo' if geo else 'metadata_ecology'] = np.clip(np.rint(prediction), 1, 64).astype(int)
    records = []
    for expert, name in enumerate(archive['model_ids']):
        p = archive['calibrated_probability'][:, expert].astype(float)
        mass = archive['total_probability_mass'][:, expert].astype(float)
        k = np.arange(1, 65)
        expected_surrogate = 2*p.cumsum(1)/(k+mass[:, None])
        policies = {'top18': np.full(len(m), 18), 'top24': np.full(len(m), 24),
            'expected_F1_surrogate_8_40': expected_surrogate[:, 7:40].argmax(1)+8,
            'expected_F1_surrogate_1_64': expected_surrogate.argmax(1)+1,
            **count_predictions}
        for policy, sizes in policies.items():
            selected = scores[np.arange(len(m)), expert, sizes-1]
            table = pd.DataFrame({'id': archive['survey_id'], 'country': archive['country'],
                'f1': selected, 'count_mae': abs(sizes-m), 'predicted_size': sizes,
                'true_size': m, 'f1_vs_top18': selected-scores[:, expert, 17]})
            for group, mask in [('all', np.ones(len(table), bool)),
                ('core', table.country.isin(['Denmark', 'Netherlands'])),
                ('outside_core', ~table.country.isin(['Denmark', 'Netherlands']))]:
                unique = table.loc[mask].groupby('id').mean(numeric_only=True)
                records.append({'expert': str(name), 'policy': policy, 'group': group,
                    'surveys': len(unique), **unique.mean().to_dict()})
    frame = pd.DataFrame(records)
    frame.to_csv(output/'cardinality_pilot.csv', index=False)
    report = {'status': 'completed_local_CPU_pilot', 'independent_assessment': False,
        'data': 'reused v30 calibration; unique-survey means across overlapping fold views',
        'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'views': len(m), 'unique_surveys': len(np.unique(archive['survey_id'])),
        'new_model_fits': len(fit_records), 'neural_fits': 0, 'fit_records': fit_records,
        'design': {'count_target': 'squared error on log1p observed survey richness',
            'metadata_ecology': ['log1p areaInM2', 'year', 'log1p geoUncertaintyInM'],
            'metadata_geo_adds': ['latitude', 'longitude'], 'seed': 20261113,
            'hyperparameter_search': False, 'selection_of_winner_on_this_data': False,
            'ranking_frozen': True, 'surrogate_is_exact_expected_F1': False,
            'counts_clipped_to_available_top64': True},
        'seconds': time.monotonic()-started, 'results': records,
        'limitations': ['Retrospective calibration diagnostics with historically consumed labels.',
            'Feature importances and associations do not establish ecological mechanisms.',
            'Top64 export and half-precision probabilities limit counterfactual decoding.',
            'Previously calibrated neural scores are not newly calibrated here.',
            'Count MAE and sample F1 optimize different decisions; no accuracy improvement is presumed.']}
    (output/'cardinality_pilot.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(frame.loc[(frame.group == 'all') & (frame.expert == 'temporal_attention'),
        ['policy', 'f1', 'count_mae', 'predicted_size']].to_string(index=False))
    print(json.dumps({'seconds': report['seconds'], 'new_model_fits': report['new_model_fits']}))


if __name__ == '__main__':
    main()
