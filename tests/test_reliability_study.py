import ast
import json
from pathlib import Path

import numpy as np
import torch

from test_v31_notebook import sensors

ROOT = Path(__file__).resolve().parents[1]


def test_actual_neural_calibration_uncertainty_export_pipeline(sensors, tmp_path):
    notebook = json.loads((ROOT/'notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb').read_text())
    space = {}
    exec(compile(''.join(notebook['cells'][1]['source']), 'frozen-study-dependencies', 'exec'), space)
    space['STUDY'] = json.loads((ROOT/'configs/reliability_study_r1.json').read_text())
    space['STUDY_MODE'] = 'smoke'
    space['STUDY_CONFIG_SHA256'] = 'synthetic-integration-test'
    exec(compile((ROOT/'src/geolifeclef/reliability.py').read_text(), 'study-helpers', 'exec'), space)
    definitions = ast.parse((ROOT/'scripts/reliability_study_core.py').read_text())
    assert isinstance(definitions.body[-1], ast.Try)
    definitions.body = definitions.body[:-1]
    exec(compile(definitions, 'study-definitions', 'exec'), space)

    store, rows = sensors
    store.labels = np.pad(store.labels, ((0, 0), (0, 5016-12)))
    store.species_ids = np.arange(5016)
    rows['region'], rows['year'], rows['geoUncertaintyInM'] = 'synthetic', 2020, 1
    rows['areaInM2'] = np.arange(len(rows))+1
    store.rows = rows
    split = {'training': np.arange(16), 'probability_calibration': np.arange(16, 20),
             'risk_calibration': np.arange(20, 26), 'assessment': np.arange(26, 32)}
    runtime, export = tmp_path/'runtime', tmp_path/'export'
    runtime.mkdir()
    export.mkdir()
    result, frame = space['r1_fit_fold'](0, split, store, runtime, export, space['R1Guard'](.75),
        {'kind': 'attention', 'id': 'synthetic', 'width': 16, 'epochs': 1, 'geo': True, 'gamma': 0.},
        [1, 2], torch.device('cpu'))
    assert len(frame) == 6 and len(result['training']) == 2
    assert all(r['best_epoch'] == 1 and r['history'][0]['selection_f1'] is None for r in result['training'])
    assert all(r['probability_calibration_surveys'] == 4 for r in result['training'])
    assert all(r['calibration_surveys'] == 6 for r in result['risk_fitting'].values())
    assert set(frame.surveyId) == set(rows.iloc[split['assessment']].surveyId)
    budgets = [int(frame[c].sum()) for c in frame if c.startswith('primary_accept_')]
    assert len(set(budgets)) == 1 and budgets[0] == 5
    with np.load(export/'fold0_predictions.npz', allow_pickle=False) as archive:
        assert archive['mean_probability'].shape == (12, 5016)
        assert archive['member_top18_columns'].shape == (12, 2, 18)
        assert len(archive['true_offsets']) == 13
        assert np.array_equal(archive['survey_id'], rows.iloc[np.arange(20, 32)].surveyId)
    assert np.isfinite(frame.sample_F1).all()
    assert 'macro_f1_all_species' in result['multilabel_metrics']
