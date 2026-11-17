"""Exercise checkpoint reuse and normalization reconstruction without training."""
import ast
import json
from pathlib import Path

import numpy as np
import torch

from test_v31_notebook import sensors

ROOT = Path(__file__).resolve().parents[1]


def test_saved_checkpoint_path_reproduces_without_any_training(sensors, tmp_path):
    nb = json.loads((ROOT/'notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb').read_text())
    ns = {}
    exec(compile(''.join(nb['cells'][1]['source']), 'frozen-dependencies', 'exec'), ns)
    ns['STUDY_MODE'] = 'full'
    ns['FOLLOWUP'] = json.loads((ROOT/'configs/reliability_followup_d1_d3.json').read_text())
    for name in ['src/geolifeclef/reliability.py', 'src/geolifeclef/decision_reliability.py']:
        exec(compile((ROOT/name).read_text(), name, 'exec'), ns)
    for name in ['scripts/reliability_study_core.py', 'scripts/reliability_inference_core.py']:
        tree = ast.parse((ROOT/name).read_text())
        tree.body = [n for n in tree.body if not isinstance(n, ast.Try)]
        exec(compile(tree, name, 'exec'), ns)
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference attempted neural training')
    ns['v29'].train_candidate = forbidden
    store, rows = sensors
    store.labels = np.pad(store.labels, ((0, 0), (0, 5016-12)))
    store.species_ids = np.arange(5016)
    rows['region'], rows['year'], rows['geoUncertaintyInM'] = 'synthetic', 2020, 1
    rows['areaInM2'] = np.arange(len(rows))+1
    store.rows = rows
    split = {'training': np.arange(16), 'probability_calibration': np.arange(16, 20),
             'risk_calibration': np.arange(20, 26), 'assessment': np.arange(26, 32)}
    source, temporary, export = [tmp_path/n for n in ['source', 'runtime', 'export']]
    for p in [source, temporary, export]:
        p.mkdir()
    v29, legacy = ns['v29'], ns['legacy']
    stats = v29.fit_normalization(store, split['training'])
    records, probabilities, sets = [], [], []
    take = np.arange(16, 32)
    for seed in [1, 2]:
        config = {'seed': seed, 'kind': 'attention', 'width': 16, 'geo': True, 'id': f'synthetic{seed}'}
        model = v29.create_model(store, config, split['training'])
        path = source/f'fixed_r1_{config["id"]}.pt'
        torch.save(model.state_dict(), path)
        calibration = {'slope': 1., 'intercept': 0.}
        raw = v29.predict(model, store, take, stats, torch.device('cpu'), config)
        p = v29.calibrated(raw, calibration)
        probabilities.append(p)
        sets.append(legacy.top_rank(p[4:], 18)[0])
        records.append({'config': config, 'probability_calibration': calibration,
                        'checkpoint_sha256': legacy.sha256_file(path)})
    (source/'fold0_report.json').write_text(json.dumps({'training': records}))
    np.savez(source/'fold0_predictions.npz', survey_id=store.train_ids[20:32],
             member_top18_columns=np.stack(sets, axis=1))
    mean = sum(probabilities)/2
    frequency = legacy._frequency(store.labels, split['training'])
    original = ns['decision_frame'](mean[10:], store.labels[26:32], rows.iloc[26:32], frequency, 'top18')
    original.to_csv(source/'fold0_assessment.csv', index=False)
    result = ns['infer_saved_fold'](0, split, store, source, temporary, export,
                                   ns['R1Guard'](.75), torch.device('cpu'))
    assert result['passed'] and result['survey_F1_match_fraction'] == 1
    assert all(r['original_top18_set_match_fraction'] == 1 for r in result['member_sets'])
    assert (export/'fold0_normalization.json').exists()
    with np.load(export/'fold0_mean_probability.npz', allow_pickle=False) as a:
        assert a['mean_probability'].dtype == np.float32
        np.testing.assert_array_equal(a['mean_probability'], mean[4:])
