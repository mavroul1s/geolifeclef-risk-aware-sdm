import hashlib
import json

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import v33_notebook_core as core
from scripts.build_v33_notebook import ROOT, OUTPUT
from test_v31_notebook import sensors


def test_new_convolution_specialist_actual_width_full_vocabulary_and_backward(sensors):
    store, _ = sensors
    store.labels = np.pad(store.labels, ((0, 0), (0, 5016-12)))
    store.species_ids = np.arange(5016)
    config = core.CONV_CONFIGS[0]
    model, _ = core.previous.make_specialist(store, config, np.arange(16))
    assert isinstance(model, core.v29.SensorConv)
    # Exercise residual branch even though tiny fixture prevalence cannot qualify.
    model.head = core.previous.RareResidualHead(model.head, np.array([2, 3]))
    stats = core.v29.fit_normalization(store, np.arange(16))
    x = core.v29.batch_inputs(store, np.arange(2), stats, torch.device('cpu'))
    logits, richness = model(x, True)
    assert logits.shape == (2, 5016) and richness.shape == (2,)
    loss = core.previous.asymmetric_loss(logits, torch.as_tensor(store.labels[:2], dtype=torch.float32),
        torch.ones(5016), config['negative_clip'], config['negative_gamma'])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.rare[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_25_policies_preserve_counts_and_prefix():
    base = [list(range(n)) for n in (8, 18, 40)]
    expert = {'rank': np.tile(np.arange(200, 72, -1), (3, 1)), 'value': np.ones((3, 128))}
    for policy in core.POLICIES:
        core.v31.validate_residual(base, core.decode(base, expert, expert, policy), policy)
    assert core.decode(base, None, None, core.POLICIES[0]) == base
    assert core.self_tests()['passed']


def test_selection_uses_only_calibration_and_rejects_outside_core_loss():
    rows = pd.DataFrame({'surveyId': np.arange(60), 'country': ['Denmark']*30+['France']*30})
    old = np.full(60, .2)
    bundles = [{'split': {'calibration': np.arange(60)}, 'data': {'assessment': 'DO NOT READ'},
                'trials': {p['id']: old.copy() for p in core.POLICIES}} for _ in range(2)]
    p = core.POLICIES[1]
    for b in bundles:
        b['trials'][p['id']] += np.r_[np.full(30, .1), np.full(30, -.01)]
    assert core.select_policy(bundles, rows)[0]['id'] == 'control'
    for b in bundles:
        b['trials'][p['id']] = old+.01
    assert core.select_policy(bundles, rows)[0] == p


def test_runtime_estimate_accounts_for_complete_fold_wall_time():
    records = [{'group': g, 'member': i, 'training_surveys': 100, 'history': [{'seconds': 10}]}
               for g, cs in core.groups() for i in range(len(cs))]
    b = {'records': records, 'wall_seconds': 5000, 'split': {'training': np.arange(100)}}
    estimate = sum(c['epochs'] for _, cs in core.groups() for c in cs)*20*1.5+2700
    assert core.production_estimate([b], {'attention': 1, 'conv': 1}, 200) == estimate
    assert core.production_estimate([b], {'attention': 0, 'conv': 0}, 200) == 2700
    assert core.remaining_plan_estimate([b], 110, 200) == 5000*1.1*1.25+estimate+600


def test_standalone_hashes_exact_best_csv_and_frozen_policy(tmp_path):
    notebook = nbformat.read(OUTPUT, as_version=4)
    nbformat.validate(notebook)
    assert OUTPUT.stat().st_size < 1_000_000
    space = {}
    for cell in notebook.cells[1:3]:
        assert not cell.outputs and cell.execution_count is None
        exec(compile(cell.source, 'standalone-v33', 'exec'), space)
    assert space['self_tests']()['passed']
    assert space['previous'] is not core.previous
    assert space['V33_SOURCE_HASH'] == hashlib.sha256((ROOT/'scripts/v33_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest()
    for name, digest in space['FROZEN_SOURCE_HASHES'].items():
        assert hashlib.sha256((ROOT/f'scripts/{name}_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest() == digest
    control = pd.read_csv(ROOT/'results/v32_kaggle_output/GLC25_PA_submission_v32.csv')
    species = np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    prediction = space['decode_control'](space['CONTROL_B64'], control, species)
    proof = space['legacy'].write_submission(tmp_path/'roundtrip.csv', control, control.surveyId.to_numpy(), prediction, species)
    assert proof['sha256'] == core.CONTROL_HASH
    with pytest.raises(ValueError, match='payload mismatch'):
        space['decode_control'](space['CONTROL_B64'][:-4]+'AAAA', control, species)
    report = json.loads((ROOT/'results/v32_kaggle_output/v32_report.json').read_text())
    assert core.FROZEN_V32_POLICY == report['policy']
    manifest = json.loads((ROOT/'results/v32_kaggle_output/v32_manifest.json').read_text())
    for name, digest in manifest['outputs'].items():
        assert hashlib.sha256((ROOT/'results/v32_kaggle_output'/name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('fault', ['none', 'gate', 'budget', 'export', 'late'])
def test_real_pipeline_reference_new_models_production_and_failure_contracts(sensors, tmp_path, monkeypatch, fault):
    store, rows = sensors
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core.os, 'cpu_count', lambda: 2)
    store.labels = np.pad(store.labels, ((0, 0), (0, 5016-12)))
    store.species_ids = np.arange(5016)
    store.cache = tmp_path/'sensors'; store.cache.mkdir()
    np.save(store.cache/'train_sentinel64.npy', store.high_train)
    np.save(store.cache/'test_sentinel64.npy', store.high_test)
    data = tmp_path/'official'; data.mkdir()
    rows.to_csv(data/'GLC25_PA_metadata_train.csv', index=False)
    template = pd.DataFrame({'surveyId': rows.surveyId, 'predictions': ['']*32})
    template.to_csv(data/'GLC25_SAMPLE_SUBMISSION.csv', index=False)
    monkeypatch.setattr(core.legacy, 'EXPECTED_TEST_ROWS', 32)
    base = [list(range(100, 108)) for _ in range(32)]
    proof = core.legacy.write_submission(tmp_path/'control.csv', template, store.test_ids, base, store.species_ids)
    monkeypatch.setattr(core, 'CONTROL_HASH', proof['sha256'])
    splits = [{'training': np.arange(16), 'selection': np.arange(16, 20), 'calibration': np.arange(20, 24),
               'assessment': np.arange(24+4*f, 28+4*f)} for f in range(2)]
    monkeypatch.setattr(core.v31.previous, 'make_splits', lambda *a, **k: (splits, [{'minimum_distance_km': 21.}]*2))
    monkeypatch.setattr(core.legacy, 'require_gpu', lambda: torch.device('cpu'))
    monkeypatch.setattr(core.legacy, 'discover_data_root', lambda: data)
    monkeypatch.setattr(core, 'decode_control', lambda *a: base)
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a, **k: {})
    monkeypatch.setattr(core.legacy, 'FeatureStore', lambda p: store)
    monkeypatch.setattr(core.legacy, 'load_rows_and_pairs', lambda *a: (rows, rows, None))
    for module, field in ((core, 'ATTENTION_CONFIGS'), (core, 'CONV_CONFIGS'),
                          (core.previous, 'BAG_CONFIGS'), (core.previous, 'SPECIALISTS'), (core.v31, 'CONFIGS')):
        monkeypatch.setattr(module, field, tuple({**c, 'width': 16, 'epochs': 1} for c in getattr(module, field)))
    monkeypatch.setattr(core.previous, 'BAG_EPOCHS', (1, 1, 1))
    monkeypatch.setattr(core.v31, 'EPOCHS', (1, 1, 1))
    # Run every frozen reference fit for integration coverage, then control its
    # decoded predictions to make synthetic acceptance deterministic, not accuracy evidence.
    original_fold = core.previous.fit_fold
    def baseline(*a, **k):
        b = original_fold(*a, **k)
        for d in b['data'].values():
            d['base'] = [list(range(100, 108)) for _ in d['base']]
            d['bag']['value'][:] = 0
            d['specialist']['value'][:] = 0
        assert len(b['records']) == 5 and len(b['reference_records']) == 6
        return b
    monkeypatch.setattr(core.previous, 'fit_fold', baseline)
    geographic = core.v31.geographic_gain
    def small_gain(*a, **k):
        result = geographic(*a, **k)
        result['geographic_gain_positive'] = result['gain'] > 0
        return result
    monkeypatch.setattr(core.v31, 'geographic_gain', small_gain)
    policy = {'id': 'smoke', 'attention': 1., 'conv': 1., 'swaps': 2}
    monkeypatch.setattr(core, 'select_policy', lambda *a: (core.POLICIES[0] if fault == 'gate' else policy, []))
    monkeypatch.setattr(core, 'V33_SOURCE_HASH', 'synthetic', raising=False)
    monkeypatch.setattr(core, 'FROZEN_SOURCE_HASHES', {}, raising=False)
    train, production = core.previous.train_specialist, []
    def checked_train(*args, **kwargs):
        if args[-1] == 'production':
            production.append(args[2].copy())
            assert np.array_equal(args[2], np.arange(32))
        else:
            assert np.array_equal(args[2], np.arange(16))
        return train(*args, **kwargs)
    monkeypatch.setattr(core.previous, 'train_specialist', checked_train)
    if fault == 'budget':
        monkeypatch.setattr(core, 'remaining_plan_estimate', lambda *a: 99*3600)
    if fault == 'export':
        def fail(*a, **k):
            raise ValueError('export interrupted')
        monkeypatch.setattr(core, 'save_compact', fail)
    if fault == 'late':
        save = core.legacy.save_json
        def sixth_file(path, content):
            save(path, content)
            if path.name == 'v33_manifest.json':
                save(path.parent/'unexpected_extra.json', {'force': 'late export contract failure'})
        monkeypatch.setattr(core.legacy, 'save_json', sixth_file)
    export = tmp_path/'artifacts/v33_export'
    if fault in ('budget', 'export', 'late'):
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            core.run_v33('')
        assert not (export/'GLC25_PA_submission_v33.csv').exists()
        failure = json.loads((export/'failure_report.json').read_text())
        assert not failure['eligible_for_submission']
        if fault == 'budget':
            assert len(production) == 0 and len(failure['completed_folds']) == 1
        else:
            assert (export/'failed_DO_NOT_SUBMIT.csv').exists()
        if fault == 'late':
            report = json.loads((export/'v33_report.json').read_text())
            assert report['status'] == 'failed' and not report['eligible_for_submission']
            assert not (export/'v33_manifest.json').exists()
            assert (export/'failed_manifest_not_a_valid_export.json').exists()
    else:
        result = core.run_v33('')
        assert result['status'] == 'complete' and result['eligible_for_submission'] == (fault == 'none')
        assert len(production) == (3 if fault == 'none' else 0)
        assert len(list(export.iterdir())) == 5 and result['output_bytes'] < 16_000_000
        report = json.loads((export/'v33_report.json').read_text())
        assert len(report['training']['development']) == 2
        for fold in report['training']['development']:
            assert len(fold['reference_members']) == 11 and len(fold['members']) == 3
            assert len(fold['member_calibration_diagnostics']) == 3
        if fault == 'none':
            assert report['production_diagnostics']['cardinality_equal_to_scored_v32_per_row']
            assert report['production_diagnostics']['all_PA_rows_used']
        else:
            assert hashlib.sha256((export/result['prediction_file']).read_bytes()).hexdigest() == core.CONTROL_HASH
        output = pd.read_csv(export/result['prediction_file'])
        assert all(len(x.split()) == 8 for x in output.predictions)
        with np.load(export/'calibration_top128_v33.npz', allow_pickle=False) as archive:
            assert archive['ranked_species_columns'].shape == (8, 5, 128)
            assert archive['expert_ids'].tolist() == ['attention_ensemble', 'convolution_ensemble']+[
                c['id'] for _, cs in core.groups() for c in cs]
            assert archive['true_offsets'][-1] == len(archive['true_species_columns'])
        manifest = json.loads((export/'v33_manifest.json').read_text())
        for name, digest in manifest['outputs'].items():
            assert hashlib.sha256((export/name).read_bytes()).hexdigest() == digest
    assert not (tmp_path/'artifacts/v33_runtime').exists()


def test_gpu_fail_fast_precedes_feature_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def missing():
        raise RuntimeError('Enable a Kaggle GPU')
    monkeypatch.setattr(core.legacy, 'require_gpu', missing)
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a: pytest.fail('GPU preflight was too late'))
    with pytest.raises(RuntimeError, match='Enable a Kaggle GPU'):
        core.run_v33('')
    assert not (tmp_path/'artifacts/v33_runtime').exists()
