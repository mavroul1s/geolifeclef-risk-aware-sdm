import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from scripts import v32_notebook_core as core
from scripts.build_v32_notebook import ROOT, OUTPUT
from test_v31_notebook import sensors


def test_asymmetric_loss_soft_labels_stability_and_easy_negative_gradients():
    z = torch.tensor([[-100., -7., 0., 100.]], requires_grad=True)
    y = torch.tensor([[0., 0., .3, 1.]])
    loss = core.asymmetric_loss(z, y, torch.ones(4))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad[0, :2].abs().sum() == 0
    z = torch.tensor([[-2., .1, 3.]], requires_grad=True)
    y = torch.tensor([[.2, 1., 0.]])
    weight = torch.tensor([2., 1., 3.])
    assert torch.allclose(core.asymmetric_loss(z, y, weight, clip=0, gamma=0),
                          F.binary_cross_entropy_with_logits(z, y, pos_weight=weight), atol=1e-6)
    assert core.self_tests()['passed']


def test_rare_branch_is_train_only_and_identity_initialized(sensors):
    store, _ = sensors
    store.labels = np.zeros((1200, 12), np.uint8)
    store.labels[:5, 2] = 1
    store.labels[:500, 0] = 1
    store.labels[1000:, 4] = 1  # Query-only taxon must NOT get a rare branch.
    config = {**core.SPECIALISTS[1], 'width': 16}
    model, rare = core.make_specialist(store, config, np.arange(1000))
    assert rare.tolist() == [2]
    assert model.head.rare_columns.tolist() == [2]
    features = torch.randn(4, 32)
    model.eval()
    assert torch.equal(model.head(features), model.head.base(features))
    changed = store.labels.copy()
    store.labels[1000:] = 1
    second, repeated = core.make_specialist(store, config, np.arange(1000))
    assert np.array_equal(rare, repeated)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), second.state_dict().values()))
    output = model.head(features)
    output.sum().backward()
    assert model.head.rare[-1].weight.grad.abs().sum() > 0
    assert np.array_equal(changed[:1000], store.labels[:1000])


def test_new_policy_count_and_protected_prefix_contract():
    base = [list(range(n)) for n in (8, 18, 40)]
    expert = {'rank': np.tile(np.arange(200, 72, -1), (3, 1)), 'value': np.ones((3, 128))}
    for policy in core.POLICIES:
        result = core.decode(base, expert, expert, policy)
        core.previous.validate_residual(base, result, policy)
    assert len(core.POLICIES) == 22
    assert core.decode(base, None, None, core.POLICIES[0]) == base


@pytest.mark.parametrize('config', core.SPECIALISTS, ids=lambda c: c['id'])
def test_production_width_and_full_vocabulary_backward(sensors, config):
    store, _ = sensors
    store.labels = np.pad(store.labels, ((0, 0), (0, 5016-12)))
    store.species_ids = np.arange(5016)
    take = np.arange(16)
    stats = core.v29.fit_normalization(store, take)
    model, _ = core.make_specialist(store, config, take)
    if config['rare_branch']:
        # Force a branch for architecture QA; the train-only membership rule is
        # tested separately with 1,000 rows. Here all raw sensors are real-shaped.
        model.head = core.RareResidualHead(model.head, np.array([2, 3]))
    x = core.v29.batch_inputs(store, take[:2], stats, torch.device('cpu'))
    logits, richness = model(x, config['geo'])
    assert logits.shape == (2, 5016) and richness.shape == (2,)
    loss = core.asymmetric_loss(logits, torch.as_tensor(store.labels[:2], dtype=torch.float32),
                               torch.ones(5016), config['negative_clip'], config['negative_gamma'])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_selection_does_not_read_assessment_and_rejects_geographic_regression():
    n = 60
    rows = pd.DataFrame({'surveyId': np.arange(n), 'country': ['Denmark']*30+['France']*30})
    controls = np.full(n, .2)
    candidate = core.POLICIES[1]
    bundles = [{'number': f, 'split': {'calibration': np.arange(n)},
        'trials': {p['id']: controls.copy() for p in core.POLICIES},
        'data': {'assessment': 'MUST NOT READ'}} for f in range(2)]
    for b in bundles:
        b['trials'][candidate['id']] += np.r_[np.full(30, .1), np.full(30, -.01)]
    assert core.select_policy(bundles, rows)[0]['id'] == 'control'
    for b in bundles:
        b['trials'][candidate['id']] = controls+.01
    assert core.select_policy(bundles, rows)[0] == candidate


def test_production_estimate_includes_every_active_model_and_margin():
    records = []
    for group, configs in (('bag', core.BAG_CONFIGS), ('specialist', core.SPECIALISTS)):
        records.extend({'group': group, 'member': i, 'training_surveys': 100,
                        'history': [{'seconds': 10}]} for i in range(len(configs)))
    bundles = [{'records': records}]*2
    assert core.production_estimate(bundles, {'bag': 0, 'specialist': 0}, 200) == 2700
    expected = (sum(core.BAG_EPOCHS)+sum(c['epochs'] for c in core.SPECIALISTS))*20*1.5+2700
    assert core.production_estimate(bundles, {'bag': 1, 'specialist': 1}, 200) == expected


def test_standalone_notebook_control_roundtrip_and_frozen_source_integrity(tmp_path):
    notebook = nbformat.read(OUTPUT, as_version=4)
    nbformat.validate(notebook)
    assert OUTPUT.stat().st_size < 1_000_000
    space = {}
    for cell in notebook.cells[1:3]:
        assert not cell.outputs and cell.execution_count is None
        exec(compile(cell.source, 'standalone-v32', 'exec'), space)
    assert space['self_tests']()['passed']
    assert space['previous'] is not core.previous
    assert space['V32_SOURCE_HASH'] == hashlib.sha256((ROOT/'scripts/v32_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest()
    for name, digest in space['FROZEN_SOURCE_HASHES'].items():
        assert hashlib.sha256((ROOT/f'scripts/{name}_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest() == digest
    control = pd.read_csv(ROOT/'results/v31_kaggle_output/GLC25_PA_submission_v31.csv')
    species = np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    prediction = space['decode_control'](space['CONTROL_B64'], control, species)
    proof = space['legacy'].write_submission(tmp_path/'roundtrip.csv', control, control.surveyId.to_numpy(), prediction, species)
    assert proof['sha256'] == core.CONTROL_HASH
    assert core.FROZEN_V31_POLICY == json.loads((ROOT/'results/v31_kaggle_output/v31_report.json').read_text())['selected_policy']
    manifest = json.loads((ROOT/'results/v31_kaggle_output/v31_manifest.json').read_text())
    for name, digest in manifest['outputs'].items():
        assert hashlib.sha256((ROOT/'results/v31_kaggle_output'/name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('fault', ['none', 'export', 'gate', 'budget'])
def test_real_miniature_pipeline_five_new_models(sensors, tmp_path, monkeypatch, fault):
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
    monkeypatch.setattr(core.previous.previous, 'make_splits', lambda *a, **k: (splits, [{'minimum_distance_km': 21.}]*2))
    monkeypatch.setattr(core.legacy, 'require_gpu', lambda: torch.device('cpu'))
    monkeypatch.setattr(core.legacy, 'discover_data_root', lambda: data)
    monkeypatch.setattr(core, 'decode_control', lambda *a: base)
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a, **k: {})
    monkeypatch.setattr(core.legacy, 'FeatureStore', lambda p: store)
    monkeypatch.setattr(core.legacy, 'load_rows_and_pairs', lambda *a: (rows, rows, None))
    monkeypatch.setattr(core, 'BAG_CONFIGS', tuple({**c, 'width': 16, 'epochs': 1} for c in core.BAG_CONFIGS))
    monkeypatch.setattr(core, 'BAG_EPOCHS', (1, 1, 1))
    monkeypatch.setattr(core, 'SPECIALISTS', tuple({**c, 'width': 16, 'epochs': 1} for c in core.SPECIALISTS))
    monkeypatch.setattr(core.previous, 'CONFIGS', tuple({**c, 'width': 16, 'epochs': 1} for c in core.previous.CONFIGS))
    monkeypatch.setattr(core.previous, 'EPOCHS', (1, 1, 1))
    monkeypatch.setattr(core.v29, 'decode_ensemble', lambda b, p, policy: [list(range(100, 108)) for _ in p])
    # Baseline pipeline is real; freeze its decoded output wrong to make acceptance
    # deterministic in 32 rows. No synthetic score is reported as model evidence.
    original_fold = core.previous.fit_fold
    def baseline(*a, **k):
        b = original_fold(*a, **k)
        for d in b['data'].values():
            d['neural']['value'][:] = 0
            d['habitat']['value'][:] = 0
        return b
    monkeypatch.setattr(core.previous, 'fit_fold', baseline)
    geographic = core.previous.geographic_gain
    def small_gain(*a, **k):
        result = geographic(*a, **k)
        result['geographic_gain_positive'] = result['gain'] > 0
        return result
    monkeypatch.setattr(core.previous, 'geographic_gain', small_gain)
    policy = {'id': 'smoke', 'bag': 1., 'specialist': 1., 'swaps': 2}
    monkeypatch.setattr(core, 'select_policy', lambda *a: (core.POLICIES[0] if fault == 'gate' else policy, []))
    monkeypatch.setattr(core, 'V32_SOURCE_HASH', 'synthetic', raising=False)
    monkeypatch.setattr(core, 'FROZEN_SOURCE_HASHES', {}, raising=False)
    ordinary, specialist, production = core.v29.train_candidate, core.train_specialist, []
    def checked_ordinary(*args, **kwargs):
        if kwargs.get('phase') == 'production':
            production.append(args[2].copy())
            assert np.array_equal(args[2], np.arange(32)) and len(args[3]) == 0
        return ordinary(*args, **kwargs)
    def checked_specialist(*args, **kwargs):
        if args[-1] == 'production':
            production.append(args[2].copy())
            assert np.array_equal(args[2], np.arange(32))
        return specialist(*args, **kwargs)
    monkeypatch.setattr(core.v29, 'train_candidate', checked_ordinary)
    monkeypatch.setattr(core, 'train_specialist', checked_specialist)
    if fault == 'export':
        def fail(*a, **k):
            raise ValueError('export interrupted')
        monkeypatch.setattr(core, 'save_compact', fail)
    if fault == 'budget':
        monkeypatch.setattr(core, 'production_estimate', lambda *a: 99*3600)
    export = tmp_path/'artifacts/v32_export'
    if fault in ('export', 'budget'):
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            core.run_v32('')
        assert not (export/'GLC25_PA_submission_v32.csv').exists()
        if fault == 'export':
            assert (export/'failed_DO_NOT_SUBMIT.csv').exists()
        else:
            assert len(production) == 0
        assert not json.loads((export/'failure_report.json').read_text())['eligible_for_submission']
    else:
        result = core.run_v32('')
        assert result['status'] == 'complete'
        assert result['eligible_for_submission'] == (fault == 'none')
        assert len(production) == (5 if fault == 'none' else 0)
        assert len(list(export.iterdir())) == 5 and result['output_bytes'] < 16_000_000
        report = json.loads((export/'v32_report.json').read_text())
        if fault == 'none':
            assert report['production_diagnostics']['all_PA_rows_used']
            assert report['production_diagnostics']['cardinality_equal_to_scored_v31_per_row']
            assert all(r['training_surveys'] == 32 for r in report['training']['production'])
        else:
            assert hashlib.sha256((export/result['prediction_file']).read_bytes()).hexdigest() == core.CONTROL_HASH
        output = pd.read_csv(export/result['prediction_file'])
        assert all(len(x.split()) == 8 for x in output.predictions)
        with np.load(export/'calibration_top128_v32.npz', allow_pickle=False) as evidence:
            assert evidence['ranked_species_columns'].shape == (8, 2, 128)
            assert evidence['true_offsets'][-1] == len(evidence['true_species_columns'])
            assert evidence['expert_ids'].tolist() == ['new_seed_bag', 'asymmetric_rare_specialists']
            assert 'habitat_distance' not in evidence.files
        manifest = json.loads((export/'v32_manifest.json').read_text())
        for name, digest in manifest['outputs'].items():
            assert hashlib.sha256((export/name).read_bytes()).hexdigest() == digest
    assert not (tmp_path/'artifacts/v32_runtime').exists()


def test_gpu_preflight_precedes_large_feature_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def gpu_missing():
        raise RuntimeError('Enable a Kaggle GPU')
    monkeypatch.setattr(core.legacy, 'require_gpu', gpu_missing)
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a: pytest.fail('Predictors loaded before GPU check'))
    with pytest.raises(RuntimeError, match='Enable a Kaggle GPU'):
        core.run_v32('')
    assert not (tmp_path/'artifacts/v32_runtime').exists()
