import hashlib
import json
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import v31_notebook_core as core
from scripts.build_v31_notebook import ROOT, OUTPUT


@pytest.fixture
def sensors():
    torch.set_num_threads(2)
    rng, n = np.random.default_rng(101), 32
    rows = pd.DataFrame({'surveyId': np.arange(n), 'lat': np.linspace(30, 65, n),
                         'lon': np.linspace(-10, 30, n), 'country': ['France', 'Denmark']*16})
    vectors = {k: rng.normal(size=(n, 5)).astype(np.float32) for k in core.legacy.MODALITIES}
    rasters = {k: rng.normal(size=(n, *shape)).astype(np.float16) for k, shape in core.legacy.RASTER_SHAPES.items()}
    labels = (rng.random((n, 12)) < .2).astype(np.uint8)
    labels[:, 0] = 1
    store = SimpleNamespace(train=vectors, test=vectors, raster_train=rasters, raster_test=rasters,
        high_train=rng.uniform(size=(n, 4, 64, 64)).astype(np.float16), labels=labels, species_ids=np.arange(12),
        dims={k: 5 for k in vectors}, train_ids=rows.surveyId.to_numpy(), test_ids=rows.surveyId.to_numpy(),
        static_train=core.v29.candidate_static(rows), eco_train=core.v29.candidate_static(rows, False))
    store.high_test = store.high_train
    store.static_test, store.eco_test = store.static_train, store.eco_train
    return store, rows


def test_habitat_train_only_and_exact_sparse_neighbour_average(sensors):
    store, _ = sensors
    train, query = np.arange(24), np.arange(24, 32)
    environment = store.train['environment']
    expert = core.HabitatAnalogues().fit(environment, store.labels, train)
    expected_mean = environment[train].mean(0)
    assert np.allclose(expert.mean, expected_mean)
    actual = expert.predict_rank(environment[query], torch.device('cpu'))
    q = expert.transform(environment[query])
    distances = ((q[:, None]-expert.latent[None])**2).sum(2)
    ranked = np.argsort(distances, axis=1)
    ordered = np.take_along_axis(distances, ranked, 1)
    weights = expert.neighbour_weights(ordered)
    probability = np.einsum('bk,bks->bs', weights, store.labels[train][ranked])
    assert np.allclose(actual['value'], np.take_along_axis(probability, actual['rank'], 1), atol=.001)
    assert np.allclose(actual['distance'], ordered[:, 0], atol=1e-5)
    assert np.allclose(weights.sum(1), 1)
    # Changing query labels and covariates cannot affect the fitted metric or labels.
    labels = store.labels.copy(); labels[query] = 1-labels[query]
    changed_env = environment.copy(); changed_env[query] = 10000
    second = core.HabitatAnalogues().fit(changed_env, labels, train)
    repeated = second.predict_rank(environment[query], torch.device('cpu'))
    assert np.array_equal(actual['rank'], repeated['rank'])
    assert np.array_equal(actual['value'], repeated['value'])
    assert not np.intersect1d(expert.indices, query).size


def test_habitat_missing_and_constant_features():
    environment = np.full((8, 5), np.nan, np.float32)
    labels = np.eye(8, dtype=np.uint8)
    expert = core.HabitatAnalogues().fit(environment, labels, np.arange(6))
    prediction = expert.predict_rank(environment[6:], torch.device('cpu'))
    assert np.isfinite(prediction['value']).all() and np.isfinite(prediction['distance']).all()
    assert np.allclose(prediction['value'][:, :6], 1/6, atol=.001)
    with pytest.raises(ValueError, match='distinct'):
        core.HabitatAnalogues().fit(environment, labels, np.array([0, 0]))


def test_habitat_full_vocabulary_and_128_neighbours():
    rng = np.random.default_rng(31)
    environment = rng.normal(size=(260, 74)).astype(np.float32)
    labels = np.zeros((260, 5016), np.uint8)
    labels[:, 1] = 1
    labels[np.arange(260), np.arange(260)+100] = 1
    expert = core.HabitatAnalogues().fit(environment, labels, np.arange(256))
    assert expert.latent.shape == (256, 32)
    actual = expert.predict_rank(environment[256:], torch.device('cpu'))
    assert actual['rank'].shape == (4, 128)
    assert (actual['rank'][:, 0] == 1).all() and (actual['value'][:, 0] == 1).all()
    distance = np.tile(np.arange(128), (4, 1)).astype(np.float32)
    weight = expert.neighbour_weights(distance)
    assert np.allclose(weight.sum(1), 1) and (weight[:, :32].sum(1) > .5).all()


def test_residual_preserves_exact_count_prefix_and_limits():
    rng = np.random.default_rng(15)
    base = [rng.choice(200, size=k, replace=False).tolist() for k in range(8, 41)]
    expert = {'rank': np.stack([rng.permutation(200)[:128] for _ in base]), 'value': np.ones((len(base), 128))}
    for policy in core.POLICIES:
        output = core.residual_decode(base, expert, expert, policy)
        assert [len(x) for x in output] == list(range(8, 41))
        core.validate_residual(base, output, policy)
    assert core.residual_decode(base, None, None, core.POLICIES[0]) == base
    zero = {'rank': expert['rank'], 'value': np.zeros_like(expert['value'])}
    assert core.residual_decode(base, None, zero, {'neural': 0, 'habitat': 1, 'swaps': 4}) == base
    strong = {'rank': np.arange(100, 110)[None], 'value': np.ones((1, 10))}
    result = core.residual_decode([list(range(10))], strong, None, {'neural': 1, 'habitat': 0, 'swaps': 2})
    assert result == [list(range(8))+[100, 101]]
    with pytest.raises(ValueError, match='contract'):
        core.validate_residual([list(range(10))], [list(range(9))], {'swaps': 2})


def test_geographic_gate_rejects_v30_pattern_and_counts_distinct_surveys():
    countries = ['Denmark']*100+['Netherlands']*100+['France']*30
    gain = core.geographic_gain(np.r_[np.ones(200)*.005, np.ones(30)*-.001], countries)
    assert gain['gain'] > 0 and gain['outside_core_gain'] < 0 and not gain['geographic_gain_positive']
    doubled = core.geographic_gain(np.ones(60), ['France']*60, np.tile(np.arange(30), 2))
    assert doubled['outside_core_rows'] == 30 and doubled['supported_countries'] == 1
    assert doubled['geographic_gain_positive']
    # Actual archived v30 fails this new post-hoc development gate despite positive pooled gain.
    frame = pd.read_csv(ROOT/'results/v30_kaggle_output/regression_per_survey_v30.csv')
    gain = core.geographic_gain(frame.delta_f1, frame.country)
    assert gain['gain'] > 0 and gain['outside_core_gain'] < 0
    assert not gain['geographic_gain_positive']


def test_select_policy_does_not_choose_dominant_country_only_gain():
    rows = pd.DataFrame({'surveyId': np.arange(130), 'country': ['Denmark']*100+['France']*30})
    bundles = []
    for f in range(2):
        trials = {p['id']: np.ones(130)*.3 for p in core.POLICIES}
        trials[core.POLICIES[1]['id']] += np.r_[np.ones(100)*.1, np.ones(30)*-.01]
        trials[core.POLICIES[2]['id']] += .001
        bundles.append({'split': {'calibration': np.arange(130)}, 'trials': trials})
    selected, records = core.select_policy(bundles, rows)
    assert selected == core.POLICIES[2]
    assert not records[1]['allowed'] and records[2]['allowed']


def test_embedded_sources_control_and_production_recipe(tmp_path):
    notebook = nbformat.read(OUTPUT, as_version=4)
    nbformat.validate(notebook)
    assert OUTPUT.stat().st_size < 1_000_000
    source = (ROOT/'scripts/v31_notebook_core.py').read_text(encoding='utf-8')
    assert notebook.metadata.glc_v31.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    namespace = {}
    for cell in notebook.cells[1:3]:
        assert cell.outputs == [] and cell.execution_count is None
        exec(compile(cell.source, '<embedded-v31>', 'exec'), namespace)
    assert namespace['self_tests']()['passed']
    for name, digest in notebook.metadata.glc_v31.embedded_sources_sha256.items():
        actual = (ROOT/f'scripts/{name}_notebook_core.py').read_text(encoding='utf-8')
        assert hashlib.sha256(actual.encode()).hexdigest() == digest
    template = pd.read_csv(ROOT/'results/v29_kaggle_output/GLC25_PA_submission_v29.csv')
    species = np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    prediction = namespace['decode_control'](namespace['CONTROL_B64'], template, species)
    proof, decision = core.publish(tmp_path, template, template.surveyId.to_numpy(), prediction, species, True)
    assert proof['sha256'] == core.CONTROL_HASH
    assert not decision['eligible_for_submission'] and decision['prediction_file'] == 'unchanged_v29_DO_NOT_SUBMIT.csv'
    report = json.loads((ROOT/'results/v29_kaggle_output/v29_report.json').read_text())
    assert core.EPOCHS == (12, 15, 18)
    assert all(c['seed']+31000 not in [x['seed'] for x in core.CONFIGS] for c in core.CONFIGS)
    actual_calibrations = report['calibrations']
    for actual, expected in zip(actual_calibrations, core.PRODUCTION_CALIBRATIONS):
        assert actual['slope'] == expected['slope'] and actual['intercept'] == expected['intercept']


def test_production_admission_before_fit(sensors, tmp_path, monkeypatch):
    store, rows = sensors
    records = [{'group': 'replica', 'member': i, 'history': [{'seconds': 100}]} for i in range(3)]
    bundles = [{'records': records, 'split': {'training': np.arange(16)}}]*2
    class Expired:
        def require(self, seconds, stage):
            assert seconds > 2700 and 'whole' in stage
            raise TimeoutError('before fit')
    monkeypatch.setattr(core.v29, 'train_candidate', lambda *a, **k: pytest.fail('too late'))
    with pytest.raises(TimeoutError, match='before fit'):
        core.fit_production(bundles, {'neural': 1, 'habitat': 0}, rows, rows, store, [], tmp_path, Expired(), torch.device('cpu'))


def test_gpu_preflight_immediate_and_no_temporary_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a, **k: pytest.fail('GPU check too late'))
    with pytest.raises(RuntimeError, match='GPU'):
        core.run_v31('')
    assert not (tmp_path/'artifacts/v31_runtime').exists()
    failure = json.loads((tmp_path/'artifacts/v31_export/failure_report.json').read_text())
    assert not failure['eligible_for_submission']


def test_v30_archive_integrity_and_best_v29_unchanged():
    manifest = json.loads((ROOT/'results/v30_kaggle_output/v30_manifest.json').read_text())
    for name, digest in manifest['outputs'].items():
        assert hashlib.sha256((ROOT/'results/v30_kaggle_output'/name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((ROOT/'results/v29_kaggle_output/GLC25_PA_submission_v29.csv').read_bytes()).hexdigest() == core.CONTROL_HASH


@pytest.mark.parametrize('fail_after_prediction', [False, True])
def test_real_small_fit_production_and_compact_export(sensors, tmp_path, monkeypatch, fail_after_prediction):
    store, rows = sensors
    real_self_tests = core.self_tests()
    monkeypatch.setattr(core, 'self_tests', lambda: real_self_tests)
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
    monkeypatch.setattr(core.previous, 'make_splits', lambda *a, **k: (splits, [{'minimum_distance_km': 21.}]*2))
    monkeypatch.setattr(core.legacy, 'require_gpu', lambda: torch.device('cpu'))
    monkeypatch.setattr(core.legacy, 'discover_data_root', lambda: data)
    monkeypatch.setattr(core, 'decode_control', lambda *a: base)
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a, **k: {})
    monkeypatch.setattr(core.legacy, 'FeatureStore', lambda p: store)
    monkeypatch.setattr(core.legacy, 'load_rows_and_pairs', lambda *a: (rows, rows, None))
    monkeypatch.setattr(core, 'CONFIGS', tuple({**c, 'width': 16, 'epochs': 1} for c in core.CONFIGS))
    monkeypatch.setattr(core, 'EPOCHS', (1, 1, 1))
    monkeypatch.setattr(core.v29, 'decode_ensemble', lambda b, p, policy: [list(range(100, 108)) for _ in p])
    # The miniature cannot satisfy >=30 outside-country support. Force only that
    # statistical support gate; all train/predict/calibrate/residual/export code is real.
    geographic = core.geographic_gain
    def miniature_gain(*a, **k):
        result = geographic(*a, **k)
        result['geographic_gain_positive'] = result['gain'] > 0
        return result
    monkeypatch.setattr(core, 'geographic_gain', miniature_gain)
    # Exercise both active production experts, irrespective of tiny-fixture policy noise.
    policy = {'id': 'smoke_both', 'neural': 1., 'habitat': .5, 'swaps': 2}
    monkeypatch.setattr(core, 'select_policy', lambda *a: (policy, []))
    for name in ('V31_SOURCE_HASH', 'V30_SOURCE_HASH', 'V29_SOURCE_HASH', 'LEGACY_SOURCE_HASH'):
        monkeypatch.setattr(core, name, 'synthetic', raising=False)
    train, production_indices = core.v29.train_candidate, []
    def checked_train(*args, **kwargs):
        if kwargs.get('phase') == 'production':
            production_indices.append(args[2].copy())
            assert np.array_equal(args[2], np.arange(32)) and len(args[3]) == 0
        return train(*args, **kwargs)
    monkeypatch.setattr(core.v29, 'train_candidate', checked_train)
    if fail_after_prediction:
        def fail(*a, **k):
            raise ValueError('export interrupted')
        monkeypatch.setattr(core, 'save_compact', fail)
        with pytest.raises(ValueError, match='export interrupted'):
            core.run_v31('')
        export = tmp_path/'artifacts/v31_export'
        assert not (export/'GLC25_PA_submission_v31.csv').exists()
        assert (export/'failed_DO_NOT_SUBMIT.csv').exists()
        assert not json.loads((export/'failure_report.json').read_text())['eligible_for_submission']
        assert not (tmp_path/'artifacts/v31_runtime').exists()
        return
    result = core.run_v31('')
    assert result['status'] == 'complete' and result['eligible_for_submission']
    assert len(production_indices) == 3
    export = tmp_path/'artifacts/v31_export'
    assert len(list(export.iterdir())) == 5 and result['output_bytes'] < 16_000_000
    report = json.loads((export/'v31_report.json').read_text())
    assert report['production_diagnostics']['all_PA_rows_used']
    assert report['production_diagnostics']['cardinality_equal_to_scored_v29_per_row']
    assert all(r['training_surveys'] == 32 for r in report['training']['production'])
    assert not (tmp_path/'artifacts/v31_runtime').exists()
    output = pd.read_csv(export/'GLC25_PA_submission_v31.csv')
    assert all(len(x.split()) == 8 for x in output.predictions)
    with np.load(export/'calibration_top128_v31.npz', allow_pickle=False) as evidence:
        assert evidence['ranked_species_columns'].shape == (8, 2, 128)
        assert evidence['true_offsets'][-1] == len(evidence['true_species_columns'])
    manifest = json.loads((export/'v31_manifest.json').read_text())
    for name, digest in manifest['outputs'].items():
        assert hashlib.sha256((export/name).read_bytes()).hexdigest() == digest
