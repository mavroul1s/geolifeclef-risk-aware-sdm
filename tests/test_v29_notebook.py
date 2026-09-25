import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.v29_notebook_core as core
from scripts.build_v29_notebook import consumed_paths, make_notebook, OUTPUT, ROOT, LIMIT
from scripts.build_v28_notebook import packed_consumed_ids, packed_v27_submission


@pytest.fixture
def small_store():
    torch.set_num_threads(2)
    rng = np.random.default_rng(402)
    n, species = 32, 12
    rows = pd.DataFrame({'surveyId': np.arange(n), 'lat': np.linspace(35, 60, n),
                         'lon': np.linspace(-5, 30, n), 'country': ['France', 'Italy'] * (n // 2)})
    vectors = {k: rng.normal(size=(n, 5)).astype(np.float32) for k in core.legacy.MODALITIES}
    rasters = {k: rng.normal(size=(n, *shape)).astype(np.float16)
               for k, shape in core.legacy.RASTER_SHAPES.items()}
    labels = (rng.uniform(size=(n, species)) < .2).astype(np.uint8)
    labels[:, -1] = 0
    labels[0, -1] = 1  # A singleton species must remain trainable.
    store = SimpleNamespace(train=vectors, test=vectors, raster_train=rasters, raster_test=rasters,
        high_train=rng.uniform(0, 1, size=(n, 4, 64, 64)).astype(np.float16), labels=labels,
        species_ids=np.arange(species), dims={k: 5 for k in vectors},
        static_train=core.candidate_static(rows), eco_train=core.candidate_static(rows, False))
    store.high_test, store.static_test, store.eco_test = store.high_train, store.static_train, store.eco_train
    return store, rows


@pytest.mark.parametrize('kind,geo,gamma', [('attention', True, 0.), ('conv', True, 0.), ('attention', False, 2.)])
def test_actual_training_and_full_refit_on_cpu(small_store, tmp_path, kind, geo, gamma):
    store, rows = small_store
    training, selection = np.arange(24), np.arange(24, 32)
    stats = core.fit_normalization(store, training)
    config = {'id': kind, 'kind': kind, 'geo': geo, 'seed': 42, 'epochs': 3, 'width': 16, 'gamma': gamma}
    initial = core.create_model(store, config, training).head.weight.detach().clone()
    model, record = core.train_candidate(store, rows, training, selection, stats, config, tmp_path,
                                         core.legacy.RuntimeGuard(), torch.device('cpu'))
    assert record['best_epoch'] in (1, 3)
    assert record['all_species_outputs_trained'] == 12
    assert not torch.equal(initial[-1], model.head.weight[-1])
    probabilities = core.predict(model, store, selection, stats, torch.device('cpu'), config, views=2)
    assert probabilities.shape == (8, 12) and np.isfinite(probabilities).all()
    assert np.all((probabilities > 0) & (probabilities < 1))
    full, refit = core.train_candidate(store, rows, np.arange(32), np.array([], dtype=int), stats,
                                      config, tmp_path, core.legacy.RuntimeGuard(), torch.device('cpu'),
                                      fixed_epochs=2, phase='full')
    assert refit['training_surveys'] == 32 and refit['selection_f1'] is None
    assert all(x['selection_f1'] is None for x in refit['history'])
    assert core.predict(full, store, selection, stats, torch.device('cpu'), config, test=True).shape == (8, 12)


def test_model_construction_seed_and_ecology_invariance(small_store):
    store, rows = small_store
    config = {**core.CONFIGS[0], 'width': 16}
    first = core.create_model(store, config, np.arange(24))
    second = core.create_model(store, config, np.arange(24))
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    changed = rows.assign(lat=-90, lon=179, country='Ukraine')
    assert np.array_equal(core.candidate_static(rows, False), core.candidate_static(changed, False))
    assert not np.array_equal(core.candidate_static(rows), core.candidate_static(changed))


def test_shared_platt_calibration_lowers_nll():
    rng = np.random.default_rng(405)
    truth = rng.uniform(.01, .8, size=(500, 12))
    labels = (rng.random(truth.shape) < truth).astype(np.uint8)
    biased = np.clip(truth * .2, 1e-6, 1 - 1e-6)
    result = core.fit_platt(biased, labels)
    corrected = core.calibrated(biased, result)
    def nll(p):
        return -(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()
    assert result['optimizer_success']
    assert nll(corrected) < nll(biased)


def test_policies_and_audit_no_change_are_honest():
    rows = pd.DataFrame({'surveyId': np.arange(40), 'lat': np.repeat([40, 50], 20),
                         'lon': [3.] * 40, 'country': ['France'] * 40})
    base = [list(range(8)) for _ in range(40)]
    target = np.zeros((40, 12), np.uint8)
    target[:, :8] = 1
    p = np.full((40, 12), .05, np.float32)
    p[:, 4:] = .8
    policy, trials = core.select_policy(base, p, target, rows)
    assert policy['id'] == 'control'
    frame, audit = core.audit_report(base, p, policy, target, rows)
    assert audit['gain'] == 0 and audit['bootstrap']['ci95'] == [0, 0]
    for policy in core.POLICIES:
        predicted = core.decode_ensemble(base, p, policy)
        assert all(8 <= len(x) <= 12 and len(set(x)) == len(x) for x in predicted)
    assert frame.delta_f1.sum() == 0


def test_notebook_source_and_embedded_reference(tmp_path):
    if not OUTPUT.exists():
        pytest.fail('Build v29 notebook before running tests')
    notebook = json.loads(OUTPUT.read_text(encoding='utf-8'))
    assert OUTPUT.stat().st_size < LIMIT
    assert len(notebook['cells']) == 4
    core_source = (ROOT / 'scripts/v29_notebook_core.py').read_text(encoding='utf-8')
    assert notebook['metadata']['glc_v29']['core_sha256'] == hashlib.sha256(core_source.encode()).hexdigest()
    namespace = {}
    for cell in notebook['cells'][1:3]:
        assert cell['outputs'] == [] and cell['execution_count'] is None
        source = ''.join(cell['source'])
        assert not any(isinstance(n, ast.Import) and any(x.name.startswith('scripts') for x in n.names)
                       for n in ast.walk(ast.parse(source)))
        exec(compile(source, '<notebook>', 'exec'), namespace)
    assert namespace['self_tests']()['passed']
    template = pd.read_csv(ROOT / 'results/v27_kaggle_output/GLC25_PA_submission_v27.csv')
    species = np.load(ROOT / 'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    predictions, consumed = namespace['decode_payloads'](namespace['CONTROL_B64'], namespace['CONSUMED_B64'], template, species)
    proof, decision = core.publish_predictions(tmp_path, template, template.surveyId.to_numpy(), predictions, species, True)
    assert proof['sha256'] == core.CONTROL_HASH
    assert not decision['eligible_for_submission']
    assert decision['prediction_file'] == 'unchanged_v27_DO_NOT_SUBMIT.csv'
    assert len(consumed) == 86592
    # Different predictions still must not be called a submission when their gate fails.
    replacement = 0 if 0 not in predictions[0] else next(i for i in range(5016) if i not in predictions[0])
    predictions[0][0] = replacement
    other = tmp_path / 'different'
    other.mkdir()
    _, decision = core.publish_predictions(other, template, template.surveyId.to_numpy(), predictions, species, False)
    assert decision['different_from_v27'] and not decision['eligible_for_submission']
    assert decision['prediction_file'] == 'candidate_DO_NOT_SUBMIT.csv'
    ready = tmp_path / 'ready'
    ready.mkdir()
    _, decision = core.publish_predictions(ready, template, template.surveyId.to_numpy(), predictions, species, True)
    assert decision['eligible_for_submission'] and decision['prediction_file'] == 'GLC25_PA_submission_v29.csv'


def test_registered_official_split_has_no_block_or_id_leakage():
    path = ROOT / 'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv'
    if not path.exists():
        pytest.skip('Local official PA metadata not installed')
    consumed = np.unique(np.concatenate([pd.read_csv(p, usecols=['surveyId']).surveyId.to_numpy() for p in consumed_paths()]))
    rows = pd.read_csv(path, usecols=['surveyId', 'lat', 'lon', 'country']).drop_duplicates('surveyId').reset_index(drop=True)
    split, manifest = core.make_split(rows, consumed)
    assert manifest['counts'] == {'training': 42840, 'selection': 18753, 'calibration': 8045, 'assessment': 2395}
    blocks = core.legacy.spatial_blocks(rows)
    assert manifest['minimum_evaluation_distance_km'] >= 20
    assert not np.intersect1d(rows.surveyId.to_numpy()[split['assessment']], consumed).size
    roles = list(split)
    for i, role in enumerate(roles):
        for other in roles[i + 1:]:
            assert not np.intersect1d(split[role], split[other]).size
            assert not set(blocks[split[role]]) & set(blocks[split[other]])


def test_budget_estimate_covers_whole_ensemble():
    records = [{'best_epoch': 10, 'history': [{'seconds': 60}, {'seconds': 120}]}] * 3
    assert core.full_refit_estimate(records, 200, 100) == pytest.approx(3 * 90 * 2 * 10 * 1.4 + 2700)


def test_gpu_missing_fails_before_data_processing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a, **k: pytest.fail('GPU check was too late'))
    with pytest.raises(RuntimeError, match='GPU'):
        core.run_v29('', '')
    report = json.loads((tmp_path / 'artifacts/v29_export/failure_report.json').read_text())
    assert report['status'] == 'failed' and not report['official_submission_made']
    assert not (tmp_path / 'artifacts/v29_runtime').exists()


def test_dual_resolution_writer_contract(tmp_path, monkeypatch):
    rows = pd.DataFrame({'surveyId': [1, 2, 3]})
    def fake_reader(task):
        return tuple([np.full(width, task[-1], np.float32) for width in core.legacy.REMOTE_DIMS.values()] +
                     [np.full(shape, task[-1], np.float16) for shape in core.legacy.RASTER_SHAPES.values()] +
                     [np.full((4, 64, 64), task[-1], np.float16)])
    monkeypatch.setattr(core, 'read_multiresolution', fake_reader)
    metadata = core.write_multiresolution(tmp_path, rows, 'PA', 'train', tmp_path,
                                          core.legacy.RuntimeGuard(), 2)
    assert metadata['sentinel_pixels'] == 64
    for name, shape in core.legacy.RASTER_SHAPES.items():
        array = np.load(tmp_path / f'train_{name}_raster.npy')
        assert array.shape == (3, *shape) and np.all(array[2] == 3)
    assert np.load(tmp_path / 'train_sentinel64.npy').shape == (3, 4, 64, 64)


def test_sentinel_indices_follow_band_metadata_or_documented_rgbnir(small_store):
    assert core.sentinel_band_indices(['B02', 'B03', 'B04', 'B08'], ['undefined'] * 4) == [2, 1, 0, 3]
    assert core.sentinel_band_indices([None] * 4, ['red', 'green', 'blue', 'undefined']) == [0, 1, 2, 3]
    assert core.sentinel_band_indices([None] * 4, ['undefined'] * 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='conflicting'):
        core.sentinel_band_indices(['B08', None, None, None], ['undefined'] * 4)
    store, _ = small_store
    store.high_train[:] = np.array([.2, .3, .1, .8], np.float16)[None, :, None, None]
    stats = core.fit_normalization(store, np.arange(24))
    batch = core.batch_inputs(store, np.array([0]), stats, torch.device('cpu'))
    assert float(batch['sentinel'][0, 4].mean()) == pytest.approx(.6, abs=.001)
    assert float(batch['sentinel'][0, 5].mean()) == pytest.approx(-.5 / 1.1, abs=.001)


def test_complete_miniature_pipeline_and_cleanup(small_store, tmp_path, monkeypatch):
    """Exercise the actual orchestration, refit, calibration, scoring and publication.

    Synthetic sensors/reference replace unavailable competition files, not the new
    training/prediction functions. This is not a score/performance benchmark.
    """
    store, rows = small_store
    monkeypatch.chdir(tmp_path)
    store.labels = np.pad(store.labels, ((0, 0), (0, 5016 - 12)))
    store.species_ids = np.arange(5016)
    store.train_ids = store.test_ids = rows.surveyId.to_numpy()
    store.cache = tmp_path / 'sensors'
    store.cache.mkdir()
    np.save(store.cache / 'train_sentinel64.npy', store.high_train)
    np.save(store.cache / 'test_sentinel64.npy', store.high_test)
    data = tmp_path / 'official'
    data.mkdir()
    rows.to_csv(data / 'GLC25_PA_metadata_train.csv', index=False)
    template = pd.DataFrame({'surveyId': rows.surveyId, 'predictions': [''] * len(rows)})
    template.to_csv(data / 'GLC25_SAMPLE_SUBMISSION.csv', index=False)
    monkeypatch.setattr(core.legacy, 'EXPECTED_TEST_ROWS', len(rows))
    baseline = [list(range(8)) for _ in range(len(rows))]
    proof = core.legacy.write_submission(tmp_path / 'control.csv', template, store.test_ids,
                                         baseline, store.species_ids)
    monkeypatch.setattr(core, 'CONTROL_HASH', proof['sha256'])
    split = {'training': np.arange(16), 'selection': np.arange(16, 24),
             'calibration': np.arange(24, 28), 'assessment': np.arange(28, 32)}
    monkeypatch.setattr(core, 'make_split', lambda *a, **k: (split, {'minimum_evaluation_distance_km': 21.}))
    monkeypatch.setattr(core.legacy, 'require_gpu', lambda: torch.device('cpu'))
    monkeypatch.setattr(core.legacy, 'discover_data_root', lambda: data)
    monkeypatch.setattr(core, 'decode_consumed', lambda x: np.arange(28))
    monkeypatch.setattr(core, 'decode_payloads', lambda *a: (baseline, np.arange(28)))
    monkeypatch.setattr(core.legacy, 'prepare_feature_store', lambda *a, **k: {})
    monkeypatch.setattr(core.legacy, 'FeatureStore', lambda p: store)
    monkeypatch.setattr(core.legacy, 'load_rows_and_pairs', lambda *a: (rows, rows, None))
    monkeypatch.setattr(core.legacy.POGridIndex, 'build', lambda *a, **k: None)
    def reference(*args):
        bundle = {'predictions': {}, 'components': {}, 'frequencies': None, 'graph': None}
        for role in ('selection', 'calibration', 'assessment'):
            bundle['predictions'][role] = {'base_lists': [list(range(8)) for _ in split[role]],
                                          'candidate': None, 'predicted_count': None, 'risk': None}
            bundle['components'][role] = {'spatial': None, 'po': None}
        return bundle, {'synthetic_reference': True}
    monkeypatch.setattr(core.legacy, '_build_models_for_fold', reference)
    monkeypatch.setattr(core.legacy, 'compose_predictions', lambda base, *a: base)
    monkeypatch.setattr(core, 'CONFIGS', ({**core.CONFIGS[0], 'width': 16, 'epochs': 1},))
    monkeypatch.setattr(core, 'select_policy', lambda *a: ({'id': 'mini', 'alpha': 1., 'count_scale': 1.}, []))
    monkeypatch.setattr(core, 'V29_SOURCE_HASH', 'synthetic', raising=False)
    monkeypatch.setattr(core, 'LEGACY_SOURCE_HASH', 'synthetic', raising=False)
    result = core.run_v29('', '')
    export = tmp_path / 'artifacts/v29_export'
    assert result['status'] == 'complete' and len(list(export.iterdir())) == 4
    report = json.loads((export / 'v29_report.json').read_text())
    assert report['training']['development'][0]['training_surveys'] == 16
    assert report['training']['full_data'][0]['training_surveys'] == 32
    assert report['integrity']['all_5016_species']
    assert not (tmp_path / 'artifacts/v29_runtime').exists()
    manifest = json.loads((export / 'v29_manifest.json').read_text())
    for name, digest in manifest['outputs'].items():
        assert hashlib.sha256((export / name).read_bytes()).hexdigest() == digest
