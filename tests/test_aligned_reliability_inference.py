import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from geolifeclef.inference_reproduction import reference_topk_gap_ulps

ROOT = Path(__file__).resolve().parents[1]


def test_alignment_reuses_training_stats_and_restores_role_batch_boundaries(tmp_path):
    source, cache, data = [tmp_path/n for n in ['source', 'cache', 'data']]
    for p in [source, cache, data]:
        p.mkdir()
    ids = np.arange(32)
    rows = pd.DataFrame({'surveyId': ids, 'lat': 1., 'lon': 2., 'country': 'a',
                         'region': 'r', 'year': 2020, 'areaInM2': 1., 'geoUncertaintyInM': 1.})
    rows.to_csv(data/'GLC25_PA_metadata_train.csv', index=False)
    split = {'training': ids[:16], 'selection': ids[16:20], 'calibration': ids[20:26], 'assessment': ids[26:]}
    digest = lambda b: hashlib.sha256(b).hexdigest()
    file_digest = lambda p: digest(Path(p).read_bytes())
    training_hash = digest(ids[:16].astype('<i8').tobytes())
    original_manifest = {'effective_role_hashes': [{'training': training_hash}]*2}
    (source/'r1_manifest.json').write_text(json.dumps(original_manifest))
    manifest = {'outputs': {}}
    probabilities = np.linspace(.01, .8, 20).astype(np.float16)
    ranks = np.argsort(-probabilities, kind='stable')[:18]
    for fold in [0, 1]:
        name = f'fold{fold}_normalization.json'
        (cache/name).write_text(json.dumps({'vector': {'environment': {'mean': [1., 2.], 'std': [3., 4.]}}}))
        manifest['outputs'][name] = file_digest(cache/name)
        (source/f'fold{fold}_report.json').write_text(json.dumps({'role_counts': {'probability_calibration': 4}}))
        np.savez(source/f'fold{fold}_predictions.npz',
                 member_top18_columns=np.tile(ranks, (12, 2, 1)), role=np.array(['risk_calibration']*6+['assessment']*6))
    (cache/'inference_manifest.json').write_text(json.dumps(manifest))
    calls = []
    def predict(model, store, indices, stats, device, config, **kwargs):
        calls.append(indices.copy())
        return np.tile(probabilities, (len(indices), 1))
    class Model:
        def requires_grad_(self, enabled):
            assert enabled is False
            return self
    v29 = SimpleNamespace(create_model=lambda *a, **k: Model(), predict=predict)
    ns = {'Path': Path, 'json': json, 'np': np, 'pd': pd, 'v29': v29,
        'legacy': SimpleNamespace(sha256_file=file_digest, sha256_bytes=digest, discover_data_root=lambda: data),
        'v31': SimpleNamespace(previous=SimpleNamespace(make_splits=lambda rows: ([split, split], []))),
        'r1_read_predictors': lambda task: ('real', 'real', 'real'),
        'FIRST_INFERENCE_MANIFEST_SHA256': file_digest(cache/'inference_manifest.json'),
        'STUDY': {'neural_seeds': [1, 2]}, 'reference_topk_gap_ulps': reference_topk_gap_ulps}
    tree = ast.parse((ROOT/'scripts/reliability_aligned_inference.py').read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.Try)]
    exec(compile(tree, 'alignment-test', 'exec'), ns)
    ns['find_input_manifest'] = lambda name: source/'r1_manifest.json' if name.startswith('r1_export') else cache/'inference_manifest.json'
    audit, required = ns['install_aligned_inference']()
    stats = v29.fit_normalization(SimpleNamespace(train_ids=ids), ids[:16])
    assert stats['vector']['environment']['mean'].dtype == np.float32
    out = v29.predict(None, None, ids[16:], stats, None, {'id': 'r1_fold0_seed1', 'seed': 1})
    assert [len(c) for c in calls] == [4, 12]
    assert out.shape == (16, 20) and required == 16
    assert audit[0]['reference_sets_within_two_raw_ULPs'] == 1
    assert ns['r1_read_predictors'](('data', 16)) == ('real', 'real', 'real')
    assert ns['r1_read_predictors'](('data', 0))[2].shape == (4, 64, 64)
