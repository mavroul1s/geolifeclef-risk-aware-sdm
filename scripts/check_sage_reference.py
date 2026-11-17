"""Reproduce fixed published probability rows without reading any target labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geolifeclef.range_io import BoundedHTTPFile
from geolifeclef.sage_inference import load_network, normalization_from_source, normalize


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    data, out = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/neon_bridge.json').read_text())
    ids = np.load(ROOT / 'data/external_replication_e1/test_indices.npy')[:128]
    record = json.loads((ROOT / 'data/external_replication_e1/zenodo_21335338.json').read_text())
    archive = next(f for f in record['files'] if f['key'] == 'predictions.tar')
    index = json.loads((ROOT / 'data/independent_data_audit/sage_predictions_tar_index.json').read_text())
    arrays, reports, configs = {'cell_ids': ids}, [], {}
    paths = [ROOT / 'configs/neon_bridge.json', Path(__file__),
             ROOT / 'src/geolifeclef/sage_inference.py', ROOT / 'src/geolifeclef/range_io.py',
             ROOT / 'data/neon_confirmation_feasibility/sage_configs.tar',
             data / 'published_source/src/data_modules/datamodule.py',
             ROOT / 'data/external_replication_e1/test_indices.npy']
    for run in config['model_runs']:
        with tarfile.open(ROOT / 'data/neon_confirmation_feasibility/sage_configs.tar') as tar:
            model_config = yaml.safe_load(tar.extractfile(run + '/config.yaml'))
        configs[run] = model_config
        subset = model_config['data']['init_args']['predictors_subset']
        predictors = []
        for group, columns in subset.items():
            if group == 'location':
                continue
            path = data / 'reference_predictors' / f'eval_{group}.h5'
            with h5py.File(path) as f:
                # h5py requires sorted fancy indices; preserve the published test order.
                values = f['predictor_data'][:]
                predictors.append(values[ids][:, columns])
            paths.append(path)
        raw = np.concatenate(predictors, axis=1).astype(np.float32)
        means, stds = normalization_from_source(data / 'published_source/src/data_modules/datamodule.py', subset)
        checkpoint = data / 'checkpoints' / run / 'best.ckpt'
        model = load_network(checkpoint, model_config)
        with torch.inference_mode():
            prediction = torch.sigmoid(model(normalize(raw, means, stds))).numpy()
        local = ROOT / 'data/external_replication_e1' / run / 'test_predictions.h5'
        fetched = requests = 0
        if local.exists():
            with h5py.File(local) as f:
                reference = f['probabilities'][:128]
            paths.append(local)
        else:
            member = next(m for m in index['members'] if m['name'] == run + '/test_predictions.h5')
            remote = BoundedHTTPFile(archive['links']['self'], member['data_offset'], member['size'],
                                    budget=config['maximum_remote_reference_bytes'],
                                    cache_dir=data / 'reference_cache' / run)
            with remote:
                with h5py.File(remote, 'r') as f:
                    reference = f['probabilities'][:128]
                fetched, requests = remote.fetched_bytes, remote.requests
        error = np.abs(prediction.astype(np.float64) - reference)
        passed = bool(np.isfinite(error).all()
                      and error.max() <= config['reference_max_absolute_probability_error']
                      and error.mean() <= config['reference_mean_absolute_probability_error'])
        report = {'run': run, 'rows': 128, 'classes': prediction.shape[1],
                  'max_absolute_probability_error': float(error.max()),
                  'mean_absolute_probability_error': float(error.mean()),
                  'reference_probability_array_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
                  'new_reference_network_bytes': fetched, 'reference_range_requests': requests,
                  'reference_passed': passed, 'checkpoint_sha256': sha(checkpoint)}
        reports.append(report)
        arrays[run + '__raw'] = raw
        arrays[run + '__means'], arrays[run + '__stds'] = means, stds
        arrays[run + '__published'] = reference
        arrays[run + '__cpu'] = prediction
        print(json.dumps(report), flush=True)
        paths.append(checkpoint)
    np.savez_compressed(data / 'cpu_reference.npz', **arrays)
    (data / 'model_configs.json').write_text(json.dumps(configs, indent=2) + '\n')
    result = {'status': 'passed' if all(r['reference_passed'] for r in reports) else 'failed',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
              'torch_version': torch.__version__, 'device': 'CPU', 'dtype': 'float32',
              'selection_fixed_before_comparison': 'first 128 published test rows; all 5771 classes',
              'target_labels_read': False, 'NEON_outcomes_read': False, 'runs': reports,
              'input_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)},
              'reference_bundle_sha256': sha(data / 'cpu_reference.npz')}
    (out / 'cpu_reference.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
