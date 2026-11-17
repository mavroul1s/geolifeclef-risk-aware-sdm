"""Verify all GPU probability columns and frozen input/output provenance locally."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT / 'results/neon_bridge'
    contract = json.loads((out / 'gpu_reference_contract.json').read_text())
    state = json.loads((out / 'kaggle_gpu_state.json').read_text())
    config = json.loads((ROOT / 'configs/neon_bridge.json').read_text())
    assert state['status'] == 'complete' and state['private'] and state['GPU_requested']
    assert not state['competition_submission_made']
    assert state['notebook_sha256'] == contract['notebook_sha256'] == sha(ROOT / 'notebooks/sage_gpu_reference.ipynb')
    assert contract['CPU_reference_report_sha256'] == sha(out / 'cpu_reference.json')
    paths = ROOT / 'artifacts/neon_bridge_gpu'
    for item in state['output_files']:
        path = paths / item['name']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    report = json.loads((paths / 'gpu_report.json').read_text())
    assert not report['neural_training_performed'] and not report['target_labels_read']
    assert report['elapsed_seconds'] <= 1200 and report['GPU_name']
    results = []
    with np.load(paths / 'gpu_probabilities.npz') as gpu, np.load(ROOT / 'data/neon_bridge/cpu_reference.npz') as cpu:
        np.testing.assert_array_equal(gpu['cell_ids'], cpu['cell_ids'])
        assert set(gpu.files) == {'cell_ids', *contract['admitted_runs']}
        for run in contract['admitted_runs']:
            p = gpu[run]
            assert p.shape == (128, 5771) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            comparisons = {}
            for name in ['cpu', 'published']:
                delta = np.abs(p.astype(np.float64) - cpu[run + '__' + name])
                comparisons[name] = {'max_absolute_error': float(delta.max()), 'mean_absolute_error': float(delta.mean()),
                                     'passed': bool(delta.max() <= config['reference_max_absolute_probability_error']
                                                    and delta.mean() <= config['reference_mean_absolute_probability_error'])}
            results.append({'run': run, 'probabilities_compared': int(p.size), 'comparisons': comparisons})
    result = {'status': 'passed' if all(c['passed'] for r in results for c in r['comparisons'].values()) else 'failed',
              'GPU_name': report['GPU_name'], 'torch_version': report['torch_version'],
              'notebook_elapsed_seconds': report['elapsed_seconds'], 'runs': results,
              'input_contract_and_output_hashes_verified': True, 'outputs': state['output_files'],
              'scope': 'Published-input inference reproduction only; no fresh field-data performance or new scientific effect.',
              'ResNet_excluded_due_to_failed_CPU_reference': contract['excluded_runs']}
    (out / 'gpu_reference_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
