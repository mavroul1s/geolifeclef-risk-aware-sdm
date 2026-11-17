"""Package only passing CPU-reference pipelines for one bounded private GPU check."""
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    data, out = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'
    report = json.loads((out / 'cpu_reference.json').read_text())
    admitted = [r['run'] for r in report['runs'] if r['reference_passed']]
    if not admitted:
        raise ValueError('No pipeline passed the fixed CPU reference')
    checkpoint_index = json.loads((data / 'checkpoint_index.json').read_text())
    model_record = json.loads((ROOT / 'data/external_replication_e1/zenodo_21335338.json').read_text())
    archive = next(f for f in model_record['files'] if f['key'] == 'checkpoints.tar')
    configs = json.loads((data / 'model_configs.json').read_text())
    columns = np.linspace(0, 5770, 256, dtype=int)
    arrays, members, minimal_configs = {'probe_columns': columns}, {}, {}
    with np.load(data / 'cpu_reference.npz') as reference:
        arrays['cell_ids'] = reference['cell_ids']
        for run in admitted:
            for name in ['raw', 'means', 'stds']:
                arrays[run + '__' + name] = reference[run + '__' + name]
            arrays[run + '__probe'] = reference[run + '__cpu'][:, columns]
            m = next(m for m in checkpoint_index['members'] if m['name'] == run + '/checkpoints/best.ckpt')
            members[run] = {**m, 'sha256': next(r['checkpoint_sha256'] for r in report['runs'] if r['run'] == run)}
            minimal_configs[run] = {'model': {'init_args': {'net': configs[run]['model']['init_args']['net']}}}
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    encoded = base64.b64encode(stream.getvalue()).decode()
    code = '''import base64, hashlib, io, json, signal, time, urllib.request
from pathlib import Path
import numpy as np
import torch
started = time.monotonic()
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('20-minute reference budget exceeded')))
signal.alarm(1200)
assert torch.cuda.is_available(), 'Actual CUDA GPU required'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
device = torch.device('cuda')
'''
    code += (ROOT / 'src/geolifeclef/sage_inference.py').read_text() + '\n'
    code += f"bundle = np.load(io.BytesIO(base64.b64decode({encoded!r})))\n"
    code += f"members = {members!r}\nconfigs = json.loads({json.dumps(minimal_configs)!r})\narchive_url = {archive['links']['self']!r}\n"
    code += '''outputs = {'cell_ids': bundle['cell_ids']}
reports = []
for run, member in members.items():
    path = Path('/kaggle/working') / (run + '.ckpt')
    digest = hashlib.sha256()
    with path.open('wb') as target:
        for offset in range(0, member['size'], 8 * 1024**2):
            count = min(8 * 1024**2, member['size'] - offset)
            left = member['data_offset'] + offset
            request = urllib.request.Request(archive_url, headers={'Range': f'bytes={left}-{left+count-1}'})
            with urllib.request.urlopen(request, timeout=40) as response:
                assert response.status == 206
                assert response.headers.get('Content-Range', '').startswith(f'bytes {left}-{left+count-1}/')
                body = response.read(count + 1)
            assert len(body) == count
            target.write(body); digest.update(body)
    assert digest.hexdigest() == member['sha256'], 'Checkpoint digest mismatch'
    model = load_network(path, configs[run]).to(device).eval()
    x = normalize(bundle[run+'__raw'], bundle[run+'__means'], bundle[run+'__stds']).to(device)
    torch.cuda.synchronize(); t0 = time.monotonic()
    with torch.inference_mode():
        prediction = torch.sigmoid(model(x)).cpu().numpy()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    delta = np.abs(prediction[:, bundle['probe_columns']].astype(np.float64) - bundle[run+'__probe'])
    passed = bool(np.isfinite(prediction).all() and delta.max() <= 1e-5 and delta.mean() <= 1e-6)
    outputs[run] = prediction
    reports.append({'run': run, 'shape': list(prediction.shape), 'checkpoint_sha256': digest.hexdigest(),
                    'probe_max_absolute_error': float(delta.max()), 'probe_mean_absolute_error': float(delta.mean()),
                    'probe_passed': passed, 'inference_seconds': elapsed})
    path.unlink()
    del model
    torch.cuda.empty_cache()
np.savez_compressed('/kaggle/working/gpu_probabilities.npz', **outputs)
report = {'status': 'probe_passed_pending_full_local_comparison' if all(r['probe_passed'] for r in reports) else 'failed',
          'GPU_name': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
          'CUDA_version': torch.version.cuda, 'elapsed_seconds': time.monotonic()-started,
          'runs': reports, 'neural_training_performed': False, 'target_labels_read': False,
          'NEON_credential_included': False, 'NEON_performance_evaluated': False}
Path('/kaggle/working/gpu_report.json').write_text(json.dumps(report, indent=2)+'\\n')
print(json.dumps(report))
signal.alarm(0)
assert all(r['probe_passed'] for r in reports)
'''
    compile(code, '<gpu_reference>', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# Published SAGE inference reference\n', 'No training, field-data evaluation, or competition submission. Full local comparison is required after collection.']},
                          {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    body = json.dumps(notebook, indent=1) + '\n'
    credential = json.loads((ROOT.parent / 'kaggle_api_key/kaggle.json').read_text())
    neon_path = ROOT.parent / 'neon_api_key/neon_api_key'
    for secret in [credential['key'], neon_path.read_text().strip()]:
        if secret and secret in body:
            raise ValueError('Credential found in notebook')
    path = ROOT / 'notebooks/sage_gpu_reference.ipynb'
    path.write_text(body)
    contract = {'at_utc': datetime.now(timezone.utc).isoformat(), 'admitted_runs': admitted,
                'excluded_runs': [r['run'] for r in report['runs'] if not r['reference_passed']],
                'admission': 'CPU reference passes per pipeline; failed pipelines remain excluded without relaxed tolerances',
                'GPU_job_limit': 1, 'wall_time_limit_minutes': 20, 'neural_training': False,
                'CPU_reference_report_sha256': hashlib.sha256((out / 'cpu_reference.json').read_bytes()).hexdigest(),
                'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'full_local_comparison_required': True, 'NEON_outcomes_read': False,
                'note': 'A reference check of the admitted pipeline; does not establish fresh NEON input compatibility or ecological gains.'}
    (out / 'gpu_reference_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(contract))


if __name__ == '__main__':
    main()
