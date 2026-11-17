"""Build one technical rerun; never overwrite the original failed audit record."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'notebooks/geolifeclef_reliability_checkpoint_inference.ipynb'
    original = json.loads(path.read_text())
    notebook = json.loads(json.dumps(original))
    for cell in notebook['cells']:
        if cell['id'] == 'saved-inference':
            tree = ast.parse(''.join(cell['source']))
            tree.body = [n for n in tree.body if not isinstance(n, ast.Try)]
            cell['source'] = ast.unparse(tree).splitlines(keepends=True)
    settings = ROOT/'configs/reliability_inference_alignment.json'
    source_manifest = ROOT/'results/reliability_followup_d1_d3/inference_summary/inference_manifest.json'
    config = f'ALIGNMENT_CONFIG_SHA256 = {hashlib.sha256(settings.read_bytes()).hexdigest()!r}\n'
    config += f'FIRST_INFERENCE_MANIFEST_SHA256 = {hashlib.sha256(source_manifest.read_bytes()).hexdigest()!r}\n'
    sources = [('boundary-audit', (ROOT/'src/geolifeclef/inference_reproduction.py').read_text()),
        ('alignment-settings', config), ('aligned-run', (ROOT/'scripts/reliability_aligned_inference.py').read_text())]
    for name, text in sources:
        compile(text, name, 'exec')
        notebook['cells'].append({'cell_type': 'code', 'id': name, 'metadata': {},
                                 'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)})
    notebook['cells'][0]['source'].append('Technical alignment rerun; original exact-set failure retained. Numerical tie sensitivity is reported separately.\n')
    target = ROOT/'notebooks/geolifeclef_reliability_aligned_inference.ipynb'
    target.write_text(json.dumps(notebook, indent=1)+'\n')
    record = {'amended_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_config_changed': False, 'original_exact_member_guard_failure_preserved': True,
        'new_reliability_comparisons_inspected': False, 'calibration_thresholds_and_technical_reproduction_inspected': True,
        'alignment_config_sha256': hashlib.sha256(settings.read_bytes()).hexdigest(),
        'parent_notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'aligned_notebook_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in [
            'scripts/reliability_aligned_inference.py', 'src/geolifeclef/inference_reproduction.py',
            'scripts/build_aligned_reliability_inference_notebook.py']},
        'technical_reason': 'Original code concatenated probability-calibration and evaluation inference batches and demanded literal equality of legacy top-k species sets despite float16 boundaries. Align the original call boundaries/EMA and audit rank equivalence in raw rounding units. Do not change F1 or regional utility margins.',
        'attempt_1_reproduction': [json.loads((ROOT/f'results/reliability_followup_d1_d3/inference_summary/fold{f}_reproduction.json').read_text()) for f in [0, 1]]}
    (ROOT/'results/reliability_followup_d1_d3/alignment_amendment.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'notebook': str(target), 'sha256': record['aligned_notebook_sha256']}))


if __name__ == '__main__':
    main()
