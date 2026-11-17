"""Build an inference-only notebook using completed private R1 checkpoints."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    config = (ROOT/'configs/reliability_followup_d1_d3.json').read_text()
    study = (ROOT/'configs/reliability_study_r1.json').read_text()
    original = json.loads((ROOT/'notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb').read_text())
    tree = ast.parse((ROOT/'scripts/reliability_study_core.py').read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.Try) and
                 not (isinstance(n, ast.FunctionDef) and n.name in ['run_reliability_study', 'r1_fit_fold'])]
    cells = [{'cell_type': 'markdown', 'id': 'inference-intro', 'metadata': {}, 'source': [
        '# R1 checkpoint reuse: decision reliability follow-up\n',
        'Private inference-only run, no new neural training or submission.\n',
        'Exploratory development work on historically consumed labels.\n']} , original['cells'][1]]
    manifest = ROOT/'results/reliability_study_r1/full_output/r1_export/r1_manifest.json'
    settings = f'STUDY = json.loads({study!r})\nSTUDY_MODE = "full"\n'
    settings += f'FOLLOWUP = json.loads({config!r})\n'
    settings += f'FOLLOWUP_CONFIG_SHA256 = {hashlib.sha256(config.encode()).hexdigest()!r}\n'
    settings += f'SOURCE_MANIFEST_SHA256 = {hashlib.sha256(manifest.read_bytes()).hexdigest()!r}\n'
    sources = [('inference-settings', settings),
        ('r1-helpers', (ROOT/'src/geolifeclef/reliability.py').read_text()),
        ('decision-helpers', (ROOT/'src/geolifeclef/decision_reliability.py').read_text()),
        ('frozen-preprocessing', ast.unparse(tree)),
        ('saved-inference', (ROOT/'scripts/reliability_inference_core.py').read_text())]
    for name, source in sources:
        compile(source, name, 'exec')
        cells.append({'cell_type': 'code', 'id': name, 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.splitlines(keepends=True)})
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'}}}
    path = ROOT/'notebooks/geolifeclef_reliability_checkpoint_inference.ipynb'
    path.write_text(json.dumps(notebook, indent=1)+'\n')
    frozen = ROOT/'results/reliability_followup_d1_d3/protocol_freeze.json'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    record = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'fresh_assessment': False,
        'config_sha256': hashlib.sha256(config.encode()).hexdigest(),
        'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'new_neural_training_permitted': False, 'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'designed_after_R1_assessment_inspection': True,
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            'src/geolifeclef/decision_reliability.py', 'scripts/reliability_inference_core.py',
            'scripts/build_reliability_inference_notebook.py']}}
    if frozen.exists():
        old = json.loads(frozen.read_text())
        if old['config_sha256'] != record['config_sha256']:
            raise ValueError('Refuse silent change to frozen diagnostic configuration')
        record['initial_frozen_at_utc'] = old.get('initial_frozen_at_utc', old['frozen_at_utc'])
    frozen.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'notebook': str(path), 'sha256': record['notebook_sha256']}))


if __name__ == '__main__':
    build()
