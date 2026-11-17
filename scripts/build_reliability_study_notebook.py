"""Build isolated GPU study notebooks without changing competitive recipes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(mode):
    study_text = (ROOT/'configs/reliability_study_r1.json').read_text()
    study = json.loads(study_text)
    frozen = json.loads((ROOT/'notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb').read_text())
    source_path = ROOT/'scripts'/('reliability_gpu_probe.py' if mode == 'probe' else 'reliability_study_core.py')
    source = source_path.read_text()
    compile(source, str(source_path), 'exec')
    helpers = (ROOT/'src/geolifeclef/reliability.py').read_text() if mode != 'probe' else ''
    config = f'STUDY = json.loads({study_text!r})\nSTUDY_MODE = {mode!r}\n'
    config += f'STUDY_CONFIG_SHA256 = {hashlib.sha256(study_text.encode()).hexdigest()!r}\n'
    cells = [{'cell_type': 'markdown', 'id': 'study-intro', 'metadata': {}, 'source': [
        '# Controlled GeoLifeCLEF reliability study R1\n',
        f'Mode: **{mode}**. Official competition inputs, GPU, no submission.\n',
        'All existing PA labels were historically consumed: no independent accuracy claim.\n',
        'Fixed architecture, epochs, seeds, role mapping and comparisons.\n']}]
    cells.append(frozen['cells'][1])
    for label, code in [('study-config', config), ('study-helpers', helpers), ('study-run', source)]:
        if code:
            cells.append({'cell_type': 'code', 'id': label, 'metadata': {}, 'execution_count': None,
                          'outputs': [], 'source': code.splitlines(keepends=True)})
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False, 'language': 'python', 'sourceType': 'notebook'},
        'reliability_study': {'mode': mode, 'fresh_assessment': False,
            'config_sha256': hashlib.sha256(study_text.encode()).hexdigest(),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'helpers_sha256': hashlib.sha256(helpers.encode()).hexdigest()}}}
    data = (json.dumps(notebook, indent=1)+'\n').encode()
    if len(data) >= 1_000_000:
        raise ValueError('Notebook exceeds source-size envelope')
    output = ROOT/f'notebooks/geolifeclef_reliability_r1_{mode}.ipynb'
    output.write_bytes(data)
    print(json.dumps({'mode': mode, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe', 'smoke', 'full'], required=True)
    build(parser.parse_args().mode)
