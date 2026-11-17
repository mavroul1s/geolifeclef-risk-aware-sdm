"""Build a private CPU reproduction notebook from audited, frozen v35 cells."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    existing = json.loads((ROOT/'notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb').read_text())
    core = (ROOT/'scripts/kaggle_reliability_pilot_core.py').read_text()
    compile(core, 'reliability-pilot', 'exec')
    cells = [{'cell_type': 'markdown', 'id': 'reliability-intro', 'metadata': {}, 'source': [
        '# GeoLifeCLEF: reliability, survey protocol and geographic coverage\n',
        'CPU reproduction of exploratory research from 13 November 2026.\n',
        'Official competition input only. No neural training or submission.\n',
        'Cached matched-v32 development results retain v34 provenance. All labels\n',
        'were historically consumed: this is not independent confirmation.\n',
        'The frozen v35 definition/evidence cells supply verified data readers,\n',
        'split reconstruction and cached comparator scores; run_v35 is never called.\n']}]
    cells.extend(existing['cells'][1:3])
    cells.append({'cell_type': 'code', 'id': 'reliability-cpu-run', 'metadata': {},
                  'source': core.splitlines(keepends=True), 'execution_count': None, 'outputs': []})
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
            'kaggle': {'accelerator': 'none', 'isGpuEnabled': False, 'isInternetEnabled': False,
                       'language': 'python', 'sourceType': 'notebook'},
            'research_audit': {'fresh_assessment': False, 'neural_training': False,
                'pilot_source_sha256': hashlib.sha256(core.encode()).hexdigest()}}}
    path = ROOT/'notebooks/geolifeclef_reliability_audit_2026_11_13.ipynb'
    data = (json.dumps(notebook, indent=1)+'\n').encode()
    if len(data) >= 1_000_000:
        raise ValueError('Notebook exceeds the source-size envelope')
    path.write_bytes(data)
    print(json.dumps({'notebook': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}))
    return path


if __name__ == '__main__':
    build()
