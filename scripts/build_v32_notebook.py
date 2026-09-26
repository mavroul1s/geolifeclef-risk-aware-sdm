"""Build one offline Kaggle notebook with verified frozen sources and exact v31 CSV."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'notebooks/geolifeclef_v32_asymmetric_rare_specialist_ensemble.ipynb'
CONTROL_HASH = 'da070a8ac5708ef5d7fb38cdbecf862aa9d036e4d246bc1e4b364572bb7fe367'


def pack_control():
    path = ROOT/'results/v31_kaggle_output/GLC25_PA_submission_v31.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_HASH:
        raise ValueError('Not the exact scored v31 CSV')
    frame = pd.read_csv(path)
    species = np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy', allow_pickle=False)
    if len(frame) != 14784 or not frame.surveyId.is_unique or len(species) != 5016:
        raise ValueError('Invalid v31 dimensions')
    lookup = pd.Index(species)
    lists = [lookup.get_indexer(np.fromstring(text, sep=' ', dtype=np.int64)) for text in frame.predictions]
    if any((row < 0).any() or not 8 <= len(row) <= 40 or len(row) != len(np.unique(row)) for row in lists):
        raise ValueError('Invalid v31 species lists')
    raw = np.array(list(map(len, lists)), np.uint8).tobytes()+np.concatenate(lists).astype('<u2').tobytes()
    packed = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    return base64.b64encode(packed).decode(), hashlib.sha256(packed).hexdigest(), hashlib.sha256(raw).hexdigest()


def build(output=OUTPUT):
    names = ('v27', 'v29', 'v30', 'v31', 'v32')
    sources = {n: (ROOT/f'scripts/{n}_notebook_core.py').read_text(encoding='utf-8') for n in names}
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    manifest = json.loads((ROOT/'results/v31_kaggle_output/v31_manifest.json').read_text())
    for name, field in (('v27', 'embedded_legacy_sha256'), ('v29', 'embedded_v29_sha256'),
                        ('v30', 'embedded_v30_sha256'), ('v31', 'source_sha256')):
        if hashes[name] != manifest[field]:
            raise ValueError(f'Frozen scored source changed: {name}')
    control = pack_control()
    loader = 'import types as _types\n'
    for index, name in enumerate(names[:-1]):
        encoded = base64.b64encode(lzma.compress(sources[name].encode(), preset=9)).decode()
        loader += (f"_{name}_source = lzma.decompress(base64.b64decode({encoded!r}))\n"
                   f"assert hashlib.sha256(_{name}_source).hexdigest() == {hashes[name]!r}\n"
                   f"_{name} = _types.ModuleType('v32_frozen_{name}')\n")
        if index:
            old, symbol = names[index-1], ('legacy' if name == 'v29' else 'previous')
            statement = f'import scripts.{old}_notebook_core as {symbol}'
            assert sources[name].count(statement) == 1
            loader += (f"_{name}.__dict__[{symbol!r}] = _{old}\n"
                       f"_{name}_source = _{name}_source.replace({statement.encode()!r}, b'')\n")
        loader += f"exec(compile(_{name}_source, 'frozen-{name}', 'exec'), _{name}.__dict__)\ndel _{name}_source\n"
    loader += 'previous = _v31\ndel _v27, _v29, _v30, _v31\n'
    statement = 'import scripts.v31_notebook_core as previous'
    assert sources['v32'].count(statement) == 1
    standalone = sources['v32'].replace(statement, loader)
    payload = (f"V32_SOURCE_HASH = {hashes['v32']!r}\nFROZEN_SOURCE_HASHES = {dict((n, hashes[n]) for n in names[:-1])!r}\n"
        f"CONTROL_PAYLOAD_HASH = {control[1]!r}\nCONTROL_RAW_HASH = {control[2]!r}\nCONTROL_B64 = {control[0]!r}\n"
        "print({'source_sha256': V32_SOURCE_HASH, 'control': 'exact scored v31', 'fresh_holdout_remaining': 0})\n")
    intro = """# GeoLifeCLEF v32 — asymmetric-loss rare specialists + seed ensemble

An unscored candidate targeting the competition winner, **not a demonstrated SOTA**.
Our best v31: **0.24072 public / 0.21301 private**. Winning private target: 0.23021.

## Run

1. Attach **GeoLifeCLEF25 @ CVPR & LifeCLEF** (`geolifeclef-2025`), the competition input.
2. Select **GPU T4 x1**, restart, and **Run All**. Internet OFF is supported. No extra dataset.
3. Submit **only** `v32_export/GLC25_PA_submission_v32.csv`, and only when
   `eligible_for_submission` is `true`. Never submit reports, NPZ, ZIP or DO_NOT_SUBMIT files.

## What changes

V31 diagnostics favored neural seed diversity; habitat-only changes were negative.
V32 adds a new three-model seed bag (the proven 12/15/18-epoch recipe), plus two wider
multimodal attention models (20/24 epochs). The latter use asymmetric loss with clipped
easy negative labels; the geography-free model adds a nonlinear branch only for taxa
with training frequency >=5 and training prevalence <=0.005. All 5,016 species remain
in the main loss and evaluation. These are testable hypotheses, not claimed gains.

All production models use **all 88,987 PA surveys**. The exact scored v31 CSV is embedded
and hash-verified. Every test row retains its v31 number of species and leading 60%.
Calibration chooses a seed-only, specialist-only or mixed policy, allowing at most
2/4/8 tail swaps. Zero change is an explicit option. Stronger changes are not forced.

22 policies are selected only on calibration rows, before two regression checks against
the **frozen v31 recipe refit** (not v29). Both fold gains, spatial-bootstrap lower bound,
country-macro gain and gain outside Denmark/Netherlands must be positive. Failure means
no new eligible CSV; retain v31. All PA IDs were assessed before: **no fresh holdout remains**.
The bootstrap and gates are development diagnostics, not independent proof.

## Runtime and export

The complete v31 run took 3.1845 hours on T4. V32 has 22 development fits and at most
5 production fits, including wider models, so it is heavier. Full v32 T4 runtime is
**not yet measured**. A 10.75-hour cooperative guard reserves headroom inside 12 hours;
whole-production admission uses measured development speed with a 1.5x margin plus
45 minutes. Slow hardware can stop safely instead of finishing. GPU and spatial-split
checks happen before loading the large predictors. A failed export revokes its CSV.

Successful output: exactly five files, <=16 MB total, including diagnostics useful for
the next iteration. Temporary features and model checkpoints are removed.

## Provenance and limitations

Inspired by rare-specialist ensembles described in the organizers' GeoLifeCLEF 2025
overview and asymmetric loss in Tighnari v2; this is not a reproduction of their full
systems. No pretrained external weights, extra data, or test labels are used.
Sources: https://www.dei.unipd.it/~faggioli/temp/clef2025/paper_234.pdf and
https://ceur-ws.org/Vol-4038/paper_246.pdf .

Only the test control CSV is byte-exact; development uses re-trained v31 weights.
Calibration transfers from v29 across seeds for the seed bag and from selection-only
folds to full-data specialist training. That transfer may fail under distribution shift.
Repeated development can overfit; no policy can guarantee a better leaderboard result.
"""
    cells = [{'cell_type': 'markdown', 'id': 'v32-intro', 'metadata': {}, 'source': intro.splitlines(keepends=True)}]
    run = "V32_RESULT = run_v32(CONTROL_B64)\nprint(json.dumps(V32_RESULT, indent=2))\nprint(V32_RESULT['message'])\nV32_RESULT\n"
    for name, source in (('core', standalone), ('evidence', payload), ('run', run)):
        compile(source, name, 'exec')
        cells.append({'cell_type': 'code', 'id': f'v32-{name}', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.splitlines(keepends=True)})
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'accelerator': 'gpu', 'isGpuEnabled': True, 'isInternetEnabled': False, 'language': 'python', 'sourceType': 'notebook'},
        'glc_v32': {'experiment': 'v32_asymmetric_rare_specialist_ensemble', 'source_sha256': hashes['v32'],
            'embedded_sources_sha256': {n: hashes[n] for n in names[:-1]}, 'control_sha256': CONTROL_HASH,
            'fresh_assessment': False, 'official_submission_made': False, 'max_total_hours': 10.75,
            'maximum_output_bytes': 16_000_000, 'required_input': ['geolifeclef-2025']}}}
    data = (json.dumps(notebook, ensure_ascii=False, indent=1)+'\n').encode()
    if len(data) >= 1_000_000:
        raise ValueError('Notebook exceeds Kaggle 1 MB source limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(json.dumps({'notebook': str(output), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}, indent=2))
    return notebook


if __name__ == '__main__':
    build()
