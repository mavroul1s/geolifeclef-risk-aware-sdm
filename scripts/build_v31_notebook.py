"""Build the offline v31 notebook; all frozen sources and v29 CSV are hash checked."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

from scripts.build_v30_notebook import pack_control, CONTROL_HASH

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'notebooks/geolifeclef_v31_full_data_bagged_habitat_residual.ipynb'


def build(output=OUTPUT):
    names = ('v27', 'v29', 'v30', 'v31')
    sources = {n: (ROOT/f'scripts/{n}_notebook_core.py').read_text(encoding='utf-8') for n in names}
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    manifest = json.loads((ROOT/'results/v29_kaggle_output/v29_manifest.json').read_text())
    v30_manifest = json.loads((ROOT/'results/v30_kaggle_output/v30_manifest.json').read_text())
    assert hashes['v27'] == manifest['embedded_legacy_sha256'], 'Frozen reader source changed'
    assert hashes['v29'] == manifest['source_sha256'], 'Scored v29 source changed'
    assert hashes['v30'] == v30_manifest['source_sha256'], 'Scored v30 source changed'
    control = pack_control(ROOT/'results/v29_kaggle_output/GLC25_PA_submission_v29.csv',
                           ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    loader = 'import types as _types\n'
    for name in names[:-1]:
        encoded = base64.b64encode(lzma.compress(sources[name].encode(), preset=9)).decode()
        loader += (f"_{name}_source = lzma.decompress(base64.b64decode({encoded!r}))\n"
                   f"assert hashlib.sha256(_{name}_source).hexdigest() == {hashes[name]!r}\n"
                   f"_{name} = _types.ModuleType('v31_frozen_{name}')\n")
        if name != 'v27':
            old, symbol = ('v27', 'legacy') if name == 'v29' else ('v29', 'previous')
            statement = f'import scripts.{old}_notebook_core as {symbol}'
            assert sources[name].count(statement) == 1
            loader += (f"_{name}.__dict__[{symbol!r}] = _{old}\n"
                       f"_{name}_source = _{name}_source.replace({statement.encode()!r}, b'')\n")
        loader += f"exec(compile(_{name}_source, 'frozen-{name}', 'exec'), _{name}.__dict__)\ndel _{name}_source\n"
    loader += 'previous = _v30\ndel _v27, _v29, _v30\n'
    statement = 'import scripts.v30_notebook_core as previous'
    assert sources['v31'].count(statement) == 1
    standalone = sources['v31'].replace(statement, loader)
    payload = (f"V31_SOURCE_HASH = {hashes['v31']!r}\nV30_SOURCE_HASH = {hashes['v30']!r}\n"
        f"V29_SOURCE_HASH = {hashes['v29']!r}\nLEGACY_SOURCE_HASH = {hashes['v27']!r}\n"
        f"CONTROL_PAYLOAD_HASH = {control[1]!r}\nCONTROL_RAW_HASH = {control[2]!r}\nCONTROL_B64 = {control[0]!r}\n"
        "print({'source_sha256': V31_SOURCE_HASH, 'control': 'exact scored v29', 'fresh_holdout_remaining': 0})\n")
    intro = """# GeoLifeCLEF v31 — full-data bagging + environmental analogues

**An unscored candidate, not a demonstrated SOTA result.** V30 regressed to
0.23451 public / 0.20942 private. V29 remains the best: 0.23900 / 0.21093.

This version freezes the exact scored v29 CSV, preserves **every row's species count**
and its leading 60%, and allows at most 2 or 4 tail replacements. It adds independently
seeded replicas of the three v29 neural models and an environmental-neighbour expert.
Production uses **all 88,987 PA surveys**, with no v30 calibration-anchor exclusion.
The habitat metric uses official environmental features, not coordinates, country or IDs.

## Run and submit

1. Attach **GeoLifeCLEF25 @ CVPR & LifeCLEF** (`geolifeclef-2025`). No extra dataset.
2. Select **GPU T4 x1**, restart the session, then **Run All**. Internet is not needed.
3. Submit **only** `v31_export/GLC25_PA_submission_v31.csv`, and only if the final
   `eligible_for_submission` is `true`. Never submit the ZIP, reports, diagnostic CSV,
   NPZ, or anything named `DO_NOT_SUBMIT`.

If a gate fails, the notebook returns the unchanged best-v29 prediction file explicitly
marked **DO_NOT_SUBMIT**, avoiding another identical submission.

## Evidence and limits

The 15 ranking policies are selected on calibration partitions, then checked on the
same two geographically buffered development folds as v30. Positive pooled gain is
insufficient: both folds, equal-country macro gain (countries with >=30 surveys),
gain outside Denmark/Netherlands, and the spatial-bootstrap lower bound must pass.
**All PA IDs were assessed in earlier experiments: these are repeated checks, not a
fresh independent audit.** Geographic gates are a response to v30 diagnostics, so they
too are development choices. The reference is a fixed-epoch v29 recipe refit; only
the official-test control CSV is byte-exact. Production replicas reuse the scored
v29 calibration coefficients across seeds, an explicit transfer assumption.

The original v29 production epoch schedule is frozen at 12/15/18 for both development
groups and production. There are 12 development fits and at most 3 production fits.
Whole-production admission is based on measured development speed. A **10.75-hour
cooperative guard** leaves headroom within the requested 12-hour session, but full
v31 T4 runtime is unmeasured: a slow run may stop safely instead of completing.

The successful export contains exactly five files, capped at **16 MB total**. Large
temporary caches and checkpoints are removed. No external data or pretrained weights.
"""
    cells = [{'cell_type': 'markdown', 'id': 'v31-intro', 'metadata': {}, 'source': intro.splitlines(keepends=True)}]
    run = "V31_RESULT = run_v31(CONTROL_B64)\nprint(json.dumps(V31_RESULT, indent=2))\nprint(V31_RESULT['message'])\nV31_RESULT\n"
    for name, source in (('core', standalone), ('evidence', payload), ('run', run)):
        compile(source, name, 'exec')
        cells.append({'cell_type': 'code', 'id': f'v31-{name}', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.splitlines(keepends=True)})
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'accelerator': 'gpu', 'isGpuEnabled': True, 'isInternetEnabled': False, 'language': 'python', 'sourceType': 'notebook'},
        'glc_v31': {'experiment': 'v31_full_data_bagged_habitat_residual', 'source_sha256': hashes['v31'],
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
