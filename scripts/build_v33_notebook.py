"""Build offline v33 with immutable, hash-verified v27-v32 sources and v32 CSV."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'notebooks/geolifeclef_v33_diverse_asymmetric_multisensor.ipynb'
CONTROL_HASH = '71c863d9cb05f3efc63ac54dd7faa05efd2fc137a11da9adcd2064c82c88dfee'


def pack_control():
    path = ROOT/'results/v32_kaggle_output/GLC25_PA_submission_v32.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_HASH:
        raise ValueError('Not the exact scored v32 CSV')
    frame = pd.read_csv(path)
    species = np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy', allow_pickle=False)
    if len(frame) != 14784 or not frame.surveyId.is_unique or len(species) != 5016:
        raise ValueError('Invalid v32 dimensions')
    lookup = pd.Index(species)
    lists = [lookup.get_indexer(np.fromstring(text, sep=' ', dtype=np.int64)) for text in frame.predictions]
    if any((row < 0).any() or not 8 <= len(row) <= 40 or len(row) != len(np.unique(row)) for row in lists):
        raise ValueError('Invalid v32 species lists')
    raw = np.array(list(map(len, lists)), np.uint8).tobytes()+np.concatenate(lists).astype('<u2').tobytes()
    packed = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    return base64.b64encode(packed).decode(), hashlib.sha256(packed).hexdigest(), hashlib.sha256(raw).hexdigest()


def build(output=OUTPUT):
    names = ('v27', 'v29', 'v30', 'v31', 'v32', 'v33')
    sources = {n: (ROOT/f'scripts/{n}_notebook_core.py').read_text(encoding='utf-8') for n in names}
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    manifest = json.loads((ROOT/'results/v32_kaggle_output/v32_manifest.json').read_text())
    expected = {**manifest['embedded_sources_sha256'], 'v32': manifest['source_sha256']}
    for name in names[:-1]:
        if hashes[name] != expected[name]:
            raise ValueError(f'Frozen scored source changed: {name}')
    control = pack_control()
    loader = 'import types as _types\n'
    for index, name in enumerate(names[:-1]):
        encoded = base64.b64encode(lzma.compress(sources[name].encode(), preset=9)).decode()
        loader += (f"_{name}_source = lzma.decompress(base64.b64decode({encoded!r}))\n"
                   f"assert hashlib.sha256(_{name}_source).hexdigest() == {hashes[name]!r}\n"
                   f"_{name} = _types.ModuleType('v33_frozen_{name}')\n")
        if index:
            old, symbol = names[index-1], ('legacy' if name == 'v29' else 'previous')
            statement = f'import scripts.{old}_notebook_core as {symbol}'
            assert sources[name].count(statement) == 1
            loader += (f"_{name}.__dict__[{symbol!r}] = _{old}\n"
                       f"_{name}_source = _{name}_source.replace({statement.encode()!r}, b'')\n")
        loader += f"exec(compile(_{name}_source, 'frozen-{name}', 'exec'), _{name}.__dict__)\ndel _{name}_source\n"
    loader += 'previous = _v32\ndel _v27, _v29, _v30, _v31, _v32\n'
    statement = 'import scripts.v32_notebook_core as previous'
    assert sources['v33'].count(statement) == 1
    standalone = sources['v33'].replace(statement, loader)
    payload = (f"V33_SOURCE_HASH = {hashes['v33']!r}\nFROZEN_SOURCE_HASHES = {dict((n, hashes[n]) for n in names[:-1])!r}\n"
        f"CONTROL_PAYLOAD_HASH = {control[1]!r}\nCONTROL_RAW_HASH = {control[2]!r}\nCONTROL_B64 = {control[0]!r}\n"
        "print({'source_sha256': V33_SOURCE_HASH, 'control': 'exact scored v32', 'fresh_holdout_remaining': 0})\n")
    intro = """# GeoLifeCLEF v33 — diverse asymmetric multisensor ensemble

Candidate improvement, **not established SOTA**. Best v32: **0.24225 public /
0.21514 private**. Winning private target: 0.23021. V32 completed in **5.7681 hours
on Tesla T4**, with 7.16 MB exported, and is the new immutable scored control.

## Run and submit

1. Attach **GeoLifeCLEF25 @ CVPR & LifeCLEF** (`geolifeclef-2025`), the competition input.
2. Select **GPU T4 x1**, restart, then **Run All**. Internet is not required.
3. Submit **only** `v33_export/GLC25_PA_submission_v33.csv`, and only when
   `eligible_for_submission` is `true`. Never submit ZIP, NPZ, diagnostics or DO_NOT_SUBMIT files.

## Evidence-driven change

On repeated spatial checks v32 gained 0.0016916 over matched v31. Its specialist-only
ablation retained 0.0010292 gain, versus 0.0001460 for seed-bag-only. The combination
was strongest. This favors diversifying the successful asymmetric-loss specialists.
It does not identify the separate causal effects of wider models, loss and rare head.

V33 adds **three** models: two new-seed replicas of the proven width-192 attention
specialists (20/24 epochs) and a **pyramid-raster + temporal-convolution specialist**
(width 192, 24 epochs). The convolution model uses the same clipped asymmetric loss
and a training-defined rare-species residual head. It adds architecture diversity
to the previously attention-only specialist group. All 5,016 species stay in the loss.
No external data or pretrained weights are used; all production fits use all PA rows.

The exact scored v32 CSV is embedded. Each test row keeps its original species count
and leading 60%; selected policies allow at most 2/4/8 tail replacements. Calibration
selects among 25 unchanged/attention-only/convolution-only/mixed policies. Individual
model diagnostics are reported separately and do not select members in this run.

## Checks and limitations

The frozen selected v32 recipe is refit inside two fixed 20-km buffered spatial folds.
Only the official-test CSV is byte-exact, not the development weights. Selection rows
calibrate probabilities; calibration rows select the policy; regression checks occur
after policy freeze. Both fold gains, block-bootstrap lower bound, country-macro gain
and gain outside Denmark/Netherlands must be positive before production and eligibility.
**All PA IDs were assessed before: these are repeated development checks, not a fresh
holdout or independent proof.** Production calibration transfers from smaller folds.
If rejected, the exact v32 control is returned under DO_NOT_SUBMIT; keep the scored v32.

## Budget and small output

V33 has **28 development fits and at most 3 production fits**. Its full T4 runtime
is unmeasured. The cooperative **10.75-hour guard** reserves headroom within 12 hours.
After fold 0, the remaining plan estimate includes its entire measured wall time
(all reference fits, inference, diagnostics), scaled with a 1.25x margin, plus
production's 1.5x training-speed estimate and overhead. A slow session can stop safely;
there is no unconditional runtime guarantee against stalled I/O or hardware.

Successful output is exactly **five files, <=16 MB total**: the eligible prediction
CSV, report JSON, manifest JSON, per-survey regression CSV and sampled calibration
NPZ. The NPZ now includes both aggregate experts AND all three individual model
rankings, improving next-iteration diagnosis. No large checkpoints or caches remain.
Failures revoke submission eligibility, including a late failure after report creation.
"""
    cells = [{'cell_type': 'markdown', 'id': 'v33-intro', 'metadata': {}, 'source': intro.splitlines(keepends=True)}]
    run = "V33_RESULT = run_v33(CONTROL_B64)\nprint(json.dumps(V33_RESULT, indent=2))\nprint(V33_RESULT['message'])\nV33_RESULT\n"
    for name, source in (('core', standalone), ('evidence', payload), ('run', run)):
        compile(source, name, 'exec')
        cells.append({'cell_type': 'code', 'id': f'v33-{name}', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.splitlines(keepends=True)})
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'accelerator': 'gpu', 'isGpuEnabled': True, 'isInternetEnabled': False, 'language': 'python', 'sourceType': 'notebook'},
        'glc_v33': {'experiment': 'v33_diverse_asymmetric_multisensor', 'source_sha256': hashes['v33'],
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
