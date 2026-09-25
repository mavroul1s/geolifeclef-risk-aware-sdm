"""Assemble a verified, offline, single-file v30 Kaggle deliverable."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks/geolifeclef_v30_calibrated_multiscale_attention.ipynb'
CONTROL_HASH = '9ec7afb25eb1651b161b0f8c6ecaa64fd553d17dd24b4808b1f07446b85f8194'


def pack_control(path, species_path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_HASH:
        raise ValueError('Not the exact scored v29 CSV')
    frame = pd.read_csv(path)
    species = np.load(species_path, allow_pickle=False)
    if len(frame) != 14784 or not frame.surveyId.is_unique or len(species) != 5016:
        raise ValueError('Invalid v29 dimensions')
    lookup = pd.Index(species)
    lists = [lookup.get_indexer(np.fromstring(text, sep=' ', dtype=np.int64)) for text in frame.predictions]
    if any((row < 0).any() or not 8 <= len(row) <= 40 or len(row) != len(np.unique(row)) for row in lists):
        raise ValueError('Invalid v29 species lists')
    raw = np.array(list(map(len, lists)), np.uint8).tobytes() + np.concatenate(lists).astype('<u2').tobytes()
    packed = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    return base64.b64encode(packed).decode(), hashlib.sha256(packed).hexdigest(), hashlib.sha256(raw).hexdigest()


def make_notebook(core, previous, legacy, control):
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    encode = lambda s: base64.b64encode(lzma.compress(s.encode(), preset=9)).decode()
    if previous.count('import scripts.v27_notebook_core as legacy') != 1 or core.count('import scripts.v29_notebook_core as previous') != 1:
        raise ValueError('Unexpected repository import contract')
    loader = ("import types as _types\n"
        f"LEGACY_SOURCE_HASH = {sha(legacy)!r}\nV29_SOURCE_HASH = {sha(previous)!r}\n"
        f"_legacy_source = lzma.decompress(base64.b64decode({encode(legacy)!r}))\n"
        f"_previous_source = lzma.decompress(base64.b64decode({encode(previous)!r}))\n"
        "assert hashlib.sha256(_legacy_source).hexdigest() == LEGACY_SOURCE_HASH\n"
        "assert hashlib.sha256(_previous_source).hexdigest() == V29_SOURCE_HASH\n"
        "_legacy_module = _types.ModuleType('v30_frozen_readers')\n"
        "exec(compile(_legacy_source, 'v30-frozen-readers', 'exec'), _legacy_module.__dict__)\n"
        "previous = _types.ModuleType('v30_frozen_v29')\n"
        "previous.__dict__['legacy'] = _legacy_module\n"
        "_previous_source = _previous_source.replace(b'import scripts.v27_notebook_core as legacy', b'')\n"
        "exec(compile(_previous_source, 'v30-frozen-v29', 'exec'), previous.__dict__)\n"
        "del _legacy_source, _previous_source, _legacy_module\n")
    standalone = core.replace('import scripts.v29_notebook_core as previous', loader)
    payload = (f"V30_SOURCE_HASH = {sha(core)!r}\nCONTROL_PAYLOAD_HASH = {control[1]!r}\n"
               f"CONTROL_RAW_HASH = {control[2]!r}\nCONTROL_B64 = {control[0]!r}\n"
               "print({'source_sha256': V30_SOURCE_HASH, 'control': 'exact scored v29', 'fresh_holdout_remaining': 0})\n")
    intro = """# GeoLifeCLEF v30 — calibrated multiscale attention

Candidate improvement, **not a demonstrated SOTA result**. The best scored v29 is
0.23900 public / 0.21093 private, and completed in 3.014 hours on a Tesla T4.
V30 retains its exact CSV as control, compares attention-only ensembles with a new
two-scale image/temporal attention model, and learns adaptive species-list length.
Final models keep spatially separated calibration anchors out of weight training;
their probability/count calibration is fitted on the exact production models.

## Run and submit

1. Attach only **GeoLifeCLEF25 @ CVPR & LifeCLEF** (`geolifeclef-2025`).
2. Enable **GPU T4 x1**, restart the session, then **Run All**. No Internet or extra dataset.
3. Submit **only** `v30_export/GLC25_PA_submission_v30.csv`, and only when the final
   `eligible_for_submission` is `true`. Do not submit a ZIP, JSON, NPZ, diagnostic CSV,
   or a file containing `DO_NOT_SUBMIT` in its name.

All 88,987 PA survey IDs have been assessed in earlier experiments. V30 therefore uses
two repeated spatial development folds, NOT a fresh independent holdout. Policy selection
and the within-run regression check use disjoint blocks with 20km training buffers.
Production calibration is also excluded by block and 20km buffer from weight training.
The deployed models consequently use fewer PA training rows than v29, a real tradeoff.

The 10.75-hour cooperative budget uses batch-level checks and measured refit admission.
V30's full GPU runtime has not been measured; it can stop safely if the session is too slow.
Temporary caches/checkpoints are deleted. Exactly five compact files, capped at 16 MB total,
remain: prediction CSV, regression CSV, report, manifest, and a compressed top-64 calibration
diagnostic file. Only the eligible `GLC25_PA_submission_v30.csv` is for Kaggle submission.
"""
    run = "V30_RESULT = run_v30(CONTROL_B64)\nprint(json.dumps(V30_RESULT, indent=2))\nprint(V30_RESULT['message'])\nV30_RESULT\n"
    cells = [{'cell_type':'markdown','id':'v30-intro','metadata':{},'source':intro.splitlines(keepends=True)}]
    for name, source in (('core',standalone),('evidence',payload),('run',run)):
        compile(source, name, 'exec')
        cells.append({'cell_type':'code','id':f'v30-{name}','metadata':{},'execution_count':None,
                      'outputs':[],'source':source.splitlines(keepends=True)})
    return {'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'},
        'kaggle':{'accelerator':'gpu','isGpuEnabled':True,'isInternetEnabled':False,'language':'python','sourceType':'notebook'},
        'glc_v30':{'experiment':'v30_calibrated_multiscale_attention','source_sha256':sha(core),
            'v29_source_sha256':sha(previous),'legacy_source_sha256':sha(legacy),'control_sha256':CONTROL_HASH,
            'fresh_assessment':False,'official_submission_made':False,'max_total_hours':10.75,
            'required_input':['geolifeclef-2025'],'maximum_output_bytes':16000000}}}


def build(output=OUTPUT):
    core = (ROOT / 'scripts/v30_notebook_core.py').read_text(encoding='utf-8')
    previous = (ROOT / 'scripts/v29_notebook_core.py').read_text(encoding='utf-8')
    legacy = (ROOT / 'scripts/v27_notebook_core.py').read_text(encoding='utf-8')
    manifest = json.loads((ROOT / 'results/v29_kaggle_output/v29_manifest.json').read_text())
    if hashlib.sha256(previous.encode()).hexdigest() != manifest['source_sha256']:
        raise ValueError('The scored v29 source was changed')
    control = pack_control(ROOT / 'results/v29_kaggle_output/GLC25_PA_submission_v29.csv',
                            ROOT / 'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    notebook = make_notebook(core, previous, legacy, control)
    data = (json.dumps(notebook, ensure_ascii=False, indent=1)+'\n').encode()
    if len(data) >= 1_000_000:
        raise ValueError('Kaggle source exceeds 1 MB')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(json.dumps({'notebook':str(output),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}, indent=2))
    return notebook


if __name__ == '__main__':
    build()
