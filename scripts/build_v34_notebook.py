"""Build offline v34 with immutable, hash-verified v27-v32 sources and v32 CSV."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'notebooks/geolifeclef_v34_cooccurrence_community_reranking.ipynb'
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
    names = ('v27', 'v29', 'v30', 'v31', 'v32', 'v34')
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
                   f"_{name} = _types.ModuleType('v34_frozen_{name}')\n")
        if index:
            old, symbol = names[index-1], ('legacy' if name == 'v29' else 'previous')
            statement = f'import scripts.{old}_notebook_core as {symbol}'
            assert sources[name].count(statement) == 1
            loader += (f"_{name}.__dict__[{symbol!r}] = _{old}\n"
                       f"_{name}_source = _{name}_source.replace({statement.encode()!r}, b'')\n")
        loader += f"exec(compile(_{name}_source, 'frozen-{name}', 'exec'), _{name}.__dict__)\ndel _{name}_source\n"
    loader += 'previous = _v32\ndel _v27, _v29, _v30, _v31, _v32\n'
    statement = 'import scripts.v32_notebook_core as previous'
    assert sources['v34'].count(statement) == 1
    standalone = sources['v34'].replace(statement, loader)
    payload = (f"V34_SOURCE_HASH = {hashes['v34']!r}\nFROZEN_SOURCE_HASHES = {dict((n, hashes[n]) for n in names[:-1])!r}\n"
        f"CONTROL_PAYLOAD_HASH = {control[1]!r}\nCONTROL_RAW_HASH = {control[2]!r}\nCONTROL_B64 = {control[0]!r}\n"
        "print({'source_sha256': V34_SOURCE_HASH, 'control': 'exact scored v32', 'fresh_holdout_remaining': 0})\n")
    intro = """# GeoLifeCLEF v34 — co-occurrence community reranking

Candidate improvement, **not established SOTA**. Best v32: **0.24225 public /
0.21514 private**. V33 was rejected and returned the exact v32 CSV; its equal
Kaggle score was not a new model result.

## Run and submit

1. Attach **GeoLifeCLEF25 @ CVPR & LifeCLEF** (`geolifeclef-2025`), the official competition input.
2. Select **GPU T4 x1**, restart, then **Run All**. Internet and extra datasets are not required.
3. Submit **only** `v34_export/GLC25_PA_submission_v34.csv`, and only when
   `eligible_for_submission` is `true`. Never submit ZIP, NPZ or diagnostic CSVs.
4. If rejected, output contains **NO_SUBMISSION.json**, not a duplicate submission.
   Keep the already scored v32. There is no need to submit that identical file again.

## Evidence-driven change

The new convolution specialist in v33 consistently hurt development F1. Instead
of more seeds or larger networks, v34 refits the **unchanged v32 recipe** and learns
species co-occurrence from each spatial training partition. A 100-observation prior
shrinks conditional frequencies toward marginal prevalence. Clipped log association
reweights the average bag/specialist top128 scores using at most 12 protected species
as context. Self-edges are zero. No query labels, external data or weights are used.

The test control is the exact scored v32 CSV. Every row keeps its count and leading
60%; at most 2/4 tail replacements are allowed. The 17-policy calibration search
includes the unchanged control AND zero-co-occurrence probability reranking.
The latter separates graph effects from merely reusing probability magnitudes.

Before building this notebook, a bounded learned-F1 editor failed a local blocked
pilot and was rejected. The graph hypothesis had a small positive signal in both
views of the archived sampled calibration: the strongest pilot policy gained
0.0004093 per deduplicated survey, with positive country-macro and outside-DK/NL
gains. **This is exploratory reused evidence, not fresh validation or an expected
Kaggle gain.** All pilot results, including negative ones, are recorded in the repo.

## Checks and limitations

Two fixed 20-km-buffered spatial folds refit the frozen v32 recipe. The graph reads
only training labels. Selection calibrates neural probabilities; calibration selects
the policy; assessment is read after policy freeze. Positive gains in both folds,
country-macro, outside Denmark/Netherlands and a block-bootstrap lower bound are
required. The no-graph assessment ablation is diagnostic, not another policy search.

All **88,987 PA IDs were assessed previously**. These repeated development checks
are not independent evidence. Co-occurrence can reflect sampling/geography, and
incorrect predicted anchors can propagate errors. Production trains all five v32
members on all PA rows with the frozen scored v32 calibration coefficients. Only
the control CSV, not refitted model weights, is byte-exact.

## Runtime and compact output

**22 development neural fits + at most 5 production neural fits**, the same count
as v32 (5.7681 hours on T4), plus inexpensive graph operations. Full v34 T4 runtime
is unmeasured. A cooperative **10.75-hour guard**, whole-fold admission estimate
and production reserve provide headroom within 12 hours; this is not an absolute
guarantee against stalled I/O or hardware.

Successful export is exactly **five files, <=16 MB total**: an eligible submission
CSV OR NO_SUBMISSION.json, report, manifest, per-survey diagnostics and compact
sampled rankings. Caches/checkpoints are removed. Export failures revoke eligibility.

Label-dependency modeling is established in multi-label classification, e.g.
[Chen et al., CVPR 2019](https://arxiv.org/abs/1904.03582). This is a bounded
statistical reranker, **not** an implementation of their GCN or a novel-method claim.
"""
    cells = [{'cell_type': 'markdown', 'id': 'v34-intro', 'metadata': {}, 'source': intro.splitlines(keepends=True)}]
    run = "V34_RESULT = run_v34(CONTROL_B64)\nprint(json.dumps(V34_RESULT, indent=2))\nprint(V34_RESULT['message'])\nV34_RESULT\n"
    for name, source in (('core', standalone), ('evidence', payload), ('run', run)):
        compile(source, name, 'exec')
        cells.append({'cell_type': 'code', 'id': f'v34-{name}', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.splitlines(keepends=True)})
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'accelerator': 'gpu', 'isGpuEnabled': True, 'isInternetEnabled': False, 'language': 'python', 'sourceType': 'notebook'},
        'glc_v34': {'experiment': 'v34_cooccurrence_community_reranking', 'source_sha256': hashes['v34'],
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
