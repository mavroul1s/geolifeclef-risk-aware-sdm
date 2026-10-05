"""Build an offline v35 notebook with authenticated cached spatial benchmarks."""
import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb'


def pack_reference():
    folder = ROOT/'results/v34_kaggle_output'
    manifest = json.loads((folder/'v34_manifest.json').read_text())
    if set(manifest['outputs']) != {'GLC25_PA_submission_v34.csv','calibration_top128_v34.npz',
                                    'regression_per_survey_v34.csv','v34_report.json'}:
        raise ValueError('Unexpected archived manifest paths')
    for name, digest in manifest['outputs'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Archived reference output changed: {name}')
    source = (ROOT/'scripts/v34_notebook_core.py').read_text(encoding='utf-8')
    if hashlib.sha256(source.encode()).hexdigest() != manifest['source_sha256']:
        raise ValueError('The cached benchmark no longer matches its audited source')
    archive = np.load(folder/'calibration_top128_v34.npz', allow_pickle=False)
    regression = pd.read_csv(folder/'regression_per_survey_v34.csv')
    if len(regression) != 31570 or not regression.surveyId.is_unique:
        raise ValueError('Invalid cached assessment')
    species = archive['species_ids'].astype('<i8')
    pairs = pd.read_csv(ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv', usecols=['surveyId','speciesId'])
    ids = np.sort(pairs.surveyId.unique())
    row = pd.Index(ids).get_indexer(pairs.surveyId)
    col = pd.Index(species).get_indexer(pairs.speciesId)
    if min(row.min(), col.min()) < 0:
        raise ValueError('Unknown training labels')
    labels = sparse.csr_matrix((np.ones(len(row), np.uint8), (row, col)), shape=(len(ids),len(species)))
    labels.data[:] = 1
    digest = hashlib.sha256()
    for start in range(0, len(ids), 1024):
        digest.update(labels[start:start+1024].toarray().tobytes())
    reference = {'schema': 1, 'species_sha256': hashlib.sha256(species.tobytes()).hexdigest(),
        'train_metadata_sha256': hashlib.sha256((ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv').read_bytes()).hexdigest(),
        'labels_sha256': digest.hexdigest(), 'folds': [],
        'provenance': {'reference': 'matched unchanged v32 recipe predictions from the completed v34 run, NOT v34 graph predictions',
            'v34_manifest_sha256': hashlib.sha256((folder/'v34_manifest.json').read_bytes()).hexdigest(),
            'v34_source_sha256': manifest['source_sha256'], 'files': manifest['outputs'],
            'fresh_assessment': False}}
    for fold in range(2):
        positions = np.flatnonzero(archive['fold'] == fold)
        if len(positions) != 1500:
            raise ValueError('Unexpected cached calibration size')
        predictions, f1 = [], []
        for i in positions:
            pred = archive['reference_columns_padded65535'][i]
            pred = pred[pred != 65535].astype(int).tolist()
            a, b = archive['true_offsets'][i:i+2]
            truth = set(archive['true_species_columns'][a:b].astype(int))
            predictions.append(pred); f1.append(2*len(set(pred)&truth)/(len(pred)+len(truth)))
        frame = regression[regression.fold == fold]
        reference['folds'].append({'id_hashes': manifest['splits'][fold]['id_hashes'],
            'calibration': {'ids': archive['survey_id'][positions].astype(int).tolist(), 'predictions': predictions, 'f1': f1},
            'assessment': {'ids': frame.surveyId.astype(int).tolist(), 'f1': frame.matched_v32_f1.tolist()}})
    raw = json.dumps(reference, separators=(',',':'), ensure_ascii=True).encode()
    packed = lzma.compress(raw, preset=9)
    return base64.b64encode(packed).decode(), hashlib.sha256(packed).hexdigest(), hashlib.sha256(raw).hexdigest()


def build(output=OUTPUT):
    names = ('v27','v29','v30','v31','v32','v35')
    sources = {n:(ROOT/f'scripts/{n}_notebook_core.py').read_text(encoding='utf-8') for n in names}
    hashes = {n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
    manifest = json.loads((ROOT/'results/v32_kaggle_output/v32_manifest.json').read_text())
    expected = {**manifest['embedded_sources_sha256'],'v32':manifest['source_sha256']}
    for name in names[:-1]:
        if hashes[name] != expected[name]:
            raise ValueError(f'Frozen source changed: {name}')
    reference = pack_reference()
    loader = 'import types as _types\n'
    for index,name in enumerate(names[:-1]):
        encoded = base64.b64encode(lzma.compress(sources[name].encode(),preset=9)).decode()
        loader += (f"_{name}_source = lzma.decompress(base64.b64decode({encoded!r}))\n"
                   f"assert hashlib.sha256(_{name}_source).hexdigest() == {hashes[name]!r}\n"
                   f"_{name} = _types.ModuleType('v35_frozen_{name}')\n")
        if index:
            old,symbol = names[index-1],('legacy' if name == 'v29' else 'previous')
            statement = f'import scripts.{old}_notebook_core as {symbol}'
            assert sources[name].count(statement) == 1
            loader += (f"_{name}.__dict__[{symbol!r}] = _{old}\n"
                       f"_{name}_source = _{name}_source.replace({statement.encode()!r}, b'')\n")
        loader += f"exec(compile(_{name}_source, 'frozen-{name}', 'exec'), _{name}.__dict__)\ndel _{name}_source\n"
    loader += 'previous = _v32\ndel _v27, _v29, _v30, _v31, _v32\n'
    statement = 'import scripts.v32_notebook_core as previous'
    assert sources['v35'].count(statement) == 1
    standalone = sources['v35'].replace(statement,loader)
    payload = (f"V35_SOURCE_HASH = {hashes['v35']!r}\nFROZEN_SOURCE_HASHES = {dict((n,hashes[n]) for n in names[:-1])!r}\n"
        f"REFERENCE_PAYLOAD_HASH = {reference[1]!r}\nREFERENCE_RAW_HASH = {reference[2]!r}\nREFERENCE_B64 = {reference[0]!r}\n"
        "print({'source_sha256':V35_SOURCE_HASH,'cached_reference':'unchanged v32 recipe from v34 run','fresh_holdout_remaining':0})\n")
    intro = """# GeoLifeCLEF v35 — full-set balanced attention

**Candidate, not proven improvement or SOTA.** V34 regressed to 0.24203 public /
0.21483 private. Keep the best v32 (0.24225 / 0.21514) until a new actual score wins.

## Run

1. Attach **GeoLifeCLEF25 @ CVPR & LifeCLEF** (`geolifeclef-2025`), official input only.
2. Select **T4 x1**, restart, **Run All**. Internet off is fine; no extra dataset.
3. Submit **only** `v35_export/GLC25_PA_submission_v35.csv` when
   `eligible_for_submission` is true. If `NO_SUBMISSION.json` appears, do not submit
   any diagnostic file; retain v32. Nothing is uploaded/submitted automatically.

## The change

V34 passed development by only +0.000430 F1; its bootstrap lower bound was
0.000000427. That weak reused signal did not transfer. V35 abandons graph reranking
and the permanent v29/v32 count/prefix lock. It predicts complete species sets.

Three attention models train for up to **48 epochs**. EMA checkpoints at 12/24/36/48
are selected using selection-only ranking F1 (with country/outside-core components).
The first model extends the successful geo-ASL recipe. Two new sensor-balanced
models separately pool Sentinel, Landsat, climate, environmental and static tokens,
then fuse them. They use geo/no-geo diversity, a training-defined rare residual
head, and a small per-survey normalized positive-ranking auxiliary objective.
All 5,016 species remain in the classifier/loss. No external pretrained weights.

Calibration chooses one of 15 probability-ensemble/count policies: equal,
ecology-lean or sensor-balanced pair, crossed with 3 expected-F1-surrogate count
scales or fixed top18/top24. Counts remain within 8..40 but are no longer inherited
from v29. Checkpoint choice and probability calibration use selection rows only.
Production uses all PA rows; selected epoch counts retain the development
learning-rate schedule prefix rather than silently compressing it.

## Spend the budget on new models

The historical matched-v32 development scores are embedded as a small verified
cache. Before trusting it, the notebook checks **official training metadata,
all four role ID hashes in both folds, all PA labels and species order**, and
recomputes cached calibration F1.
No old reference neural fits are repeated: **6 new development fits + at most
3 production fits**, not 22 old-reference fits before the new work.

This cache contains PA development results, not test labels or pretrained models.
The reference is the unchanged v32 recipe refit during v34, not the v34 graph result.
Calibration comparison uses the 1,500 saved views per fold (some survey IDs repeat
across folds and are deduplicated for country-support checks). Assessment covers
the complete 31,570 historical spatial-check surveys.

## Stricter promotion, honest failure output

Require calibration and assessment gain >=0.001, outside-DK/NL gain >=0.0005,
positive gains in both folds and country-macro, and assessment bootstrap lower
bound >=0.00025. These margins were introduced **after observing v34's failure**;
they are safeguards, not a new independent statistical guarantee.
If rejected, report the actual attempted model's scores rather than hiding them
behind an identical control with zero delta.

All 88,987 PA IDs have already been assessed. Countries poorly represented or
absent from PA remain a major limitation (especially Ukraine/UK). These checks
are **repeated development**, not a fresh holdout. Full-set replacement can regress
even when these gates pass. The pooling/loss/schedule/decoder changes are a joint
hypothesis, not a completed causal ablation or exact Bayes-optimal F1 algorithm.

## Runtime/output

Cooperative **10.75-hour guard**, whole remaining-plan admission with safety
margins, GPU preflight before loading predictors. Full v35 T4 duration is unmeasured;
there is no absolute 12-hour guarantee against stalled I/O/hardware.
Exactly five final files, <=16 MB: eligible CSV OR NO_SUBMISSION.json, report,
manifest, per-survey diagnostic CSV, and sampled individual-model top128 scores
with full probability mass and truth for both calibration and assessment.
Temporary feature caches/checkpoints are removed. Failures revoke eligibility.

The distinction between marginal-probability ranking and F1-optimal set decisions
is discussed by [Dembczynski et al., ICML 2013](https://proceedings.mlr.press/v28/dembczynski13.html).
Our count decoder is a ratio-of-expectations surrogate, not their exact algorithm.
"""
    cells = [{'cell_type':'markdown','id':'v35-intro','metadata':{},'source':intro.splitlines(keepends=True)}]
    run = "V35_RESULT = run_v35(REFERENCE_B64)\nprint(json.dumps(V35_RESULT,indent=2))\nprint(V35_RESULT['message'])\nV35_RESULT\n"
    for name,source in (('core',standalone),('evidence',payload),('run',run)):
        compile(source,name,'exec')
        cells.append({'cell_type':'code','id':f'v35-{name}','metadata':{},'execution_count':None,'outputs':[],
                      'source':source.splitlines(keepends=True)})
    notebook = {'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},
        'kaggle':{'accelerator':'gpu','isGpuEnabled':True,'isInternetEnabled':False,'language':'python','sourceType':'notebook'},
        'glc_v35':{'experiment':'v35_full_set_balanced_attention','source_sha256':hashes['v35'],
            'embedded_sources_sha256':{n:hashes[n] for n in names[:-1]},'reference_sha256':reference[1],
            'fresh_assessment':False,'official_submission_made':False,'max_total_hours':10.75,
            'maximum_output_bytes':16_000_000,'required_input':['geolifeclef-2025']}}}
    data = (json.dumps(notebook,ensure_ascii=False,indent=1)+'\n').encode()
    if len(data) >= 1_000_000:
        raise ValueError('Notebook exceeds 1MB source limit')
    output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(data)
    print(json.dumps({'notebook':str(output),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},indent=2))
    return notebook


if __name__ == '__main__':
    build()
