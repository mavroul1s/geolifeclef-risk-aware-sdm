"""Replay the RELEASED v34 decoder on consumed archived calibration predictions.

This checks implementation and output size; it is not a fresh validation run.
No policy chosen here is hard-coded into the Kaggle notebook.
"""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scripts import v34_notebook_core as core

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'results/v32_kaggle_output/calibration_top128_v32.npz'
    with np.load(source,allow_pickle=False) as archive:
        ids, folds, countries, species = [archive[k] for k in ('survey_id','fold','country','species_ids')]
        base = [r[r != 65535].astype(int).tolist() for r in archive['reference_columns_padded65535']]
        experts = [{'rank':archive['ranked_species_columns'][:,i], 'value':archive['probability'][:,i]} for i in range(2)]
    pairs = pd.read_csv(ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv',
        usecols=['surveyId','lat','lon','country','speciesId'])
    rows = pairs.drop_duplicates('surveyId').reset_index(drop=True)
    labels = np.zeros((len(rows),len(species)),np.uint8)
    row_index = pd.Index(rows.surveyId).get_indexer(pairs.surveyId)
    col_index = pd.Index(species).get_indexer(pairs.speciesId)
    if min(row_index.min(),col_index.min()) < 0:
        raise ValueError('Unknown label or survey')
    labels[row_index,col_index] = 1
    splits, manifests = core.v31.previous.make_splits(rows)
    base = core.previous.decode(base,*experts,core.FROZEN_V32_POLICY)
    index = pd.Index(rows.surveyId).get_indexer(ids)
    bundles,old = [],core.legacy.score_prediction_lists(labels[index],base)
    scores = {p['id']:np.empty(len(base)) for p in core.POLICIES}
    for f,split in enumerate(splits):
        pos = np.flatnonzero(folds == f)
        assert set(index[pos]).issubset(split['calibration'])
        graph = core.CommunityGraph().fit(labels,split['training'])
        data = {'base':[base[i] for i in pos], **{k:{s:v[pos] for s,v in e.items()} for k,e in zip(('bag','specialist'),experts)}}
        data['community'] = graph.predict_rank(data['base'],data['bag'],data['specialist'])
        for policy in core.POLICIES:
            scores[policy['id']][pos] = core.legacy.score_prediction_lists(labels[index[pos]],core.decode(data,policy))
        bundles.append({'number':f,'split':{'calibration':index[pos]},'data':{'calibration':data},'graph':graph.record})
    trials = [{**p, **core.v31.geographic_gain(scores[p['id']]-old,countries,ids),
        'fold_gains':[float((scores[p['id']]-old)[folds == f].mean()) for f in (0,1)]} for p in core.POLICIES]
    with tempfile.TemporaryDirectory(prefix='v34-size-',dir=ROOT/'artifacts') as directory:
        path = Path(directory)/'calibration.npz'
        core.save_compact(path,bundles,rows,SimpleNamespace(labels=labels,species_ids=species),core.POLICIES[0])
        compact_bytes = path.stat().st_size
    report = {'scope':'released implementation replay on consumed calibration; not fresh evidence or Kaggle score prediction',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'core_sha256':hashlib.sha256((ROOT/'scripts/v34_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest(),
        'rows':len(ids),'unique_surveys':len(set(ids)),'graph_training':[b['graph'] for b in bundles],
        'fold_manifests':manifests,'compact_six_expert_npz_bytes':compact_bytes,
        'policy_is_not_selected_for_notebook_by_this_script':True,'policies':trials}
    core.legacy.save_json(ROOT/'results/v34_archived_replay.json',report)
    print(json.dumps({'compact_npz_bytes':compact_bytes,'top_exploratory_policies':sorted(trials,key=lambda p:p['robust_gain'],reverse=True)[:3]},indent=2))


if __name__ == '__main__':
    main()
