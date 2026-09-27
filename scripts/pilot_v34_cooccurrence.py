"""Exploratory candidate-label co-occurrence correction on archived calibration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scripts import pilot_v34_editor_core as core

ROOT = Path(__file__).resolve().parents[1]


def fit_graph(labels):
    y = sparse.csr_matrix(labels, dtype=np.float32)
    count = np.asarray(y.sum(0)).ravel()
    prior = (count+.5)/(len(labels)+1.)
    pairs = (y.T@y).toarray()
    conditional = (pairs+100.*prior[None])/(count[:, None]+100.)
    lift = np.clip(np.log(conditional/prior[None]), -2., 2.)
    np.fill_diagonal(lift, 0.)
    return lift


def main():
    archive = np.load(ROOT/'results/v32_kaggle_output/calibration_top128_v32.npz', allow_pickle=False)
    pairs = pd.read_csv(ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv',
        usecols=['surveyId','lat','lon','country','speciesId'])
    rows = pairs.drop_duplicates('surveyId').reset_index(drop=True)
    labels = np.zeros((len(rows), len(archive['species_ids'])), np.uint8)
    labels[pd.Index(rows.surveyId).get_indexer(pairs.surveyId), pd.Index(archive['species_ids']).get_indexer(pairs.speciesId)] = 1
    splits, _ = core.v31.previous.make_splits(rows)
    base = [r[r != 65535].astype(int).tolist() for r in archive['reference_columns_padded65535']]
    bag, specialist = [{'rank': archive['ranked_species_columns'][:, i], 'value': archive['probability'][:, i]} for i in range(2)]
    base = core.previous.decode(base, bag, specialist, core.FROZEN_V32_POLICY)
    index = pd.Index(rows.surveyId).get_indexer(archive['survey_id'])
    old = core.legacy.score_prediction_lists(labels[index], base)
    scores = {}
    for fold, split in enumerate(splits):
        print('graph fold', fold, flush=True)
        graph = fit_graph(labels[split['training']])
        positions = np.flatnonzero(archive['fold'] == fold)
        assert set(index[positions]).issubset(set(split['calibration']))
        experts = {alpha: {'rank': [], 'value': []} for alpha in (0., .25, .5, 1.)}
        for i in positions:
            p = np.full(labels.shape[1], 1e-7, np.float32)
            for e in (bag,specialist):
                p[e['rank'][i]] += e['value'][i].astype(np.float32)/2.
            anchors = base[i][:min(12, int(np.ceil(.6*len(base[i]))))]
            weight = p[anchors]; weight /= weight.sum()
            context = weight@graph[anchors]
            for alpha in experts:
                corrected = np.log(p)+alpha*context
                rank = np.argsort(-corrected)[:128]
                experts[alpha]['rank'].append(rank)
                experts[alpha]['value'].append(np.exp(corrected[rank]))
        for alpha, lists in experts.items():
            expert = {k: np.asarray(v) for k,v in lists.items()}
            for weight in (.5, 1., 2.):
                for swaps in (2,4):
                    name = f'graph{alpha}_weight{weight}_swap{swaps}'
                    prediction = core.v31.residual_decode([base[i] for i in positions], expert, None,
                        {'neural':weight,'habitat':0.,'swaps':swaps})
                    scores.setdefault(name,np.zeros(len(base)))[positions] = core.legacy.score_prediction_lists(labels[index[positions]],prediction)
    report = {'scope':'exploratory reused calibration, not independent evidence; graphs fit only spatial training rows',
        'policies':[{ 'id':k, **core.v31.geographic_gain(v-old,archive['country'],archive['survey_id']),
            'fold_gains':[float((v-old)[archive['fold']==f].mean()) for f in (0,1)]} for k,v in scores.items()]}
    core.legacy.save_json(ROOT/'results/v34_cooccurrence_pilot.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
