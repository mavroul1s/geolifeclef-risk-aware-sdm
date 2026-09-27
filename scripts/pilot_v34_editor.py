"""Small exploratory pilot; no fresh evidence and no Kaggle/production writes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts import pilot_v34_editor_core as core

ROOT = Path(__file__).resolve().parents[1]


def subset(data, indices):
    return {"base": [data["base"][i] for i in indices], **{k: {n: v[indices] for n, v in data[k].items()}
            for k in ("bag", "specialist")}}


def main():
    source = ROOT/'results/v32_kaggle_output/calibration_top128_v32.npz'
    with np.load(source, allow_pickle=False) as archive:
        ids, countries, folds = [archive[k] for k in ('survey_id', 'country', 'fold')]
        base = [r[r != 65535].astype(int).tolist() for r in archive['reference_columns_padded65535']]
        data = {k: {'rank': archive['ranked_species_columns'][:, i], 'value': archive['probability'][:, i]}
                for i, k in enumerate(('bag', 'specialist'))}
        data['base'] = core.previous.decode(base, data['bag'], data['specialist'], core.FROZEN_V32_POLICY)
        labels = np.zeros((len(ids), len(archive['species_ids'])), np.uint8)
        for i, (a, b) in enumerate(zip(archive['true_offsets'][:-1], archive['true_offsets'][1:])):
            labels[i, archive['true_species_columns'][a:b]] = 1
    metadata = pd.read_csv(ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv',
        usecols=['surveyId', 'lat', 'lon', 'country']).drop_duplicates('surveyId').set_index('surveyId').loc[ids].reset_index()
    block = core.legacy.spatial_blocks(metadata)
    test = np.array([int(hashlib.sha256(('v34-pilot:'+str(b)).encode()).hexdigest()[:8], 16)%10 >= 7 for b in block])
    train = ~test
    distance = core.legacy.nearest_distance_km(metadata.loc[test, ['lat','lon']].to_numpy(), metadata.loc[train, ['lat','lon']].to_numpy())
    train[np.flatnonzero(train)[distance < 20.]] = False
    train_ids, test_ids = np.flatnonzero(train), np.flatnonzero(test)
    assert not set(ids[train]) & set(ids[test])
    assert not set(block[train]) & set(block[test])
    # Two folds share survey IDs: normalize repetitions, never split by fold.
    multiplicity = pd.Series(ids[train]).value_counts()
    weight = np.array([1./multiplicity[i] for i in ids[train]])
    model, fit = core.fit_editor(subset(data, train_ids), labels[train], weight)
    evaluated = subset(data, test_ids)
    utilities = core.score_edits(evaluated, model)
    baseline = core.legacy.score_prediction_lists(labels[test], evaluated['base'])
    trials = []
    for policy in core.POLICIES:
        prediction = core.decode(evaluated['base'], utilities, policy)
        score = core.legacy.score_prediction_lists(labels[test], prediction)
        trials.append({**policy, **core.v31.geographic_gain(score-baseline, countries[test], ids[test]),
            'fold_gains': [float((score-baseline)[folds[test] == f].mean()) for f in (0,1)],
            'mean_count_change': float(np.mean([len(a)-len(b) for a,b in zip(prediction,evaluated['base'])]))})
    report = {'scope': 'exploratory on consumed sampled calibration; NOT independent validation; NOT policy selection for Kaggle',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'training_rows': int(train.sum()),
        'evaluation_rows': int(test.sum()), 'training_unique_surveys': len(set(ids[train])),
        'evaluation_unique_surveys': len(set(ids[test])), 'training_blocks': len(set(block[train])),
        'evaluation_blocks': len(set(block[test])), 'buffer_km': 20, 'fit': fit,
        'baseline_f1': float(baseline.mean()), 'policies': trials}
    core.legacy.save_json(ROOT/'results/v34_exploratory_pilot.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
