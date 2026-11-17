"""Develop the content-mass objective on consumed GeoLifeCLEF outputs only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from geolifeclef.ecological_selection import QuotaSelection, paired_ecological_comparison

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    output = ROOT/'results/ecological_selection_pilot';output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    old = json.loads((ROOT/'configs/reliability_followup_d1_d3.json').read_text())
    parameters = config['risk_regressor']
    parameters['random_state'] = 20261116
    (output/'design.json').write_text(json.dumps({'designed_at_utc': datetime.now(timezone.utc).isoformat(),
        'fresh_data': False, 'external_outcomes_inspected': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'target': '80% predicted rare occurrence mass, base score 1-SDC, same quotas',
        'primary_candidate': 'rare_count_constraint',
        'controls': ['SDC', 'mean_confidence', 'lean_risk', 'raw_rare_constraint', 'total_count_constraint']}, indent=2)+'\n')
    all_frames, records = [], []
    for fold in [0, 1]:
        source = ROOT/'results/reliability_followup_d1_d3/inference_output/aligned/inference_export'
        frame = pd.read_csv(source/f'fold{fold}_decisions.csv', float_precision='round_trip')
        frame = frame[(frame.predictor == 'ensemble') & (frame.decoder == 'expected_F1_surrogate')].copy()
        original = ROOT/'results/reliability_study_r1/full_output/r1_export'
        with np.load(original/f'fold{fold}_predictions.npz', allow_pickle=False) as archive:
            frequency = archive['training_frequency']
        rare = (frequency >= 1) & (frequency <= 25)
        with np.load(source/f'fold{fold}_mean_probability.npz', allow_pickle=False) as archive:
            probability, ids = archive['mean_probability'], archive['survey_id']
        mass = np.concatenate([p[:, rare].sum(1) for p in np.array_split(probability, max(1, len(probability)//256))]).astype(float)
        del probability
        frame['predicted_rare_mass'] = pd.Series(mass, index=ids).loc[frame.surveyId].to_numpy()
        cal = frame[frame.role == 'risk_calibration']
        assessed = frame[frame.role == 'assessment'].copy().reset_index(drop=True)
        features = old['lean_features']+['predicted_rare_mass']
        predicted = {}
        for target, column, used in [('lean_risk', 'loss', old['lean_features']),
                                      ('rare_count', 'rare_truth', features), ('total_count', 'true_richness', features)]:
            model = HistGradientBoostingRegressor(**parameters)
            model.fit(cal[used].to_numpy(float), cal[column].to_numpy(float))
            predicted[target] = np.maximum(model.predict(assessed[used].to_numpy(float)), 0)
        groups = assessed.country.astype(str)+'|area='+assessed.area_bin.astype(str)
        selector = QuotaSelection(assessed.surveyId, .8, groups)
        base_risk = 1-assessed.expected_F1_surrogate.to_numpy()
        masks = {'SDC': selector.accept(base_risk),
                 'mean_confidence': selector.accept(-assessed.mean_set_confidence),
                 'lean_risk': selector.accept(predicted['lean_risk'])}
        for method, estimate in [('raw_rare_constraint', assessed.predicted_rare_mass),
                                  ('rare_count_constraint', predicted['rare_count']),
                                  ('total_count_constraint', predicted['total_count'])]:
            masks[method], record = selector.preserve_mass(base_risk, estimate)
            records.append({'fold': fold, 'method': method, **record})
        for name, chosen in masks.items():
            assessed['accept_'+name] = chosen
        all_frames.append(assessed)
    frame = pd.concat(all_frames, ignore_index=True)
    rows, contrasts = [], {}
    for method in masks:
        accepted = frame['accept_'+method].to_numpy(bool)
        s = frame.loc[accepted]
        rows.append({'method': method, 'retained': len(s), 'F1': float(s.sample_F1.mean()),
            'rare_occurrence_coverage': float(s.rare_truth.sum()/frame.rare_truth.sum()),
            'rare_hits': int(s.rare_hits.sum()),
            'rare_precision': float(s.rare_hits.sum()/s.rare_predictions.sum()) if s.rare_predictions.sum() else None})
        contrasts[method] = paired_ecological_comparison(frame.sample_F1, frame.rare_truth, accepted,
            frame.accept_SDC, frame.spatial_block)
    pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
    report = {'status': 'completed_development_pilot', 'new_neural_fits': 0,
        'external_assessment_opened': False, 'numerical_tie_sensitivity': True,
        'comparisons_vs_SDC': contrasts, 'proxy_constraint_records': records,
        'seconds': time.monotonic()-started,
        'limitations': ['All PA labels historically consumed', 'Development pilot, no independent confirmation',
            'A proxy constraint is not a guarantee about actual rare occurrences',
            'Rarity is training PA prevalence 1..25; external primary rarity is PO relative prevalence']}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
