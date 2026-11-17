"""Exploratory feature ablations after inspecting R1; not registered primary tests."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from geolifeclef.reliability import R1_METADATA_FEATURES, r1_accept, r1_paired_selected_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    source = ROOT/'results/reliability_study_r1/full_output/r1_export'
    output = ROOT/'results/reliability_study_r1/analysis'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/reliability_study_r1.json').read_text())
    meta = R1_METADATA_FEATURES+['log_pa_distance']
    decision = ['probability_mass', 'max_probability', 'mean_top18_confidence', 'expected_F1_surrogate']
    entropy = ['top18_entropy', 'all_label_entropy']
    ensemble = ['ensemble_variance', 'top18_variance', 'ensemble_mutual_information', 'set_disagreement']
    variants = {'metadata_distance': meta, 'plus_mass_only': meta+['probability_mass'],
        'plus_decision_confidence': meta+decision,
        'plus_decision_and_entropy': meta+decision+entropy,
        'plus_ensemble_only': meta+ensemble,
        'full_model_signals': meta+decision+entropy+ensemble,
        'full_model_and_environment': meta+decision+entropy+ensemble+['environment_novelty']}
    frames, records = [], []
    for fold in [0, 1]:
        f = pd.read_csv(source/f'fold{fold}_confidence_roles.csv', float_precision='round_trip')
        cal = f[f.role == 'risk_calibration']
        assessed = f[f.role == 'assessment'].copy()
        group = assessed.country.astype(str)+'|area='+assessed.area_bin.astype(str)
        for name, features in variants.items():
            model = HistGradientBoostingRegressor(**config['risk_regressor'])
            model.fit(cal[features].to_numpy(float), cal.loss.to_numpy(float))
            risk = model.predict(assessed[features].to_numpy(float))
            accepted = r1_accept(risk, assessed.surveyId, .8, group)
            assessed['accept_'+name] = accepted
            records.append({'fold': fold, 'variant': name, 'F1': float(assessed.loc[accepted, 'sample_F1'].mean()),
                'retained': int(accepted.sum()), 'rare_hits': int(assessed.loc[accepted, 'rare_hits'].sum())})
        frames.append(assessed)
    frame = pd.concat(frames).reset_index(drop=True)
    comparisons = {}
    for name in variants:
        accepted = frame['accept_'+name]
        records.append({'fold': 'pooled', 'variant': name, 'F1': float(frame.loc[accepted, 'sample_F1'].mean()),
            'retained': int(accepted.sum()), 'rare_hits': int(frame.loc[accepted, 'rare_hits'].sum())})
        comparisons[name] = r1_paired_selected_bootstrap(frame.sample_F1, accepted,
            frame.accept_plus_decision_and_entropy, frame.spatial_block)
    pd.DataFrame(records).to_csv(output/'posthoc_signal_ablation.csv', index=False)
    report = {'status': 'completed_posthoc_CPU_ablation', 'designed_after_R1_assessment_inspection': True,
        'independent_confirmation': False, 'new_GPU_fits': 0, 'risk_regressor_fits': 14,
        'features': variants, 'risk_regressor': config['risk_regressor'], 'results': records,
        'comparisons_against_decision_and_entropy': comparisons,
        'seconds': time.monotonic()-started,
        'limitations': ['Exploratory post-hoc comparisons; not registered primary hypotheses.',
            'Correlated-feature ablations measure predictive utility, not causal mechanisms.',
            'All fits use risk-calibration labels only; assessment outcomes evaluate the fixed feature families.',
            'Local library versions can differ from Kaggle; all ablations are refit in the same local environment.']}
    (output/'posthoc_signal_ablation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(pd.DataFrame(records).query("fold == 'pooled'").to_string(index=False))


if __name__ == '__main__':
    main()
