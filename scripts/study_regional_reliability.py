"""Exploratory D3: select regional variants on risk-calibration OOF outcomes only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from geolifeclef.decision_reliability import grouped_risk_prediction
from geolifeclef.reliability import r1_accept, r1_hash, r1_paired_selected_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def accept(frame, risk, coverage):
    return r1_accept(risk, frame.surveyId, coverage, frame.country.astype(str)+'|area='+frame.area_bin.astype(str))


def summarize(frame, accepted):
    selected = frame.loc[accepted]
    rare = int(frame.rare_truth.sum())
    return {'surveys': len(frame), 'retained': int(accepted.sum()),
        'F1': float(selected.sample_F1.mean()), 'rare_hits': int(selected.rare_hits.sum()),
        'rare_occurrence_coverage': float(selected.rare_truth.sum()/rare) if rare else None,
        'conditional_rare_recall': float(selected.rare_hits.sum()/selected.rare_truth.sum()) if selected.rare_truth.sum() else None,
        'rare_recall_over_full_cohort': float(selected.rare_hits.sum()/rare) if rare else None}


def main():
    started = time.monotonic()
    config_path = ROOT/'configs/reliability_followup_d1_d3.json'
    config = json.loads(config_path.read_text())
    parameters = json.loads((ROOT/'configs/reliability_study_r1.json').read_text())['risk_regressor']
    source = ROOT/'results/reliability_study_r1/full_output/r1_export'
    output = ROOT/'results/reliability_followup_d1_d3/regional'
    output.mkdir(parents=True, exist_ok=True)
    # Record implementation before producing any new assessment comparison.
    design = {'analysis_started_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'designed_after_R1': True, 'independent_confirmation': False,
        'candidate_selection': config['D3_selection']}
    (output/'analysis_design.json').write_text(json.dumps(design, indent=2)+'\n')
    manifest = json.loads((source/'r1_manifest.json').read_text())
    frames, calibration_records, fitting_records = [], [], []
    for fold in [0, 1]:
        name = f'fold{fold}_confidence_roles.csv'
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != manifest['outputs'][name]:
            raise ValueError('Input confidence artifact digest mismatch')
        frame = pd.read_csv(source/name, float_precision='round_trip')
        frame['mean_set_confidence'] = frame.mean_top18_confidence
        calibration = frame[frame.role == 'risk_calibration'].copy().reset_index(drop=True)
        assessed = frame[frame.role == 'assessment'].copy().reset_index(drop=True)
        features = config['lean_features']
        # All rows in a geographic block stay together in every inner split.
        inner = calibration.spatial_block.map(lambda b: r1_hash('D3-inner:'+str(b)) % 3).to_numpy()
        supported = calibration.country.value_counts()
        supported = supported[supported >= config['supported_country_minimum_surveys']].index
        for variant in config['D3_variants']:
            oof = np.full(len(calibration), np.nan)
            for k in range(config['D3_inner_geographic_folds']):
                train, validation = calibration.loc[inner != k], calibration.loc[inner == k]
                if not len(train) or not len(validation):
                    raise ValueError('Empty geographic calibration split')
                oof[inner == k], _ = grouped_risk_prediction(train, validation.drop(columns=['loss', 'sample_F1']),
                    features, parameters, variant, config['D3_local_minimum_calibration_surveys'],
                    config['D3_blend_pseudocount'])
            if not np.isfinite(oof).all():
                raise ValueError('Missing OOF calibration prediction')
            selected = accept(calibration, oof, .8)
            macro = calibration.loc[selected & calibration.country.isin(supported)].groupby('country').sample_F1.mean().mean()
            calibration_records.append({'fold': fold, 'variant': variant, 'OOF_country_macro_F1': float(macro),
                'OOF_selected_F1': float(calibration.loc[selected, 'sample_F1'].mean()),
                'inner_blocks': [int(calibration.loc[inner == k, 'spatial_block'].nunique()) for k in range(3)],
                'calibration_surveys': len(calibration), 'supported_countries': len(supported)})
            risk, record = grouped_risk_prediction(calibration, assessed.drop(columns=['loss', 'sample_F1']),
                features, parameters, variant, config['D3_local_minimum_calibration_surveys'],
                config['D3_blend_pseudocount'])
            assessed['risk_'+variant] = risk
            fitting_records.append({'fold': fold, **record})
        assessed['risk_mean_set_confidence'] = -assessed.mean_set_confidence
        assessed['risk_metadata_distance'], _ = grouped_risk_prediction(calibration,
            assessed.drop(columns=['loss', 'sample_F1']), features[:4], parameters, 'global_lean')
        frames.append(assessed)
    calibration_scores = pd.DataFrame(calibration_records)
    mean_scores = calibration_scores.groupby('variant').OOF_country_macro_F1.mean()
    chosen = max(config['D3_variants'], key=lambda name: mean_scores[name])
    # Candidate is now fixed. Assessment outcomes never choose it.
    (output/'calibration_selection.json').write_text(json.dumps({'chosen_variant': chosen,
        'selection_role': 'risk_calibration with spatial OOF predictions',
        'scores': calibration_records, 'assessment_outcomes_used': False,
        'unique_calibration_surveys_per_fold': len(calibration),
        'calibration_IDs_shared_between_outer_folds': True}, indent=2)+'\n')
    pooled = pd.concat(frames).sort_values('surveyId').reset_index(drop=True)
    if pooled.surveyId.duplicated().any():
        raise ValueError('Overlapping assessment surveys')
    methods = config['D3_variants']+['mean_set_confidence', 'metadata_distance']
    metrics, countries, referral, masks = [], [], [], {}
    for method in methods:
        for coverage in config['coverage_grid']:
            mask = np.zeros(len(pooled), bool)
            for fold, group in pooled.groupby('fold'):
                mask[group.index] = accept(group, group['risk_'+method], coverage)
            metrics.append({'method': method, 'coverage': coverage, **summarize(pooled, mask)})
            if coverage == .8:
                masks[method] = mask
                for country, group in pooled.groupby('country'):
                    selected = group.loc[mask[group.index]]
                    countries.append({'method': method, 'country': country, 'surveys': len(group),
                        'retained': len(selected), 'F1': float(selected.sample_F1.mean()) if len(selected) else None})
                for cost in config['referral_cost_grid']:
                    referral.append({'method': method, 'cost_per_referral_in_sample_loss_units': cost,
                        'system_loss': float((pooled.loc[mask, 'loss'].sum()+(~mask).sum()*cost)/len(pooled)),
                        'full_coverage_loss': float(pooled.loss.mean()),
                        'assumption': 'referred surveys incur the stated cost and zero residual prediction loss'})
    contrasts = {}
    for baseline in ['global_lean', 'mean_set_confidence']:
        contrasts[chosen+'__vs__'+baseline] = r1_paired_selected_bootstrap(
            pooled.sample_F1, masks[chosen], masks[baseline], pooled.spatial_block)
    c = pd.DataFrame(countries)
    candidate_country = c[c.method == chosen].set_index('country')
    baseline_country = c[c.method == 'global_lean'].set_index('country')
    supported = candidate_country.surveys >= config['supported_country_minimum_surveys']
    country_delta = candidate_country.F1-baseline_country.F1
    # A pure country intercept cannot affect ranking inside country/area quotas.
    control_changes = 0
    for _, group in pooled.groupby('fold'):
        offsets = group.country.map(lambda name: (r1_hash(name) % 100)/10).to_numpy()
        original = accept(group, group.risk_global_lean, .8)
        shifted = accept(group, group.risk_global_lean+offsets, .8)
        control_changes += int(np.sum(original != shifted))
    pd.DataFrame(metrics).to_csv(output/'risk_coverage.csv', index=False)
    c.to_csv(output/'country_results.csv', index=False)
    pd.DataFrame(referral).to_csv(output/'referral_costs.csv', index=False)
    # Compact per-survey audit table supports later comparisons without refitting.
    for name, mask in masks.items():
        pooled['accept_'+name] = mask
    keep = ['surveyId', 'fold', 'country', 'spatial_block', 'sample_F1', 'rare_truth', 'rare_hits']
    keep += ['risk_'+name for name in methods]+['accept_'+name for name in methods]
    pooled[keep].to_csv(output/'assessment_audit.csv', index=False)
    report = {'status': 'completed_exploratory_D3', 'fresh_assessment': False,
        'new_neural_fits': 0, 'selected_using_assessment_outcomes': False,
        'chosen_variant': chosen, 'contrasts': contrasts,
        'supported_country_minimum_gain_vs_global': float(country_delta[supported].min()),
        'country_regression_margin_met_vs_global': bool(country_delta[supported].min() >= -.005),
        'country_intercept_control_changed_acceptance': control_changes,
        'fitting': fitting_records, 'seconds': time.monotonic()-started,
        'limitations': ['Exploratory method design after R1', 'Shared calibration cohort across outer model folds',
            'No independent confirmation or risk guarantee', 'Referral costs are hypothetical, not measured conservation costs']}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['status', 'chosen_variant', 'contrasts', 'country_regression_margin_met_vs_global']}))


if __name__ == '__main__':
    main()
