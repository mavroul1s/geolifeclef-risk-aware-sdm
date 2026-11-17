"""Analyze completed R1 exports without fitting or selecting new methods."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geolifeclef.reliability import r1_accept, r1_paired_selected_bootstrap
from geolifeclef.reliability_analysis import selection_gain_decomposition

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/reliability_study_r1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    report = json.loads((args.input/'r1_report.json').read_text())
    manifest = json.loads((args.input/'r1_manifest.json').read_text())
    if report['status'] != 'complete' or report['mode'] != 'full':
        raise ValueError('Scientific analysis requires a completed full run')
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != report['config_sha256']:
        raise ValueError('Configuration differs from the executed notebook')
    verified = []
    for name, expected in manifest['outputs'].items():
        path = args.input/name
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Output integrity failure: {name}')
        verified.append(name)
    frame = pd.concat([pd.read_csv(args.input/f'fold{fold}_assessment.csv') for fold in report['folds']])
    frame = frame.sort_values('surveyId').reset_index(drop=True)
    if frame.surveyId.duplicated().any():
        raise ValueError('Assessment overlap')
    contrast = config['primary_contrast']
    ca, ba = contrast['candidate'], contrast['baseline']
    a, b = frame['primary_accept_'+ca].astype(bool), frame['primary_accept_'+ba].astype(bool)
    groups = frame.groupby(['fold', 'country', 'area_bin'], dropna=False)
    for _, group in groups:
        if group['primary_accept_'+ca].sum() != group['primary_accept_'+ba].sum():
            raise ValueError('Primary methods have unequal group budgets')
    primary = r1_paired_selected_bootstrap(frame.sample_F1, a, b, frame.spatial_block)
    countries = []
    for country, part in frame.groupby('country'):
        accepted_a, accepted_b = part['primary_accept_'+ca], part['primary_accept_'+ba]
        score_a = float(part.loc[accepted_a, 'sample_F1'].mean()) if accepted_a.any() else None
        score_b = float(part.loc[accepted_b, 'sample_F1'].mean()) if accepted_b.any() else None
        countries.append({'country': country, 'surveys': len(part), 'candidate_retained': int(accepted_a.sum()),
            'baseline_retained': int(accepted_b.sum()), 'candidate_F1': score_a, 'baseline_F1': score_b,
            'gain': score_a-score_b if score_a is not None and score_b is not None else None})
    country = pd.DataFrame(countries)
    country.to_csv(args.output/'primary_country_comparison.csv', index=False)
    supported = country[country.surveys >= contrast['supported_country_minimum_surveys']]
    country_margin = float(supported.gain.min())
    fold_gains = []
    for fold, part in frame.groupby('fold'):
        aa, bb = part['primary_accept_'+ca], part['primary_accept_'+ba]
        fold_gains.append({'fold': int(fold), 'candidate_F1': float(part.loc[aa, 'sample_F1'].mean()),
                          'baseline_F1': float(part.loc[bb, 'sample_F1'].mean()),
                          'gain': float(part.loc[aa, 'sample_F1'].mean()-part.loc[bb, 'sample_F1'].mean())})
    curves = pd.read_csv(args.input/'risk_coverage.csv')
    pooled = []
    for key, part in curves.groupby(['policy', 'method', 'coverage', 'group']):
        n = part.retained_surveys.sum()
        pooled.append(dict(zip(['policy', 'method', 'coverage', 'group'], key),
            retained_surveys=int(n), sample_F1=float((part.sample_F1*part.retained_surveys).sum()/n) if n else None))
    pd.DataFrame(pooled).to_csv(args.output/'pooled_risk_coverage.csv', index=False)
    decomposition = []
    for method in config['selectors']:
        accepted = np.zeros(len(frame), bool)
        # Selection budgets were defined within fold, so replay that exact rule.
        for _, part in frame.groupby('fold'):
            accepted[part.index] = r1_accept(part['risk_'+method], part.surveyId, .8)
        for stratification, group_values in [
            ('country', frame.country.astype(str)), ('plot_area', frame.area_bin.astype(str)),
            ('country_area', frame.country.astype(str)+'|area='+frame.area_bin.astype(str))]:
            # Area cutpoints and the predictive model differ by fold.
            group_values = 'fold='+frame.fold.astype(str)+'|'+group_values
            d = selection_gain_decomposition(frame.sample_F1, accepted, group_values)
            decomposition.append({'method': method, 'stratification': 'fold_'+stratification,
                **{k: d[k] for k in ['gain', 'composition', 'within', 'full_mean', 'accepted_mean']}})
    pd.DataFrame(decomposition).to_csv(args.output/'selection_gain_decomposition.csv', index=False)
    seed_metrics = []
    for fold in report['folds']:
        with np.load(args.input/f'fold{fold}_predictions.npz', allow_pickle=False) as archive:
            role = archive['role'] == 'assessment'
            sets = archive['member_top18_columns'][role]
            offsets = archive['true_offsets']
            columns = archive['true_species_columns']
            truth = [set(columns[offsets[i]:offsets[i+1]].tolist()) for i in np.flatnonzero(role)]
            for member, seed in enumerate(report['seeds']):
                scores = [2*len(set(predicted)&actual)/(18+len(actual))
                          for predicted, actual in zip(sets[:, member], truth)]
                seed_metrics.append({'fold': fold, 'seed': seed, 'sample_F1': float(np.mean(scores))})
    pd.DataFrame(seed_metrics).to_csv(args.output/'individual_seed_metrics.csv', index=False)
    secondary_comparisons = {}
    for candidate, baseline in [(ca, 'mean_top18_confidence'), (ca, 'model_only_risk'), ('full_risk', ca)]:
        secondary_comparisons[candidate+'__vs__'+baseline] = r1_paired_selected_bootstrap(
            frame.sample_F1, frame['primary_accept_'+candidate],
            frame['primary_accept_'+baseline], frame.spatial_block)
    rare_summary = []
    for method in ['full_coverage', ca, ba, 'mean_top18_confidence']:
        mask = np.ones(len(frame), bool) if method == 'full_coverage' else frame['primary_accept_'+method]
        retained = frame.loc[mask]
        truth, hits = int(retained.rare_truth.sum()), int(retained.rare_hits.sum())
        rare_summary.append({'method': method, 'retained_surveys': len(retained),
            'rare_true_occurrences': truth, 'rare_true_positives': hits,
            'conditional_rare_recall': hits/truth if truth else None,
            'rare_occurrence_coverage': truth/int(frame.rare_truth.sum()),
            'rare_recovery_over_full_cohort': hits/int(frame.rare_truth.sum())})
    pd.DataFrame(rare_summary).to_csv(args.output/'rare_species_coverage.csv', index=False)
    result = {'status': 'completed_development_analysis', 'fresh_assessment': False,
        'config_sha256': report['config_sha256'], 'verified_manifest_files': len(verified),
        'assessment_surveys': len(frame), 'primary_candidate': ca, 'primary_baseline': ba,
        'primary_policy': contrast['coverage_policy'], 'primary_coverage': .8,
        'candidate_F1': float(frame.loc[a, 'sample_F1'].mean()),
        'baseline_F1': float(frame.loc[b, 'sample_F1'].mean()),
        'primary': primary, 'fold_gains': fold_gains,
        'minimum_supported_country_gain': country_margin,
        'gain_margin_met': primary['gain'] >= contrast['minimum_absolute_F1_gain'],
        'country_regression_margin_met': country_margin >= -contrast['maximum_supported_country_F1_regression'],
        'all_supported_country_area_budgets_match': True,
        'both_fold_gains_positive': all(r['gain'] > 0 for r in fold_gains),
        'full_coverage_F1': float(frame.sample_F1.mean()),
        'secondary_expected_F1_decoder_F1': float(frame.secondary_F1.mean()),
        'fixed_top18_count_MAE': float((frame.predicted_richness-frame.true_richness).abs().mean()),
        'secondary_expected_F1_decoder_count_MAE': float((frame.secondary_count-frame.true_richness).abs().mean()),
        'secondary_comparisons_exploratory': secondary_comparisons,
        'rare_definition': '1 to 25 positive PA surveys in the corresponding fold training role',
        'rare_species_coverage': rare_summary,
        'spearman_correlations': {
            'top18_entropy_vs_mean_top18_confidence': float(frame.top18_entropy.corr(frame.mean_top18_confidence, method='spearman')),
            'top18_entropy_vs_F1': float(frame.top18_entropy.corr(frame.sample_F1, method='spearman')),
            'mean_top18_confidence_vs_F1': float(frame.mean_top18_confidence.corr(frame.sample_F1, method='spearman'))},
        'single_seed_quality': seed_metrics,
        'runtime_hours': report['runtime_hours'],
        'interpretation_limits': ['Reused adaptive development data, no independent confirmation.',
            'One three-member ensemble per fold; three individual members are not three independent ensemble trials.',
            'Spatial intervals condition on saved fits and fixed selection rules.',
            'Exported full mean probabilities are float16; exact primary scores and selection features are in CSV.',
            'Selective F1 is not a full-coverage competition result.']}
    result['practical_primary_criteria_met'] = result['gain_margin_met'] and result['country_regression_margin_met']
    (args.output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    plot(args.output, pd.DataFrame(pooled), country, ca, ba)
    print(json.dumps({k: result[k] for k in ['status', 'assessment_surveys', 'candidate_F1', 'baseline_F1', 'primary', 'practical_primary_criteria_met']}))


def plot(output, curves, countries, candidate, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    methods = {baseline: 'Metadata + distance', candidate: 'Metadata + distance + model signals',
               'mean_top18_confidence': 'Mean predicted-set confidence', 'area_only': 'Plot area only'}
    for ax, policy, title in zip(axes[:2], ['unconstrained', 'country_area'],
        ['Unconstrained selection', 'Identical country / area quotas']):
        for method, label in methods.items():
            part = curves[(curves.policy == policy) & (curves.method == method) & (curves.group == 'all')].sort_values('coverage')
            ax.plot(part.coverage, part.sample_F1, 'o-', label=label)
        ax.set(xlabel='Fraction retained', ylabel='Retained-survey F1', title=title)
        ax.legend(fontsize=7)
    country = countries[countries.surveys >= 30].sort_values('gain')
    axes[2].barh(country.country, country.gain)
    axes[2].axvline(0, color='black', linewidth=.6)
    axes[2].axvline(-.005, color='red', linestyle='--', linewidth=.8)
    axes[2].set(xlabel='Primary F1 gain at 80% retention', title='Supported countries (n >= 30)')
    fig.suptitle('Controlled development study R1 — historically consumed data', fontsize=11)
    for suffix in ['png', 'svg']:
        fig.savefig(output/f'reliability_r1.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
