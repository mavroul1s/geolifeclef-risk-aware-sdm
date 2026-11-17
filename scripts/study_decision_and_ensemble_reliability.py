"""D1/D2 analysis after verified inference-only checkpoint reproduction."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from geolifeclef.reliability import r1_hash, r1_paired_selected_bootstrap
from scripts.study_regional_reliability import accept, summarize

ROOT = Path(__file__).resolve().parents[1]


def paired_full_coverage_change(difference, blocks, repetitions=2000):
    difference = np.asarray(difference, float)
    _, inverse = np.unique(np.asarray(blocks).astype(str), return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=difference)
    rng = np.random.default_rng(20261114)
    draws = rng.integers(len(count), size=(repetitions, len(count)))
    gains = total[draws].sum(1)/count[draws].sum(1)
    return {'gain': float(difference.mean()), 'ci95': np.quantile(gains, [.025, .975]).tolist(),
        'clusters': len(count), 'interpretation': 'descriptive conditional spatial interval on reused development data'}


def exact_R1_averaging_reference():
    """Averaging comparison from immutable original sets, unaffected by new inference."""
    source = ROOT/'results/reliability_study_r1/full_output/r1_export'
    frames = []
    for fold in [0, 1]:
        assessed = pd.read_csv(source/f'fold{fold}_assessment.csv', float_precision='round_trip')
        assessed = assessed.set_index('surveyId')[['spatial_block', 'sample_F1']].rename(columns={'sample_F1': 'ensemble_F1'})
        with np.load(source/f'fold{fold}_predictions.npz', allow_pickle=False) as archive:
            take = np.flatnonzero(archive['role'] == 'assessment')
            ids = archive['survey_id'][take]
            sets = archive['member_top18_columns'][take]
            offsets, columns = archive['true_offsets'], archive['true_species_columns']
            for member in range(sets.shape[1]):
                scores = []
                for j, i in enumerate(take):
                    truth = set(columns[offsets[i]:offsets[i+1]].tolist())
                    scores.append(2*len(set(sets[j, member].tolist()) & truth)/(18+len(truth)))
                assessed.loc[ids, 'member_'+str(member)+'_F1'] = scores
        frames.append(assessed)
    frame = pd.concat(frames)
    members = [c for c in frame if c.startswith('member_')]
    contrast = paired_full_coverage_change(frame.ensemble_F1-frame[members].mean(axis=1), frame.spatial_block)
    return {'ensemble_full_F1': float(frame.ensemble_F1.mean()),
            'mean_individual_full_F1': float(frame[members].mean(axis=1).mean()), 'contrast': contrast,
            'source': 'original immutable R1 CSV and member species sets, not regenerated predictions'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--allow-tie-sensitivity', action='store_true',
                        help='Explicitly label outputs passing the recorded technical numerical audit')
    args = parser.parse_args()
    started = time.monotonic()
    config_path = ROOT/'configs/reliability_followup_d1_d3.json'
    config = json.loads(config_path.read_text())
    parameters = json.loads((ROOT/'configs/reliability_study_r1.json').read_text())['risk_regressor']
    output = ROOT/'results/reliability_followup_d1_d3/decisions_ensembles'
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.input/'inference_report.json').read_text())
    manifest = json.loads((args.input/'inference_manifest.json').read_text())
    sensitivity = report['status'] == 'complete_with_numerical_tie_sensitivity'
    allowed = report['status'] == 'complete' or (sensitivity and args.allow_tie_sensitivity and
                                                report.get('numerical_sensitivity_audit_passed'))
    if not allowed or report['new_neural_fits'] != 0:
        raise ValueError('Completed, reproduced inference-only output required')
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != report['config_sha256']:
        raise ValueError('Diagnostic configuration differs from executed notebook')
    for name, digest in manifest['outputs'].items():
        with (args.input/name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Inference artifact digest mismatch: '+name)
    (output/'analysis_design.json').write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config_sha256': report['config_sha256'], 'designed_after_R1': True}, indent=2)+'\n')
    frames = [pd.read_csv(args.input/f'fold{f}_decisions.csv', float_precision='round_trip') for f in [0, 1]]
    frame = pd.concat(frames, ignore_index=True)
    metrics, fold_metrics, full_metrics, contrasts, scored = [], [], [], {}, []
    for (predictor, decoder), group in frame.groupby(['predictor', 'decoder']):
        pools = []
        for fold, fold_frame in group.groupby('fold'):
            calibration = fold_frame[fold_frame.role == 'risk_calibration']
            assessed = fold_frame[fold_frame.role == 'assessment'].copy().reset_index(drop=True)
            risks = {'random': assessed.surveyId.map(lambda i: r1_hash(f'20261113:{i}')).to_numpy(float),
                     'mean_set_confidence': -assessed.mean_set_confidence.to_numpy(),
                     'set_entropy': assessed.set_entropy.to_numpy()}
            for method, features in [('metadata_distance', config['lean_features'][:4]),
                                     ('lean_risk', config['lean_features'])]:
                model = HistGradientBoostingRegressor(**parameters)
                model.fit(calibration[features].to_numpy(float), calibration.loss.to_numpy(float))
                risks[method] = model.predict(assessed[features].to_numpy(float))
            for name, risk in risks.items():
                assessed['risk_'+name] = risk
            pools.append(assessed)
        pooled = pd.concat(pools).sort_values('surveyId').reset_index(drop=True)
        if pooled.surveyId.duplicated().any():
            raise ValueError('Repeated survey within a predictor/decoder assessment')
        full_metrics.append({'predictor': predictor, 'decoder': decoder,
            'full_F1': float(pooled.sample_F1.mean()),
            'count_MAE': float((pooled.predicted_richness-pooled.true_richness).abs().mean()),
            'mean_set_size': float(pooled.predicted_richness.mean()),
            'entropy_F1_spearman': float(pooled.set_entropy.corr(pooled.sample_F1, method='spearman')),
            'confidence_F1_spearman': float(pooled.mean_set_confidence.corr(pooled.sample_F1, method='spearman'))})
        masks = {}
        for method in config['D1_selectors']:
            for coverage in config['coverage_grid']:
                mask = np.zeros(len(pooled), bool)
                for fold, part in pooled.groupby('fold'):
                    accepted = accept(part, part['risk_'+method], coverage)
                    mask[part.index] = accepted
                    if coverage == .8:
                        fold_metrics.append({'predictor': predictor, 'decoder': decoder, 'method': method,
                                             'fold': fold, **summarize(part, accepted)})
                metrics.append({'predictor': predictor, 'decoder': decoder, 'method': method,
                                'coverage': coverage, **summarize(pooled, mask)})
                if coverage == .8:
                    masks[method] = mask
        if predictor == 'ensemble':
            for baseline in ['mean_set_confidence', 'metadata_distance']:
                contrasts[decoder+'__lean_vs__'+baseline] = r1_paired_selected_bootstrap(
                    pooled.sample_F1, masks['lean_risk'], masks[baseline], pooled.spatial_block)
        scored.append(pooled[['surveyId', 'predictor', 'decoder', 'spatial_block', 'sample_F1']])
    full = pd.DataFrame(full_metrics)
    score_frame = pd.concat(scored, ignore_index=True)
    for decoder in config['decoders']:
        part = score_frame[score_frame.decoder == decoder]
        pivot = part.pivot(index='surveyId', columns='predictor', values='sample_F1')
        members = [c for c in pivot if c.startswith('seed_')]
        blocks = part.drop_duplicates('surveyId').set_index('surveyId').loc[pivot.index, 'spatial_block']
        contrasts[decoder+'__ensemble_vs_mean_individual_F1'] = paired_full_coverage_change(
            pivot.ensemble-pivot[members].mean(axis=1), blocks)
    ensemble = score_frame[score_frame.predictor == 'ensemble']
    pivot = ensemble.pivot(index='surveyId', columns='decoder', values='sample_F1')
    blocks = ensemble.drop_duplicates('surveyId').set_index('surveyId').loc[pivot.index, 'spatial_block']
    for decoder in config['decoders'][1:]:
        contrasts[decoder+'__vs_top18_full_coverage'] = paired_full_coverage_change(pivot[decoder]-pivot.top18, blocks)
    pd.DataFrame(metrics).to_csv(output/'risk_coverage.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(output/'fold_results_80.csv', index=False)
    full.to_csv(output/'full_coverage.csv', index=False)
    pd.DataFrame(metrics).query("predictor == 'ensemble' and coverage == .8").to_csv(output/'D1_ensemble_80.csv', index=False)
    pd.DataFrame(metrics).query('coverage == .8').to_csv(output/'D2_all_predictors_80.csv', index=False)
    original_reference = exact_R1_averaging_reference()
    result = {'status': 'completed_exploratory_D1_D2', 'new_neural_fits': 0,
        'inference_status': report['status'], 'numerical_tie_sensitivity': sensitivity,
        'original_exact_member_set_guard_passed': report.get('original_exact_member_set_guard_passed', True),
        'original_immutable_R1_averaging_reference': original_reference,
        'checkpoint_inferences': 6, 'fresh_assessment': False, 'verified_inference_outputs': len(manifest['outputs']),
        'contrasts': contrasts, 'seconds': time.monotonic()-started,
        'limitations': ['Methods and feature families designed after R1',
            'Member mean is expected individual-model F1, not a fourth prediction model',
            'Three members are not three independent ensemble trials',
            'Thresholds use probability calibration only; errors use disjoint risk-calibration labels',
            'No independent confirmation or competition submission',
            'Any failed exact-set reproduction is preserved; numerical rank-equivalence audit is separate']}
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(output, pd.DataFrame(metrics), full, sensitivity=sensitivity)
    print(full[full.predictor == 'ensemble'].to_string(index=False))
    print(json.dumps(result['contrasts']))


def plot(output, curves, full, *, sensitivity=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, decoder in zip(axes[:2], ['top18', 'expected_F1_surrogate']):
        for method in ['random', 'mean_set_confidence', 'set_entropy', 'lean_risk']:
            group = curves[(curves.predictor == 'ensemble') & (curves.decoder == decoder) & (curves.method == method)].sort_values('coverage')
            ax.plot(group.coverage, group.F1, 'o-', label=method.replace('_', ' '))
        ax.set(title=decoder.replace('_', ' '), xlabel='Fraction retained', ylabel='Retained-survey F1')
        ax.legend(fontsize=7)
    for predictor, group in full.groupby('predictor'):
        group = group.set_index('decoder').loc[['top18', 'calibrated_threshold', 'expected_F1_surrogate']]
        axes[2].plot(range(3), group.full_F1, 'o-', label=predictor)
    axes[2].set_xticks(range(3), ['Top-18', 'Threshold', 'F1 surrogate'])
    axes[2].set(title='Full coverage: individuals and ensemble', ylabel='Sample F1')
    axes[2].legend(fontsize=7)
    fig.suptitle('D1/D2 — reused development data'+(' · numerical tie sensitivity' if sensitivity else ''), fontsize=11)
    for suffix in ['png', 'svg']:
        fig.savefig(output/f'decision_ensemble_diagnostics.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
