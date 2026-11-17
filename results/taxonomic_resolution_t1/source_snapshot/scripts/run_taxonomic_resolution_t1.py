"""Frozen development stress test; no assessment-driven model selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geolifeclef.external_reliability import (META_FEATURES, RISK_FEATURES, apply_calibration,
    prediction_context, decode_counts, decision_features, decision_outcomes, read_rows)
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.neon_assessment import score_fixed_sets
from geolifeclef.partial_set_bounds import fixed_set_f1_bounds, linear_contrast_bounds
from geolifeclef.reliability import r1_hash
from geolifeclef.sage_inference import load_network, normalize
from geolifeclef.taxonomic_censoring import genus_contract, censor_genera, paired_spatial_interval

DATA, OUT = ROOT/'data/taxonomic_resolution_t1', ROOT/'results/taxonomic_resolution_t1'
SOURCES = ['configs/taxonomic_resolution_t1.json', 'docs/taxonomic_resolution_t1_protocol.md',
    'scripts/run_taxonomic_resolution_t1.py', 'scripts/validate_taxonomic_resolution_t1.py',
    'src/geolifeclef/taxonomic_censoring.py', 'tests/test_taxonomic_censoring.py',
    *['src/geolifeclef/'+n+'.py' for n in ['external_reliability', 'ecological_selection',
       'neon_assessment', 'partial_set_bounds', 'reliability', 'sage_inference']]]
INPUTS = ['configs/confirmation_e4.json', 'results/confirmation_e4/fit_parameters.json',
    'data/confirmation_e4/label_contract.npz', 'data/confirmation_e4/source_calibration_roles.csv',
    'data/confirmation_e4/neon_assessment_roles.csv', 'data/confirmation_e4/frozen_policy.npz',
    'artifacts/neon_e4_gpu/e4_probabilities.npz', 'data/neon_bridge/model_configs.json',
    'data/neon_bridge/checkpoints/aggregation_s0/best.ckpt', 'data/neon_bridge/cpu_reference.npz',
    'data/neon_bridge/species_names.csv', 'data/external_replication_e1/cohort_roles.csv',
    'data/external_replication_e1/eligible_species.npy', 'data/external_replication_e1/rare_species_mask.npy',
    'data/external_replication_e1/targets/species_properties_1km.csv',
    'data/external_replication_e1/sealed_eval_test_targets.h5',
    *['data/neon_bridge/reference_predictors/eval_'+n+'.h5' for n in
      ['chelsa', 'soilgrids', 'topography', 'human_footprint']]]
HISTORY = [(n, relative) for n, relative in [
    ('external_replication_e1', True), ('ranking_study_e2', True), ('confirmation_e3', True),
    ('neon_confirmation_feasibility', True), ('neon_bridge', False), ('confirmation_e4', False)]]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def verify_history():
    checks = {}
    for name, relative in HISTORY:
        path = ROOT/'results'/name/'artifact_manifest.json'
        files = json.loads(path.read_text())['files']
        base = path.parent if relative else ROOT
        for key, receipt in files.items():
            p = base/key
            if p.stat().st_size != receipt['bytes'] or sha(p) != receipt['sha256']:
                raise ValueError('Historical artifact changed: '+str(p))
        checks[name] = {'files_verified': len(files), 'manifest_sha256': sha(path)}
    return checks


def freeze():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    identity = {name: sha(ROOT/name) for name in SOURCES+INPUTS}
    path = OUT/'protocol_freeze.json'
    if path.exists():
        lock = json.loads(path.read_text())
        if lock['identity'] != identity:
            raise ValueError('Frozen T1 source/input changed; preserve and disclose repairs')
        return lock
    checks = verify_history()
    for name in SOURCES:
        p = OUT/'source_snapshot'/name
        p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, p)
    lock = {'at_utc': now(), 'identity': identity, 'historical_artifacts': checks,
            'assessment_labels_materialized_by_T1': False,
            'data_reuse': 'all assessment outcomes historically consumed; local development lock only'}
    save(path, lock)
    return lock


def policy(config):
    freeze()
    if (OUT/'policy_freeze.json').exists():
        raise ValueError('Policy already frozen; do not refit')
    started = time.monotonic(); torch.set_num_threads(2)
    cohort = pd.read_csv(ROOT/'data/external_replication_e1/cohort_roles.csv')
    assess = cohort[cohort.role.eq('assessment')].reset_index(drop=True)
    if len(assess) != 5251 or assess.spatial_block.nunique() != 73:
        raise ValueError('Wrong assessment role')
    source = pd.read_csv(ROOT/'data/confirmation_e4/source_calibration_roles.csv')
    neon = pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv')
    if set(source.cell_id) & set(assess.cell_id):
        raise ValueError('Calibration/assessment overlap')
    label = np.load(ROOT/'data/confirmation_e4/label_contract.npz')
    universe, rare = label['universe'], label['rare']
    native = np.load(ROOT/'data/external_replication_e1/eligible_species.npy')
    eligible = native[cohort.role.eq('assessment').to_numpy()] & universe
    if not eligible.any(1).all():
        raise ValueError('Empty eligible universe')
    props = pd.read_csv(ROOT/'data/external_replication_e1/targets/species_properties_1km.csv',
        usecols=['species_id', 'species_name', 'genus', 'relative_prevalence'])
    names = pd.read_csv(ROOT/'data/neon_bridge/species_names.csv')
    assert np.array_equal(props.species_id, names.Index)
    assert props.species_name.tolist() == names['Species Name'].tolist()
    genus_names, genus_codes, weights = genus_contract(props.species_name, props.genus,
                                                      props.relative_prevalence, universe)
    pd.DataFrame({'genus': genus_names, 'rarity_weight': weights}).to_csv(OUT/'genus_weights.csv', index=False)
    save(OUT/'taxonomy_audit.json', {'classes': len(names), 'universe_classes': int(universe.sum()),
        'canonical_genera': len(genus_names), 'universe_genera': int(len(np.unique(genus_codes[universe]))),
        'first_token_genus_identity_all_classes': True, 'missing_or_ambiguous_genus': 0,
        'columns_read': props.columns.tolist(), 'test_prevalence_used': False,
        'eligible_class_count_range': [int(eligible.sum(1).min()), int(eligible.sum(1).max())],
        'NEON_parent_rank_name_history_semantics_resolved': False})
    ref = np.load(ROOT/'data/neon_bridge/cpu_reference.npz')
    model_config = json.loads((ROOT/'data/neon_bridge/model_configs.json').read_text())[config['model']]
    model = load_network(ROOT/'data/neon_bridge/checkpoints/aggregation_s0/best.ckpt', model_config)
    means, stds = ref['aggregation_s0__means'], ref['aggregation_s0__stds']
    def infer(raw):
        with torch.inference_mode():
            return np.concatenate([torch.sigmoid(model(normalize(raw[i:i+256], means, stds))).numpy()
                                   for i in range(0, len(raw), 256)])
    reference = infer(ref['aggregation_s0__raw'])
    errors = {}
    for key in ['cpu', 'published']:
        error = np.abs(reference.astype(float)-ref['aggregation_s0__'+key])
        errors[key] = {'max': float(error.max()), 'mean': float(error.mean())}
        assert error.max() <= 1e-5 and error.mean() <= 1e-6
    parts = []
    for group in ['chelsa', 'soilgrids', 'topography', 'human_footprint']:
        with h5py.File(ROOT/'data/neon_bridge/reference_predictors'/('eval_'+group+'.h5')) as f:
            parts.append(read_rows(f['predictor_data'], assess.cell_id.to_numpy(int)))
    raw = np.concatenate(parts, axis=1)
    assert raw.shape == (5251, 52)
    assessment_probability = infer(raw)
    np.savez_compressed(DATA/'assessment_probabilities.npz', probability=assessment_probability)
    # Reconstruct E4 only on its original source role and verify frozen outputs.
    gpu = np.load(ROOT/'artifacts/neon_e4_gpu/e4_probabilities.npz')
    params = json.loads((ROOT/'results/confirmation_e4/fit_parameters.json').read_text())
    e4 = json.loads((ROOT/'configs/confirmation_e4.json').read_text())
    calibration = params['probability_calibration']
    old_eligible = np.concatenate([label['source_eligible'], label['neon_eligible']])
    old_probability = apply_calibration(np.concatenate([gpu['source'], gpu['neon']]), calibration, old_eligible)
    old_context = prediction_context(old_probability, old_eligible)
    old_count = decode_counts(old_context, config['decoder'])
    old_metadata = pd.concat([source[META_FEATURES[:-1]], neon[META_FEATURES[:-1]]], ignore_index=True)
    old_features = decision_features(old_context, old_count, old_metadata, rare)
    risk = np.flatnonzero(source.role.eq('risk_calibration'))
    save(OUT/'source_target_access.json', {'at_utc': now(), 'role': 'original E1 risk_calibration',
        'rows': len(risk), 'T1_assessment_requested': False})
    with h5py.File(ROOT/'data/external_replication_e1/sealed_eval_test_targets.h5') as f:
        y = read_rows(f['targets'], source.iloc[risk].prediction_row.to_numpy(int))
    y = np.where(label['source_eligible'][risk], y, -1)
    quality = decision_outcomes(y, old_eligible[risk], old_context['rank'][risk], old_count[risk], rare).sample_F1
    head = HistGradientBoostingRegressor(**e4['risk_regressor'])
    head.fit(old_features.iloc[risk][RISK_FEATURES], quality)
    frozen = np.load(ROOT/'data/confirmation_e4/frozen_policy.npz')
    reproduced = np.clip(head.predict(old_features.iloc[len(source):][RISK_FEATURES]), 0, 1)
    head_error = float(np.abs(reproduced-frozen['HGB_quality']).max())
    assert head_error <= 1e-12
    assert abs(float(quality.mean())-params['source_riskcal_mean_quality']) <= 1e-15
    probability = apply_calibration(assessment_probability, calibration, eligible)
    context = prediction_context(probability, eligible)
    count = decode_counts(context, config['decoder'])
    features = decision_features(context, count, assess[META_FEATURES[:-1]], rare)
    arrays = {'rank': context['rank'].astype(np.int16), 'count': count,
        'HGB_quality': np.clip(head.predict(features[RISK_FEATURES]), 0, 1), 'SDC': features.SDC.to_numpy(),
        'mean_confidence': features.mean_set_confidence.to_numpy(), 'entropy': features.set_entropy.to_numpy(),
        'random': np.array([r1_hash('T1:'+str(x))/2**64 for x in assess.cell_id])}
    risks = {name: arrays[name] if name in ['random', 'entropy'] else 1-arrays['HGB_quality' if name == 'HGB' else name]
             for name in config['selectors']}
    groups = (assess.country.astype(str)+'|'+assess.area_bin.astype(str)).to_numpy()
    selector = QuotaSelection(assess.cell_id.to_numpy(), config['coverage'], groups)
    for name, values in risks.items():
        arrays['accept_'+name] = selector.accept(values)
        for rows, quota in selector.groups:
            assert arrays['accept_'+name][rows].sum() == quota
    np.savez_compressed(DATA/'fixed_policy.npz', **arrays)
    np.savez_compressed(DATA/'contract.npz', eligible=eligible, rare=rare, universe=universe,
                        genus_codes=genus_codes, genus_weights=weights)
    features.to_csv(DATA/'features.csv', index=False); assess.to_csv(DATA/'assessment_roles.csv', index=False)
    save(OUT/'inference_validation.json', {'reference': errors, 'reconstructed_E4_HGB_max_error': head_error,
        'new_HGB_fits': 1, 'new_probability_calibration_fits': 0, 'new_backbone_fits': 0,
        'new_GPU_jobs': 0, 'elapsed_seconds_CPU_policy': time.monotonic()-started,
        'probability_calibration': calibration, 'parameters_reused_from': 'results/confirmation_e4/fit_parameters.json'})
    save(OUT/'policy_freeze.json', {'at_utc': now(), 'T1_assessment_labels_accessed': False,
        'protocol_freeze_sha256': sha(OUT/'protocol_freeze.json'),
        'files': {str(p.relative_to(ROOT)): sha(p) for p in DATA.iterdir() if p.is_file()}})
    print(json.dumps({'status': 'fixed_policy_before_assessment', 'cells': len(assess),
                      'E4_HGB_max_error': head_error, 'reference': errors}), flush=True)


def seed_stats(values):
    values = np.asarray(values, float)
    return {'mean': float(values.mean()), 'seed_min': float(values.min()), 'seed_max': float(values.max())}


def replay(config):
    freeze()
    if (OUT/'summary.json').exists():
        raise ValueError('T1 completed; do not overwrite')
    lock = json.loads((OUT/'policy_freeze.json').read_text())
    assert lock['protocol_freeze_sha256'] == sha(OUT/'protocol_freeze.json')
    for name, digest in lock['files'].items():
        assert sha(ROOT/name) == digest
    cohort = pd.read_csv(DATA/'assessment_roles.csv'); p = np.load(DATA/'fixed_policy.npz')
    contract = np.load(DATA/'contract.npz'); eligible, rare = contract['eligible'], contract['rare']
    save(OUT/'assessment_access.json', {'first_materialized_at_utc': now(),
        'policy_freeze_sha256': sha(OUT/'policy_freeze.json'), 'rows': len(cohort),
        'role': 'historically consumed E1 assessment; adaptive development, not fresh confirmation'})
    with h5py.File(ROOT/'data/external_replication_e1/sealed_eval_test_targets.h5') as f:
        oracle_truth = read_rows(f['targets'], cohort.prediction_row.to_numpy(int))
    oracle_truth = np.where(eligible, oracle_truth, -1).astype(np.int8)
    oracle = decision_outcomes(oracle_truth, eligible, p['rank'], p['count'], rare)
    q = oracle.sample_F1.to_numpy(); rare_truth = oracle.rare_truth.to_numpy()
    np.savez_compressed(DATA/'oracle.npz', truth=oracle_truth, F1=q, rare_truth=rare_truth)
    chosen = {name: p['accept_'+name] for name in config['selectors']}
    bootstrap = {k: v for k, v in config['bootstrap'].items() if k != 'cluster'}
    oracle_report = {'full_F1': float(q.mean()), 'cells': len(q), 'spatial_blocks': cohort.spatial_block.nunique(),
        'countries': cohort.country.nunique(), 'rare_occurrences': int(rare_truth.sum()),
        'selected_F1': {k: float(q[v].mean()) for k, v in chosen.items()},
        'accepted_cells': {k: int(v.sum()) for k, v in chosen.items()}, 'contrasts': {}}
    for a, b in config['contrasts']:
        oracle_report['contrasts'][a+'_'+b] = paired_spatial_interval(q, cohort.spatial_block, chosen[a], chosen[b], **bootstrap)
    save(OUT/'oracle_summary.json', oracle_report)
    arrays = {k: [] for k in ['unknown_as_absence', 'known_labels_only', 'lower', 'upper',
        'known_fraction', 'hidden_positive_fraction', 'admitted', 'realized_genus_fraction']}
    rows, country_rows, receipts = [], [], []
    for mechanism in config['mechanisms']:
        weights = np.ones_like(contract['genus_weights']) if mechanism.startswith('uniform') else contract['genus_weights']
        for dose in config['doses']:
            for seed in config['mask_seeds']:
                known, n_genera, hidden_genera = censor_genera(oracle_truth, eligible, contract['genus_codes'],
                    weights, cohort.cell_id, dose, seed)
                truth = np.where(known, oracle_truth, -1)
                scored = score_fixed_sets(truth, known, p['rank'], p['count'])
                denominator = p['count']+scored['known_truth_count']
                naive = np.divide(2*scored['hits'], denominator, out=np.ones(len(q)), where=denominator > 0)
                low, high = fixed_set_f1_bounds(truth, known, eligible, p['rank'], p['count'])
                assert np.all(low <= q+1e-12) and np.all(q <= high+1e-12)
                fraction = known.sum(1)/eligible.sum(1)
                hidden_positive = np.divide(((oracle_truth > 0)&~known).sum(1), oracle.true_richness,
                    out=np.zeros(len(q)), where=oracle.true_richness.to_numpy() > 0)
                realized = np.divide(hidden_genera, n_genera, out=np.zeros(len(q)), where=n_genera > 0)
                admitted = fraction >= config['minimum_known_label_fraction_diagnostic']
                values = [naive, scored['F1'], low, high, fraction, hidden_positive, admitted, realized]
                for key, value in zip(arrays, values):
                    arrays[key].append(value)
                row = {'mechanism': mechanism, 'dose': dose, 'seed': seed,
                    'known_fraction_mean': float(fraction.mean()), 'known_fraction_median': float(np.median(fraction)),
                    'hidden_positive_fraction_mean': float(hidden_positive.mean()),
                    'hidden_positive_occurrence_fraction': float(((oracle_truth > 0)&~known).sum()/(oracle_truth > 0).sum()),
                    'realized_genus_fraction_mean': float(realized.mean()), 'zero_positive_cells': int((n_genera == 0).sum()),
                    'bound_width_mean': float((high-low).mean()), 'admitted_cells': int(admitted.sum()),
                    'admitted_countries': int(cohort.loc[admitted, 'country'].nunique()),
                    'admitted_blocks': int(cohort.loc[admitted, 'spatial_block'].nunique()),
                    'admitted_rare_occurrence_retention': float(rare_truth[admitted].sum()/max(rare_truth.sum(), 1)),
                    'admitted_oracle_F1_shift': float(q[admitted].mean()-q.mean()) if admitted.any() else None}
                for treatment, scores in [('unknown_as_absence', naive), ('known_labels_only', scored['F1'])]:
                    row[treatment+'_F1'] = float(scores.mean()); row[treatment+'_bias'] = float((scores-q).mean())
                    row[treatment+'_admitted_bias'] = float((scores-q)[admitted].mean()) if admitted.any() else None
                    for name, mask in chosen.items():
                        row[treatment+'_selective_F1_'+name] = float(scores[mask].mean())
                    for a, b in config['contrasts']:
                        key = treatment+'_'+a+'_'+b
                        value = float(scores[chosen[a]].mean()-scores[chosen[b]].mean())
                        oracle_value = oracle_report['contrasts'][a+'_'+b]['point']
                        row[key] = value; row[key+'_distortion'] = value-oracle_value
                        row[key+'_sign_reversal'] = bool(abs(oracle_value) >= config['practical_oracle_contrast_for_reversal']
                                                       and value*oracle_value < 0)
                for a, b in config['contrasts']:
                    w = chosen[a]/chosen[a].sum()-chosen[b]/chosen[b].sum()
                    bounds = linear_contrast_bounds(low, high, w)
                    oracle_value = oracle_report['contrasts'][a+'_'+b]['point']
                    assert bounds[0]-1e-12 <= oracle_value <= bounds[1]+1e-12
                    key = 'bound_'+a+'_'+b
                    row[key+'_lower'], row[key+'_upper'] = bounds
                    row[key+'_sign_identified'] = bool(bounds[0] > 0 or bounds[1] < 0)
                for country in sorted(cohort.country.unique()):
                    take = cohort.country.eq(country).to_numpy()
                    country_rows.append({'mechanism': mechanism, 'dose': dose, 'seed': seed, 'country': country,
                        'cells': int(take.sum()), 'admitted_cells': int((take&admitted).sum()),
                        'retention': float(admitted[take].mean()),
                        'rare_occurrence_retention': float(rare_truth[take&admitted].sum()/rare_truth[take].sum()) if rare_truth[take].sum() else None})
                rows.append(row)
                receipts.append({'mechanism': mechanism, 'dose': dose, 'seed': seed,
                    'known_mask_sha256': hashlib.sha256(np.packbits(known, bitorder='little').tobytes()).hexdigest()})
            print(json.dumps({'completed': mechanism, 'dose': dose, 'settings': len(rows)}), flush=True)
    stacked = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(DATA/'replay_cells.npz', **stacked)
    table = pd.DataFrame(rows); table.to_csv(OUT/'replay_summary.csv', index=False)
    pd.DataFrame(country_rows).to_csv(OUT/'country_admission.csv', index=False)
    save(OUT/'mask_receipts.json', receipts)
    summaries = []
    for mechanism in config['mechanisms']:
        for dose in config['doses']:
            take = np.flatnonzero((table.mechanism == mechanism)&(table.dose == dose))
            section = {'mechanism': mechanism, 'dose': dose, 'seeds': len(take), 'treatments': {}, 'diagnostics': {}}
            for key in ['known_fraction_mean', 'hidden_positive_occurrence_fraction', 'realized_genus_fraction_mean',
                        'bound_width_mean', 'admitted_cells', 'admitted_countries', 'admitted_blocks',
                        'admitted_rare_occurrence_retention', 'admitted_oracle_F1_shift']:
                section['diagnostics'][key] = seed_stats(table.iloc[take][key])
            for treatment in ['unknown_as_absence', 'known_labels_only']:
                mean_scores = stacked[treatment][take].mean(0)
                bias = paired_spatial_interval(mean_scores-q, cohort.spatial_block, **bootstrap)
                outcome = {'bias': bias, 'F1': seed_stats(table.iloc[take][treatment+'_F1']),
                    'bias_seed_variability': seed_stats(table.iloc[take][treatment+'_bias']),
                    'admitted_bias': seed_stats(table.iloc[take][treatment+'_admitted_bias']), 'contrasts': {}}
                for a, b in config['contrasts']:
                    key = a+'_'+b; column = treatment+'_'+key
                    contrast = paired_spatial_interval(mean_scores, cohort.spatial_block, chosen[a], chosen[b], **bootstrap)
                    distortion = paired_spatial_interval(mean_scores-q, cohort.spatial_block, chosen[a], chosen[b], **bootstrap)
                    outcome['contrasts'][key] = {'observed': contrast, 'distortion': distortion,
                        'distortion_seed_variability': seed_stats(table.iloc[take][column+'_distortion']),
                        'sign_reversals_of_practical_oracle': int(table.iloc[take][column+'_sign_reversal'].sum()),
                        'completion_lower': seed_stats(table.iloc[take]['bound_'+key+'_lower']),
                        'completion_upper': seed_stats(table.iloc[take]['bound_'+key+'_upper']),
                        'completion_sign_identified_seeds': int(table.iloc[take]['bound_'+key+'_sign_identified'].sum())}
                section['treatments'][treatment] = outcome
            summaries.append(section)
    def passes(section):
        threshold = config['practical_score_bias']
        return any(t['bias']['interval'][1] < -threshold or t['bias']['interval'][0] > threshold
                   for t in section['treatments'].values())
    moderate = [s for s in summaries if s['dose'] == .2]
    result = {'status': 'controlled_development_complete', 'at_utc': now(), 'oracle': oracle_report,
        'replay_settings': len(rows), 'nonzero_settings': int(table.dose.gt(0).sum()),
        'unique_zero_dose_control': 1, 'new_backbone_fits': 0, 'new_GPU_jobs': 0, 'reconstructed_HGB_fits': 1,
        'practical_score_bias_criterion_met': all(passes(s) for s in moderate),
        'sections': summaries, 'new_method_or_independent_confirmation': False,
        'interpretation': 'synthetic informative genus censoring of consumed recorded benchmark labels; spatial intervals conditional on fixed policies; mask seeds are not independent ecological samples'}
    save(OUT/'summary.json', result)
    print(json.dumps({'status': result['status'], 'practical_score_bias_criterion_met': result['practical_score_bias_criterion_met']}), flush=True)


def seal():
    if not json.loads((OUT/'validation.json').read_text())['passed']:
        raise ValueError('Validation required')
    # Include compact reports and local input/output caches, preserving old manifests.
    paths = {ROOT/n for n in SOURCES}
    paths.update(DATA.rglob('*')); paths.update(OUT.rglob('*'))
    paths.add(ROOT/'docs/taxonomic_resolution_t1_results.md')
    manifest = OUT/'artifact_manifest.json'
    save(manifest, {'at_utc': now(), 'path_base': 'repository root', 'files': {
        str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(paths) if p.is_file() and p != manifest and '__pycache__' not in str(p)}})
    print('Sealed T1 artifacts')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'policy', 'replay', 'seal'])
    args = parser.parse_args(); config = json.loads((ROOT/'configs/taxonomic_resolution_t1.json').read_text())
    if args.phase == 'freeze':
        print(json.dumps(freeze()))
    elif args.phase == 'policy':
        policy(config)
    elif args.phase == 'replay':
        replay(config)
    else:
        seal()


if __name__ == '__main__':
    main()
