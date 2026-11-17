"""Two-stage locked external ecological-service replication on released SAGE fits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import hdf5plugin  # registers release compression
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from geolifeclef.ecological_selection import QuotaSelection, paired_ecological_comparison
from geolifeclef.external_reliability import (META_FEATURES, RISK_FEATURES, apply_calibration,
    choose_threshold, decode_counts, decision_features, decision_outcomes, fit_calibration,
    guarded_target_rows, joint_ecological_comparison, prediction_context, read_rows)
from geolifeclef.reliability import r1_hash

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/external_replication_e1'
OUT = ROOT/'results/external_replication_e1'
METHODS = ['random', 'mean_confidence', 'entropy', 'SDC', 'lean_risk',
           'raw_rare_constraint', 'rare_count_constraint', 'total_count_constraint']
SOURCES = ['configs/external_reliability_e1.json', 'docs/external_reliability_e1_protocol.md',
    'scripts/run_external_e1.py', 'scripts/prepare_external_e1.py',
    'src/geolifeclef/external_reliability.py', 'src/geolifeclef/ecological_selection.py',
    'src/geolifeclef/reliability.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def freeze():
    sources = {s: sha(ROOT/s) for s in SOURCES}
    inputs = {s: sha(DATA/s) for s in ['cohort_roles.csv', 'eligible_species.npy',
        'rare_species_mask.npy', 'test_indices.npy', 'sealed_eval_test_targets.h5']}
    path = OUT/'protocol_freeze.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['source_hashes'] != sources or old['input_hashes'] != inputs:
            raise RuntimeError('Locked source/input changed; amendment and new run directory required')
        return old
    snap = OUT/'source_snapshot';snap.mkdir(parents=True, exist_ok=True)
    for s in SOURCES:
        target = snap/s;target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(ROOT/s, target)
    result = {'locked_at_utc': now(), 'assessment_targets_accessed': False,
        'source_hashes': sources, 'input_hashes': inputs,
        'scope': 'local prospective lock after GLC development, not third-party registration'}
    save(path, result)
    return result


def target_rows(cohort, role, phase, admitted=False):
    with h5py.File(DATA/'sealed_eval_test_targets.h5') as f:
        return guarded_target_rows(f['targets'], cohort, role, phase, admitted)


def validate_metadata(cohort):
    if cohort.cell_id.duplicated().any() or cohort.prediction_row.duplicated().any():
        raise ValueError('Duplicate identity')
    if cohort.groupby('spatial_block').role.nunique().max() != 1:
        raise ValueError('Spatial block leaked across roles')
    from sklearn.neighbors import BallTree
    distances = {}
    roles = cohort.role.unique()
    for i, a in enumerate(roles):
        for b in roles[i+1:]:
            ac = np.deg2rad(cohort.loc[cohort.role.eq(a), ['lat','lon']])
            bc = np.deg2rad(cohort.loc[cohort.role.eq(b), ['lat','lon']])
            distance = float(BallTree(ac, metric='haversine').query(bc)[0].min()*6371.0088)
            if distance < 50:
                raise ValueError('Cross-role buffer failed')
            distances[a+'__'+b] = distance
    with h5py.File(DATA/'sealed_eval_test_targets.h5') as f:
        locations = read_rows(f['locations'], cohort.prediction_row.to_numpy())
        area = read_rows(f['sampled_area_m2'], cohort.prediction_row.to_numpy())
    if not np.allclose(locations, cohort[['lon','lat']], atol=1e-5, rtol=0):
        raise ValueError('Published test target row locations do not match cell IDs')
    if not np.allclose(area, cohort.area, atol=.1, rtol=1e-4):
        raise ValueError('Published target areas do not match contributing plot metadata')
    return {'minimum_cross_role_km': distances, 'locations_and_areas_match': True}


def check_mask(y, eligible):
    if not np.array_equal(y >= 0, eligible):
        raise ValueError('Native mask differs from published missing-target convention')
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('Non-binary published targets')


def fit(config, cohort, eligible, rare):
    freeze()
    if (OUT/'evaluation_admission.json').exists():
        raise RuntimeError('Policies already frozen; use assess, never refit inspected assessment')
    save(OUT/'metadata_validation.json', validate_metadata(cohort))
    # These are the only PA outcomes materialized during policy fitting.
    cal = np.flatnonzero(cohort.role.eq('probability_calibration'))
    risk = np.flatnonzero(cohort.role.eq('risk_calibration'))
    assess = np.flatnonzero(cohort.role.eq('assessment'))
    yc, yr = target_rows(cohort, 'probability_calibration', 'fit'), target_rows(cohort, 'risk_calibration', 'fit')
    check_mask(yc, eligible[cal]);check_mask(yr, eligible[risk])
    secondary = (yc > 0).sum(0);secondary = (secondary >= 1) & (secondary <= 25)
    np.save(DATA/'secondary_rare_mask.npy', secondary, allow_pickle=False)
    rare_risk = int((yr[:, rare] > 0).sum())
    admission = rare_risk >= config['minimum_risk_calibration_rare_occurrences']
    save(OUT/'calibration_admission.json', {'at_utc': now(), 'riskcal_rare_occurrences': rare_risk,
        'primary_admitted': admission, 'secondary_rare_species': int(secondary.sum()),
        'assessment_targets_accessed': False})
    ameta = cohort.iloc[assess].reset_index(drop=True)
    groups = (ameta.country.astype(str)+'|'+ameta.area_bin.astype(str)).to_numpy()
    selectors = {str(c): QuotaSelection(ameta.cell_id, c, groups) for c in config['coverage_grid']}
    policy_hashes = {}
    for model in config['models']:
        started = time.monotonic()
        destination = OUT/'policies'/f'{model}.npz';destination.parent.mkdir(exist_ok=True)
        detail = destination.with_suffix('.json')
        if destination.exists() and detail.exists():
            policy_hashes[model] = {'npz': sha(destination), 'json': sha(detail)}
            print(json.dumps({'model': model, 'cached_frozen_policy': True}), flush=True)
            continue
        prediction_file = DATA/model/'test_predictions.h5'
        print(json.dumps({'model': model, 'stage': 'read_predictions'}), flush=True)
        with h5py.File(prediction_file) as f:
            p = read_rows(f['probabilities'], cohort.prediction_row.to_numpy())
        calibration = fit_calibration(p[cal], yc, eligible[cal], config['calibration_bounds'])
        p = apply_calibration(p, calibration, eligible)
        ctx = prediction_context(p, eligible)
        cctx = {k: v[cal] for k,v in ctx.items()}
        threshold, grid = choose_threshold(cctx, yc, eligible[cal], rare, config['threshold_grid'])
        arrays = {'cell_id': ameta.cell_id.to_numpy(), 'rank': ctx['rank'][assess].astype(np.int16)}
        keys, masks, counts, proxy_records = [], [], [], []
        calibration_metrics = []
        for decoder in config['decoders']:
            count = decode_counts(ctx, decoder, threshold)
            features = decision_features(ctx, count, cohort[META_FEATURES[:-1]], rare)
            out = decision_outcomes(yr, eligible[risk], ctx['rank'][risk], count[risk], rare)
            predictors = {}
            for name, response in [('lean_risk', 'loss'), ('rare_count', 'rare_truth'), ('total_count', 'true_richness')]:
                reg = HistGradientBoostingRegressor(**config['risk_regressor'])
                reg.fit(features.iloc[risk][RISK_FEATURES], out[response])
                predictors[name] = reg.predict(features.iloc[assess][RISK_FEATURES])
                if name != 'lean_risk':
                    predictors[name] = np.maximum(predictors[name], 0)
            af = features.iloc[assess]
            risks = {'random': np.array([r1_hash('E1-random:'+str(i))/2**64 for i in ameta.cell_id]),
                'mean_confidence': 1-af.mean_set_confidence.to_numpy(),
                'entropy': af.set_entropy.to_numpy(), 'SDC': 1-af.SDC.to_numpy(),
                'lean_risk': predictors['lean_risk']}
            counts.append(count[assess])
            calibration_metrics.append({'decoder': decoder, 'riskcal_F1': float(out.sample_F1.mean()),
                'mean_riskcal_richness': float(out.true_richness.mean())})
            for coverage in config['coverage_grid']:
                selector = selectors[str(coverage)]
                chosen = {name: selector.accept(value) for name,value in risks.items()}
                for name, mass in [('raw_rare_constraint', af.predicted_rare_mass.to_numpy()),
                                   ('rare_count_constraint', predictors['rare_count']),
                                   ('total_count_constraint', predictors['total_count'])]:
                    chosen[name], audit = selector.preserve_mass(risks['SDC'], mass)
                    proxy_records.append({'decoder': decoder, 'coverage': coverage, 'method': name, **audit})
                for method in METHODS:
                    keys.append(f'{decoder}|{coverage}|{method}');masks.append(chosen[method])
                reference = chosen['SDC']
                for indices, _ in selector.groups:
                    if any(chosen[m][indices].sum() != reference[indices].sum() for m in METHODS):
                        raise AssertionError('Unequal group budgets')
            arrays[decoder+'_features'] = af[RISK_FEATURES+['set_entropy']].to_numpy(np.float32)
            arrays[decoder+'_proxies'] = np.column_stack([predictors['rare_count'], predictors['total_count']])
        arrays.update(keys=np.array(keys), accept=np.array(masks), counts=np.array(counts),
                      decoders=np.array(config['decoders']))
        np.savez_compressed(destination, **arrays)
        save(detail, {'model': model, 'frozen_at_utc': now(), 'prediction_sha256': sha(prediction_file),
            'calibration': calibration, 'threshold': threshold, 'threshold_grid': grid,
            'riskcal_metrics': calibration_metrics, 'proxy_constraints': proxy_records,
            'risk_features': RISK_FEATURES, 'seconds': time.monotonic()-started,
            'assessment_targets_accessed': False})
        policy_hashes[model] = {'npz': sha(destination), 'json': sha(detail)}
        print(json.dumps({'model': model, 'stage': 'policy_frozen', 'seconds': time.monotonic()-started}), flush=True)
        del p, ctx, cctx
    save(OUT/'evaluation_admission.json', {'admitted_at_utc': now(), 'protocol_sha256': sha(OUT/'protocol_freeze.json'),
        'policy_hashes': policy_hashes, 'assessment_targets_accessed': False,
        'secondary_rare_mask_sha256': sha(DATA/'secondary_rare_mask.npy'),
        'primary_calibration_admitted': admission})
    print('All assessment policies frozen; assessment outcomes remain closed.', flush=True)


def summarize(out, counts, accepted, rare_total):
    take = out.loc[accepted]
    n = len(take);truth = int(take.true_richness.sum());hits = int(take.hits.sum());pred = int(counts[accepted].sum())
    rh = int(take.rare_hits.sum());rp = int(take.rare_predictions.sum());rt = int(take.rare_truth.sum())
    return {'retained': n, 'sample_F1': float(take.sample_F1.mean()),
        'sample_F1_zero_empty': float(take.sample_F1_zero_empty.mean()),
        'micro_F1': 2*hits/max(pred+truth, 1), 'richness_MAE': float(np.abs(counts[accepted]-take.true_richness).mean()),
        'rare_occurrence_coverage': rt/rare_total if rare_total else None,
        'rare_true_positives': rh, 'rare_precision': rh/rp if rp else None,
        'rare_recall_accepted': rh/rt if rt else None, 'rare_recall_all': rh/rare_total if rare_total else None,
        'predicted_empty_fraction': float((counts[accepted] == 0).mean()),
        'observed_empty_fraction': float(take.true_richness.eq(0).mean())}


def assess(config, cohort, eligible, rare):
    freeze()  # rejects silent changes since fitting
    admission = json.loads((OUT/'evaluation_admission.json').read_text())
    if admission['protocol_sha256'] != sha(OUT/'protocol_freeze.json'):
        raise ValueError('Protocol lock changed')
    if admission['secondary_rare_mask_sha256'] != sha(DATA/'secondary_rare_mask.npy'):
        raise ValueError('Calibration-derived secondary rarity changed')
    for model, checks in admission['policy_hashes'].items():
        for suffix, digest in checks.items():
            if sha(OUT/'policies'/f'{model}.{suffix}') != digest:
                raise ValueError('Frozen assessment policy changed')
    access = OUT/'assessment_access.json'
    if not access.exists():
        save(access, {'first_assessment_target_access_utc': now(), 'all_policies_previously_frozen': True,
            'admission_sha256': sha(OUT/'evaluation_admission.json')})
    indices = np.flatnonzero(cohort.role.eq('assessment'));meta = cohort.iloc[indices].reset_index(drop=True)
    y = target_rows(cohort, 'assessment', 'assess', True);check_mask(y, eligible[indices])
    secondary = np.load(DATA/'secondary_rare_mask.npy', allow_pickle=False)
    rows, regions, individual = [], [], {}
    primary_scores, primary_a, primary_b = [], [], []
    rare_truth = (y[:, rare] > 0).sum(1)
    np.savez_compressed(OUT/'assessment_truth_summaries.npz', cell_id=meta.cell_id.to_numpy(),
                        rare_truth=rare_truth, true_richness=(y > 0).sum(1))
    for model in config['models']:
        with np.load(OUT/'policies'/f'{model}.npz', allow_pickle=False) as policy:
            if not np.array_equal(policy['cell_id'], meta.cell_id):
                raise ValueError('Policy row identity changed')
            per_model = {}
            keys = policy['keys'].tolist()
            for d, decoder in enumerate(config['decoders']):
                count = policy['counts'][d]
                out = decision_outcomes(y, eligible[indices], policy['rank'], count, rare)
                out_secondary = decision_outcomes(y, eligible[indices], policy['rank'], count, secondary)
                for coverage in config['coverage_grid']:
                    baseline = policy['accept'][keys.index(f'{decoder}|{coverage}|SDC')]
                    for method in METHODS:
                        accepted = policy['accept'][keys.index(f'{decoder}|{coverage}|{method}')]
                        result = summarize(out, count, accepted, int(rare_truth.sum()))
                        sr = int(out_secondary.rare_truth.sum())
                        result['secondary_rare_occurrence_coverage'] = float(out_secondary.loc[accepted, 'rare_truth'].sum()/sr) if sr else None
                        rows.append({'model': model, 'decoder': decoder, 'coverage': coverage, 'method': method, **result})
                        if coverage == .8 and method in ['SDC','rare_count_constraint','raw_rare_constraint','total_count_constraint']:
                            for dimension in ['country','continent']:
                                for name, positions in meta.groupby(dimension).groups.items():
                                    ix = np.array(list(positions));a = accepted[ix]
                                    sub = out.iloc[ix].reset_index(drop=True)
                                    regions.append({'model': model, 'decoder': decoder, 'method': method,
                                        'dimension': dimension, 'name': name, 'cells': len(ix),
                                        **summarize(sub, count[ix], a, int(sub.rare_truth.sum()))})
                    if decoder == config['primary_decoder'] and coverage == config['primary_coverage']:
                        candidate = policy['accept'][keys.index(f'{decoder}|{coverage}|rare_count_constraint')]
                        primary_scores.append(out.sample_F1.to_numpy());primary_a.append(candidate);primary_b.append(baseline)
                        per_model = paired_ecological_comparison(out.sample_F1, rare_truth, candidate,
                            baseline, meta.spatial_block, confidence=.975)
            individual[model] = per_model
    table = pd.DataFrame(rows);table.to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(regions).to_csv(OUT/'regional_summary.csv', index=False)
    joint = joint_ecological_comparison(primary_scores, rare_truth, primary_a, primary_b, meta.spatial_block)
    points = table[(table.decoder == config['primary_decoder']) & table.coverage.eq(config['primary_coverage']) & table.method.eq('rare_count_constraint')]
    targets = config['practical_targets']
    checks = {'calibration_admitted': admission['primary_calibration_admitted'],
        'minimum_mean_rare_gain': joint['rare_coverage_gain'] >= targets['minimum_rare_occurrence_coverage_gain'],
        'rare_gain_interval_positive': joint['rare_coverage_interval'] is not None and joint['rare_coverage_interval'][0] > 0,
        'F1_interval_noninferior': joint['F1_interval'] is not None and joint['F1_interval'][0] >= -targets['maximum_absolute_F1_loss'],
        'all_model_point_F1_noninferior': min(joint['model_F1_gains']) >= -targets['maximum_absolute_F1_loss'],
        'all_models_minimum_realized_rare_coverage': bool(points.rare_occurrence_coverage.ge(targets['minimum_realized_rare_occurrence_coverage']).all())}
    save(OUT/'report.json', {'status': 'completed_locked_external_analysis', 'completed_at_utc': now(),
        'assessment_cells': len(meta), 'rare_occurrences': int(rare_truth.sum()), 'rare_species': int(rare.sum()),
        'primary_success': bool(all(checks.values())), 'primary_checks': checks,
        'primary_joint_comparison': joint, 'model_primary_comparisons': individual,
        'new_neural_fits': 0, 'competition_submissions': 0,
        'limitations': ['Published pretrained backbone training can include evaluation cells',
            'Public benchmark subset; locally locked analysis, not secret test',
            'Fixed-model conditional spatial intervals; not biological causality or coverage guarantee',
            'Rarity means low recorded PO prevalence, not conservation threat status',
            'Other decoder and method contrasts are descriptive secondary analyses']})
    print(json.dumps({'primary_success': all(checks.values()), 'checks': checks, 'joint': joint}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('phase', choices=['fit','assess'])
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    cohort = pd.read_csv(DATA/'cohort_roles.csv', dtype={'area_bin': str})
    eligible = np.load(DATA/'eligible_species.npy', allow_pickle=False)
    rare = np.load(DATA/'rare_species_mask.npy', allow_pickle=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (fit if args.phase == 'fit' else assess)(config, cohort, eligible, rare)


if __name__ == '__main__':
    main()
