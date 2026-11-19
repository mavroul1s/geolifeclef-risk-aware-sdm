"""Bounded shared-concept verification replay; frozen predictions, zero fits."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from geolifeclef.taxonomic_queries import QueryState, acquisition_order, contrast_interval, simulate_keys
from geolifeclef.external_reliability import read_rows
from run_taxonomic_resolution_t1 import sha, save, verify_history
from validate_taxonomy_audits import verify_manifest

DATA, OUT = ROOT/'data/taxonomic_verification_t2', ROOT/'results/taxonomic_verification_t2'
SOURCES = ['configs/taxonomic_verification_t2.json', 'docs/taxonomic_verification_t2_protocol.md',
    'docs/taxonomic_verification_t2_prior_work.md', 'src/geolifeclef/taxonomic_queries.py',
    'tests/test_taxonomic_queries.py', 'scripts/run_taxonomic_verification_t2.py',
    'scripts/validate_taxonomic_verification_t2.py', 'scripts/plot_taxonomic_verification_t2.py']
INPUTS = ['data/taxonomic_resolution_t1/'+n for n in
    ['fixed_policy.npz', 'contract.npz', 'assessment_roles.csv', 'oracle.npz']]
INPUTS += ['data/external_replication_e1/'+n for n in
    ['cohort_roles.csv', 'eligible_species.npy', 'sealed_eval_test_targets.h5']]
INPUTS += ['src/geolifeclef/external_reliability.py', 'scripts/run_taxonomic_resolution_t1.py',
    'scripts/validate_taxonomy_audits.py', 'results/taxonomic_resolution_t1/policy_freeze.json']
EXTRA_MANIFESTS = ['taxonomic_resolution_t1', 'neon_taxonomy_a1', 'neon_taxonomy_a2',
                   'neon_taxonomy_a3', 'neon_taxonomy_a3u', 'publication_validation_2026_11_19']


def now(): return datetime.now(timezone.utc).isoformat()


def freeze():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    identity = {n: sha(ROOT/n) for n in SOURCES+INPUTS}
    path = OUT/'protocol_freeze.json'
    if path.exists():
        assert json.loads(path.read_text())['identity'] == identity, 'Frozen T2 source/input changed'
        return
    old = verify_history()
    for name in EXTRA_MANIFESTS:
        p = ROOT/'results'/name/'artifact_manifest.json'
        old[name] = {'files_verified': verify_manifest(p), 'manifest_sha256': sha(p)}
    for name in SOURCES:
        target = OUT/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    save(path, {'at_utc': now(), 'identity': identity, 'earlier_artifacts': old,
        'role': 'consumed-cohort development; no fresh outcomes or new fitting', 'T2_replay_started': False})


def pack_case(keys, answers, orders, path):
    offsets = np.r_[0, np.cumsum([len(c) for c in keys.cells])]
    cells = np.concatenate(keys.cells) if len(keys.cells) else np.array([], int)
    np.savez_compressed(path, genus=keys.genus, offsets=offsets, cells=cells, answer=answers,
                        **{'order_'+k: v for k, v in orders.items()})


def sign(lo, hi, margin):
    return 1 if lo > margin else (-1 if hi < -margin else 0)


def replay(config):
    freeze()
    if (OUT/'replay_access.json').exists():
        raise ValueError('T2 replay already started; preserve partial/completed results')
    started = time.monotonic()
    save(OUT/'replay_access.json', {'at_utc': now(), 'protocol_sha256': sha(OUT/'protocol_freeze.json'),
        'all_outcomes_already_consumed': True})
    cohort = pd.read_csv(ROOT/'data/taxonomic_resolution_t1/assessment_roles.csv')
    source = pd.read_csv(ROOT/'data/external_replication_e1/cohort_roles.csv')
    take = source.role.eq('assessment').to_numpy()
    assert source.loc[take, 'cell_id'].tolist() == cohort.cell_id.tolist()
    native = np.load(ROOT/'data/external_replication_e1/eligible_species.npy')[take]
    contract = np.load(ROOT/'data/taxonomic_resolution_t1/contract.npz')
    eligible, universe, genera, rare = [contract[k] for k in ['eligible', 'universe', 'genus_codes', 'rare']]
    np.testing.assert_array_equal(eligible, native & universe)
    with h5py.File(ROOT/'data/external_replication_e1/sealed_eval_test_targets.h5') as f:
        full = read_rows(f['targets'], cohort.prediction_row.to_numpy(int))
    truth = (full > 0) & eligible
    original = np.load(ROOT/'data/taxonomic_resolution_t1/oracle.npz')
    np.testing.assert_array_equal(np.where(eligible, full, -1), original['truth'])
    policy = np.load(ROOT/'data/taxonomic_resolution_t1/fixed_policy.npz')
    pred = np.zeros_like(eligible)
    for i, n in enumerate(policy['count']): pred[i, policy['rank'][i, :n]] = True
    a, b = policy['accept_HGB'], policy['accept_SDC']
    assert len(cohort) == 5251 and a.sum() == b.sum() == 4201
    weights = a/a.sum()-b/b.sum()
    den = pred.sum(1)+truth.sum(1)
    quality = np.divide(2*(pred & truth).sum(1), den, out=np.ones(len(den)), where=den > 0)
    np.testing.assert_array_equal(quality, original['F1'])
    oracle_contrast = float(weights@quality)
    np.savez_compressed(DATA/'oracle_contract.npz', eligible=eligible, prediction=pred, truth=truth,
        weights=weights, quality=quality, genera=genera, rare=rare)
    cases, curves, events = [], [], []
    country = cohort.country.astype(str).to_numpy()
    for seed in config['mask_seeds']:
        initial, problems = simulate_keys(full, native, universe, genera, cohort.spatial_block,
            seed, config['hidden_concept_probability'])
        np.savez_compressed(DATA/f'initial_{seed}.npz', positive=np.packbits(initial, axis=1))
        previous_known = None
        for unit in config['query_units']:
            keys, answers = problems[unit]
            state = QueryState(eligible, initial, genera, pred, keys)
            if previous_known is not None: np.testing.assert_array_equal(state.known, previous_known)
            previous_known = state.known.copy()
            orders = {name: acquisition_order(state, weights, name, seed+101)[0] for name in config['policies']}
            pack_case(keys, answers, orders, DATA/f'case_{seed}_{unit}.npz')
            cases.append({'seed': seed, 'unit': unit, 'hidden_keys': len(answers),
                'hidden_records': sum(map(len, keys.cells)), 'outside_target_keys': int((answers < 0).sum()),
                'max_linked_cells': max(map(len, keys.cells), default=0),
                'mean_linked_cells': float(np.mean(list(map(len, keys.cells)))) if len(answers) else 0.,
                'initial_known_fraction': float(state.known.sum()/eligible.sum()),
                'initial_hidden_rare': int((truth & ~state.positive & rare).sum())})
            for name, order in orders.items():
                state = QueryState(eligible, initial, genera, pred, keys)
                lower, upper = state.bounds(); lo, hi = contrast_interval(lower, upper, weights)
                thresholds = {f: min(int(np.ceil(f*len(answers))), len(order)) for f in config['budget_fractions']}
                first_width = 0 if hi-lo <= config['width_tolerance'] else None
                first_sign = 0 if sign(lo, hi, config['practical_contrast_margin']) else None
                record_cost = outside_cost = 0
                affected = np.zeros(len(cohort), bool)
                old_lo, old_hi = lo, hi
                for charged in range(len(order)+1):
                    if time.monotonic()-started > config['maximum_replay_seconds']:
                        raise TimeoutError('Frozen T2 replay budget exceeded; preserve partial results')
                    if charged:
                        key = int(order[charged-1]); rows = keys.cells[key]
                        local_lo, local_hi = state.bounds(rows)
                        before_lo, before_hi = contrast_interval(local_lo, local_hi, weights[rows])
                        state.query(key, int(answers[key]))
                        local_lo, local_hi = state.bounds(rows)
                        after_lo, after_hi = contrast_interval(local_lo, local_hi, weights[rows])
                        lo += after_lo-before_lo; hi += after_hi-before_hi
                        record_cost += len(rows); outside_cost += int(answers[key] < 0); affected[rows] = True
                        assert lo >= old_lo-1e-11 and hi <= old_hi+1e-11
                        old_lo, old_hi = lo, hi
                    if first_width is None and hi-lo <= config['width_tolerance']: first_width = charged
                    if first_sign is None and sign(lo, hi, config['practical_contrast_margin']): first_sign = charged
                    fractions = [f for f, n in thresholds.items() if n == charged]
                    if not fractions: continue
                    lower, upper = state.bounds(); check_lo, check_hi = contrast_interval(lower, upper, weights)
                    assert np.all(lower <= quality+1e-12) and np.all(quality <= upper+1e-12)
                    np.testing.assert_allclose([lo, hi], [check_lo, check_hi], atol=1e-10)
                    declared = sign(check_lo, check_hi, config['practical_contrast_margin'])
                    assert not declared or declared*oracle_contrast > config['practical_contrast_margin']
                    for fraction in fractions:
                        curves.append({'seed': seed, 'unit': unit, 'policy': name, 'budget_fraction': fraction,
                            'queries': charged, 'queried_records': record_cost, 'outside_target_queries': outside_cost,
                            'affected_cells': int(affected.sum()), 'queried_countries': len(set(country[affected])),
                            'lower': check_lo, 'upper': check_hi, 'width': check_hi-check_lo,
                            'mean_cell_width': float(np.mean(upper-lower)), 'known_fraction': float(state.known.sum()/eligible.sum()),
                            'known_rare_positives': int((state.positive & rare).sum()), 'practical_sign': declared,
                            'oracle_contained': True})
                if name in {'uniform', 'global_width'}:
                    np.testing.assert_array_equal(state.known, eligible)
                    np.testing.assert_array_equal(state.positive, truth)
                assert abs(hi-lo) <= 1e-9, 'All relevant queries must identify the fixed contrast'
                events.append({'seed': seed, 'unit': unit, 'policy': name,
                    'first_width_queries': first_width, 'first_practical_sign_queries': first_sign,
                    'available_keys': len(answers), 'executed_queries': len(order),
                    'order_sha256': hashlib.sha256(np.asarray(order, '<i8').tobytes()).hexdigest()})
            pd.DataFrame(curves).to_csv(OUT/'curves.csv', index=False)
            pd.DataFrame(events).to_csv(OUT/'query_events.csv', index=False)
            pd.DataFrame(cases).to_csv(OUT/'case_metadata.csv', index=False)
            print(json.dumps({'seed': seed, 'unit': unit, 'keys': len(answers),
                'trajectories_complete': len(events), 'seconds': round(time.monotonic()-started, 2)}), flush=True)
    frame = pd.DataFrame(curves); primary = frame[frame.budget_fraction.eq(config['primary_budget_fraction'])]
    comparisons = {}; passed = True
    rng = np.random.default_rng(config['bootstrap']['seed'])
    draws = rng.integers(len(config['mask_seeds']), size=(config['bootstrap']['repetitions'], len(config['mask_seeds'])))
    tail = (1-config['bootstrap']['confidence'])/2
    for unit in config['query_units']:
        table = primary[primary.unit.eq(unit)].pivot(index='seed', columns='policy', values='width')
        comparisons[unit] = {}
        for control in config['primary_controls']:
            reduction = table[control].to_numpy()-table.paired_width.to_numpy()
            ratio = float(reduction.mean()/table[control].mean()) if table[control].mean() > 0 else 0.
            good = ratio >= config['minimum_relative_width_reduction'] and int((reduction > 1e-12).sum()) >= config['minimum_better_seeds']
            passed &= good
            comparisons[unit][control] = {'mean_absolute_width_reduction': float(reduction.mean()),
                'relative_mean_width_reduction': ratio, 'better_seeds': int((reduction > 1e-12).sum()),
                'mask_bootstrap_interval': np.quantile(reduction[draws].mean(1), [tail, 1-tail]).tolist(),
                'practical_target_passed': bool(good)}
    save(OUT/'summary.json', {'status': 'completed_development', 'at_utc': now(), 'cells': len(cohort),
        'countries': cohort.country.nunique(), 'spatial_blocks': cohort.spatial_block.nunique(),
        'trajectories': len(events), 'checkpoints': len(curves), 'oracle_contrast': oracle_contrast,
        'primary_comparisons': comparisons, 'practical_target_passed': bool(passed),
        'wrong_sign_declarations': 0, 'oracle_containment_failures': 0,
        'same_initial_information_across_cost_units': True, 'elapsed_replay_seconds': time.monotonic()-started,
        'new_model_fits': 0, 'new_GPU_jobs': 0, 'fresh_confirmation': False,
        'ranges': 'conservative independent-binary outer ranges; not sharp shared-key bounds or CIs'})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'replay']); args = parser.parse_args()
    if args.phase == 'freeze': freeze(); print('T2 protocol/source/input lock verified')
    else: replay(json.loads((ROOT/'configs/taxonomic_verification_t2.json').read_text()))


if __name__ == '__main__': main()
