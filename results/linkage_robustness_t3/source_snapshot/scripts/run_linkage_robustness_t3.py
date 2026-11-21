"""Frozen T3 linkage/cost sensitivity on saved T2 records; no model fits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from geolifeclef.taxonomic_queries import PublicKeys, QueryState, acquisition_order, contrast_interval
from geolifeclef.linkage_verification import (corrupt_memberships, record_keys, comparison_members,
    IdentityOracle, resolve_group, verification_cost)
from run_taxonomic_resolution_t1 import sha, save, verify_history
from validate_taxonomy_audits import verify_manifest

DATA, OUT = ROOT/'data/linkage_robustness_t3', ROOT/'results/linkage_robustness_t3'
T2 = ROOT/'data/taxonomic_verification_t2'
SOURCES = ['configs/linkage_robustness_t3.json', 'docs/linkage_robustness_t3_protocol.md',
    'src/geolifeclef/linkage_verification.py', 'tests/test_linkage_verification.py',
    'scripts/run_linkage_robustness_t3.py', 'scripts/validate_linkage_robustness_t3.py',
    'scripts/plot_linkage_robustness_t3.py']
INPUTS = ['src/geolifeclef/taxonomic_queries.py', 'scripts/run_taxonomic_resolution_t1.py',
    'scripts/validate_taxonomy_audits.py', 'data/taxonomic_resolution_t1/assessment_roles.csv',
    'data/taxonomic_verification_t2/oracle_contract.npz',
    'results/taxonomic_verification_t2/query_events.csv', 'results/taxonomic_verification_t2/summary.json',
    'results/taxonomic_verification_t2/artifact_manifest.json']
INPUTS += ['data/taxonomic_verification_t2/'+f'{prefix}_{seed}{suffix}.npz'
    for seed in range(202611190, 202611200)
    for prefix, suffix in [('initial', ''), ('case', '_block_shared')]]
HISTORY = ['taxonomic_resolution_t1', 'neon_taxonomy_a1', 'neon_taxonomy_a2', 'neon_taxonomy_a3',
           'neon_taxonomy_a3u', 'publication_validation_2026_11_19', 'taxonomic_verification_t2']


def now(): return datetime.now(timezone.utc).isoformat()


def resident(path):
    with np.load(path) as f: return {k: f[k] for k in f.files}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    identity = {name: sha(ROOT/name) for name in SOURCES+INPUTS}
    lock = OUT/'protocol_freeze.json'
    if lock.exists():
        assert json.loads(lock.read_text())['identity'] == identity, 'Frozen T3 source/input changed'
        return
    history = verify_history()
    for name in HISTORY:
        path = ROOT/'results'/name/'artifact_manifest.json'
        history[name] = {'files_verified': verify_manifest(path), 'manifest_sha256': sha(path)}
    for name in SOURCES:
        target = OUT/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    save(lock, {'at_utc': now(), 'identity': identity, 'earlier_artifacts': history,
        'role': 'consumed-record development; construction feasibility precedes main replay', 'main_replay_started': False})


def declared_sign(lo, hi, margin): return 1 if lo > margin else (-1 if hi < -margin else 0)


def event_record(charged, oracle, lo, hi, true_contrast, margin):
    s = declared_sign(lo, hi, margin)
    return {'groups': charged, 'resolutions': oracle.resolutions, 'audits': oracle.audits,
        'lower': lo, 'upper': hi, 'oracle_contained': bool(lo-1e-10 <= true_contrast <= hi+1e-10),
        'practical_sign': s, 'correct_sign': bool(not s or s*true_contrast > margin)}


def replay(config):
    freeze()
    if (OUT/'replay_access.json').exists(): raise ValueError('T3 already started; preserve completed/partial results')
    started = time.monotonic()
    save(OUT/'replay_access.json', {'at_utc': now(), 'protocol_sha256': sha(OUT/'protocol_freeze.json'),
        'all_outcomes_already_consumed': True, 'new_field_composition_requests': 0})
    base = resident(T2/'oracle_contract.npz')
    eligible, prediction, truth, genera, weights, quality = [base[k] for k in
        ['eligible', 'prediction', 'truth', 'genera', 'weights', 'quality']]
    true_contrast = float(weights@quality)
    cohort = pd.read_csv(ROOT/'data/taxonomic_resolution_t1/assessment_roles.csv')
    blocks = cohort.spatial_block.to_numpy(str); countries = cohort.country.to_numpy(str)
    original_events = pd.read_csv(ROOT/'results/taxonomic_verification_t2/query_events.csv')
    local_best = original_events[original_events.unit.eq('cell_local')].groupby('seed').first_practical_sign_queries.min()
    curves, events, receipts = [], [], []
    for seed in config['mask_seeds']:
        original = resident(T2/f'case_{seed}_block_shared.npz')
        original_keys = PublicKeys(original['genus'], tuple(original['cells'][a:b]
            for a, b in zip(original['offsets'][:-1], original['offsets'][1:])))
        initial = np.unpackbits(resident(T2/f'initial_{seed}.npz')['positive'], axis=1, count=eligible.shape[1]).astype(bool)
        original_known = QueryState(eligible, initial, genera, prediction, original_keys).known
        old_records = Counter((int(cell), int(genus), int(answer))
            for genus, members, answer in zip(original_keys.genus, original_keys.cells, original['answer']) for cell in members)
        for error in config['membership_error_rates']:
            keys, actual, offsets, receipt = corrupt_memberships(original_keys, original['answer'], blocks,
                error, seed+config['corruption_seed_offset'], max(config['membership_error_rates']))
            assert receipt['realized_swaps'] == receipt['requested_swaps']
            new_records = Counter((int(cell), int(genus), int(answer))
                for k, genus in enumerate(keys.genus) for cell, answer in zip(keys.cells[k], actual[offsets[k]:offsets[k+1]]))
            assert new_records == old_records
            public = QueryState(eligible, initial, genera, prediction, keys)
            np.testing.assert_array_equal(public.known, original_known)
            support = comparison_members(eligible, public.known, genera, weights, keys)
            relevant = np.array([len(m) > 0 for m in support])
            orders = {name: acquisition_order(public, weights, name, seed+101)[0] for name in config['policies']}
            local_keys = record_keys(keys)
            receipt.update(seed=seed, error_rate=error, public_keys=len(keys.genus), inventory_multiset_unchanged=True,
                initial_known_unchanged=True, local_best_practical_cost=float(local_best[seed]))
            receipts.append(receipt)
            rate_tag = int(round(1000*error))
            np.savez_compressed(DATA/f'case_{seed}_{rate_tag}.npz', genus=keys.genus, offsets=offsets,
                cells=np.concatenate(keys.cells), private_answers=actual,
                support=np.concatenate([np.isin(c, s) for c, s in zip(keys.cells, support)]),
                **{'order_'+name: order for name, order in orders.items()})
            for mode in config['modes']:
                for name, raw_order in orders.items():
                    order = raw_order[relevant[raw_order]] if mode == 'audit_support' else raw_order
                    state = QueryState(eligible, initial, genera, prediction, local_keys)
                    oracle = IdentityOracle(actual); low, high = state.bounds(); lo, hi = contrast_interval(low, high, weights)
                    first_sign = first_width = None
                    checkpoints = {f: min(int(np.ceil(f*len(keys.genus))), len(order)) for f in config['budget_fractions']}
                    failures = wrong_signs = 0; invalid_at_any_checkpoint = False
                    last_lo, last_hi = lo, hi; affected = np.zeros(len(eligible), bool)
                    for charged in range(len(order)+1):
                        if time.monotonic()-started > config['maximum_replay_seconds']:
                            raise TimeoutError('Frozen T3 budget exceeded; keep partial results')
                        if charged:
                            group = int(order[charged-1])
                            rows = support[group] if mode == 'audit_support' else keys.cells[group]
                            a, b = state.bounds(rows); old_l, old_h = contrast_interval(a, b, weights[rows])
                            updated = resolve_group(state, group, keys, offsets, oracle, mode, support)
                            a, b = state.bounds(rows); new_l, new_h = contrast_interval(a, b, weights[rows])
                            lo += new_l-old_l; hi += new_h-old_h; affected[updated] = True
                            assert lo >= last_lo-1e-10 and hi <= last_hi+1e-10
                            last_lo, last_hi = lo, hi
                        current = declared_sign(lo, hi, config['practical_margin'])
                        if first_sign is None and current:
                            first_sign = event_record(charged, oracle, lo, hi, true_contrast, config['practical_margin'])
                        if first_width is None and hi-lo <= config['width_tolerance']:
                            first_width = event_record(charged, oracle, lo, hi, true_contrast, config['practical_margin'])
                        fractions = [f for f, n in checkpoints.items() if n == charged]
                        if not fractions: continue
                        low, high = state.bounds(); check_lo, check_hi = contrast_interval(low, high, weights)
                        np.testing.assert_allclose([lo, hi], [check_lo, check_hi], atol=1e-9)
                        contained = bool(check_lo-1e-10 <= true_contrast <= check_hi+1e-10)
                        cell_failures = int(((low > quality+1e-12) | (high < quality-1e-12)).sum())
                        bad_known = int((state.known & (state.positive != truth)).sum())
                        wrong = bool(current and current*true_contrast <= config['practical_margin'])
                        if mode != 'unchecked_all':
                            assert bad_known == 0 and cell_failures == 0 and contained and not wrong
                        failures += len(fractions)*int(not contained); wrong_signs += len(fractions)*int(wrong)
                        invalid_at_any_checkpoint |= not contained
                        for fraction in fractions:
                            curves.append({'seed': seed, 'error_rate': error, 'mode': mode, 'policy': name,
                                'budget_fraction': fraction, 'groups': charged, 'resolutions': oracle.resolutions,
                                'audits': oracle.audits, 'resolved_records': int(state.queried.sum()),
                                'lower': check_lo, 'upper': check_hi, 'width': check_hi-check_lo,
                                'oracle_contained': contained, 'cell_containment_failures': cell_failures,
                                'incorrect_known_labels': bad_known, 'wrong_practical_sign': wrong,
                                'known_fraction': float(state.known.sum()/eligible.sum()),
                                'affected_countries': len(set(countries[affected]))})
                    if mode != 'unchecked_all':
                        assert first_sign is not None and first_width is not None and hi-lo < 1e-8
                        assert first_sign['correct_sign'] and first_sign['oracle_contained']
                    row = {'seed': seed, 'error_rate': error, 'mode': mode, 'policy': name,
                        'executed_groups': len(order), 'available_groups': len(keys.genus),
                        'total_resolutions': oracle.resolutions, 'total_audits': oracle.audits,
                        'contrast_failures_at_checkpoints': failures, 'wrong_sign_checkpoints': wrong_signs,
                        'any_contrast_failure': invalid_at_any_checkpoint,
                        'order_sha256': hashlib.sha256(np.asarray(order, '<i8').tobytes()).hexdigest(),
                        'local_best_practical_cost': float(local_best[seed])}
                    for prefix, endpoint in [('sign', first_sign), ('tolerance', first_width)]:
                        for field in ['groups', 'resolutions', 'audits', 'lower', 'upper', 'oracle_contained', 'correct_sign']:
                            row[prefix+'_'+field] = None if endpoint is None else endpoint[field]
                    if error == 0 and mode in {'unchecked_all', 'audit_all'}:
                        old = original_events[(original_events.seed == seed) & original_events.unit.eq('block_shared') & original_events.policy.eq(name)].iloc[0]
                        assert first_sign['groups'] == old.first_practical_sign_queries
                        assert first_width['groups'] == old.first_width_queries
                    events.append(row)
            pd.DataFrame(curves).to_csv(OUT/'curves.csv', index=False)
            pd.DataFrame(events).to_csv(OUT/'query_events.csv', index=False)
            pd.DataFrame(receipts).to_csv(OUT/'corruption_receipts.csv', index=False)
            print(json.dumps({'seed': seed, 'error_rate': error, 'trajectories_complete': len(events),
                'seconds': round(time.monotonic()-started, 2)}), flush=True)
    events = pd.DataFrame(events); frame = pd.DataFrame(curves)
    costs = []
    for row in events.to_dict('records'):
        for price in config['audit_prices']:
            record = {k: row[k] for k in ['seed', 'error_rate', 'mode', 'policy', 'local_best_practical_cost']}
            for endpoint in ['sign', 'tolerance']:
                r, a = row[endpoint+'_resolutions'], row[endpoint+'_audits']
                record[endpoint+'_cost'] = None if pd.isna(r) else verification_cost(r, a, price)
            costs.append({'audit_price': price, **record})
    cost_frame = pd.DataFrame(costs); cost_frame.to_csv(OUT/'cost_frontiers.csv', index=False)
    comparisons = []; passed = True; rng = np.random.default_rng(config['bootstrap']['seed'])
    draws = rng.integers(10, size=(config['bootstrap']['repetitions'], 10)); tail = (1-config['bootstrap']['confidence'])/2
    for error in config['membership_error_rates'][1:]:
        take = cost_frame[(cost_frame.error_rate == error) & cost_frame['mode'].eq('audit_support') &
            cost_frame.audit_price.eq(config['reference_audit_price'])]
        table = take.pivot(index='seed', columns='policy', values='sign_cost')
        local = take.groupby('seed').local_best_practical_cost.first().reindex(table.index)
        for name in [p for p in config['policies'] if p != 'paired_width']+['best_local_including_paired']:
            control = local if name == 'best_local_including_paired' else table[name]
            delta = control.to_numpy()-table.paired_width.to_numpy()
            ratio = float(delta.mean()/control.mean()); better = int((delta > 1e-10).sum())
            good = ratio >= config['minimum_relative_cost_reduction'] and better >= config['minimum_better_seeds']
            passed &= good
            comparisons.append({'error_rate': error, 'control': name, 'relative_mean_cost_reduction': ratio,
                'better_seeds': better, 'mean_cost_saving': float(delta.mean()),
                'mask_interval': np.quantile(delta[draws].mean(1), [tail, 1-tail]).tolist(), 'criterion_passed': bool(good)})
    save(OUT/'summary.json', {'status': 'completed_development', 'at_utc': now(), 'trajectories': len(events),
        'checkpoints': len(frame), 'price_evaluations': len(costs), 'oracle_contrast': true_contrast,
        'reference_robustness_criterion_passed': bool(passed), 'primary_comparisons': comparisons,
        'guarded_contrast_failures': int((~frame[frame['mode'] != 'unchecked_all'].oracle_contained).sum()),
        'unchecked_contrast_failures': int((~frame[frame['mode'] == 'unchecked_all'].oracle_contained).sum()),
        'elapsed_replay_seconds': time.monotonic()-started, 'new_model_fits': 0, 'new_GPU_jobs': 0,
        'fresh_confirmation': False, 'perfect_audit_oracle_assumed': True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'replay']); args = parser.parse_args()
    if args.phase == 'freeze': freeze(); print('T3 source/input protocol verified')
    else: replay(json.loads((ROOT/'configs/linkage_robustness_t3.json').read_text()))


if __name__ == '__main__': main()
