"""Frozen audit of generic scope-matched adaptive verification controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from geolifeclef.taxonomic_queries import PublicKeys, QueryState, acquisition_order, contrast_interval
from geolifeclef.linkage_verification import record_keys, resolve_group, IdentityOracle, verification_cost
from geolifeclef.adaptive_verification import ScopeCredit, priorities, choose_batch, equal_support_weight
from run_linkage_robustness_t3 import resident, now
from run_taxonomic_resolution_t1 import sha, save, verify_history
from validate_taxonomy_audits import verify_manifest

DATA, OUT = ROOT/'data/adaptive_baselines_t4', ROOT/'results/adaptive_baselines_t4'
T2, T3 = ROOT/'data/taxonomic_verification_t2', ROOT/'data/linkage_robustness_t3'
SOURCES = ['configs/adaptive_baselines_t4.json', 'docs/adaptive_baselines_t4_protocol.md',
    'src/geolifeclef/adaptive_verification.py', 'tests/test_adaptive_verification.py',
    'scripts/run_adaptive_baselines_t4.py', 'scripts/validate_adaptive_baselines_t4.py',
    'scripts/plot_adaptive_baselines_t4.py']
INPUTS = ['src/geolifeclef/taxonomic_queries.py', 'src/geolifeclef/linkage_verification.py',
    'scripts/run_linkage_robustness_t3.py', 'scripts/validate_linkage_robustness_t3.py',
    'scripts/run_taxonomic_resolution_t1.py', 'scripts/validate_taxonomy_audits.py',
    'data/taxonomic_verification_t2/oracle_contract.npz', 'results/linkage_robustness_t3/artifact_manifest.json']
INPUTS += [f'data/linkage_robustness_t3/case_{seed}_{tag}.npz'
    for seed in range(202611190, 202611200) for tag in [0, 10, 50]]
INPUTS += [f'data/taxonomic_verification_t2/initial_{seed}.npz' for seed in range(202611190, 202611200)]
HISTORY = ['taxonomic_resolution_t1', 'neon_taxonomy_a1', 'neon_taxonomy_a2', 'neon_taxonomy_a3',
    'neon_taxonomy_a3u', 'publication_validation_2026_11_19', 'taxonomic_verification_t2', 'linkage_robustness_t3']


def freeze():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    identity = {p: sha(ROOT/p) for p in SOURCES+INPUTS}; path = OUT/'protocol_freeze.json'
    if path.exists():
        assert json.loads(path.read_text())['identity'] == identity, 'Frozen T4 source/input changed'
        return
    old = verify_history()
    for name in HISTORY:
        p = ROOT/'results'/name/'artifact_manifest.json'
        old[name] = {'files_verified': verify_manifest(p), 'manifest_sha256': sha(p)}
    for p in SOURCES:
        target = OUT/'source_snapshot'/p; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/p, target)
    save(path, {'at_utc': now(), 'identity': identity, 'earlier_manifests': old,
        'role': 'consumed-record development; analytical equivalence precedes query replay', 'replay_started': False})


def practical(lo, hi, margin): return 1 if lo > margin else (-1 if hi < -margin else 0)


def execute(case, initial, base, policy, price, seed, config, started):
    e, p, g, w, truth, q = [base[k] for k in ['eligible', 'prediction', 'genera', 'weights', 'truth', 'quality']]
    keys = PublicKeys(case['genus'], tuple(case['cells'][a:b] for a, b in zip(case['offsets'][:-1], case['offsets'][1:])))
    state = QueryState(e, initial, g, p, record_keys(keys))
    graph = ScopeCredit(state, keys, case['offsets'], case['support'])
    support = tuple(c[case['support'][a:b]] for c, a, b in zip(keys.cells, case['offsets'][:-1], case['offsets'][1:]))
    oracle = IdentityOracle(case['private_answers']); queried = np.zeros(len(keys.genus), bool)
    tie = np.random.default_rng(seed+101).random(len(keys.genus)); true_contrast = float(w@q)
    order, batch_starts, curves = [], [], []; sign_event = width_event = None
    low, high = state.bounds(); lo, hi = contrast_interval(low, high, w); last_lo, last_hi = lo, hi

    def capture(endpoint=False):
        lower, upper = state.bounds(); a, b = contrast_interval(lower, upper, w)
        np.testing.assert_allclose([a, b], [lo, hi], atol=1e-9)
        assert np.all(lower <= q+1e-12) and np.all(q <= upper+1e-12)
        if endpoint: assert not np.any(state.known & (state.positive != truth))
        curves.append({'groups': len(order), 'resolutions': oracle.resolutions, 'audits': oracle.audits,
            'cost': verification_cost(oracle.resolutions, oracle.audits, price), 'lower': a, 'upper': b,
            'width': b-a, 'oracle_contained': True, 'endpoint_label_check': endpoint})

    def event():
        assert practical(lo, hi, config['practical_margin'])*true_contrast > config['practical_margin'] or hi-lo <= config['width_tolerance']
        return {'groups': len(order), 'resolutions': oracle.resolutions, 'audits': oracle.audits,
            'cost': verification_cost(oracle.resolutions, oracle.audits, price), 'lower': lo, 'upper': hi}

    capture()
    while sign_event is None or width_event is None:
        if time.monotonic()-started > config['maximum_replay_seconds']: raise TimeoutError('Frozen T4 replay budget exceeded')
        batch_starts.append(len(order))
        score, expected, probability = priorities(graph, state, policy, price, len(order), oracle.resolutions, oracle.audits,
            **config['mismatch_prior'])
        batch = choose_batch(score, graph.members, queried, tie, config['batch_size'])
        if not len(batch): raise ValueError('Relevant groups exhausted before the finite-cohort endpoints')
        for group in batch:
            rows = support[group]; lower, upper = state.bounds(rows)
            before_lo, before_hi = contrast_interval(lower, upper, w[rows])
            resolve_group(state, int(group), keys, case['offsets'], oracle, 'audit_support', support)
            graph.observe(state, rows, int(keys.genus[group])); queried[group] = True; order.append(int(group))
            lower, upper = state.bounds(rows); after_lo, after_hi = contrast_interval(lower, upper, w[rows])
            lo += after_lo-before_lo; hi += after_hi-before_hi
            assert lo >= last_lo-1e-10 and hi <= last_hi+1e-10; last_lo, last_hi = lo, hi
            new_endpoint = False
            if sign_event is None and practical(lo, hi, config['practical_margin']): sign_event = event(); new_endpoint = True
            if width_event is None and hi-lo <= config['width_tolerance']: width_event = event(); new_endpoint = True
            if new_endpoint: capture(True)
            if sign_event is not None and width_event is not None: break
        capture()
    assert sign_event is not None and width_event is not None
    return np.array(order, int), np.array(batch_starts, int), curves, sign_event, width_event


def replay(config):
    freeze()
    if (OUT/'replay_access.json').exists(): raise ValueError('T4 already started; preserve prior/partial results')
    started = time.monotonic(); save(OUT/'replay_access.json', {'at_utc': now(), 'protocol_sha256': sha(OUT/'protocol_freeze.json'),
        'all_outcomes_already_consumed': True, 'private_oracle_used_only_for_charged_answers_and_validation': True})
    base = resident(T2/'oracle_contract.npz'); e, p, g, w = [base[k] for k in ['eligible', 'prediction', 'genera', 'weights']]
    factor = equal_support_weight(w); events, curves, equivalence = [], [], []; executed = 0
    for seed in config['mask_seeds']:
        initial = np.unpackbits(resident(T2/f'initial_{seed}.npz')['positive'], axis=1, count=e.shape[1]).astype(bool)
        for error in config['error_rates']:
            tag = int(round(1000*error)); case = resident(T3/f'case_{seed}_{tag}.npz')
            keys = PublicKeys(case['genus'], tuple(case['cells'][a:b] for a, b in zip(case['offsets'][:-1], case['offsets'][1:])))
            public = QueryState(e, initial, g, p, keys)
            state = QueryState(e, initial, g, p, record_keys(keys)); graph = ScopeCredit(state, keys, case['offsets'], case['support'])
            _, paired = acquisition_order(public, w, 'paired_width', seed+101)
            discrepancy = float(np.max(np.abs(paired/factor-graph.initial_credit))); assert discrepancy < 1e-12
            tie = np.random.default_rng(seed+101).random(len(keys.genus))
            canonical = choose_batch(graph.initial_credit, graph.members, np.zeros(len(keys.genus), bool), tie, len(keys.genus))
            equivalence.append({'seed': seed, 'error_rate': error, 'max_unscaled_score_error': discrepancy,
                'floating_point_order_position_differences': int((canonical != case['order_paired_width']).sum()),
                'constant_absolute_weight': factor, 'no_distinct_paired_arm': True})
            for policy in config['policies']:
                reuse = None
                for price_index, price in enumerate(config['audit_prices']):
                    reused = price_index > 0 and policy in {'static_scope_width', 'adaptive_scope_width'}
                    if reused: order, batches, raw_curves, first_sign, first_width = reuse
                    else:
                        order, batches, raw_curves, first_sign, first_width = execute(case, initial, base, policy, price, seed, config, started)
                        executed += 1
                        if policy in {'static_scope_width', 'adaptive_scope_width'}: reuse = (order, batches, raw_curves, first_sign, first_width)
                    path = DATA/f'trace_{seed}_{tag}_{price_index}_{policy}.npz'
                    np.savez_compressed(path, order=order, batch_starts=batches)
                    prefix = {'seed': seed, 'error_rate': error, 'audit_price': price, 'price_index': price_index,
                        'policy': policy, 'reused_query_path': reused}
                    row = {**prefix, 'executed_groups': len(order), 'batch_count': len(batches),
                        'order_sha256': hashlib.sha256(np.asarray(order, '<i8').tobytes()).hexdigest()}
                    for name, endpoint in [('sign', first_sign), ('tolerance', first_width)]:
                        for key, value in endpoint.items(): row[name+'_'+key] = value
                        row[name+'_cost'] = verification_cost(endpoint['resolutions'], endpoint['audits'], price)
                    events.append(row)
                    for curve in raw_curves:
                        curves.append({**prefix, **curve, 'cost': verification_cost(curve['resolutions'], curve['audits'], price)})
            pd.DataFrame(events).to_csv(OUT/'query_events.csv', index=False)
            pd.DataFrame(curves).to_csv(OUT/'batch_curves.csv', index=False)
            pd.DataFrame(equivalence).to_csv(OUT/'equivalence.csv', index=False)
            print(json.dumps({'seed': seed, 'error_rate': error, 'priced_paths': len(events), 'executed_trajectories': executed,
                'seconds': round(time.monotonic()-started, 2)}), flush=True)
    frame = pd.DataFrame(events); comparisons = []; passed = True
    rng = np.random.default_rng(config['bootstrap']['seed']); draws = rng.integers(10, size=(config['bootstrap']['repetitions'], 10))
    tail = (1-config['bootstrap']['confidence'])/2
    for error in config['error_rates'][1:]:
        for price in config['audit_prices']:
            table = frame[(frame.error_rate == error) & (frame.audit_price == price)].pivot(index='seed', columns='policy', values='sign_cost')
            best = table[['static_scope_width', 'static_scope_width_per_cost']].min(axis=1)
            delta = best-table.adaptive_scope_width_per_cost; relative = float(delta.mean()/best.mean())
            good = relative >= config['minimum_relative_cost_reduction'] and (delta > 1e-10).sum() >= config['minimum_better_seeds']
            passed &= good
            comparisons.append({'error_rate': error, 'audit_price': price, 'relative_mean_cost_reduction': relative,
                'better_seeds': int((delta > 1e-10).sum()), 'mean_cost_saving': float(delta.mean()),
                'mask_interval': np.quantile(delta.to_numpy()[draws].mean(1), [tail, 1-tail]).tolist(), 'criterion_passed': bool(good)})
    save(OUT/'summary.json', {'status': 'completed_development', 'at_utc': now(), 'priced_paths': len(events),
        'executed_trajectories': executed, 'reprice_only_paths': int(frame.reused_query_path.sum()),
        'recorded_curve_points': len(curves), 'scope_equivalence_cases': len(equivalence),
        'maximum_equivalence_error': max(r['max_unscaled_score_error'] for r in equivalence),
        'static_paired_score_is_not_distinct': True, 'reference_operational_criterion_passed': bool(passed),
        'primary_comparisons': comparisons, 'oracle_containment_failures': 0, 'incorrect_endpoint_labels': 0,
        'elapsed_replay_seconds': time.monotonic()-started, 'new_model_fits': 0, 'new_GPU_jobs': 0,
        'fresh_confirmation': False, 'perfect_audit_oracle_assumed': True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'replay']); args = parser.parse_args()
    if args.phase == 'freeze': freeze(); print('T4 source/input protocol verified')
    else: replay(json.loads((ROOT/'configs/adaptive_baselines_t4.json').read_text()))


if __name__ == '__main__': main()
