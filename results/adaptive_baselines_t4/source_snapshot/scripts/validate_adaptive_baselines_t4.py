"""Independent dense T4 stopping, operation-count and batch-priority checks."""
import hashlib
import json
import numpy as np
import pandas as pd
from run_adaptive_baselines_t4 import ROOT, DATA, OUT, T2, T3, freeze, resident, now
from run_taxonomic_resolution_t1 import save, sha
from validate_linkage_robustness_t3 import dense, dense_bounds, contrast, operations


def public_context(case, order, n, eligible, initial, genera, prediction):
    positive, known, _ = dense(case, order, 'audit_support', n, eligible, initial, genera)
    low, high = dense_bounds(eligible, positive, known, prediction)
    records = np.flatnonzero(case['support'])
    groups = np.repeat(np.arange(len(case['genus'])), np.diff(case['offsets']))[records]
    cells = case['cells'][records]; gen = case['genus'][groups]
    pairs, ids = np.unique(np.column_stack((cells, gen)), axis=0, return_inverse=True)
    classes = {g: np.flatnonzero(genera == g) for g in np.unique(genera)}
    unknown = np.array([int((eligible[c, classes[g]] & ~known[c, classes[g]]).sum()) for c, g in pairs])
    chosen = np.zeros(len(case['genus']), bool); chosen[order[:n]] = True
    remaining = np.bincount(ids[~chosen[groups]], minlength=len(pairs))
    total = (eligible & ~known).sum(1)
    credit = (high-low)[pairs[:, 0]]*unknown/(np.maximum(total[pairs[:, 0]], 1)*np.maximum(remaining, 1))
    basis = np.bincount(groups, weights=credit[ids], minlength=len(case['genus']))
    incidence = np.bincount(groups, weights=((unknown > 0) & ((high-low)[pairs[:, 0]] > 0))[ids], minlength=len(case['genus']))
    return basis, incidence, np.bincount(groups, minlength=len(case['genus'])), positive, known, low, high


def expected_batch(case, order, n, initial_credit, current_credit, incidence, members, policy, price, seed, config):
    r, a = operations(case, order, 'audit_support', n)
    alpha, beta = config['mismatch_prior']['alpha'], config['mismatch_prior']['beta']
    forecast = alpha/(alpha+beta) if policy.startswith('static_') or policy.endswith('_fixed') else (alpha+r-n)/(alpha+beta+a)
    score = initial_credit.copy() if policy.startswith('static_') else (incidence if policy.startswith('adaptive_incidence') else current_credit)
    if '_per_cost' in policy: score = score/(1+np.maximum(members-1, 0)*(price+forecast))
    tie = np.random.default_rng(seed+101).random(len(score)); chosen = np.zeros(len(score), bool); chosen[order[:n]] = True
    result = np.lexsort((tie, -score)); result = result[(members[result] > 0) & ~chosen[result]]
    return result[:config['batch_size']]


def main():
    freeze(); config = json.loads((ROOT/'configs/adaptive_baselines_t4.json').read_text())
    events = pd.read_csv(OUT/'query_events.csv'); assert len(events) == 360
    base = resident(T2/'oracle_contract.npz'); e, p, g, w, truth, quality = [base[k] for k in
        ['eligible', 'prediction', 'genera', 'weights', 'truth', 'quality']]
    endpoint_checks = priority_checks = operation_checks = 0
    for seed in config['mask_seeds']:
        initial = np.unpackbits(resident(T2/f'initial_{seed}.npz')['positive'], axis=1, count=e.shape[1]).astype(bool)
        for error in config['error_rates']:
            tag = int(round(1000*error)); case = resident(T3/f'case_{seed}_{tag}.npz')
            initial_basis = None
            if seed == config['independent_priority_seed']:
                initial_basis = public_context(case, np.array([], int), 0, e, initial, g, p)[0]
            for policy in config['policies']:
                unpriced_order = None
                for price_index, price in enumerate(config['audit_prices']):
                    row = events[(events.seed == seed) & (events.error_rate == error) & (events.policy == policy) & (events.audit_price == price)].iloc[0]
                    trace = resident(DATA/f'trace_{seed}_{tag}_{price_index}_{policy}.npz'); order = trace['order']; starts = trace['batch_starts']
                    assert len(order) == len(set(order)) == row.executed_groups
                    assert hashlib.sha256(np.asarray(order, '<i8').tobytes()).hexdigest() == row.order_sha256
                    assert np.array_equal(starts, np.arange(0, len(order), config['batch_size']))
                    if policy in {'static_scope_width', 'adaptive_scope_width'}:
                        if unpriced_order is None: unpriced_order = order
                        else: np.testing.assert_array_equal(order, unpriced_order)
                    for endpoint in ['sign', 'tolerance']:
                        stop = int(row[endpoint+'_groups']); r, a = operations(case, order, 'audit_support', stop)
                        assert r == row[endpoint+'_resolutions'] and a == row[endpoint+'_audits']
                        assert np.isclose(r+price*a, row[endpoint+'_cost']); operation_checks += 1
                        if seed not in config['independent_endpoint_seeds']: continue
                        for n in sorted({stop, max(0, stop-1)}):
                            positive, known, _ = dense(case, order, 'audit_support', n, e, initial, g)
                            lo, hi = dense_bounds(e, positive, known, p); interval = contrast(lo, hi, w)
                            assert not np.any(known & (positive != truth))
                            assert np.all(lo <= quality+1e-12) and np.all(quality <= hi+1e-12)
                            if endpoint == 'sign': criterion = interval[0] > config['practical_margin'] or interval[1] < -config['practical_margin']
                            else: criterion = interval[1]-interval[0] <= config['width_tolerance']+1e-12
                            assert criterion == (n == stop), (seed, error, policy, endpoint, n, interval)
                            if n == stop: np.testing.assert_allclose(interval, [row[endpoint+'_lower'], row[endpoint+'_upper']], atol=1e-9)
                            endpoint_checks += 1
                    if seed != config['independent_priority_seed']: continue
                    chosen_starts = sorted({int(starts[0]), int(starts[len(starts)//2]), int(starts[-1])})
                    for n in chosen_starts:
                        credit, incidence, members, *_ = public_context(case, order, n, e, initial, g, p)
                        expected = expected_batch(case, order, n, initial_basis, credit, incidence, members,
                            policy, price, seed, config)
                        actual = order[n:n+config['batch_size']]
                        np.testing.assert_array_equal(actual, expected[:len(actual)]); priority_checks += 1
            print(json.dumps({'seed': seed, 'error_rate': error, 'dense_endpoint_prefixes': endpoint_checks,
                'batch_decisions': priority_checks}), flush=True)
    summary = json.loads((OUT/'summary.json').read_text()); assert endpoint_checks == 288 and priority_checks == 108 and operation_checks == 720
    gate = True
    for entry in summary['primary_comparisons']:
        take = events[(events.error_rate == entry['error_rate']) & (events.audit_price == entry['audit_price'])]
        table = take.pivot(index='seed', columns='policy', values='sign_cost')
        best = table[['static_scope_width', 'static_scope_width_per_cost']].min(axis=1)
        delta = best-table.adaptive_scope_width_per_cost
        passed = delta.mean()/best.mean() >= config['minimum_relative_cost_reduction'] and (delta > 1e-10).sum() >= config['minimum_better_seeds']
        assert bool(passed) == entry['criterion_passed']; gate &= passed
    assert bool(gate) == summary['reference_operational_criterion_passed']
    times = [json.loads((OUT/n).read_text())['at_utc'] for n in ['protocol_freeze.json', 'replay_access.json', 'summary.json']]
    assert times == sorted(times) and len(set(times)) == 3
    save(OUT/'validation.json', {'passed': True, 'at_utc': now(), 'dense_endpoint_prefixes': endpoint_checks,
        'independent_batch_decisions': priority_checks, 'operation_cost_endpoints': operation_checks,
        'all_order_hashes': len(events), 'unpriced_paths_identical_across_prices': True,
        'reference_criterion_recomputed': True, 'chronology_and_frozen_inputs_verified': True,
        'trace_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(DATA.glob('trace_*.npz'))}})


if __name__ == '__main__': main()
