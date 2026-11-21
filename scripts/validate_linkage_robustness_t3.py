"""Independent dense T3 labels, audit charges, public priorities and primary costs."""
from collections import Counter
import hashlib
import json
import numpy as np
import pandas as pd

from run_linkage_robustness_t3 import ROOT, DATA, OUT, T2, freeze, resident, now
from run_taxonomic_resolution_t1 import save, sha
from validate_taxonomic_verification_t2 import dense_bounds, independent_orders


def scope_from_initial(case, eligible, initial, genera, weights):
    unknown = np.zeros_like(eligible)
    classes = {g: np.flatnonzero(genera == g) for g in np.unique(genera)}
    for key, genus in enumerate(case['genus']):
        for cell in case['cells'][case['offsets'][key]:case['offsets'][key+1]]:
            unknown[cell, classes[genus]] = True
    known = eligible & (~unknown | initial)
    scope = np.zeros(len(case['cells']), bool)
    for key, genus in enumerate(case['genus']):
        for record in range(int(case['offsets'][key]), int(case['offsets'][key+1])):
            cell = int(case['cells'][record])
            scope[record] = bool(weights[cell] != 0 and np.any(eligible[cell, classes[genus]] & ~known[cell, classes[genus]]))
    np.testing.assert_array_equal(scope, case['support'])
    return scope, known


def effective_order(case, policy, mode):
    order = case['order_'+policy]
    if mode == 'audit_support':
        relevant = np.array([case['support'][a:b].any() for a, b in zip(case['offsets'][:-1], case['offsets'][1:])])
        order = order[relevant[order]]
    return order


def operations(case, order, mode, n):
    resolutions = audits = 0
    for key in order[:n]:
        members = np.arange(case['offsets'][key], case['offsets'][key+1])
        if mode == 'audit_support': members = members[case['support'][members]]
        representative = members[np.argmin(case['cells'][members])]
        resolutions += 1
        if mode != 'unchecked_all':
            audits += len(members)-1
            resolutions += int((case['private_answers'][members] != case['private_answers'][representative]).sum())
    return resolutions, audits


def dense(case, order, mode, n, eligible, initial, genera):
    selected = np.zeros(len(case['genus']), bool); selected[order[:n]] = True
    positive = initial.copy(); unknown = np.zeros_like(eligible)
    classes = {g: np.flatnonzero(genera == g) for g in np.unique(genera)}
    affected = np.zeros(len(eligible), bool)
    for key, genus in enumerate(case['genus']):
        records = np.arange(case['offsets'][key], case['offsets'][key+1])
        reveal = case['support'][records] if mode == 'audit_support' else np.ones(len(records), bool)
        reveal = reveal & selected[key]
        members = records[reveal]
        if len(members):
            representative = members[np.argmin(case['cells'][members])]
            answers = (np.full(len(members), case['private_answers'][representative])
                       if mode == 'unchecked_all' else case['private_answers'][members])
            for record, answer in zip(members, answers):
                cell = int(case['cells'][record]); affected[cell] = True
                if answer >= 0 and eligible[cell, answer]: positive[cell, answer] = True
        for record in records[~reveal]:
            unknown[case['cells'][record], classes[genus]] = True
    known = eligible & (~unknown | positive)
    return positive, known, affected


def contrast(low, high, weights):
    return [float(np.where(weights >= 0, weights*low, weights*high).sum()),
            float(np.where(weights >= 0, weights*high, weights*low).sum())]


def main():
    freeze(); config = json.loads((ROOT/'configs/linkage_robustness_t3.json').read_text())
    base = resident(T2/'oracle_contract.npz'); eligible, prediction, truth, genera, weights, quality = [base[k] for k in
        ['eligible', 'prediction', 'truth', 'genera', 'weights', 'quality']]
    curves = pd.read_csv(OUT/'curves.csv'); events = pd.read_csv(OUT/'query_events.csv')
    assert len(curves) == 2520 and len(events) == 360
    checkpoint_checks = stopping_checks = operation_checks = order_checks = 0
    times = [json.loads((OUT/name).read_text())['at_utc'] for name in ['protocol_freeze.json', 'replay_access.json', 'summary.json']]
    assert times == sorted(times) and len(set(times)) == 3
    for seed in config['mask_seeds']:
        initial = np.unpackbits(resident(T2/f'initial_{seed}.npz')['positive'], axis=1, count=eligible.shape[1]).astype(bool)
        source = resident(T2/f'case_{seed}_block_shared.npz')
        expected = Counter((int(c), int(g), int(source['answer'][key])) for key, g in enumerate(source['genus'])
            for c in source['cells'][source['offsets'][key]:source['offsets'][key+1]])
        for error in config['membership_error_rates']:
            case = resident(DATA/f'case_{seed}_{int(round(1000*error))}.npz')
            actual = Counter((int(c), int(g), int(a)) for key, g in enumerate(case['genus'])
                for c, a in zip(case['cells'][case['offsets'][key]:case['offsets'][key+1]],
                                case['private_answers'][case['offsets'][key]:case['offsets'][key+1]]))
            assert expected == actual
            assert all(len(set(case['cells'][a:b])) == b-a for a, b in zip(case['offsets'][:-1], case['offsets'][1:]))
            scope, known0 = scope_from_initial(case, eligible, initial, genera, weights)
            if seed == config['independent_validation_seed']:
                ranking = independent_orders(eligible, initial, known0, prediction, genera, case, weights, seed)
                for name in config['policies']:
                    np.testing.assert_array_equal(case['order_'+name], ranking[name]); order_checks += 1
            for mode in config['modes']:
                for policy in config['policies']:
                    order = effective_order(case, policy, mode)
                    row = events[(events.seed == seed) & (events.error_rate == error) & (events['mode'] == mode) & (events.policy == policy)].iloc[0]
                    assert hashlib.sha256(np.asarray(order, '<i8').tobytes()).hexdigest() == row.order_sha256
                    assert row.executed_groups == len(order)
                    for endpoint in ['sign', 'tolerance']:
                        stop = row[endpoint+'_groups']
                        if pd.isna(stop): continue
                        stop = int(stop); resolutions, audits = operations(case, order, mode, stop)
                        assert resolutions == row[endpoint+'_resolutions'] and audits == row[endpoint+'_audits']
                        operation_checks += 1
                    if mode == 'audit_support':
                        stop = int(row.sign_groups)
                        for n in sorted({stop, max(0, stop-1)}):
                            positive, known, _ = dense(case, order, mode, n, eligible, initial, genera)
                            lo, hi = dense_bounds(eligible, positive, known, prediction); interval = contrast(lo, hi, weights)
                            certified = interval[0] > config['practical_margin'] or interval[1] < -config['practical_margin']
                            assert certified == (n == stop)
                            assert not np.any(known & (positive != truth))
                            stopping_checks += 1
                    if seed != config['independent_validation_seed']: continue
                    for fraction in config['independent_validation_fractions']:
                        n = min(int(np.ceil(fraction*len(case['genus']))), len(order))
                        positive, known, _ = dense(case, order, mode, n, eligible, initial, genera)
                        lo, hi = dense_bounds(eligible, positive, known, prediction); interval = contrast(lo, hi, weights)
                        computed = curves[(curves.seed == seed) & (curves.error_rate == error) & (curves['mode'] == mode) &
                            (curves.policy == policy) & (curves.budget_fraction == fraction)].iloc[0]
                        r, a = operations(case, order, mode, n)
                        np.testing.assert_allclose([*interval, known.sum()/eligible.sum()],
                            [computed.lower, computed.upper, computed.known_fraction], atol=1e-10)
                        assert computed.resolutions == r and computed.audits == a
                        assert computed.incorrect_known_labels == int((known & (positive != truth)).sum())
                        assert computed.cell_containment_failures == int(((lo > quality+1e-12) | (hi < quality-1e-12)).sum())
                        checkpoint_checks += 1
            print(json.dumps({'seed': seed, 'error_rate': error, 'dense_checkpoints': checkpoint_checks,
                'primary_stopping_prefixes': stopping_checks}), flush=True)
    frontiers = pd.read_csv(OUT/'cost_frontiers.csv')
    for row in events.itertuples():
        take = frontiers[(frontiers.seed == row.seed) & (frontiers.error_rate == row.error_rate) &
            (frontiers['mode'] == row.mode) & (frontiers.policy == row.policy)]
        assert len(take) == 4
        for endpoint in ['sign', 'tolerance']:
            resolutions, audits = getattr(row, endpoint+'_resolutions'), getattr(row, endpoint+'_audits')
            if not pd.isna(resolutions):
                np.testing.assert_allclose(take[endpoint+'_cost'], resolutions+take.audit_price*audits)
    assert checkpoint_checks == 108 and stopping_checks == 240
    summary = json.loads((OUT/'summary.json').read_text()); recomputed = True
    for comparison in summary['primary_comparisons']:
        error = comparison['error_rate']; control = comparison['control']
        take = frontiers[(frontiers.error_rate == error) & (frontiers['mode'] == 'audit_support') &
            (frontiers.audit_price == config['reference_audit_price'])]
        table = take.pivot(index='seed', columns='policy', values='sign_cost')
        values = (take.groupby('seed').local_best_practical_cost.first().reindex(table.index)
            if control == 'best_local_including_paired' else table[control])
        delta = values-table.paired_width
        passed = delta.mean()/values.mean() >= config['minimum_relative_cost_reduction'] and (delta > 1e-10).sum() >= config['minimum_better_seeds']
        assert bool(passed) == comparison['criterion_passed']; recomputed &= passed
    assert bool(recomputed) == summary['reference_robustness_criterion_passed']
    save(OUT/'validation.json', {'passed': True, 'at_utc': now(), 'independent_dense_checkpoints': checkpoint_checks,
        'independent_primary_stopping_prefixes': stopping_checks, 'operation_cost_endpoints': operation_checks,
        'independent_acquisition_orders': order_checks, 'all_order_hashes_checked': len(events),
        'corruption_inventory_invariants_all_cases': True, 'reference_criterion_recomputed': True,
        'cache_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(DATA.glob('case_*.npz'))}})


if __name__ == '__main__': main()
