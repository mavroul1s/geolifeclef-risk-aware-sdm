"""Dense, independently reconstructed checks of incremental T2 queries."""
from collections import Counter
import hashlib
import json
import numpy as np
import pandas as pd

from run_taxonomic_verification_t2 import ROOT, DATA, OUT, freeze, now
from run_taxonomic_resolution_t1 import sha, save


def dense_state(eligible, initial, genera, case, queried):
    positive = initial.copy(); unknown = np.zeros_like(eligible)
    classes = {g: np.flatnonzero(genera == g) for g in np.unique(genera)}
    for key, genus in enumerate(case['genus']):
        cells = case['cells'][case['offsets'][key]:case['offsets'][key+1]]
        if queried[key]:
            answer = int(case['answer'][key])
            if answer >= 0: positive[cells, answer] |= eligible[cells, answer]
        else:
            for cell in cells: unknown[cell, classes[genus]] = True
    known = eligible & (~unknown | positive)
    return positive, known


def dense_bounds(eligible, positive, known, prediction):
    hits = (prediction & positive).sum(1); positives = positive.sum(1)
    up = (prediction & ~known).sum(1); uo = (eligible & ~prediction & ~known).sum(1)
    ld = prediction.sum(1)+positives+uo; ud = prediction.sum(1)+positives+up
    return (np.divide(2*hits, ld, out=np.ones(len(ld)), where=ld > 0),
            np.divide(2*(hits+up), ud, out=np.ones(len(ud)), where=ud > 0))


def independent_orders(eligible, positive, known, prediction, genera, case, weights, seed):
    lo, hi = dense_bounds(eligible, positive, known, prediction)
    unknown = eligible & ~known
    total = unknown.sum(1)
    remaining = Counter()
    for key, genus in enumerate(case['genus']):
        for cell in case['cells'][case['offsets'][key]:case['offsets'][key+1]]:
            remaining[int(cell), int(genus)] += 1
    classes = {g: np.flatnonzero(genera == g) for g in np.unique(genera)}
    credit, relevant = {}, {}
    for (cell, genus), n in remaining.items():
        count = int(unknown[cell, classes[genus]].sum())
        credit[cell, genus] = (hi[cell]-lo[cell])*count/max(int(total[cell]), 1)/n
        relevant[cell, genus] = bool(count and weights[cell] != 0)
    global_score = np.zeros(len(case['genus'])); paired_score = global_score.copy(); incidence = global_score.copy()
    for key, genus in enumerate(case['genus']):
        for cell in case['cells'][case['offsets'][key]:case['offsets'][key+1]]:
            c = credit[int(cell), int(genus)]
            global_score[key] += c/len(weights); paired_score[key] += c*abs(weights[cell])
            incidence[key] += relevant[int(cell), int(genus)]
    tie = np.random.default_rng(seed+101).random(len(global_score)); support = incidence > 0
    output = {}
    for name, score in [('uniform', np.zeros_like(tie)), ('uniform_support', np.zeros_like(tie)),
                        ('global_width', global_score), ('incidence_support', incidence), ('paired_width', paired_score)]:
        order = np.lexsort((tie, -score))
        if name in {'uniform_support', 'incidence_support', 'paired_width'}: order = order[support[order]]
        output[name] = order
    return output


def main():
    freeze()
    config = json.loads((ROOT/'configs/taxonomic_verification_t2.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text()); curves = pd.read_csv(OUT/'curves.csv')
    event = pd.read_csv(OUT/'query_events.csv')
    assert len(curves) == 700 and len(event) == 100
    times = [json.loads((OUT/name).read_text())['at_utc'] for name in ['protocol_freeze.json', 'replay_access.json', 'summary.json']]
    assert times == sorted(times) and len(set(times)) == 3
    oracle = np.load(DATA/'oracle_contract.npz')
    eligible, prediction, truth, genera, weights, quality = [oracle[k] for k in
        ['eligible', 'prediction', 'truth', 'genera', 'weights', 'quality']]
    np.testing.assert_allclose(float(weights@quality), summary['oracle_contrast'], atol=1e-15)
    dense_checks = order_checks = 0
    for seed in config['mask_seeds']:
        initial = np.unpackbits(np.load(DATA/f'initial_{seed}.npz')['positive'], axis=1, count=eligible.shape[1]).astype(bool)
        record_sets = []
        for unit in config['query_units']:
            case = np.load(DATA/f'case_{seed}_{unit}.npz')
            records = Counter()
            for key, genus in enumerate(case['genus']):
                for cell in case['cells'][case['offsets'][key]:case['offsets'][key+1]]:
                    records[int(cell), int(genus), int(case['answer'][key])] += 1
            record_sets.append(records)
            if seed in config['independent_validation_seeds']:
                pos0, known0 = dense_state(eligible, initial, genera, case, np.zeros(len(case['genus']), bool))
                orders = independent_orders(eligible, pos0, known0, prediction, genera, case, weights, seed)
            for name in config['policies']:
                order = case['order_'+name]
                assert len(order) == len(set(order)) and np.all(order >= 0) and np.all(order < len(case['genus']))
                receipt = event[(event.seed == seed) & (event.unit == unit) & (event.policy == name)].iloc[0]
                assert hashlib.sha256(np.asarray(order, '<i8').tobytes()).hexdigest() == receipt.order_sha256
                if seed not in config['independent_validation_seeds']: continue
                np.testing.assert_array_equal(order, orders[name]); order_checks += 1
                for fraction in config['independent_validation_fractions']:
                    n = min(int(np.ceil(fraction*len(case['genus']))), len(order))
                    queried = np.zeros(len(case['genus']), bool); queried[order[:n]] = True
                    pos, known = dense_state(eligible, initial, genera, case, queried)
                    assert not np.any(pos & ~truth) and not np.any(known & ((pos != truth) & eligible))
                    lower, upper = dense_bounds(eligible, pos, known, prediction)
                    assert np.all(lower <= quality+1e-12) and np.all(quality <= upper+1e-12)
                    lo = np.where(weights >= 0, weights*lower, weights*upper).sum()
                    hi = np.where(weights >= 0, weights*upper, weights*lower).sum()
                    row = curves[(curves.seed == seed) & (curves.unit == unit) & (curves.policy == name) & (curves.budget_fraction == fraction)].iloc[0]
                    np.testing.assert_allclose([lo, hi, known.sum()/eligible.sum(), np.mean(upper-lower)],
                        [row.lower, row.upper, row.known_fraction, row.mean_cell_width], atol=1e-11)
                    assert row.queries == n
                    assert row.queried_records == sum(len(case['cells'][case['offsets'][key]:case['offsets'][key+1]]) for key in order[:n])
                    if fraction == 1 and name in {'uniform', 'global_width'}:
                        np.testing.assert_array_equal(pos, truth); np.testing.assert_array_equal(known, eligible)
                    dense_checks += 1
            print(json.dumps({'seed': seed, 'unit': unit, 'dense_checks': dense_checks}), flush=True)
        assert record_sets[0] == record_sets[1], 'Cost units must represent identical censored records'
    primary = curves[curves.budget_fraction == .25]; gate = True
    for unit in config['query_units']:
        table = primary[primary.unit == unit].pivot(index='seed', columns='policy', values='width')
        for control in config['primary_controls']:
            delta = table[control]-table.paired_width
            relative = delta.mean()/table[control].mean() if table[control].mean() > 0 else 0.
            good = relative >= .1 and int((delta > 1e-12).sum()) >= 8
            assert bool(good) == summary['primary_comparisons'][unit][control]['practical_target_passed']
            gate &= good
    assert bool(gate) == summary['practical_target_passed']
    files = [DATA/'oracle_contract.npz', *DATA.glob('case_*.npz'), *DATA.glob('initial_*.npz')]
    save(OUT/'validation.json', {'passed': True, 'at_utc': now(), 'independent_dense_checkpoints': dense_checks,
        'cell_rows_per_checkpoint': len(eligible), 'independent_acquisition_orders': order_checks,
        'all_trajectory_order_hashes': len(event), 'identical_hidden_records_across_cost_units': True,
        'primary_criterion_recomputed': True, 'chronology_and_inputs_verified': True,
        'cache_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files}})
    print(json.dumps({'passed': True, 'dense_checkpoints': dense_checks, 'independent_orders': order_checks}))


if __name__ == '__main__': main()
