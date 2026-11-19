"""Post-replay independent dense checks of all secondary stopping costs.

Validate the reported crossing and its immediately preceding prefix. Nested
query information makes the ranges monotone; these two reconstructions check
the first-crossing costs without reusing incremental query bookkeeping.
This adds validation only, without changing the frozen outcomes or criterion.
"""
import json
import numpy as np
import pandas as pd

from run_taxonomic_verification_t2 import ROOT, DATA, OUT, freeze, now
from run_taxonomic_resolution_t1 import save
from validate_taxonomic_verification_t2 import dense_state, dense_bounds


def main():
    freeze()
    config = json.loads((ROOT/'configs/taxonomic_verification_t2.json').read_text())
    events = pd.read_csv(OUT/'query_events.csv')
    oracle = np.load(DATA/'oracle_contract.npz')
    eligible, prediction, genera, weights = [oracle[k] for k in ['eligible', 'prediction', 'genera', 'weights']]
    checks = 0
    for (seed, unit), frame in events.groupby(['seed', 'unit'], sort=False):
        case = np.load(DATA/f'case_{seed}_{unit}.npz')
        initial = np.unpackbits(np.load(DATA/f'initial_{seed}.npz')['positive'], axis=1, count=eligible.shape[1]).astype(bool)
        for row in frame.itertuples():
            order = case['order_'+row.policy]
            for field, criterion in [
                ('first_width_queries', lambda lo, hi: hi-lo <= config['width_tolerance']+1e-12),
                ('first_practical_sign_queries', lambda lo, hi: lo > config['practical_contrast_margin'] or hi < -config['practical_contrast_margin'])]:
                cost = getattr(row, field)
                assert pd.notna(cost), 'This completed finite-cohort contrast must eventually resolve'
                cost = int(cost)
                for n in sorted({cost, max(cost-1, 0)}):
                    take = np.zeros(len(case['genus']), bool); take[order[:n]] = True
                    positive, known = dense_state(eligible, initial, genera, case, take)
                    low, high = dense_bounds(eligible, positive, known, prediction)
                    lo = float(np.where(weights >= 0, weights*low, weights*high).sum())
                    hi = float(np.where(weights >= 0, weights*high, weights*low).sum())
                    assert bool(criterion(lo, hi)) == (n == cost), (seed, unit, row.policy, field, n, lo, hi)
                    checks += 1
        print(json.dumps({'seed': int(seed), 'unit': unit, 'event_boundary_checks': checks}), flush=True)
    save(OUT/'event_validation.json', {'passed': True, 'at_utc': now(), 'dense_prefix_checks': checks,
        'trajectory_events_verified': len(events), 'secondary_costs_verified': 2*len(events),
        'cell_rows_per_check': len(eligible), 'incremental_query_state_reused': False,
        'role': 'post-replay validation; frozen source/outcomes and practical criterion unchanged'})


if __name__ == '__main__': main()
