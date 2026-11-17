"""Independent full-matrix scoring and budget checks for the T1 replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from geolifeclef.taxonomic_censoring import censor_genera
from run_taxonomic_resolution_t1 import DATA, OUT, sha, save, verify_history, freeze


def main():
    freeze()
    config = json.loads((ROOT/'configs/taxonomic_resolution_t1.json').read_text())
    lock = json.loads((OUT/'policy_freeze.json').read_text())
    for name, digest in lock['files'].items():
        assert sha(ROOT/name) == digest
    assert json.loads((OUT/'protocol_freeze.json').read_text())['at_utc'] < lock['at_utc']
    assert lock['at_utc'] < json.loads((OUT/'assessment_access.json').read_text())['first_materialized_at_utc']
    history = verify_history()
    c = pd.read_csv(DATA/'assessment_roles.csv'); p = np.load(DATA/'fixed_policy.npz')
    contract = np.load(DATA/'contract.npz'); oracle = np.load(DATA/'oracle.npz')
    replay = np.load(DATA/'replay_cells.npz'); table = pd.read_csv(OUT/'replay_summary.csv')
    receipts = json.loads((OUT/'mask_receipts.json').read_text())
    eligible, truth, q = contract['eligible'], oracle['truth'], oracle['F1']
    n, k = truth.shape
    pred = np.zeros((n, k), bool)
    for row in range(n):
        classes = p['rank'][row, :p['count'][row]]
        assert len(classes) == len(np.unique(classes))
        pred[row, classes] = True
    assert not np.any(pred&~eligible)
    positive = truth > 0
    den = pred.sum(1)+positive.sum(1)
    independent_oracle = np.divide(2*(pred&positive).sum(1), den, out=np.ones(n), where=den > 0)
    assert np.allclose(independent_oracle, q, atol=1e-14)
    # All policies must meet identical country/area allocations, including total.
    groups = (c.country.astype(str)+'|'+c.area_bin.astype(str)).to_numpy()
    quotas = {}
    for group in np.unique(groups):
        take = groups == group
        amounts = [int(p['accept_'+name][take].sum()) for name in config['selectors']]
        assert len(set(amounts)) == 1
        quotas[group] = amounts[0]
    assert sum(quotas.values()) == round(config['coverage']*len(c))
    containment, largest_error = 0, 0.
    for index, record in enumerate(receipts):
        mechanism, dose, seed = record['mechanism'], record['dose'], record['seed']
        weights = np.ones_like(contract['genus_weights']) if mechanism.startswith('uniform') else contract['genus_weights']
        known, positive_genera, hidden_genera = censor_genera(truth, eligible, contract['genus_codes'], weights,
                                                            c.cell_id, dose, seed)
        assert hashlib.sha256(np.packbits(known, bitorder='little').tobytes()).hexdigest() == record['known_mask_sha256']
        assert np.array_equal(hidden_genera, np.ceil(dose*positive_genera).astype(int))
        observed_positive = positive&known
        tp = (pred&observed_positive).sum(1)
        positive_count = observed_positive.sum(1)
        naive_den = pred.sum(1)+positive_count
        known_pred = (pred&known).sum(1)
        known_den = known_pred+positive_count
        naive = np.divide(2*tp, naive_den, out=np.ones(n), where=naive_den > 0)
        known_score = np.divide(2*tp, known_den, out=np.ones(n), where=known_den > 0)
        missing_pred = (pred&~known).sum(1)
        missing_other = (eligible&~known&~pred).sum(1)
        low_den = pred.sum(1)+positive_count+missing_other
        high_den = pred.sum(1)+positive_count+missing_pred
        lower = np.divide(2*tp, low_den, out=np.ones(n), where=low_den > 0)
        upper = np.divide(2*(tp+missing_pred), high_den, out=np.ones(n), where=high_den > 0)
        assert np.all(lower <= q+1e-12) and np.all(q <= upper+1e-12)
        for key, value in [('unknown_as_absence', naive), ('known_labels_only', known_score),
                           ('lower', lower), ('upper', upper)]:
            error = float(np.max(np.abs(value-replay[key][index])))
            largest_error = max(error, largest_error)
            assert error < 1e-12
        containment += n
        admission = known.sum(1)/eligible.sum(1) >= config['minimum_known_label_fraction_diagnostic']
        assert np.array_equal(admission, replay['admitted'][index])
        if dose == 0:
            assert np.array_equal(known, eligible)
            assert np.array_equal(naive, q) and np.array_equal(known_score, q)
            assert np.array_equal(lower, upper)
        # Semantics on a deterministic cross-section: no partially hidden eligible
        # genus, no hidden genus without an oracle positive, no restored positives.
        for row in np.linspace(0, n-1, 19, dtype=int):
            hidden = np.unique(contract['genus_codes'][eligible[row]&~known[row]])
            positive_codes = np.unique(contract['genus_codes'][positive[row]])
            assert len(hidden) == hidden_genera[row] and np.isin(hidden, positive_codes).all()
            for code in hidden:
                assert not known[row, eligible[row]&(contract['genus_codes'] == code)].any()
        for a, b in config['contrasts']:
            pa, pb = p['accept_'+a], p['accept_'+b]
            w = pa/pa.sum()-pb/pb.sum()
            lo = float((w[w > 0]*lower[w > 0]).sum()+(w[w < 0]*upper[w < 0]).sum())
            hi = float((w[w > 0]*upper[w > 0]).sum()+(w[w < 0]*lower[w < 0]).sum())
            assert np.isclose(lo, table.iloc[index]['bound_'+a+'_'+b+'_lower'], atol=1e-12)
            assert np.isclose(hi, table.iloc[index]['bound_'+a+'_'+b+'_upper'], atol=1e-12)
            for name, scores in [('unknown_as_absence', naive), ('known_labels_only', known_score)]:
                actual = scores[pa].mean()-scores[pb].mean()
                assert np.isclose(actual, table.iloc[index][name+'_'+a+'_'+b], atol=1e-12)
        if (index+1)%10 == 0:
            print('Independently verified', index+1, 'settings', flush=True)
    # Independently draw spatial counts for one nonzero paired distortion.
    result = json.loads((OUT/'summary.json').read_text())
    section = next(s for s in result['sections'] if s['dose'] == .2 and s['mechanism'].startswith('uniform'))
    rows = np.flatnonzero((table.dose == .2)&table.mechanism.str.startswith('uniform'))
    value = replay['known_labels_only'][rows].mean(0)-q
    names, inverse = np.unique(c.spatial_block.astype(str), return_inverse=True)
    rng = np.random.default_rng(config['bootstrap']['seed'])
    draw = rng.integers(len(names), size=(config['bootstrap']['repetitions'], len(names)))
    count = np.bincount(inverse); sums = np.bincount(inverse, weights=value)
    boot = sums[draw].sum(1)/count[draw].sum(1)
    tail = (1-config['bootstrap']['confidence'])/2
    assert np.allclose(np.quantile(boot, [tail, 1-tail]), section['treatments']['known_labels_only']['bias']['interval'], atol=1e-12)
    report = {'passed': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
        'settings_independently_scored': len(receipts), 'cell_replay_oracle_containment_checks': containment,
        'maximum_scoring_error': largest_error, 'mask_hashes_verified': len(receipts),
        'fixed_policy_quotas_verified': len(quotas), 'all_selectors_accepted_cells': int(p['accept_HGB'].sum()),
        'zero_dose_identity': True, 'chronology_and_input_identities': True,
        'independent_spatial_interval_reproduction': True, 'historical_manifests': history,
        'scope': 'correctness/integrity, not novelty or new independent ecological confirmation'}
    save(OUT/'validation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
