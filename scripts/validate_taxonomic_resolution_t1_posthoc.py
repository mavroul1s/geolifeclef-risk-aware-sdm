"""Construct feasible endpoint witnesses and independently audit T1 policies."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from geolifeclef.taxonomic_censoring import censor_genera
from run_taxonomic_resolution_t1 import DATA, OUT, save, sha, now


def hash_number(text):
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:15], 16)


def main():
    target = OUT/'posthoc'; lock = json.loads((target/'analysis_lock.json').read_text())
    for name, digest in {**lock['primary_and_cache_identities'], **lock['sources']}.items():
        assert sha(ROOT/name) == digest
    cfg = json.loads((ROOT/'configs/taxonomic_resolution_t1.json').read_text())
    c = pd.read_csv(DATA/'assessment_roles.csv'); p = np.load(DATA/'fixed_policy.npz')
    y = np.load(DATA/'oracle.npz')['truth']; contract = np.load(DATA/'contract.npz')
    bounds = np.load(DATA/'posthoc_genus_bounds.npz'); receipts = json.loads((OUT/'mask_receipts.json').read_text())
    # Python sorting and explicit group allocations are independent of QuotaSelection.
    groups = (c.country.astype(str)+'|'+c.area_bin.astype(str)).to_numpy()
    names, sizes = np.unique(groups, return_counts=True)
    total = int(np.ceil(len(c)*cfg['coverage'])); exact = sizes*total/len(c)
    quotas = np.floor(exact).astype(int)
    order = sorted(range(len(names)), key=lambda i: (-(exact[i]-quotas[i]), hash_number(names[i])))
    for i in order[:total-int(quotas.sum())]:
        quotas[i] += 1
    for name in cfg['selectors']:
        risk = p[name] if name in ['random', 'entropy'] else 1-p['HGB_quality' if name == 'HGB' else name]
        accepted = np.zeros(len(c), bool)
        for group, quota in zip(names, quotas):
            rows = np.flatnonzero(groups == group)
            ranking = sorted(rows, key=lambda i: (float(risk[i]), hash_number('20261113:'+str(c.iloc[i].cell_id))))
            accepted[ranking[:quota]] = True
        assert np.array_equal(accepted, p['accept_'+name])
    cases, max_error = 0, 0.
    selected_rows = np.linspace(0, len(c)-1, 200, dtype=int)
    for index, record in enumerate(receipts):
        weights = np.ones_like(contract['genus_weights']) if record['mechanism'].startswith('uniform') else contract['genus_weights']
        known, _, _ = censor_genera(y, contract['eligible'], contract['genus_codes'], weights,
                                    c.cell_id, record['dose'], record['seed'])
        for row in selected_rows:
            pred = np.zeros(y.shape[1], bool); pred[p['rank'][row, :p['count'][row]]] = True
            unknown = contract['eligible'][row]&~known[row]
            hidden = np.unique(contract['genus_codes'][unknown])
            low = (y[row] > 0)&known[row]; low[unknown&~pred] = True
            high = (y[row] > 0)&known[row]; high[unknown&pred] = True
            for code in hidden:
                classes = np.flatnonzero(unknown&(contract['genus_codes'] == code))
                if not low[classes].any():
                    low[classes[0]] = True
                if not high[classes].any():
                    high[classes[0]] = True
                assert low[classes].any() and high[classes].any()
            for witness, key in [(low, 'lower'), (high, 'upper')]:
                assert np.array_equal(witness[known[row]], y[row, known[row]] > 0)
                den = int(pred.sum()+witness.sum())
                value = 2*int((witness&pred).sum())/den if den else 1.
                error = abs(value-bounds[key][index, row]); max_error = max(max_error, error)
                assert error < 1e-12
            cases += 1
        if (index+1)%10 == 0:
            print('Endpoint witness settings verified', index+1, flush=True)
    save(target/'validation.json', {'passed': True, 'at_utc': now(),
        'primary_files_and_posthoc_sources_unchanged': True,
        'independent_fixed_selectors_verified': len(cfg['selectors']),
        'independent_country_area_allocations': len(names), 'feasible_endpoint_witness_pairs': cases,
        'maximum_endpoint_witness_error': max_error,
        'constraint_scope': 'simulator-known eligible positive genera; not a validated NEON assumption'})


if __name__ == '__main__':
    main()
