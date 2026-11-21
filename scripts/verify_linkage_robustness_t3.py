"""Verify T3 historical locks, corruption doses, validation and final manifest."""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from run_linkage_robustness_t3 import ROOT, DATA, OUT, T2, SOURCES, INPUTS, HISTORY, freeze, resident
from run_taxonomic_resolution_t1 import verify_history, save
from validate_taxonomy_audits import verify_manifest, seal

REPORTING = ['scripts/plot_linkage_robustness_t3_final.py', 'scripts/summarize_linkage_robustness_t3.py', 'scripts/verify_linkage_robustness_t3.py',
             'docs/linkage_robustness_t3_results.md']
MAINTAINED = ['README.md', 'results/README.md', 'docs/next_experiment_budget.md',
    'docs/paper_claim_evidence_matrix.md', 'docs/paper_methods_results_draft.md',
    'docs/paper_outline_working.md', 'docs/thesis_paper_status.md']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seal', action='store_true'); args = parser.parse_args()
    manifest = OUT/'artifact_manifest.json'
    if args.seal and manifest.exists(): raise ValueError('T3 already sealed; verify without --seal')
    freeze(); history = verify_history()
    for name in HISTORY: history[name] = {'files_verified': verify_manifest(ROOT/'results'/name/'artifact_manifest.json')}
    config = json.loads((ROOT/'configs/linkage_robustness_t3.json').read_text())
    receipt = pd.read_csv(OUT/'corruption_receipts.csv')
    blocks = pd.read_csv(ROOT/'data/taxonomic_resolution_t1/assessment_roles.csv').spatial_block.to_numpy(str)
    cases_checked = 0
    for seed in config['mask_seeds']:
        original = resident(T2/f'case_{seed}_block_shared.npz'); previous = np.zeros(len(original['cells']), bool)
        original_labels = np.repeat(original['answer'], np.diff(original['offsets']))
        for error in config['membership_error_rates']:
            case = resident(DATA/f'case_{seed}_{int(round(error*1000))}.npz')
            changed = case['private_answers'] != original_labels
            moved = case['cells'] != original['cells']
            np.testing.assert_array_equal(changed, moved)
            assert np.all(previous <= moved); previous = moved
            row = receipt[(receipt.seed == seed) & (receipt.error_rate == error)].iloc[0]
            assert changed.sum() == 2*row.realized_swaps == 2*row.requested_swaps
            assert np.isclose(changed.mean(), row.realized_rate)
            np.testing.assert_array_equal(case['genus'], original['genus'])
            np.testing.assert_array_equal(case['offsets'], original['offsets'])
            for key in range(len(case['genus'])):
                a, b = case['offsets'][key:key+2]; cells = case['cells'][a:b]
                assert len(cells) == len(set(cells))
                assert np.all(blocks[cells] == blocks[original['cells'][a]])
                assert np.any(~changed[a:b]), 'Each proposed group must retain an original record'
            cases_checked += 1
    validation = json.loads((OUT/'validation.json').read_text())
    assert validation['passed'] and validation['independent_dense_checkpoints'] == 108
    assert validation['independent_primary_stopping_prefixes'] == 240
    assert validation['independent_acquisition_orders'] == 12
    events = pd.read_csv(OUT/'query_events.csv')
    assert validation['operation_cost_endpoints'] == int(events.sign_groups.notna().sum()+events.tolerance_groups.notna().sum())
    suites = ET.parse(OUT/'full_tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(s.attrib[k]) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert tests['tests'] == 417 and not any(tests[k] for k in ['failures', 'errors', 'skipped'])
    summary = json.loads((OUT/'summary.json').read_text())
    assert summary['trajectories'] == 360 and summary['checkpoints'] == 2520 and summary['price_evaluations'] == 1440
    assert summary['guarded_contrast_failures'] == 0
    report = {'earlier_manifests': history, 'full_tests': tests, 'trajectories': 360,
        'checkpoints': 2520, 'price_evaluations': 1440, 'corruption_cases_verified': cases_checked,
        'independent_dense_checkpoints': 108, 'independent_primary_stopping_prefixes': 240,
        'independent_acquisition_orders': 12, 'operation_cost_endpoints': validation['operation_cost_endpoints'],
        'reference_robustness_criterion_passed': summary['reference_robustness_criterion_passed'],
        'perfect_target_label_audit_is_assumed': True, 'field_error_rates_or_costs_measured': False,
        'new_model_fits': 0, 'new_GPU_jobs': 0, 'fresh_confirmation': False}
    if args.seal:
        for name in MAINTAINED:
            target = OUT/'documentation_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
        save(OUT/'integrity_validation.json', report)
        names = SOURCES+INPUTS+REPORTING
        names += [str(p.relative_to(ROOT)) for base in [DATA, OUT] for p in base.rglob('*') if p.is_file() and p != manifest]
        count = seal(manifest, names)
    else:
        count = verify_manifest(manifest)
        assert json.loads((OUT/'integrity_validation.json').read_text()) == report
    print(json.dumps({'passed': True, 'T3_manifest_entries_verified': count, **report}, indent=2))


if __name__ == '__main__': main()
