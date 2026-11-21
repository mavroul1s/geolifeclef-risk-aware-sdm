"""Verify and seal the T4 generic-baseline/equivalence audit."""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
import pandas as pd
from run_adaptive_baselines_t4 import ROOT, DATA, OUT, SOURCES, INPUTS, HISTORY, freeze
from run_taxonomic_resolution_t1 import verify_history, save
from validate_taxonomy_audits import verify_manifest, seal

REPORTING = ['scripts/summarize_adaptive_baselines_t4.py', 'scripts/verify_adaptive_baselines_t4.py',
    'docs/adaptive_baselines_t4_results.md']
MAINTAINED = ['README.md', 'results/README.md', 'docs/next_experiment_budget.md', 'docs/thesis_paper_status.md',
    'docs/paper_claim_evidence_matrix.md', 'docs/paper_methods_results_draft.md', 'docs/paper_outline_working.md']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seal', action='store_true'); args = parser.parse_args()
    path = OUT/'artifact_manifest.json'
    if args.seal and path.exists(): raise ValueError('T4 already sealed; verify without --seal')
    freeze(); history = verify_history()
    for name in HISTORY: history[name] = {'files_verified': verify_manifest(ROOT/'results'/name/'artifact_manifest.json')}
    summary = json.loads((OUT/'summary.json').read_text()); validation = json.loads((OUT/'validation.json').read_text())
    assert summary['priced_paths'] == 360 and summary['executed_trajectories'] == 300 and summary['reprice_only_paths'] == 60
    assert summary['static_paired_score_is_not_distinct'] and summary['maximum_equivalence_error'] < 1e-12
    assert validation['passed'] and validation['dense_endpoint_prefixes'] == 288
    assert validation['independent_batch_decisions'] == 108 and validation['operation_cost_endpoints'] == 720
    equivalence = pd.read_csv(OUT/'equivalence.csv')
    assert len(equivalence) == 30 and equivalence.max_unscaled_score_error.max() < 1e-12
    events = pd.read_csv(OUT/'query_events.csv'); assert len(events) == 360 and events.reused_query_path.sum() == 60
    suites = ET.parse(OUT/'full_tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(s.attrib[k]) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert tests['tests'] == 426 and not any(tests[k] for k in ['failures', 'errors', 'skipped'])
    report = {'earlier_manifests': history, 'full_tests': tests, 'executed_query_trajectories': 300,
        'priced_paths': 360, 'reprice_only_paths': 60, 'scope_equivalence_cases': 30,
        'dense_endpoint_prefixes': 288, 'independent_batch_decisions': 108, 'operation_cost_endpoints': 720,
        'static_paired_score_is_not_distinct': True,
        'reference_operational_criterion_passed': summary['reference_operational_criterion_passed'],
        'new_general_method_claim': False, 'field_costs_measured': False,
        'new_model_fits': 0, 'new_GPU_jobs': 0, 'fresh_confirmation': False}
    if args.seal:
        for name in MAINTAINED:
            target = OUT/'documentation_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
        save(OUT/'integrity_validation.json', report)
        names = SOURCES+INPUTS+REPORTING
        names += [str(p.relative_to(ROOT)) for base in [DATA, OUT] for p in base.rglob('*') if p.is_file() and p != path]
        count = seal(path, names)
    else:
        count = verify_manifest(path); assert json.loads((OUT/'integrity_validation.json').read_text()) == report
    print(json.dumps({'passed': True, 'T4_manifest_entries_verified': count, **report}, indent=2))


if __name__ == '__main__': main()
