"""Verify T2 source/input identities, independent receipts and sealed artifacts."""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from run_taxonomic_verification_t2 import ROOT, DATA, OUT, SOURCES, INPUTS, EXTRA_MANIFESTS, freeze
from run_taxonomic_resolution_t1 import verify_history, save
from validate_taxonomy_audits import verify_manifest, seal

REPORTING = ['scripts/validate_taxonomic_verification_t2_events.py',
    'scripts/validate_taxonomic_verification_t2_resident.py', 'scripts/summarize_taxonomic_verification_t2.py',
    'scripts/verify_taxonomic_verification_t2.py', 'docs/taxonomic_verification_t2_results.md']
MAINTAINED = ['README.md', 'results/README.md', 'docs/next_experiment_budget.md',
    'docs/paper_claim_evidence_matrix.md', 'docs/paper_methods_results_draft.md',
    'docs/paper_outline_working.md', 'docs/taxonomic_verification_t2_plan.md']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seal', action='store_true'); args = parser.parse_args()
    manifest = OUT/'artifact_manifest.json'
    if args.seal and manifest.exists(): raise ValueError('T2 already sealed; run without --seal')
    freeze(); history = verify_history()
    for name in EXTRA_MANIFESTS:
        history[name] = {'files_verified': verify_manifest(ROOT/'results'/name/'artifact_manifest.json')}
    receipts = {}
    for name in ['validation', 'event_validation', 'resident_checkpoints', 'resident_events']:
        receipts[name] = json.loads((OUT/(name+'.json')).read_text())
        assert receipts[name]['passed'] is True
    assert receipts['validation']['independent_dense_checkpoints'] == 60
    assert receipts['validation']['independent_acquisition_orders'] == 20
    assert receipts['event_validation']['dense_prefix_checks'] == 400
    assert receipts['event_validation']['secondary_costs_verified'] == 200
    suites = ET.parse(OUT/'full_tests.xml').getroot().findall('testsuite')
    tests = {k: sum(int(s.attrib[k]) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert tests['tests'] == 407 and not any(tests[k] for k in ['failures', 'errors', 'skipped'])
    summary = json.loads((OUT/'summary.json').read_text())
    assert summary['trajectories'] == 100 and summary['checkpoints'] == 700
    assert not summary['practical_target_passed'] and summary['oracle_containment_failures'] == 0
    report = {'earlier_manifests': history, 'full_tests': tests, 'trajectories': 100, 'checkpoints': 700,
        'independent_dense_checkpoints': 60, 'independent_acquisition_orders': 20,
        'secondary_stopping_costs_verified': 200, 'dense_stopping_prefixes': 400,
        'practical_target_passed': False, 'secondary_endpoints_remain_secondary': True,
        'new_model_fits': 0, 'new_GPU_jobs': 0, 'fresh_confirmation': False}
    if args.seal:
        for name in MAINTAINED:
            target = OUT/'documentation_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
        save(OUT/'integrity_validation.json', report)
        names = SOURCES+INPUTS+REPORTING
        names += [str(p.relative_to(ROOT)) for base in [OUT, DATA] for p in base.rglob('*') if p.is_file() and p != manifest]
        count = seal(manifest, names)
    else:
        count = verify_manifest(manifest)
        assert json.loads((OUT/'integrity_validation.json').read_text()) == report
    print(json.dumps({'passed': True, 'T2_manifest_entries_verified': count, **report}, indent=2))


if __name__ == '__main__': main()
