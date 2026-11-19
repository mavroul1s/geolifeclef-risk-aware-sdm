"""Verify the preserved A2/A3/A3u development batch and its earlier artifacts.

Run without arguments to verify an existing release. --seal records a new
manifest only when none exists; it never replaces sealed results. Full local
data caches are required for input verification, but remain excluded from Git.
"""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from run_taxonomic_resolution_t1 import verify_history, sha, save

PUBLICATION = ROOT / 'results/publication_validation_2026_11_19'
STUDIES = ('neon_taxonomy_a2', 'neon_taxonomy_a3', 'neon_taxonomy_a3u')
DOCS = ('README.md', 'docs/neon_taxonomy_a2_a3_results.md',
        'docs/taxonomy_audits_validation_2026_11_19.md',
        'docs/taxonomic_verification_t2_plan.md', 'docs/next_experiment_budget.md',
        'docs/paper_claim_evidence_matrix.md', 'docs/paper_methods_results_draft.md',
        'docs/paper_outline_working.md', 'results/README.md')


def verify_manifest(path):
    files = json.loads(path.read_text())['files']
    for name, identity in files.items():
        source = ROOT / name
        assert source.stat().st_size == identity['bytes'], name
        assert sha(source) == identity['sha256'], name
    return len(files)


def seal(path, names):
    files = {name: {'bytes': (ROOT / name).stat().st_size, 'sha256': sha(ROOT / name)}
             for name in sorted(set(names))}
    if path.exists():
        assert json.loads(path.read_text())['files'] == files, 'Sealed manifest differs: ' + str(path)
        return verify_manifest(path)
    save(path, {'scope': 'completed development audits; no fresh confirmation',
                'path_base': 'repository root; ignored caches are hashes only', 'files': files})
    return verify_manifest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    if args.seal and (PUBLICATION / 'artifact_manifest.json').exists():
        raise ValueError('Publication already sealed; verify without --seal')
    history = verify_history()
    for name in ('taxonomic_resolution_t1', 'neon_taxonomy_a1'):
        history[name] = {'files_verified': verify_manifest(ROOT / 'results' / name / 'artifact_manifest.json')}
    studies = {}
    for name in STUDIES:
        runner = importlib.import_module('run_' + name)
        runner.freeze()  # Checks the pre-replay source/input lock, without refreezing.
        output = ROOT / 'results' / name
        assert json.loads((output / 'validation.json').read_text())['passed'] is True
        manifest = output / 'artifact_manifest.json'
        if args.seal:
            files = [str(p.relative_to(ROOT)) for base in (output, ROOT / 'data' / name)
                     for p in base.rglob('*') if p.is_file() and p != manifest]
            files += list(json.loads((output / 'protocol_freeze.json').read_text())['identity'])
            if name == 'neon_taxonomy_a2':
                files.append('scripts/diagnose_neon_taxonomy_a2.py')
            count = seal(manifest, files)
        else:
            count = verify_manifest(manifest)
        studies[name] = {'files_verified': count, 'independent_validation_passed': True}
    suites = ET.parse(PUBLICATION / 'full_tests.xml').getroot().findall('testsuite')
    totals = {key: sum(int(s.attrib[key]) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    assert totals['tests'] > 0 and not any(totals[k] for k in ('failures', 'errors', 'skipped'))
    final = json.loads((ROOT / 'results/neon_taxonomy_a3u/mapping_audit.json').read_text())
    guard = json.loads((ROOT / 'results/neon_taxonomy_a3u/validation.json').read_text())
    report = {'historical_artifacts': history, 'development_audits': studies,
              'full_test_suite': totals, 'new_model_fits': 0, 'new_GPU_jobs': 0,
              'fresh_confirmation': False, 'original_E4_primary_preserved': True,
              'final_contract': {key: final[key] for key in
                  ('cells', 'scorable_cells', 'scorable_sites', 'added_positive_pairs', 'removed_positive_pairs')},
              'guard_unknown_pairs_added_from_A3': guard['extra_unknown_pairs_from_A3']}
    if args.seal:
        for name in DOCS:
            target = PUBLICATION / 'documentation_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        save(PUBLICATION / 'validation.json', report)
        files = [str(p.relative_to(ROOT)) for p in PUBLICATION.rglob('*') if p.is_file()]
        files += ['scripts/validate_taxonomy_audits.py']
        files += ['results/' + name + '/artifact_manifest.json' for name in STUDIES]
        seal(PUBLICATION / 'artifact_manifest.json', files)
    else:
        verify_manifest(PUBLICATION / 'artifact_manifest.json')
        assert report == json.loads((PUBLICATION / 'validation.json').read_text())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
