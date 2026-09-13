"""Shared entry point for the one master notebook and Kaggle API bootstrap."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    os.environ.setdefault('GLC_PIPELINE_STARTED_AT', str(time.time()))
    os.environ['PYTHONUNBUFFERED'] = '1'
    started = float(os.environ['GLC_PIPELINE_STARTED_AT'])
    def remaining():
        seconds = 10.5 * 3600 - (time.time() - started)
        if seconds <= 0:
            raise TimeoutError('The complete v21 notebook exceeded 10.5 hours')
        return seconds
    root = Path('/kaggle/input')
    data = next((p for p in (root / 'competitions/geolifeclef-2025', root / 'geolifeclef-2025') if p.is_dir()), None)
    if data is None:
        raise FileNotFoundError('Attach the GeoLifeCLEF 2025 competition')
    candidates = [p for p in root.rglob('geolifeclef-v20-frozen-control') if p.is_dir()]
    bundles = [q.parent for p in candidates for q in p.rglob('provenance.json')]
    if len(bundles) != 1:
        raise ValueError('Exactly one pinned private v20 frozen-control input is required')
    subprocess.run([sys.executable, '-m', 'pytest'], check=True, timeout=remaining())
    subprocess.run([sys.executable, '-m', 'scripts.run_ood_po_expert', '--data-root', str(data),
                    '--frozen-dir', str(bundles[0]), '--max-hours', '10.5'], check=True, timeout=remaining())
    subprocess.run([sys.executable, '-m', 'pytest'], check=True, timeout=remaining())
    report_path = Path('artifacts/ood_po_expert_v21/v21_report.json')
    report = json.loads(report_path.read_text())
    report['total_pipeline_hours'] = (time.time() - started) / 3600
    report['notebook_tests_before_and_after_passed'] = True
    report['source_commit'] = os.environ.get('GLC_SOURCE_COMMIT', 'local-master-notebook')
    if report['total_pipeline_hours'] >= 10.5:
        raise TimeoutError('v21 final checks exceeded the registered total budget')
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'v21_complete': True, 'hours': report['total_pipeline_hours'],
                      'official_submission_allowed': report['submission_gate']['eligible_for_official_submission']}))


if __name__ == '__main__':
    main()
