"""Read-only monitoring of an already launched v21/v22; never pushes or submits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from scripts.kaggle_artifacts import KaggleReader, SafeKaggleError, sanitize_text


OUTPUT_NAMES = ('v21_report.json', 'failure.json', 'split_manifest.json', 'po_manifest.json',
                'frozen_policies.json', 'pre_assessment_freeze.json', 'assessment_per_survey.csv',
                'GLC25_PA_submission.csv', 'frozen_v20_provenance.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, choices=(21, 22), default=21)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-hours', type=float, default=11.)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(f'artifacts/v{args.version}_review')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = KaggleReader()
    deadline = time.monotonic() + args.max_hours * 3600
    errors = 0
    while time.monotonic() < deadline:
        try:
            state = reader.status()
            state['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
            (args.output_dir / 'monitor_status.json').write_text(json.dumps(state, indent=2))
            print(json.dumps(state), flush=True)
            errors = 0
            if state['status'] in ('complete', 'error', 'failed', 'cancelled', 'canceled'):
                output = reader.output(version=args.version)
                log = sanitize_text(str(output.get('log', '')), reader.secrets)
                (args.output_dir / 'execution.log').write_text(log, encoding='utf-8')
                by_name = {entry['fileName']: entry for entry in output['files']}
                folder = 'ood_po_expert_v21' if args.version == 21 else 'retained_po_v22'
                prefix = f'geolifeclef-risk-aware-sdm/artifacts/{folder}/'
                names = OUTPUT_NAMES if args.version == 21 else (
                    'v22_report.json', 'failure.json', 'split_manifest.json', 'po_manifests.json',
                    'frozen_policies.json', 'pre_assessment_freeze.json', 'assessment_per_survey.csv',
                    'GLC25_PA_submission.csv', 'frozen_v21_provenance.json', 'unchanged_v21_submission.csv')
                downloaded = []
                for name in names:
                    entry = by_name.get(prefix + name)
                    if entry is not None:
                        downloaded.append(reader.download_url(entry['url'], args.output_dir / name))
                (args.output_dir / 'retrieval_manifest.json').write_text(json.dumps({
                    'version': args.version, 'status': state['status'], 'files': downloaded}, indent=2))
                print(json.dumps({'terminal_status': state['status'], 'downloaded': [x['file'] for x in downloaded],
                                  'log_characters': len(log)}), flush=True)
                return 0 if state['status'] == 'complete' else 1
        except SafeKaggleError:
            errors += 1
            print(json.dumps({'read_error': True, 'consecutive_errors': errors}), flush=True)
            if errors >= 5:
                return 2
        time.sleep(50)
    print(json.dumps({'monitor_deadline_reached': True}), flush=True)
    return 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('Read-only monitor failed; raw error details withheld to protect credentials/URLs.', flush=True)
        raise SystemExit(2)
