"""Fail-closed validation, one official v21 attempt, and read-only score retrieval."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scripts.kaggle_artifacts import KaggleReader, SafeKaggleError, ROOT, COMPETITION
from scripts.ood_po_protocol import submission_eligibility
from scripts.run_ood_po_expert import validate_submission
from scripts.stage_frozen_v20 import sha256_file, verify_bundle


def validate_report(report: dict, expected_commit: str) -> None:
    """Recompute numerical permission; do not trust a lone eligibility boolean."""
    if report.get('experiment') != 'v21_competition_only_ood_po_expert' or report.get('status') != 'complete':
        raise ValueError('Not a completed v21 experiment')
    if report.get('source_commit') != expected_commit:
        raise ValueError('Source commit differs from the authorized run')
    for field in ('external_data_or_weights', 'test_labels_used', 'post_calibration_finetuning',
                  'original_v20_retrained_to_recover_outputs'):
        if report.get(field) is not False:
            raise ValueError(f'Required negative integrity declaration missing: {field}')
    if report.get('notebook_tests_before_and_after_passed') is not True:
        raise ValueError('Both notebook test passes are required')
    hours = report.get('total_pipeline_hours')
    if not isinstance(hours, (float, int)) or not math.isfinite(hours) or not 0 < hours < 10.5:
        raise ValueError('Registered notebook runtime exceeded or missing')
    if report.get('registered_max_total_hours') != 10.5 or report.get('frozen_v20_source_version') != 20:
        raise ValueError('Wrong runtime registration or frozen source')
    assessment = report['assessment']
    if assessment.get('used_for_selection') is not False:
        raise ValueError('Assessment was used for selection')
    split = report['split']
    if split.get('assessment_original_v20_training_only') is not True or split.get('minimum_evaluation_distance_km', 0) < 20 - 1e-7:
        raise ValueError('Assessment ancestry or geographic buffer invalid')
    if report['submission'].get('rows') != 14784 or report['submission'].get('species_vocabulary') != 5016:
        raise ValueError('Submission dimensions differ from the competition')
    expected_integrity = {'all_5016_species', 'new_assessment_excludes_old_v20_heldouts',
        'twenty_km_training_buffer', 'original_v20_bundle_verified', 'unchanged_v20_csv_exact',
        'production_policy_has_nonzero_po', 'policies_unchanged_since_freeze',
        'production_csv_unchanged_since_freeze', 'competition_only', 'test_labels_unused', 'registered_runtime'}
    integrity = report['integrity']
    if not expected_integrity.issubset(integrity) or not all(v is True for v in integrity.values()):
        raise ValueError('Required integrity checks failed or missing')
    selected = report['selected_policies']
    if selected['deployment']['po'].get('alpha', 0) <= 0:
        raise ValueError('Production candidate has no PO component')
    gate = submission_eligibility(selected['evaluation']['po'], assessment['versus_frozen_v20'],
                                 assessment['versus_zero_po'], integrity)
    if not gate['eligible_for_official_submission'] or report['submission_gate'] != gate:
        raise ValueError('v21 failed the independently recomputed submission gate')
    metrics = assessment['metrics']
    for name in ('frozen_v20_recipe', 'zero_po', 'po'):
        item = metrics[name]
        if item.get('species') != 5016 or item.get('surveys') != split['partition_counts']['assessment']:
            raise ValueError('Assessment dimensions disagree')
    for control, comparison in (('frozen_v20_recipe', 'versus_frozen_v20'), ('zero_po', 'versus_zero_po')):
        delta = metrics['po']['sample_f1'] - metrics[control]['sample_f1']
        if not math.isclose(delta, assessment[comparison]['mean_difference'], abs_tol=1e-12):
            raise ValueError('Reported assessment gains disagree with scores')


@contextmanager
def official_client():
    credential = json.loads((ROOT / 'api_key/kaggle_2.json').read_text(encoding='utf-8-sig'))
    before = {name: os.environ.get(name) for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY')}
    try:
        os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = credential['username'], credential['key']
        sys.path.insert(0, str(ROOT / '.venv/kaggle-api'))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            yield api
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def score_value(value):
    if value is None or str(value).strip().lower() in ('', 'none', 'null', 'nan'):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def matched_scores(api, message):
    matched = [row for row in api.competition_submissions(COMPETITION)
               if getattr(row, 'description', None) == message]
    return [{'ref': str(getattr(row, 'ref', '')), 'status': str(getattr(row, 'status', 'unknown')),
             'public_score': score_value(getattr(row, 'publicScore', None)),
             'private_score': score_value(getattr(row, 'privateScore', None))} for row in matched]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'submit', 'scores'])
    parser.add_argument('--version', type=int, default=21)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/v21_review')
    parser.add_argument('--frozen-dir', type=Path, default=ROOT / 'artifacts/v20_frozen_bundle_v21')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / 'official_submission_receipt.json'
    message = f'v21 PO OOD expert {args.source_commit[:12]}: frozen gated evaluation'
    if args.action == 'scores':
        if not receipt_path.is_file():
            raise ValueError('No recorded submission attempt to inspect')
        with official_client() as api:
            scores = matched_scores(api, message)
        result = {'message': message, 'matching_submissions': scores}
        (args.output_dir / 'official_scores.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    reader = KaggleReader()
    if reader.status()['status'] != 'complete':
        raise ValueError('Kaggle must be complete before validation/submission')
    output = reader.output(version=args.version)
    entries = {entry['fileName']: entry for entry in output['files']}
    prefix = 'geolifeclef-risk-aware-sdm/artifacts/ood_po_expert_v21/'
    for name in ('v21_report.json', 'GLC25_PA_submission.csv', 'frozen_policies.json', 'pre_assessment_freeze.json'):
        entry = entries.get(prefix + name)
        if entry is None:
            raise ValueError('Missing required v21 output: ' + name)
        reader.download_url(entry['url'], args.output_dir / name)
    report = json.loads((args.output_dir / 'v21_report.json').read_text())
    validate_report(report, args.source_commit)
    freeze = json.loads((args.output_dir / 'pre_assessment_freeze.json').read_text())
    if freeze['policies_sha256'] != sha256_file(args.output_dir / 'frozen_policies.json'):
        raise ValueError('Frozen policy digest mismatch')
    bundle = verify_bundle(args.frozen_dir)
    species = np.load(args.frozen_dir / 'species_ids.npy')
    template = pd.read_csv(ROOT / 'artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv').surveyId.to_numpy()
    submission = validate_submission(args.output_dir / 'GLC25_PA_submission.csv', template, species)
    if submission['submission_sha256'] != report['submission']['submission_sha256'] or submission['submission_sha256'] != freeze['production_submission_sha256']:
        raise ValueError('Submission digest differs from report/freeze')
    if submission['submission_sha256'] == bundle['test']['original_submission_sha256']:
        raise ValueError('Candidate is identical to unchanged v20; refusing redundant submission')
    if args.action == 'validate':
        print(json.dumps({'validated': True, 'version': args.version, 'source_commit': args.source_commit,
                          'submission_sha256': submission['submission_sha256']}))
        return
    # The exclusive receipt precedes any request that could create a submission.
    # An uncertain response is inspected, never automatically retried.
    if receipt_path.exists():
        raise ValueError('Submission receipt already exists; retrieve scores instead of resubmitting')
    with official_client() as api:
        if matched_scores(api, message):
            raise ValueError('An official submission with this unique run message already exists')
        receipt = {'status': 'attempt_started', 'message': message, 'kernel_version': args.version,
                   'source_commit': args.source_commit, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                   'submission_sha256': submission['submission_sha256'], 'automatic_retry_allowed': False}
        with receipt_path.open('x', encoding='utf-8') as handle:
            json.dump(receipt, handle, indent=2)
        response = api.competition_submit(str(args.output_dir / 'GLC25_PA_submission.csv'), message, COMPETITION, quiet=True)
        if isinstance(response, str):
            raise SafeKaggleError('Official client did not confirm submission; inspect scores before any further action')
        receipt['status'] = 'request_returned'
        receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, SafeKaggleError):
        # An external client can also raise ValueError with a signed URL.
        # Emit neither raw exception text nor a traceback at the CLI boundary.
        print('Validation or Kaggle operation failed safely; inspect stored reports and run status before retrying.', file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        print('Kaggle operation failed or is uncertain; raw details withheld. Do not retry a recorded submission.', file=sys.stderr)
        raise SystemExit(2)
