"""Portable exclusion history recovered from the original frozen v25 notebook."""
import hashlib
import json
from pathlib import Path


def historical_consumed_paths(root, last_version):
    """Equivalent union inputs; never fabricate per-version assessment records."""
    if last_version < 24:
        raise ValueError('The frozen consolidated history covers v21 through v24')
    root = Path(root)
    base = root/'results/competition_history'
    union = base/'consumed_assessment_v21_v24.csv'
    metadata = json.loads((base/'consumed_assessment_v21_v24_provenance.json').read_text())
    if hashlib.sha256(union.read_bytes()).hexdigest() != metadata['csv_sha256']:
        raise ValueError('Consumed assessment history digest mismatch')
    return [union, *[root/f'results/v{v}_kaggle_output/assessment_per_survey_v{v}.csv'
                     for v in range(25, last_version+1)]]
