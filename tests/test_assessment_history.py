import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geolifeclef.assessment_history import historical_consumed_paths

ROOT = Path(__file__).resolve().parents[1]


def test_exclusion_union_exactly_matches_original_embedded_v25_history():
    metadata = json.loads((ROOT/'results/competition_history/consumed_assessment_v21_v24_provenance.json').read_text())
    book_path = ROOT/metadata['source_notebook']
    assert hashlib.sha256(book_path.read_bytes()).hexdigest() == metadata['source_notebook_sha256']
    constants = {}
    for cell in json.loads(book_path.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('CONSUMED_ASSESSMENT_IDS_'):
                        constants[target.id] = ast.literal_eval(node.value)
    packed = base64.b64decode(constants['CONSUMED_ASSESSMENT_IDS_B64'])
    assert hashlib.sha256(packed).hexdigest() == constants['CONSUMED_ASSESSMENT_IDS_SHA256']
    original = np.cumsum(np.frombuffer(lzma.decompress(packed), dtype='<u4').astype(np.int64))
    restored = pd.read_csv(historical_consumed_paths(ROOT, 24)[0]).surveyId.to_numpy(np.int64)
    assert len(original) == constants['CONSUMED_ASSESSMENT_IDS_COUNT'] == 55325
    assert np.array_equal(restored, original)
    assert len(historical_consumed_paths(ROOT, 28)) == 5


def test_corrupted_history_is_rejected(tmp_path):
    base = tmp_path/'results/competition_history'; base.mkdir(parents=True)
    (base/'consumed_assessment_v21_v24.csv').write_text('surveyId\n123\n')
    (base/'consumed_assessment_v21_v24_provenance.json').write_text(json.dumps({'csv_sha256': '0'*64}))
    with pytest.raises(ValueError, match='digest mismatch'):
        historical_consumed_paths(tmp_path, 28)
