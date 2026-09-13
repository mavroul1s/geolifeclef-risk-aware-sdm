import time

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.prepare_po_expert import prepare_po, coordinate_features, _stream_environment
from scripts.run_ood_po_expert import normalized_environment


def fixture_data(root, repeat=False):
    root.mkdir()
    rows = pd.DataFrame({
        'publisher': ['A', 'A', 'A', 'B', 'A', 'A', 'A'],
        'lat': [45.01, 45.012, 45.02, 45.01, 45.08, 0., 45.1],
        'lon': [6.01, 6.011, 6.02, 6.01, 6.08, 0., 6.1],
        'speciesId': [0, 0, 1, 2, 3, 4, 99999], 'surveyId': np.arange(1, 8),
        'geoUncertaintyInM': [10] * 7,
    })
    for key in ('year', 'month', 'day', 'taxonRank', 'date', 'dayOfYear'):
        rows[key] = 1
    raw = pd.concat([rows, rows.iloc[[0, 1, 0]]], ignore_index=True) if repeat else rows
    raw.to_csv(root / 'GLC25_P0_metadata_train.csv', index=False)
    pa = pd.DataFrame({'surveyId': [100, 101], 'lat': [0., .2], 'lon': [0., .2]})
    test = pd.DataFrame({'surveyId': [102], 'lat': [0.4], 'lon': [0.4]})
    env = root / 'EnvironmentalValues'
    env.mkdir()
    for family in ('landcover', 'temperature'):
        values = pd.DataFrame({'surveyId': rows.surveyId, family + '-1': [1.] * len(rows), family + '-2': [0.] * len(rows)})
        values.to_csv(env / f'GLC25-PO-train-{family}.csv', index=False)
        for source, ids in (('train', pa.surveyId), ('test', test.surveyId)):
            pd.DataFrame({'surveyId': ids, family + '-1': np.arange(len(ids)), family + '-2': [0.] * len(ids)}).to_csv(env / f'GLC25-PA-{source}-{family}.csv', index=False)
    return pa, test


def test_streamed_pseudo_surveys_preserve_vocabulary_and_ignore_record_density(tmp_path):
    pa, test = fixture_data(tmp_path / 'a')
    pa2, test2 = fixture_data(tmp_path / 'b', repeat=True)
    first = prepare_po(tmp_path / 'a', tmp_path / 'oa', pa, test, np.arange(5016), time.monotonic() + 60)
    second = prepare_po(tmp_path / 'b', tmp_path / 'ob', pa2, test2, np.arange(5016), time.monotonic() + 60)
    a = sparse.load_npz(first['paths']['po_labels'])
    b = sparse.load_npz(second['paths']['po_labels'])
    assert a.shape == b.shape == (3, 5016)
    assert (a != b).nnz == 0
    assert first['counts']['near_pa_coordinate_excluded_rows'] == 1
    assert first['counts']['unknown_or_invalid_species_rows'] == 1
    for key in ('po_features', 'po_weights', 'po_environment_raw'):
        np.testing.assert_array_equal(np.load(first['paths'][key]), np.load(second['paths'][key]))
    support = np.load(first['paths']['po_support'])
    weights = np.load(first['paths']['po_weights'])
    same_cell = np.all(support['cells'] == support['cells'][0], axis=1)
    assert np.isclose(weights[same_cell][0], weights[same_cell][1])
    assert first['pa_labels_accessed'] is False
    assert first['po_covered_species'] == 4


def test_environment_join_rejects_missing_duplicate_and_expired_budget(tmp_path):
    path = tmp_path / 'env.csv'
    pd.DataFrame({'surveyId': [2, 1], 'value': [20, 10]}).to_csv(path, index=False)
    values, _ = _stream_environment(path, np.array([1, 2]), ['value'], time.monotonic() + 60)
    assert values[:, 0].tolist() == [10, 20]
    with pytest.raises(ValueError, match='absent'):
        _stream_environment(path, np.array([3]), ['value'], time.monotonic() + 60)
    pd.DataFrame({'surveyId': [1, 1], 'value': [10, 20]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        _stream_environment(path, np.array([1]), ['value'], time.monotonic() + 60)
    with pytest.raises(TimeoutError):
        _stream_environment(path, np.array([1]), ['value'], time.monotonic() - 1)


def test_matching_pa_training_only_normalization_and_missing_indicators():
    pa = np.array([[1., np.nan], [3., np.nan], [999., 9.]])
    a, test, po, stats = normalized_environment(pa, np.array([[2., 5.]]), np.array([[4., np.nan]]), np.array([0, 1]))
    assert stats['mean'] == [2., 0.]
    assert a.shape == (3, 4)
    assert po[0, 3] == 1
    assert a[2, 0] == 12
    assert np.isfinite(a).all() and np.isfinite(po).all()
    assert coordinate_features(np.array([[45., 6.]])).shape == (1, 34)

