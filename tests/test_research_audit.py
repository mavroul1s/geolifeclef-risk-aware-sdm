import itertools

import numpy as np
import pytest

from geolifeclef.research_audit import cluster_bootstrap, prefix_f1, set_size_ceiling


def test_prefix_scores_match_explicit_sets_and_reject_duplicates():
    ranks = np.array([[[4, 1, 3, 2]], [[0, 2, 1, 4]]])
    truth = [np.array([1, 2]), np.array([0, 1, 3])]
    scores = prefix_f1(ranks, truth)
    for i, k in itertools.product(range(2), range(1, 5)):
        predicted, actual = set(ranks[i, 0, :k]), set(truth[i])
        assert scores[i, 0, k-1] == pytest.approx(2*len(predicted & actual)/(k+len(actual)))
    with pytest.raises(ValueError, match="duplicate"):
        prefix_f1(np.array([[[1, 1]]]), [np.array([1])])


def test_support_and_size_ceiling_matches_exhaustive_feasible_sets():
    # Check the bound against all subsets of a six-label output vocabulary.
    for m, available, lower, upper in itertools.product(range(1, 5), range(5), [1, 2], [3, 6]):
        if available > m:
            continue
        truth = set(range(m))
        vocabulary = list(range(available)) + list(range(10, 16-available))
        best = max(2*len(set(p) & truth)/(len(p)+m)
                   for k in range(lower, upper+1)
                   for p in itertools.combinations(vocabulary, k))
        assert float(set_size_ceiling(m, available, lower, upper)) == pytest.approx(best)


def test_bootstrap_preserves_whole_clusters_and_survey_weighting():
    values = np.array([1., 1., 1., -1.])
    result = cluster_bootstrap(values, ['a', 'a', 'a', 'b'], repetitions=1000)
    assert result['mean'] == .5  # Equal block weighting would incorrectly return 0.
    assert result['ci95'] == [-1., 1.]
    assert result == cluster_bootstrap(values, ['a', 'a', 'a', 'b'], repetitions=1000)
    with pytest.raises(ValueError):
        cluster_bootstrap([np.nan], ['a'])


def test_archive_check_accepts_only_exact_or_hash_proven_line_ending_recovery(tmp_path, monkeypatch):
    import hashlib
    from scripts import audit_paper_readiness as audit

    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    path = tmp_path/'submission.csv'
    original = b'surveyId,predictions\r\n1,2 3\r\n'
    expected = hashlib.sha256(original).hexdigest()
    path.write_bytes(original)
    assert audit.check_archived_hash(path, expected)['verification'] == 'exact_sha256'
    path.write_bytes(original.replace(b'\r\n', b'\n'))
    assert audit.check_archived_hash(path, expected)['verification'] == 'original_digest_recovered_by_CRLF_only'
    path.write_bytes(b'surveyId,predictions\n1,2 4\n')
    with pytest.raises(ValueError, match='does not match'):
        audit.check_archived_hash(path, expected)
