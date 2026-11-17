import itertools

import numpy as np
import pandas as pd
import pytest

from geolifeclef.reliability import (
    R1_METADATA_FEATURES, R1_MODEL_FEATURES, r1_accept, r1_area_bins,
    r1_binary_entropy, r1_fit_selectors, r1_signals,
)


def test_quota_selection_is_optimal_for_fixed_group_budgets_and_permutation_invariant():
    risk = np.array([.9, .1, .4, .2, .8, .3])
    ids = np.arange(6)+100
    groups = np.array(['a']*3+['b']*3)
    chosen = r1_accept(risk, ids, 2/3, groups)
    feasible = [take for take in itertools.combinations(range(6), 4)
                if sum(groups[list(take)] == 'a') == 2]
    assert risk[chosen].sum() == pytest.approx(min(risk[list(t)].sum() for t in feasible))
    p = np.array([4, 0, 3, 5, 2, 1])
    assert set(ids[chosen]) == set(ids[p][r1_accept(risk[p], ids[p], 2/3, groups[p])])
    for c in [.2, .4, .8, 1.]:
        a = r1_accept(risk, ids, c, groups)
        assert a.sum() == int(np.ceil(len(ids)*c))
        for group in np.unique(groups):
            q = a[groups == group].sum()
            ideal = (groups == group).sum()*a.mean()
            assert abs(q-ideal) <= 1


def test_identical_members_have_zero_epistemic_signals_and_disagreement():
    p = np.array([[.8, .3, .1], [.2, .4, .6]], np.float32)
    ranks = np.argsort(-p, axis=1)[:, :2]
    sets = np.repeat(ranks[:, None], 3, axis=1)
    signals, _ = r1_signals(p, p*p, r1_binary_entropy(p).mean(1), sets, k=2)
    assert np.allclose(signals['ensemble_variance'], 0)
    assert np.allclose(signals['ensemble_mutual_information'], 0)
    assert np.allclose(signals['set_disagreement'], 0)


def test_known_ensemble_has_positive_variance_information_and_set_disagreement():
    members = np.array([[[.9, .1]], [[.1, .9]]], np.float32)
    mean = members.mean(0)
    sets = np.array([[[0], [1]]])
    signals, _ = r1_signals(mean, np.square(members).mean(0),
        r1_binary_entropy(members).mean((0, 2)), sets, k=1)
    assert signals['ensemble_variance'][0] == pytest.approx(.16)
    assert signals['ensemble_mutual_information'][0] > .3
    assert signals['set_disagreement'][0] == 1


def test_area_edges_use_training_covariates_and_missingness_is_explicit():
    training = [1, 10, 100, 1000, 10000]
    bins, edges = r1_area_bins([2, 20, np.nan, -1], training)
    _, repeated = r1_area_bins([2, 20, 1e10], training)
    assert np.array_equal(edges, repeated)
    assert bins[2:].tolist() == ['missing', 'missing']


def test_error_regressors_cannot_read_assessment_labels():
    rng = np.random.default_rng(12)
    names = list(dict.fromkeys(R1_METADATA_FEATURES+R1_MODEL_FEATURES+['log_pa_distance', 'environment_novelty']))
    cal = pd.DataFrame(rng.normal(size=(80, len(names))), columns=names)
    cal['loss'] = rng.uniform(size=80)
    assessment = cal.drop(columns='loss').iloc[:12].copy()
    assessment['surveyId'] = np.arange(12)
    config = dict(max_iter=2, max_leaf_nodes=3, min_samples_leaf=5, early_stopping=False, random_state=1)
    first, _ = r1_fit_selectors(cal, assessment, config)
    # Assessment may contain adversarial target columns; fitted risks must be unchanged.
    assessment['loss'] = 999
    assessment['sample_F1'] = -999
    second, _ = r1_fit_selectors(cal, assessment, config)
    assert set(first) == set(second)
    assert all(np.array_equal(first[k], second[k]) for k in first)
