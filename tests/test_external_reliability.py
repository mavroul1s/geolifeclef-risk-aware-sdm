import numpy as np
import pandas as pd
import pytest

from geolifeclef.external_reliability import (
    apply_calibration, decode_counts, decision_features, decision_outcomes,
    fit_calibration, prediction_context, guarded_target_rows, joint_ecological_comparison)


def test_native_range_mask_is_applied_before_decision_and_confidence():
    eligible = np.array([[1, 0, 1]], bool)
    p = apply_calibration([[.7, .99, .1]], {'slope': 1., 'intercept': 0.}, eligible)
    ctx = prediction_context(p, eligible)
    counts = decode_counts(ctx, 'top18')
    assert p[0, 1] == 0 and counts[0] == 2
    out = decision_outcomes([[1, -1, 0]], eligible, ctx['rank'], counts, np.array([1, 1, 0], bool))
    assert out.sample_F1.iloc[0] == pytest.approx(2/3)
    assert out.rare_truth.iloc[0] == 1


def test_empty_prediction_uses_declared_event_confidence_and_both_f1_conventions():
    eligible = np.ones((2, 3), bool)
    p = np.array([[.01, .01, .01], [.8, .1, .01]], np.float32)
    ctx = prediction_context(p, eligible)
    count = decode_counts(ctx, 'expected_F1_0_80')
    assert count.tolist() == [0, 1]
    features = decision_features(ctx, count, pd.DataFrame({'id': [0, 1]}), np.array([1, 0, 0], bool))
    assert features.SDC.iloc[0] == pytest.approx(.99**3)
    out = decision_outcomes([[0, 0, 0], [1, 0, 0]], eligible, ctx['rank'], count, np.array([1, 0, 0], bool))
    assert out.sample_F1.tolist() == [1, 1]
    assert out.sample_F1_zero_empty.tolist() == [0, 1]


def test_no_eligible_species_never_yields_spurious_predictions():
    eligible = np.zeros((1, 3), bool)
    ctx = prediction_context(np.zeros((1, 3)), eligible)
    for decoder in ['top18', 'expected_F1_0_80', 'expected_F1_8_40', 'threshold_0_80']:
        assert decode_counts(ctx, decoder, .1).tolist() == [0]


def test_feature_generation_has_no_target_argument_or_hidden_target_dependency():
    p = np.array([[.8, .2, .1]], np.float32)
    ctx = prediction_context(p, np.ones_like(p, bool))
    meta = pd.DataFrame({'area': [1]})
    a = decision_features(ctx, np.array([1]), meta, np.array([1, 0, 0], bool))
    b = decision_features(ctx, np.array([1]), meta.assign(sample_F1=-1e9), np.array([1, 0, 0], bool))
    np.testing.assert_array_equal(a.SDC, b.SDC)
    assert a.predicted_rare_mass.iloc[0] == pytest.approx(.8)


def test_calibration_respects_mask_and_returns_valid_probability_transform():
    p = np.tile([.9, .1, .5], (20, 1))
    eligible = np.tile([1, 1, 0], (20, 1)).astype(bool)
    y = np.tile([1, 0, -1], (20, 1))
    result = fit_calibration(p, y, eligible, {'slope': [.25, 4.], 'intercept': [-12., 8.]})
    assert result['eligible_pairs'] == 40
    calibrated = apply_calibration(p, result, eligible)
    assert calibrated[:, 0].mean() > calibrated[:, 1].mean()
    assert np.all(calibrated[:, 2] == 0)


def test_inconsistent_published_target_mask_fails_closed():
    with pytest.raises(ValueError):
        decision_outcomes([[1, 1]], np.array([[1, 0]], bool), np.array([[0, 1]]),
                          np.array([1]), np.array([1, 0], bool))


def test_target_guard_prevents_any_assessment_indexing_in_fit_phase():
    class Trap:
        shape = (3, 2)
        dtype = np.dtype('int8')
        chunks = None
        def __getitem__(self, rows):
            assert 2 not in rows, 'assessment materialized'
            return np.ones((len(rows), 2), np.int8)
    cohort = pd.DataFrame({'role': ['probability_calibration','risk_calibration','assessment'],
                           'prediction_row': [0,1,2]})
    for phase, admitted in [('fit', False), ('fit', True), ('assess', False)]:
        with pytest.raises(PermissionError):
            guarded_target_rows(Trap(), cohort, 'assessment', phase, admitted)
    assert guarded_target_rows(Trap(), cohort, 'risk_calibration', 'fit').shape == (1,2)


def test_joint_bootstrap_does_not_count_duplicated_models_as_independent_sites():
    from geolifeclef.ecological_selection import paired_ecological_comparison
    score = np.array([.2,.5,.8,.4,.3,.1]);rare = np.array([2,1,3,1,2,1])
    a = np.array([1,0,1,0,1,0], bool);b = ~a
    blocks = np.array(['a','a','b','b','c','c'])
    single = paired_ecological_comparison(score, rare, a, b, blocks, repetitions=200, confidence=.975)
    joint = joint_ecological_comparison(np.tile(score, (4,1)), rare, np.tile(a,(4,1)),
        np.tile(b,(4,1)), blocks, repetitions=200)
    for key in ['F1_gain','rare_coverage_gain','F1_interval','rare_coverage_interval']:
        np.testing.assert_allclose(joint[key], single[key])
    assert joint['clusters'] == 3 and joint['models'] == 4


def test_shared_model_bootstrap_averages_contrasts_before_interval():
    score = np.array([[.1,.9,.2,.8], [.9,.1,.8,.2]])
    a = np.array([[1,0,1,0], [1,0,1,0]], bool);b = ~a
    result = joint_ecological_comparison(score, np.ones(4), a, b, ['a','a','b','b'], repetitions=100)
    assert result['F1_gain'] == pytest.approx(0)
    np.testing.assert_allclose(result['F1_interval'], [0,0], atol=1e-12)


def test_external_report_writer_serializes_numpy_scalars_without_nonfinite_values(tmp_path):
    import json
    from scripts.run_external_e1 import save
    path = tmp_path/'report.json'
    save(path, {'passed': np.bool_(False), 'count': np.int64(3), 'metric': np.float64(.1)})
    assert json.loads(path.read_text()) == {'passed': False, 'count': 3, 'metric': .1}
    with pytest.raises(ValueError):
        save(path, {'metric': float('nan')})
