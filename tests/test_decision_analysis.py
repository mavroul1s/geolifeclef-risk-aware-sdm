import numpy as np
import pytest

from scripts.study_decision_and_ensemble_reliability import paired_full_coverage_change


def test_paired_full_coverage_interval_preserves_constant_improvement():
    r = paired_full_coverage_change(np.full(12, .02), ['a']*10+['b']*2, repetitions=100)
    assert r['gain'] == pytest.approx(.02)
    assert r['ci95'] == pytest.approx([.02, .02])
    assert r['clusters'] == 2


def test_spatial_interval_uses_survey_weighted_paired_changes():
    changes = np.array([.1]*10+[-.1]*2)
    r = paired_full_coverage_change(changes, ['a']*10+['b']*2, repetitions=1000)
    assert r['gain'] == pytest.approx(changes.mean())
    assert r['ci95'] == pytest.approx([-.1, .1])
