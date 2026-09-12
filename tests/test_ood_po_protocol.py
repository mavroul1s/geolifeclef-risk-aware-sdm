import numpy as np
import pandas as pd
import pytest

from scripts.ood_po_protocol import (
    BUFFER_KM, assessment_metrics, frozen_v20_blend, make_partitions,
    mix_probabilities, nearest_training_support, paired_block_bootstrap,
    registered_policies, sample_f1, select_mixture_policy, spatial_block_ids,
    submission_eligibility,
)
from scripts.prepare_environmental_challenger import spatial_partitions
from scripts.run_environmental_challenger import policy_counts, ranked_f1, top_rank


def geography():
    coordinates = np.array([(lat + .99, lon + .99) for lat in range(40, 60) for lon in range(-10, 30)])
    frame = pd.DataFrame(coordinates, columns=["lat", "lon"])
    frame["surveyId"] = np.arange(len(frame))
    frame["country"] = "France"
    frame["speciesId"] = np.arange(len(frame)) % 71
    return frame


def test_new_geography_is_label_id_and_order_independent_and_old_audit_excluded():
    frame = geography()
    frame.loc[:10, "country"] = "Switzerland"
    first, report, _ = make_partitions(frame, minimum_partition_size=1)
    changed = frame.copy()
    changed.surveyId += 10000
    changed.speciesId = 900
    second, _, _ = make_partitions(changed, minimum_partition_size=1)
    np.testing.assert_array_equal(first, second)
    reverse, _, _ = make_partitions(changed.iloc[::-1], minimum_partition_size=1)
    np.testing.assert_array_equal(first, reverse[::-1])
    old = spatial_partitions(frame)
    assert np.all(first[old != 0] == -1)
    assert np.all(first[:11] == -1)
    assert report["assessment_original_v20_training_only"] is True
    assert set(np.unique(first)) == {-1, 0, 1, 2, 3}
    # Evaluation groups have no cross-stage block overlap.
    blocks = spatial_block_ids(frame)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not set(blocks[first == i]) & set(blocks[first == j])


def test_buffer_removes_close_training_and_covers_all_three_evaluations():
    frame = geography()
    # Dense samples on both sides of every degree boundary expose adjacent-block leakage.
    shifted = frame.copy()
    shifted[["lat", "lon"]] -= .98
    shifted.surveyId += len(frame)
    frame = pd.concat([frame, shifted], ignore_index=True)
    split, manifest, distance = make_partitions(frame, np.zeros(len(frame), dtype=np.int8), minimum_partition_size=1)
    assert manifest["training_rows_excluded_by_buffer"] > 0
    for stage in (1, 2, 3):
        assert distance[split == stage].min() >= BUFFER_KM - 1e-7
    assert np.allclose(distance[split == 0], 0)
    with pytest.raises(ValueError, match="no adaptive retry"):
        make_partitions(frame, minimum_partition_size=10000)


def test_support_uses_training_only_and_rejects_invalid_coordinates():
    support = nearest_training_support(np.array([[0, 0]]), np.array([[0, 0], [0, 1]]))
    assert support["distance_km"][1] == pytest.approx(111.195, abs=.01)
    assert support["within_20km"].tolist() == [1, 0]
    assert support["within_50km"].tolist() == [1, 0]
    frame = geography()
    frame.loc[0, "lat"] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        make_partitions(frame, minimum_partition_size=1)


def test_unchanged_v20_arithmetic_and_top20_tie_parity():
    rng = np.random.default_rng(91)
    components = [rng.random((3, 5016)).astype(np.float16) for _ in range(3)]
    challenger = components[1].astype(np.float32)
    challenger += components[2]
    challenger *= .5
    original = .25 * components[0].astype(np.float32) + .75 * challenger
    baseline = frozen_v20_blend(*components)
    np.testing.assert_array_equal(baseline, original)
    baseline[:] = .5  # Ties exercise the exact old top_rank implementation.
    labels = (rng.random(baseline.shape) < .01).astype(np.uint8)
    mixed = mix_probabilities(baseline, components[0], np.array([0, 25, 500]), registered_policies()[0])
    assert mixed is baseline
    ranks, values = top_rank(original * 0 + .5)
    expected = ranked_f1(labels, ranks, policy_counts(values, {"kind": "top_k", "k": 20}))
    np.testing.assert_array_equal(sample_f1(labels, mixed), expected)
    assert mixed.shape[1] == 5016


def test_conservative_grid_and_distance_gate_have_no_row_rescaling():
    baseline = np.full((3, 60), .2, np.float32)
    expert = np.full((3, 60), .8, np.float32)
    policy = {"alpha": .2, "gate": "pa_distance", "k": 20}
    mixed = mix_probabilities(baseline, expert, np.array([0, 50, 500]), policy)
    np.testing.assert_allclose(mixed[:, 0], [.2, .32, .32])
    assert len(registered_policies()) == 12
    with pytest.raises(ValueError, match="outside preregistered"):
        mix_probabilities(baseline, expert, np.ones(3), {"alpha": .5, "gate": "uniform"})
    expert[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite probability"):
        mix_probabilities(baseline, expert, np.ones(3), registered_policies()[0])


def test_calibration_rejects_a_bad_expert_and_ties_choose_unchanged():
    base = np.zeros((4, 60), dtype=np.float32)
    base[:, :20] = .55
    expert = np.zeros_like(base)
    expert[:, 20:40] = 1
    targets = (base > 0).astype(np.uint8)
    selected, trials = select_mixture_policy(targets, base, expert, np.full(4, 100))
    assert selected["alpha"] == 0
    assert selected["gate"] == "uniform"
    assert selected["calibration_f1"] == 1
    assert len(trials) == 12


def test_metrics_frequency_groups_use_training_support_and_empty_priority_country():
    labels = np.zeros((2, 60), dtype=np.uint8)
    labels[0, :20] = 1
    labels[1, 20:40] = 1
    frequency = np.full(60, 50)
    frequency[:10] = 0
    frequency[10:20] = 3
    metric, scores = assessment_metrics(labels, labels.astype(np.float32), np.array(["France", "Bulgaria"]), np.array([21, 201]), frequency)
    assert scores.tolist() == [1, 1]
    assert metric["species_recall"]["zero_training"]["micro_recall"] == 1
    assert metric["species_recall"]["rare_1_to_25"]["micro_recall"] == 1
    assert metric["priority_countries"]["Switzerland"] == {"n": 0, "sample_f1": None}
    assert metric["by_pa_distance"]["20_to_50km"]["n"] == 1
    assert metric["cardinality"]["prediction_min"] == 20


def test_paired_spatial_bootstrap_is_deterministic_and_keeps_whole_groups():
    control = np.full(10, .4)
    candidate = np.array([.5] * 9 + [.2])
    blocks = np.array(["dense"] * 9 + ["sparse"])
    report = paired_block_bootstrap(candidate, control, blocks)
    assert report == paired_block_bootstrap(candidate, control, blocks)
    assert report["mean_difference"] == pytest.approx(.07)
    assert report["ci95"] == pytest.approx([-.2, .1])
    assert report["spatial_blocks"] == 2
    with pytest.raises(ValueError, match="two independent"):
        paired_block_bootstrap(candidate, control, np.full(10, "one"))


def test_submission_gate_requires_positive_assessment_ci_zero_po_and_all_integrity():
    policy = {"alpha": .025, "gate": "uniform", "k": 20}
    comparison = paired_block_bootstrap(np.full(10, .5), np.full(10, .4), np.arange(10))
    checks = submission_eligibility(policy, comparison, comparison, {"labels_untouched": True, "ids_match": True})
    assert all(checks.values())
    assert all(type(value) is bool for value in checks.values())
    for bad in ({**comparison, "mean_difference": 0}, {**comparison, "ci95": [-.1, .2]}):
        assert not submission_eligibility(policy, bad, comparison, {"ids_match": True})["eligible_for_official_submission"]
    assert not submission_eligibility(policy, comparison, {**comparison, "mean_difference": 0}, {"ids_match": True})["eligible_for_official_submission"]
    assert not submission_eligibility({**policy, "alpha": 0}, comparison, comparison, {"ids_match": True})["eligible_for_official_submission"]
    assert not submission_eligibility(policy, comparison, comparison, {})["eligible_for_official_submission"]
    assert not submission_eligibility(policy, comparison, comparison, {"ids_match": False})["eligible_for_official_submission"]
