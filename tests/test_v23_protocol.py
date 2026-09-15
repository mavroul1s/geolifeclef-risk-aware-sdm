import numpy as np
import pandas as pd

import scripts.v23_protocol as protocol


def _components(rows):
    return {"pa": np.linspace(0, 1, rows), "po": np.linspace(1, 0, rows),
            "disagreement": np.linspace(0, 1, rows)}


def test_v23_ood_gate_and_variable_cardinality_are_bounded():
    components = protocol.ood_components(
        np.array([0.0, 20.0, 500.0]), np.array([0.0, 10.0, 500.0]),
        np.array([100.0, 10.0, 0.1]), np.array([0.0, 0.02, 0.2]),
    )
    for gate in protocol.GATES:
        values = protocol.gate_values(components, gate)
        assert values.shape == (3,)
        assert np.all((values >= 0) & (values <= 1))
    policy = {"alpha": 0.1, "gate": "pa_po_disagreement",
              "k_near": 16, "k_far": 20, "transition": 0.5}
    counts = protocol.policy_counts(policy, components)
    assert set(counts).issubset({16, 20})


def test_v23_policy_selection_scores_per_survey_and_includes_unchanged_control():
    base = np.zeros((2, 32), dtype=np.float32)
    expert = np.zeros((2, 32), dtype=np.float32)
    base[0, :20], base[1, :20] = np.linspace(1, 0.2, 20), np.linspace(1, 0.2, 20)
    expert[:] = base
    targets = np.zeros((2, 32), dtype=np.uint8)
    targets[:, :18] = 1
    selected, trials = protocol.select_policy(targets, base, expert, _components(2))
    assert selected in trials
    # Regression for Kaggle kernel v23: selected records include the audit score
    # and must remain valid inputs to final mixing and cardinality selection.
    mixed = protocol.mix(base, expert, _components(2), selected)
    counts = protocol.policy_counts(selected, _components(2))
    assert mixed.shape == base.shape
    assert counts.shape == (2,)
    assert any(item["alpha"] == 0 for item in trials)
    assert all(16 <= item["k_near"] <= 20 and 20 <= item["k_far"] <= 28 for item in trials)
    with np.testing.assert_raises_regex(ValueError, "Unregistered v23 mixture"):
        protocol.mix(base, expert, _components(2), {**selected, "alpha": 0.123})


def test_v23_crossfit_uses_only_buckets_not_consumed_by_v22(monkeypatch):
    rows = pd.DataFrame({"surveyId": np.arange(2400), "lat": 40.0, "lon": 20.0,
                         "country": "synthetic"})
    buckets = np.arange(len(rows)) % 100
    monkeypatch.setattr(protocol, "spatial_partitions", lambda frame: np.zeros(len(frame), dtype=np.int8))
    monkeypatch.setattr(protocol, "make_partitions",
        lambda frame, original, minimum_partition_size: (np.zeros(len(frame), dtype=np.int8), {}, None))
    monkeypatch.setattr(protocol, "hashed_blocks", lambda frame, seed: buckets[:len(frame)])
    monkeypatch.setattr(protocol, "spatial_block_ids",
        lambda frame: np.asarray([f"block-{value}" for value in frame.surveyId]))
    monkeypatch.setattr(protocol, "nearest_training_support",
        lambda train, query: {"distance_km": np.full(len(query), 25.0),
                              "within_20km": np.zeros(len(query), dtype=int),
                              "within_50km": np.zeros(len(query), dtype=int)})

    class FakeTree:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, values, k=1):
            return np.full((len(values), 1), 25.0 / protocol.EARTH_RADIUS_KM), np.zeros((len(values), 1), int)

    monkeypatch.setattr(protocol, "BallTree", FakeTree)
    folds = protocol.crossfit_partitions(rows, minimum=10)
    assert len(folds) == 2
    assert all(np.all(buckets[split == 3] >= 40) for split, _, _ in folds)
    assert not np.any((folds[0][0] == 3) & (folds[1][0] == 3))


def test_v23_submission_gate_is_fail_closed():
    assessment = {"comparisons": {
        "frozen_v22": {"mean_difference": 0.01, "ci95": [0.001, 0.02]},
        "zero_po": {"mean_difference": 0.01, "ci95": [0.0, 0.02]},
        "retained_po": {"mean_difference": 0.001, "ci95": [-0.01, 0.02]},
    }, "fold_sample_f1": [
        {"single_head_ensemble": 0.31, "frozen_v22": 0.30},
        {"single_head_ensemble": 0.32, "frozen_v22": 0.31},
    ]}
    policies = {**{f"fold_{fold}": {"single_head_ensemble": {"selected": {"alpha": 0.1}}}
                  for fold in (0, 1)},
                "deployment": {"single_head_ensemble": {"selected": {"alpha": 0.1}}}}
    assert protocol.submission_gate(assessment, {"ok": True}, policies)["eligible_for_manual_submission"]
    policies["deployment"]["single_head_ensemble"]["selected"]["alpha"] = 0
    assert not protocol.submission_gate(assessment, {"ok": True}, policies)["eligible_for_manual_submission"]
