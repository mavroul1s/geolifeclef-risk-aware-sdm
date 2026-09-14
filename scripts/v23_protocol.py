"""Preregistered v23 geography, OOD gating, and variable-cardinality policy."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from scripts.ood_po_protocol import (
    EARTH_RADIUS_KM,
    make_partitions,
    nearest_training_support,
    spatial_block_ids,
)
from scripts.prepare_environmental_challenger import spatial_partitions
from scripts.run_environmental_challenger import ranked_f1, top_rank
from scripts.v22_protocol import hashed_blocks


V22_SPLIT_SEED = 20250922
V23_SEEDS = (20250923, 3408, 9173)
LARGE_SEED = 27183
CHECKPOINT_EPOCHS = (12, 24, 36, 48, 60, 72)
ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35)
GATES = ("uniform", "pa_distance", "pa_po", "pa_po_disagreement")
CARDINALITIES = ((20, 20), (16, 20), (18, 22), (20, 24), (20, 28))


def _ids_sha256(values: np.ndarray) -> str:
    ordered = np.sort(np.asarray(values, dtype=np.int64)).astype("<i8", copy=False)
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def crossfit_partitions(rows: pd.DataFrame, minimum: int = 100):
    """Use only the untouched remainder of former v21 training for assessment.

    v22 consumed buckets 0--39. They are fixed development-only checkpoint and
    calibration partitions here. Buckets 40--69 and 70--99 are the two new,
    disjoint v23 outer assessments. No salt retry or label-based assignment is
    possible. The other outer fold may be used for training, as in cross-fit.
    """
    original = spatial_partitions(rows)
    previous, _, _ = make_partitions(rows, original, minimum_partition_size=minimum)
    former_v21_training = previous == 0
    bucket = hashed_blocks(rows, V22_SPLIT_SEED)
    block = spatial_block_ids(rows)
    coordinates = rows[["lat", "lon"]].to_numpy(dtype=np.float64)
    selection = former_v21_training & (bucket < 20)
    calibration = former_v21_training & (bucket >= 20) & (bucket < 40)
    assessments = (
        former_v21_training & (bucket >= 40) & (bucket < 70),
        former_v21_training & (bucket >= 70),
    )
    if len(rows) == 88987 and (selection.sum() != 7435 or calibration.sum() != 7044):
        raise ValueError("Consumed v22 development partitions no longer match their frozen counts")
    result = []
    for fold, assessment in enumerate(assessments):
        split = np.full(len(rows), -1, dtype=np.int8)
        split[original == 0] = 0
        split[selection] = 1
        split[calibration] = 2
        split[assessment] = 3
        evaluation = split > 0
        evaluation_blocks = np.unique(block[evaluation])
        split[(split == 0) & np.isin(block, evaluation_blocks)] = -1
        training = np.flatnonzero(split == 0)
        tree = BallTree(np.deg2rad(coordinates[evaluation]), metric="haversine")
        distance = tree.query(np.deg2rad(coordinates[training]), k=1)[0][:, 0] * EARTH_RADIUS_KM
        split[training[distance < 20.0]] = -1
        support = nearest_training_support(rows.loc[split == 0], rows)
        counts = {
            name: int((split == index).sum())
            for index, name in enumerate(("training", "checkpoint_selection", "calibration", "assessment"))
        }
        if min(counts.values()) < minimum:
            raise ValueError(f"Insufficient fixed v23 fold {fold}: {counts}")
        if np.any(bucket[split == 3] < 40) or np.any(previous[split == 3] != 0):
            raise AssertionError("v23 assessment overlaps a consumed or ineligible partition")
        if np.any(support["distance_km"][split > 0] < 20.0 - 1e-7):
            raise AssertionError("v23 geographic buffer violated")
        manifest = {
            "fold": fold,
            "protocol": "v23_remaining_v21_training_crossfit",
            "assignment_seed": V22_SPLIT_SEED,
            "assessment_bucket_range": [40, 70] if fold == 0 else [70, 100],
            "checkpoint_selection_source": "consumed_v22_fold_0_development_only",
            "calibration_source": "consumed_v22_fold_1_development_only",
            "partition_counts": counts,
            "partition_blocks": {
                name: int(np.unique(block[split == index]).size)
                for index, name in enumerate(counts)
            },
            "assessment_ids_sha256": _ids_sha256(rows.surveyId.to_numpy()[split == 3]),
            "minimum_evaluation_distance_km": float(support["distance_km"][split > 0].min()),
            "assessment_previously_consumed": False,
            "selection_and_calibration_previously_consumed": True,
            "labels_or_ids_used_for_assignment": False,
            "adaptive_split_retries": 0,
        }
        result.append((split, support, manifest))
    if np.any((result[0][0] == 3) & (result[1][0] == 3)):
        raise AssertionError("v23 outer assessments overlap")
    return result


def po_context(po_coordinates: np.ndarray, po_species_support: np.ndarray,
               query_coordinates: np.ndarray) -> dict[str, np.ndarray]:
    """Compute label-free PO coverage descriptors from eight nearby groups."""
    po_coordinates = np.asarray(po_coordinates, dtype=np.float64)
    query_coordinates = np.asarray(query_coordinates, dtype=np.float64)
    support = np.asarray(po_species_support, dtype=np.float64)
    if po_coordinates.ndim != 2 or po_coordinates.shape[1] != 2 or support.shape != (len(po_coordinates),):
        raise ValueError("Invalid PO support geometry")
    if query_coordinates.ndim != 2 or query_coordinates.shape[1] != 2:
        raise ValueError("Invalid PO query geometry")
    if not np.isfinite(po_coordinates).all() or not np.isfinite(query_coordinates).all() or (support <= 0).any():
        raise ValueError("PO context requires finite coordinates and positive support")
    tree = BallTree(np.deg2rad(po_coordinates), metric="haversine")
    distance, indices = tree.query(np.deg2rad(query_coordinates), k=min(8, len(po_coordinates)))
    distance *= EARTH_RADIUS_KM
    local_support = (support[indices] / (1.0 + distance)).sum(axis=1)
    return {"distance_km": distance[:, 0], "local_species_support": local_support}


def ood_components(pa_distance: np.ndarray, po_distance: np.ndarray,
                   po_support: np.ndarray, disagreement: np.ndarray) -> dict[str, np.ndarray]:
    values = [np.asarray(item, dtype=np.float64) for item in
              (pa_distance, po_distance, po_support, disagreement)]
    if any(item.ndim != 1 or item.shape != values[0].shape for item in values):
        raise ValueError("OOD component vectors must have one shared shape")
    if any(not np.isfinite(item).all() for item in values) or any((item < 0).any() for item in values):
        raise ValueError("OOD components must be finite and nonnegative")
    pa = np.clip(np.log1p(values[0]) / np.log(201.0), 0.0, 1.0)
    po_distance_risk = np.clip(np.log1p(values[1]) / np.log(101.0), 0.0, 1.0)
    po_support_risk = 1.0 - np.clip(np.log1p(values[2]) / np.log(65.0), 0.0, 1.0)
    po = 0.5 * (po_distance_risk + po_support_risk)
    disagreement_score = np.clip(values[3] / 0.05, 0.0, 1.0)
    return {"pa": pa, "po": po, "disagreement": disagreement_score}


def gate_values(components: dict[str, np.ndarray], gate: str) -> np.ndarray:
    if set(components) != {"pa", "po", "disagreement"} or gate not in GATES:
        raise ValueError("Unregistered v23 OOD gate")
    if gate == "uniform":
        return np.ones_like(components["pa"], dtype=np.float64)
    if gate == "pa_distance":
        return components["pa"]
    if gate == "pa_po":
        return 0.6 * components["pa"] + 0.4 * components["po"]
    return 0.45 * components["pa"] + 0.30 * components["po"] + 0.25 * components["disagreement"]


def registered_policies() -> list[dict]:
    return [
        {"alpha": alpha, "gate": gate, "k_near": near, "k_far": far, "transition": 0.5}
        for alpha in ALPHAS for gate in GATES for near, far in CARDINALITIES
        if gate != "uniform" or near == far
    ]


def _probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite nonempty probability matrix")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities outside [0, 1]")
    return values


def mix(base: np.ndarray, expert: np.ndarray, components: dict[str, np.ndarray], policy: dict) -> np.ndarray:
    base, expert = _probabilities(base), _probabilities(expert)
    if base.shape != expert.shape or policy not in registered_policies():
        raise ValueError("Unregistered v23 mixture")
    gate = gate_values(components, policy["gate"])
    if gate.shape != (len(base),):
        raise ValueError("OOD gate rows do not match probabilities")
    if policy["alpha"] == 0:
        return base
    weight = (policy["alpha"] * gate).astype(np.float32)[:, None]
    return (1.0 - weight) * base.astype(np.float32) + weight * expert.astype(np.float32)


def policy_counts(policy: dict, components: dict[str, np.ndarray]) -> np.ndarray:
    if policy not in registered_policies():
        raise ValueError("Unregistered v23 cardinality policy")
    gate = gate_values(components, policy["gate"])
    return np.where(gate < policy["transition"], policy["k_near"], policy["k_far"]).astype(np.int64)


def per_survey_f1(targets: np.ndarray, probabilities: np.ndarray, policy: dict,
                  components: dict[str, np.ndarray]) -> np.ndarray:
    probabilities = _probabilities(probabilities)
    targets = np.asarray(targets)
    if targets.shape != probabilities.shape or not np.isin(targets, (0, 1)).all():
        raise ValueError("Binary targets must match probabilities")
    indices, _ = top_rank(probabilities, maximum=max(max(pair) for pair in CARDINALITIES))
    return ranked_f1(targets, indices, policy_counts(policy, components))


def select_policy(targets: np.ndarray, base: np.ndarray, expert: np.ndarray,
                  components: dict[str, np.ndarray]):
    trials = []
    for policy in registered_policies():
        values = mix(base, expert, components, policy)
        score = float(per_survey_f1(targets, values, policy, components).mean())
        trials.append({**policy, "calibration_f1": score})
    # Exact ties prefer unchanged v22/top20, then smaller alpha/cardinality change.
    def key(item):
        intervention = abs(item["k_near"] - 20) + abs(item["k_far"] - 20)
        return item["calibration_f1"], -item["alpha"], -intervention, item["gate"] == "uniform"
    return dict(max(trials, key=key)), trials


CHECKPOINT_POLICY = {
    "alpha": 0.10,
    "gate": "uniform",
    "k_near": 20,
    "k_far": 20,
    "transition": 0.5,
}


def checkpoint_score(targets: np.ndarray, base: np.ndarray, expert: np.ndarray) -> float:
    zeros = np.zeros(len(base), dtype=np.float64)
    components = {"pa": zeros, "po": zeros, "disagreement": zeros}
    values = mix(base, expert, components, CHECKPOINT_POLICY)
    return float(per_survey_f1(targets, values, CHECKPOINT_POLICY, components).mean())


def submission_gate(assessment: dict, integrity: dict, policies: dict) -> dict[str, bool]:
    comparisons = assessment["comparisons"]
    checks = {
        "positive_spatial_ci_over_frozen_v22": comparisons["frozen_v22"]["ci95"][0] > 0,
        "positive_gain_over_zero_po": comparisons["zero_po"]["mean_difference"] > 0,
        "primary_beats_retained_po": comparisons["retained_po"]["mean_difference"] > 0,
        "primary_beats_v22_each_fold": all(
            fold["single_head_ensemble"] > fold["frozen_v22"]
            for fold in assessment["fold_sample_f1"]
        ),
        "nonzero_primary_weight_each_fold": all(
            policies[f"fold_{fold}"]["single_head_ensemble"]["selected"]["alpha"] > 0
            for fold in (0, 1)
        ),
        "nonzero_production_weight": policies["deployment"]["single_head_ensemble"]["selected"]["alpha"] > 0,
        "all_integrity_checks_pass": bool(integrity) and all(value is True for value in integrity.values()),
    }
    checks["eligible_for_manual_submission"] = all(checks.values())
    return checks
