"""Preregistered v21 geography, conservative calibration and one-shot assessment.

Only the original v20 training rows are eligible. The original v20 heldouts are
excluded, including from normalization and the matched control's training. A
compatible control must be fitted afresh on partition 0 with the v20 recipe;
the original v20 weights have already seen the new assessment rows.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from scripts.prepare_environmental_challenger import spatial_partitions
from scripts.run_environmental_challenger import policy_counts, ranked_f1, top_rank


SPLIT_SALT = 20250921
EARTH_RADIUS_KM = 6371.0088
BUFFER_KM = 20.0
TOP_K = 20
ALPHAS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2)
GATES = ("uniform", "pa_distance")
SPLIT_NAMES = {0: "training", 1: "checkpoint_selection", 2: "calibration", 3: "assessment"}


def _coordinates(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        values = values[["lat", "lon"]].apply(pd.to_numeric, errors="raise").to_numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Finite latitude/longitude coordinates of shape (n, 2) required")
    if np.any(np.abs(values[:, 0]) > 90) or np.any(np.abs(values[:, 1]) > 180):
        raise ValueError("Latitude/longitude outside geographic bounds")
    return values


def spatial_block_ids(rows: pd.DataFrame | np.ndarray) -> np.ndarray:
    """The fixed one-degree grouping used for assignment and uncertainty."""
    blocks = np.floor(_coordinates(rows)).astype(np.int64)
    return np.asarray([f"{lat}:{lon}" for lat, lon in blocks], dtype=str)


def nearest_training_support(
    train_coordinates: pd.DataFrame | np.ndarray,
    query_coordinates: pd.DataFrame | np.ndarray,
) -> dict[str, np.ndarray]:
    """Distances and PA density use only the actual, buffered PA fit rows."""
    train = _coordinates(train_coordinates)
    query = _coordinates(query_coordinates)
    if not len(train):
        raise ValueError("No training coordinates available")
    tree = BallTree(np.deg2rad(train), metric="haversine")
    radians = np.deg2rad(query)
    if not len(query):
        return {"distance_km": np.empty(0), "within_20km": np.empty(0, dtype=np.int64),
                "within_50km": np.empty(0, dtype=np.int64)}
    distance = tree.query(radians, k=1, return_distance=True)[0][:, 0] * EARTH_RADIUS_KM
    return {"distance_km": distance,
            "within_20km": tree.query_radius(radians, r=20 / EARTH_RADIUS_KM, count_only=True),
            "within_50km": tree.query_radius(radians, r=50 / EARTH_RADIUS_KM, count_only=True)}


def make_partitions(
    rows: pd.DataFrame,
    original_v20_partitions: np.ndarray | None = None,
    minimum_partition_size: int = 100,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Return split (-1/0/1/2/3), manifest, and distance to buffered train.

    Hash percentages are 70/10/10/10 within the original v20 training rows.
    Remove training rows within 20 km of ANY checkpoint/calibration/assessment
    row. No retries, country overrides, label stratification or split tuning.
    Survey IDs appear only in provenance hashes, never in assignment.
    """
    coordinates = _coordinates(rows)
    old = spatial_partitions(rows) if original_v20_partitions is None else np.asarray(original_v20_partitions)
    if old.shape != (len(rows),) or not np.isin(old, (0, 1, 2, 3)).all():
        raise ValueError("Invalid original v20 partition array")
    if minimum_partition_size < 1:
        raise ValueError("minimum_partition_size must be positive")
    blocks = np.floor(coordinates).astype(np.int64)
    bucket = ((blocks[:, 0] * 73856093) ^ (blocks[:, 1] * 19349663) ^ SPLIT_SALT) % 100
    splits = np.select([bucket < 70, bucket < 80, bucket < 90], [0, 1, 2], default=3).astype(np.int8)
    splits[old != 0] = -1
    evaluation = splits > 0
    if not evaluation.any():
        raise ValueError("Registered geography produced no evaluation rows; no adaptive retry")
    candidate_training = np.flatnonzero(splits == 0)
    evaluation_tree = BallTree(np.deg2rad(coordinates[evaluation]), metric="haversine")
    buffered = np.zeros(len(rows), dtype=bool)
    if len(candidate_training):
        distance = evaluation_tree.query(np.deg2rad(coordinates[candidate_training]), k=1)[0][:, 0]
        buffered[candidate_training[distance * EARTH_RADIUS_KM < BUFFER_KM]] = True
        splits[buffered] = -1
    counts = {name: int((splits == index).sum()) for index, name in SPLIT_NAMES.items()}
    if any(count < minimum_partition_size for count in counts.values()):
        raise ValueError(f"Registered geography below minimum {minimum_partition_size}: {counts}; no adaptive retry")
    support = nearest_training_support(coordinates[splits == 0], coordinates)
    if np.any(support["distance_km"][evaluation] < BUFFER_KM - 1e-7):
        raise AssertionError("Geographic training buffer violated")
    block_ids = spatial_block_ids(coordinates)
    diagnostics = {
        "protocol": "v21_original_v20_training_only_one_degree_buffered",
        "split_salt": SPLIT_SALT,
        "nominal_percentages": [70, 10, 10, 10],
        "buffer_km": BUFFER_KM,
        "buffer_applies_to": ["checkpoint_selection", "calibration", "assessment"],
        "original_v20_heldouts_excluded": int((old != 0).sum()),
        "training_rows_excluded_by_buffer": int(buffered.sum()),
        "partition_counts": counts,
        "excluded_rows": int((splits == -1).sum()),
        "partition_blocks": {name: int(len(np.unique(block_ids[splits == index])))
                             for index, name in SPLIT_NAMES.items()},
        "assessment_original_v20_training_only": bool(np.all(old[splits == 3] == 0)),
        "minimum_evaluation_distance_km": float(support["distance_km"][evaluation].min()),
        "support": {name: {
            "distance_km_quantiles": np.quantile(support["distance_km"][splits == index], [0, .25, .5, .75, 1]).tolist(),
            "mean_pa_rows_within_20km": float(support["within_20km"][splits == index].mean()),
            "mean_pa_rows_within_50km": float(support["within_50km"][splits == index].mean()),
        } for index, name in SPLIT_NAMES.items()},
        "labels_or_survey_ids_used_for_assignment": False,
        "adaptive_split_retries": 0,
    }
    if "surveyId" in rows:
        ids = rows.surveyId.to_numpy(dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Partition rows must contain unique survey IDs")
        diagnostics["partition_ids_sha256"] = {
            name: hashlib.sha256(np.sort(ids[splits == index]).astype("<i8").tobytes()).hexdigest()
            for index, name in SPLIT_NAMES.items()}
    if "country" in rows:
        countries = rows.country.fillna("unknown").astype(str).to_numpy()
        diagnostics["partition_countries"] = {
            name: {str(country): int(((splits == index) & (countries == country)).sum())
                   for country in np.unique(countries[splits == index])}
            for index, name in SPLIT_NAMES.items()}
    return splits, diagnostics, support["distance_km"]


def _probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Nonempty finite probability matrix required")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probabilities outside [0, 1]")
    return values


def frozen_v20_blend(reference: np.ndarray, challenger_2025: np.ndarray,
                     challenger_3407: np.ndarray) -> np.ndarray:
    """Exact v20 arithmetic: 75% two-seed mean plus 25% reference."""
    arrays = [_probabilities(values) for values in (reference, challenger_2025, challenger_3407)]
    if len({values.shape for values in arrays}) != 1:
        raise ValueError("v20 component probability shapes differ")
    challenger = arrays[1].astype(np.float32)
    challenger += arrays[2]
    challenger *= 0.5
    return 0.25 * arrays[0].astype(np.float32) + 0.75 * challenger


def registered_policies() -> list[dict]:
    return [{"alpha": alpha, "gate": gate, "k": TOP_K, "species_gate": "all",
             "row_scaling": "none"} for alpha in ALPHAS for gate in GATES]


def mix_probabilities(base: np.ndarray, expert: np.ndarray, distances_km: np.ndarray,
                      policy: dict) -> np.ndarray:
    """PA-finetuned probabilities; no per-row mass rescaling or species pruning."""
    base, expert = _probabilities(base), _probabilities(expert)
    distance = np.asarray(distances_km, dtype=np.float64)
    if base.shape != expert.shape:
        raise ValueError("Base/expert probability shapes differ")
    if distance.shape != (len(base),) or not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("Finite nonnegative PA distances required")
    if (policy.get("alpha") not in ALPHAS or policy.get("gate") not in GATES
            or policy.get("k", TOP_K) != TOP_K or policy.get("species_gate", "all") != "all"
            or policy.get("row_scaling", "none") != "none"):
        raise ValueError("Policy outside preregistered v21 grid")
    alpha = float(policy["alpha"])
    if alpha == 0:
        return base  # Preserve exact dtype, values and tie ordering of unchanged-v20.
    gate = np.ones(len(base)) if policy["gate"] == "uniform" else np.clip(np.log1p(distance) / np.log(51.0), 0, 1)
    weight = (alpha * gate).astype(np.float32)[:, None]
    return (1 - weight) * base.astype(np.float32) + weight * expert.astype(np.float32)


def _targets(targets: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    targets = np.asarray(targets)
    if targets.shape != shape or not np.isin(targets, (0, 1)).all():
        raise ValueError("Binary targets must match probability shape")
    return targets


def sample_f1(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probabilities = _probabilities(probabilities)
    targets = _targets(targets, probabilities.shape)
    # Use the original maximum=50 path even for top20 to retain exact tie parity.
    indices, values = top_rank(probabilities)
    return ranked_f1(targets, indices, policy_counts(values, {"kind": "top_k", "k": TOP_K}))


def select_mixture_policy(targets: np.ndarray, base: np.ndarray, expert: np.ndarray,
                          distances_km: np.ndarray) -> tuple[dict, list[dict]]:
    """Run ONCE on calibration; call separately for PO and zero-PO experts.

    Ties favor the smaller alpha then uniform gate. Caller must persist both
    selections before loading assessment labels and may not refit afterward.
    """
    trials = []
    for policy in registered_policies():
        probability = mix_probabilities(base, expert, distances_km, policy)
        score = float(sample_f1(targets, probability).mean())
        trials.append({**policy, "calibration_f1": score})
    return dict(max(trials, key=lambda trial: trial["calibration_f1"])), trials


def assessment_metrics(targets: np.ndarray, probabilities: np.ndarray, countries: np.ndarray,
                       distances_km: np.ndarray, training_species_counts: np.ndarray) -> tuple[dict, np.ndarray]:
    """Reporting only: country/distance F1, training-frequency recall/cardinality."""
    probabilities = _probabilities(probabilities)
    targets = _targets(targets, probabilities.shape)
    countries = np.asarray(countries, dtype=str)
    distances = np.asarray(distances_km, dtype=np.float64)
    frequency = np.asarray(training_species_counts)
    if countries.shape != (len(targets),) or distances.shape != (len(targets),):
        raise ValueError("Country/distance rows must match targets")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Finite nonnegative PA distances required")
    if frequency.shape != (targets.shape[1],) or not np.isfinite(frequency).all() or np.any(frequency < 0):
        raise ValueError("Training species counts must match vocabulary")
    indices, values = top_rank(probabilities)
    counts = policy_counts(values, {"kind": "top_k", "k": TOP_K})
    scores = ranked_f1(targets, indices, counts)
    def group(mask):
        return {"n": int(mask.sum()), "sample_f1": float(scores[mask].mean()) if mask.any() else None}
    by_country = {str(country): group(countries == country) for country in np.unique(countries)}
    by_distance = {name: group((distances >= low) & (distances < high)) for name, low, high in
                   (("0_to_20km", 0, 20), ("20_to_50km", 20, 50), ("50_to_100km", 50, 100),
                    ("100_to_200km", 100, 200), ("200km_plus", 200, np.inf))}
    selected = indices[:, :min(TOP_K, targets.shape[1])]
    hit_matrix = np.take_along_axis(targets, selected, axis=1)
    recall = {}
    for name, mask in (("zero_training", frequency == 0), ("rare_1_to_25", (frequency >= 1) & (frequency <= 25)),
                       ("common_over_25", frequency > 25)):
        positives = int(targets[:, mask].sum())
        hits = int((hit_matrix * mask[selected]).sum())
        recall[name] = {"species": int(mask.sum()), "target_positives": positives, "true_positives": hits,
                        "micro_recall": hits / positives if positives else None}
    report = {
        "sample_f1": float(scores.mean()), "surveys": len(scores),
        "species": targets.shape[1], "by_country": by_country,
        "country_macro_f1": float(np.mean([value["sample_f1"] for value in by_country.values()])),
        "priority_countries": {country: group(countries == country) for country in ("Bulgaria", "Ukraine", "Switzerland")},
        "by_pa_distance": by_distance, "species_recall": recall,
        "frequency_groups_defined_from": "buffered PA training labels only",
        "cardinality": {"prediction_min": int(counts.min()), "prediction_max": int(counts.max()),
                        "prediction_mean": float(counts.mean()), "target_mean": float(targets.sum(axis=1).mean())},
        "used_for_selection": False,
    }
    return report, scores


def paired_block_bootstrap(candidate_scores: np.ndarray, control_scores: np.ndarray,
                           blocks: np.ndarray, iterations: int = 500, seed: int = SPLIT_SALT) -> dict:
    """Resample whole one-degree blocks and preserve paired survey differences.

    The statistic remains sample-averaged F1, using each resampled block's sum
    and size. This is spatial uncertainty, not a hidden-test score estimate.
    """
    candidate, control = np.asarray(candidate_scores, dtype=np.float64), np.asarray(control_scores, dtype=np.float64)
    blocks = np.asarray(blocks, dtype=str)
    if candidate.ndim != 1 or candidate.shape != control.shape or blocks.shape != candidate.shape or not len(candidate):
        raise ValueError("Paired score/block vectors must have equal nonzero length")
    if not np.isfinite(candidate).all() or not np.isfinite(control).all():
        raise ValueError("Finite paired scores required")
    if np.any(candidate < 0) or np.any(candidate > 1) or np.any(control < 0) or np.any(control > 1):
        raise ValueError("Sample F1 outside [0, 1]")
    if iterations != 500 or seed != SPLIT_SALT:
        raise ValueError("Bootstrap iterations/seed are preregistered at 500/20250921")
    unique, inverse = np.unique(blocks, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two independent spatial blocks required")
    differences = candidate - control
    block_sums = np.bincount(inverse, weights=differences)
    block_sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        draw = rng.integers(0, len(unique), size=len(unique))
        estimates[iteration] = block_sums[draw].sum() / block_sizes[draw].sum()
    interval = np.quantile(estimates, [.025, .975]).tolist()
    return {"mean_difference": float(differences.mean()), "ci95": interval,
            "iterations": iterations, "seed": seed, "spatial_blocks": len(unique),
            "surveys": len(candidate), "unit": "one_degree_spatial_block",
            "statistic": "paired_sample_mean_f1_difference"}


def submission_eligibility(selected_policy: dict, versus_frozen_v20: dict,
                           versus_zero_po: dict, integrity_checks: dict[str, bool]) -> dict[str, bool]:
    """Fail-closed numerical gate only; never writes or submits an artifact."""
    def positive(comparison):
        value = comparison.get("mean_difference", np.nan)
        return bool(np.isfinite(value) and value > 0)
    ci = versus_frozen_v20.get("ci95", [np.nan, np.nan])
    lower = ci[0] if len(ci) == 2 else np.nan
    registered = {key: selected_policy.get(key, default) for key, default in
                  (("alpha", None), ("gate", None), ("k", TOP_K), ("species_gate", "all"), ("row_scaling", "none"))}
    checks = {
        "registered_policy": registered in registered_policies(),
        "nonzero_po_weight": bool(selected_policy.get("alpha", 0) > 0),
        "positive_gain_over_frozen_v20": positive(versus_frozen_v20),
        "positive_spatial_ci_lower_over_frozen_v20": bool(np.isfinite(lower) and lower > 0),
        "positive_gain_over_zero_po": positive(versus_zero_po),
        "spatial_bootstrap_valid": bool(versus_frozen_v20.get("spatial_blocks", 0) >= 2
                                         and versus_frozen_v20.get("iterations") == 500
                                         and versus_frozen_v20.get("seed") == SPLIT_SALT),
        "all_integrity_checks_pass": bool(integrity_checks) and all(value is True for value in integrity_checks.values()),
    }
    checks["eligible_for_official_submission"] = all(checks.values())
    return checks
