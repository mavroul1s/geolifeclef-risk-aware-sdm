#!/usr/bin/env python
"""Train one rare-aware model and create an OOD-aware GeoLifeCLEF submission."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geolifeclef.losses import asymmetric_loss
from geolifeclef.models import RareAwareCompetitiveFusionSDM, count_trainable_parameters
from geolifeclef.utils import set_seed
from scripts.train_spatial_competition import MODALITIES, MultimodalNPZDataset, collate


EARTH_RADIUS_KM = 6371.0088
OOD_HOLDOUT_COUNTRIES = {
    "Austria",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Czech Republic",
    "Greece",
    "Hungary",
    "Italy",
    "Latvia",
    "Montenegro",
    "Norway",
    "Poland",
    "Romania",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "Switzerland",
}


def adaptive_cardinalities(
    log_cardinality: np.ndarray,
    multiplier: float = 1.0,
    *,
    minimum: int = 4,
    maximum: int = 50,
) -> np.ndarray:
    predicted = np.expm1(np.asarray(log_cardinality, dtype=np.float64)) * multiplier
    return np.clip(np.rint(predicted), minimum, maximum).astype(np.int64)


def label_sets(labels: np.ndarray, species_ids: np.ndarray) -> list[set[int]]:
    return [set(map(int, species_ids[np.flatnonzero(row)])) for row in labels]


def sample_f1_from_lists(truth: list[set[int]], predictions: list[list[int]]) -> float:
    if len(truth) != len(predictions):
        raise ValueError("Truth and prediction rows do not align")
    scores = []
    for expected, predicted_values in zip(truth, predictions):
        predicted = set(predicted_values)
        scores.append(2 * len(expected & predicted) / max(len(expected) + len(predicted), 1))
    return float(np.mean(scores))


def rank_blend_predictions(
    probabilities: np.ndarray,
    neural_species_ids: np.ndarray,
    cardinalities: np.ndarray,
    *,
    po_candidates: list[dict[int, float]] | None = None,
    pa_candidates: list[dict[int, float]] | None = None,
    ood_mask: np.ndarray | None = None,
    po_weight_id: float = 0.0,
    po_weight_ood: float = 0.0,
    pa_weight: float = 0.0,
    neural_pool: int = 96,
) -> list[list[int]]:
    probabilities = np.asarray(probabilities)
    neural_species_ids = np.asarray(neural_species_ids)
    cardinalities = np.asarray(cardinalities, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(neural_species_ids):
        raise ValueError("Probability/species dimensions do not match")
    if len(probabilities) != len(cardinalities):
        raise ValueError("Probability/cardinality rows do not align")
    rows = len(probabilities)
    po_candidates = po_candidates or [{} for _ in range(rows)]
    pa_candidates = pa_candidates or [{} for _ in range(rows)]
    ood_mask = np.zeros(rows, dtype=bool) if ood_mask is None else np.asarray(ood_mask, bool)
    if not (len(po_candidates) == len(pa_candidates) == len(ood_mask) == rows):
        raise ValueError("Spatial candidate rows do not align")

    pool = min(max(neural_pool, int(cardinalities.max(initial=1))), probabilities.shape[1])
    if pool == probabilities.shape[1]:
        top_indices = np.broadcast_to(np.arange(pool), probabilities.shape)
    else:
        top_indices = np.argpartition(probabilities, -pool, axis=1)[:, -pool:]
    predictions: list[list[int]] = []
    for row_index, candidate_indices in enumerate(top_indices):
        ordered = candidate_indices[
            np.argsort(probabilities[row_index, candidate_indices])[::-1]
        ]
        top_probability = max(float(probabilities[row_index, ordered[0]]), 1e-8)
        scores = {
            int(neural_species_ids[index]): float(probabilities[row_index, index])
            / top_probability
            for index in ordered
        }
        po_weight = po_weight_ood if ood_mask[row_index] else po_weight_id
        for species_id, score in po_candidates[row_index].items():
            scores[int(species_id)] = scores.get(int(species_id), 0.0) + po_weight * float(score)
        for species_id, score in pa_candidates[row_index].items():
            scores[int(species_id)] = scores.get(int(species_id), 0.0) + pa_weight * float(score)
        count = min(max(int(cardinalities[row_index]), 1), len(scores))
        predictions.append(
            [species_id for species_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:count]]
        )
    return predictions


class PresenceOnlyIndex:
    def __init__(self, metadata_path: Path, *, chunksize: int = 500_000):
        coordinates: list[np.ndarray] = []
        species: list[np.ndarray] = []
        rows_seen = 0
        for chunk in pd.read_csv(
            metadata_path,
            usecols=["lat", "lon", "speciesId"],
            chunksize=chunksize,
        ):
            rows_seen += len(chunk)
            chunk = chunk.dropna(subset=["lat", "lon", "speciesId"])
            coordinates.append(chunk[["lat", "lon"]].to_numpy(dtype=np.float64))
            species.append(chunk["speciesId"].to_numpy(dtype=np.int32))
        if not coordinates:
            raise ValueError("No usable PO observations were found")
        self.coordinates = np.concatenate(coordinates)
        self.species_ids = np.concatenate(species)
        self.tree = BallTree(np.deg2rad(self.coordinates), metric="haversine")
        self.rows_seen = rows_seen

    def candidates(
        self,
        query_coordinates: np.ndarray,
        *,
        neighbors: int = 96,
        radius_km: float = 35.0,
        min_count: int = 2,
        maximum_candidates: int = 32,
    ) -> list[dict[int, float]]:
        neighbors = min(neighbors, len(self.species_ids))
        distances, indices = self.tree.query(
            np.deg2rad(np.asarray(query_coordinates, dtype=np.float64)), k=neighbors
        )
        distances *= EARTH_RADIUS_KM
        result: list[dict[int, float]] = []
        for row_distances, row_indices in zip(distances, indices):
            valid = row_distances <= radius_km
            if not valid.any():
                result.append({})
                continue
            local_species = self.species_ids[row_indices[valid]]
            local_distances = row_distances[valid]
            unique_species, inverse, counts = np.unique(
                local_species, return_inverse=True, return_counts=True
            )
            weights = np.exp(-local_distances / 10.0)
            weighted_counts = np.bincount(inverse, weights=weights)
            keep = counts >= min_count
            if not keep.any():
                keep[np.argmax(weighted_counts)] = True
            unique_species = unique_species[keep]
            weighted_counts = weighted_counts[keep]
            order = np.argsort(weighted_counts)[::-1][:maximum_candidates]
            scale = max(float(weighted_counts[order[0]]), 1e-8)
            result.append(
                {
                    int(unique_species[index]): float(weighted_counts[index] / scale)
                    for index in order
                }
            )
        return result


def pa_consensus_candidates(
    query_coordinates: np.ndarray,
    reference_rows: pd.DataFrame,
    reference_pairs: pd.DataFrame,
    *,
    neighbors: int = 5,
    radius_km: float = 10.0,
    agreement: float = 0.8,
) -> tuple[list[dict[int, float]], np.ndarray]:
    reference_rows = reference_rows.drop_duplicates("surveyId")
    reference_ids = reference_rows["surveyId"].to_numpy(dtype=np.int64)
    reference_coordinates = reference_rows[["lat", "lon"]].to_numpy(dtype=np.float64)
    tree = BallTree(np.deg2rad(reference_coordinates), metric="haversine")
    labels_by_survey = (
        reference_pairs.groupby("surveyId")["speciesId"].agg(lambda values: set(map(int, values))).to_dict()
    )
    distances, indices = tree.query(
        np.deg2rad(np.asarray(query_coordinates, dtype=np.float64)),
        k=min(neighbors, len(reference_rows)),
    )
    distances *= EARTH_RADIUS_KM
    candidates: list[dict[int, float]] = []
    for row_distances, row_indices in zip(distances, indices):
        valid_indices = row_indices[row_distances <= radius_km]
        if len(valid_indices) < 3:
            candidates.append({})
            continue
        counts: Counter[int] = Counter()
        for reference_index in valid_indices:
            counts.update(labels_by_survey.get(int(reference_ids[reference_index]), set()))
        threshold = math.ceil(agreement * len(valid_indices))
        candidates.append(
            {
                species_id: count / len(valid_indices)
                for species_id, count in counts.items()
                if count >= threshold
            }
        )
    return candidates, distances[:, 0]


@torch.no_grad()
def predict_with_aux(
    model: RareAwareCompetitiveFusionSDM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    log_cardinality: list[np.ndarray] = []
    for batch in loader:
        labels = batch["labels"]
        inputs = {name: batch[name].to(device, non_blocking=True) for name in MODALITIES}
        logits, _, predicted_log_cardinality = model.forward_with_aux(inputs)
        targets.append(labels.numpy().astype(np.uint8))
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        log_cardinality.append(predicted_log_cardinality.float().cpu().numpy())
    return (
        np.concatenate(targets),
        np.concatenate(probabilities),
        np.concatenate(log_cardinality),
    )


def train_stage(
    model: RareAwareCompetitiveFusionSDM,
    train_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    deadline: float,
    reserve_seconds: float,
    selection_loader: DataLoader | None = None,
    checkpoint_path: Path | None = None,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = {name: batch[name].to(device, non_blocking=True) for name in MODALITIES}
            if torch.rand((), device=device) < 0.5:
                inputs["sentinel"] = torch.flip(inputs["sentinel"], dims=(-1,))
            if torch.rand((), device=device) < 0.5:
                inputs["sentinel"] = torch.flip(inputs["sentinel"], dims=(-2,))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, rare_logits, predicted_log_cardinality = model.forward_with_aux(inputs)
                classification_loss = asymmetric_loss(logits, labels)
                rare_loss = asymmetric_loss(rare_logits, labels[:, model.rare_indices])
                true_log_cardinality = torch.log1p(labels.sum(dim=1))
                cardinality_loss = F.smooth_l1_loss(
                    predicted_log_cardinality.float(), true_log_cardinality.float()
                )
                loss = classification_loss + 0.35 * rare_loss + 0.08 * cardinality_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
        scheduler.step()
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
        }
        if selection_loader is not None:
            targets, probabilities, predicted_log_cardinality = predict_with_aux(
                model, selection_loader, device
            )
            counts = adaptive_cardinalities(predicted_log_cardinality)
            predictions = rank_blend_predictions(
                probabilities,
                np.arange(probabilities.shape[1]),
                counts,
            )
            score = sample_f1_from_lists(
                label_sets(targets, np.arange(targets.shape[1])), predictions
            )
            record["selection_adaptive_f1"] = score
            if score > best_score:
                best_score = score
                if checkpoint_path is not None:
                    torch.save({"model_state": model.state_dict(), "score": score}, checkpoint_path)
        seconds = time.monotonic() - epoch_started
        record["seconds"] = seconds
        record["examples_per_second"] = samples / max(seconds, 1e-6)
        history.append(record)
        print(record)
        if time.monotonic() + seconds * 1.25 + reserve_seconds >= deadline and epoch < epochs:
            raise TimeoutError("Insufficient registered Kaggle time to finish the single-model run")
    return history


def tune_postprocessing(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predicted_log_cardinality: np.ndarray,
    species_ids: np.ndarray,
    po_candidates: list[dict[int, float]],
    pa_candidates: list[dict[int, float]],
    ood_mask: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    truth = label_sets(targets, species_ids)
    trials: list[dict[str, float]] = []
    for cardinality_multiplier in (0.75, 0.9, 1.0, 1.15, 1.3):
        counts = adaptive_cardinalities(predicted_log_cardinality, cardinality_multiplier)
        for po_weight_ood in (0.0, 0.2, 0.4, 0.6):
            for pa_weight in (0.0, 0.4, 0.8):
                predictions = rank_blend_predictions(
                    probabilities,
                    species_ids,
                    counts,
                    po_candidates=po_candidates,
                    pa_candidates=pa_candidates,
                    ood_mask=ood_mask,
                    po_weight_id=0.05,
                    po_weight_ood=po_weight_ood,
                    pa_weight=pa_weight,
                )
                score = sample_f1_from_lists(truth, predictions)
                trials.append(
                    {
                        "cardinality_multiplier": cardinality_multiplier,
                        "po_weight_id": 0.05,
                        "po_weight_ood": po_weight_ood,
                        "pa_weight": pa_weight,
                        "sample_f1": score,
                    }
                )
    return max(trials, key=lambda trial: trial["sample_f1"]), trials


def write_submission(
    output_path: Path,
    template_path: Path,
    sample_ids: np.ndarray,
    predictions: list[list[int]],
) -> dict[str, object]:
    template = pd.read_csv(template_path)
    if list(template.columns) != ["surveyId", "predictions"]:
        raise ValueError(f"Unexpected sample-submission columns: {list(template.columns)}")
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    template_ids = template["surveyId"].to_numpy(dtype=np.int64)
    if len(sample_ids) != len(predictions) or len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Test IDs and prediction rows must align and be unique")
    if set(template_ids.tolist()) != set(sample_ids.tolist()):
        raise ValueError("Prediction IDs do not match the official sample submission")
    prediction_by_id = {
        int(survey_id): " ".join(map(str, row))
        for survey_id, row in zip(sample_ids, predictions)
    }
    submission = pd.DataFrame(
        {
            "surveyId": template_ids,
            "predictions": [prediction_by_id[int(value)] for value in template_ids],
        }
    )
    cardinalities = submission["predictions"].str.split().str.len()
    if bool(submission["predictions"].eq("").any()) or int(cardinalities.min()) < 1:
        raise ValueError("Submission contains an empty prediction row")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return {
        "submission_path": str(output_path),
        "submission_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": int(len(submission)),
        "unique_survey_ids": int(submission["surveyId"].nunique()),
        "prediction_count": {
            "minimum": int(cardinalities.min()),
            "median": float(cardinalities.median()),
            "mean": float(cardinalities.mean()),
            "maximum": int(cardinalities.max()),
        },
        "template_order_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--stage-one-epochs", type=int, default=12)
    parser.add_argument("--full-finetune-epochs", type=int, default=6)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rare-max-occurrences", type=int, default=50)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--cleanup-cache", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_labels_used") is not False:
        raise ValueError("Official test manifest must confirm that test labels were not used")
    deadline = started + args.max_hours * 3600 - float(manifest["preparation_seconds"])
    if deadline <= started:
        raise TimeoutError("Preprocessing exhausted the Kaggle time budget")

    train_dataset = MultimodalNPZDataset(args.data_dir / "full_train.npz")
    test_dataset = MultimodalNPZDataset(args.data_dir / "official_test.npz")
    train_ids = train_dataset.arrays["sample_id"].astype(np.int64)
    test_ids = test_dataset.arrays["sample_id"].astype(np.int64)
    species_ids = np.asarray(manifest["species_ids"], dtype=np.int64)
    label_frequency = train_dataset.arrays["labels"].sum(axis=0)
    rare_indices = np.flatnonzero(label_frequency <= args.rare_max_occurrences)

    metadata = pd.read_csv(args.data_root / "GLC25_PA_metadata_train.csv")
    pairs = metadata[["surveyId", "speciesId"]].dropna().drop_duplicates().astype("int64")
    rows = metadata.drop_duplicates("surveyId").set_index("surveyId")
    aligned_rows = rows.loc[train_ids]
    holdout_mask = aligned_rows["country"].isin(OOD_HOLDOUT_COUNTRIES).to_numpy()
    holdout_hash = (train_ids * 2654435761) % 2
    selection_indices = np.flatnonzero(holdout_mask & (holdout_hash == 0))
    tuning_indices = np.flatnonzero(holdout_mask & (holdout_hash == 1))
    core_indices = np.flatnonzero(~holdout_mask)
    if min(len(selection_indices), len(tuning_indices)) < 1000:
        raise ValueError("Geographic holdout is unexpectedly small")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    core_loader = DataLoader(Subset(train_dataset, core_indices), shuffle=True, **loader_kwargs)
    selection_loader = DataLoader(
        Subset(train_dataset, selection_indices), shuffle=False, **loader_kwargs
    )
    tuning_loader = DataLoader(
        Subset(train_dataset, tuning_indices), shuffle=False, **loader_kwargs
    )
    full_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RareAwareCompetitiveFusionSDM(
        len(species_ids),
        static_features=int(train_dataset.arrays["static"].shape[1]),
        rare_indices=torch.from_numpy(rare_indices),
        model_dim=args.model_dim,
    ).to(device)
    stage_one_checkpoint = args.output_dir / "rare_aware_geographic_best.pt"
    stage_one_history = train_stage(
        model,
        core_loader,
        device,
        epochs=args.stage_one_epochs,
        learning_rate=8e-4,
        deadline=deadline,
        reserve_seconds=3600,
        selection_loader=selection_loader,
        checkpoint_path=stage_one_checkpoint,
    )
    saved = torch.load(stage_one_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])

    tuning_targets, tuning_probabilities, tuning_log_cardinality = predict_with_aux(
        model, tuning_loader, device
    )
    core_ids = train_ids[core_indices]
    core_rows = rows.loc[core_ids].reset_index()
    core_pairs = pairs.loc[pairs["surveyId"].isin(core_ids)]
    tuning_coordinates = aligned_rows.iloc[tuning_indices][["lat", "lon"]].to_numpy()
    tuning_pa_candidates, tuning_pa_distance = pa_consensus_candidates(
        tuning_coordinates, core_rows, core_pairs
    )
    po_index = PresenceOnlyIndex(args.data_root / "GLC25_P0_metadata_train.csv")
    tuning_po_candidates = po_index.candidates(tuning_coordinates)
    tuning_ood_mask = tuning_pa_distance > 10.0
    policy, policy_trials = tune_postprocessing(
        tuning_targets,
        tuning_probabilities,
        tuning_log_cardinality,
        species_ids,
        tuning_po_candidates,
        tuning_pa_candidates,
        tuning_ood_mask,
    )
    neural_only_predictions = rank_blend_predictions(
        tuning_probabilities,
        species_ids,
        adaptive_cardinalities(tuning_log_cardinality),
    )
    neural_only_score = sample_f1_from_lists(
        label_sets(tuning_targets, species_ids), neural_only_predictions
    )

    full_history = train_stage(
        model,
        full_loader,
        device,
        epochs=args.full_finetune_epochs,
        learning_rate=2e-4,
        deadline=deadline,
        reserve_seconds=1800,
    )
    final_checkpoint = args.output_dir / "rare_aware_full.pt"
    torch.save({"model_state": model.state_dict(), "policy": policy}, final_checkpoint)
    test_targets, test_probabilities, test_log_cardinality = predict_with_aux(
        model, test_loader, device
    )
    if np.any(test_targets):
        raise ValueError("Official test placeholder labels must be all zero")

    test_metadata = pd.read_csv(args.data_root / "GLC25_PA_metadata_test.csv")
    test_rows = test_metadata.drop_duplicates("surveyId").set_index("surveyId").loc[test_ids]
    test_coordinates = test_rows[["lat", "lon"]].to_numpy()
    all_rows = rows.loc[train_ids].reset_index()
    test_pa_candidates, test_pa_distance = pa_consensus_candidates(
        test_coordinates, all_rows, pairs
    )
    test_po_candidates = po_index.candidates(test_coordinates)
    test_ood_mask = test_pa_distance > 10.0
    test_cardinalities = adaptive_cardinalities(
        test_log_cardinality, policy["cardinality_multiplier"]
    )
    predictions = rank_blend_predictions(
        test_probabilities,
        species_ids,
        test_cardinalities,
        po_candidates=test_po_candidates,
        pa_candidates=test_pa_candidates,
        ood_mask=test_ood_mask,
        po_weight_id=policy["po_weight_id"],
        po_weight_ood=policy["po_weight_ood"],
        pa_weight=policy["pa_weight"],
    )
    submission_path = args.output_dir / "GLC25_PA_submission.csv"
    submission_report = write_submission(
        submission_path, args.sample_submission, test_ids, predictions
    )
    report = {
        "protocol": "one rare-aware multimodal model with adaptive cardinality and OOD spatial priors",
        "seed": args.seed,
        "train_samples": int(len(train_dataset)),
        "test_samples": int(len(test_dataset)),
        "pa_species": int(len(species_ids)),
        "rare_pa_species": int(len(rare_indices)),
        "po_rows_seen": int(po_index.rows_seen),
        "geographic_holdout": {
            "countries": sorted(OOD_HOLDOUT_COUNTRIES),
            "core_samples": int(len(core_indices)),
            "selection_samples": int(len(selection_indices)),
            "tuning_samples": int(len(tuning_indices)),
            "selection_best_adaptive_f1": float(saved["score"]),
            "tuning_neural_only_adaptive_f1": neural_only_score,
            "tuning_optimized_f1": policy["sample_f1"],
        },
        "policy": policy,
        "policy_trials": policy_trials,
        "test_geography": {
            "within_10km_of_pa": int((~test_ood_mask).sum()),
            "farther_than_10km_from_pa": int(test_ood_mask.sum()),
        },
        "stage_one_history": stage_one_history,
        "full_finetune_history": full_history,
        "parameters": count_trainable_parameters(model),
        "test_labels_used": False,
        "submission": submission_report,
        "total_pipeline_hours": (
            time.monotonic() - started + float(manifest["preparation_seconds"])
        )
        / 3600,
        "registered_max_total_hours": args.max_hours,
        "official_score": None,
    }
    (args.output_dir / "sota_single_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if args.cleanup_cache:
        for name in ("full_train.npz", "official_test.npz"):
            path = args.data_dir / name
            if path.exists():
                path.unlink()
        shutil.rmtree(args.data_dir / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
