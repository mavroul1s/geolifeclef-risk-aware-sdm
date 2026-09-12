#!/usr/bin/env python
"""Evaluate a probability ensemble of full spatial fusion checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Standalone script execution puts ``scripts/`` on sys.path, not the repository
# root. Add the root explicitly so Kaggle can import shared training helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geolifeclef.models import CompetitiveFusionSDM
from scripts.train_spatial_competition import (
    MultimodalNPZDataset,
    apply_prediction_policy,
    collate,
    fit_prediction_policy,
    predict,
    sample_f1_from_predictions,
)


def mean_probabilities(
    checkpoints: list[Path],
    loader: DataLoader,
    device: torch.device,
    *,
    species: int,
    static_features: int,
    model_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    targets_reference: np.ndarray | None = None
    probability_sum: np.ndarray | None = None
    for checkpoint_path in checkpoints:
        model = CompetitiveFusionSDM(
            species, static_features=static_features, model_dim=model_dim
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        targets, probabilities = predict(model, loader, device, multimodal=True)
        if targets_reference is None:
            targets_reference = targets
            probability_sum = probabilities.astype(np.float32)
        else:
            if not np.array_equal(targets_reference, targets):
                raise ValueError("Checkpoint predictions were not evaluated on identical targets")
            probability_sum += probabilities
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if targets_reference is None or probability_sum is None:
        raise ValueError("At least one checkpoint is required")
    return targets_reference, probability_sum / len(checkpoints)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cleanup-cache", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    run_reports = [
        json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
        for run_dir in args.run_dirs
    ]
    expected_hashes = manifest["split_sha256"]
    if any(report["split_sha256"] != expected_hashes for report in run_reports):
        raise ValueError("A seed run used different split hashes")
    checkpoints = [run_dir / "competitive_fusion_best.pt" for run_dir in args.run_dirs]
    if any(not checkpoint.is_file() for checkpoint in checkpoints):
        raise FileNotFoundError("One or more fusion checkpoints are missing")

    calibration = MultimodalNPZDataset(args.data_dir / "calibration.npz")
    validation = MultimodalNPZDataset(args.data_dir / "validation.npz")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    calibration_loader = DataLoader(calibration, **loader_kwargs)
    validation_loader = DataLoader(validation, **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species = int(manifest["species"])
    static_width = int(calibration.arrays["static"].shape[1])
    calibration_targets, calibration_probabilities = mean_probabilities(
        checkpoints,
        calibration_loader,
        device,
        species=species,
        static_features=static_width,
        model_dim=args.model_dim,
    )
    policy = fit_prediction_policy(calibration_targets, calibration_probabilities)
    validation_targets, validation_probabilities = mean_probabilities(
        checkpoints,
        validation_loader,
        device,
        species=species,
        static_features=static_width,
        model_dim=args.model_dim,
    )
    predictions = apply_prediction_policy(validation_probabilities, policy)
    fusion_scores = np.asarray(
        [report["comparison"]["fusion"]["validation_sample_f1"] for report in run_reports],
        dtype=np.float64,
    )
    reference_scores = np.asarray(
        [report["comparison"]["reference"]["validation_sample_f1"] for report in run_reports],
        dtype=np.float64,
    )
    seeds = [int(report["seed"]) for report in run_reports]
    ensemble_score = sample_f1_from_predictions(validation_targets, predictions)
    report = {
        "seeds": seeds,
        "split_sha256": expected_hashes,
        "same_split_verified": True,
        "primary_metric": "sample-averaged F1",
        "fusion_seed_scores": fusion_scores.tolist(),
        "fusion_mean_sample_f1": float(fusion_scores.mean()),
        "fusion_std_sample_f1": float(fusion_scores.std(ddof=1)),
        "reference_seed_scores": reference_scores.tolist(),
        "reference_mean_sample_f1": float(reference_scores.mean()),
        "ensemble_prediction_policy": policy,
        "ensemble_validation_sample_f1": ensemble_score,
        "ensemble_mean_predicted_species": float(predictions.sum(1).mean()),
        "all_fusion_runs_beat_matched_reference": bool(np.all(fusion_scores > reference_scores)),
        "ensemble_beats_mean_reference": bool(ensemble_score > reference_scores.mean()),
        "official_2025_private_leaderboard_target": 0.2302,
        "official_target_directly_comparable": False,
        "next_gate": (
            "Retrain on all PA surveys and run official PA-test inference only if robustness gates pass."
        ),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.cleanup_cache:
        for name in ("train.npz", "calibration.npz", "validation.npz"):
            path = args.data_dir / name
            if path.exists():
                path.unlink()
        shutil.rmtree(args.data_dir / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
