#!/usr/bin/env python
"""Retrain the promoted fusion ensemble on all PA labels and predict official PA test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geolifeclef.losses import asymmetric_loss
from geolifeclef.models import CompetitiveFusionSDM, count_trainable_parameters
from geolifeclef.utils import set_seed
from scripts.train_spatial_competition import (
    MODALITIES,
    MultimodalNPZDataset,
    collate,
    predict,
)


def top_k_species_ids(
    probabilities: np.ndarray, species_ids: np.ndarray, k: int
) -> list[list[int]]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(species_ids):
        raise ValueError("Probability/species dimensions do not match")
    if not 1 <= k <= probabilities.shape[1]:
        raise ValueError("k is outside the species dimension")
    candidate_indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    rows: list[list[int]] = []
    for row_index, indices in enumerate(candidate_indices):
        ordered = indices[np.argsort(probabilities[row_index, indices])[::-1]]
        rows.append([int(species_ids[index]) for index in ordered])
    return rows


def write_submission(
    output_path: Path,
    template_path: Path,
    sample_ids: np.ndarray,
    probabilities: np.ndarray,
    species_ids: np.ndarray,
    *,
    k: int,
) -> dict[str, object]:
    template = pd.read_csv(template_path)
    if list(template.columns) != ["surveyId", "predictions"]:
        raise ValueError(f"Unexpected sample-submission columns: {list(template.columns)}")
    if len(sample_ids) != len(probabilities) or len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Test sample IDs must be unique and align with probability rows")
    template_ids = template["surveyId"].to_numpy(dtype=np.int64)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if set(template_ids.tolist()) != set(sample_ids.tolist()):
        raise ValueError("Prediction IDs do not match the official sample submission")
    prediction_lists = top_k_species_ids(probabilities, species_ids, k)
    prediction_by_id = {
        int(survey_id): " ".join(map(str, predictions))
        for survey_id, predictions in zip(sample_ids, prediction_lists)
    }
    submission = pd.DataFrame(
        {
            "surveyId": template_ids,
            "predictions": [prediction_by_id[int(survey_id)] for survey_id in template_ids],
        }
    )
    token_counts = submission["predictions"].str.split().str.len()
    if not bool((token_counts == k).all()):
        raise ValueError("Submission rows do not all contain the registered top-k predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "submission_path": str(output_path),
        "submission_sha256": digest,
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "unique_survey_ids": int(submission["surveyId"].nunique()),
        "prediction_policy": {"kind": "top_k", "k": k},
        "template_order_preserved": True,
    }


def train_full_model(
    model: CompetitiveFusionSDM,
    train_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    deadline: float,
) -> list[dict[str, float | int]]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = {name: batch[name].to(device, non_blocking=True) for name in MODALITIES}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = asymmetric_loss(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
        scheduler.step()
        seconds = time.monotonic() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
            "seconds": seconds,
            "examples_per_second": samples / max(seconds, 1e-6),
        }
        history.append(record)
        print(record)
        if time.monotonic() + seconds * 1.25 + 300 >= deadline and epoch < epochs:
            raise TimeoutError("Insufficient registered time to finish all full-data epochs")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 3407, 7919])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--cleanup-cache", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_labels_used") is not False:
        raise ValueError("Official test manifest must explicitly confirm no test labels were used")
    preparation_seconds = float(manifest["preparation_seconds"])
    remaining_seconds = args.max_hours * 3600 - preparation_seconds
    if remaining_seconds <= 0:
        raise TimeoutError("Preprocessing exhausted the registered Kaggle time budget")
    deadline = started + remaining_seconds
    train_dataset = MultimodalNPZDataset(args.data_dir / "full_train.npz", augment=True)
    test_dataset = MultimodalNPZDataset(args.data_dir / "official_test.npz")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species_ids = np.asarray(manifest["species_ids"], dtype=np.int64)
    static_width = int(train_dataset.arrays["static"].shape[1])
    probability_sum: np.ndarray | None = None
    seed_reports: list[dict[str, object]] = []
    test_ids_reference: np.ndarray | None = None
    for seed in args.seeds:
        set_seed(seed)
        model = CompetitiveFusionSDM(
            len(species_ids), static_features=static_width, model_dim=args.model_dim
        ).to(device)
        history = train_full_model(
            model, train_loader, device, epochs=args.epochs, deadline=deadline
        )
        checkpoint_path = args.output_dir / f"competitive_fusion_full_seed_{seed}.pt"
        torch.save(
            {"model_state": model.state_dict(), "seed": seed, "epochs": len(history)},
            checkpoint_path,
        )
        test_targets, probabilities = predict(model, test_loader, device, multimodal=True)
        if np.any(test_targets):
            raise ValueError("Official test placeholder labels must contain only zeros")
        test_ids = test_dataset.arrays["sample_id"].astype(np.int64)
        if test_ids_reference is None:
            test_ids_reference = test_ids
            probability_sum = probabilities.astype(np.float32)
        else:
            if not np.array_equal(test_ids_reference, test_ids):
                raise ValueError("Seed predictions were not produced for identical test rows")
            probability_sum += probabilities
        history_path = args.output_dir / f"full_seed_{seed}_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        seed_reports.append(
            {"seed": seed, "epochs": len(history), "checkpoint": str(checkpoint_path)}
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if probability_sum is None or test_ids_reference is None:
        raise ValueError("No full-data seed models were trained")
    ensemble_probabilities = probability_sum / len(args.seeds)
    submission_path = args.output_dir / "GLC25_PA_submission.csv"
    submission_report = write_submission(
        submission_path,
        args.sample_submission,
        test_ids_reference,
        ensemble_probabilities,
        species_ids,
        k=args.top_k,
    )
    report = {
        "protocol": "three-seed full-PA ensemble and official unlabeled test inference",
        "seeds": args.seeds,
        "seed_runs": seed_reports,
        "epochs_per_seed": args.epochs,
        "parameters_per_model": count_trainable_parameters(
            CompetitiveFusionSDM(
                len(species_ids), static_features=static_width, model_dim=args.model_dim
            )
        ),
        "train_samples": int(len(train_dataset)),
        "test_samples": int(len(test_dataset)),
        "species": int(len(species_ids)),
        "calibration_policy_source": "v17 spatial calibration ensemble",
        "test_labels_used": False,
        "submission": submission_report,
        "preparation_hours": preparation_seconds / 3600,
        "training_and_inference_hours": (time.monotonic() - started) / 3600,
        "total_pipeline_hours": (time.monotonic() - started + preparation_seconds) / 3600,
        "registered_max_total_hours": args.max_hours,
        "official_score": None,
    }
    (args.output_dir / "submission_manifest.json").write_text(
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
