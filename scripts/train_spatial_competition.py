#!/usr/bin/env python
"""Fair spatial head-to-head: Landsat reference versus competitive multimodal SDM."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geolifeclef.losses import asymmetric_loss
from geolifeclef.metrics import top_k_predictions
from geolifeclef.models import CompetitiveFusionSDM, TemporalCNN, count_trainable_parameters
from geolifeclef.utils import set_seed


MODALITIES = ("landsat", "climate", "sentinel", "static")


class MultimodalNPZDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: Path, *, augment: bool = False):
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        self.augment = augment

    def __len__(self) -> int:
        return int(len(self.arrays["labels"]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            name: torch.from_numpy(self.arrays[name][index].astype(np.float32, copy=False))
            for name in (*MODALITIES, "labels")
        }
        if self.augment:
            if torch.rand(()) < 0.5:
                item["sentinel"] = torch.flip(item["sentinel"], dims=(-1,))
            if torch.rand(()) < 0.5:
                item["sentinel"] = torch.flip(item["sentinel"], dims=(-2,))
        return item


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.stack([item[name] for item in batch]) for name in batch[0]}


def sample_f1_from_predictions(targets: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(targets).astype(bool)
    predicted = np.asarray(predictions).astype(bool)
    true_positives = (truth & predicted).sum(axis=1)
    denominator = truth.sum(axis=1) + predicted.sum(axis=1)
    return float(np.mean(2 * true_positives / np.maximum(denominator, 1)))


def threshold_min_k_predictions(
    probabilities: np.ndarray, threshold: float, minimum_k: int, maximum_k: int = 50
) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    minimum_k = min(minimum_k, probabilities.shape[1])
    maximum_k = min(maximum_k, probabilities.shape[1])
    predicted = probabilities >= threshold
    for row_index, row in enumerate(probabilities):
        count = int(predicted[row_index].sum())
        if count < minimum_k:
            indices = np.argpartition(row, -minimum_k)[-minimum_k:]
            predicted[row_index, indices] = True
        elif count > maximum_k:
            indices = np.argpartition(row, -maximum_k)[-maximum_k:]
            predicted[row_index] = False
            predicted[row_index, indices] = True
    return predicted.astype(np.uint8)


def fit_prediction_policy(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for k in (1, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30):
        if k > probabilities.shape[1]:
            continue
        score = sample_f1_from_predictions(targets, top_k_predictions(probabilities, k))
        candidates.append({"kind": "top_k", "k": k, "calibration_sample_f1": score})
    for threshold in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30):
        for minimum_k in (1, 2, 4, 8, 12, 14, 16):
            predictions = threshold_min_k_predictions(probabilities, threshold, minimum_k)
            score = sample_f1_from_predictions(targets, predictions)
            candidates.append(
                {
                    "kind": "threshold_min_k",
                    "threshold": threshold,
                    "minimum_k": minimum_k,
                    "maximum_k": 50,
                    "calibration_sample_f1": score,
                }
            )
    return max(candidates, key=lambda candidate: float(candidate["calibration_sample_f1"]))


def apply_prediction_policy(probabilities: np.ndarray, policy: dict[str, object]) -> np.ndarray:
    if policy["kind"] == "top_k":
        return top_k_predictions(probabilities, int(policy["k"]))
    return threshold_min_k_predictions(
        probabilities,
        float(policy["threshold"]),
        int(policy["minimum_k"]),
        int(policy["maximum_k"]),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    multimodal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets, probabilities = [], []
    for batch in loader:
        labels = batch["labels"]
        if multimodal:
            inputs = {name: batch[name].to(device, non_blocking=True) for name in MODALITIES}
            logits = model(inputs)
        else:
            logits = model(batch["landsat"].to(device, non_blocking=True))
        targets.append(labels.numpy().astype(np.uint8))
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    calibration_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    *,
    multimodal: bool,
    epochs: int,
    learning_rate: float,
    deadline: float,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    checkpoint_path = output_dir / f"{name}_best.pt"
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                if multimodal:
                    inputs = {
                        modality: batch[modality].to(device, non_blocking=True)
                        for modality in MODALITIES
                    }
                    logits = model(inputs)
                else:
                    logits = model(batch["landsat"].to(device, non_blocking=True))
                loss = asymmetric_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
        scheduler.step()
        targets, probabilities = predict(
            model, calibration_loader, device, multimodal=multimodal
        )
        selection_score = sample_f1_from_predictions(
            targets, top_k_predictions(probabilities, 16)
        )
        seconds = time.monotonic() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
            "calibration_top16_sample_f1": selection_score,
            "seconds": seconds,
            "examples_per_second": samples / max(seconds, 1e-6),
        }
        history.append(record)
        print({"model": name, **record})
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {"model_state": model.state_dict(), "calibration_sample_f1": best_score},
                checkpoint_path,
            )
        estimated_next_epoch = seconds * 1.25 + 300
        if time.monotonic() + estimated_next_epoch >= deadline:
            print(f"Stopping {name} before the registered wall-clock deadline.")
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model, history


def evaluate_model(
    name: str,
    model: nn.Module,
    calibration_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    multimodal: bool,
) -> dict[str, object]:
    calibration_targets, calibration_probabilities = predict(
        model, calibration_loader, device, multimodal=multimodal
    )
    policy = fit_prediction_policy(calibration_targets, calibration_probabilities)
    validation_targets, validation_probabilities = predict(
        model, validation_loader, device, multimodal=multimodal
    )
    validation_predictions = apply_prediction_policy(validation_probabilities, policy)
    return {
        "model": name,
        "prediction_policy": policy,
        "validation_sample_f1": sample_f1_from_predictions(
            validation_targets, validation_predictions
        ),
        "validation_mean_predicted_species": float(validation_predictions.sum(1).mean()),
        "validation_mean_true_species": float(validation_targets.sum(1).mean()),
        "parameters": count_trainable_parameters(model),
    }


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reference-epochs", type=int, default=4)
    parser.add_argument("--fusion-epochs", type=int, default=6)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--cleanup-cache", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    preparation_seconds = float(manifest.get("preparation_seconds", 0.0))
    training_budget_hours = max(0.25, args.max_hours - preparation_seconds / 3600)
    deadline = started + training_budget_hours * 3600
    train_dataset = MultimodalNPZDataset(args.data_dir / "train.npz", augment=True)
    calibration_dataset = MultimodalNPZDataset(args.data_dir / "calibration.npz")
    validation_dataset = MultimodalNPZDataset(args.data_dir / "validation.npz")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species = int(manifest["species"])
    static_width = int(train_dataset.arrays["static"].shape[1])

    reference = TemporalCNN(6, species, channels=64, dropout=0.15)
    reference, reference_history = train_model(
        "landsat_reference",
        reference,
        train_loader,
        calibration_loader,
        device,
        args.output_dir,
        multimodal=False,
        epochs=args.reference_epochs,
        learning_rate=8e-4,
        deadline=deadline,
    )
    reference_result = evaluate_model(
        "landsat_reference",
        reference,
        calibration_loader,
        validation_loader,
        device,
        multimodal=False,
    )
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fusion = CompetitiveFusionSDM(
        species, static_features=static_width, model_dim=args.model_dim
    )
    fusion, fusion_history = train_model(
        "competitive_fusion",
        fusion,
        train_loader,
        calibration_loader,
        device,
        args.output_dir,
        multimodal=True,
        epochs=args.fusion_epochs,
        learning_rate=3e-4,
        deadline=deadline,
    )
    fusion_result = evaluate_model(
        "competitive_fusion",
        fusion,
        calibration_loader,
        validation_loader,
        device,
        multimodal=True,
    )
    write_history(args.output_dir / "landsat_reference_history.csv", reference_history)
    write_history(args.output_dir / "competitive_fusion_history.csv", fusion_history)
    reference_score = float(reference_result["validation_sample_f1"])
    fusion_score = float(fusion_result["validation_sample_f1"])
    report = {
        "split_sha256": manifest["split_sha256"],
        "same_split_verified": True,
        "holdout_country": manifest["holdout_country"],
        "primary_metric": "sample-averaged F1",
        "official_2025_private_leaderboard_target": 0.2302,
        "official_target_directly_comparable": False,
        "comparison": {"reference": reference_result, "fusion": fusion_result},
        "fusion_absolute_gain": fusion_score - reference_score,
        "fusion_wins_internal_holdout": fusion_score > reference_score,
        "preparation_hours": preparation_seconds / 3600,
        "training_hours": (time.monotonic() - started) / 3600,
        "total_pipeline_hours": (time.monotonic() - started + preparation_seconds) / 3600,
        "registered_max_total_hours": args.max_hours,
        "device": str(device),
        "warning": (
            "Internal Netherlands holdout estimates OOD performance; only the official hidden test "
            "can establish a leaderboard/SOTA win."
        ),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if args.cleanup_cache:
        for name in ("train.npz", "calibration.npz", "validation.npz"):
            path = args.data_dir / name
            if path.exists():
                path.unlink()
        shutil.rmtree(args.data_dir / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
