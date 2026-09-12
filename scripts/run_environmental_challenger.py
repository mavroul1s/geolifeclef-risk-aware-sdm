#!/usr/bin/env python
"""One bounded run: matched reference, two challenger seeds, frozen calibration and audit."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from geolifeclef.environmental_model import EnvironmentalChallenger
from geolifeclef.losses import asymmetric_loss
from geolifeclef.models import CompetitiveFusionSDM, count_trainable_parameters
from geolifeclef.utils import set_seed
from scripts.prepare_environmental_challenger import FEATURES, prepare


class MappedDataset(Dataset):
    def __init__(self, root: Path, prefix: str, indices: np.ndarray | None = None, augment: bool = False):
        self.paths = {name: root / f"{prefix}_{name}.npy" for name in FEATURES}
        if prefix == "train":
            self.paths["labels"] = root / "train_labels.npy"
        self.arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in self.paths.items()}
        self.indices = np.arange(len(self.arrays["static"])) if indices is None else np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = {name: torch.from_numpy(np.array(values[self.indices[index]], dtype=np.float32)) for name, values in self.arrays.items()}
        if self.augment:
            if torch.rand(()) < 0.5:
                item["sentinel"] = item["sentinel"].flip(-1)
            if torch.rand(()) < 0.5:
                item["sentinel"] = item["sentinel"].flip(-2)
            if torch.rand(()) < 0.5:
                item["sentinel"] = item["sentinel"].transpose(-1, -2)
        return item


def top_rank(probabilities: np.ndarray, maximum: int = 50) -> tuple[np.ndarray, np.ndarray]:
    maximum = min(maximum, probabilities.shape[1])
    indices = np.argpartition(probabilities, -maximum, axis=1)[:, -maximum:]
    values = np.take_along_axis(probabilities, indices, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1), np.take_along_axis(values, order, axis=1)


def policy_counts(sorted_probabilities: np.ndarray, policy: dict) -> np.ndarray:
    if policy["kind"] == "top_k":
        return np.full(len(sorted_probabilities), min(policy["k"], sorted_probabilities.shape[1]), dtype=np.int64)
    return np.clip((sorted_probabilities >= policy["threshold"]).sum(axis=1), min(policy["minimum_k"], sorted_probabilities.shape[1]), sorted_probabilities.shape[1])


def ranked_f1(targets: np.ndarray, indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
    hits = np.take_along_axis(targets, indices, axis=1).cumsum(axis=1)
    return 2 * hits[np.arange(len(hits)), counts - 1] / np.maximum(targets.sum(axis=1) + counts, 1)


def policies() -> list[dict]:
    return ([{"kind": "top_k", "k": k} for k in (8, 12, 16, 18, 20, 24, 30)] +
            [{"kind": "threshold_min_k", "threshold": t, "minimum_k": k, "maximum_k": 50} for t in (0.08, 0.12, 0.16, 0.20, 0.25, 0.30) for k in (4, 12, 18)])


def select_policy(targets: np.ndarray, reference: np.ndarray, challenger: np.ndarray) -> tuple[dict, list[dict]]:
    trials = []
    # Weight zero and top-18 are explicit controls, not a forced nonzero new component.
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        indices, values = top_rank((1 - weight) * reference + weight * challenger)
        for policy in policies():
            score = float(ranked_f1(targets, indices, policy_counts(values, policy)).mean())
            trials.append({"challenger_weight": weight, "policy": policy, "calibration_f1": score})
    return max(trials, key=lambda trial: trial["calibration_f1"]), trials


@torch.no_grad()
def predict(model, loader, device, deadline=float("inf")) -> np.ndarray:
    model.eval()
    result = []
    for batch in loader:
        if time.monotonic() >= deadline:
            raise TimeoutError("Inference reached the registered deadline")
        inputs = {name: batch[name].to(device, non_blocking=True) for name in FEATURES}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
        probability = logits.float().sigmoid().cpu().numpy()
        if not np.isfinite(probability).all():
            raise FloatingPointError("Nonfinite prediction; refusing submission")
        result.append(probability.astype(np.float16))
    return np.concatenate(result)


def train_one(model, train_loader, selection_loader, selection_targets, device, output, name, epochs, deadline, minimum_epochs=8):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_epoch, history = -1.0, 0, []
    checkpoint = output / f"{name}_best.pt"
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        total_loss, samples, epoch_complete = 0.0, 0, True
        for batch in train_loader:
            if time.monotonic() > deadline - 120:
                epoch_complete = False
                break
            inputs = {key: batch[key].to(device, non_blocking=True) for key in FEATURES}
            target = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                if isinstance(model, EnvironmentalChallenger):
                    logits, auxiliary = model.forward_with_aux(inputs)
                else:
                    logits, auxiliary = model(inputs), ()
            # Float32 loss avoids log/underflow problems in mixed precision.
            loss = asymmetric_loss(logits.float(), target)
            for specialist_logits in auxiliary:
                loss = loss + 0.15 * asymmetric_loss(specialist_logits.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {name}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            samples += len(target)
        if not epoch_complete:
            break
        scheduler.step()
        probabilities = predict(model, selection_loader, device, deadline)
        indices, values = top_rank(probabilities)
        score = float(ranked_f1(selection_targets, indices, policy_counts(values, {"kind": "top_k", "k": 18})).mean())
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state": model.state_dict(), "selection_top18_f1": best, "epoch": epoch}, checkpoint)
        record = {"model": name, "epoch": epoch, "training_loss": total_loss / samples, "selection_top18_f1": score, "seconds": time.monotonic() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        (output / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch >= minimum_epochs and (epoch - best_epoch >= 8 or time.monotonic() + record["seconds"] * 1.2 + 120 > deadline):
            break
    if len(history) < minimum_epochs:
        raise TimeoutError(f"{name} completed only {len(history)} epochs, below registered minimum {minimum_epochs}")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    return {"name": name, "best_epoch": best_epoch, "epochs_completed": len(history), "selection_top18_f1": best, "parameters": count_trainable_parameters(model), "history": history}


def audit_metrics(targets, probabilities, policy, countries):
    indices, values = top_rank(probabilities)
    scores = ranked_f1(targets, indices, policy_counts(values, policy))
    by_country = {str(country): {"n": int((countries == country).sum()), "sample_f1": float(scores[countries == country].mean())} for country in np.unique(countries)}
    ood = np.isin(countries, ["Italy", "Switzerland"])
    return {"sample_f1": float(scores.mean()), "country_macro_f1": float(np.mean([v["sample_f1"] for v in by_country.values()])), "untouched_country_ood_f1": float(scores[ood].mean()) if ood.any() else None, "by_country": by_country}, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/environmental_challenger"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/environmental_challenger"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    # Launch time includes extraction/setup/tests when called by the Kaggle runner.
    started = time.monotonic()
    elapsed_setup = max(0.0, time.time() - float(os.environ.get("GLC_PIPELINE_STARTED_AT", time.time())))
    deadline = started + args.max_hours * 3600 - elapsed_setup
    if args.epochs < args.minimum_epochs:
        raise ValueError("Requested epochs below registered minimum")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("This registered Kaggle experiment requires an attached GPU")
    torch.set_num_threads(min(os.cpu_count() or 2, 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare(args.data_root, args.data_dir, workers=6, deadline=deadline - 3600)
    (args.output_dir / "data_manifest.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")
    partitions = np.load(args.data_dir / "partitions.npy")
    species = np.load(args.data_dir / "species_ids.npy")
    ids = np.load(args.data_dir / "train_ids.npy")
    country = np.load(args.data_dir / "train_countries.npy")
    labels = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    partition_indices = [np.flatnonzero(partitions == i) for i in range(4)]
    # Selection is the only label-bearing evaluation performed during training.
    selection_targets = np.array(labels[partition_indices[1]])
    calibration_targets = np.array(labels[partition_indices[2]])
    loaders = {name: DataLoader(MappedDataset(args.data_dir, "train", partition_indices[i]), batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda") for i, name in ((1, "selection"), (2, "calibration"), (3, "audit"))}
    loaders["test"] = DataLoader(MappedDataset(args.data_dir, "test"), batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    environment_dim = int(np.load(args.data_dir / "train_environment.npy", mmap_mode="r").shape[1])
    static_dim = int(np.load(args.data_dir / "train_static.npy", mmap_mode="r").shape[1])
    model_runs = []
    # Model count is fixed before looking at calibration or audit results.
    specifications = [("reference_2025", 2025, False), ("challenger_2025", 2025, True), ("challenger_3407", 3407, True)]
    for run_index, (name, seed, challenger) in enumerate(specifications):
        set_seed(seed)
        remaining = len(specifications) - run_index
        # Reserve 45 minutes for inference, final checks, and publication of artifacts.
        training_seconds = (deadline - time.monotonic() - 2700) / remaining
        if training_seconds < 600:
            raise TimeoutError("Insufficient budget to start another registered model")
        model = EnvironmentalChallenger(len(species), environment_dim, static_dim) if challenger else CompetitiveFusionSDM(len(species), static_dim, model_dim=192)
        train_loader = DataLoader(MappedDataset(args.data_dir, "train", partition_indices[0], augment=True), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=device.type == "cuda")
        run = train_one(model, train_loader, loaders["selection"], selection_targets, device, args.output_dir, name, args.epochs, time.monotonic() + training_seconds, args.minimum_epochs)
        run["seed"] = seed
        model_runs.append(run)
        # No training or normalization update may occur after these predictions.
        for split in ("calibration", "audit", "test"):
            probability = predict(model, loaders[split], device, deadline)
            np.save(args.output_dir / f"{name}_{split}_probabilities.npy", probability, allow_pickle=False)
        del model, train_loader, probability
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    def predictions(split):
        reference = np.load(args.output_dir / f"reference_2025_{split}_probabilities.npy").astype(np.float32)
        challenger = np.load(args.output_dir / f"challenger_2025_{split}_probabilities.npy").astype(np.float32)
        challenger += np.load(args.output_dir / f"challenger_3407_{split}_probabilities.npy")
        challenger *= 0.5
        return reference, challenger
    reference_cal, challenger_cal = predictions("calibration")
    selected, trials = select_policy(calibration_targets, reference_cal, challenger_cal)
    reference_policy = max((trial for trial in trials if trial["challenger_weight"] == 0), key=lambda trial: trial["calibration_f1"])["policy"]
    # Freeze and persist decisions BEFORE reading audit labels. Audit never chooses model/policy.
    (args.output_dir / "frozen_policy.json").write_text(json.dumps({"selected": selected, "reference_policy": reference_policy, "trials": trials, "further_training": False}, indent=2), encoding="utf-8")
    del reference_cal, challenger_cal
    weight = selected["challenger_weight"]
    reference_audit, challenger_audit = predictions("audit")
    audit_targets = np.array(labels[partition_indices[3]])
    audit_countries = country[partition_indices[3]]
    selected_audit, scores = audit_metrics(audit_targets, (1 - weight) * reference_audit + weight * challenger_audit, selected["policy"], audit_countries)
    reference_audit_metrics, reference_scores = audit_metrics(audit_targets, reference_audit, reference_policy, audit_countries)
    top18_reference, _ = audit_metrics(audit_targets, reference_audit, {"kind": "top_k", "k": 18}, audit_countries)
    # Equal-seed-count architectural comparison, not only 2-seed versus 1-seed.
    single_cal = np.load(args.output_dir / "challenger_2025_calibration_probabilities.npy").astype(np.float32)
    single_policy, _ = select_policy(calibration_targets, single_cal, single_cal)
    single_audit = np.load(args.output_dir / "challenger_2025_audit_probabilities.npy")
    single_metrics, _ = audit_metrics(audit_targets, single_audit, single_policy["policy"], audit_countries)
    pd.DataFrame({"surveyId": ids[partition_indices[3]], "country": audit_countries, "selected_sample_f1": scores, "reference_sample_f1": reference_scores}).to_csv(args.output_dir / "untouched_audit_per_survey.csv", index=False)
    del reference_audit, challenger_audit, single_cal, single_audit, audit_targets
    reference_test, challenger_test = predictions("test")
    probabilities = (1 - weight) * reference_test + weight * challenger_test
    indices, values = top_rank(probabilities)
    counts = policy_counts(values, selected["policy"])
    test_ids = np.load(args.data_dir / "test_ids.npy")
    lookup = {int(sid): index for index, sid in enumerate(test_ids)}
    template = pd.read_csv(args.data_root / "GLC25_SAMPLE_SUBMISSION.csv")
    submission_path = args.output_dir / "GLC25_PA_submission.csv"
    with submission_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["surveyId", "predictions"])
        for sid in template.surveyId:
            index = lookup[int(sid)]
            predictions_row = species[indices[index, :counts[index]]]
            writer.writerow([int(sid), " ".join(map(str, predictions_row))])
    report = {"protocol": "competition-only environmental challenger with frozen checkpoints and untouched spatial/country audit", "model_runs": model_runs, "environment_features": environment_dim, "training_samples": int(len(partition_indices[0])), "partition_counts": prep["partition_counts"], "normalization_fit_partition": "training only", "matched_reference": "v18 architecture, shared v20 preprocessing/splits; not an exact v18 reproduction", "selected": selected, "untouched_audit": {"selected": selected_audit, "reference_calibrated": reference_audit_metrics, "reference_top18": top18_reference, "challenger_single_seed_calibrated": single_metrics, "selected_minus_reference": selected_audit["sample_f1"] - reference_audit_metrics["sample_f1"], "used_for_selection": False}, "official_score": None, "sota_proven": False, "prior_best_official_private_score": 0.18900, "competition_winner_private_target": 0.2302, "test_labels_used": False, "external_data_or_weights": False, "post_calibration_finetuning": False, "submission": {"rows": len(template), "unique_survey_ids": len(lookup), "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(), "prediction_policy": selected["policy"], "prediction_count_min": int(counts.min()), "prediction_count_max": int(counts.max())}, "total_pipeline_hours": (time.monotonic() - started + elapsed_setup) / 3600, "registered_max_total_hours": args.max_hours}
    (args.output_dir / "challenger_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    # Remove only generated prepared arrays, after final artifact verification.
    for path in args.data_dir.glob("*.npy"):
        path.unlink()


if __name__ == "__main__":
    main()
