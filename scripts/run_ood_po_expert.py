"""One v21 pipeline: new geographic control, PO expert, frozen deployment, assessment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import DataLoader

from geolifeclef.environmental_model import EnvironmentalChallenger
from geolifeclef.models import CompetitiveFusionSDM
from geolifeclef.utils import set_seed
from scripts.prepare_environmental_challenger import prepare, spatial_partitions
from scripts.prepare_po_expert import prepare_po, coordinate_features, expand_features
from scripts.train_po_expert import fit_expert, expert_predict
from scripts.run_environmental_challenger import MappedDataset, predict, train_one
from scripts.ood_po_protocol import (
    make_partitions, frozen_v20_blend, nearest_training_support, select_mixture_policy,
    mix_probabilities, assessment_metrics, paired_block_bootstrap, spatial_block_ids,
    submission_eligibility, sample_f1,
)
from scripts.stage_frozen_v20 import verify_bundle, submission_bytes, sha256_file


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized_environment(pa, test, po, fit_indices):
    """Identical PA-only fitting for PO and zero-PO; retain all raw columns."""
    fit = np.array(pa[fit_indices], dtype=np.float32)
    fit[~np.isfinite(fit)] = np.nan
    finite = np.isfinite(fit)
    count = finite.sum(axis=0)
    mean = np.nansum(fit, axis=0, dtype=np.float64) / np.maximum(count, 1)
    variance = np.nansum((fit - mean) ** 2, axis=0) / np.maximum(count, 1)
    std = np.where(variance < 1e-12, 1., np.sqrt(variance))
    def transform(values):
        values = np.asarray(values, dtype=np.float32)
        valid = np.isfinite(values)
        scaled = np.clip(np.nan_to_num((values - mean) / std, nan=0, posinf=0, neginf=0), -12, 12)
        return np.concatenate((scaled.astype(np.float32), (~valid).astype(np.float32)), axis=1)
    return (transform(pa), transform(test), transform(po),
            {"mean": mean.tolist(), "std": std.tolist(), "fit_rows": len(fit_indices),
             "fit_source": "matching PA training only, shared by PO and zero-PO",
             "all_raw_columns_retained": True, "missing_indicator_for_every_column": True})


def validate_submission(path, template_ids, species):
    frame = pd.read_csv(path, dtype={"predictions": str})
    if list(frame.columns) != ["surveyId", "predictions"]:
        raise ValueError("Invalid official CSV columns")
    if not np.array_equal(frame.surveyId.to_numpy(), template_ids) or frame.surveyId.duplicated().any():
        raise ValueError("Official CSV IDs differ from template order")
    vocabulary = set(map(int, species))
    for row in frame.predictions:
        values = [int(value) for value in row.split()]
        if len(values) != 20 or len(set(values)) != 20 or not set(values).issubset(vocabulary):
            raise ValueError("Official CSV has invalid top20 species")
    return {"rows": len(frame), "species_vocabulary": len(vocabulary),
            "prediction_policy": {"kind": "top_k", "k": 20},
            "submission_sha256": sha256_file(path), "template_order_verified": True,
            "prediction_count_min": 20, "prediction_count_max": 20}


def matched_control(data_dir, output, partitions, labels, device, deadline, workers, epochs):
    """Scientific refit for independent assessment; never recreates v20 outputs."""
    indices = [np.flatnonzero(partitions == i) for i in range(4)]
    loaders = {name: DataLoader(MappedDataset(data_dir, "train", indices[i]), batch_size=128,
                               num_workers=workers, pin_memory=device.type == "cuda")
               for i, name in ((1, "selection"), (2, "calibration"), (3, "assessment"))}
    env_dim = np.load(data_dir / "train_environment.npy", mmap_mode="r").shape[1]
    static_dim = np.load(data_dir / "train_static.npy", mmap_mode="r").shape[1]
    runs = []
    for i, (name, seed, challenger) in enumerate((
        ("reference_2025", 2025, False), ("challenger_2025", 2025, True),
        ("challenger_3407", 3407, True),
    )):
        set_seed(seed)
        # Preserve a minimum of three hours for both experts and final checks.
        seconds = min(3600., (deadline - time.monotonic() - 10800) / (3 - i))
        if seconds < 600:
            raise TimeoutError("Insufficient budget for the registered matched control")
        model = (EnvironmentalChallenger(labels.shape[1], env_dim, static_dim) if challenger
                 else CompetitiveFusionSDM(labels.shape[1], static_dim, model_dim=192))
        train = DataLoader(MappedDataset(data_dir, "train", indices[0], augment=True),
                           batch_size=128, shuffle=True, drop_last=True, num_workers=workers,
                           pin_memory=device.type == "cuda")
        run = train_one(model, train, loaders["selection"], np.array(labels[indices[1]]),
                        device, output, name, epochs, time.monotonic() + seconds, minimum_epochs=8)
        runs.append(run)
        for split in ("calibration", "assessment"):
            values = predict(model, loaders[split], device, deadline)
            np.save(output / f"{name}_{split}.npy", values, allow_pickle=False)
        del model, train, values
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for split in ("calibration", "assessment"):
        values = frozen_v20_blend(*(np.load(output / f"{name}_{split}.npy", mmap_mode="r")
                                   for name in ("reference_2025", "challenger_2025", "challenger_3407")))
        np.save(output / f"frozen_control_{split}.npy", values, allow_pickle=False)
    return runs


def run(args):
    started = time.monotonic()
    elapsed_setup = max(0., time.time() - float(os.environ.get("GLC_PIPELINE_STARTED_AT", time.time())))
    deadline = started + args.max_hours * 3600 - elapsed_setup
    if args.max_hours > 10.5:
        raise ValueError("v21 total runtime may not exceed 10.5 hours")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("The registered competition run requires a Kaggle T4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Registered accelerator is a T4")
    torch.set_num_threads(min(os.cpu_count() or 2, 4))
    output, data_dir = args.output_dir, args.data_dir
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data_root / "GLC25_PA_metadata_train.csv")
    rows = raw.drop_duplicates("surveyId").reset_index(drop=True)
    test_rows = pd.read_csv(args.data_root / "GLC25_PA_metadata_test.csv").drop_duplicates("surveyId").reset_index(drop=True)
    species = np.sort(raw.speciesId.dropna().unique().astype(np.int64))
    del raw
    if len(species) != 5016 or len(test_rows) != 14784:
        raise ValueError("Competition dimensions differ from the preregistration")
    original = spatial_partitions(rows)
    outer, split_manifest, outer_distance = make_partitions(rows, original)
    ids, test_ids = rows.surveyId.to_numpy(np.int64), test_rows.surveyId.to_numpy(np.int64)
    provenance = verify_bundle(args.frozen_dir, calibration_ids=ids[original == 2], test_ids=test_ids)
    if not np.array_equal(species, np.load(args.frozen_dir / "species_ids.npy")):
        raise ValueError("Frozen-v20 and v21 species ordering differ")
    save_json(output / "frozen_v20_provenance.json", provenance)
    save_json(output / "split_manifest.json", split_manifest)
    pd.DataFrame({"surveyId": ids, "v20_partition": original, "v21_partition": outer,
                  "distance_to_v21_training_km": outer_distance}).to_csv(output / "partition_ids.csv", index=False)
    print(json.dumps({"stage": "registered_geography", "counts": split_manifest["partition_counts"]}), flush=True)
    prep = prepare(args.data_root, data_dir, workers=6, deadline=deadline - 21600,
                   partition_override=outer)
    prep["protocol"] = "v21_matched_control_new_training_normalization"
    save_json(output / "control_data_manifest.json", prep)
    labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    po_dir = data_dir / "po"
    po_manifest = prepare_po(args.data_root, po_dir, rows, test_rows, species, deadline - 18000)
    save_json(output / "po_manifest.json", po_manifest)
    shutil.copyfile(po_dir / "po_support.npz", output / "po_support.npz")
    control_runs = matched_control(data_dir, output, outer, labels, device, deadline,
                                   args.workers, args.control_epochs)
    geo_pa = coordinate_features(rows[["lat", "lon"]].to_numpy())
    geo_test = coordinate_features(test_rows[["lat", "lon"]].to_numpy())
    geo_po = np.load(po_dir / "po_features.npy", mmap_mode="r")
    env_pa = np.load(po_dir / "pa_environment_raw.npy", mmap_mode="r")
    env_test = np.load(po_dir / "test_environment_raw.npy", mmap_mode="r")
    env_po = np.load(po_dir / "po_environment_raw.npy", mmap_mode="r")
    po_labels = sparse.load_npz(po_dir / "po_labels.npz")
    po_weights = np.load(po_dir / "po_weights.npy")
    policies, expert_runs = {}, []
    deployment_distance = nearest_training_support(rows.loc[original == 0], rows)["distance_km"]
    test_distance = nearest_training_support(rows.loc[original == 0], test_rows)["distance_km"]
    # The production predictor is separate and never informs outer choices.
    # It follows the frozen recipe on the original v20 training partition.
    for path_name, partitions, distances in (("evaluation", outer, outer_distance),
                                            ("deployment", original, deployment_distance)):
        ix = [np.flatnonzero(partitions == i) for i in range(4)]
        a, b, c, normalization = normalized_environment(env_pa, env_test, env_po, ix[0])
        save_json(output / f"{path_name}_expert_normalization.json", normalization)
        features, test_features, po_features = expand_features(geo_pa, a), expand_features(geo_test, b), expand_features(geo_po, c)
        del a, b, c
        base_path = (output / "frozen_control_calibration.npy" if path_name == "evaluation"
                     else args.frozen_dir / "v20_calibration_probabilities.npy")
        base = np.load(base_path, mmap_mode="r")
        policies[path_name] = {}
        for use_po in (True, False):
            values = None
            tag = "po" if use_po else "zero_po"
            name = f"{path_name}_{tag}"
            seconds = min(3600., deadline - time.monotonic() - 1800)
            if seconds < 600:
                raise TimeoutError("Insufficient budget for a required expert control")
            model, history = fit_expert(features, labels, ix[0], ix[1], po_features, po_labels,
                                        po_weights, device, output, name, time.monotonic() + seconds,
                                        use_po=use_po, epochs=args.expert_epochs,
                                        pretrain_epochs=args.po_epochs)
            expert_runs.append(history)
            calibration = expert_predict(model, features[ix[2]], device, deadline)
            np.save(output / f"{name}_calibration.npy", calibration, allow_pickle=False)
            selected, trials = select_mixture_policy(np.array(labels[ix[2]]), base, calibration, distances[ix[2]])
            policies[path_name][tag] = {"selected": selected, "trials": trials,
                                      "further_training": False, "checkpoint_sha256": sha256_file(output / f"{name}_best.pt")}
            # Persist every policy before assessment. No assessment labels here.
            save_json(output / "frozen_policies.json", policies)
            if path_name == "evaluation":
                values = expert_predict(model, features[ix[3]], device, deadline)
                np.save(output / f"{name}_assessment.npy", values, allow_pickle=False)
            elif use_po:
                values = expert_predict(model, test_features, device, deadline)
                np.save(output / "deployment_po_test.npy", values, allow_pickle=False)
            del model, calibration, values
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del features, test_features, po_features, base
        gc.collect()
    # Freeze actual deployment outputs before outer assessment reporting.
    deployment_policy = policies["deployment"]["po"]["selected"]
    base_test = np.load(args.frozen_dir / "v20_test_probabilities.npy", mmap_mode="r")
    expert_test = np.load(output / "deployment_po_test.npy", mmap_mode="r")
    test_probability = mix_probabilities(base_test, expert_test, test_distance, deployment_policy)
    template_ids = pd.read_csv(args.data_root / "GLC25_SAMPLE_SUBMISSION.csv").surveyId.to_numpy(np.int64)
    submission_path = output / "GLC25_PA_submission.csv"
    if deployment_policy["alpha"] == 0:
        shutil.copyfile(args.frozen_dir / "v20_original_submission.csv", submission_path)
    else:
        submission_path.write_bytes(submission_bytes(test_probability, species, test_ids, template_ids))
    submission = validate_submission(submission_path, template_ids, species)
    shutil.copyfile(args.frozen_dir / "v20_original_submission.csv", output / "unchanged_v20_submission.csv")
    freeze = {"policies_sha256": sha256_file(output / "frozen_policies.json"),
              "production_submission_sha256": submission["submission_sha256"],
              "assessment_reporting_started": False, "production_uses_original_v20": True}
    save_json(output / "pre_assessment_freeze.json", freeze)
    del test_probability, base_test, expert_test
    # Only now evaluate the outer predictor against its own never-fitted labels.
    assessment_ix = np.flatnonzero(outer == 3)
    targets = np.array(labels[assessment_ix])
    countries = rows.country.fillna("unknown").to_numpy(dtype=str)[assessment_ix]
    training_counts = np.asarray(labels[np.flatnonzero(outer == 0)]).sum(axis=0)
    base = np.load(output / "frozen_control_assessment.npy", mmap_mode="r")
    metrics, scores = {}, {}
    metrics["frozen_v20_recipe"], scores["frozen_v20_recipe"] = assessment_metrics(
        targets, base, countries, outer_distance[assessment_ix], training_counts)
    for tag in ("po", "zero_po"):
        expert = np.load(output / f"evaluation_{tag}_assessment.npy", mmap_mode="r")
        probability = mix_probabilities(base, expert, outer_distance[assessment_ix], policies["evaluation"][tag]["selected"])
        metrics[tag], scores[tag] = assessment_metrics(targets, probability, countries,
                                                       outer_distance[assessment_ix], training_counts)
        metrics[f"{tag}_standalone"], _ = assessment_metrics(targets, expert, countries,
                                                              outer_distance[assessment_ix], training_counts)
    blocks = spatial_block_ids(rows.iloc[assessment_ix])
    vs_base = paired_block_bootstrap(scores["po"], scores["frozen_v20_recipe"], blocks)
    vs_zero = paired_block_bootstrap(scores["po"], scores["zero_po"], blocks)
    integrity = {
        "all_5016_species": len(species) == 5016 and po_labels.shape[1] == 5016,
        "new_assessment_excludes_old_v20_heldouts": bool(np.all(original[assessment_ix] == 0)),
        "twenty_km_training_buffer": bool(np.all(outer_distance[outer > 0] >= 20 - 1e-7)),
        "original_v20_bundle_verified": True,
        "unchanged_v20_csv_exact": sha256_file(output / "unchanged_v20_submission.csv") == provenance["test"]["original_submission_sha256"],
        "production_policy_has_nonzero_po": bool(deployment_policy["alpha"] > 0),
        "policies_unchanged_since_freeze": sha256_file(output / "frozen_policies.json") == freeze["policies_sha256"],
        "production_csv_unchanged_since_freeze": sha256_file(submission_path) == freeze["production_submission_sha256"],
        "competition_only": True, "test_labels_unused": True,
        "registered_runtime": bool(time.monotonic() < deadline),
    }
    eligibility = submission_eligibility(policies["evaluation"]["po"]["selected"], vs_base, vs_zero, integrity)
    pd.DataFrame({"surveyId": ids[assessment_ix], "country": countries, "block": blocks,
                  "pa_distance_km": outer_distance[assessment_ix], **scores}).to_csv(output / "assessment_per_survey.csv", index=False)
    report = {
        "experiment": "v21_competition_only_ood_po_expert", "status": "complete",
        "protocol": "fresh buffered geographic assessment of matched frozen-v20 recipe",
        "assessment_scope": "recipe transfer; exact production v20 checkpoints are not independently reassessed",
        "split": split_manifest, "po": po_manifest, "control_runs": control_runs, "expert_runs": expert_runs,
        "selected_policies": {path: {tag: entry["selected"] for tag, entry in content.items()} for path, content in policies.items()},
        "assessment": {"metrics": metrics, "versus_frozen_v20": vs_base, "versus_zero_po": vs_zero, "used_for_selection": False},
        "integrity": integrity, "submission_gate": eligibility, "submission": submission,
        "frozen_v20_source_version": 20, "original_v20_retrained_to_recover_outputs": False,
        "external_data_or_weights": False, "test_labels_used": False, "post_calibration_finetuning": False,
        "official_public_score": None, "official_private_score": None,
        "prior_best_official_private_score": .19360, "competition_winner_private_target": .2302,
        "sota_proven": False, "registered_max_total_hours": args.max_hours,
        "total_pipeline_hours": (time.monotonic() - started + elapsed_setup) / 3600,
    }
    save_json(output / "v21_report.json", report)
    print(json.dumps({"stage": "complete", "assessment": report["assessment"], "submission_gate": eligibility,
                      "total_pipeline_hours": report["total_pipeline_hours"]}, indent=2), flush=True)
    # Prepared arrays can exceed Kaggle output quotas. Only remove our own
    # generated cache after all final reports, checkpoints and proofs exist.
    if data_dir.resolve().is_relative_to(Path.cwd().resolve() / "data" / "processed"):
        shutil.rmtree(data_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ood_po_expert_v21"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/ood_po_expert_v21"))
    parser.add_argument("--control-epochs", type=int, default=24)
    parser.add_argument("--expert-epochs", type=int, default=28)
    parser.add_argument("--po-epochs", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=10.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save_json(args.output_dir / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                  "message": str(error), "official_submission_allowed": False})
        raise


if __name__ == "__main__":
    main()
