"""Run the preregistered, compute-bounded v23 cross-fit experiment."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from scripts.ood_po_protocol import (
    assessment_metrics,
    nearest_training_support,
    paired_block_bootstrap,
    sample_f1,
    spatial_block_ids,
)
from scripts.prepare_environmental_challenger import prepare, spatial_partitions
from scripts.prepare_po_expert import coordinate_features, expand_features, prepare_po
from scripts.run_ood_po_expert import matched_control, normalized_environment, save_json
from scripts.run_retained_po import fit_v21_control
from scripts.stage_frozen_v20 import sha256_file, verify_bundle
from scripts.stage_frozen_v21 import verify_v21
from scripts.stage_frozen_v22 import OFFICIAL_CSV_SHA256, reconstruct, verify_v22
from scripts.train_diverse_po_v23 import fit_adaptation, predict, pretrain
from scripts.train_retained_po import fit_arm as fit_v22_arm
from scripts.train_retained_po import predict as predict_v22
from scripts.train_retained_po import pretrain as pretrain_v22
from scripts.v22_protocol import select_policy as select_v22_policy
from scripts.v23_protocol import (
    LARGE_SEED,
    V23_SEEDS,
    crossfit_partitions,
    mix,
    ood_components,
    per_survey_f1,
    po_context,
    policy_counts,
    select_policy,
    submission_gate,
)
from scripts.run_environmental_challenger import top_rank


FAMILIES = ("single_head_ensemble", "retained_po", "zero_po", "large_single_head")
OUTPUT_CSV = "GLC25_PA_submission_v23.csv"
EXPECTED_KERNEL_VERSION = 23
MAX_HOURS = 10.5


def prepare_labels(raw, rows, species, path):
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                       shape=(len(rows), len(species)))
    matrix[:] = 0
    pairs = raw[["surveyId", "speciesId"]].dropna().drop_duplicates()
    row_indices = pd.Index(rows.surveyId).get_indexer(pairs.surveyId)
    columns = pd.Index(species).get_indexer(pairs.speciesId)
    if (row_indices < 0).any() or (columns < 0).any():
        raise ValueError("PA label alignment failed")
    matrix[row_indices, columns] = 1
    matrix.flush()
    return matrix


def load_po(directory: Path):
    support = np.load(directory / "po_support.npz", allow_pickle=False)
    return {
        "pa": np.load(directory / "pa_environment_raw.npy", mmap_mode="r"),
        "test": np.load(directory / "test_environment_raw.npy", mmap_mode="r"),
        "po": np.load(directory / "po_environment_raw.npy", mmap_mode="r"),
        "geo": np.load(directory / "po_features.npy", mmap_mode="r"),
        "labels": sparse.load_npz(directory / "po_labels.npz"),
        "weights": np.load(directory / "po_weights.npy", allow_pickle=False),
        "support_coordinates": np.asarray(support["coordinates"], dtype=np.float64),
        "support_species": np.asarray(support["unique_species"], dtype=np.float64),
    }


def expert_features(pool, fit_indices, rows, test_rows, output, name):
    pa, test, po, normalization = normalized_environment(
        pool["pa"], pool["test"], pool["po"], fit_indices,
    )
    save_json(output / f"{name}_normalization.json", normalization)
    pa = expand_features(coordinate_features(rows[["lat", "lon"]].to_numpy()), pa)
    test = expand_features(coordinate_features(test_rows[["lat", "lon"]].to_numpy()), test)
    po = expand_features(pool["geo"], po)
    return pa, test, po


def fit_matched_v22(rows, test_rows, labels, split, distance, old_pool, new_pool,
                    data_dir, output, device, deadline):
    output.mkdir(parents=True, exist_ok=True)
    control = fit_v21_control(rows, test_rows, labels, split, distance, old_pool,
                              data_dir, output, device, deadline)
    indices = [np.flatnonzero(split == index) for index in range(4)]
    pa, _, po = expert_features(new_pool, indices[0], rows, test_rows, output, "matched_v22")
    pretrained, pretraining = pretrain_v22(
        po, new_pool["labels"], new_pool["weights"], device, output,
        min(deadline - 2400, time.monotonic() + 3600),
    )
    model, adaptation = fit_v22_arm(
        pretrained, "retained_po", pa, labels, indices[0], indices[1],
        np.load(output / "frozen_v21_selection.npy", mmap_mode="r"), distance,
        po, new_pool["labels"], new_pool["weights"], device, output,
        min(deadline - 1800, time.monotonic() + 3600),
    )
    predictions = {}
    for index, role in ((1, "selection"), (2, "calibration"), (3, "assessment")):
        expert = predict_v22(model, pa[indices[index]], device, deadline)
        base = np.load(output / f"frozen_v21_{role}.npy", mmap_mode="r")
        if role == "calibration":
            policy, trials = select_v22_policy(np.asarray(labels[indices[index]]), base, expert,
                                               distance[indices[index]])
        predictions[role] = (base, expert)
    policy_record = {"selected": policy, "trials": trials,
                     "checkpoint_sha256": sha256_file(output / "retained_po_best.pt")}
    role_index = {"selection": 1, "calibration": 2, "assessment": 3}
    for role, (base, expert) in predictions.items():
        from scripts.v22_protocol import mix as v22_mix
        np.save(output / f"frozen_v22_{role}.npy",
                v22_mix(base, expert, distance[indices[role_index[role]]], policy),
                allow_pickle=False)
    del model, pretrained, pa, po
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"matched_v21": control, "v22_pretraining": pretraining,
            "v22_adaptation": adaptation, "v22_policy": policy_record}


def _ensemble(arrays):
    if not arrays:
        raise ValueError("Empty v23 ensemble")
    mean = np.zeros(np.asarray(arrays[0]).shape, dtype=np.float32)
    for values in arrays:
        mean += np.asarray(values, dtype=np.float32) / len(arrays)
    disagreement = np.zeros(len(mean), dtype=np.float64)
    for values in arrays:
        delta = np.asarray(values, dtype=np.float32) - mean
        disagreement += np.mean(delta * delta, axis=1) / len(arrays)
    return mean.astype(np.float16), np.sqrt(disagreement)


def _role_features(pa, test, indices, deployment):
    return {
        "selection": pa[indices[1]],
        "calibration": pa[indices[2]],
        "target": test if deployment else pa[indices[3]],
    }


def fit_v23_models(rows, test_rows, labels, split, distance, pool, base_selection,
                   base_calibration, output, device, deadline, *, deployment=False):
    indices = [np.flatnonzero(split == index) for index in range(4)]
    pa, test, po = expert_features(pool, indices[0], rows, test_rows, output, "v23")
    features = _role_features(pa, test, indices, deployment)
    query_rows = {
        "selection": rows.iloc[indices[1]],
        "calibration": rows.iloc[indices[2]],
        "target": test_rows if deployment else rows.iloc[indices[3]],
    }
    po_contexts = {
        role: po_context(pool["support_coordinates"], pool["support_species"],
                         frame[["lat", "lon"]].to_numpy())
        for role, frame in query_rows.items()
    }
    predictions = {family: {role: [] for role in features} for family in FAMILIES}
    histories = {family: [] for family in FAMILIES}
    pretraining_histories = []
    for seed in V23_SEEDS:
        pretrained, history = pretrain(
            po, pool["labels"], pool["weights"], device, output, deadline,
            seed=seed, width=512, blocks=3,
        )
        pretraining_histories.append(history)
        for arm, family in (("single_head", "single_head_ensemble"),
                            ("retained_po", "retained_po")):
            model, training = fit_adaptation(
                pretrained, arm, pa, labels, indices[0], indices[1], base_selection,
                predictions[family]["selection"], po, pool["labels"], pool["weights"],
                device, output, deadline, seed=seed, width=512, blocks=3,
            )
            histories[family].append(training)
            for role, values in features.items():
                predictions[family][role].append(predict(model, values, device, deadline))
            del model
        model, training = fit_adaptation(
            None, "zero_po", pa, labels, indices[0], indices[1], base_selection,
            predictions["zero_po"]["selection"], po, pool["labels"], pool["weights"],
            device, output, deadline, seed=seed, width=512, blocks=3,
        )
        histories["zero_po"].append(training)
        for role, values in features.items():
            predictions["zero_po"][role].append(predict(model, values, device, deadline))
        del model, pretrained
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    large_pretrained, large_pretraining = pretrain(
        po, pool["labels"], pool["weights"], device, output, deadline,
        seed=LARGE_SEED, width=768, blocks=4,
    )
    large_model, large_history = fit_adaptation(
        large_pretrained, "single_head", pa, labels, indices[0], indices[1], base_selection,
        [], po, pool["labels"], pool["weights"], device, output, deadline,
        seed=LARGE_SEED, width=768, blocks=4,
    )
    histories["large_single_head"].append(large_history)
    for role, values in features.items():
        predictions["large_single_head"][role].append(predict(large_model, values, device, deadline))
    pretraining_histories.append(large_pretraining)
    del large_model, large_pretrained
    ensembles, disagreements, components = {}, {}, {}
    for family in FAMILIES:
        ensembles[family], disagreements[family], components[family] = {}, {}, {}
        for role in features:
            ensemble, disagreement = _ensemble(predictions[family][role])
            ensembles[family][role] = ensemble
            disagreements[family][role] = disagreement
            context = po_contexts[role]
            role_indices = indices[1] if role == "selection" else indices[2]
            pa_distance = (distance[role_indices] if role != "target" or not deployment
                           else nearest_training_support(rows.iloc[indices[0]], test_rows)["distance_km"])
            if role == "target" and not deployment:
                pa_distance = distance[indices[3]]
            components[family][role] = ood_components(
                pa_distance, context["distance_km"], context["local_species_support"], disagreement,
            )
            np.save(output / f"{family}_{role}.npy", ensemble, allow_pickle=False)
            np.save(output / f"{family}_{role}_disagreement.npy", disagreement, allow_pickle=False)
        predictions[family]["seed_count"] = len(predictions[family]["selection"])
    if not deployment:
        for seed, values in zip(V23_SEEDS, predictions["single_head_ensemble"]["target"]):
            np.save(output / f"single_head_seed_{seed}_target.npy", values, allow_pickle=False)
    policies = {}
    calibration_targets = np.asarray(labels[indices[2]])
    for family in FAMILIES:
        selected, trials = select_policy(calibration_targets, base_calibration,
                                         ensembles[family]["calibration"],
                                         components[family]["calibration"])
        policies[family] = {"selected": selected, "trials": trials,
                            "seed_count": predictions[family]["seed_count"]}
    training = {"po_pretraining": pretraining_histories, "families": histories,
                "architecture_comparison_is_diagnostic": True,
                "assessment_used_for_selection": False}
    del pa, test, po
    gc.collect()
    return policies, training, components


def deployment_partitions(rows):
    original = spatial_partitions(rows)
    v22 = __import__("scripts.run_retained_po", fromlist=["deployment_partitions"]).deployment_partitions(rows)
    eligible = np.flatnonzero(v22 == 2)
    blocks = spatial_block_ids(rows.iloc[eligible])
    unique, counts = np.unique(blocks, return_counts=True)
    order = sorted(range(len(unique)), key=lambda index: (-counts[index],
        hashlib.sha256(f"20251223:{unique[index]}".encode()).digest()))
    selection_blocks, totals = set(), [0, 0]
    for index in order:
        side = 0 if totals[0] <= totals[1] else 1
        if side == 0:
            selection_blocks.add(unique[index])
        totals[side] += int(counts[index])
    split = np.full(len(rows), -1, dtype=np.int8)
    split[original == 0] = 0
    selected = np.isin(blocks, list(selection_blocks))
    split[eligible[selected]] = 1
    split[eligible[~selected]] = 2
    if min((split == 1).sum(), (split == 2).sum()) < 100:
        raise ValueError("Insufficient fixed v23 deployment development split")
    return split


def _validate_submission(path, template_ids, species):
    frame = pd.read_csv(path, dtype={"predictions": str})
    if list(frame.columns) != ["surveyId", "predictions"]:
        raise ValueError("Invalid v23 CSV columns")
    if frame.surveyId.duplicated().any() or not np.array_equal(frame.surveyId.to_numpy(), template_ids):
        raise ValueError("v23 CSV IDs differ from template order")
    vocabulary = set(map(int, species))
    counts = []
    for row in frame.predictions:
        values = [int(value) for value in row.split()]
        if len(values) != len(set(values)) or not set(values).issubset(vocabulary):
            raise ValueError("Invalid v23 prediction row")
        counts.append(len(values))
    if min(counts) < 16 or max(counts) > 28:
        raise ValueError("v23 cardinality outside the registered grid")
    return {"rows": len(frame), "species_vocabulary": len(vocabulary),
            "prediction_count_min": min(counts), "prediction_count_max": max(counts),
            "prediction_count_mean": float(np.mean(counts)), "template_order_verified": True,
            "submission_sha256": sha256_file(path)}


def submission_bytes(probabilities, species, test_ids, template_ids, policy, components):
    indices, _ = top_rank(probabilities, maximum=28)
    counts = policy_counts(policy, components)
    lookup = {int(value): index for index, value in enumerate(test_ids)}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["surveyId", "predictions"])
    for survey_id in template_ids:
        row = lookup[int(survey_id)]
        selected = species[indices[row, :counts[row]]]
        writer.writerow([int(survey_id), " ".join(map(str, selected))])
    return stream.getvalue().encode("utf-8")


def _metrics(targets, probabilities, policy, components, countries, distances, training_counts,
             *, baseline=None, species_ids=None):
    scores = per_survey_f1(targets, probabilities, policy, components)
    counts = policy_counts(policy, components)
    report = {
        "sample_f1": float(scores.mean()), "surveys": len(scores), "species": targets.shape[1],
        "cardinality": {"prediction_min": int(counts.min()), "prediction_max": int(counts.max()),
                        "prediction_mean": float(counts.mean()),
                        "target_mean": float(np.asarray(targets).sum(axis=1).mean())},
        "by_country": {}, "by_pa_distance": {}, "species_recall": {}, "used_for_selection": False,
        "ood_component_means": {name: float(np.mean(values)) for name, values in components.items()},
    }
    for country in np.unique(countries):
        mask = countries == country
        report["by_country"][str(country)] = {"n": int(mask.sum()), "sample_f1": float(scores[mask].mean())}
    for name, low, high in (("20_to_50km", 20, 50), ("50_to_100km", 50, 100),
                            ("100_to_200km", 100, 200), ("200km_plus", 200, np.inf)):
        mask = (distances >= low) & (distances < high)
        report["by_pa_distance"][name] = {"n": int(mask.sum()),
            "sample_f1": float(scores[mask].mean()) if mask.any() else None}
    max_indices, _ = top_rank(probabilities, maximum=28)
    frequency = np.asarray(training_counts)
    for name, species_mask in (("zero_training", frequency == 0),
                               ("rare_1_to_25", (frequency >= 1) & (frequency <= 25)),
                               ("common_over_25", frequency > 25)):
        positives = int(np.asarray(targets)[:, species_mask].sum())
        hits = 0
        for row, count in enumerate(counts):
            chosen = max_indices[row, :count]
            hits += int(np.asarray(targets)[row, chosen][species_mask[chosen]].sum())
        report["species_recall"][name] = {"species": int(species_mask.sum()),
            "target_positives": positives, "true_positives": hits,
            "micro_recall": hits / positives if positives else None}
    if baseline is not None and species_ids is not None:
        base_indices, _ = top_rank(baseline, maximum=20)
        added, removed, added_hits, removed_hits = 0, 0, 0, 0
        species_delta = np.zeros(targets.shape[1], dtype=np.int64)
        prediction_delta = np.zeros(targets.shape[1], dtype=np.int64)
        for row, count in enumerate(counts):
            old = set(map(int, base_indices[row, :20]))
            new = set(map(int, max_indices[row, :count]))
            add, drop = new - old, old - new
            added += len(add)
            removed += len(drop)
            for index in add:
                prediction_delta[index] += 1
                if targets[row, index]:
                    added_hits += 1
                    species_delta[index] += 1
            for index in drop:
                prediction_delta[index] -= 1
                if targets[row, index]:
                    removed_hits += 1
                    species_delta[index] -= 1
        order_gain = np.argsort(-species_delta, kind="stable")[:20]
        order_loss = np.argsort(species_delta, kind="stable")[:20]
        report["changes_vs_frozen_v22_top20"] = {
            "added_predictions": added, "removed_predictions": removed,
            "added_true_presences": added_hits, "removed_true_presences": removed_hits,
            "net_true_presence_change": added_hits - removed_hits,
            "top_species_true_presence_gains": [
                {"species_id": int(species_ids[index]), "net_true_presence_change": int(species_delta[index]),
                 "net_prediction_count_change": int(prediction_delta[index])} for index in order_gain
                if species_delta[index] > 0],
            "top_species_true_presence_losses": [
                {"species_id": int(species_ids[index]), "net_true_presence_change": int(species_delta[index]),
                 "net_prediction_count_change": int(prediction_delta[index])} for index in order_loss
                if species_delta[index] < 0],
        }
    return report, scores


def run(args):
    started = float(os.environ.get("GLC_PIPELINE_STARTED_AT", time.time()))
    deadline = time.monotonic() + MAX_HOURS * 3600 - (time.time() - started)
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("At least one Kaggle T4 is required; v23 uses cuda:0 only")
    device = torch.device("cuda:0")
    torch.set_num_threads(min(os.cpu_count() or 2, 4))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cache = Path("data/processed/diverse_po_v23")
    cache.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data_root / "GLC25_PA_metadata_train.csv")
    rows = raw.drop_duplicates("surveyId").reset_index(drop=True)
    test_rows = pd.read_csv(args.data_root / "GLC25_PA_metadata_test.csv").drop_duplicates("surveyId").reset_index(drop=True)
    template = pd.read_csv(args.data_root / "GLC25_SAMPLE_SUBMISSION.csv").surveyId.to_numpy()
    species = np.sort(raw.speciesId.dropna().unique().astype(np.int64))
    if len(species) != 5016 or len(test_rows) != 14784 or len(np.unique(template)) != 14784:
        raise ValueError("GeoLifeCLEF dimensions differ from the registered v23 contract")
    labels = prepare_labels(raw, rows, species, cache / "pa_labels.npy")
    del raw
    folds = crossfit_partitions(rows)
    save_json(output / "split_manifest.json", {"folds": [item[2] for item in folds],
              "checkpoint_selection_calibration_assessment_separate": True,
              "v20_v21_v22_assessments_not_reused_as_v23_assessment": True})
    pd.DataFrame({"surveyId": rows.surveyId, "fold_0": folds[0][0],
                  "fold_1": folds[1][0]}).to_csv(output / "partition_ids.csv", index=False)
    original = spatial_partitions(rows)
    verify_bundle(args.frozen_v20_dir, test_ids=test_rows.surveyId.to_numpy())
    verify_v21(args.frozen_v21_dir)
    v22_proof = verify_v22(args.frozen_v22_dir, v20=args.frozen_v20_dir,
        v21=args.frozen_v21_dir, train_metadata=args.data_root / "GLC25_PA_metadata_train.csv",
        test_metadata=args.data_root / "GLC25_PA_metadata_test.csv",
        template=args.data_root / "GLC25_SAMPLE_SUBMISSION.csv")
    if not np.array_equal(species, np.load(args.frozen_v20_dir / "species_ids.npy")):
        raise ValueError("Frozen vocabulary differs from competition PA vocabulary")
    manifests = {mode: prepare_po(args.data_root, cache / mode, rows, test_rows, species,
                                   deadline - 7 * 3600, mode=mode) for mode in ("v21", "v22")}
    save_json(output / "po_manifests.json", manifests)
    old_pool, new_pool = load_po(cache / "v21"), load_po(cache / "v22")
    policies, training, component_paths = {}, {}, {}
    for fold, (split, support, manifest) in enumerate(folds):
        name = f"fold_{fold}"
        directory = output / name
        directory.mkdir()
        data_dir = cache / f"{name}_modalities"
        prepare(args.data_root, data_dir, workers=6, deadline=deadline - 4 * 3600,
                partition_override=split)
        control_dir = directory / "matched_v22"
        training[name] = {"control": fit_matched_v22(rows, test_rows, labels, split,
            support["distance_km"], old_pool, new_pool, data_dir, control_dir, device, deadline)}
        policies[name], training[name]["v23"], components = fit_v23_models(
            rows, test_rows, labels, split, support["distance_km"], new_pool,
            np.load(control_dir / "frozen_v22_selection.npy", mmap_mode="r"),
            np.load(control_dir / "frozen_v22_calibration.npy", mmap_mode="r"),
            directory, device, deadline,
        )
        component_paths[name] = components
        save_json(output / "frozen_policies.json", policies)
        if not data_dir.resolve().is_relative_to(cache.resolve()):
            raise ValueError("Unsafe v23 cache cleanup")
        shutil.rmtree(data_dir)
    production = deployment_partitions(rows)
    _, frozen_v22_calibration, frozen_v22_test = reconstruct(
        args.frozen_v20_dir, args.frozen_v21_dir, args.frozen_v22_dir, rows, test_rows,
    )
    v22_production = __import__("scripts.run_retained_po", fromlist=["deployment_partitions"]).deployment_partitions(rows)
    eligible = np.flatnonzero(v22_production == 2)
    position = {row: index for index, row in enumerate(eligible)}
    selection_positions = [position[row] for row in np.flatnonzero(production == 1)]
    calibration_positions = [position[row] for row in np.flatnonzero(production == 2)]
    production_distance = nearest_training_support(rows.loc[production == 0], rows)["distance_km"]
    directory = output / "deployment"
    directory.mkdir()
    policies["deployment"], training["deployment"], components = fit_v23_models(
        rows, test_rows, labels, production, production_distance, new_pool,
        frozen_v22_calibration[selection_positions], frozen_v22_calibration[calibration_positions],
        directory, device, deadline, deployment=True,
    )
    component_paths["deployment"] = components
    save_json(output / "frozen_policies.json", policies)
    shutil.copyfile(args.frozen_v22_dir / "GLC25_PA_submission.csv", output / "unchanged_v22_submission.csv")
    primary_policy = policies["deployment"]["single_head_ensemble"]["selected"]
    primary_test = np.load(directory / "single_head_ensemble_target.npy", mmap_mode="r")
    selected_test = mix(frozen_v22_test, primary_test,
                        components["single_head_ensemble"]["target"], primary_policy)
    destination = output / OUTPUT_CSV
    if primary_policy["alpha"] == 0 and primary_policy["k_near"] == primary_policy["k_far"] == 20:
        shutil.copyfile(output / "unchanged_v22_submission.csv", destination)
    else:
        destination.write_bytes(submission_bytes(selected_test, species, test_rows.surveyId.to_numpy(),
                                                 template, primary_policy,
                                                 components["single_head_ensemble"]["target"]))
    submission = _validate_submission(destination, template, species)
    checkpoint_hashes = {path.relative_to(output).as_posix(): sha256_file(path)
                         for path in sorted(output.rglob("*.pt"))}
    freeze = {"policies_sha256": sha256_file(output / "frozen_policies.json"),
              "submission_sha256": sha256_file(destination),
              "unchanged_v22_sha256": sha256_file(output / "unchanged_v22_submission.csv"),
              "assessment_reporting_started": False,
              "all_models_and_policies_frozen": True,
              "checkpoint_sha256": checkpoint_hashes}
    save_json(output / "pre_assessment_freeze.json", freeze)
    frames, fold_metrics, fold_scores = [], {}, []
    for fold, (split, support, manifest) in enumerate(folds):
        name = f"fold_{fold}"
        directory = output / name
        indices = np.flatnonzero(split == 3)
        targets = np.asarray(labels[indices])
        countries = rows.country.fillna("unknown").to_numpy(dtype=str)[indices]
        counts = np.asarray(labels[np.flatnonzero(split == 0)]).sum(axis=0)
        base = np.load(directory / "matched_v22" / "frozen_v22_assessment.npy", mmap_mode="r")
        frozen_v20 = np.load(directory / "matched_v22" / "frozen_control_assessment.npy", mmap_mode="r")
        frozen_v21 = np.load(directory / "matched_v22" / "frozen_v21_assessment.npy", mmap_mode="r")
        scores = {"frozen_v20": sample_f1(targets, frozen_v20),
                  "frozen_v21": sample_f1(targets, frozen_v21),
                  "frozen_v22": sample_f1(targets, base)}
        metrics = {
            "frozen_v20": assessment_metrics(targets, frozen_v20, countries,
                support["distance_km"][indices], counts)[0],
            "frozen_v21": assessment_metrics(targets, frozen_v21, countries,
                support["distance_km"][indices], counts)[0],
            "frozen_v22": assessment_metrics(targets, base, countries,
                support["distance_km"][indices], counts)[0],
        }
        for family in FAMILIES:
            policy = policies[name][family]["selected"]
            family_components = component_paths[name][family]["target"]
            probability = mix(base, np.load(directory / f"{family}_target.npy", mmap_mode="r"),
                              family_components, policy)
            metrics[family], scores[family] = _metrics(
                targets, probability, policy, family_components, countries,
                support["distance_km"][indices], counts, baseline=base, species_ids=species,
            )
        metrics["single_head_ensemble"]["individual_seed_sample_f1"] = {}
        primary_policy = policies[name]["single_head_ensemble"]["selected"]
        primary_components = component_paths[name]["single_head_ensemble"]["target"]
        for seed in V23_SEEDS:
            seed_path = directory / f"single_head_seed_{seed}_target.npy"
            seed_probability = mix(base, np.load(seed_path, mmap_mode="r"),
                                   primary_components, primary_policy)
            metrics["single_head_ensemble"]["individual_seed_sample_f1"][str(seed)] = float(
                per_survey_f1(targets, seed_probability, primary_policy, primary_components).mean())
        fold_metrics[name] = metrics
        fold_scores.append({key: float(value.mean()) for key, value in scores.items()})
        frames.append(pd.DataFrame({"surveyId": rows.surveyId.to_numpy()[indices], "fold": fold,
            "country": countries, "block": spatial_block_ids(rows.iloc[indices]),
            "pa_distance_km": support["distance_km"][indices], **scores}))
    frame = pd.concat(frames, ignore_index=True)
    if frame.surveyId.duplicated().any() or len(frame) != 11032:
        raise ValueError("v23 assessment folds overlap or changed")
    frame.to_csv(output / "assessment_per_survey.csv", index=False)
    comparisons = {name: paired_block_bootstrap(frame.single_head_ensemble.to_numpy(),
                   frame[name].to_numpy(), frame.block.to_numpy())
                   for name in ("frozen_v22", "retained_po", "zero_po", "large_single_head")}
    assessment = {"sample_f1": {name: float(frame[name].mean()) for name in
                  ("frozen_v20", "frozen_v21", "frozen_v22", *FAMILIES)}, "fold_sample_f1": fold_scores,
                  "metrics": fold_metrics, "comparisons": comparisons,
                  "surveys": len(frame), "blocks": int(frame.block.nunique()),
                  "used_for_selection": False, "now_consumed": True,
                  "warning": "Internal cross-fit recipe-transfer assessment, not a hidden-test score."}
    integrity = {
        "source_commit_exact": os.environ.get("GLC_SOURCE_COMMIT") == args.expected_commit,
        "kernel_version_exact": int(os.environ.get("GLC_KERNEL_VERSION", EXPECTED_KERNEL_VERSION)) == EXPECTED_KERNEL_VERSION,
        "all_5016_species": len(species) == 5016,
        "new_assessment_only": all(not item[2]["assessment_previously_consumed"] for item in folds),
        "selection_calibration_assessment_separate": True,
        "twenty_km_buffer": all(item[2]["minimum_evaluation_distance_km"] >= 20.0 - 1e-7 for item in folds),
        "exact_unchanged_v22": sha256_file(output / "unchanged_v22_submission.csv") == OFFICIAL_CSV_SHA256,
        "frozen_v22_parity": v22_proof["parity"]["exact_rank_probability_parity"] is True,
        "policies_unchanged": sha256_file(output / "frozen_policies.json") == freeze["policies_sha256"],
        "checkpoint_hashes_unchanged": all(
            (output / name).is_file() and sha256_file(output / name) == digest
            for name, digest in freeze["checkpoint_sha256"].items()),
        "submission_unchanged_after_freeze": sha256_file(destination) == freeze["submission_sha256"],
        "competition_only": True, "test_labels_unused": True, "no_post_assessment_training": True,
        "notebook_tests_before_passed": os.environ.get("GLC_TESTS_BEFORE") == "1",
        "single_cuda_device_used": device.index == 0, "registered_runtime": time.time() - started < MAX_HOURS * 3600,
    }
    gate = submission_gate(assessment, integrity, policies)
    report = {
        "experiment": "v23_diverse_single_head_crossfit", "status": "complete",
        "source_commit": args.expected_commit, "kernel": "con1los/geolifeclef-risk-aware-sdm-phase-1",
        "kernel_version": EXPECTED_KERNEL_VERSION, "frozen_v22_source_version": 22,
        "frozen_v22_source_commit": v22_proof["source_commit"], "species": len(species),
        "test_rows": len(test_rows), "split": [item[2] for item in folds],
        "selected_policies": policies, "training": training, "assessment": assessment,
        "integrity": integrity, "submission_gate": gate, "submission": submission,
        "output_csv": OUTPUT_CSV, "registered_max_total_hours": MAX_HOURS,
        "total_pipeline_hours": (time.time() - started) / 3600,
        "notebook_tests_before_passed": os.environ.get("GLC_TESTS_BEFORE") == "1",
        "notebook_tests_after_passed": False, "external_data_or_weights": False,
        "official_submission_made": False,
    }
    save_json(output / "v23_report.json", report)
    print(json.dumps({"stage": "v23_complete", "internal_scores": assessment["sample_f1"],
                      "manual_submission_gate": gate, "hours": report["total_pipeline_hours"]}), flush=True)
    if time.time() - started >= MAX_HOURS * 3600:
        raise TimeoutError("v23 exceeded the 10.5-hour total guard")
    del labels, old_pool, new_pool
    gc.collect()
    if cache.resolve().is_relative_to((Path.cwd() / "data" / "processed").resolve()):
        shutil.rmtree(cache)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-v20-dir", type=Path, required=True)
    parser.add_argument("--frozen-v21-dir", type=Path, required=True)
    parser.add_argument("--frozen-v22-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/diverse_po_v23"))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save_json(args.output_dir / "failure.json", {"status": "failed",
                  "error_type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
