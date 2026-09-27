"""v33: diversify the successful asymmetric specialists; frozen scored v32 control.

No fresh assessment labels remain. Eligibility is a development gate, not proof
of better hidden-test performance. Only official competition data are required.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import json
import lzma
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

import scripts.v32_notebook_core as previous

v31, v29, legacy = previous.previous, previous.v29, previous.legacy
EXPERIMENT = "v33_diverse_asymmetric_multisensor"
CONTROL_HASH = "79c221bd82cf25a50de7eb9e5658824ba3784a6c250df98e613ac49b4bdb2cd8"
MAX_HOURS = 10.75
FROZEN_V32_POLICY = {"id": "bag1_specialist1_swap4", "bag": 1., "specialist": 1., "swaps": 4}
ATTENTION_CONFIGS = tuple({**c, "id": "replica_"+c["id"], "seed": c["seed"]+33000} for c in previous.SPECIALISTS)
CONV_CONFIGS = ({"id": "asymmetric_pyramid_conv", "kind": "conv", "geo": True, "width": 192,
    "seed": 20263313, "epochs": 24, "rare_branch": True, "negative_clip": .02, "negative_gamma": 2.},)
POLICIES = ({"id": "control", "attention": 0., "conv": 0., "swaps": 0},) + tuple(
    {"id": f"attention{a:g}_conv{c:g}_swap{k}", "attention": a, "conv": c, "swaps": k}
    for a, c in ((.5, 0.), (1., 0.), (0., .5), (0., 1.), (.5, .5), (1., .5), (.5, 1.), (1., 1.))
    for k in (2, 4, 8))


def groups():
    return (("attention", ATTENTION_CONFIGS), ("conv", CONV_CONFIGS))


def decode(base, attention, conv, policy, guard=None):
    return v31.residual_decode(base, attention, conv,
        {"neural": policy["attention"], "habitat": policy["conv"], "swaps": policy["swaps"]}, guard)


def member_diagnostics(data, target, countries, guard=None):
    """Fixed diagnostics only, never another model/policy search."""
    old = legacy.score_prediction_lists(target, data["base"])
    result = {}
    for name, expert in data["members"].items():
        alone = [row[:len(base)].astype(int).tolist() for row, base in zip(expert["rank"], data["base"])]
        bounded = decode(data["base"], expert, None, {"attention": 1., "conv": 0., "swaps": 4}, guard)
        scores = legacy.score_prediction_lists(target, bounded)
        result[name] = {"standalone_fixed_reference_count_f1": float(legacy.score_prediction_lists(target, alone).mean()),
            "bounded_4_swap_f1": float(scores.mean()), **v31.geographic_gain(scores-old, countries),
            "used_for_member_selection": False}
    return result


def fit_fold(number, split, rows, store, temporary, guard, device):
    started = time.monotonic()
    baseline = previous.fit_fold(number, split, rows, store, temporary, guard, device)
    data = {role: {"base": previous.decode(d["base"], d["bag"], d["specialist"], FROZEN_V32_POLICY, guard),
                   "members": {}} for role, d in baseline["data"].items()}
    # Account for ALL eleven frozen reference fits, not just the last five.
    reference_records = baseline["reference_records"]+baseline["records"]
    del baseline
    records, calibrations = [], {}
    stats = v29.fit_normalization(store, split["training"])
    directory = temporary/f"v33_fold_{number}"; directory.mkdir()
    for group, configs in groups():
        sums = {role: np.zeros((len(split[role]), len(store.species_ids)), np.float32) for role in data}
        for i, source in enumerate(configs):
            guard.require(3.5*3600, "start v33 development specialist")
            config = {**source, "seed": source["seed"]+number*1000}
            model, record = previous.train_specialist(store, rows, split["training"], stats, config,
                                                     directory, guard, device, "development")
            p = v29.predict(model, store, split["selection"], stats, device, config, views=2, guard=guard)
            cal = v29.fit_platt(p, np.asarray(store.labels[split["selection"]]))
            calibrations[config["id"]] = cal
            for role in data:
                p = v29.predict(model, store, split[role], stats, device, config, views=2, guard=guard)
                calibrated = v29.calibrated(p, cal)
                sums[role] += calibrated/len(configs)
                rank, value = v31.compact_rank(calibrated)
                data[role]["members"][config["id"]] = {"rank": rank, "value": value}
            records.append({**record, "group": group, "member": i})
            del model, p, calibrated
            previous.release(device)
        for role, probability in sums.items():
            rank, value = v31.compact_rank(probability)
            data[role][group] = {"rank": rank, "value": value}
        del sums, probability
    target, d = np.asarray(store.labels[split["calibration"]]), data["calibration"]
    trials = {p["id"]: legacy.score_prediction_lists(target, decode(d["base"], d["attention"], d["conv"], p, guard)) for p in POLICIES}
    diagnostics = member_diagnostics(d, target, rows.iloc[split["calibration"]].country, guard)
    # Assessment labels are NOT read here; diagnostics are computed after policy freeze.
    return {"number": number, "split": split, "data": data, "records": records, "calibrations": calibrations,
        "reference_records": reference_records, "trials": trials, "calibration_members": diagnostics,
        "wall_seconds": time.monotonic()-started}


def select_policy(bundles, rows):
    trials = []
    take = np.concatenate([b["split"]["calibration"] for b in bundles])
    reference = np.concatenate([b["trials"]["control"] for b in bundles])
    for policy in POLICIES:
        scores = np.concatenate([b["trials"][policy["id"]] for b in bundles])
        gains = [float((b["trials"][policy["id"]]-b["trials"]["control"]).mean()) for b in bundles]
        geo = v31.geographic_gain(scores-reference, rows.iloc[take].country, rows.iloc[take].surveyId)
        allowed = policy["id"] == "control" or (min(gains) > 0 and geo["geographic_gain_positive"])
        trials.append({**policy, **geo, "fold_gains": gains, "allowed": bool(allowed), "sample_f1": float(scores.mean())})
    chosen = max((t for t in trials if t["allowed"]), key=lambda t: (t["robust_gain"], -t["swaps"], -t["attention"]-t["conv"]))
    return next(dict(p) for p in POLICIES if p["id"] == chosen["id"]), trials


def regression_check(bundles, policy, rows, store, guard):
    frames, summaries = [], []
    for b in bundles:
        take, d = b["split"]["assessment"], b["data"]["assessment"]
        target = np.asarray(store.labels[take])
        prediction = decode(d["base"], d["attention"], d["conv"], policy, guard)
        old, new = [legacy.score_prediction_lists(target, p) for p in (d["base"], prediction)]
        ablations = {}
        for group, _ in groups():
            p = decode(d["base"], d["attention"], d["conv"], {**policy, group: 0.}, guard)
            ablations["without_"+group] = float((legacy.score_prediction_lists(target, p)-old).mean())
        frames.append(pd.DataFrame({"surveyId": rows.iloc[take].surveyId.to_numpy(), "fold": b["number"],
            "country": rows.iloc[take].country.to_numpy(), "spatial_block": legacy.spatial_blocks(rows.iloc[take]),
            "matched_v32_f1": old, "v33_f1": new, "delta_f1": new-old,
            "predicted_cardinality": list(map(len, prediction)),
            "swaps": [len(set(a)-set(c)) for a, c in zip(prediction, d["base"])]}))
        summaries.append({"fold": b["number"], "gain": float((new-old).mean()), "ablations_not_used_for_selection": ablations,
            "member_diagnostics_not_used_for_selection": member_diagnostics(d, target, rows.iloc[take].country, guard),
            "multilabel": v29.multilabel_summary(target, prediction),
            "species_groups": legacy.species_group_metrics(target, prediction, legacy._frequency(store.labels, b["split"]["training"]))})
    frame = pd.concat(frames, ignore_index=True)
    if not frame.surveyId.is_unique:
        raise ValueError("Duplicate regression survey")
    return frame, {**v31.geographic_gain(frame.delta_f1, frame.country), "folds": summaries,
        "surveys": len(frame), "matched_v32_f1": float(frame.matched_v32_f1.mean()), "v33_f1": float(frame.v33_f1.mean()),
        "bootstrap": legacy.paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy(), iterations=1000, seed=20263307),
        "fresh_assessment": False, "used_for_policy_selection_in_this_run": False,
        "by_country": legacy.summarize_by_group(frame, "country", ("matched_v32_f1", "v33_f1", "delta_f1"))}


def production_estimate(bundles, policy, full_rows):
    seconds = 45*60
    for group, configs in groups():
        if not policy[group]:
            continue
        for i, config in enumerate(configs):
            speed = max(np.median([h["seconds"] for h in r["history"]])/r["training_surveys"]
                        for b in bundles for r in b["records"] if r["group"] == group and r["member"] == i)
            seconds += speed*full_rows*config["epochs"]*1.5
    return float(seconds)


def remaining_plan_estimate(bundles, next_training_rows, full_rows):
    first = bundles[0]
    # Includes prediction, all eleven reference fits, calibration, and diagnostics.
    development = first["wall_seconds"]*max(1., next_training_rows/len(first["split"]["training"]))*1.25
    return float(development+production_estimate(bundles, {"attention": 1., "conv": 1.}, full_rows)+10*60)


def fit_production(bundles, policy, rows, test_rows, store, control, directory, guard, device):
    estimate = production_estimate(bundles, policy, len(rows))
    guard.require(estimate, "whole v33 production ensemble admission")
    take = np.arange(len(rows)); stats = v29.fit_normalization(store, take)
    experts, records = {"attention": None, "conv": None}, []
    for group, configs in groups():
        if not policy[group]:
            continue
        probability = np.zeros((len(test_rows), len(store.species_ids)), np.float32)
        for config in configs:
            model, record = previous.train_specialist(store, rows, take, stats, config, directory, guard, device, "production")
            cal = {k: float(np.mean([b["calibrations"][config["id"]][k] for b in bundles])) for k in ("slope", "intercept")}
            p = v29.predict(model, store, np.arange(len(test_rows)), stats, device, config, test=True, views=2, guard=guard)
            probability += v29.calibrated(p, cal)/len(configs)
            records.append({**record, "group": group, "calibration": cal,
                            "calibration_source": "mean selection-only fold coefficients; full-data transfer assumption"})
            del model, p
            previous.release(device)
        rank, value = v31.compact_rank(probability)
        experts[group] = {"rank": rank, "value": value}
        del probability
    prediction = decode(control, experts["attention"], experts["conv"], policy, guard)
    swaps = np.array([len(set(a)-set(b)) for a, b in zip(prediction, control)])
    return prediction, records, {"training_rows": len(take), "all_PA_rows_used": True,
        "calibration_anchor_rows_removed": 0, "cardinality_equal_to_scored_v32_per_row": True,
        "admission_estimate_seconds": estimate, "changed_rows": int((swaps > 0).sum()),
        "mean_swaps": float(swaps.mean()), "maximum_swaps": int(swaps.max())}


def decode_control(payload, template, species):
    decoder = v31.previous
    decoder.CONTROL_PAYLOAD_HASH, decoder.CONTROL_RAW_HASH = CONTROL_PAYLOAD_HASH, CONTROL_RAW_HASH
    return decoder.decode_control(payload, template, species)


def publish(export, template, ids, prediction, species, gate):
    tentative = export/"candidate_DO_NOT_SUBMIT.csv"
    proof = legacy.write_submission(tentative, template, ids, prediction, species)
    differs = proof["sha256"] != CONTROL_HASH
    eligible = bool(gate and differs)
    name = "GLC25_PA_submission_v33.csv" if eligible else (tentative.name if differs else "unchanged_v32_DO_NOT_SUBMIT.csv")
    if name != tentative.name:
        tentative.replace(export/name)
    return proof, {"eligible_for_submission": eligible, "different_from_v32": differs, "prediction_file": name,
        "message": "SUBMIT ONLY THIS CSV" if eligible else "DO NOT SUBMIT THIS OUTPUT; keep the scored v32"}


def save_compact(path, bundles, rows, store, per_fold=1500):
    ids, folds, countries, bases, ranks, values, truth = [], [], [], [], [], [], []
    names = [c["id"] for _, configs in groups() for c in configs]
    for b in bundles:
        take, d = b["split"]["calibration"], b["data"]["calibration"]
        positions = np.sort(np.random.default_rng(20263308+b["number"]).choice(len(take), min(per_fold, len(take)),
                            replace=False, p=v29.sample_weights(rows, take)))
        selected = take[positions]
        base = np.full((len(selected), 40), 65535, np.uint16)
        for j, pos in enumerate(positions):
            base[j, :len(d["base"][pos])] = d["base"][pos]
        experts = [d["attention"], d["conv"]]+[d["members"][n] for n in names]
        ranks.append(np.stack([e["rank"][positions] for e in experts], axis=1))
        values.append(np.stack([e["value"][positions] for e in experts], axis=1))
        bases.append(base); ids.append(rows.iloc[selected].surveyId.to_numpy()); folds.append(np.full(len(selected), b["number"], np.uint8))
        countries.append(rows.iloc[selected].country.fillna("unknown").to_numpy(dtype="U64"))
        truth.extend(np.flatnonzero(store.labels[j]).astype(np.uint16) for j in selected)
    np.savez_compressed(path, survey_id=np.concatenate(ids), fold=np.concatenate(folds), country=np.concatenate(countries),
        reference_columns_padded65535=np.concatenate(bases), ranked_species_columns=np.concatenate(ranks),
        probability=np.concatenate(values), true_species_columns=np.concatenate(truth),
        true_offsets=np.r_[0, np.cumsum(list(map(len, truth)))].astype(np.uint32), species_ids=store.species_ids,
        expert_ids=np.array(["attention_ensemble", "convolution_ensemble"]+names),
        reference_version=np.array("matched frozen v32 recipe, not exact historical weights"),
        evidence_scope=np.array("sampled consumed calibration; top128 per group AND per member; not a fresh audit"))


def self_tests():
    assert len(POLICIES) == 25
    assert len(ATTENTION_CONFIGS) == 2 and len(CONV_CONFIGS) == 1
    assert CONV_CONFIGS[0]["kind"] == "conv"
    base = [list(range(10))]
    e = {"rank": np.arange(30, 10, -1)[None], "value": np.ones((1, 20))}
    assert decode(base, None, None, POLICIES[0]) == base
    v31.validate_residual(base, decode(base, e, e, POLICIES[-1]), POLICIES[-1])
    return {"passed": True, "tests": 5}


def run_v33(control_b64):
    guard = legacy.RuntimeGuard(MAX_HOURS)
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
    temporary, export = working/"v33_runtime", working/"v33_export"
    legacy._clean_directory(temporary, working); legacy._clean_directory(export, working)
    temporary.mkdir(parents=True); export.mkdir(parents=True)
    store, bundles = None, []
    try:
        device = legacy.require_gpu()
        torch.set_num_threads(min(os.cpu_count() or 2, 6))
        tests = self_tests()
        root = legacy.discover_data_root()
        preflight = pd.read_csv(root/"GLC25_PA_metadata_train.csv", usecols=["surveyId", "lat", "lon", "country"]).drop_duplicates("surveyId").reset_index(drop=True)
        _, manifests = v31.previous.make_splits(preflight)
        guard.stamp("preflight", folds=manifests, fresh_assessment=False)
        original_writer = legacy._write_remote_arrays
        try:
            legacy._write_remote_arrays = v29.write_multiresolution
            features = legacy.prepare_feature_store(root, temporary/"features", guard, workers=6)
        finally:
            legacy._write_remote_arrays = original_writer
        store = legacy.FeatureStore(temporary/"features")
        store.high_train = np.load(store.cache/"train_sentinel64.npy", mmap_mode="r")
        store.high_test = np.load(store.cache/"test_sentinel64.npy", mmap_mode="r")
        rows, test_rows, pairs = legacy.load_rows_and_pairs(root, store.train_ids, store.test_ids)
        del pairs, preflight
        store.static_train, store.static_test = v29.candidate_static(rows), v29.candidate_static(test_rows)
        store.eco_train, store.eco_test = v29.candidate_static(rows, False), v29.candidate_static(test_rows, False)
        splits, manifests = v31.previous.make_splits(rows)
        template = pd.read_csv(root/"GLC25_SAMPLE_SUBMISSION.csv")
        control = decode_control(control_b64, template, store.species_ids)
        order = pd.Index(template.surveyId).get_indexer(store.test_ids)
        if (order < 0).any() or not template.surveyId.is_unique:
            raise ValueError("Test/template IDs mismatch")
        control = [control[i] for i in order]
        proof = legacy.write_submission(temporary/"control.csv", template, store.test_ids, control, store.species_ids)
        if proof["sha256"] != CONTROL_HASH:
            raise ValueError("Exact scored v32 round trip failed")
        for i, split in enumerate(splits):
            if bundles:
                estimate = remaining_plan_estimate(bundles, len(split["training"]), len(rows))
                guard.stamp("remaining_plan_admission", estimate_seconds=estimate)
                guard.require(estimate, "remaining development and largest production plan admission")
            bundles.append(fit_fold(i, split, rows, store, temporary, guard, device))
        policy, trials = select_policy(bundles, rows)
        legacy.save_json(temporary/"frozen_policy.json", {"policy": policy, "trials": trials})
        guard.stamp("policy_frozen", policy=policy)
        frame, regression = regression_check(bundles, policy, rows, store, guard)
        gates = {"new_policy": policy["id"] != "control", "positive_each_fold": all(f["gain"] > 0 for f in regression["folds"]),
            "spatial_ci_positive": regression["bootstrap"]["ci95"][0] > 0,
            "geographic_transfer_positive": regression["geographic_gain_positive"],
            "geographic_buffers": min(m["minimum_distance_km"] for m in manifests) >= 20,
            "exact_scored_control": proof["sha256"] == CONTROL_HASH}
        prediction, fitted, diagnostics = control, [], {"skipped": "development gate failed"}
        if all(gates.values()):
            prediction, fitted, diagnostics = fit_production(bundles, policy, rows, test_rows, store, control, temporary, guard, device)
        v31.validate_residual(control, prediction, policy)
        gates["within_budget"] = guard.elapsed_hours() < MAX_HOURS
        csv_proof, decision = publish(export, template, store.test_ids, prediction, store.species_ids, all(gates.values()))
        frame.to_csv(export/"regression_per_survey_v33.csv", index=False)
        save_compact(export/"calibration_top128_v33.npz", bundles, rows, store)
        report = {"experiment": EXPERIMENT, "status": "complete", **decision, "runtime_hours": guard.elapsed_hours(),
            "self_tests": tests, "policy": policy, "calibration_trials": trials, "gates": gates,
            "regression": regression, "submission_validation": csv_proof, "official_submission_made": False,
            "training": {"development": [{"fold": b["number"], "members": b["records"], "reference_members": b["reference_records"],
                "calibrations": b["calibrations"], "member_calibration_diagnostics": b["calibration_members"],
                "wall_seconds": b["wall_seconds"]} for b in bundles], "production": fitted},
            "production_diagnostics": diagnostics,
            "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__},
            "validation_status": "all 88,987 PA IDs previously assessed; repeated spatial development, no fresh holdout",
            "limitations": ["No guarantee of hidden-test improvement, winning or completion on an unmeasured GPU session.",
                "Reference v32 is a frozen-recipe refit; only the official-test control CSV is byte-exact.",
                "Repeated development gates and bootstrap are not independent evidence after many experiments.",
                "Production calibration transfers from smaller training folds; rare branch membership can change.",
                "Only two aggregate experts select the fusion policy; individual-member diagnostics are not model selection.",
                "Fixed cardinality and protected leading 60% can prevent useful larger changes."]}
        legacy.save_json(export/"v33_report.json", report)
        manifest = {"experiment": EXPERIMENT, "source_sha256": V33_SOURCE_HASH, "embedded_sources_sha256": FROZEN_SOURCE_HASHES,
            "splits": manifests, "features": features, "attention_configs": ATTENTION_CONFIGS, "conv_configs": CONV_CONFIGS,
            "policies": POLICIES, "frozen_v32_policy": FROZEN_V32_POLICY, "control_sha256": CONTROL_HASH,
            "runtime_cap_hours": MAX_HOURS, "no_external_data_or_weights": True, "fresh_assessment": False,
            "outputs": {p.name: legacy.sha256_file(p) for p in sorted(export.iterdir())}}
        legacy.save_json(export/"v33_manifest.json", manifest)
        size = sum(p.stat().st_size for p in export.iterdir())
        if len(list(export.iterdir())) != 5 or size > 16_000_000:
            raise ValueError("Compact five-file/16MB export contract exceeded")
        guard.require(0, "final compact export")
        return {"status": "complete", **decision, "runtime_hours": guard.elapsed_hours(), "output_bytes": size,
            "export_directory": str(export), "regression_gain": regression["gain"], "fresh_assessment": False, "policy": policy}
    except Exception as error:
        ready = export/"GLC25_PA_submission_v33.csv"
        if ready.exists():
            ready.replace(export/"failed_DO_NOT_SUBMIT.csv")
        # A late failure must also revoke the eligibility flag in an existing report.
        report_path = export/"v33_report.json"
        if report_path.exists():
            failed = json.loads(report_path.read_text(encoding="utf-8"))
            failed.update(status="failed", eligible_for_submission=False, message="DO NOT SUBMIT; export failed")
            failed["prediction_file"] = "failed_DO_NOT_SUBMIT.csv" if (export/"failed_DO_NOT_SUBMIT.csv").exists() else None
            legacy.save_json(report_path, failed)
        manifest_path = export/"v33_manifest.json"
        if manifest_path.exists():
            manifest_path.replace(export/"failed_manifest_not_a_valid_export.json")
        legacy.save_json(export/"failure_report.json", {"experiment": EXPERIMENT, "status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "runtime_hours": guard.elapsed_hours(), "official_submission_made": False,
            "eligible_for_submission": False, "completed_folds": [{"fold": b["number"], "members": b["records"],
                "calibrations": b["calibrations"], "wall_seconds": b["wall_seconds"]} for b in bundles]})
        raise
    finally:
        del store
        previous.release(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        legacy._clean_directory(temporary, working)
