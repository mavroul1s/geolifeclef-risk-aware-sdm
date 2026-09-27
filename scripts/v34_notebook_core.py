"""v34: shrinkage co-occurrence reranking around the exact scored v32 CSV.

No fresh assessment remains. This is a candidate, not established improvement.
"""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits

import scripts.v32_notebook_core as previous

v31, v29, legacy = previous.previous, previous.v29, previous.legacy
EXPERIMENT = "v34_cooccurrence_community_reranking"
CONTROL_HASH = "71c863d9cb05f3efc63ac54dd7faa05efd2fc137a11da9adcd2064c82c88dfee"
MAX_HOURS = 10.75
FROZEN_V32_POLICY = {"id": "bag1_specialist1_swap4", "bag": 1., "specialist": 1., "swaps": 4}
GRAPH_STRENGTHS = (0., .25, .5, 1.)
POLICIES = ({"id": "control", "alpha": 0., "weight": 0., "swaps": 0},) + tuple(
    {"id": f"graph{a:g}_weight{w:g}_swap{k}", "alpha": a, "weight": w, "swaps": k}
    for a in GRAPH_STRENGTHS for w in (1., 2.) for k in (2, 4))
# Same numerical calibration used in the successful v32 production run.
SPECIALIST_PRODUCTION_CALIBRATIONS = {
    "asymmetric_geo": {"slope": 2.149121353985206, "intercept": -3.0941123693055497},
    "rare_ecology": {"slope": 1.6738341207715663, "intercept": -1.9531678851363052}}


class CommunityGraph:
    """Train-only pair association, shrunk to the training marginal prevalence.

    Conditional P(j|i)=(cooccurrence(i,j)+100*prior(j))/(count(i)+100).
    Clip log(P(j|i)/prior(j)) to [-2,2]; remove self-edges. Associations are
    observational (including survey effort/geography), not causal interactions.
    """
    def fit(self, labels, indices, guard=None):
        indices = np.asarray(indices, np.int64)
        if not len(indices) or len(indices) != len(np.unique(indices)):
            raise ValueError("Empty or repeated graph training rows")
        if guard:
            guard.require(30*60, "community graph fit")
        started = time.monotonic()
        y = sparse.csr_matrix(np.asarray(labels[indices]), dtype=np.float32)
        count = np.asarray(y.sum(0)).ravel()
        prior = (count+.5)/(len(indices)+1.)
        with threadpool_limits(limits=4):
            conditional = (y.T@y).toarray()
            conditional += 100.*prior[None]
            conditional /= count[:, None]+100.
            conditional /= prior[None]
            np.log(conditional, out=conditional)
            np.clip(conditional, -2., 2., out=conditional)
        np.fill_diagonal(conditional, 0.)
        self.lift = conditional
        self.record = {"training_rows": len(indices), "species": len(count), "pseudocount": 100.,
            "log_lift_clip": [-2., 2.], "diagonal_zero": True, "minimum_species_count_filter": None,
            "training_indices_sha256": hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
            "seconds": time.monotonic()-started, "bytes": self.lift.nbytes,
            "causal_interpretation": False}
        return self

    def predict_rank(self, base, bag, specialist, strengths=GRAPH_STRENGTHS, guard=None):
        n, species = len(base), len(self.lift)
        if len(bag["rank"]) != n or len(specialist["rank"]) != n:
            raise ValueError("Community expert row mismatch")
        output = {str(a): {"rank": [], "value": []} for a in strengths}
        with threadpool_limits(limits=4):
            for start in range(0, n, 128):
                if guard:
                    guard.require(20*60, "community context inference")
                stop = min(start+128, n)
                probability = np.full((stop-start, species), 1e-7, np.float32)
                row_index = np.arange(stop-start)[:, None]
                for expert in (bag, specialist):
                    probability[row_index, expert["rank"][start:stop]] += expert["value"][start:stop].astype(np.float32)/2.
                anchor_rows, anchor_cols, anchor_weights = [], [], []
                for j, old in enumerate(base[start:stop]):
                    anchors = old[:min(12, math.ceil(.6*len(old)))]
                    weight = probability[j, anchors].copy()
                    weight /= weight.sum()
                    anchor_rows.extend([j]*len(anchors)); anchor_cols.extend(anchors); anchor_weights.extend(weight)
                seeds = sparse.csr_matrix((anchor_weights, (anchor_rows, anchor_cols)), shape=probability.shape, dtype=np.float32)
                context = seeds @ self.lift
                log_probability = np.log(probability)
                for a in strengths:
                    corrected = log_probability+a*context
                    # Stable, full sort gives species-column deterministic tie breaking.
                    rank = np.argsort(-corrected, axis=1, kind="stable")[:, :min(128, species)]
                    value = np.exp(np.take_along_axis(corrected, rank, axis=1))
                    output[str(a)]["rank"].append(rank.astype(np.uint16))
                    output[str(a)]["value"].append(value.astype(np.float16))
        return {key: {k: np.concatenate(v) for k, v in expert.items()} for key, expert in output.items()}


def decode(data, policy, guard=None):
    expert = data["community"][str(float(policy["alpha"]))] if policy["swaps"] else None
    return v31.residual_decode(data["base"], expert, None,
        {"neural": policy["weight"], "habitat": 0., "swaps": policy["swaps"]}, guard)


def fit_fold(number, split, rows, store, temporary, guard, device):
    started = time.monotonic()
    bundle = previous.fit_fold(number, split, rows, store, temporary, guard, device)
    graph = CommunityGraph().fit(store.labels, split["training"], guard)
    for role, data in bundle["data"].items():
        data["base"] = previous.decode(data["base"], data["bag"], data["specialist"], FROZEN_V32_POLICY, guard)
        data["community"] = graph.predict_rank(data["base"], data["bag"], data["specialist"], guard=guard)
    target = np.asarray(store.labels[split["calibration"]])
    bundle["trials"] = {p["id"]: legacy.score_prediction_lists(target, decode(bundle["data"]["calibration"], p, guard)) for p in POLICIES}
    bundle["graph"] = graph.record
    bundle["wall_seconds"] = time.monotonic()-started
    # Assessment labels have not been read by the added graph or policy search.
    return bundle


def select_policy(bundles, rows):
    take = np.concatenate([b["split"]["calibration"] for b in bundles])
    reference = np.concatenate([b["trials"]["control"] for b in bundles])
    trials = []
    for policy in POLICIES:
        scores = np.concatenate([b["trials"][policy["id"]] for b in bundles])
        gains = [float((b["trials"][policy["id"]]-b["trials"]["control"]).mean()) for b in bundles]
        geo = v31.geographic_gain(scores-reference, rows.iloc[take].country, rows.iloc[take].surveyId)
        allowed = policy["id"] == "control" or (min(gains) > 0 and geo["geographic_gain_positive"])
        trials.append({**policy, **geo, "fold_gains": gains, "allowed": bool(allowed), "sample_f1": float(scores.mean())})
    chosen = max((t for t in trials if t["allowed"]), key=lambda t: (t["robust_gain"], -t["swaps"], -t["alpha"], -t["weight"]))
    return next(dict(p) for p in POLICIES if p["id"] == chosen["id"]), trials


def regression_check(bundles, policy, rows, store, guard):
    frames, summaries = [], []
    for b in bundles:
        take, data = b["split"]["assessment"], b["data"]["assessment"]
        target = np.asarray(store.labels[take])
        prediction = decode(data, policy, guard)
        old, new = [legacy.score_prediction_lists(target, p) for p in (data["base"], prediction)]
        no_graph = decode(data, {**policy, "alpha": 0.}, guard)
        no_graph_score = legacy.score_prediction_lists(target, no_graph)
        frames.append(pd.DataFrame({"surveyId": rows.iloc[take].surveyId.to_numpy(), "fold": b["number"],
            "country": rows.iloc[take].country.to_numpy(), "spatial_block": legacy.spatial_blocks(rows.iloc[take]),
            "matched_v32_f1": old, "v34_f1": new, "delta_f1": new-old,
            "without_cooccurrence_f1": no_graph_score, "cooccurrence_increment": new-no_graph_score,
            "predicted_cardinality": list(map(len, prediction)),
            "swaps": [len(set(a)-set(c)) for a, c in zip(prediction, data["base"])]}))
        summaries.append({"fold": b["number"], "gain": float((new-old).mean()),
            "without_cooccurrence_gain": float((no_graph_score-old).mean()),
            "cooccurrence_increment_not_used_for_selection": float((new-no_graph_score).mean()),
            "multilabel": v29.multilabel_summary(target, prediction),
            "species_groups": legacy.species_group_metrics(target, prediction, legacy._frequency(store.labels, b["split"]["training"]))})
    frame = pd.concat(frames, ignore_index=True)
    if not frame.surveyId.is_unique:
        raise ValueError("Duplicate regression survey")
    return frame, {**v31.geographic_gain(frame.delta_f1, frame.country), "folds": summaries,
        "surveys": len(frame), "matched_v32_f1": float(frame.matched_v32_f1.mean()), "v34_f1": float(frame.v34_f1.mean()),
        "bootstrap": legacy.paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy(), iterations=1000, seed=20263407),
        "fresh_assessment": False, "used_for_policy_selection_in_this_run": False,
        "by_country": legacy.summarize_by_group(frame, "country", ("matched_v32_f1", "v34_f1", "delta_f1"))}


def production_estimate(bundles, full_rows):
    # Same five v32 production models, plus a graph and inference safety reserve.
    return previous.production_estimate(bundles, {"bag": 1., "specialist": 1.}, full_rows)+15*60


def remaining_plan_estimate(bundles, next_training_rows, full_rows):
    first = bundles[0]
    development = first["wall_seconds"]*max(1., next_training_rows/len(first["split"]["training"]))*1.25
    return float(development+production_estimate(bundles, full_rows)+10*60)


def fit_production(bundles, policy, rows, test_rows, store, control, directory, guard, device):
    estimate = production_estimate(bundles, len(rows))
    guard.require(estimate, "whole v34 production ensemble admission")
    take = np.arange(len(rows)); stats = v29.fit_normalization(store, take)
    experts, records = {}, []
    for group, configs in (("bag", previous.BAG_CONFIGS), ("specialist", previous.SPECIALISTS)):
        probability = np.zeros((len(test_rows), len(store.species_ids)), np.float32)
        for i, config in enumerate(configs):
            if group == "bag":
                model, record = v29.train_candidate(store, rows, take, np.array([], np.int64), stats, config,
                    directory, guard, device, fixed_epochs=previous.BAG_EPOCHS[i], phase="production")
                cal = v31.PRODUCTION_CALIBRATIONS[i]
            else:
                model, record = previous.train_specialist(store, rows, take, stats, config, directory, guard, device, "production")
                cal = SPECIALIST_PRODUCTION_CALIBRATIONS[config["id"]]
            p = v29.predict(model, store, np.arange(len(test_rows)), stats, device, config, test=True, views=2, guard=guard)
            probability += v29.calibrated(p, cal)/len(configs)
            records.append({**record, "group": group, "calibration": cal, "calibration_source": "frozen scored v32 production coefficients"})
            del model, p
            previous.release(device)
        rank, value = v31.compact_rank(probability)
        experts[group] = {"rank": rank, "value": value}
        del probability
    graph = CommunityGraph().fit(store.labels, take, guard)
    data = {"base": control, **experts}
    data["community"] = graph.predict_rank(control, experts["bag"], experts["specialist"], strengths=(float(policy["alpha"]),), guard=guard)
    prediction = decode(data, policy, guard)
    swaps = np.array([len(set(a)-set(b)) for a, b in zip(prediction, control)])
    return prediction, records, {"training_rows": len(take), "all_PA_rows_used": True, "graph": graph.record,
        "calibration_anchor_rows_removed": 0, "cardinality_equal_to_scored_v32_per_row": True,
        "admission_estimate_seconds": estimate, "changed_rows": int((swaps > 0).sum()),
        "mean_swaps": float(swaps.mean()), "maximum_swaps": int(swaps.max())}


def decode_control(payload, template, species):
    decoder = v31.previous
    decoder.CONTROL_PAYLOAD_HASH, decoder.CONTROL_RAW_HASH = CONTROL_PAYLOAD_HASH, CONTROL_RAW_HASH
    return decoder.decode_control(payload, template, species)


def publish(export, temporary, template, ids, prediction, species, gate):
    tentative = temporary/"candidate.csv"
    proof = legacy.write_submission(tentative, template, ids, prediction, species)
    differs = proof["sha256"] != CONTROL_HASH
    eligible = bool(gate and differs)
    name = "GLC25_PA_submission_v34.csv" if eligible else None
    decision = {"eligible_for_submission": eligible, "different_from_v32": differs, "prediction_file": name,
        "message": "SUBMIT ONLY GLC25_PA_submission_v34.csv" if eligible else "NO NEW SUBMISSION GENERATED; keep your scored v32"}
    if eligible:
        tentative.replace(export/name)
    else:
        # Do not hand the user another duplicate control disguised as a new CSV.
        legacy.save_json(export/"NO_SUBMISSION.json", decision)
    return proof, decision


def save_compact(path, bundles, rows, store, policy, per_fold=1500):
    ids, folds, countries, bases, ranks, values, truth = [], [], [], [], [], [], []
    for b in bundles:
        take, data = b["split"]["calibration"], b["data"]["calibration"]
        positions = np.sort(np.random.default_rng(20263408+b["number"]).choice(len(take), min(per_fold, len(take)),
                            replace=False, p=v29.sample_weights(rows, take)))
        selected = take[positions]
        base = np.full((len(selected), 40), 65535, np.uint16)
        for j, pos in enumerate(positions):
            base[j, :len(data["base"][pos])] = data["base"][pos]
        experts = [data["bag"], data["specialist"]]+[data["community"][str(a)] for a in GRAPH_STRENGTHS]
        ranks.append(np.stack([e["rank"][positions] for e in experts], axis=1))
        values.append(np.stack([e["value"][positions] for e in experts], axis=1))
        bases.append(base); ids.append(rows.iloc[selected].surveyId.to_numpy()); folds.append(np.full(len(selected), b["number"], np.uint8))
        countries.append(rows.iloc[selected].country.fillna("unknown").to_numpy(dtype="U64"))
        truth.extend(np.flatnonzero(store.labels[j]).astype(np.uint16) for j in selected)
    np.savez_compressed(path, survey_id=np.concatenate(ids), fold=np.concatenate(folds), country=np.concatenate(countries),
        reference_columns_padded65535=np.concatenate(bases), ranked_species_columns=np.concatenate(ranks),
        score=np.concatenate(values), true_species_columns=np.concatenate(truth),
        true_offsets=np.r_[0, np.cumsum(list(map(len, truth)))].astype(np.uint32), species_ids=store.species_ids,
        expert_ids=np.array(["bag", "specialist"]+[f"community_alpha{a:g}" for a in GRAPH_STRENGTHS]),
        selected_policy=np.array(policy["id"]), reference_version=np.array("matched frozen v32 recipe, not exact historical weights"),
        evidence_scope=np.array("sampled consumed calibration; community values are ranking scores NOT calibrated probabilities"))


def self_tests():
    assert len(POLICIES) == 17
    base = [list(range(10))]
    e = {"rank": np.arange(30, 10, -1)[None], "value": np.ones((1, 20))}
    data = {"base": base, "community": {str(a): e for a in GRAPH_STRENGTHS}}
    for p in POLICIES:
        v31.validate_residual(base, decode(data, p), p)
    assert decode(data, POLICIES[0]) == base
    graph = CommunityGraph().fit(np.eye(40, dtype=np.uint8), np.arange(20))
    assert np.isfinite(graph.lift).all() and np.all(np.diag(graph.lift) == 0)
    assert np.abs(graph.lift).max() <= 2.
    return {"passed": True, "tests": 5}


def run_v34(control_b64):
    guard = legacy.RuntimeGuard(MAX_HOURS)
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
    temporary, export = working/"v34_runtime", working/"v34_export"
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
                guard.require(estimate, "remaining development and production admission")
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
        csv_proof, decision = publish(export, temporary, template, store.test_ids, prediction, store.species_ids, all(gates.values()))
        frame.to_csv(export/"regression_per_survey_v34.csv", index=False)
        save_compact(export/"calibration_top128_v34.npz", bundles, rows, store, policy)
        report = {"experiment": EXPERIMENT, "status": "complete", **decision, "runtime_hours": guard.elapsed_hours(),
            "self_tests": tests, "policy": policy, "calibration_trials": trials, "gates": gates,
            "regression": regression, "submission_validation": csv_proof, "official_submission_made": False,
            "training": {"development": [{"fold": b["number"], "members": b["records"], "reference_members": b["reference_records"],
                "calibrations": b["calibrations"], "graph": b["graph"], "wall_seconds": b["wall_seconds"]} for b in bundles], "production": fitted},
            "production_diagnostics": diagnostics,
            "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__},
            "validation_status": "all 88,987 PA IDs previously assessed; repeated spatial development, no fresh holdout",
            "limitations": ["No guarantee of hidden-test improvement, winning or 12-hour completion on an unmeasured session.",
                "Only the scored test-control CSV is byte-exact; reference development weights are refitted.",
                "Co-occurrence includes observational/geographic/sampling effects, not causal ecological interactions.",
                "A wrong protected anchor set can propagate errors; only 2/4 tail swaps are permitted.",
                "All PA labels and the sampled calibration pilot were already consumed; gates are not independent proof.",
                "Truncated top128 ensemble scores omit weak probabilities; zero-graph policies isolate probability reranking.",
                "Production uses all PA rows and frozen v32 calibration; fold-to-full-data transfer can change behavior."]}
        legacy.save_json(export/"v34_report.json", report)
        manifest = {"experiment": EXPERIMENT, "source_sha256": V34_SOURCE_HASH, "embedded_sources_sha256": FROZEN_SOURCE_HASHES,
            "splits": manifests, "features": features, "policies": POLICIES, "frozen_v32_policy": FROZEN_V32_POLICY,
            "specialist_production_calibrations": SPECIALIST_PRODUCTION_CALIBRATIONS, "control_sha256": CONTROL_HASH,
            "runtime_cap_hours": MAX_HOURS, "no_external_data_or_weights": True, "fresh_assessment": False,
            "outputs": {p.name: legacy.sha256_file(p) for p in sorted(export.iterdir())}}
        legacy.save_json(export/"v34_manifest.json", manifest)
        size = sum(p.stat().st_size for p in export.iterdir())
        if len(list(export.iterdir())) != 5 or size > 16_000_000:
            raise ValueError("Compact five-file/16MB export contract exceeded")
        guard.require(0, "final compact export")
        return {"status": "complete", **decision, "runtime_hours": guard.elapsed_hours(), "output_bytes": size,
            "export_directory": str(export), "regression_gain": regression["gain"], "fresh_assessment": False, "policy": policy}
    except Exception as error:
        ready = export/"GLC25_PA_submission_v34.csv"
        if ready.exists():
            ready.replace(export/"failed_predictions_DO_NOT_SUBMIT.txt")
        report_path = export/"v34_report.json"
        if report_path.exists():
            failed = json.loads(report_path.read_text(encoding="utf-8"))
            failed.update(status="failed", eligible_for_submission=False, prediction_file=None, message="DO NOT SUBMIT; export failed")
            legacy.save_json(report_path, failed)
        manifest_path = export/"v34_manifest.json"
        if manifest_path.exists():
            manifest_path.replace(export/"failed_manifest_not_a_valid_export.json")
        legacy.save_json(export/"failure_report.json", {"experiment": EXPERIMENT, "status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "runtime_hours": guard.elapsed_hours(), "official_submission_made": False,
            "eligible_for_submission": False, "prediction_file": None,
            "completed_folds": [{"fold": b["number"], "graph": b["graph"], "wall_seconds": b["wall_seconds"]} for b in bundles]})
        raise
    finally:
        del store
        previous.release(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        legacy._clean_directory(temporary, working)
