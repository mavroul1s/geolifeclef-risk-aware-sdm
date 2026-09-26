"""v31: full-data seed bagging and habitat analogues, with cardinality-locked residuals.

Repeated development evidence only: every official PA ID has already been assessed.
The scored v29 CSV remains immutable; no result here establishes SOTA.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA

import scripts.v30_notebook_core as previous

v29 = previous.previous
legacy = previous.legacy
EXPERIMENT = "v31_full_data_bagged_habitat_residual"
CONTROL_HASH = previous.CONTROL_HASH
MAX_HOURS = 10.75
CONFIGS = tuple(dict(c) for c in v29.CONFIGS)
# Freeze the epoch schedule of the best *scored* production recipe, not v30.
EPOCHS = (12, 15, 18)
PRODUCTION_CALIBRATIONS = (
    {"slope": .931340859404444, "intercept": -.031887795053823956},
    {"slope": .8515955628102694, "intercept": -.23047593294208438},
    {"slope": 1.9269219246980456, "intercept": -.5758844842834113})
POLICIES = ({"id": "control", "neural": 0., "habitat": 0., "swaps": 0},) + tuple(
    {"id": f"seed{n:g}_habitat{h:g}_swap{k}", "neural": n, "habitat": h, "swaps": k}
    for n, h in [(n, h) for n in (.5, 1.) for h in (0., .25, .5)] + [(0., 1.)]
    for k in (2, 4))
CORE_COUNTRIES = ("Denmark", "Netherlands")
TOP_K = 128


def compact_rank(probability):
    rank, values = legacy.top_rank(probability, min(TOP_K, probability.shape[1]))
    return rank.astype(np.uint16), values.astype(np.float16)


class HabitatAnalogues:
    """Train-only environmental PCA and exact nearest-neighbour label averaging.

    Country, survey IDs and coordinates are intentionally absent from the metric.
    Query labels are never an input. Sparse multiplication avoids B x K x species RAM.
    """
    def fit(self, environment, labels, indices):
        self.indices = np.asarray(indices, np.int64).copy()
        if len(self.indices) < 2 or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("Habitat fitting requires distinct training rows")
        x = np.asarray(environment[self.indices], np.float32).copy()
        finite = np.isfinite(x)
        self.mean = np.where(finite, x, 0).sum(0) / np.maximum(finite.sum(0), 1)
        x = np.where(finite, x, self.mean).astype(np.float32)
        self.scale = np.maximum(x.std(0), .01)
        x = np.clip((x-self.mean)/self.scale, -6, 6).astype(np.float32)
        components = min(32, x.shape[1], len(x)-1)
        self.pca = PCA(n_components=components, svd_solver="randomized", random_state=20263101)
        # Handle an entirely constant fixture/feature collection without NaN variance.
        if not np.any(x):
            self.pca = None
            self.divisor = np.ones(components, np.float32)
            self.latent = x[:, :components]
        else:
            self.pca.fit(x)
            self.divisor = np.maximum(self.pca.explained_variance_, .05)**.25
            self.latent = (self.pca.transform(x)/self.divisor).astype(np.float32)
        self.labels = sparse.csr_matrix(np.asarray(labels[self.indices], np.uint8)).astype(np.float32)
        if not np.isfinite(self.latent).all():
            raise FloatingPointError("Nonfinite habitat training metric")
        return self

    def transform(self, environment):
        x = np.asarray(environment, np.float32)
        x = np.where(np.isfinite(x), x, self.mean)
        x = np.clip((x-self.mean)/self.scale, -6, 6).astype(np.float32)
        return (x[:, :len(self.divisor)] if self.pca is None else
                self.pca.transform(x)/self.divisor).astype(np.float32)

    @staticmethod
    def neighbour_weights(distance):
        small = min(32, distance.shape[1])
        temperature = np.maximum(np.median(distance[:, :small], axis=1), 1e-4)
        weights = np.exp(-(distance-distance[:, :1])/temperature[:, None])
        local = weights[:, :small] / np.maximum(weights[:, :small].sum(1, keepdims=True), 1e-12)
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
        weights *= .5
        weights[:, :small] += .5*local
        return weights.astype(np.float32)

    @torch.no_grad()
    def predict_rank(self, environment, device, guard=None):
        latent = torch.as_tensor(self.latent, device=device)
        squared = latent.square().sum(1)[None]
        count = min(128, len(self.indices))
        ranks, values, distances = [], [], []
        batch = 128 if device.type == "cuda" else 16
        # TF32 can perturb very close neighbours; float32 matmul is intentional.
        for start in range(0, len(environment), batch):
            if guard is not None:
                guard.require(20*60, "habitat analogue inference")
            q = torch.as_tensor(self.transform(environment[start:start+batch]), device=device)
            d = (q.square().sum(1)[:, None] + squared - 2*q@latent.T).clamp_min_(0)
            nearest, indices = torch.topk(d, count, largest=False, sorted=True)
            indices, nearest = indices.cpu().numpy(), nearest.cpu().numpy()
            weights = self.neighbour_weights(nearest)
            assignment = sparse.csr_matrix((weights.ravel(), indices.ravel(),
                np.arange(0, (len(q)+1)*count, count)), shape=(len(q), len(self.indices)))
            probability = (assignment @ self.labels).toarray()
            rank, value = compact_rank(probability)
            ranks.append(rank); values.append(value); distances.append(nearest[:, 0])
        return {"rank": np.concatenate(ranks), "value": np.concatenate(values),
                "distance": np.concatenate(distances)}


def residual_decode(base, neural, habitat, policy, guard=None):
    """Protect the first 60%, preserve EVERY row's count, change at most 2/4 tail items."""
    if policy["swaps"] == 0:
        return [list(row) for row in base]
    result = []
    for i, old in enumerate(base):
        if guard is not None and i % 1024 == 0:
            guard.require(10*60, "bounded residual decoding")
        old_set = set(old)
        protected = set(old[:math.ceil(.6*len(old))])
        scores = {int(s): 1/(10+j) for j, s in enumerate(old)}
        for expert, weight in ((neural, policy["neural"]), (habitat, policy["habitat"])):
            if not weight:
                continue
            for j, (s, value) in enumerate(zip(expert["rank"][i], expert["value"][i])):
                if value > 0:  # Never promote zero-evidence habitat species.
                    scores[int(s)] = scores.get(int(s), 0.) + weight/(10+j)
        tail = sorted((s for s in old if s not in protected), key=lambda s: (scores[s], -old.index(s)))
        outside = sorted((s for s in scores if s not in old_set), key=lambda s: (-scores[s], s))
        removed, added = set(), []
        for outgoing, incoming in zip(tail[:policy["swaps"]], outside[:policy["swaps"]]):
            if scores[incoming] <= scores[outgoing] + 1e-12:
                break
            removed.add(outgoing); added.append(incoming)
        result.append([s for s in old if s not in removed] + added)
    validate_residual(base, result, policy)
    return result


def validate_residual(base, result, policy):
    if len(base) != len(result):
        raise ValueError("Residual row count changed")
    for old, new in zip(base, result):
        if (len(new) != len(old) or len(new) != len(set(new)) or
            not set(old[:math.ceil(.6*len(old))]).issubset(new) or
            len(set(new)-set(old)) > policy["swaps"]):
            raise ValueError("Residual count/prefix/swap contract broken")


def geographic_gain(delta, countries, ids=None):
    frame = pd.DataFrame({"country": np.asarray(countries).astype(str), "delta": delta})
    # Both calibration folds query the same survey IDs. Count each survey once
    # for minimum-support country checks; predictions still come from separate fits.
    if ids is not None:
        frame["surveyId"] = np.asarray(ids)
        frame = frame.groupby(["surveyId", "country"], as_index=False).delta.mean()
    country = frame.groupby("country").delta.agg(["mean", "size"])
    supported = country[country["size"] >= 30]
    outside = frame.loc[~frame.country.isin(CORE_COUNTRIES), "delta"]
    macro = float(supported["mean"].mean()) if len(supported) else 0.
    transfer = float(outside.mean()) if len(outside) else 0.
    return {"gain": float(frame.delta.mean()), "country_macro_gain_min30": macro,
        "outside_core_gain": transfer, "outside_core_rows": len(outside),
        "supported_countries": len(supported),
        "robust_gain": float(.5*frame.delta.mean()+.25*macro+.25*transfer),
        "geographic_gain_positive": bool(frame.delta.mean() > 0 and macro > 0 and
                                          transfer > 0 and len(outside) >= 30)}


def fit_fold(number, split, rows, store, temporary, guard, device):
    directory = temporary / f"fold_{number}"
    directory.mkdir()
    stats = v29.fit_normalization(store, split["training"])
    records, calibrations, role_data = [], [], {r: {} for r in ("calibration", "assessment")}
    for group in ("reference", "replica"):
        sums = {r: np.zeros((len(split[r]), len(store.species_ids)), np.float32) for r in role_data}
        for i, source in enumerate(CONFIGS):
            guard.require(3.5*3600, "start fixed-recipe development fit")
            config = {**source, "seed": source["seed"]+number*1000+(31000 if group == "replica" else 0),
                      "id": f"{group}_{source['id']}"}
            model, record = v29.train_candidate(store, rows, split["training"], split["selection"],
                stats, config, directory, guard, device, fixed_epochs=EPOCHS[i], phase="development")
            p = v29.predict(model, store, split["selection"], stats, device, config, views=2, guard=guard)
            cal = v29.fit_platt(p, np.asarray(store.labels[split["selection"]]))
            for role in role_data:
                p = v29.predict(model, store, split[role], stats, device, config, views=2, guard=guard)
                sums[role] += v29.calibrated(p, cal)/len(CONFIGS)
            records.append({**record, "group": group, "member": i}); calibrations.append(cal)
            del model, p
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for role, p in sums.items():
            if group == "reference":
                role_data[role]["base"] = v29.decode_ensemble([[] for _ in p], p, {"alpha": 1., "count_scale": .8})
            else:
                rank, value = compact_rank(p)
                role_data[role]["neural"] = {"rank": rank, "value": value}
        del sums, p
    expert = HabitatAnalogues().fit(store.train["environment"], store.labels, split["training"])
    for role in role_data:
        role_data[role]["habitat"] = expert.predict_rank(store.train["environment"][split[role]], device, guard)
    trials = {}
    cal_data = role_data["calibration"]
    target = np.asarray(store.labels[split["calibration"]])
    for policy in POLICIES:
        prediction = residual_decode(cal_data["base"], cal_data["neural"], cal_data["habitat"], policy, guard)
        trials[policy["id"]] = legacy.score_prediction_lists(target, prediction)
    guard.stamp("v31_fold_complete", fold=number)
    return {"number": number, "split": split, "data": role_data, "records": records,
            "calibrations": calibrations, "trials": trials}


def select_policy(bundles, rows):
    trials = []
    take = np.concatenate([b["split"]["calibration"] for b in bundles])
    reference = np.concatenate([b["trials"]["control"] for b in bundles])
    for policy in POLICIES:
        scores = np.concatenate([b["trials"][policy["id"]] for b in bundles])
        gains = [float((b["trials"][policy["id"]]-b["trials"]["control"]).mean()) for b in bundles]
        geo = geographic_gain(scores-reference, rows.iloc[take].country, rows.iloc[take].surveyId)
        allowed = policy["id"] == "control" or (min(gains) > 0 and geo["geographic_gain_positive"])
        trials.append({**policy, **geo, "fold_gains": gains, "allowed": bool(allowed),
                       "sample_f1": float(scores.mean())})
    chosen = max((t for t in trials if t["allowed"]), key=lambda t: (t["robust_gain"], -t["swaps"], -t["neural"]-t["habitat"]))
    return next(dict(p) for p in POLICIES if p["id"] == chosen["id"]), trials


def regression_check(bundles, policy, rows, store, guard=None):
    frames, summaries = [], []
    for b in bundles:
        take, data = b["split"]["assessment"], b["data"]["assessment"]
        target = np.asarray(store.labels[take])
        base = data["base"]
        prediction = residual_decode(base, data["neural"], data["habitat"], policy, guard)
        old, new = [legacy.score_prediction_lists(target, x) for x in (base, prediction)]
        ablations = {}
        for expert in ("neural", "habitat"):
            ablated = residual_decode(base, data["neural"], data["habitat"], {**policy, expert: 0.}, guard)
            ablations[f"without_{expert}_gain"] = float((legacy.score_prediction_lists(target, ablated)-old).mean())
        frames.append(pd.DataFrame({"surveyId": rows.iloc[take].surveyId.to_numpy(), "fold": b["number"],
            "country": rows.iloc[take].country.to_numpy(), "spatial_block": legacy.spatial_blocks(rows.iloc[take]),
            "matched_v29_f1": old, "v31_f1": new, "delta_f1": new-old,
            "predicted_cardinality": list(map(len, prediction)),
            "swaps": [len(set(a)-set(c)) for a, c in zip(prediction, base)]}))
        summaries.append({"fold": b["number"], "gain": float((new-old).mean()), "ablations_not_used_for_selection": ablations,
            "multilabel": v29.multilabel_summary(target, prediction),
            "species_groups": legacy.species_group_metrics(target, prediction, legacy._frequency(store.labels, b["split"]["training"]))})
    frame = pd.concat(frames, ignore_index=True)
    if not frame.surveyId.is_unique:
        raise ValueError("Repeated regression survey")
    return frame, {**geographic_gain(frame.delta_f1, frame.country), "folds": summaries,
        "surveys": len(frame), "matched_v29_f1": float(frame.matched_v29_f1.mean()), "v31_f1": float(frame.v31_f1.mean()),
        "bootstrap": legacy.paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy(), iterations=1000, seed=20263107),
        "fresh_assessment": False, "used_for_policy_selection_in_this_run": False,
        "by_country": legacy.summarize_by_group(frame, "country", ("matched_v29_f1", "v31_f1", "delta_f1"))}


def production_estimate(bundles, full_rows):
    return sum(max(np.median([h["seconds"] for h in r["history"]])/len(b["split"]["training"])
        for b in bundles for r in b["records"] if r["group"] == "replica" and r["member"] == i)
        *full_rows*EPOCHS[i]*1.4 for i in range(len(CONFIGS))) + 45*60


def fit_production(bundles, policy, rows, test_rows, store, control, directory, guard, device):
    take = np.arange(len(rows))
    records = []
    neural = habitat = None
    if policy["neural"]:
        guard.require(production_estimate(bundles, len(rows)), "whole full-data replica ensemble admission")
        stats = v29.fit_normalization(store, take)
        probability = np.zeros((len(test_rows), len(store.species_ids)), np.float32)
        for i, source in enumerate(CONFIGS):
            config = {**source, "seed": source["seed"]+31000, "id": f"replica_{source['id']}"}
            model, record = v29.train_candidate(store, rows, take, np.array([], np.int64), stats,
                config, directory, guard, device, fixed_epochs=EPOCHS[i], phase="production")
            p = v29.predict(model, store, np.arange(len(test_rows)), stats, device, config, test=True, views=2, guard=guard)
            probability += v29.calibrated(p, PRODUCTION_CALIBRATIONS[i])/len(CONFIGS)
            records.append({**record, "calibration": PRODUCTION_CALIBRATIONS[i],
                            "calibration_source": "frozen scored v29; transfer between seeds is an explicit assumption"})
            del model, p
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rank, value = compact_rank(probability)
        neural = {"rank": rank, "value": value}
    if policy["habitat"]:
        guard.require(30*60, "full-data habitat fit admission")
        expert = HabitatAnalogues().fit(store.train["environment"], store.labels, take)
        habitat = expert.predict_rank(store.test["environment"], device, guard)
    prediction = residual_decode(control, neural, habitat, policy, guard)
    swaps = np.array([len(set(a)-set(b)) for a, b in zip(prediction, control)])
    return prediction, records, {"training_rows": len(take), "all_PA_rows_used": True,
        "calibration_anchor_rows_removed": 0, "cardinality_equal_to_scored_v29_per_row": True,
        "production_epochs_frozen_from_scored_v29": EPOCHS, "changed_rows": int((swaps>0).sum()),
        "mean_swaps": float(swaps.mean()), "maximum_swaps": int(swaps.max()),
        "habitat_metric": "official environmental values only; train-only PCA; no geography or IDs"}


def decode_control(payload, template, species):
    # Keep the frozen decoder's globals isolated from notebook source globals.
    previous.CONTROL_PAYLOAD_HASH, previous.CONTROL_RAW_HASH = CONTROL_PAYLOAD_HASH, CONTROL_RAW_HASH
    return previous.decode_control(payload, template, species)


def publish(export, template, ids, predictions, species, gate):
    tentative = export / "candidate_DO_NOT_SUBMIT.csv"
    proof = legacy.write_submission(tentative, template, ids, predictions, species)
    differs = proof["sha256"] != CONTROL_HASH
    eligible = bool(gate and differs)
    name = "GLC25_PA_submission_v31.csv" if eligible else (tentative.name if differs else "unchanged_v29_DO_NOT_SUBMIT.csv")
    if name != tentative.name:
        tentative.replace(export/name)
    return proof, {"eligible_for_submission": eligible, "different_from_v29": differs, "prediction_file": name,
        "message": "SUBMIT ONLY THIS CSV" if eligible else "DO NOT SUBMIT THIS OUTPUT; keep the scored v29"}


def save_compact(path, bundles, rows, store, per_fold=1500):
    ids, folds, countries, bases, ranks, values, distances, truth = [], [], [], [], [], [], [], []
    for b in bundles:
        take = b["split"]["calibration"]
        positions = np.sort(np.random.default_rng(20263108+b["number"]).choice(len(take), min(per_fold, len(take)),
            replace=False, p=v29.sample_weights(rows, take)))
        selected, data = take[positions], b["data"]["calibration"]
        base = np.full((len(positions), 40), 65535, np.uint16)
        for j, pos in enumerate(positions):
            base[j, :len(data["base"][pos])] = data["base"][pos]
        bases.append(base); ids.append(rows.iloc[selected].surveyId.to_numpy()); folds.append(np.full(len(selected), b["number"], np.uint8))
        countries.append(rows.iloc[selected].country.fillna("unknown").to_numpy(dtype="U64"))
        ranks.append(np.stack([data[k]["rank"][positions] for k in ("neural", "habitat")], axis=1))
        values.append(np.stack([data[k]["value"][positions] for k in ("neural", "habitat")], axis=1))
        distances.append(data["habitat"]["distance"][positions])
        truth.extend(np.flatnonzero(store.labels[j]).astype(np.uint16) for j in selected)
    np.savez_compressed(path, survey_id=np.concatenate(ids), fold=np.concatenate(folds), country=np.concatenate(countries),
        reference_columns_padded65535=np.concatenate(bases), ranked_species_columns=np.concatenate(ranks),
        probability=np.concatenate(values), habitat_distance=np.concatenate(distances),
        true_species_columns=np.concatenate(truth), true_offsets=np.r_[0, np.cumsum(list(map(len, truth)))].astype(np.uint32),
        species_ids=store.species_ids, expert_ids=np.array(["seed_replica", "environmental_analogues"]),
        evidence_scope=np.array("sampled previously consumed calibration; truncated top128, not full probabilities or a fresh audit"))


def self_tests():
    assert len(POLICIES) == 15
    base = [list(range(10))]
    expert = {"rank": np.arange(30, 10, -1)[None], "value": np.ones((1, 20))}
    assert residual_decode(base, None, None, POLICIES[0]) == base
    result = residual_decode(base, expert, expert, {"neural": 1., "habitat": .5, "swaps": 2})
    validate_residual(base, result, {"swaps": 2})
    assert np.allclose(HabitatAnalogues.neighbour_weights(np.array([[0., 1., 2.]] )).sum(1), 1)
    assert not geographic_gain(np.r_[np.ones(100), -np.ones(30)], ["Denmark"]*100+["France"]*30)["geographic_gain_positive"]
    return {"passed": True, "tests": 5}


def run_v31(control_b64):
    guard = legacy.RuntimeGuard(MAX_HOURS)
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
    temporary, export = working/"v31_runtime", working/"v31_export"
    legacy._clean_directory(temporary, working); legacy._clean_directory(export, working)
    temporary.mkdir(parents=True); export.mkdir(parents=True)
    store = None
    try:
        device = legacy.require_gpu()  # Fail before opening the large official predictors.
        torch.set_num_threads(min(os.cpu_count() or 2, 6))
        tests = self_tests()
        root = legacy.discover_data_root()
        preflight = pd.read_csv(root/"GLC25_PA_metadata_train.csv", usecols=["surveyId", "lat", "lon", "country"]).drop_duplicates("surveyId").reset_index(drop=True)
        _, manifests = previous.make_splits(preflight)
        guard.stamp("preflight", folds=manifests, fresh_assessment=False, epochs=EPOCHS)
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
        splits, manifests = previous.make_splits(rows)
        template = pd.read_csv(root/"GLC25_SAMPLE_SUBMISSION.csv")
        control = decode_control(control_b64, template, store.species_ids)
        order = pd.Index(template.surveyId).get_indexer(store.test_ids)
        if (order < 0).any() or not template.surveyId.is_unique:
            raise ValueError("Test/template IDs mismatch")
        control = [control[i] for i in order]
        proof = legacy.write_submission(temporary/"control.csv", template, store.test_ids, control, store.species_ids)
        if proof["sha256"] != CONTROL_HASH:
            raise ValueError("Exact scored v29 round trip failed")
        bundles = [fit_fold(i, split, rows, store, temporary, guard, device) for i, split in enumerate(splits)]
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
        validate_residual(control, prediction, policy)
        gates["within_budget"] = guard.elapsed_hours() < MAX_HOURS
        submission, decision = publish(export, template, store.test_ids, prediction, store.species_ids, all(gates.values()))
        frame.to_csv(export/"regression_per_survey_v31.csv", index=False)
        save_compact(export/"calibration_top128_v31.npz", bundles, rows, store)
        report = {"experiment": EXPERIMENT, "status": "complete", "runtime_hours": guard.elapsed_hours(),
            "official_submission_made": False, "official_public_score": None, "official_private_score": None,
            "control_scores": {"public": .23900, "private": .21093}, "private_target": .23021,
            "selected_policy": policy, "policy_trials": trials, "regression_check": regression,
            "submission_gate": {**gates, **decision}, "submission": submission, "self_tests": tests,
            "training": {"development": [{"fold": b["number"], "members": b["records"], "calibrations": b["calibrations"]} for b in bundles],
                         "production": fitted}, "production_diagnostics": diagnostics,
            "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__},
            "validation_status": "all 88,987 PA IDs previously assessed; repeated spatial development, no fresh holdout",
            "limitations": ["No guarantee of SOTA, leaderboard gain or full completion on an unmeasured GPU session.",
                "Geographic gates were added after examining v30: they are development choices, not independent audit evidence.",
                "Matched v29 is a fixed-epoch recipe refit, not the original model weights; only the test CSV is exact.",
                "Production uses frozen v29 probability calibration on new seeds; transfer is an assumption.",
                "Conservative tail swaps may reject useful but larger changes; all row counts intentionally stay at v29."]}
        legacy.save_json(export/"v31_report.json", report)
        manifest = {"experiment": EXPERIMENT, "source_sha256": V31_SOURCE_HASH, "embedded_v30_sha256": V30_SOURCE_HASH,
            "embedded_v29_sha256": V29_SOURCE_HASH, "embedded_legacy_sha256": LEGACY_SOURCE_HASH,
            "splits": manifests, "features": features, "configs": CONFIGS, "fixed_epochs": EPOCHS, "policies": POLICIES,
            "control_sha256": CONTROL_HASH, "runtime_cap_hours": MAX_HOURS, "no_external_data_or_weights": True,
            "fresh_assessment": False, "outputs": {p.name: legacy.sha256_file(p) for p in sorted(export.iterdir())}}
        legacy.save_json(export/"v31_manifest.json", manifest)
        size = sum(p.stat().st_size for p in export.iterdir())
        if len(list(export.iterdir())) != 5 or size > 16_000_000:
            raise ValueError("Compact five-file/16MB export contract exceeded")
        return {"status": "complete", **decision, "runtime_hours": guard.elapsed_hours(), "output_bytes": size,
            "export_directory": str(export), "regression_gain": regression["gain"], "fresh_assessment": False, "policy": policy}
    except Exception as error:
        # A failure after CSV writing must not leave a ready-to-submit filename.
        ready = export/"GLC25_PA_submission_v31.csv"
        if ready.exists():
            ready.replace(export/"failed_DO_NOT_SUBMIT.csv")
        legacy.save_json(export/"failure_report.json", {"experiment": EXPERIMENT, "status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "runtime_hours": guard.elapsed_hours(), "official_submission_made": False,
            "eligible_for_submission": False})
        raise
    finally:
        del store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        legacy._clean_directory(temporary, working)
