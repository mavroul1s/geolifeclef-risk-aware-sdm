"""v30: calibrated multiscale attention; repeated spatial validation, not a fresh audit."""
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
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import HistGradientBoostingRegressor

import scripts.v29_notebook_core as previous

legacy = previous.legacy
EXPERIMENT = "v30_calibrated_multiscale_attention"
CONTROL_HASH = "9ec7afb25eb1651b161b0f8c6ecaa64fd553d17dd24b4808b1f07446b85f8194"
MAX_HOURS = 10.75
SPLIT_SEED = 20260926
CONFIGS = tuple(dict(c) for c in previous.CONFIGS) + (
    {"id": "multiscale_ecology", "kind": "multiscale", "geo": False, "width": 192,
     "gamma": 0.0, "epochs": 48, "seed": 20263004},)
# The weak convolutional member remains in the matched v29 reference, not candidates.
WEIGHTS = {"attention_pair": (0.5, 0., 0.5, 0.),
           "balanced": (1/3, 0., 1/3, 1/3),
           "geo_lean": (0.6, 0., 0.2, 0.2),
           "ecology_lean": (0.2, 0., 0.4, 0.4)}
POLICIES = ({"id": "control", "weights": "attention_pair", "learned_count": 0., "alpha": 0.},) + tuple(
    {"id": f"{name}_count{int(count*100)}_a{int(alpha*100)}", "weights": name,
     "learned_count": count, "alpha": alpha}
    for name in WEIGHTS for count in (0., 0.5, 1.) for alpha in (0.75, 1.))
ORIGINAL_FACTORY = previous.create_model


def registered_roles(rows):
    """Greedy label-blind size/country balancing of indivisible one-degree blocks."""
    blocks = legacy.spatial_blocks(rows)
    countries = rows.country.fillna("unknown").astype(str).to_numpy()
    names, country_index = np.unique(countries, return_inverse=True)
    unique = np.unique(blocks)
    vectors = {b: np.bincount(country_index[blocks == b], minlength=len(names)) for b in unique}
    fractions = np.array([.18, .18, .10, .10, .44])
    total = np.bincount(country_index, minlength=len(names))
    assigned = np.zeros((5, len(names)), np.int64)
    allocation = {}
    order = sorted(unique, key=lambda b: (-int(vectors[b].sum()),
        hashlib.sha256(f"v30:{SPLIT_SEED}:{b}".encode()).hexdigest()))
    for block in order:
        v = vectors[block]
        # Squared normalized allocation cost, weighted by the block's country mix.
        global_cost = (assigned.sum(1) + v.sum()) / (len(rows) * fractions)
        country_cost = (((assigned + v) / np.maximum(fractions[:, None] * total, 1)) *
                        (v / v.sum())[None]).sum(1)
        role = int(np.argmin(.5 * global_cost + .5 * country_cost))
        assigned[role] += v
        allocation[block] = role
    return np.array([allocation[b] for b in blocks]), blocks


def validate_split(rows, split, *, production=False, minimums=None):
    minimums = minimums or ({"training": 40000, "anchor": 1000} if production else
                            {"training": 18000, "selection": 1000, "calibration": 1000, "assessment": 1000})
    if any(len(split[name]) < n for name, n in minimums.items()):
        raise ValueError(f"Registered v30 partitions too small: { {k:len(v) for k,v in split.items()} }")
    blocks = legacy.spatial_blocks(rows)
    roles = list(split)
    for i, a in enumerate(roles):
        for b in roles[i+1:]:
            if np.intersect1d(split[a], split[b]).size or set(blocks[split[a]]) & set(blocks[split[b]]):
                raise ValueError("Shared IDs or blocks across v30 roles")
    evaluation = np.concatenate([v for k, v in split.items() if k != "training"])
    xy = rows[["lat", "lon"]].to_numpy(np.float64)
    distance = legacy.nearest_distance_km(xy[split["training"]], xy[evaluation]).min()
    if distance < 20:
        raise ValueError("v30 geographic buffer failed")
    return {"counts": {k: len(v) for k, v in split.items()},
            "blocks": {k: len(np.unique(blocks[v])) for k, v in split.items()},
            "id_hashes": {k: legacy.sha256_bytes(rows.surveyId.to_numpy(np.int64)[v].astype('<i8').tobytes())
                          for k, v in split.items()}, "minimum_distance_km": float(distance),
            "labels_used_for_assignment": False, "fresh_assessment": False,
            "protocol": "previously_consumed_spatial_development_regression_check",
            "seed": SPLIT_SEED}


def make_splits(rows, *, minimums=None):
    roles, _ = registered_roles(rows)
    xy = rows[["lat", "lon"]].to_numpy(np.float64)
    splits, manifests = [], []
    for fold in range(2):
        split = {"selection": np.flatnonzero(roles == 2), "calibration": np.flatnonzero(roles == 3),
                 "assessment": np.flatnonzero(roles == fold)}
        held = np.concatenate(list(split.values()))
        train = np.flatnonzero(~np.isin(roles, [2, 3, fold]))
        split["training"] = train[legacy.nearest_distance_km(xy[held], xy[train]) >= 20]
        manifests.append(validate_split(rows, split, minimums=minimums))
        splits.append(split)
    if np.intersect1d(splits[0]["assessment"], splits[1]["assessment"]).size:
        raise ValueError("Outer regression partitions overlap")
    return splits, manifests


def production_split(rows, *, minimums=None):
    blocks = legacy.spatial_blocks(rows)
    bucket = np.array([legacy.stable_bucket(f"v30-production:{SPLIT_SEED}:{b}") for b in blocks])
    anchor, train = np.flatnonzero(bucket < 8), np.flatnonzero(bucket >= 8)
    if not len(anchor) or not len(train):
        raise ValueError("Empty production split")
    xy = rows[["lat", "lon"]].to_numpy(np.float64)
    train = train[legacy.nearest_distance_km(xy[anchor], xy[train]) >= 20]
    split = {"training": train, "anchor": anchor}
    return split, validate_split(rows, split, production=True, minimums=minimums)


class MultiscaleAttention(previous.SensorAttention):
    """Shared CNN for full landscape and central habitat; direct tabular skip paths."""
    def __init__(self, env_dim, static_dim, species, width=192):
        super().__init__(env_dim, static_dim, species, width)
        self.position = nn.Parameter(torch.randn(1, 171, width) * .02)
        self.norm = nn.LayerNorm(4 * width)
        self.head = nn.Linear(4 * width, species)
        self.richness = nn.Linear(4 * width, 1)

    def forward(self, x, geo=False):
        image = self.image(x["sentinel"]).flatten(2).transpose(1, 2)
        center = F.interpolate(x["sentinel"][:, :, 16:48, 16:48], size=(64, 64),
                               mode="bilinear", align_corners=False)
        center = self.image(center).flatten(2).transpose(1, 2)
        land = self.land(x["landsat"].permute(0, 3, 1, 2).flatten(2))
        climate = self.climate(x["bioclim"].permute(0, 2, 1, 3).flatten(2))
        env = self.environment(x["environment"])[:, None]
        static = self.static(x["static_geo" if geo else "static_eco"])[:, None]
        if self.training:
            image, center, land, climate = [s * torch.bernoulli(s.new_full((len(s), 1, 1), .9))
                                             for s in (image, center, land, climate)]
        tokens = torch.cat([self.cls.expand(len(image), -1, -1), image, center, land, climate, env, static], 1)
        encoded = self.encoder(tokens + self.position)
        feature = self.norm(torch.cat([encoded[:, 0], encoded[:, 1:].mean(1), env[:, 0], static[:, 0]], 1))
        return self.head(feature), self.richness(feature).squeeze(1)


def create_model(store, config, indices):
    if config["kind"] != "multiscale":
        return ORIGINAL_FACTORY(store, config, indices)
    legacy.set_seed(config["seed"])
    model = MultiscaleAttention(store.dims["environment"],
        store.static_train.shape[1] if config["geo"] else store.eco_train.shape[1],
        len(store.species_ids), config["width"])
    frequencies = legacy._frequency(store.labels, indices)
    prior = np.clip((frequencies + .5) / (len(indices) + 1), 1e-5, .95)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor(np.log(prior / (1-prior)), dtype=torch.float32))
        model.richness.bias.fill_(math.log1p(frequencies.sum() / max(len(indices), 1)))
    return model


def train_model(*args, **kwargs):
    saved = previous.create_model
    try:
        previous.create_model = create_model
        return previous.train_candidate(*args, **kwargs)
    finally:
        previous.create_model = saved


def combine(arrays, weights):
    result = np.zeros(arrays[0].shape, np.float32)
    for probability, weight in zip(arrays, weights):
        if weight:
            result += np.asarray(probability, np.float32) * weight
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite ensemble probabilities")
    return result


def count_features(probability, rows, distance):
    _, values = legacy.top_rank(probability, min(40, probability.shape[1]))
    values = np.asarray(values, np.float32)
    summaries = [probability.sum(1), values[:, 0], values.mean(1), values.std(1)]
    summaries += [values[:, :min(k, values.shape[1])].sum(1) for k in (5, 10, 20, 40)]
    summaries += [np.log1p(distance)]
    # No country target averages or species labels in inference features.
    return np.column_stack([*summaries, previous.candidate_static(rows, False)]).astype(np.float32)


def fit_count(probability, targets, rows, distance):
    maximum = min(40, probability.shape[1])
    oracle = legacy.oracle_f1_counts(probability, targets, minimum=min(8, maximum), maximum=maximum)
    model = HistGradientBoostingRegressor(loss="squared_error", max_iter=120,
        max_leaf_nodes=12, min_samples_leaf=60, learning_rate=.05, l2_regularization=10.,
        early_stopping=False, random_state=20263006)
    model.fit(count_features(probability, rows, distance), oracle,
              sample_weight=previous.sample_weights(rows.reset_index(drop=True), np.arange(len(rows))) * len(rows))
    return model, {"fit_rows": len(rows), "target": "oracle top-k on model-unseen calibration-fit rows",
                   "oracle_mean": float(oracle.mean()), "fitted_on_assessment": False,
                   "features_exclude_country_and_label_statistics": True}


def count_values(probability, rows, distance, model):
    ranked, values = legacy.top_rank(probability, min(64, probability.shape[1]))
    maximum, minimum = min(40, probability.shape[1]), min(8, probability.shape[1])
    k = np.arange(1, maximum+1)
    surrogate = 2 * np.cumsum(values[:, :maximum], axis=1) / (
        k[None] + .8 * probability.sum(1)[:, None] + 1e-8)
    original = surrogate[:, minimum-1:].argmax(1) + minimum
    learned = np.clip(model.predict(count_features(probability, rows, distance)), minimum, maximum)
    return ranked, original, learned


def decode(base, ranked, original_count, learned_count, policy):
    alpha = policy["alpha"]
    if not alpha:
        return [list(row) for row in base]
    minimum, maximum = min(8, ranked.shape[1]), min(40, ranked.shape[1])
    weight = policy["learned_count"]
    counts = np.clip(np.rint((1-weight)*original_count + weight*learned_count), minimum, maximum).astype(int)
    if alpha == 1:
        return [ranked[i, :k].astype(int).tolist() for i, k in enumerate(counts)]
    result = []
    for i, old in enumerate(base):
        scores = {int(s): (1-alpha)*(1-.7*j/max(len(old)-1, 1)) for j,s in enumerate(old)}
        for j, s in enumerate(ranked[i]):
            scores[int(s)] = scores.get(int(s), 0.) + alpha*(1-.85*j/max(ranked.shape[1]-1, 1))
        count = int(np.clip(round((1-alpha)*len(old)+alpha*counts[i]), minimum, maximum))
        result.append([s for s,_ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:count]])
    return result


def fold_reference(arrays):
    probability = combine(arrays, (1/3, 1/3, 1/3, 0.))
    return previous.decode_ensemble([[] for _ in probability], probability, {"alpha": 1., "count_scale": .8})


def fit_fold(number, split, rows, store, temporary, guard, device):
    directory = temporary / f"fold_{number}"
    directory.mkdir()
    stats = previous.fit_normalization(store, split["training"])
    records, calibrations = [], []
    for i, config in enumerate(CONFIGS):
        remaining_fits = (2-number)*len(CONFIGS)-i
        budget = max(120., (guard.remaining_seconds()-3.5*3600) / remaining_fits)
        guard.require(3.75*3600, "start development member")
        config = {**config, "seed": config["seed"] + number*1000}
        model, record = train_model(store, rows, split["training"], split["selection"], stats,
            config, directory, guard, device, training_budget_seconds=budget)
        p = previous.predict(model, store, split["selection"], stats, device, config, views=2, guard=guard)
        calibration = previous.fit_platt(p, np.asarray(store.labels[split["selection"]]))
        for role in ("selection", "calibration", "assessment"):
            if role != "selection":
                p = previous.predict(model, store, split[role], stats, device, config, views=2, guard=guard)
            calibrated = previous.calibrated(p, calibration)
            np.save(directory / f"{role}_{i}.npy", calibrated.astype(np.float16), allow_pickle=False)
            if role == "calibration":
                lists = previous.decode_ensemble([[] for _ in p], calibrated, {"alpha": 1., "count_scale": .8})
                record["calibration_standalone_f1"] = float(legacy.score_prediction_lists(
                    np.asarray(store.labels[split[role]]), lists).mean())
                record["calibration_by_country"] = legacy.summarize_by_group(pd.DataFrame({
                    "country": rows.iloc[split[role]].country.to_numpy(), "f1": legacy.score_prediction_lists(
                        np.asarray(store.labels[split[role]]), lists)}), "country", ("f1",))
            del calibrated
        records.append(record)
        calibrations.append(calibration)
        del model, p
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    def load(role):
        return [np.load(directory / f"{role}_{i}.npy", mmap_mode="r") for i in range(len(CONFIGS))]
    arrays_selection, arrays_calibration = load("selection"), load("calibration")
    references = {role: fold_reference(load(role)) for role in ("calibration", "assessment")}
    xy = rows[["lat", "lon"]].to_numpy()
    distances = {role: legacy.nearest_distance_km(xy[split["training"]], xy[split[role]])
                 for role in ("selection", "calibration", "assessment")}
    counts, count_records, decoded, trials = {}, {}, {}, {}
    for name, weights in WEIGHTS.items():
        p = combine(arrays_selection, weights)
        counts[name], count_records[name] = fit_count(p, np.asarray(store.labels[split["selection"]]),
            rows.iloc[split["selection"]], distances["selection"])
        p = combine(arrays_calibration, weights)
        decoded[name] = count_values(p, rows.iloc[split["calibration"]], distances["calibration"], counts[name])
    target = np.asarray(store.labels[split["calibration"]])
    for policy in POLICIES:
        prediction = decode(references["calibration"], *decoded[policy["weights"]], policy)
        trials[policy["id"]] = legacy.score_prediction_lists(target, prediction)
    return {"number": number, "directory": directory, "split": split, "references": references,
            "distance": distances, "count_models": counts, "trials": trials,
            "records": records, "calibrations": calibrations, "count_records": count_records}


def select_policy(bundles, rows):
    trials = []
    for policy in POLICIES:
        scores = np.concatenate([b["trials"][policy["id"]] for b in bundles])
        reference = np.concatenate([b["trials"]["control"] for b in bundles])
        selected_rows = pd.concat([rows.iloc[b["split"]["calibration"]] for b in bundles])
        delta = scores-reference
        frame = pd.DataFrame({"country": selected_rows.country.to_numpy(),
                              "block": legacy.spatial_blocks(selected_rows), "delta": delta})
        country = frame.groupby("country").delta.agg(['mean', 'size'])
        country = country[country['size'] >= 60]
        robust = .6*delta.mean() + .2*(country['mean'].mean() if len(country) else 0.) + .2*frame.groupby('block').delta.mean().mean()
        fold_gains = [float((b["trials"][policy["id"]]-b["trials"]["control"]).mean()) for b in bundles]
        allowed = policy["alpha"] == 0 or (min(fold_gains) > 0 and robust > 0)
        trials.append({**policy, "sample_f1": float(scores.mean()), "gain": float(delta.mean()),
                       "robust_gain": float(robust), "fold_gains": fold_gains, "allowed": bool(allowed)})
    chosen = max((t for t in trials if t["allowed"]), key=lambda x: (x["robust_gain"], -x["alpha"], -x["learned_count"]))
    return next(dict(p) for p in POLICIES if p["id"] == chosen["id"]), trials


def regression_check(bundles, policy, rows, store):
    frames, summaries = [], []
    for bundle in bundles:
        take = bundle["split"]["assessment"]
        arrays = [np.load(bundle["directory"] / f"assessment_{i}.npy", mmap_mode="r") for i in range(len(CONFIGS))]
        probability = combine(arrays, WEIGHTS[policy["weights"]])
        ranked, original, learned = count_values(probability, rows.iloc[take], bundle["distance"]["assessment"],
                                               bundle["count_models"][policy["weights"]])
        base = bundle["references"]["assessment"]
        prediction = decode(base, ranked, original, learned, policy)
        target = np.asarray(store.labels[take])
        old, new = legacy.score_prediction_lists(target, base), legacy.score_prediction_lists(target, prediction)
        frames.append(pd.DataFrame({"surveyId": rows.iloc[take].surveyId.to_numpy(), "fold": bundle["number"],
            "country": rows.iloc[take].country.to_numpy(), "spatial_block": legacy.spatial_blocks(rows.iloc[take]),
            "matched_v29_f1": old, "v30_f1": new, "delta_f1": new-old,
            "true_cardinality": target.sum(1), "predicted_cardinality": list(map(len, prediction))}))
        frequency = legacy._frequency(store.labels, bundle["split"]["training"])
        summaries.append({"fold": bundle["number"], "surveys": len(take), "gain": float((new-old).mean()),
            "v30_multilabel": previous.multilabel_summary(target, prediction),
            "v30_species_groups": legacy.species_group_metrics(target, prediction, frequency)})
    frame = pd.concat(frames, ignore_index=True)
    if not frame.surveyId.is_unique:
        raise ValueError("Repeated survey in outer regression scores")
    bootstrap = legacy.paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy(),
                                             iterations=1000, seed=20263007)
    return frame, {"surveys": len(frame), "matched_v29_f1": float(frame.matched_v29_f1.mean()),
        "v30_f1": float(frame.v30_f1.mean()), "gain": float(frame.delta_f1.mean()),
        "folds": summaries, "bootstrap": bootstrap, "fresh_assessment": False,
        "all_labels_previously_consumed": True, "used_for_policy_selection_in_this_run": False,
        "by_country": legacy.summarize_by_group(frame, "country", ("matched_v29_f1", "v30_f1", "delta_f1")),
        "warning": "Repeated spatial development regression check, not untouched assessment or SOTA evidence."}


def fit_production(bundles, policy, rows, test_rows, store, control, split, directory, guard, device):
    weights = WEIGHTS[policy["weights"]]
    active = [i for i, w in enumerate(weights) if w]
    epochs = {i: max(1, int(np.median([b["records"][i]["best_epoch"] for b in bundles]))) for i in active}
    estimates = {i: max(np.median([h["seconds"] for h in b["records"][i]["history"]]) /
                       len(b["split"]["training"]) for b in bundles) * len(split["training"]) * epochs[i] * 1.4
                 for i in active}
    guard.require(sum(estimates.values()) + 45*60, "whole calibrated production ensemble admission")
    stats = previous.fit_normalization(store, split["training"])
    anchor_p = np.zeros((len(split["anchor"]), len(store.species_ids)), np.float32)
    test_p = np.zeros((len(test_rows), len(store.species_ids)), np.float32)
    records = []
    for position, i in enumerate(active):
        guard.require(sum(estimates[j] for j in active[position:]) + 35*60, "remaining production ensemble admission")
        config = {**CONFIGS[i], "seed": CONFIGS[i]["seed"]+2000}
        model, record = train_model(store, rows, split["training"], np.array([], np.int64), stats,
            config, directory, guard, device, fixed_epochs=epochs[i], phase="production")
        p = previous.predict(model, store, split["anchor"], stats, device, config, views=2, guard=guard)
        # Unlike v29, these probabilities come from the exact production weights,
        # and their calibration labels were never used in fitting those weights.
        calibration = previous.fit_platt(p, np.asarray(store.labels[split["anchor"]]))
        anchor_p += weights[i] * previous.calibrated(p, calibration)
        p = previous.predict(model, store, np.arange(len(test_rows)), stats, device, config,
                             views=2, test=True, guard=guard)
        test_p += weights[i] * previous.calibrated(p, calibration)
        records.append({**record, "anchor_calibration": calibration, "calibration_transferred_from_development": False})
        del model, p
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    xy = rows[["lat", "lon"]].to_numpy()
    anchor_distance = legacy.nearest_distance_km(xy[split["training"]], xy[split["anchor"]])
    test_distance = legacy.nearest_distance_km(xy[split["training"]], test_rows[["lat", "lon"]].to_numpy())
    count, count_record = fit_count(anchor_p, np.asarray(store.labels[split["anchor"]]),
                                    rows.iloc[split["anchor"]], anchor_distance)
    prediction = decode(control, *count_values(test_p, test_rows, test_distance, count), policy)
    diagnostic = {"probability_mass_mean": float(test_p.sum(1).mean()),
                  "predicted_count_mean": float(np.mean(list(map(len, prediction)))),
                  "anchor_probability_mass_mean": float(anchor_p.sum(1).mean()),
                  "count_fit": count_record, "epoch_counts_locked_from_development": epochs,
                  "training_rows": len(split["training"]), "calibration_rows": len(split["anchor"]),
                  "anchor_not_in_weight_training": not np.intersect1d(split["anchor"], split["training"]).size}
    return prediction, records, diagnostic


def save_compact_diagnostics(path, bundles, rows, store, per_fold=2000):
    """Small reusable calibration evidence, not checkpoints or full probability matrices."""
    all_ids, all_folds, all_countries, all_rank, all_probability, all_mass, truth = [], [], [], [], [], [], []
    for bundle in bundles:
        take = bundle["split"]["calibration"]
        weights = previous.sample_weights(rows, take)
        rng = np.random.default_rng(20263008+bundle["number"])
        positions = np.sort(rng.choice(len(take), min(per_fold, len(take)), replace=False, p=weights))
        selected = take[positions]
        rank, probability, mass = [], [], []
        for i in range(len(CONFIGS)):
            p = np.asarray(np.load(bundle["directory"] / f"calibration_{i}.npy", mmap_mode="r")[positions], np.float32)
            ranked, values = legacy.top_rank(p, min(64, p.shape[1]))
            rank.append(ranked.astype(np.uint16)); probability.append(values.astype(np.float16)); mass.append(p.sum(1))
        all_rank.append(np.stack(rank, axis=1)); all_probability.append(np.stack(probability, axis=1))
        all_mass.append(np.stack(mass, axis=1)); all_ids.append(rows.surveyId.to_numpy()[selected])
        all_folds.append(np.full(len(selected), bundle["number"], np.uint8))
        all_countries.append(rows.iloc[selected].country.fillna('unknown').to_numpy(dtype='U64'))
        truth.extend(np.flatnonzero(store.labels[i]).astype(np.uint16) for i in selected)
    counts = np.array(list(map(len, truth)), np.uint16)
    np.savez_compressed(path, survey_id=np.concatenate(all_ids), fold=np.concatenate(all_folds),
        country=np.concatenate(all_countries), ranked_species_columns=np.concatenate(all_rank),
        calibrated_probability=np.concatenate(all_probability), total_probability_mass=np.concatenate(all_mass),
        true_species_columns=np.concatenate(truth), true_offsets=np.r_[0, np.cumsum(counts)].astype(np.uint32),
        species_ids=store.species_ids, model_ids=np.array([c['id'] for c in CONFIGS]),
        evidence_scope=np.array('previously consumed calibration only; top64 is truncated, not full probabilities'))


def decode_control(payload, template, species):
    packed = base64.b64decode(payload)
    if legacy.sha256_bytes(packed) != CONTROL_PAYLOAD_HASH:
        raise ValueError("v29 control payload mismatch")
    raw = lzma.decompress(packed)
    if legacy.sha256_bytes(raw) != CONTROL_RAW_HASH:
        raise ValueError("v29 control raw hash mismatch")
    n = len(template)
    counts, flat = np.frombuffer(raw[:n], np.uint8), np.frombuffer(raw[n:], '<u2')
    if n != legacy.EXPECTED_TEST_ROWS or len(species) != 5016 or counts.sum() != len(flat) or counts.min() < 8 or counts.max() > 40 or flat.max() >= len(species):
        raise ValueError("Invalid frozen v29 control dimensions")
    offsets = np.r_[0, np.cumsum(counts)]
    result = [flat[a:b].astype(int).tolist() for a,b in zip(offsets[:-1], offsets[1:])]
    if any(len(x) != len(set(x)) for x in result):
        raise ValueError("Duplicate frozen species")
    return result


def publish(export, template, ids, predictions, species, gate):
    tentative = export / 'candidate_DO_NOT_SUBMIT.csv'
    proof = legacy.write_submission(tentative, template, ids, predictions, species)
    differs = proof['sha256'] != CONTROL_HASH
    eligible = bool(gate and differs)
    name = 'GLC25_PA_submission_v30.csv' if eligible else ('candidate_DO_NOT_SUBMIT.csv' if differs else 'unchanged_v29_DO_NOT_SUBMIT.csv')
    if tentative.name != name:
        tentative.replace(export / name)
    return proof, {'eligible_for_submission': eligible, 'prediction_file': name, 'different_from_v29': differs,
                   'message': 'SUBMIT ONLY THIS CSV' if eligible else 'DO NOT SUBMIT THIS OUTPUT'}


def self_tests():
    assert len(POLICIES) == 25 and all(abs(sum(w)-1) < 1e-6 for w in WEIGHTS.values())
    base = [list(range(8))]
    assert decode(base, np.arange(12)[None], np.array([8]), np.array([10]), POLICIES[0]) == base
    model = MultiscaleAttention(5, 2, 12, width=16).eval()
    with torch.no_grad():
        result, count = model({'environment': torch.zeros(2,5), 'static_eco': torch.zeros(2,2),
            'sentinel': torch.zeros(2,6,64,64), 'landsat': torch.zeros(2,6,4,21), 'bioclim': torch.zeros(2,4,19,12)})
    assert result.shape == (2,12) and count.shape == (2,) and torch.isfinite(result).all()
    return {'passed': True, 'tests': 4}


def run_v30(control_b64):
    guard = legacy.RuntimeGuard(MAX_HOURS)
    working = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('artifacts')
    temporary, export = working / 'v30_runtime', working / 'v30_export'
    legacy._clean_directory(temporary, working); legacy._clean_directory(export, working)
    temporary.mkdir(parents=True); export.mkdir(parents=True)
    store = None
    try:
        device = legacy.require_gpu()
        torch.set_num_threads(min(os.cpu_count() or 2, 6))
        tests = self_tests()
        root = legacy.discover_data_root()
        preflight = pd.read_csv(root / 'GLC25_PA_metadata_train.csv', usecols=['surveyId','lat','lon','country']).drop_duplicates('surveyId').reset_index(drop=True)
        _, manifests = make_splits(preflight)
        _, production_manifest = production_split(preflight)
        guard.stamp('preflight', folds=manifests, production=production_manifest, fresh_assessment=False)
        original_writer = legacy._write_remote_arrays
        try:
            legacy._write_remote_arrays = previous.write_multiresolution
            features = legacy.prepare_feature_store(root, temporary / 'features', guard, workers=6)
        finally:
            legacy._write_remote_arrays = original_writer
        store = legacy.FeatureStore(temporary / 'features')
        store.high_train = np.load(store.cache / 'train_sentinel64.npy', mmap_mode='r')
        store.high_test = np.load(store.cache / 'test_sentinel64.npy', mmap_mode='r')
        rows, test_rows, pairs = legacy.load_rows_and_pairs(root, store.train_ids, store.test_ids)
        del pairs, preflight
        store.static_train, store.static_test = previous.candidate_static(rows), previous.candidate_static(test_rows)
        store.eco_train, store.eco_test = previous.candidate_static(rows, False), previous.candidate_static(test_rows, False)
        splits, manifests = make_splits(rows)
        production, production_manifest = production_split(rows)
        template = pd.read_csv(root / 'GLC25_SAMPLE_SUBMISSION.csv')
        control = decode_control(control_b64, template, store.species_ids)
        order = pd.Index(template.surveyId).get_indexer(store.test_ids)
        if (order < 0).any() or not template.surveyId.is_unique:
            raise ValueError('Test/template IDs mismatch')
        control = [control[i] for i in order]
        proof = legacy.write_submission(temporary / 'control.csv', template, store.test_ids, control, store.species_ids)
        if proof['sha256'] != CONTROL_HASH:
            raise ValueError('Exact v29 control round trip failed')
        bundles = [fit_fold(i, split, rows, store, temporary, guard, device) for i,split in enumerate(splits)]
        policy, trials = select_policy(bundles, rows)
        legacy.save_json(temporary / 'frozen_policy.json', {'policy':policy, 'trials':trials})
        guard.stamp('policy_frozen', policy=policy)
        frame, regression = regression_check(bundles, policy, rows, store)
        gates = {'selected_new_method': policy['alpha'] > 0,
                 'positive_each_regression_fold': all(f['gain'] > 0 for f in regression['folds']),
                 'spatial_ci_positive': regression['bootstrap']['ci95'][0] > 0,
                 'exact_frozen_control': proof['sha256'] == CONTROL_HASH,
                 'geographic_buffers': min(m['minimum_distance_km'] for m in manifests+[production_manifest]) >= 20}
        fitted, production_diagnostics = [], {'skipped': 'development regression gate failed'}
        if all(gates.values()):
            prediction, fitted, production_diagnostics = fit_production(bundles, policy, rows, test_rows,
                store, control, production, temporary, guard, device)
        else:
            prediction = control
        gates['within_budget'] = guard.elapsed_hours() < MAX_HOURS
        submission, decision = publish(export, template, store.test_ids, prediction, store.species_ids, all(gates.values()))
        frame.to_csv(export / 'regression_per_survey_v30.csv', index=False)
        save_compact_diagnostics(export / 'calibration_top64_v30.npz', bundles, rows, store)
        report = {'experiment':EXPERIMENT, 'status':'complete', 'runtime_hours':guard.elapsed_hours(),
            'official_submission_made':False, 'official_public_score':None, 'official_private_score':None,
            'control_scores':{'public':.23900,'private':.21093}, 'private_target':.23021,
            'selected_policy':policy, 'policy_trials':trials, 'regression_check':regression,
            'submission_gate':{**gates, **decision}, 'submission':submission,
            'training':{'development':[{'fold':b['number'], 'members':b['records'], 'calibrations':b['calibrations'],
                         'count_models':b['count_records']} for b in bundles], 'production':fitted},
            'production_diagnostics':production_diagnostics, 'self_tests':tests,
            'hardware':{'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None, 'torch':torch.__version__},
            'validation_status':'all 88,987 PA IDs previously assessed; no fresh holdout remains',
            'limitations':['Repeated validation can be optimistically biased by previous experiment choices.',
                           'Matched v29 is a recipe refit; exact deployed v29 is embedded only for official test.',
                           'Production models exclude calibration-anchor blocks and buffers; they do not train on all PA.',
                           'No assertion of SOTA or guaranteed 12-hour completion.']}
        legacy.save_json(export / 'v30_report.json', report)
        manifest = {'experiment':EXPERIMENT, 'source_sha256':V30_SOURCE_HASH, 'embedded_v29_sha256':V29_SOURCE_HASH,
            'embedded_legacy_sha256':LEGACY_SOURCE_HASH, 'splits':manifests, 'production_split':production_manifest,
            'features':features, 'configs':CONFIGS, 'policies':POLICIES, 'runtime_cap_hours':MAX_HOURS,
            'no_external_data_or_weights':True, 'fresh_assessment':False,
            'outputs':{p.name:legacy.sha256_file(p) for p in sorted(export.iterdir())}}
        legacy.save_json(export / 'v30_manifest.json', manifest)
        size = sum(p.stat().st_size for p in export.iterdir())
        if len(list(export.iterdir())) != 5 or size > 16_000_000:
            raise ValueError('Compact export contract exceeded')
        return {'status':'complete', **decision, 'runtime_hours':guard.elapsed_hours(), 'output_bytes':size,
                'export_directory':str(export), 'regression_gain':regression['gain'],
                'fresh_assessment':False, 'policy':policy}
    except Exception as error:
        legacy.save_json(export / 'failure_report.json', {'experiment':EXPERIMENT, 'status':'failed',
            'error':str(error), 'traceback':traceback.format_exc(), 'runtime_hours':guard.elapsed_hours(),
            'official_submission_made':False})
        raise
    finally:
        del store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        legacy._clean_directory(temporary, working)
