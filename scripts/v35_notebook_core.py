"""v35: full-set, checkpoint-selected attention ensemble; cached v32 comparator.

All development labels were consumed before. No independent performance claim.
"""
from __future__ import annotations

import base64
import copy
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
from torch import nn
from torch.nn import functional as F

import scripts.v32_notebook_core as previous

v31, v29, legacy = previous.previous, previous.v29, previous.legacy
EXPERIMENT = "v35_full_set_balanced_attention"
CONTROL_HASH = "71c863d9cb05f3efc63ac54dd7faa05efd2fc137a11da9adcd2064c82c88dfee"
MAX_HOURS = 10.75
CHECKPOINT_EPOCHS = (12, 24, 36, 48)
CONFIGS = (
    {"id": "long_geo", "kind": "attention", "geo": True, "width": 192, "seed": 20263501,
     "epochs": 48, "rare_branch": False, "negative_clip": .02, "negative_gamma": 4., "ranking_weight": 0.},
    {"id": "balanced_ecology", "kind": "balanced_attention", "geo": False, "width": 192, "seed": 20263502,
     "epochs": 48, "rare_branch": True, "negative_clip": .02, "negative_gamma": 2., "ranking_weight": .08},
    {"id": "balanced_geo", "kind": "balanced_attention", "geo": True, "width": 192, "seed": 20263503,
     "epochs": 48, "rare_branch": True, "negative_clip": .02, "negative_gamma": 4., "ranking_weight": .08})
ENSEMBLES = {"equal": (1., 1., 1.), "ecology_lean": (1., 2., 1.), "balanced_pair": (0., 1., 1.)}
POLICIES = tuple({"id": f"{name}_{mode}", "ensemble": name, "count_mode": mode}
                 for name in ENSEMBLES for mode in ("scale0.7", "scale1", "scale1.3", "fixed18", "fixed24"))
MIN_GAIN = .001
MIN_OUTSIDE_GAIN = .0005
MIN_CI = .00025


class BalancedAttention(v29.SensorAttention):
    """Pool each sensor separately so 64 image tokens cannot dominate 1 env token."""
    def __init__(self, env_dim, static_dim, species, width=192):
        super().__init__(env_dim, static_dim, species, width)
        self.fusion = nn.Sequential(nn.Linear(6*width, 2*width), nn.GELU(), nn.Dropout(.15),
                                    legacy.ResidualVectorBlock(2*width, .15))

    def forward(self, x, geo=True):
        image = self.image(x["sentinel"]).flatten(2).transpose(1, 2)
        land = self.land(x["landsat"].permute(0, 3, 1, 2).flatten(2))
        climate = self.climate(x["bioclim"].permute(0, 2, 1, 3).flatten(2))
        env, static = self.environment(x["environment"])[:, None], self.static(x["static_geo" if geo else "static_eco"])[:, None]
        sensors = [image, land, climate, env, static]
        if self.training:
            sensors = [s*torch.bernoulli(s.new_full((len(s), 1, 1), .9)) for s in sensors]
        tokens = torch.cat([self.cls.expand(len(image), -1, -1)]+sensors, 1)
        encoded = self.encoder(tokens+self.position[:, :tokens.shape[1]])
        pooled, offset = [encoded[:, 0]], 1
        for sensor in sensors:
            pooled.append(encoded[:, offset:offset+sensor.shape[1]].mean(1))
            offset += sensor.shape[1]
        feature = self.norm(self.fusion(torch.cat(pooled, 1)))
        return self.head(feature), self.richness(feature).squeeze(1)


def create_model(store, config, indices):
    if config["kind"] == "attention":
        model = v29.create_model(store, config, indices)
    else:
        legacy.set_seed(config["seed"])
        model = BalancedAttention(store.dims["environment"],
            store.static_train.shape[1] if config["geo"] else store.eco_train.shape[1], len(store.species_ids), config["width"])
        frequency = legacy._frequency(store.labels, indices)
        prior = np.clip((frequency+.5)/(len(indices)+1.), 1e-5, .95)
        with torch.no_grad():
            model.head.bias.copy_(torch.tensor(np.log(prior/(1-prior)), dtype=torch.float32))
            model.richness.bias.fill_(math.log1p(frequency.sum()/len(indices)))
    frequency = legacy._frequency(store.labels, indices)
    rare = np.flatnonzero((frequency >= 5) & (frequency/len(indices) <= .005))
    if config["rare_branch"] and len(rare):
        model.head = previous.RareResidualHead(model.head, rare)
    return model, rare


def ranking_loss(logits, target):
    """Each survey's positive mass sums to one; supports soft mixup targets.

    Auxiliary ranking surrogate, NOT an exact expected-F1 loss.
    """
    distribution = target.float()/target.sum(1, keepdim=True).clamp_min(1e-6)
    return -(distribution*F.log_softmax(logits.float(), dim=1)).sum(1).mean()


def checkpoint_score(probability, target, countries):
    rank, _ = legacy.top_rank(probability, min(30, probability.shape[1]))
    values = np.mean([legacy.f1_from_ranked(target, rank, np.full(len(rank), min(k, rank.shape[1])))
                      for k in (12, 18, 24, 30)], axis=0)
    frame = pd.DataFrame({"score": values, "country": np.asarray(countries).astype(str)})
    country = frame.groupby("country").score.agg(["mean", "size"])
    supported = country.loc[country["size"] >= 30, "mean"]
    outside = frame.loc[~frame.country.isin(v31.CORE_COUNTRIES), "score"]
    macro = float(supported.mean()) if len(supported) else float(values.mean())
    transfer = float(outside.mean()) if len(outside) else float(values.mean())
    return float(.5*values.mean()+.25*macro+.25*transfer)


def train_model(store, rows, indices, selection, stats, config, directory, guard, device, *, production_epochs=None):
    started = time.monotonic()
    phase = "production" if production_epochs is not None else "development"
    epochs = int(production_epochs or config["epochs"])
    if epochs > config["epochs"]:
        raise ValueError("Production schedule exceeds registered schedule")
    model, rare = create_model(store, config, indices)
    model = model.to(device); ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.02)
    # Keep the same learning-rate prefix when production stops at a selected epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng, weights = np.random.default_rng(config["seed"]), v29.sample_weights(rows, indices)
    frequency = legacy._frequency(store.labels, indices)
    positive = torch.as_tensor(np.clip((20/np.maximum(frequency, 1))**.25, 1, 3), dtype=torch.float32, device=device)
    best, best_epoch, history = -np.inf, 0, []
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/f"{phase}_{config['id']}.pt"
    batch = 48 if device.type == "cuda" else 16
    for epoch in range(1, epochs+1):
        start, total, seen = time.monotonic(), 0., 0
        uniform = rng.permutation(indices)[:len(indices)//2]
        order = rng.permutation(np.r_[uniform, rng.choice(indices, len(indices)-len(uniform), p=weights)])
        model.train()
        for begin in range(0, len(order), batch):
            guard.require(60*60, f"{phase} v35 training")
            take = order[begin:begin+batch]
            x = v29.batch_inputs(store, take, stats, device, augment=True, rng=rng)
            target = torch.as_tensor(np.asarray(store.labels[take], np.float32), device=device)
            if len(take) > 1 and rng.random() < .5:
                mix, permutation = float(rng.beta(.2, .2)), torch.randperm(len(take), device=device)
                x = {k: mix*v+(1-mix)*v[permutation] for k, v in x.items()}
                target = mix*target+(1-mix)*target[permutation]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, richness = model(x, config["geo"])
                loss = 100*previous.asymmetric_loss(logits, target, positive, config["negative_clip"], config["negative_gamma"])
                loss += config["ranking_weight"]*ranking_loss(logits, target)
                loss += .04*F.smooth_l1_loss(richness.float(), torch.log1p(target.sum(1)))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite v35 loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.lerp_(current, .02)
            total += float(loss.detach())*len(take); seen += len(take)
        scheduler.step()
        score = None
        if production_epochs is None and (epoch in CHECKPOINT_EPOCHS or epoch == epochs):
            p = v29.predict(ema, store, selection, stats, device, config, views=1, guard=guard)
            score = checkpoint_score(p, np.asarray(store.labels[selection]), rows.iloc[selection].country)
            if score > best:
                best, best_epoch = score, epoch
                torch.save(ema.state_dict(), checkpoint)
            del p
        record = {"epoch": epoch, "loss": total/max(seen, 1), "selection_score": score, "seconds": time.monotonic()-start,
                  "learning_rate_after_epoch": optimizer.param_groups[0]["lr"]}
        history.append(record); guard.stamp("v35_train", model=config["id"], phase=phase, **record)
    if production_epochs is not None:
        best_epoch = epochs
        torch.save(ema.state_dict(), checkpoint)
    if best_epoch < 1:
        raise RuntimeError("No selected checkpoint")
    ema.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    del model, optimizer
    previous.release(device)
    return ema, {"config": config, "training_surveys": len(indices), "best_epoch": best_epoch,
        "selection_score": float(best) if production_epochs is None else None, "history": history,
        "seconds": time.monotonic()-started, "parameters": sum(p.numel() for p in ema.parameters()),
        "rare_columns": len(rare), "all_species_outputs_trained": len(store.species_ids),
        "scheduler_total_epochs": config["epochs"], "checkpoint_sha256": legacy.sha256_file(checkpoint)}


def decode(probability, mode):
    maximum = min(40, probability.shape[1]); minimum = min(8, maximum)
    rank, value = legacy.top_rank(probability, maximum)
    if mode.startswith("fixed"):
        counts = np.full(len(probability), min(maximum, max(minimum, int(mode[5:]))))
    else:
        scale = float(mode[5:])
        k = np.arange(1, maximum+1)
        objective = 2*np.cumsum(value, 1)/(k[None]+probability.sum(1)[:, None]*scale+1e-8)
        counts = objective[:, minimum-1:].argmax(1)+minimum
    return [row[:count].astype(int).tolist() for row, count in zip(rank, counts)]


def combine(members, name):
    weights = np.asarray(ENSEMBLES[name], np.float32); weights /= weights.sum()
    result = np.zeros_like(members[0], dtype=np.float32)
    for weight, member in zip(weights, members):
        if weight:
            result += np.asarray(member, np.float32)*weight
    return result


def load_reference(payload):
    packed = base64.b64decode(payload, validate=True)
    if hashlib.sha256(packed).hexdigest() != REFERENCE_PAYLOAD_HASH:
        raise ValueError("Reference payload mismatch")
    raw = lzma.decompress(packed)
    if hashlib.sha256(raw).hexdigest() != REFERENCE_RAW_HASH:
        raise ValueError("Reference raw hash mismatch")
    return json.loads(raw)


def label_hash(labels):
    digest = hashlib.sha256()
    for start in range(0, len(labels), 1024):
        digest.update(np.asarray(labels[start:start+1024], np.uint8).tobytes())
    return digest.hexdigest()


def bind_reference(reference, rows, species, splits, manifests, labels):
    """Match ALL original training/evaluation IDs before trusting cached scores."""
    if len(splits) != 2 or len(manifests) != 2 or len(reference["folds"]) != 2:
        raise ValueError("Exactly two cached folds required")
    if hashlib.sha256(np.asarray(species, '<i8').tobytes()).hexdigest() != reference["species_sha256"]:
        raise ValueError("Reference species vocabulary mismatch")
    if label_hash(labels) != reference["labels_sha256"]:
        raise ValueError("Official PA labels changed; cached comparator is invalid")
    lookup = pd.Index(rows.surveyId)
    bound = []
    for f, (split, manifest, saved) in enumerate(zip(splits, manifests, reference["folds"])):
        if manifest["id_hashes"] != saved["id_hashes"]:
            raise ValueError("Cached baseline split IDs changed; do not compare incompatible training recipes")
        roles = {}
        for role in ("calibration", "assessment"):
            ids = np.asarray(saved[role]["ids"], np.int64)
            take = lookup.get_indexer(ids)
            score = np.asarray(saved[role]["f1"], np.float64)
            if ((take < 0).any() or len(np.unique(ids)) != len(ids) or len(score) != len(ids) or
                not np.isfinite(score).all() or np.any((score < 0) | (score > 1)) or
                not set(take).issubset(set(split[role]))):
                raise ValueError("Invalid cached comparator alignment")
            if role == "assessment" and set(take) != set(split[role]):
                raise ValueError("Incomplete cached assessment")
            if role == "calibration":
                prediction = saved[role]["predictions"]
                if not np.allclose(legacy.score_prediction_lists(np.asarray(labels[take]), prediction), score, atol=1e-10, rtol=0):
                    raise ValueError("Calibration labels/cache mismatch")
            roles[role] = {"indices": take, "reference_f1": score}
        bound.append(roles)
    if len(bound) != 2:
        raise ValueError("Exactly two cached folds required")
    return bound


def fit_fold(number, split, bound, rows, store, directory, guard, device):
    started = time.monotonic()
    stats = v29.fit_normalization(store, split["training"])
    data = {r: {**b, "members": []} for r, b in bound.items()}
    records, calibrations = [], []
    for source in CONFIGS:
        guard.require(3*3600, "start v35 development model")
        config = {**source, "seed": source["seed"]+number*1000}
        model, record = train_model(store, rows, split["training"], split["selection"], stats, config,
                                    directory/f"fold{number}", guard, device)
        p = v29.predict(model, store, split["selection"], stats, device, config, views=2, guard=guard)
        cal = v29.fit_platt(p, np.asarray(store.labels[split["selection"]]))
        calibrations.append(cal); records.append(record)
        for role, d in data.items():
            p = v29.predict(model, store, d["indices"], stats, device, config, views=2, guard=guard)
            d["members"].append(v29.calibrated(p, cal).astype(np.float16))
        del model, p
        previous.release(device)
    trials = {}
    d = data["calibration"]; target = np.asarray(store.labels[d["indices"]])
    for name in ENSEMBLES:
        probability = combine(d["members"], name)
        for policy in (p for p in POLICIES if p["ensemble"] == name):
            trials[policy["id"]] = legacy.score_prediction_lists(target, decode(probability, policy["count_mode"]))
        del probability
    return {"number": number, "split": split, "data": data, "records": records, "calibrations": calibrations,
        "trials": trials, "wall_seconds": time.monotonic()-started}


def select_policy(bundles, rows):
    take = np.concatenate([b["data"]["calibration"]["indices"] for b in bundles])
    reference = np.concatenate([b["data"]["calibration"]["reference_f1"] for b in bundles])
    trials = []
    for policy in POLICIES:
        scores = np.concatenate([b["trials"][policy["id"]] for b in bundles])
        gains = [float((b["trials"][policy["id"]]-b["data"]["calibration"]["reference_f1"]).mean()) for b in bundles]
        geo = v31.geographic_gain(scores-reference, rows.iloc[take].country, rows.iloc[take].surveyId)
        allowed = min(gains) > 0 and geo["gain"] >= MIN_GAIN and geo["outside_core_gain"] >= MIN_OUTSIDE_GAIN and geo["geographic_gain_positive"]
        trials.append({**policy, **geo, "fold_gains": gains, "allowed": bool(allowed), "sample_f1": float(scores.mean())})
    # If no policy passes, retain the best ATTEMPT for honest diagnostics, not a
    # hidden all-zero control report. It cannot authorize production/submission.
    eligible = [t for t in trials if t["allowed"]]
    chosen = max(eligible or trials, key=lambda t: (t["robust_gain"], t["gain"]))
    return next(dict(p) for p in POLICIES if p["id"] == chosen["id"]), trials, bool(eligible)


def regression_check(bundles, policy, rows, store):
    frames, summaries = [], []
    for b in bundles:
        d = b["data"]["assessment"]; take = d["indices"]
        probability = combine(d["members"], policy["ensemble"])
        prediction = decode(probability, policy["count_mode"])
        target = np.asarray(store.labels[take]); new = legacy.score_prediction_lists(target, prediction)
        old = d["reference_f1"]
        members = {c["id"]: float(legacy.score_prediction_lists(target, decode(p, policy["count_mode"])).mean())
                   for c, p in zip(CONFIGS, d["members"])}
        frames.append(pd.DataFrame({"surveyId": rows.iloc[take].surveyId.to_numpy(), "fold": b["number"],
            "country": rows.iloc[take].country.to_numpy(), "spatial_block": legacy.spatial_blocks(rows.iloc[take]),
            "cached_matched_v32_f1": old, "v35_f1": new, "delta_f1": new-old,
            "predicted_cardinality": list(map(len, prediction))}))
        summaries.append({"fold": b["number"], "gain": float((new-old).mean()),
            "members_not_used_for_policy_selection": members, "multilabel": v29.multilabel_summary(target, prediction),
            "species_groups": legacy.species_group_metrics(target, prediction, legacy._frequency(store.labels, b["split"]["training"]))})
        del probability
    frame = pd.concat(frames, ignore_index=True)
    if not frame.surveyId.is_unique:
        raise ValueError("Duplicate assessment surveys")
    return frame, {**v31.geographic_gain(frame.delta_f1, frame.country), "folds": summaries,
        "surveys": len(frame), "matched_v32_f1": float(frame.cached_matched_v32_f1.mean()), "v35_f1": float(frame.v35_f1.mean()),
        "bootstrap": legacy.paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy(), iterations=1000, seed=20263507),
        "fresh_assessment": False, "used_for_policy_selection": False,
        "by_country": legacy.summarize_by_group(frame, "country", ("cached_matched_v32_f1", "v35_f1", "delta_f1"))}


def production_epochs(bundles):
    return [int(np.median([b["records"][i]["best_epoch"] for b in bundles])) for i in range(len(CONFIGS))]


def production_estimate(bundles, full_rows, *, maximum=False):
    epochs = [c["epochs"] for c in CONFIGS] if maximum else production_epochs(bundles)
    return float(sum(max(np.median([h["seconds"] for h in b["records"][i]["history"]])/b["records"][i]["training_surveys"]
                    for b in bundles)*full_rows*epochs[i]*1.5 for i in range(len(CONFIGS)))+45*60)


def remaining_plan_estimate(bundles, next_training_rows, full_rows):
    first = bundles[0]
    return float(first["wall_seconds"]*max(1., next_training_rows/len(first["split"]["training"]))*1.25
                 +production_estimate(bundles, full_rows, maximum=True)+15*60)


def fit_production(bundles, policy, rows, test_rows, store, directory, guard, device):
    estimate = production_estimate(bundles, len(rows))
    guard.require(estimate, "whole v35 production admission")
    take = np.arange(len(rows)); stats = v29.fit_normalization(store, take)
    epochs, members, records = production_epochs(bundles), [], []
    for i, config in enumerate(CONFIGS):
        if ENSEMBLES[policy["ensemble"]][i] == 0:
            members.append(None); continue
        model, record = train_model(store, rows, take, np.array([], np.int64), stats, config, directory,
                                   guard, device, production_epochs=epochs[i])
        cal = {k: float(np.mean([b["calibrations"][i][k] for b in bundles])) for k in ("slope", "intercept")}
        p = v29.predict(model, store, np.arange(len(test_rows)), stats, device, config, test=True, views=2, guard=guard)
        members.append(v29.calibrated(p, cal).astype(np.float16))
        records.append({**record, "calibration": cal, "calibration_source": "selection-only fold coefficients; full-data transfer assumption"})
        del model, p; previous.release(device)
    # A zero-weight omitted member must not allocate an unnecessary N x S array.
    weights = np.asarray(ENSEMBLES[policy["ensemble"]], np.float32); weights /= weights.sum()
    probability = np.zeros((len(test_rows), len(store.species_ids)), np.float32)
    for p, weight in zip(members, weights):
        if weight:
            probability += np.asarray(p, np.float32)*weight
    prediction = decode(probability, policy["count_mode"])
    return prediction, records, {"training_rows": len(take), "all_PA_rows_used": True,
        "calibration_anchor_rows_removed": 0, "epochs_selected_before_assessment": epochs,
        "scheduler_prefix_preserved": True, "admission_estimate_seconds": estimate,
        "mean_cardinality": float(np.mean(list(map(len, prediction)))), "full_set_predictions": True}


def save_compact(path, bundles, rows, store, per_assessment=500):
    ids, folds, roles, countries, ranks, values, masses, reference, truths = [], [], [], [], [], [], [], [], []
    for b in bundles:
        for role in ("calibration", "assessment"):
            data = b["data"][role]; take = data["indices"]
            n = len(take) if role == "calibration" else min(per_assessment, len(take))
            pos = np.sort(np.random.default_rng(20263508+b["number"]).choice(len(take), n, replace=False,
                          p=v29.sample_weights(rows, take)))
            rs, vs, ms = [], [], []
            for member in data["members"]:
                p = np.asarray(member[pos], np.float32); rank, value = v31.compact_rank(p)
                rs.append(rank); vs.append(value); ms.append(p.sum(1))
            ranks.append(np.stack(rs, 1)); values.append(np.stack(vs, 1)); masses.append(np.stack(ms, 1))
            ids.append(rows.iloc[take[pos]].surveyId.to_numpy()); folds.append(np.full(n, b["number"], np.uint8))
            roles.append(np.full(n, role)); countries.append(rows.iloc[take[pos]].country.fillna("unknown").to_numpy(dtype="U64"))
            reference.append(data["reference_f1"][pos])
            truths.extend(np.flatnonzero(store.labels[j]).astype(np.uint16) for j in take[pos])
    np.savez_compressed(path, survey_id=np.concatenate(ids), fold=np.concatenate(folds), role=np.concatenate(roles),
        country=np.concatenate(countries), ranked_species_columns=np.concatenate(ranks), probability=np.concatenate(values),
        probability_mass=np.concatenate(masses), cached_matched_v32_f1=np.concatenate(reference),
        true_species_columns=np.concatenate(truths), true_offsets=np.r_[0, np.cumsum(list(map(len, truths)))].astype(np.uint32),
        species_ids=store.species_ids, expert_ids=np.array([c["id"] for c in CONFIGS]),
        evidence_scope=np.array("previously consumed calibration and assessment; individual model top128 plus FULL probability mass"))


def self_tests():
    logits = torch.zeros(2, 12, requires_grad=True)
    target = torch.zeros(2, 12); target[0, :2] = 1; target[1, :4] = .5
    loss = ranking_loss(logits, target); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    for p in POLICIES:
        output = decode(np.full((2, 50), .05, np.float32), p["count_mode"])
        assert all(8 <= len(x) <= 40 and len(x) == len(set(x)) for x in output)
    return {"passed": True, "tests": 4}


def publish(export, temporary, template, ids, prediction, species, gate):
    proof, name, differs = None, None, False
    if prediction is not None:
        path = temporary/"candidate.csv"
        proof = legacy.write_submission(path, template, ids, prediction, species)
        differs = proof["sha256"] != CONTROL_HASH
        if gate and differs:
            name = "GLC25_PA_submission_v35.csv"
            path.replace(export/name)
    decision = {"eligible_for_submission": name is not None, "prediction_file": name, "different_from_v32": differs,
        "message": "SUBMIT ONLY GLC25_PA_submission_v35.csv" if name else "NO NEW SUBMISSION; keep the scored v32"}
    if name is None:
        legacy.save_json(export/"NO_SUBMISSION.json", decision)
    return proof, decision


def run_v35(reference_b64):
    guard = legacy.RuntimeGuard(MAX_HOURS)
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
    temporary, export = working/"v35_runtime", working/"v35_export"
    legacy._clean_directory(temporary, working); legacy._clean_directory(export, working)
    temporary.mkdir(parents=True); export.mkdir(parents=True)
    store, bundles = None, []
    try:
        device = legacy.require_gpu()
        torch.set_num_threads(min(os.cpu_count() or 2, 6))
        tests, reference = self_tests(), load_reference(reference_b64)
        root = legacy.discover_data_root()
        if legacy.sha256_file(root/"GLC25_PA_metadata_train.csv") != reference["train_metadata_sha256"]:
            raise ValueError("Official training metadata changed; cached reference invalid")
        preflight = pd.read_csv(root/"GLC25_PA_metadata_train.csv", usecols=["surveyId", "lat", "lon", "country"])
        preflight = preflight.drop_duplicates("surveyId").sort_values("surveyId").reset_index(drop=True)
        _, manifests = v31.previous.make_splits(preflight)
        if len(manifests) != len(reference["folds"]) or any(m["id_hashes"] != s["id_hashes"] for m, s in zip(manifests, reference["folds"])):
            raise ValueError("Cached reference partitions do not match official metadata")
        guard.stamp("preflight", folds=manifests, cached_reference=True, fresh_assessment=False)
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
        bound = bind_reference(reference, rows, store.species_ids, splits, manifests, store.labels)
        template = pd.read_csv(root/"GLC25_SAMPLE_SUBMISSION.csv")
        for i, split in enumerate(splits):
            if bundles:
                estimate = remaining_plan_estimate(bundles, len(split["training"]), len(rows))
                guard.stamp("remaining_plan_admission", estimate_seconds=estimate)
                guard.require(estimate, "remaining development and longest production plan")
            bundles.append(fit_fold(i, split, bound[i], rows, store, temporary, guard, device))
        policy, trials, calibration_pass = select_policy(bundles, rows)
        epochs = production_epochs(bundles)
        legacy.save_json(temporary/"frozen_policy.json", {"policy": policy, "trials": trials, "production_epochs": epochs})
        guard.stamp("policy_frozen", policy=policy, calibration_pass=calibration_pass, production_epochs=epochs)
        frame, regression = regression_check(bundles, policy, rows, store)
        gates = {"calibration_pass": calibration_pass,
            "positive_each_fold": all(f["gain"] > 0 for f in regression["folds"]),
            "minimum_practical_gain": regression["gain"] >= MIN_GAIN,
            "spatial_ci_margin": regression["bootstrap"]["ci95"][0] >= MIN_CI,
            "geographic_transfer_positive": regression["geographic_gain_positive"],
            "minimum_outside_core_gain": regression["outside_core_gain"] >= MIN_OUTSIDE_GAIN,
            "geographic_buffers": min(m["minimum_distance_km"] for m in manifests) >= 20,
            "cached_reference_ids_and_labels_verified": True}
        prediction, fitted, diagnostics = None, [], {"skipped": "development gates failed; actual candidate diagnostics retained"}
        if all(gates.values()):
            prediction, fitted, diagnostics = fit_production(bundles, policy, rows, test_rows, store, temporary, guard, device)
        gates["within_budget"] = guard.elapsed_hours() < MAX_HOURS
        proof, decision = publish(export, temporary, template, store.test_ids, prediction, store.species_ids, all(gates.values()))
        frame.to_csv(export/"regression_per_survey_v35.csv", index=False)
        save_compact(export/"predictions_top128_v35.npz", bundles, rows, store)
        report = {"experiment": EXPERIMENT, "status": "complete", **decision, "runtime_hours": guard.elapsed_hours(),
            "self_tests": tests, "policy": policy, "calibration_trials": trials, "gates": gates,
            "regression": regression, "submission_validation": proof, "official_submission_made": False,
            "training": {"development": [{"fold": b["number"], "members": b["records"], "calibrations": b["calibrations"],
                "wall_seconds": b["wall_seconds"]} for b in bundles], "production": fitted},
            "production_diagnostics": diagnostics, "production_epochs_frozen_before_assessment": epochs,
            "cached_reference": {"provenance": reference["provenance"], "payload_hash": REFERENCE_PAYLOAD_HASH,
                "reference_model_refits_this_run": 0, "calibration_rows_per_fold": [len(b["calibration"]["indices"]) for b in bound]},
            "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__},
            "train_country_counts": rows.country.value_counts().to_dict(), "test_country_counts_unlabeled": test_rows.country.value_counts().to_dict(),
            "validation_status": "all 88,987 PA surveys consumed before; repeated spatial development, no independent holdout",
            "limitations": ["No guarantee of hidden-test improvement, winning, or full 12-hour completion.",
                "Full-set replacement may regress despite passing these repeated development gates.",
                "Calibration uses the 3000 archived sampled fold views, not the full historical calibration pool.",
                "The cached comparator is the unchanged v32 RECIPE refit inside the v34 run, not its test score.",
                "New pooling, auxiliary ranking, training duration, and decoding change together; component causality is not established.",
                "Countries with no PA labels, including much of Ukraine and the UK, cannot be independently validated here.",
                "Production uses every PA row; calibration transfers from smaller folds and can shift.",
                "The ratio-of-expectations count decoder and ranking loss are surrogates, not exact expected-F1 optimization."]}
        legacy.save_json(export/"v35_report.json", report)
        manifest = {"experiment": EXPERIMENT, "source_sha256": V35_SOURCE_HASH, "embedded_sources_sha256": FROZEN_SOURCE_HASHES,
            "splits": manifests, "features": features, "configs": CONFIGS, "checkpoint_epochs": CHECKPOINT_EPOCHS,
            "ensembles": ENSEMBLES, "policies": POLICIES, "reference_sha256": REFERENCE_PAYLOAD_HASH,
            "control_submission_sha256": CONTROL_HASH, "runtime_cap_hours": MAX_HOURS,
            "gate_margins": {"gain": MIN_GAIN, "outside_core": MIN_OUTSIDE_GAIN, "ci_lower": MIN_CI},
            "no_external_data_or_weights": True, "fresh_assessment": False,
            "outputs": {p.name: legacy.sha256_file(p) for p in sorted(export.iterdir())}}
        legacy.save_json(export/"v35_manifest.json", manifest)
        size = sum(p.stat().st_size for p in export.iterdir())
        if len(list(export.iterdir())) != 5 or size > 16_000_000:
            raise ValueError("Compact five-file/16MB export contract exceeded")
        guard.require(0, "final export")
        return {"status": "complete", **decision, "runtime_hours": guard.elapsed_hours(), "output_bytes": size,
            "export_directory": str(export), "regression_gain": regression["gain"], "fresh_assessment": False, "policy": policy}
    except Exception as error:
        ready = export/"GLC25_PA_submission_v35.csv"
        if ready.exists():
            ready.replace(export/"failed_predictions_DO_NOT_SUBMIT.txt")
        report_path = export/"v35_report.json"
        if report_path.exists():
            failed = json.loads(report_path.read_text(encoding="utf-8"))
            failed.update(status="failed", eligible_for_submission=False, prediction_file=None, message="DO NOT SUBMIT; export failed")
            legacy.save_json(report_path, failed)
        manifest_path = export/"v35_manifest.json"
        if manifest_path.exists():
            manifest_path.replace(export/"failed_manifest_not_a_valid_export.json")
        legacy.save_json(export/"failure_report.json", {"experiment": EXPERIMENT, "status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "runtime_hours": guard.elapsed_hours(), "official_submission_made": False,
            "eligible_for_submission": False, "prediction_file": None,
            "completed_folds": [{"fold": b["number"], "records": b["records"], "wall_seconds": b["wall_seconds"]} for b in bundles]})
        raise
    finally:
        del store
        previous.release(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        legacy._clean_directory(temporary, working)
