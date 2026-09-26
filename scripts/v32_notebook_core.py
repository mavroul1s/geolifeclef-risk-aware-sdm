"""v32: seed diversity plus asymmetric-loss rare-species specialists.

Exact scored v31 test control; repeated (not fresh) spatial development checks.
No external weights, data, automatic submission, or claim of hidden-test improvement.
"""
from __future__ import annotations

import base64
import copy
import gc
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

import scripts.v31_notebook_core as previous

v29, legacy = previous.v29, previous.legacy
EXPERIMENT = "v32_asymmetric_rare_specialist_ensemble"
CONTROL_HASH = "da070a8ac5708ef5d7fb38cdbecf862aa9d036e4d246bc1e4b364572bb7fe367"
MAX_HOURS = 10.75
FROZEN_V31_POLICY = {"id": "seed1_habitat0.25_swap2", "neural": 1., "habitat": .25, "swaps": 2}
BAG_CONFIGS = tuple({**c, "seed": c["seed"]+62000, "id": "bag_"+c["id"]} for c in previous.CONFIGS)
BAG_EPOCHS = (12, 15, 18)
SPECIALISTS = (
    {"id": "asymmetric_geo", "kind": "attention", "geo": True, "width": 192,
     "seed": 20263211, "epochs": 20, "rare_branch": False, "negative_clip": .02, "negative_gamma": 4.},
    {"id": "rare_ecology", "kind": "attention", "geo": False, "width": 192,
     "seed": 20263212, "epochs": 24, "rare_branch": True, "negative_clip": .02, "negative_gamma": 2.})
POLICIES = ({"id": "control", "bag": 0., "specialist": 0., "swaps": 0},) + tuple(
    {"id": f"bag{a:g}_specialist{b:g}_swap{k}", "bag": a, "specialist": b, "swaps": k}
    for a, b in ((.5, 0.), (1., 0.), (0., .5), (0., 1.), (.5, .5), (1., .5), (1., 1.))
    for k in (2, 4, 8))


def asymmetric_loss(logits, target, positive_weight, clip=.02, gamma=4.):
    """Separate positive/negative terms remain correct for mixup soft labels.

    Detach focal weights (not log probabilities), keep float32 logs under AMP.
    Negative clipping ignores very easy absences; positives are never clipped.
    """
    logits = logits.float()
    p = logits.sigmoid()
    negative_probability = (1-p+clip).clamp(max=1)
    positive = -F.logsigmoid(logits)*target*positive_weight
    negative = -(1-target)*negative_probability.clamp_min(1e-8).log()
    negative *= (1-negative_probability).detach().pow(gamma)
    return (positive+negative).mean()


class RareResidualHead(nn.Module):
    """Extra nonlinear capacity for training-defined infrequent taxa only."""
    def __init__(self, original, rare_columns):
        super().__init__()
        self.base = original
        self.register_buffer("rare_columns", torch.as_tensor(rare_columns, dtype=torch.long))
        hidden = original.in_features//2
        self.rare = nn.Sequential(nn.Linear(original.in_features, hidden), nn.GELU(),
                                  nn.Dropout(.2), nn.Linear(hidden, len(rare_columns)))
        # Initially identical to the ordinary head, including the frequency prior.
        nn.init.zeros_(self.rare[-1].weight)
        nn.init.zeros_(self.rare[-1].bias)

    def forward(self, feature):
        logits = self.base(feature)
        return logits.index_add(1, self.rare_columns, self.rare(feature))


def make_specialist(store, config, indices):
    model = v29.create_model(store, config, indices)
    frequency = legacy._frequency(store.labels, indices)
    rare = np.flatnonzero((frequency >= 5) & (frequency/len(indices) <= .005))
    if config["rare_branch"] and len(rare):
        model.head = RareResidualHead(model.head, rare)
    return model, rare


def train_specialist(store, rows, indices, stats, config, output, guard, device, phase):
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, rare = make_specialist(store, config, indices)
    model = model.to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config["seed"])
    weights = v29.sample_weights(rows, indices)
    frequency = legacy._frequency(store.labels, indices)
    positive = torch.as_tensor(np.clip((20/np.maximum(frequency, 1))**.25, 1, 3),
                               dtype=torch.float32, device=device)
    history = []
    batch = 48 if device.type == "cuda" else 16
    for epoch in range(1, config["epochs"]+1):
        start, total, seen = time.monotonic(), 0., 0
        uniform = rng.permutation(indices)[:len(indices)//2]
        order = rng.permutation(np.r_[uniform, rng.choice(indices, len(indices)-len(uniform), p=weights)])
        model.train()
        for begin in range(0, len(order), batch):
            guard.require(60*60, f"{phase} specialist training")
            take = order[begin:begin+batch]
            x = v29.batch_inputs(store, take, stats, device, augment=True, rng=rng)
            target = torch.as_tensor(np.asarray(store.labels[take], np.float32), device=device)
            if len(take) > 1 and rng.random() < .5:
                mix = float(rng.beta(.2, .2))
                permutation = torch.randperm(len(take), device=device)
                x = {k: mix*v+(1-mix)*v[permutation] for k, v in x.items()}
                target = mix*target+(1-mix)*target[permutation]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, richness = model(x, config["geo"])
                loss = 100*asymmetric_loss(logits, target, positive, config["negative_clip"], config["negative_gamma"])
                loss += .04*F.smooth_l1_loss(richness.float(), torch.log1p(target.sum(1)))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite specialist loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.lerp_(current, .02)
            total += float(loss.detach())*len(take); seen += len(take)
        scheduler.step()
        record = {"epoch": epoch, "loss": total/max(seen, 1), "seconds": time.monotonic()-start}
        history.append(record)
        guard.stamp("v32_specialist_train", model=config["id"], phase=phase, **record)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/f"{phase}_{config['id']}.pt"
    torch.save(ema.state_dict(), checkpoint)
    return ema, {"config": config, "training_surveys": len(indices), "best_epoch": config["epochs"],
        "history": history, "seconds": time.monotonic()-started,
        "parameters": sum(p.numel() for p in ema.parameters()),
        "rare_columns": len(rare), "rare_definition": "training count >=5 and prevalence <=0.005",
        "rare_branch_enabled": bool(config["rare_branch"] and len(rare)),
        "all_species_outputs_trained": len(store.species_ids),
        "checkpoint_sha256": legacy.sha256_file(checkpoint),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def decode(base, bag, specialist, policy, guard=None):
    return previous.residual_decode(base, bag, specialist,
        {"neural": policy["bag"], "habitat": policy["specialist"], "swaps": policy["swaps"]}, guard)


def release(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def fit_fold(number, split, rows, store, temporary, guard, device):
    # Refits the exact frozen v31 recipe, not an older v29 comparator.
    baseline = previous.fit_fold(number, split, rows, store, temporary, guard, device)
    data = {role: {"base": previous.residual_decode(d["base"], d["neural"], d["habitat"], FROZEN_V31_POLICY, guard)}
            for role, d in baseline["data"].items()}
    records, calibrations = [], {}
    stats = v29.fit_normalization(store, split["training"])
    directory = temporary/f"new_fold_{number}"; directory.mkdir()
    for group, configs in (("bag", BAG_CONFIGS), ("specialist", SPECIALISTS)):
        sums = {r: np.zeros((len(split[r]), len(store.species_ids)), np.float32) for r in data}
        for i, source in enumerate(configs):
            guard.require(3.5*3600, "start additional development model")
            config = {**source, "seed": source["seed"]+number*1000}
            if group == "bag":
                model, record = v29.train_candidate(store, rows, split["training"], split["selection"], stats,
                    config, directory, guard, device, fixed_epochs=BAG_EPOCHS[i], phase="development")
            else:
                model, record = train_specialist(store, rows, split["training"], stats, config, directory, guard, device, "development")
            p = v29.predict(model, store, split["selection"], stats, device, config, views=2, guard=guard)
            cal = v29.fit_platt(p, np.asarray(store.labels[split["selection"]]))
            calibrations[config["id"]] = cal
            for role in data:
                p = v29.predict(model, store, split[role], stats, device, config, views=2, guard=guard)
                sums[role] += v29.calibrated(p, cal)/len(configs)
            records.append({**record, "group": group, "member": i})
            del model, p
            release(device)
        for role, probability in sums.items():
            rank, value = previous.compact_rank(probability)
            data[role][group] = {"rank": rank, "value": value}
        del sums, probability
    target, d = np.asarray(store.labels[split["calibration"]]), data["calibration"]
    trials = {p["id"]: legacy.score_prediction_lists(target, decode(d["base"], d["bag"], d["specialist"], p, guard)) for p in POLICIES}
    return {"number": number, "split": split, "data": data, "records": records, "calibrations": calibrations,
            "reference_records": baseline["records"], "trials": trials}


def select_policy(bundles, rows):
    trials = []
    take = np.concatenate([b["split"]["calibration"] for b in bundles])
    reference = np.concatenate([b["trials"]["control"] for b in bundles])
    for policy in POLICIES:
        scores = np.concatenate([b["trials"][policy["id"]] for b in bundles])
        gains = [float((b["trials"][policy["id"]]-b["trials"]["control"]).mean()) for b in bundles]
        geo = previous.geographic_gain(scores-reference, rows.iloc[take].country, rows.iloc[take].surveyId)
        allowed = policy["id"] == "control" or (min(gains) > 0 and geo["geographic_gain_positive"])
        trials.append({**policy, **geo, "fold_gains": gains, "allowed": bool(allowed), "sample_f1": float(scores.mean())})
    chosen = max((t for t in trials if t["allowed"]), key=lambda t: (t["robust_gain"], -t["swaps"], -t["bag"]-t["specialist"]))
    return next(dict(p) for p in POLICIES if p["id"] == chosen["id"]), trials


def regression_check(bundles, policy, rows, store, guard):
    frames, summaries = [], []
    for b in bundles:
        take, d = b["split"]["assessment"], b["data"]["assessment"]
        target = np.asarray(store.labels[take])
        prediction = decode(d["base"], d["bag"], d["specialist"], policy, guard)
        old, new = [legacy.score_prediction_lists(target, p) for p in (d["base"], prediction)]
        ablations = {}
        for expert in ("bag", "specialist"):
            p = decode(d["base"], d["bag"], d["specialist"], {**policy, expert: 0.}, guard)
            ablations["without_"+expert] = float((legacy.score_prediction_lists(target, p)-old).mean())
        frames.append(pd.DataFrame({"surveyId": rows.iloc[take].surveyId.to_numpy(), "fold": b["number"],
            "country": rows.iloc[take].country.to_numpy(), "spatial_block": legacy.spatial_blocks(rows.iloc[take]),
            "matched_v31_f1": old, "v32_f1": new, "delta_f1": new-old,
            "predicted_cardinality": list(map(len, prediction)),
            "swaps": [len(set(a)-set(c)) for a, c in zip(prediction, d["base"])]}))
        summaries.append({"fold": b["number"], "gain": float((new-old).mean()), "ablations_not_used_for_selection": ablations,
            "multilabel": v29.multilabel_summary(target, prediction),
            "species_groups": legacy.species_group_metrics(target, prediction, legacy._frequency(store.labels, b["split"]["training"]))})
    frame = pd.concat(frames, ignore_index=True)
    if not frame.surveyId.is_unique:
        raise ValueError("Duplicate regression survey")
    return frame, {**previous.geographic_gain(frame.delta_f1, frame.country), "folds": summaries,
        "surveys": len(frame), "matched_v31_f1": float(frame.matched_v31_f1.mean()), "v32_f1": float(frame.v32_f1.mean()),
        "bootstrap": legacy.paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy(), iterations=1000, seed=20263207),
        "fresh_assessment": False, "used_for_policy_selection_in_this_run": False,
        "by_country": legacy.summarize_by_group(frame, "country", ("matched_v31_f1", "v32_f1", "delta_f1"))}


def production_estimate(bundles, policy, full_rows):
    seconds = 45*60
    for group, configs in (("bag", BAG_CONFIGS), ("specialist", SPECIALISTS)):
        if not policy[group]:
            continue
        for i, config in enumerate(configs):
            speed = max(np.median([h["seconds"] for h in r["history"]])/r["training_surveys"]
                        for b in bundles for r in b["records"] if r["group"] == group and r["member"] == i)
            seconds += speed*full_rows*(BAG_EPOCHS[i] if group == "bag" else config["epochs"])*1.5
    return float(seconds)


def fit_production(bundles, policy, rows, test_rows, store, control, directory, guard, device):
    estimate = production_estimate(bundles, policy, len(rows))
    guard.require(estimate, "whole v32 production ensemble admission")
    take = np.arange(len(rows)); stats = v29.fit_normalization(store, take)
    experts, records = {"bag": None, "specialist": None}, []
    for group, configs in (("bag", BAG_CONFIGS), ("specialist", SPECIALISTS)):
        if not policy[group]:
            continue
        probability = np.zeros((len(test_rows), len(store.species_ids)), np.float32)
        for i, config in enumerate(configs):
            if group == "bag":
                model, record = v29.train_candidate(store, rows, take, np.array([], np.int64), stats, config,
                    directory, guard, device, fixed_epochs=BAG_EPOCHS[i], phase="production")
                cal = previous.PRODUCTION_CALIBRATIONS[i]
                source = "frozen scored v29 coefficients; between-seed transfer assumption"
            else:
                model, record = train_specialist(store, rows, take, stats, config, directory, guard, device, "production")
                cal = {k: float(np.mean([b["calibrations"][config["id"]][k] for b in bundles])) for k in ("slope", "intercept")}
                source = "mean of selection-only development coefficients; full-data transfer assumption"
            p = v29.predict(model, store, np.arange(len(test_rows)), stats, device, config, test=True, views=2, guard=guard)
            probability += v29.calibrated(p, cal)/len(configs)
            records.append({**record, "group": group, "calibration": cal, "calibration_source": source})
            del model, p
            release(device)
        rank, value = previous.compact_rank(probability)
        experts[group] = {"rank": rank, "value": value}
        del probability
    prediction = decode(control, experts["bag"], experts["specialist"], policy, guard)
    swaps = np.array([len(set(a)-set(b)) for a, b in zip(prediction, control)])
    return prediction, records, {"training_rows": len(take), "all_PA_rows_used": True,
        "calibration_anchor_rows_removed": 0, "cardinality_equal_to_scored_v31_per_row": True,
        "admission_estimate_seconds": estimate, "changed_rows": int((swaps > 0).sum()),
        "mean_swaps": float(swaps.mean()), "maximum_swaps": int(swaps.max())}


def decode_control(payload, template, species):
    previous.previous.CONTROL_PAYLOAD_HASH = CONTROL_PAYLOAD_HASH
    previous.previous.CONTROL_RAW_HASH = CONTROL_RAW_HASH
    return previous.previous.decode_control(payload, template, species)


def publish(export, template, ids, prediction, species, gate):
    tentative = export/"candidate_DO_NOT_SUBMIT.csv"
    proof = legacy.write_submission(tentative, template, ids, prediction, species)
    differs = proof["sha256"] != CONTROL_HASH
    eligible = bool(gate and differs)
    name = "GLC25_PA_submission_v32.csv" if eligible else (tentative.name if differs else "unchanged_v31_DO_NOT_SUBMIT.csv")
    if name != tentative.name:
        tentative.replace(export/name)
    return proof, {"eligible_for_submission": eligible, "different_from_v31": differs, "prediction_file": name,
        "message": "SUBMIT ONLY THIS CSV" if eligible else "DO NOT SUBMIT THIS OUTPUT; keep the scored v31"}


def save_compact(path, bundles, rows, store, per_fold=1500):
    # Reuse the tested compact format, explicitly relabel the two expert axes.
    translated = [{**b, "data": {"calibration": {"base": b["data"]["calibration"]["base"],
        "neural": b["data"]["calibration"]["bag"],
        "habitat": {**b["data"]["calibration"]["specialist"],
                    "distance": np.full(len(b["split"]["calibration"]), np.nan, np.float32)}}}} for b in bundles]
    previous.save_compact(path, translated, rows, store, per_fold)
    with np.load(path, allow_pickle=False) as archive:
        content = {k: archive[k] for k in archive.files if k != "habitat_distance"}
    content["expert_ids"] = np.array(["new_seed_bag", "asymmetric_rare_specialists"])
    content["reference_version"] = np.array("matched frozen v31 recipe, not exact historical weights")
    np.savez_compressed(path, **content)


def self_tests():
    assert len(POLICIES) == 22
    x = torch.tensor([[-100., 0., 100.]], requires_grad=True)
    loss = asymmetric_loss(x, torch.tensor([[0., .3, 1.]]), torch.ones(3))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    base = [list(range(10))]
    expert = {"rank": np.arange(30, 10, -1)[None], "value": np.ones((1, 20))}
    assert decode(base, None, None, POLICIES[0]) == base
    previous.validate_residual(base, decode(base, expert, expert, POLICIES[-1]), POLICIES[-1])
    return {"passed": True, "tests": 4}


def run_v32(control_b64):
    guard = legacy.RuntimeGuard(MAX_HOURS)
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
    temporary, export = working/"v32_runtime", working/"v32_export"
    legacy._clean_directory(temporary, working); legacy._clean_directory(export, working)
    temporary.mkdir(parents=True); export.mkdir(parents=True)
    store = None
    try:
        device = legacy.require_gpu()
        torch.set_num_threads(min(os.cpu_count() or 2, 6))
        tests = self_tests()
        root = legacy.discover_data_root()
        preflight = pd.read_csv(root/"GLC25_PA_metadata_train.csv", usecols=["surveyId", "lat", "lon", "country"]).drop_duplicates("surveyId").reset_index(drop=True)
        _, manifests = previous.previous.make_splits(preflight)
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
        splits, manifests = previous.previous.make_splits(rows)
        template = pd.read_csv(root/"GLC25_SAMPLE_SUBMISSION.csv")
        control = decode_control(control_b64, template, store.species_ids)
        order = pd.Index(template.surveyId).get_indexer(store.test_ids)
        if (order < 0).any() or not template.surveyId.is_unique:
            raise ValueError("Test/template IDs mismatch")
        control = [control[i] for i in order]
        proof = legacy.write_submission(temporary/"control.csv", template, store.test_ids, control, store.species_ids)
        if proof["sha256"] != CONTROL_HASH:
            raise ValueError("Exact scored v31 round trip failed")
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
        previous.validate_residual(control, prediction, policy)
        csv_proof, decision = publish(export, template, store.test_ids, prediction, store.species_ids, all(gates.values()))
        frame.to_csv(export/"regression_per_survey_v32.csv", index=False)
        save_compact(export/"calibration_top128_v32.npz", bundles, rows, store)
        report = {"experiment": EXPERIMENT, "status": "complete", **decision, "runtime_hours": guard.elapsed_hours(),
            "self_tests": tests, "policy": policy, "calibration_trials": trials, "gates": gates,
            "regression": regression, "submission_validation": csv_proof, "official_submission_made": False,
            "training": {"development": [{"fold": b["number"], "members": b["records"], "reference_members": b["reference_records"],
                "calibrations": b["calibrations"]} for b in bundles], "production": fitted},
            "production_diagnostics": diagnostics,
            "hardware": {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__},
            "validation_status": "all 88,987 PA IDs previously assessed; repeated spatial development, no fresh holdout",
            "limitations": ["No guarantee of hidden-test improvement, winning, or completion on an unmeasured GPU session.",
                "Reference v31 is a frozen-recipe refit; only the official-test control CSV is byte-exact.",
                "Repeated development gates and bootstrap are not independent evidence after many experiments.",
                "Production calibration transfers from other seeds (bag) or smaller training folds (specialists).",
                "Rare definitions depend only on training labels; branch membership may expand in full-data training.",
                "This is inspired by published competitor techniques, not a reproduction of their complete systems.",
                "Cardinality and leading 60% are frozen; useful larger changes can be rejected."]}
        legacy.save_json(export/"v32_report.json", report)
        manifest = {"experiment": EXPERIMENT, "source_sha256": V32_SOURCE_HASH, "embedded_sources_sha256": FROZEN_SOURCE_HASHES,
            "splits": manifests, "features": features, "bag_configs": BAG_CONFIGS, "bag_epochs": BAG_EPOCHS,
            "specialists": SPECIALISTS, "policies": POLICIES, "frozen_v31_policy": FROZEN_V31_POLICY,
            "control_sha256": CONTROL_HASH, "runtime_cap_hours": MAX_HOURS, "no_external_data_or_weights": True,
            "fresh_assessment": False, "outputs": {p.name: legacy.sha256_file(p) for p in sorted(export.iterdir())}}
        legacy.save_json(export/"v32_manifest.json", manifest)
        size = sum(p.stat().st_size for p in export.iterdir())
        if len(list(export.iterdir())) != 5 or size > 16_000_000:
            raise ValueError("Compact five-file/16MB export contract exceeded")
        return {"status": "complete", **decision, "runtime_hours": guard.elapsed_hours(), "output_bytes": size,
            "export_directory": str(export), "regression_gain": regression["gain"], "fresh_assessment": False, "policy": policy}
    except Exception as error:
        ready = export/"GLC25_PA_submission_v32.csv"
        if ready.exists():
            ready.replace(export/"failed_DO_NOT_SUBMIT.csv")
        legacy.save_json(export/"failure_report.json", {"experiment": EXPERIMENT, "status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "runtime_hours": guard.elapsed_hours(), "official_submission_made": False,
            "eligible_for_submission": False})
        raise
    finally:
        del store
        release(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        legacy._clean_directory(temporary, working)
