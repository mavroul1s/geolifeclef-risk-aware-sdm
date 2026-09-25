"""v29: full-data heterogeneous multisensor ensemble, no external weights.

The builder replaces the one repository import with a verified embedded v27 module.
v27 supplies the frozen reference recipe and established data readers only.
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

import scripts.v27_notebook_core as legacy

EXPERIMENT = "v29_full_data_multisensor_ensemble"
CONTROL_HASH = "32cd02788d910abe0cd18715da00f201a52a371a2135d17399301dd7e532f710"
MAX_HOURS = 10.75
SPLIT_SEED = 20260925
V27_POLICY = dict(legacy.POLICIES[-1])
CONFIGS = (
    {"id": "temporal_attention", "kind": "attention", "seed": 20262901,
     "width": 128, "epochs": 48, "geo": True, "gamma": 0.0},
    {"id": "multisensor_conv", "kind": "conv", "seed": 20262902,
     "width": 128, "epochs": 48, "geo": True, "gamma": 0.0},
    {"id": "ecology_attention", "kind": "attention", "seed": 20262903,
     "width": 128, "epochs": 48, "geo": False, "gamma": 2.0},
)
POLICIES = ({"id": "control", "alpha": 0.0, "count_scale": 1.0},) + tuple(
    {"id": f"ensemble_a{int(alpha*100)}_k{int(scale*100)}",
     "alpha": alpha, "count_scale": scale}
    for alpha in (0.25, 0.5, 0.75, 1.0) for scale in (0.8, 1.0, 1.2))


def decode_payloads(control_b64, consumed_b64, template, species):
    packed = base64.b64decode(control_b64)
    if legacy.sha256_bytes(packed) != CONTROL_PAYLOAD_HASH:
        raise ValueError("Frozen v27 payload hash mismatch")
    raw = lzma.decompress(packed)
    if legacy.sha256_bytes(raw) != CONTROL_RAW_HASH:
        raise ValueError("Frozen v27 raw hash mismatch")
    n = len(template)
    counts, flat = np.frombuffer(raw[:n], np.uint8), np.frombuffer(raw[n:], "<u2")
    if (n != legacy.EXPECTED_TEST_ROWS or len(species) != legacy.EXPECTED_SPECIES or
            counts.sum() != len(flat) or counts.min() < 8 or counts.max() > 40 or
            flat.max() >= len(species)):
        raise ValueError("Frozen v27 dimensions invalid")
    bounds = np.r_[0, np.cumsum(counts)]
    predictions = [flat[a:b].astype(int).tolist() for a, b in zip(bounds[:-1], bounds[1:])]
    if any(len(row) != len(set(row)) for row in predictions):
        raise ValueError("Duplicate species in frozen control")
    return predictions, decode_consumed(consumed_b64)


def decode_consumed(payload):
    packed = base64.b64decode(payload)
    if legacy.sha256_bytes(packed) != CONSUMED_HASH:
        raise ValueError("Consumed-ID hash mismatch")
    ids = np.cumsum(np.frombuffer(lzma.decompress(packed), "<u4"), dtype=np.int64)
    if len(ids) != CONSUMED_COUNT or np.any(np.diff(ids) <= 0):
        raise ValueError("Invalid consumed IDs")
    return ids


def make_split(rows, consumed, *, minimums=None):
    """One final untouched audit; all consumed labels are explicitly development data."""
    blocks = legacy.spatial_blocks(rows)
    audit = ~np.isin(rows.surveyId.to_numpy(np.int64), consumed)
    audit_blocks = np.isin(blocks, np.unique(blocks[audit]))
    bucket = np.asarray([legacy.stable_bucket(f"v29-dev:{SPLIT_SEED}:{b}") for b in blocks])
    selection = (bucket < 10) & ~audit_blocks
    calibration = (bucket >= 10) & (bucket < 20) & ~audit_blocks
    candidates = np.flatnonzero((bucket >= 20) & ~audit_blocks)
    coords = rows[["lat", "lon"]].to_numpy(np.float64)
    evaluation = audit | selection | calibration
    if not len(candidates) or not evaluation.any():
        raise ValueError("Empty registered split")
    distance = legacy.nearest_distance_km(coords[evaluation], coords[candidates])
    split = {"training": candidates[distance >= 20], "selection": np.flatnonzero(selection),
             "calibration": np.flatnonzero(calibration), "assessment": np.flatnonzero(audit)}
    limits = minimums or {"training": 10000, "selection": 500, "calibration": 500,
                          "assessment": 1000}
    if any(len(split[k]) < v for k, v in limits.items()):
        raise ValueError(f"Registered v29 split too small: { {k:len(v) for k,v in split.items()} }")
    for role in ("selection", "calibration", "assessment"):
        if set(blocks[split[role]]) & set(blocks[split["training"]]):
            raise ValueError("Training shares a held-out block")
    minimum = float(legacy.nearest_distance_km(coords[split["training"]], coords[evaluation]).min())
    return split, {"seed": SPLIT_SEED, "block_degrees": 1.0, "buffer_km": 20,
                   "minimum_evaluation_distance_km": minimum,
                   "counts": {k: len(v) for k, v in split.items()},
                   "blocks": {k: len(np.unique(blocks[v])) for k, v in split.items()},
                   "assessment_ids_sha256": legacy.sha256_bytes(
                       rows.surveyId.to_numpy(np.int64)[split["assessment"]].astype("<i8").tobytes()),
                   "consumed_ids": len(consumed), "development_labels_previously_observed": True,
                   "fresh_assessment": True, "labels_used_for_assignment": False,
                   "audit_countries": rows.iloc[split["assessment"]].country.value_counts().to_dict()}


def sentinel_band_indices(descriptions, color_names):
    """Canonical RGB-NIR: prefer TIFF metadata; otherwise published GLC RGB-NIR.

    Do not change the frozen 32px reference reader, including its old index convention.
    """
    aliases = ({"red", "b04", "b4"}, {"green", "b03", "b3"},
               {"blue", "b02", "b2"}, {"nir", "b08", "b8", "near infrared", "near-infrared"})
    descriptions = [str(value or "").strip().lower() for value in descriptions]
    colors = [str(value or "").lower() for value in color_names]
    indices = []
    for names in aliases:
        found = [i for i in range(4) if descriptions[i] in names or colors[i] in names]
        indices.append(found[0] if len(found) == 1 else None)
    if all(value is not None for value in indices[:3]) and indices[3] is None:
        remaining = set(range(4)) - set(indices[:3])
        if len(remaining) == 1:
            indices[3] = remaining.pop()
    if all(value is not None for value in indices) and len(set(indices)) == 4:
        return indices
    if any(value is not None and value != i for i, value in enumerate(indices)):
        raise ValueError(f"Partially specified conflicting Sentinel band order: {descriptions}, {colors}")
    return [0, 1, 2, 3]


def read_multiresolution(task):
    # Preserve the old rasterio 32-pixel interpolation exactly for the reference.
    original = legacy.extract_remote_features(task)
    import rasterio
    from rasterio.enums import Resampling
    path = legacy.feature_paths(Path(task[0]), task[1], task[2])[2]
    with rasterio.open(path) as dataset:
        image = dataset.read(out_shape=(4, 64, 64), out_dtype="float32",
                             resampling=Resampling.bilinear)
        order = sentinel_band_indices(dataset.descriptions, [x.name for x in dataset.colorinterp])
        image = image[order]
    image = np.clip(np.nan_to_num(image / 10000, nan=0, posinf=0, neginf=0), 0, 2)
    return (*original, image.astype(np.float16))


def write_multiresolution(root, rows, source, prefix, cache, guard, workers):
    ids = rows.surveyId.to_numpy(np.int64)
    dimensions = [(name, (width,), np.float32) for name, width in legacy.REMOTE_DIMS.items()]
    dimensions += [(f"{name}_raster", shape, np.float16)
                   for name, shape in legacy.RASTER_SHAPES.items()]
    dimensions += [("sentinel64", (4, 64, 64), np.float16)]
    arrays = [np.lib.format.open_memmap(cache / f"{prefix}_{name}.npy", mode="w+",
                                      dtype=dtype, shape=(len(ids), *shape))
              for name, shape, dtype in dimensions]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for begin in range(0, len(ids), 128):
            guard.require(7 * 3600, "dual-resolution feature preparation")
            tasks = [(str(root), source, int(sid)) for sid in ids[begin:begin + 128]]
            for offset, features in enumerate(executor.map(read_multiresolution, tasks)):
                for array, feature in zip(arrays, features):
                    array[begin + offset] = feature
            if begin % 2048 == 0:
                guard.stamp("prepare_64px", split=prefix, completed=begin + len(tasks), total=len(ids))
    for array in arrays:
        array.flush()
    return {"rows": len(ids), "seconds": time.monotonic() - started, "sentinel_pixels": 64}


def candidate_static(rows, geo=True):
    def numeric(name, default):
        value = rows[name] if name in rows else pd.Series(default, index=rows.index)
        return pd.to_numeric(value, errors="coerce").fillna(default).to_numpy(np.float32)
    area = np.log1p(np.maximum(numeric("areaInM2", 0), 0)) / 10
    year = (numeric("year", 2019) - 2019) / 5
    values = [area, year]
    if geo:
        lat, lon = numeric("lat", 0), numeric("lon", 0)
        values.extend([lat / 90, lon / 180])
        for frequency in (1, 2, 4, 8):
            for angle in (lat, lon):
                values.extend([np.sin(np.deg2rad(angle) * frequency),
                               np.cos(np.deg2rad(angle) * frequency)])
        # Explicit categories avoid collisions between unrelated countries.
        countries = rows.country.fillna("unknown").astype(str).to_numpy()
        for country in ("Denmark", "Netherlands", "France", "Italy"):
            values.append((countries == country).astype(np.float32))
        values.append((~np.isin(countries, ["Denmark", "Netherlands", "France", "Italy"])).astype(np.float32))
    return np.stack(values, axis=1).astype(np.float32)


def fit_normalization(store, indices, *, chunk_size=256):
    """Streaming statistics: never allocate N x 64 x 64 in memory."""
    result = {"vector": legacy.normalization_stats(store.train, indices), "raster": {}}
    sources = {**store.raster_train, "sentinel": store.high_train}
    for name, array in sources.items():
        total = np.zeros(array.shape[1], np.float64)
        total_sq = total.copy()
        count = 0
        sampled = np.sort(indices)[::max(1, len(indices) // 6000)]
        for begin in range(0, len(sampled), chunk_size):
            x = np.asarray(array[sampled[begin:begin + chunk_size]], np.float32)
            x = x.reshape(len(x), x.shape[1], -1)
            total += x.sum((0, 2), dtype=np.float64)
            total_sq += np.square(x, dtype=np.float64).sum((0, 2))
            count += x.shape[0] * x.shape[2]
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-8))
        result["raster"][name] = {"mean": mean.astype(np.float32),
                                   "std": np.maximum(std, 1e-4).astype(np.float32)}
    return result


def batch_inputs(store, indices, stats, device, *, test=False, augment=False, rng=None, view=0):
    vectors, rasters = (store.test, store.raster_test) if test else (store.train, store.raster_train)
    high = store.high_test if test else store.high_train
    out = {}
    for name in ("environment",):
        values = np.asarray(vectors[name][indices], np.float32)
        mean, std = stats["vector"][name]["mean"], stats["vector"][name]["std"]
        values = np.clip(np.nan_to_num((values - mean) / std, nan=0, posinf=0, neginf=0), -8, 8)
        out[name] = torch.from_numpy(values).to(device)
    out["static_geo"] = torch.from_numpy((store.static_test if test else store.static_train)[indices]).to(device)
    out["static_eco"] = torch.from_numpy((store.eco_test if test else store.eco_train)[indices]).to(device)
    for name in legacy.RASTER_MODALITIES:
        raw = np.asarray((high if name == "sentinel" else rasters[name])[indices], np.float32)
        mean = stats["raster"][name]["mean"][None, :, None, None]
        std = stats["raster"][name]["std"][None, :, None, None]
        values = np.clip(np.nan_to_num((raw - mean) / std, nan=0, posinf=0, neginf=0), -8, 8)
        if name == "sentinel":
            red, green, blue, nir = [raw[:, k] for k in range(4)]
            ndvi = (nir - red) / np.maximum(nir + red, 1e-4)
            ndwi = (green - nir) / np.maximum(green + nir, 1e-4)
            values = np.concatenate([values, np.stack([ndvi, ndwi], axis=1)], axis=1)
            turns = int(rng.integers(4)) if augment else view
            if turns:
                values = np.rot90(values, turns, axes=(-2, -1))
            if augment and rng.random() < 0.5:
                values = values[..., ::-1]
        out[name] = torch.from_numpy(np.ascontiguousarray(values)).to(device)
    return out


class SensorAttention(nn.Module):
    def __init__(self, env_dim, static_dim, species, width=128):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
                                   legacy.ConvResidual(32), nn.Conv2d(32, 64, 3, 2, 1),
                                   nn.GroupNorm(8, 64), nn.GELU(), legacy.ConvResidual(64),
                                   nn.Conv2d(64, width, 3, 2, 1))
        self.land = nn.Linear(6 * 4, width)
        self.climate = nn.Linear(4 * 12, width)
        self.environment = nn.Sequential(nn.Linear(env_dim, width), nn.GELU(), nn.LayerNorm(width))
        self.static = nn.Linear(static_dim, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, 107, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 3, 0.15, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width * 2)
        self.head = nn.Linear(width * 2, species)
        self.richness = nn.Linear(width * 2, 1)

    def forward(self, x, geo=True):
        image = self.image(x["sentinel"]).flatten(2).transpose(1, 2)  # 64 spatial tokens
        land = self.land(x["landsat"].permute(0, 3, 1, 2).flatten(2))  # 21 years, 24 values
        climate = self.climate(x["bioclim"].permute(0, 2, 1, 3).flatten(2))  # 19 years, 48 values
        env = self.environment(x["environment"])[:, None]
        static = self.static(x["static_geo" if geo else "static_eco"])[:, None]
        if self.training:
            # Whole-sensor dropout is independent of species labels.
            image, land, climate, env, static = [
                sensor * torch.bernoulli(sensor.new_full((len(sensor), 1, 1), 0.9))
                for sensor in (image, land, climate, env, static)]
        tokens = torch.cat([self.cls.expand(len(image), -1, -1), image, land, climate, env, static], 1)
        encoded = self.encoder(tokens + self.position[:, :tokens.shape[1]])
        feature = self.norm(torch.cat([encoded[:, 0], encoded[:, 1:].mean(1)], 1))
        return self.head(feature), self.richness(feature).squeeze(1)


class SensorConv(nn.Module):
    def __init__(self, env_dim, static_dim, species, width=128):
        super().__init__()
        self.image = legacy.PyramidRasterEncoder(6, width=32, output=width)
        def temporal(channels):
            return nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(width, width, 5, padding=2, groups=width),
                                  nn.Conv1d(width, width, 1), nn.GELU(),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.land, self.climate = temporal(6), temporal(4)
        self.vector = nn.Sequential(nn.Linear(env_dim + static_dim, width), nn.GELU(),
                                    legacy.ResidualVectorBlock(width, 0.2))
        self.fusion = nn.Sequential(nn.Linear(width * 4, width * 3), nn.GELU(),
                                    legacy.ResidualVectorBlock(width * 3, 0.2), nn.LayerNorm(width * 3))
        self.head, self.richness = nn.Linear(width * 3, species), nn.Linear(width * 3, 1)

    def forward(self, x, geo=True):
        # Chronological order is year, then season/month; no artificial 2D locality.
        land = x["landsat"].permute(0, 1, 3, 2).flatten(2)
        climate = x["bioclim"].flatten(2)
        static = x["static_geo" if geo else "static_eco"]
        feature = self.fusion(torch.cat([self.image(x["sentinel"]), self.land(land),
                                        self.climate(climate),
                                        self.vector(torch.cat([x["environment"], static], 1))], 1))
        return self.head(feature), self.richness(feature).squeeze(1)


def create_model(store, config, indices):
    # Seed BEFORE construction, so weights as well as minibatches are reproducible.
    legacy.set_seed(config["seed"])
    constructor = SensorAttention if config["kind"] == "attention" else SensorConv
    static_dim = store.static_train.shape[1] if config["geo"] else store.eco_train.shape[1]
    model = constructor(store.dims["environment"], static_dim, len(store.species_ids), config["width"])
    frequencies = legacy._frequency(store.labels, indices)
    prior = np.clip((frequencies + 0.5) / (len(indices) + 1), 1e-5, 0.95)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32))
        model.richness.bias.fill_(math.log1p(frequencies.sum() / max(len(indices), 1)))
    return model


@torch.no_grad()
def predict(model, store, indices, stats, device, config, *, test=False, views=1, guard=None):
    model.eval()
    out = np.empty((len(indices), len(store.species_ids)), np.float16)
    batch_size = 96 if device.type == "cuda" else 16
    for start in range(0, len(indices), batch_size):
        if guard:
            guard.require(10 * 60, "candidate inference")
        take = indices[start:start + batch_size]
        probability = None
        for view in range(views):
            x = batch_inputs(store, take, stats, device, test=test, view=view)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, _ = model(x, config["geo"])
            current = logits.float().sigmoid()
            probability = current if probability is None else probability + current
        out[start:start + len(take)] = (probability / views).cpu().numpy().astype(np.float16)
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite candidate predictions")
    return out


def sample_weights(rows, indices):
    countries = rows.iloc[indices].country.fillna("unknown").astype(str)
    counts = countries.value_counts()
    values = np.power(len(indices) / np.maximum(countries.map(counts).to_numpy(), 1), 0.25)
    values = np.clip(values / np.median(values), 0.5, 3.0)
    return values / values.sum()


def train_candidate(store, rows, indices, selection, stats, config, output, guard, device,
                    *, fixed_epochs=None, phase="development", training_budget_seconds=None):
    training_started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = create_model(store, config, indices).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    epochs = int(fixed_epochs or config["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config["seed"])
    weights = sample_weights(rows, indices)
    frequencies = legacy._frequency(store.labels, indices)
    positive = np.clip(np.power(20 / np.maximum(frequencies, 1), 0.25), 1, 3)
    positive = torch.tensor(positive, dtype=torch.float32, device=device)
    best, best_epoch, histories = -1.0, 0, []
    batch_size = 64 if device.type == "cuda" else 16
    checkpoint = output / f"{phase}_{config['id']}.pt"
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        # Half the epoch is uniform coverage; half reduces country dominance.
        uniform = rng.permutation(indices)[:len(indices) // 2]
        balanced = rng.choice(indices, size=len(indices) - len(uniform), p=weights)
        order = rng.permutation(np.concatenate([uniform, balanced]))
        model.train()
        total, seen = 0.0, 0
        for begin in range(0, len(order), batch_size):
            guard.require(45 * 60, f"{phase} training")
            take = order[begin:begin + batch_size]
            x = batch_inputs(store, take, stats, device, augment=True, rng=rng)
            target = torch.from_numpy(np.asarray(store.labels[take], np.float32)).to(device)
            if len(take) > 1 and rng.random() < 0.5:
                mix = float(rng.beta(0.2, 0.2))
                permutation = torch.randperm(len(take), device=device)
                x = {key: mix * value + (1 - mix) * value[permutation] for key, value in x.items()}
                target = mix * target + (1 - mix) * target[permutation]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, richness = model(x, config["geo"])
                logits = logits.float()
                per_label = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive,
                                                               reduction="none")
                if config["gamma"]:
                    per_label *= target + (1 - target) * logits.sigmoid().pow(config["gamma"])
                loss = 100 * per_label.mean() + 0.04 * F.smooth_l1_loss(
                    richness.float(), torch.log1p(target.sum(1)))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite v29 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.lerp_(current, 0.02)
            total += float(loss.detach()) * len(take)
            seen += len(take)
        scheduler.step()
        score = None
        if fixed_epochs is None and (epoch == 1 or epoch % 3 == 0 or epoch == epochs):
            p = predict(ema, store, selection, stats, device, config, guard=guard)
            ranked, _ = legacy.top_rank(p, min(24, p.shape[1]))
            target = np.asarray(store.labels[selection])
            score = float(np.mean([legacy.f1_from_ranked(target, ranked, np.full(len(p), k)).mean()
                                   for k in (min(12, p.shape[1]), min(18, p.shape[1]), min(24, p.shape[1]))]))
            if score > best:
                best, best_epoch = score, epoch
                torch.save(ema.state_dict(), checkpoint)
        record = {"epoch": epoch, "loss": total / max(seen, 1), "selection_f1": score,
                  "seconds": time.monotonic() - start}
        histories.append(record)
        guard.stamp("v29_train", model=config["id"], phase=phase, **record)
        if fixed_epochs is None and epoch >= 15 and epoch - best_epoch >= 12:
            break
        if (fixed_epochs is None and training_budget_seconds is not None and epoch >= 3
                and time.monotonic() - training_started >= training_budget_seconds):
            break
        # Adapt before budget exhaustion, keeping a valid EMA checkpoint.
        if fixed_epochs is None and epoch >= 6 and guard.remaining_seconds() < 5 * 3600:
            break
    if fixed_epochs is not None:
        torch.save(ema.state_dict(), checkpoint)
        best_epoch = epochs
    elif best_epoch < 1:
        raise RuntimeError("No valid checkpoint")
    ema.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ema, {"config": config, "best_epoch": best_epoch,
                 "selection_f1": best if fixed_epochs is None else None,
                 "history": histories, "training_surveys": len(indices),
                 "seconds": time.monotonic() - training_started,
                 "parameters": sum(p.numel() for p in ema.parameters()),
                 "peak_gpu_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                               if device.type == "cuda" else None),
                 "all_species_outputs_trained": len(store.species_ids),
                 "checkpoint_sha256": legacy.sha256_file(checkpoint)}


def fit_platt(probabilities, labels):
    """Fit a shared temperature/intercept to all species, with bounded binned likelihood."""
    edges = np.linspace(-16, 12, 257)
    counts, positives, sums = (np.zeros(256, np.float64) for _ in range(3))
    for begin in range(0, len(probabilities), 128):
        p = np.clip(np.asarray(probabilities[begin:begin + 128], np.float64), 1e-7, 1 - 1e-7)
        z = np.log(p / (1 - p)).ravel()
        target = np.asarray(labels[begin:begin + 128], np.float64).ravel()
        bins = np.clip(np.searchsorted(edges, z) - 1, 0, 255)
        counts += np.bincount(bins, minlength=256)
        positives += np.bincount(bins, weights=target, minlength=256)
        sums += np.bincount(bins, weights=z, minlength=256)
    x = sums / np.maximum(counts, 1)
    def objective(theta):
        logits = np.exp(theta[0]) * x + theta[1]
        return float((counts * np.logaddexp(0, logits) - positives * logits).sum() / counts.sum())
    fit = minimize(objective, [0., 0.], method="L-BFGS-B", bounds=[(-1.5, 1.5), (-8, 8)])
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise FloatingPointError("Probability calibration failed")
    return {"slope": float(np.exp(fit.x[0])), "intercept": float(fit.x[1]),
            "nll": float(fit.fun), "fit_rows": len(probabilities), "optimizer_success": bool(fit.success)}


def calibrated(probabilities, calibration):
    p = np.clip(np.asarray(probabilities, np.float32), 1e-7, 1 - 1e-7)
    return expit(calibration["slope"] * np.log(p / (1 - p)) + calibration["intercept"]).astype(np.float32)


def full_refit_estimate(records, full_rows, development_rows):
    """Conservative whole-ensemble admission, not three independent optimistic checks."""
    return float(sum(np.median([x["seconds"] for x in r["history"]]) *
                     full_rows / development_rows * r["best_epoch"] * 1.4 for r in records)
                 + 45 * 60)


def decode_ensemble(base, probability, policy):
    if policy["alpha"] == 0:
        return [list(row) for row in base]
    ranked, values = legacy.top_rank(probability, min(64, probability.shape[1]))
    minimum, maximum = min(8, probability.shape[1]), min(40, probability.shape[1])
    k = np.arange(1, maximum + 1)
    # Ratio-of-expectations F1 surrogate, not an exact expected-F1 claim.
    objective = 2 * np.cumsum(values[:, :maximum], 1) / (
        k[None, :] + probability.sum(1)[:, None] * policy["count_scale"] + 1e-8)
    counts = objective[:, minimum - 1:].argmax(1) + minimum
    alpha = policy["alpha"]
    output = []
    for row, old in enumerate(base):
        if alpha == 1:
            output.append(ranked[row, :counts[row]].astype(int).tolist())
            continue
        scores = {int(column): (1 - alpha) * (1 - 0.7 * rank / max(len(old) - 1, 1))
                  for rank, column in enumerate(old)}
        for rank, column in enumerate(ranked[row]):
            scores[int(column)] = scores.get(int(column), 0) + alpha * (1 - 0.85 * rank / max(len(ranked[row]) - 1, 1))
        count = int(np.clip(round((1 - alpha) * len(old) + alpha * counts[row]), minimum, maximum))
        output.append([column for column, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:count]])
    return output


def select_policy(base, probabilities, targets, rows):
    base_score = legacy.score_prediction_lists(targets, base)
    countries = rows.country.fillna("unknown").astype(str).to_numpy()
    blocks = legacy.spatial_blocks(rows)
    trials = []
    for policy in POLICIES:
        predictions = decode_ensemble(base, probabilities, policy)
        scores = legacy.score_prediction_lists(targets, predictions)
        delta = scores - base_score
        country_delta = [float(delta[countries == c].mean()) for c in np.unique(countries)
                         if np.count_nonzero(countries == c) >= 30]
        block_delta = [float(delta[blocks == b].mean()) for b in np.unique(blocks)]
        robust = 0.7 * delta.mean() + 0.15 * np.mean(country_delta or [0]) + 0.15 * np.mean(block_delta)
        allowed = bool(policy["alpha"] == 0 or (delta.mean() > 0 and robust > 0))
        trials.append({**policy, "sample_f1": float(scores.mean()), "gain": float(delta.mean()),
                       "robust_gain": float(robust), "allowed": allowed})
    chosen = max((t for t in trials if t["allowed"]), key=lambda t: (t["robust_gain"], -t["alpha"]))
    return {key: chosen[key] for key in ("id", "alpha", "count_scale")}, trials


def multilabel_summary(targets, predictions):
    gold = targets.sum(0, dtype=np.int64)
    predicted = np.zeros(targets.shape[1], np.int64)
    true_positive = predicted.copy()
    for i, columns in enumerate(predictions):
        predicted[columns] += 1
        true_positive[columns] += targets[i, columns].astype(np.int64)
    denominator = gold + predicted
    return {"micro_f1": float(2 * true_positive.sum() / max(denominator.sum(), 1)),
            "macro_f1_all_species": float(np.divide(2 * true_positive, denominator,
                out=np.zeros(len(gold), np.float64), where=denominator > 0).mean()),
            "macro_zero_denominator_value": 0,
            "precision": float(true_positive.sum() / max(predicted.sum(), 1)),
            "recall": float(true_positive.sum() / max(gold.sum(), 1)),
            "species_predicted": int((predicted > 0).sum())}


def audit_report(base, probability, policy, targets, rows, frequencies=None):
    predictions = decode_ensemble(base, probability, policy)
    old = legacy.score_prediction_lists(targets, base)
    new = legacy.score_prediction_lists(targets, predictions)
    frame = pd.DataFrame({"surveyId": rows.surveyId.to_numpy(), "country": rows.country.to_numpy(),
                          "spatial_block": legacy.spatial_blocks(rows), "matched_v27_f1": old,
                          "v29_f1": new, "delta_f1": new - old,
                          "true_cardinality": targets.sum(1),
                          "predicted_cardinality": list(map(len, predictions))})
    bootstrap = legacy.paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy(),
                                             iterations=1000, seed=20262904)
    report = {"surveys": len(frame), "matched_v27_f1": float(old.mean()), "v29_f1": float(new.mean()),
              "gain": float((new - old).mean()), "bootstrap": bootstrap,
              "by_country": legacy.summarize_by_group(frame, "country", ("matched_v27_f1", "v29_f1", "delta_f1")),
              "multilabel": {"matched_v27": multilabel_summary(targets, base),
                             "v29": multilabel_summary(targets, predictions)},
              "cardinality": {"true_mean": float(targets.sum(1).mean()),
                              "predicted_mean": float(frame.predicted_cardinality.mean()),
                              "mae": float(np.abs(frame.true_cardinality - frame.predicted_cardinality).mean())},
              "now_consumed": True, "used_for_selection": False,
              "warning": "Single small fresh geographic audit; not representative of all test countries. "
                         "Matched v27 is a recipe refit with one pyramid seed; exact deployed v27 is embedded only for test."}
    if frequencies is not None:
        report["species_groups"] = {"matched_v27": legacy.species_group_metrics(targets, base, frequencies),
                                    "v29": legacy.species_group_metrics(targets, predictions, frequencies)}
    return frame, report


def publish_predictions(export, template, ids, predictions, species, gate):
    """A failed gate must never expose a file that looks ready for submission."""
    tentative = export / "candidate_DO_NOT_SUBMIT.csv"
    proof = legacy.write_submission(tentative, template, ids, predictions, species)
    differs = proof["sha256"] != CONTROL_HASH
    eligible = bool(gate and differs)
    name = "GLC25_PA_submission_v29.csv" if eligible else (
        "candidate_DO_NOT_SUBMIT.csv" if differs else "unchanged_v27_DO_NOT_SUBMIT.csv")
    destination = export / name
    if destination != tentative:
        tentative.replace(destination)
    return proof, {"eligible_for_submission": eligible, "different_from_v27": differs,
                   "prediction_file": name,
                   "message": ("SUBMIT THIS CSV ONCE" if eligible else "DO NOT SUBMIT: candidate did not pass the gate")}


def self_tests():
    base = [list(range(8))]
    p = np.full((1, 12), 0.01, np.float32)
    p[0, 4:12] = 0.8
    assert decode_ensemble(base, p, POLICIES[0]) == base
    new = decode_ensemble(base, p, {"alpha": 1., "count_scale": 1.})
    assert set(new[0]) == set(range(4, 12))
    rows = pd.DataFrame({"lat": [40., 60.], "lon": [20., 5.], "country": ["France", "Ukraine"],
                         "areaInM2": [10, 10], "year": [2020, 2020]})
    assert np.array_equal(candidate_static(rows, False)[0], candidate_static(rows, False)[1])
    assert candidate_static(rows, True).shape == (2, 25)
    for kind in (SensorAttention, SensorConv):
        model = kind(5, 2, 12, width=16).eval()
        with torch.no_grad():
            logits, count = model({"environment": torch.zeros(2, 5), "static_eco": torch.zeros(2, 2),
                                   "sentinel": torch.zeros(2, 6, 64, 64),
                                   "landsat": torch.zeros(2, 6, 4, 21),
                                   "bioclim": torch.zeros(2, 4, 19, 12)}, False)
        assert logits.shape == (2, 12) and count.shape == (2,) and torch.isfinite(logits).all()
    return {"passed": True, "tests": 5}


def run_v29(control_b64, consumed_b64):
    guard = legacy.RuntimeGuard(MAX_HOURS)
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
    temporary, export = working / "v29_runtime", working / "v29_export"
    # Own version-specific directories only; never touch a previous experiment's files.
    legacy._clean_directory(temporary, working)
    legacy._clean_directory(export, working)
    temporary.mkdir(parents=True)
    export.mkdir(parents=True)
    store = None
    try:
        device = legacy.require_gpu()
        torch.set_num_threads(min(os.cpu_count() or 2, 6))
        tests = self_tests()
        root = legacy.discover_data_root()
        consumed = decode_consumed(consumed_b64)
        preflight = pd.read_csv(root / "GLC25_PA_metadata_train.csv", usecols=["surveyId", "lat", "lon", "country"])
        preflight = preflight.drop_duplicates("surveyId").reset_index(drop=True)
        _, split_manifest = make_split(preflight, consumed)
        guard.stamp("preflight", **split_manifest)
        original_writer = legacy._write_remote_arrays
        try:
            legacy._write_remote_arrays = write_multiresolution
            features = legacy.prepare_feature_store(root, temporary / "features", guard, workers=6)
        finally:
            legacy._write_remote_arrays = original_writer
        store = legacy.FeatureStore(temporary / "features")
        store.high_train = np.load(store.cache / "train_sentinel64.npy", mmap_mode="r")
        store.high_test = np.load(store.cache / "test_sentinel64.npy", mmap_mode="r")
        rows, test_rows, pairs = legacy.load_rows_and_pairs(root, store.train_ids, store.test_ids)
        del pairs, preflight
        template = pd.read_csv(root / "GLC25_SAMPLE_SUBMISSION.csv")
        # Avoid copying multiple GB of test rasters just to change row order.
        test_order = pd.Index(template.surveyId).get_indexer(store.test_ids)
        if (test_order < 0).any() or not template.surveyId.is_unique:
            raise ValueError("Test/template IDs differ")
        control, _ = decode_payloads(control_b64, consumed_b64, template, store.species_ids)
        control = [control[i] for i in test_order]
        proof = legacy.write_submission(temporary / "control.csv", template, store.test_ids, control, store.species_ids)
        if proof["sha256"] != CONTROL_HASH:
            raise ValueError("Exact v27 round trip failed")
        store.static_train, store.static_test = candidate_static(rows), candidate_static(test_rows)
        store.eco_train, store.eco_test = candidate_static(rows, False), candidate_static(test_rows, False)
        split, split_manifest = make_split(rows, consumed)
        po = legacy.POGridIndex.build(root / "GLC25_P0_metadata_train.csv", store.species_ids,
                                      rows[["lat", "lon"]].to_numpy(), guard)
        legacy.set_seed(20262900)
        bundle, reference_record = legacy._build_models_for_fold(
            "reference", split, rows, store, po, temporary, guard, device, 20262900)
        reference = {}
        for role in ("selection", "calibration", "assessment"):
            values, components = bundle["predictions"][role], bundle["components"][role]
            reference[role] = legacy.compose_predictions(
                values["base_lists"], values["candidate"], values["predicted_count"], bundle["frequencies"],
                components["spatial"], components["po"], bundle["graph"], values["risk"], V27_POLICY)
        del bundle, po, values, components
        gc.collect()
        stats = fit_normalization(store, split["training"])
        averages = {role: np.zeros((len(split[role]), len(store.species_ids)), np.float32)
                    for role in ("calibration", "assessment")}
        records, calibrations = [], []
        for model_index, config in enumerate(CONFIGS):
            guard.require(4 * 3600, "start development ensemble member")
            development_budget = max(60., (guard.remaining_seconds() - 3600) /
                                     (len(CONFIGS) - model_index + 3 * len(CONFIGS)))
            model, record = train_candidate(store, rows, split["training"], split["selection"],
                stats, config, temporary, guard, device, training_budget_seconds=development_budget)
            selection_p = predict(model, store, split["selection"], stats, device, config, views=2, guard=guard)
            calibration = fit_platt(selection_p, np.asarray(store.labels[split["selection"]]))
            for role in ("calibration", "assessment"):
                p = predict(model, store, split[role], stats, device, config, views=2, guard=guard)
                calibrated_p = calibrated(p, calibration)
                averages[role] += calibrated_p / len(CONFIGS)
                if role == "calibration":
                    standalone = decode_ensemble(reference[role], calibrated_p,
                                                 {"alpha": 1., "count_scale": 1.})
                    record["calibration_standalone_f1"] = float(legacy.score_prediction_lists(
                        np.asarray(store.labels[split[role]]), standalone).mean())
                    record["calibration_predicted_count"] = float(np.mean(list(map(len, standalone))))
                del p, calibrated_p
            records.append(record)
            calibrations.append(calibration)
            del model, selection_p
            gc.collect()
            torch.cuda.empty_cache()
        policy, trials = select_policy(reference["calibration"], averages["calibration"],
                                        np.asarray(store.labels[split["calibration"]]), rows.iloc[split["calibration"]])
        guard.stamp("policy_frozen", policy=policy, trials=trials)
        # Standalone ensemble diagnostics do not determine the policy after the audit.
        deployment_records = []
        if policy["alpha"] == 0:
            predictions = control
        else:
            all_training = np.arange(len(rows), dtype=np.int64)
            guard.require(full_refit_estimate(records, len(rows), len(split["training"])),
                          "whole full-data ensemble admission")
            full_stats = fit_normalization(store, all_training)
            test_probability = np.zeros((len(test_rows), len(store.species_ids)), np.float32)
            for config, record, calibration in zip(CONFIGS, records, calibrations):
                duration = np.median([x["seconds"] for x in record["history"]])
                expected = duration * len(rows) / len(split["training"]) * record["best_epoch"]
                guard.require(expected * 1.4 + 45 * 60, "full-data refit admission")
                full_config = {**config, "seed": config["seed"] + 100}
                model, fitted = train_candidate(store, rows, all_training, np.array([], np.int64),
                    full_stats, full_config, temporary, guard, device,
                    fixed_epochs=record["best_epoch"], phase="full_data")
                p = predict(model, store, np.arange(len(test_rows)), full_stats, device, full_config,
                            test=True, views=2, guard=guard)
                test_probability += calibrated(p, calibration) / len(CONFIGS)
                deployment_records.append(fitted)
                del model, p
                gc.collect()
                torch.cuda.empty_cache()
            predictions = decode_ensemble(control, test_probability, policy)
        # Freeze production and audit predictions before audit scoring. Production
        # refits have used all PA labels, but never alter the held-out predictor.
        audit_predictions = decode_ensemble(reference["assessment"], averages["assessment"], policy)
        freeze = {"policy": policy, "production_prediction_sha256": legacy.sha256_bytes(
                      json.dumps(predictions, separators=(",", ":")).encode()),
                  "audit_prediction_sha256": legacy.sha256_bytes(
                      json.dumps(audit_predictions, separators=(",", ":")).encode()),
                  "all_choices_frozen_before_assessment": True}
        legacy.save_json(temporary / "pre_assessment_freeze.json", freeze)
        frame, audit = audit_report(reference["assessment"], averages["assessment"], policy,
                                    np.asarray(store.labels[split["assessment"]]), rows.iloc[split["assessment"]],
                                    legacy._frequency(store.labels, split["training"]))
        integrity = {"exact_control": proof["sha256"] == CONTROL_HASH,
                     "fresh_audit": not np.intersect1d(rows.surveyId.to_numpy()[split["assessment"]], consumed).size,
                     "twenty_km_buffer": split_manifest["minimum_evaluation_distance_km"] >= 20,
                     "all_5016_species": len(store.species_ids) == 5016,
                     "choices_frozen": True, "no_external_data_or_weights": True,
                     "runtime_within_limit": guard.elapsed_hours() < MAX_HOURS}
        gate = {"calibration_selected_new_model": policy["alpha"] > 0,
                "fresh_audit_gain_positive": audit["gain"] > 0,
                "spatial_ci_lower_positive": audit["bootstrap"]["ci95"][0] > 0,
                "all_integrity": all(integrity.values())}
        submission, decision = publish_predictions(export, template, store.test_ids, predictions,
                                                    store.species_ids, all(gate.values()))
        frame.to_csv(export / "assessment_per_survey_v29.csv", index=False)
        report = {"experiment": EXPERIMENT, "status": "complete", "runtime_hours": guard.elapsed_hours(),
                  "official_submission_made": False, "official_public_score": None, "official_private_score": None,
                  "control_scores": {"public": 0.23339, "private": 0.20831}, "private_target": 0.23021,
                  "selected_policy": policy, "policy_trials": trials, "assessment": audit,
                  "submission_gate": {**gate, **decision}, "submission": submission,
                  "integrity": integrity, "pre_assessment_freeze": freeze,
                  "training": {"matched_reference": reference_record, "development": records,
                               "full_data": deployment_records}, "calibrations": calibrations,
                  "self_tests": tests, "runtime_cap_hours": MAX_HOURS,
                  "hardware": {"torch": torch.__version__, "device": str(device),
                               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                  "limitations": ["Full-data refits use all PA rows, including audit rows; the audit evaluates "
                                   "the separately frozen development predictor, not those production weights.",
                                   "Probability calibration transfer to full-data refits is an assumption.",
                                   "The final fresh audit is small and Denmark-heavy; SOTA requires an official private score."]}
        legacy.save_json(export / "v29_report.json", report)
        features["candidate_sentinel"] = "additional 4x64x64 competition imagery, derived NDVI/NDWI"
        features["candidate_band_order"] = "RGB-NIR; TIFF band/color metadata preferred, published RGB-NIR fallback"
        manifest = {"experiment": EXPERIMENT, "source_sha256": V29_SOURCE_HASH,
                    "embedded_legacy_sha256": LEGACY_SOURCE_HASH, "splits": split_manifest,
                    "features": features, "configs": CONFIGS, "policies": POLICIES,
                    "runtime_cap_hours": MAX_HOURS, "outputs": {p.name: legacy.sha256_file(p)
                    for p in sorted(export.iterdir())}, "large_arrays_exported": False}
        legacy.save_json(export / "v29_manifest.json", manifest)
        if len(list(export.iterdir())) != 4:
            raise ValueError("Unexpected export files")
        return {"status": "complete", **decision, "runtime_hours": guard.elapsed_hours(),
                "export_directory": str(export), "assessment_gain": audit["gain"],
                "spatial_ci95": audit["bootstrap"]["ci95"], "policy": policy,
                "output_bytes": sum(p.stat().st_size for p in export.iterdir())}
    except Exception as error:
        failure = {"experiment": EXPERIMENT, "status": "failed", "error": str(error),
                   "traceback": traceback.format_exc(), "runtime_hours": guard.elapsed_hours(),
                   "official_submission_made": False}
        legacy.save_json(export / "failure_report.json", failure)
        raise
    finally:
        del store
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        legacy._clean_directory(temporary, working)
