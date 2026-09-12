#!/usr/bin/env python
"""Competition-only features, fitted on the training partition, stored as memory maps."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from scripts.prepare_official_pa import feature_paths
from scripts.prepare_spatial_multimodal import static_features


FEATURES = ("landsat", "climate", "sentinel", "static", "environment")


def spatial_partitions(rows: pd.DataFrame) -> np.ndarray:
    """Fixed half-degree blocks; Italy/Switzerland are untouched country audits.

    Block separation is not a distance buffer. Adjacent blocks can be close.
    No label, survey ID, or official test statistic determines a partition.
    """
    coordinates = rows[["lat", "lon"]].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(coordinates).all():
        raise ValueError("Finite coordinates required for spatial partitioning")
    blocks = np.floor(coordinates * 2).astype(np.int64)
    hashed = ((blocks[:, 0] * 73856093) ^ (blocks[:, 1] * 19349663) ^ 2026) % 100
    # Train 85%, checkpoint selection 5%, policy calibration 5%, untouched audit 5%.
    splits = np.select([hashed < 85, hashed < 90, hashed < 95], [0, 1, 2], default=3)
    country_audit = rows["country"].fillna("").isin(["Italy", "Switzerland"]).to_numpy()
    splits[country_audit] = 3
    return splits.astype(np.int8)


def canonical_environment_name(path: Path) -> str:
    return re.sub(r"(?i)(pa[-_]?train|pa[-_]?test|train|test)", "SPLIT", path.as_posix())


def discover_environment_pairs(root: Path) -> list[tuple[Path, Path]]:
    directories = [p for p in root.iterdir() if p.is_dir() and "environmentalvalues" in p.name.lower()]
    if not directories:
        raise FileNotFoundError("Required competition EnvironmentalValues directory not found")
    files = sorted(p for directory in directories for p in directory.rglob("*.csv"))
    # Explicitly exclude PO tables: not a source of PA labels or model pretraining here.
    train = [p for p in files if re.search(r"(?i)pa[-_]?train|(?<![a-z])train(?![a-z])", p.relative_to(root).as_posix()) and not re.search(r"(?i)(?:^|[/_\-])p[0o](?:[/_\-])", p.relative_to(root).as_posix())]
    test = [p for p in files if re.search(r"(?i)pa[-_]?test|(?<![a-z])test(?![a-z])", p.relative_to(root).as_posix())]
    by_name = {canonical_environment_name(p.relative_to(root)): p for p in test}
    pairs = []
    for path in train:
        key = canonical_environment_name(path.relative_to(root))
        if key not in by_name:
            raise ValueError(f"No matching PA-test environmental table for {path.relative_to(root)}")
        pairs.append((path, by_name[key]))
    if not pairs:
        raise ValueError(f"No paired PA train/test CSVs found; files: {[str(p.relative_to(root)) for p in files[:40]]}")
    return pairs


def aligned_table(path: Path, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_candidates = [c for c in frame if re.sub(r"[^a-z]", "", str(c).lower()) == "surveyid"]
    if len(id_candidates) != 1:
        raise ValueError(f"Expected one surveyId column in {path.name}; got {list(frame.columns[:8])}")
    frame = frame.rename(columns={id_candidates[0]: "surveyId"})
    frame["surveyId"] = pd.to_numeric(frame["surveyId"], errors="raise").astype("int64")
    if frame["surveyId"].duplicated().any():
        raise ValueError(f"Duplicate survey IDs in {path.name}; refusing a many-to-many join")
    absent = np.setdiff1d(ids, frame["surveyId"].to_numpy())
    if len(absent):
        raise ValueError(f"{len(absent)} requested survey IDs absent from {path.name}")
    frame = frame.set_index("surveyId").loc[ids]
    excluded = {"speciesid", "predictions", "country", "publisher", "year", "month", "day", "lat", "lon"}
    columns = [c for c in frame if str(c).lower() not in excluded and not str(c).lower().startswith("unnamed:")]
    if any("species" in str(c).lower() for c in columns):
        raise ValueError(f"Potential target column in environmental source {path.name}")
    return frame[columns].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)


def normalize_columns(train: np.ndarray, test: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit per-position statistics on training only; missing values map to zero."""
    fit = np.asarray(train[fit_mask], dtype=np.float32)
    fit[~np.isfinite(fit)] = np.nan
    mean = np.nanmean(fit, axis=0)
    std = np.nanstd(fit, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    def transform(values):
        values = np.asarray(values, dtype=np.float32)
        return np.clip(np.nan_to_num((values - mean) / std, nan=0, posinf=0, neginf=0), -12, 12).astype(np.float16)
    return transform(train), transform(test), {"mean": mean.tolist(), "std": std.tolist()}


def read_modalities(task: tuple[Path, str, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import rasterio
    from rasterio.enums import Resampling
    root, source, survey_id, image_size = task
    landsat_path, climate_path, sentinel_path = feature_paths(root, source, survey_id)
    land = torch.load(landsat_path, map_location="cpu", weights_only=True)
    climate = torch.load(climate_path, map_location="cpu", weights_only=True)
    if tuple(land.shape) != (6, 4, 21) or tuple(climate.shape) != (4, 19, 12):
        raise ValueError(f"Unexpected cube dimensions for survey {survey_id}")
    with rasterio.open(sentinel_path) as source_file:
        image = source_file.read(out_shape=(4, image_size, image_size), out_dtype="float32", resampling=Resampling.bilinear)
    # Fixed scaling preserves cross-band and cross-location reflectance contrasts.
    image = np.clip(np.nan_to_num(image, nan=0, posinf=0, neginf=0) / 10000.0, 0, 2)
    return (land.permute(2, 1, 0).reshape(84, 6).numpy().astype(np.float32),
            climate.permute(1, 2, 0).reshape(228, 4).numpy().astype(np.float32), image.astype(np.float16))


def prepare(data_root: Path, output: Path, image_size: int = 32, workers: int = 6, deadline: float = float("inf")) -> dict:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_root / "GLC25_PA_metadata_train.csv")
    rows = raw.drop_duplicates("surveyId").reset_index(drop=True)
    test_rows = pd.read_csv(data_root / "GLC25_PA_metadata_test.csv").drop_duplicates("surveyId").reset_index(drop=True)
    template = pd.read_csv(data_root / "GLC25_SAMPLE_SUBMISSION.csv")
    ids = rows.surveyId.to_numpy(dtype=np.int64)
    test_ids = test_rows.surveyId.to_numpy(dtype=np.int64)
    if len(set(ids) & set(test_ids)) or set(test_ids) != set(template.surveyId):
        raise ValueError("PA train/test overlap or inconsistent test/template IDs")
    partitions = spatial_partitions(rows)
    fit_mask = partitions == 0
    if any(np.count_nonzero(partitions == i) < 100 for i in range(4)):
        raise ValueError("A registered spatial partition has fewer than 100 surveys")
    species = np.sort(raw.speciesId.dropna().unique().astype(np.int64))
    label_matrix = np.lib.format.open_memmap(output / "train_labels.npy", mode="w+", dtype="uint8", shape=(len(ids), len(species)))
    label_matrix[:] = 0
    pairs = raw[["surveyId", "speciesId"]].dropna().drop_duplicates()
    ri = pd.Index(ids).get_indexer(pairs.surveyId)
    ci = pd.Index(species).get_indexer(pairs.speciesId)
    label_matrix[ri, ci] = 1
    label_matrix.flush()
    for name, array in {"train_ids": ids, "test_ids": test_ids, "partitions": partitions, "species_ids": species}.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    normalization = {}
    environment_manifest = []
    environment_train, environment_test = [], []
    for train_path, test_path in discover_environment_pairs(data_root):
        a, b = aligned_table(train_path, ids), aligned_table(test_path, test_ids)
        if set(a.columns) != set(b.columns):
            raise ValueError(f"Environmental schema mismatch: {train_path.name}")
        b = b[a.columns]
        # Drop columns unavailable or constant in training, never inspect test variance.
        keep = a.loc[fit_mask].nunique(dropna=True) > 1
        a, b = a.loc[:, keep], b.loc[:, keep]
        if not len(a.columns):
            continue
        train_values, test_values, stats = normalize_columns(a.to_numpy(np.float32), b.to_numpy(np.float32), fit_mask)
        # Missingness itself is informative and remains explicitly represented.
        missing_columns = a.loc[fit_mask].isna().any(axis=0).to_numpy()
        environment_train.extend([train_values, a.isna().to_numpy()[:, missing_columns].astype(np.float16)])
        environment_test.extend([test_values, b.isna().to_numpy()[:, missing_columns].astype(np.float16)])
        environment_manifest.append({"train_source": str(train_path.relative_to(data_root)), "test_source": str(test_path.relative_to(data_root)), "columns": list(a.columns), "missing_indicator_columns": list(a.columns[missing_columns]), "normalization": stats})
        print(json.dumps({"environment_source": train_path.name, "predictors": len(a.columns), "missing_indicators": int(missing_columns.sum())}), flush=True)
    if not environment_train:
        raise ValueError("No nonconstant environmental predictors available")
    np.save(output / "train_environment.npy", np.concatenate(environment_train, axis=1), allow_pickle=False)
    np.save(output / "test_environment.npy", np.concatenate(environment_test, axis=1), allow_pickle=False)
    del environment_train, environment_test, a, b, raw, label_matrix
    for prefix, frame, survey_ids, source in (("train", rows, ids, "PA-train"), ("test", test_rows, test_ids, "PA-test")):
        arrays = {name: np.lib.format.open_memmap(output / f"{prefix}_{name}.npy", mode="w+", dtype="float16" if name == "sentinel" else "float32", shape=(len(frame), *shape)) for name, shape in {"landsat": (84, 6), "climate": (228, 4), "sentinel": (4, image_size, image_size)}.items()}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Bounded batches avoid executor queuing 100,000 image results in memory.
            for begin in range(0, len(survey_ids), 256):
                if time.monotonic() > deadline:
                    raise TimeoutError("Preparation exhausted the registered pipeline budget")
                tasks = [(data_root, source, int(sid), image_size) for sid in survey_ids[begin:begin + 256]]
                for index, values in enumerate(pool.map(read_modalities, tasks), start=begin):
                    for name, value in zip(("landsat", "climate", "sentinel"), values):
                        arrays[name][index] = value
                if begin % 4096 == 0:
                    print(json.dumps({"preparing": prefix, "surveys_done": begin + len(tasks), "total": len(survey_ids), "elapsed_minutes": (time.monotonic() - started) / 60}), flush=True)
        for array in arrays.values():
            array.flush()
        del arrays
        np.save(output / f"{prefix}_static.npy", static_features(frame), allow_pickle=False)
    for name in ("landsat", "climate", "static"):
        train_values, test_values, stats = normalize_columns(np.load(output / f"train_{name}.npy"), np.load(output / f"test_{name}.npy"), fit_mask)
        np.save(output / f"train_{name}.npy", train_values, allow_pickle=False)
        np.save(output / f"test_{name}.npy", test_values, allow_pickle=False)
        normalization[name] = stats
    countries = rows.country.fillna("unknown").astype(str).to_numpy(dtype=str)
    np.save(output / "train_countries.npy", countries, allow_pickle=False)
    report = {"protocol": "environmental_challenger_v20", "train_samples": len(ids), "test_samples": len(test_ids), "species": len(species), "species_ids": species.tolist(), "partition_counts": {name: int((partitions == i).sum()) for i, name in enumerate(("training", "checkpoint_selection", "policy_calibration", "untouched_audit"))}, "country_audit": ["Italy", "Switzerland"], "partition_ids_sha256": {name: hashlib.sha256(ids[partitions == i].astype("<i8").tobytes()).hexdigest() for i, name in enumerate(("training", "checkpoint_selection", "policy_calibration", "untouched_audit"))}, "environment_sources": environment_manifest, "normalization": normalization, "normalization_fit_partition": "training only", "sentinel_scale": "fixed reflectance/10000, clip [0,2], no per-image band stretching", "external_data_or_weights": False, "test_labels_used": False, "preparation_seconds": time.monotonic() - started}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.data_root, args.output_dir)
