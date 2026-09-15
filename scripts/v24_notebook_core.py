"""Self-contained GeoLifeCLEF v24 Kaggle pipeline.

This source is copied verbatim into the deliverable notebook by
``build_v24_notebook.py``.  The notebook depends only on the official
GeoLifeCLEF 2025 competition input and Kaggle's standard Python image.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import tarfile
import time
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import BallTree
import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT = "v24_multimodal_rare_species_sdm"
V23_COMMIT = "d307326eb55af13d1bc3b593f17997a8246df644"
V23_SUBMISSION_SHA256 = "9da01ce45a3478e8073cd93e22dbf69dde65def0f86b7ef2700e84630f2c30f8"
V23_PUBLIC_SCORE = 0.22052
V23_PRIVATE_SCORE = 0.19730
SOTA_PRIVATE_SCORE = 0.23021
EXPECTED_SPECIES = 5016
EXPECTED_TEST_ROWS = 14784
EARTH_RADIUS_KM = 6371.0088
MAX_TOTAL_HOURS = 11.25
FINAL_RESERVE_SECONDS = 35 * 60
FEATURE_PREP_LIMIT_SECONDS = 2.75 * 3600
SEEDS = {"split": 20260915, "fold_0": 20262401, "fold_1": 20262402, "deployment": 20262403,
         "bootstrap": 20262404, "po": 20262405}
MODALITIES = ("landsat", "bioclim", "sentinel", "environment", "static")
REMOTE_DIMS = {"landsat": 114, "bioclim": 76, "sentinel": 115}
POLICIES = (
    {"id": "control", "alpha_near": 0.0, "alpha_far": 0.0, "rare_weight": 0.0,
     "spatial_weight": 0.0, "cooccurrence_weight": 0.0, "cardinality_weight": 0.0},
    {"id": "mm_small", "alpha_near": 0.10, "alpha_far": 0.18, "rare_weight": 0.0,
     "spatial_weight": 0.0, "cooccurrence_weight": 0.0, "cardinality_weight": 0.25},
    {"id": "mm", "alpha_near": 0.16, "alpha_far": 0.28, "rare_weight": 0.0,
     "spatial_weight": 0.0, "cooccurrence_weight": 0.0, "cardinality_weight": 0.35},
    {"id": "mm_rare", "alpha_near": 0.14, "alpha_far": 0.28, "rare_weight": 0.055,
     "spatial_weight": 0.0, "cooccurrence_weight": 0.0, "cardinality_weight": 0.35},
    {"id": "mm_cooc", "alpha_near": 0.14, "alpha_far": 0.28, "rare_weight": 0.0,
     "spatial_weight": 0.0, "cooccurrence_weight": 0.035, "cardinality_weight": 0.35},
    {"id": "mm_spatial", "alpha_near": 0.12, "alpha_far": 0.24, "rare_weight": 0.0,
     "spatial_weight": 0.045, "cooccurrence_weight": 0.0, "cardinality_weight": 0.35},
    {"id": "balanced", "alpha_near": 0.14, "alpha_far": 0.28, "rare_weight": 0.045,
     "spatial_weight": 0.035, "cooccurrence_weight": 0.025, "cardinality_weight": 0.35},
    {"id": "balanced_conservative", "alpha_near": 0.10, "alpha_far": 0.20,
     "rare_weight": 0.030, "spatial_weight": 0.025, "cooccurrence_weight": 0.020,
     "cardinality_weight": 0.25},
    {"id": "ood_rare", "alpha_near": 0.08, "alpha_far": 0.34, "rare_weight": 0.060,
     "spatial_weight": 0.025, "cooccurrence_weight": 0.020, "cardinality_weight": 0.40},
)


def sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
                    encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def stable_bucket(text: str, modulus: int = 100) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % modulus


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class RuntimeGuard:
    def __init__(self, max_hours: float = MAX_TOTAL_HOURS):
        self.wall_started = time.time()
        self.started = time.monotonic()
        self.deadline = self.started + max_hours * 3600
        self.max_hours = max_hours

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    def elapsed_hours(self) -> float:
        return self.elapsed_seconds() / 3600

    def remaining_seconds(self) -> float:
        return self.deadline - time.monotonic()

    def require(self, reserve_seconds: float, stage: str) -> None:
        if self.remaining_seconds() <= reserve_seconds:
            raise TimeoutError(
                f"Runtime guard stopped at {stage}: {self.remaining_seconds():.0f}s remain, "
                f"but {reserve_seconds:.0f}s are reserved"
            )

    def stamp(self, stage: str, **extra: Any) -> None:
        print(json.dumps({"stage": stage, "elapsed_minutes": self.elapsed_seconds() / 60,
                          "remaining_minutes": self.remaining_seconds() / 60, **extra},
                         default=json_default), flush=True)


def discover_data_root() -> Path:
    roots = [Path("/kaggle/input"), Path("../input"), Path("data/raw")]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        direct = root / "geolifeclef-2025" / "GLC25_PA_metadata_train.csv"
        if direct.is_file():
            matches.append(direct)
        if not matches:
            matches.extend(root.glob("*/GLC25_PA_metadata_train.csv"))
    parents = sorted({path.resolve().parent for path in matches})
    valid = [path for path in parents if (path / "GLC25_PA_metadata_test.csv").is_file()
             and (path / "GLC25_SAMPLE_SUBMISSION.csv").is_file()]
    if len(valid) != 1:
        raise FileNotFoundError(
            "Attach the official geolifeclef-2025 competition data; "
            f"found {len(valid)} complete roots: {valid}"
        )
    return valid[0]


def safe_extract_tar_gz(payload_b64: str, destination: Path) -> None:
    raw = gzip.decompress(base64.b64decode(payload_b64.encode("ascii")))
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError("Unsafe frozen-v23 archive member")
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError("Frozen-v23 payload may contain regular files only")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read embedded member {member.name}")
            (destination / member.name).write_bytes(source.read())


def verify_frozen_v23(payload_b64: str, destination: Path) -> dict[str, Any]:
    safe_extract_tar_gz(payload_b64, destination)
    required = {
        "GLC25_PA_submission_v23.csv", "v23_report.json", "frozen_policies.json",
        "pre_assessment_freeze.json", "split_manifest.json", "assessment_per_survey.csv",
    }
    found = {path.name for path in destination.iterdir() if path.is_file()}
    if not required.issubset(found):
        raise ValueError(f"Embedded v23 evidence is incomplete: {sorted(required - found)}")
    report = json.loads((destination / "v23_report.json").read_text(encoding="utf-8"))
    freeze = json.loads((destination / "pre_assessment_freeze.json").read_text(encoding="utf-8"))
    csv_path = destination / "GLC25_PA_submission_v23.csv"
    checks = {
        "experiment": report.get("experiment") == "v23_diverse_single_head_crossfit",
        "status": report.get("status") == "complete",
        "source_commit": report.get("source_commit") == V23_COMMIT,
        "kernel_version": report.get("kernel_version") == 25,
        "frozen_hash_record": freeze.get("submission_sha256") == V23_SUBMISSION_SHA256,
        "submission_sha256": sha256_file(csv_path) == V23_SUBMISSION_SHA256,
        "assessment_consumed": report.get("assessment", {}).get("now_consumed") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"Frozen v23 verification failed: {checks}")
    return {"checks": checks, "report_sha256": sha256_file(destination / "v23_report.json"),
            "submission_sha256": sha256_file(csv_path), "source_commit": V23_COMMIT,
            "kernel_version": 25, "submission_reference": "56255321",
            "public_score": V23_PUBLIC_SCORE, "private_score": V23_PRIVATE_SCORE,
            "assessment_consumed": True}


def construct_patch_path(root: Path, survey_id: int) -> Path:
    text = str(int(survey_id))
    return root / text[-2:] / text[-4:-2] / f"{text}.tiff"


def feature_paths(data_root: Path, source: str, survey_id: int) -> tuple[Path, Path, Path]:
    if source not in {"PA-train", "PA-test"}:
        raise ValueError(f"Unsupported source {source}")
    token = "train" if source == "PA-train" else "test"
    landsat_stem = "landsat-time-series" if source == "PA-train" else "landsat_time_series"
    landsat = (data_root / "SateliteTimeSeries-Landsat" / "cubes" / source /
               f"GLC25-PA-{token}-{landsat_stem}_{survey_id}_cube.pt")
    bioclim = (data_root / "BioclimTimeSeries" / "cubes" / source /
               f"GLC25-PA-{token}-bioclimatic_monthly_{survey_id}_cube.pt")
    sentinel = construct_patch_path(data_root / "SatelitePatches" / source, survey_id)
    return landsat, bioclim, sentinel


def _channel_summary(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    channels, length = values.shape
    statistics = np.concatenate([
        values.mean(1), values.std(1), values.min(1), values.max(1),
        np.quantile(values, 0.10, axis=1), np.quantile(values, 0.50, axis=1),
        np.quantile(values, 0.90, axis=1),
    ]).astype(np.float32)
    if length % bins:
        positions = np.linspace(0, length, bins + 1, dtype=int)
        pooled = np.stack([values[:, positions[i]:positions[i + 1]].mean(1)
                           for i in range(bins)], axis=1)
    else:
        pooled = values.reshape(channels, bins, length // bins).mean(2)
    return np.concatenate([statistics, pooled.reshape(-1)]).astype(np.float32)


def extract_remote_features(task: tuple[str, str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_root_text, source, survey_id = task
    data_root = Path(data_root_text)
    landsat_path, bioclim_path, sentinel_path = feature_paths(data_root, source, survey_id)
    if not (landsat_path.is_file() and bioclim_path.is_file() and sentinel_path.is_file()):
        missing = [str(path) for path in (landsat_path, bioclim_path, sentinel_path)
                   if not path.is_file()]
        raise FileNotFoundError(f"Missing modality for survey {survey_id}: {missing}")
    landsat = torch.load(landsat_path, map_location="cpu", weights_only=True)
    bioclim = torch.load(bioclim_path, map_location="cpu", weights_only=True)
    if not isinstance(landsat, torch.Tensor) or tuple(landsat.shape) != (6, 4, 21):
        raise ValueError(f"Unexpected Landsat cube for {survey_id}: {getattr(landsat, 'shape', None)}")
    if not isinstance(bioclim, torch.Tensor) or tuple(bioclim.shape) != (4, 19, 12):
        raise ValueError(f"Unexpected bioclim cube for {survey_id}: {getattr(bioclim, 'shape', None)}")
    land_features = _channel_summary(landsat.numpy().reshape(6, -1), 12)
    climate_features = _channel_summary(bioclim.numpy().reshape(4, -1), 12)
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(sentinel_path) as dataset:
        image = dataset.read(out_shape=(4, 16, 16), out_dtype="float32",
                             resampling=Resampling.bilinear)
    image = np.clip(np.nan_to_num(image / 10000.0, nan=0.0, posinf=0.0, neginf=0.0), 0, 2)
    band = _channel_summary(image.reshape(4, -1), 16)
    red, nir = image[2], image[3]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-4)
    ndvi_features = _channel_summary(ndvi.reshape(1, -1), 16)
    sentinel_features = np.concatenate([band, ndvi_features]).astype(np.float32)
    if (len(land_features), len(climate_features), len(sentinel_features)) != (
        REMOTE_DIMS["landsat"], REMOTE_DIMS["bioclim"], REMOTE_DIMS["sentinel"]
    ):
        raise AssertionError("Remote feature dimensions changed")
    return land_features, climate_features, sentinel_features


def static_features(rows: pd.DataFrame) -> np.ndarray:
    def numeric(name: str, default: float) -> np.ndarray:
        source = rows[name] if name in rows else pd.Series(default, index=rows.index)
        return pd.to_numeric(source, errors="coerce").fillna(default).to_numpy(np.float32)

    lat = pd.to_numeric(rows["lat"], errors="raise").to_numpy(np.float32)
    lon = pd.to_numeric(rows["lon"], errors="raise").to_numpy(np.float32)
    year, month, day = numeric("year", 2025), numeric("month", 6), numeric("day", 15)
    uncertainty = np.log1p(np.maximum(numeric("geoUncertaintyInM", 0), 0)).astype(np.float32)
    area = np.log1p(np.maximum(numeric("areaInM2", 0), 0)).astype(np.float32)
    columns: list[np.ndarray] = [lat, lon, year, month, day, uncertainty, area]
    for frequency in (1, 2, 4, 8, 16):
        columns.extend([np.sin(np.deg2rad(lat) * frequency),
                        np.cos(np.deg2rad(lat) * frequency),
                        np.sin(np.deg2rad(lon) * frequency),
                        np.cos(np.deg2rad(lon) * frequency)])
    phase = 2 * np.pi * (month - 1 + (day - 1) / 31.0) / 12.0
    columns.extend([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    country = rows.get("country", pd.Series(["unknown"] * len(rows))).fillna("unknown").astype(str)
    buckets = np.asarray([stable_bucket(f"country:{value}", 24) for value in country], dtype=int)
    one_hot = np.zeros((len(rows), 24), dtype=np.float32)
    one_hot[np.arange(len(rows)), buckets] = 1
    return np.concatenate([np.stack(columns, axis=1), one_hot], axis=1).astype(np.float32)


def canonical_environment_name(path: Path) -> str:
    import re
    return re.sub(r"(?i)(pa[-_]?train|pa[-_]?test|train|test)", "SPLIT", path.as_posix())


def discover_environment_pairs(root: Path) -> list[tuple[Path, Path]]:
    import re
    directories = [path for path in root.iterdir()
                   if path.is_dir() and "environmentalvalues" in path.name.lower()]
    files = sorted(path for directory in directories for path in directory.rglob("*.csv"))
    train = [path for path in files
             if re.search(r"(?i)pa[-_]?train|(?<![a-z])train(?![a-z])",
                          path.relative_to(root).as_posix())
             and not re.search(r"(?i)(?:^|[/_\-])p[0o](?:[/_\-])",
                               path.relative_to(root).as_posix())]
    test = [path for path in files
            if re.search(r"(?i)pa[-_]?test|(?<![a-z])test(?![a-z])",
                         path.relative_to(root).as_posix())]
    by_name = {canonical_environment_name(path.relative_to(root)): path for path in test}
    pairs = [(path, by_name[canonical_environment_name(path.relative_to(root))])
             for path in train if canonical_environment_name(path.relative_to(root)) in by_name]
    if not pairs:
        raise FileNotFoundError("Official EnvironmentalValues PA train/test tables were not found")
    return pairs


def aligned_environment(path: Path, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_columns = [column for column in frame
                  if "".join(character for character in str(column).lower()
                             if character.isalpha()) == "surveyid"]
    if len(id_columns) != 1:
        raise ValueError(f"Expected one surveyId column in {path}")
    frame = frame.rename(columns={id_columns[0]: "surveyId"})
    frame["surveyId"] = pd.to_numeric(frame["surveyId"], errors="raise").astype("int64")
    if frame.surveyId.duplicated().any():
        raise ValueError(f"Duplicate environmental surveyId values in {path}")
    frame = frame.set_index("surveyId").loc[ids]
    excluded = {"speciesid", "predictions", "country", "publisher", "year", "month",
                "day", "lat", "lon"}
    columns = [column for column in frame
               if str(column).lower() not in excluded
               and not str(column).lower().startswith("unnamed:")
               and "species" not in str(column).lower()]
    return frame[columns].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)


def _write_remote_arrays(data_root: Path, rows: pd.DataFrame, source: str, prefix: str,
                         cache: Path, guard: RuntimeGuard, workers: int) -> dict[str, int]:
    from concurrent.futures import ThreadPoolExecutor
    ids = rows.surveyId.to_numpy(np.int64)
    arrays = {
        name: np.lib.format.open_memmap(cache / f"{prefix}_{name}.npy", mode="w+",
                                       dtype=np.float32, shape=(len(ids), dimension))
        for name, dimension in REMOTE_DIMS.items()
    }
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for begin in range(0, len(ids), 192):
            if guard.elapsed_seconds() > FEATURE_PREP_LIMIT_SECONDS:
                raise TimeoutError("Multimodal preparation exceeded its preregistered 2.75h budget")
            guard.require(FINAL_RESERVE_SECONDS + 6 * 3600, f"feature preparation {prefix}")
            batch_ids = ids[begin:begin + 192]
            tasks = [(str(data_root), source, int(survey_id)) for survey_id in batch_ids]
            for row_index, features in enumerate(executor.map(extract_remote_features, tasks),
                                                  start=begin):
                for name, values in zip(("landsat", "bioclim", "sentinel"), features):
                    arrays[name][row_index] = values
            if begin % 3072 == 0:
                guard.stamp("prepare_modalities", split=prefix,
                            completed=min(begin + len(batch_ids), len(ids)), total=len(ids))
    for values in arrays.values():
        values.flush()
    return {"rows": len(ids), "seconds": int(time.monotonic() - started)}


def prepare_feature_store(data_root: Path, cache: Path, guard: RuntimeGuard,
                          *, workers: int = 6) -> dict[str, Any]:
    cache.mkdir(parents=True, exist_ok=True)
    complete = cache / "feature_manifest.json"
    if complete.is_file():
        manifest = json.loads(complete.read_text(encoding="utf-8"))
        expected = [cache / f"{prefix}_{name}.npy" for prefix in ("train", "test")
                    for name in MODALITIES] + [cache / "labels.npy", cache / "train_ids.npy",
                                               cache / "test_ids.npy", cache / "species_ids.npy"]
        if all(path.is_file() for path in expected):
            guard.stamp("reuse_feature_cache")
            return manifest
    raw = pd.read_csv(data_root / "GLC25_PA_metadata_train.csv")
    train_rows = raw.dropna(subset=["surveyId"]).drop_duplicates("surveyId").reset_index(drop=True)
    test_rows = (pd.read_csv(data_root / "GLC25_PA_metadata_test.csv")
                 .dropna(subset=["surveyId"]).drop_duplicates("surveyId").reset_index(drop=True))
    template = pd.read_csv(data_root / "GLC25_SAMPLE_SUBMISSION.csv")
    train_rows["surveyId"] = train_rows.surveyId.astype("int64")
    test_rows["surveyId"] = test_rows.surveyId.astype("int64")
    train_ids = train_rows.surveyId.to_numpy(np.int64)
    test_ids = test_rows.surveyId.to_numpy(np.int64)
    if len(test_ids) != EXPECTED_TEST_ROWS or set(test_ids) != set(template.surveyId.astype(int)):
        raise ValueError("Official test metadata/sample submission contract changed")
    species = np.sort(raw.speciesId.dropna().unique().astype(np.int64))
    if len(species) != EXPECTED_SPECIES:
        raise ValueError(f"Expected {EXPECTED_SPECIES} PA species, found {len(species)}")
    labels = np.lib.format.open_memmap(cache / "labels.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(train_ids), len(species)))
    labels[:] = 0
    pairs = raw[["surveyId", "speciesId"]].dropna().drop_duplicates().astype("int64")
    row_index = pd.Index(train_ids).get_indexer(pairs.surveyId)
    species_index = pd.Index(species).get_indexer(pairs.speciesId)
    if (row_index < 0).any() or (species_index < 0).any():
        raise ValueError("PA labels failed alignment")
    labels[row_index, species_index] = 1
    labels.flush()
    np.save(cache / "train_ids.npy", train_ids, allow_pickle=False)
    np.save(cache / "test_ids.npy", test_ids, allow_pickle=False)
    np.save(cache / "species_ids.npy", species, allow_pickle=False)
    np.save(cache / "train_static.npy", static_features(train_rows), allow_pickle=False)
    np.save(cache / "test_static.npy", static_features(test_rows), allow_pickle=False)
    environment_train: list[np.ndarray] = []
    environment_test: list[np.ndarray] = []
    environment_sources: list[dict[str, Any]] = []
    for train_path, test_path in discover_environment_pairs(data_root):
        train_frame = aligned_environment(train_path, train_ids)
        test_frame = aligned_environment(test_path, test_ids)
        common = [column for column in train_frame.columns if column in test_frame.columns]
        train_frame, test_frame = train_frame[common], test_frame[common]
        keep = train_frame.nunique(dropna=True) > 1
        train_frame, test_frame = train_frame.loc[:, keep], test_frame.loc[:, keep]
        if train_frame.shape[1]:
            train_values, test_values = train_frame.to_numpy(np.float32), test_frame.to_numpy(np.float32)
            missing_columns = train_frame.isna().any(axis=0).to_numpy()
            environment_train.extend([train_values,
                                      train_frame.isna().to_numpy(np.float32)[:, missing_columns]])
            environment_test.extend([test_values,
                                     test_frame.isna().to_numpy(np.float32)[:, missing_columns]])
            environment_sources.append({
                "train": str(train_path.relative_to(data_root)),
                "test": str(test_path.relative_to(data_root)),
                "predictors": int(train_values.shape[1]),
                "missing_indicators": int(missing_columns.sum()),
            })
    if not environment_train:
        raise ValueError("No official soil/environmental descriptors were loaded")
    np.save(cache / "train_environment.npy", np.concatenate(environment_train, axis=1),
            allow_pickle=False)
    np.save(cache / "test_environment.npy", np.concatenate(environment_test, axis=1),
            allow_pickle=False)
    remote_reports = {
        "train": _write_remote_arrays(data_root, train_rows, "PA-train", "train", cache,
                                      guard, workers),
        "test": _write_remote_arrays(data_root, test_rows, "PA-test", "test", cache,
                                     guard, workers),
    }
    shapes = {name: list(np.load(cache / f"train_{name}.npy", mmap_mode="r").shape[1:])
              for name in MODALITIES}
    manifest = {
        "official_competition": "geolifeclef-2025", "external_data_or_weights": False,
        "train_rows": len(train_ids), "test_rows": len(test_ids), "species": len(species),
        "train_ids_sha256": sha256_bytes(train_ids.astype("<i8").tobytes()),
        "test_ids_sha256": sha256_bytes(test_ids.astype("<i8").tobytes()),
        "species_ids_sha256": sha256_bytes(species.astype("<i8").tobytes()),
        "modalities": shapes, "environment_sources": environment_sources,
        "remote_preparation": remote_reports, "summary_encoder": {
            "landsat": "per-band distribution plus 12 temporal bins",
            "bioclim": "per-channel distribution plus 12 temporal bins",
            "sentinel": "fixed-reflectance band and NDVI statistics plus 4x4 spatial pooling",
        }, "preparation_seconds": guard.elapsed_seconds(), "test_labels_used": False,
    }
    save_json(complete, manifest)
    del raw, labels, pairs, environment_train, environment_test
    gc.collect()
    return manifest


def load_rows_and_pairs(data_root: Path, train_ids: np.ndarray, test_ids: np.ndarray
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(data_root / "GLC25_PA_metadata_train.csv")
    rows = raw.drop_duplicates("surveyId").set_index("surveyId").loc[train_ids].reset_index()
    test_rows = (pd.read_csv(data_root / "GLC25_PA_metadata_test.csv")
                 .drop_duplicates("surveyId").set_index("surveyId").loc[test_ids].reset_index())
    pairs = raw[["surveyId", "speciesId"]].dropna().drop_duplicates().astype("int64")
    return rows, test_rows, pairs


class FeatureStore:
    def __init__(self, cache: Path):
        self.cache = cache
        self.train = {name: np.load(cache / f"train_{name}.npy", mmap_mode="r")
                      for name in MODALITIES}
        self.test = {name: np.load(cache / f"test_{name}.npy", mmap_mode="r")
                     for name in MODALITIES}
        self.labels = np.load(cache / "labels.npy", mmap_mode="r")
        self.train_ids = np.load(cache / "train_ids.npy", allow_pickle=False)
        self.test_ids = np.load(cache / "test_ids.npy", allow_pickle=False)
        self.species_ids = np.load(cache / "species_ids.npy", allow_pickle=False)
        self.dims = {name: int(values.shape[1]) for name, values in self.train.items()}


def spatial_blocks(rows: pd.DataFrame) -> np.ndarray:
    lat = pd.to_numeric(rows.lat, errors="raise").to_numpy(np.float64)
    lon = pd.to_numeric(rows.lon, errors="raise").to_numpy(np.float64)
    return np.asarray([f"{math.floor(a):+04d}:{math.floor(o):+04d}" for a, o in zip(lat, lon)])


def nearest_distance_km(reference_coordinates: np.ndarray,
                        query_coordinates: np.ndarray) -> np.ndarray:
    tree = BallTree(np.deg2rad(np.asarray(reference_coordinates, dtype=np.float64)),
                    metric="haversine")
    distance, _ = tree.query(np.deg2rad(np.asarray(query_coordinates, dtype=np.float64)), k=1)
    return distance[:, 0] * EARTH_RADIUS_KM


def make_outer_split(rows: pd.DataFrame, fold: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if fold not in (0, 1):
        raise ValueError("v24 has exactly two preregistered outer folds")
    blocks = spatial_blocks(rows)
    bucket = np.asarray([stable_bucket(f"v24-outer:{SEEDS['split']}:{block}") for block in blocks])
    assessment = (bucket >= fold * 12) & (bucket < (fold + 1) * 12)
    selection = (bucket >= 24) & (bucket < 30)
    calibration = (bucket >= 30) & (bucket < 38)
    evaluation = assessment | selection | calibration
    candidate_train = ~evaluation
    coordinates = rows[["lat", "lon"]].to_numpy(np.float64)
    distance = nearest_distance_km(coordinates[evaluation], coordinates[candidate_train])
    train_candidates = np.flatnonzero(candidate_train)
    training = train_candidates[distance >= 20.0]
    result = {"training": training, "selection": np.flatnonzero(selection),
              "calibration": np.flatnonzero(calibration),
              "assessment": np.flatnonzero(assessment)}
    if min(map(len, result.values())) < 500:
        raise ValueError(f"Preregistered fold {fold} produced a small partition: "
                         f"{{name: len(v) for name, v in result.items()}}")
    support = nearest_distance_km(coordinates[training], coordinates[result["assessment"]])
    manifest = {
        "fold": fold, "seed": SEEDS["split"], "block_size_degrees": 1.0,
        "assessment_bucket_range": [fold * 12, (fold + 1) * 12 - 1],
        "selection_bucket_range": [24, 29], "calibration_bucket_range": [30, 37],
        "buffer_km": 20.0, "adaptive_retries": 0,
        "partition_counts": {name: len(values) for name, values in result.items()},
        "partition_blocks": {name: int(np.unique(blocks[values]).size)
                             for name, values in result.items()},
        "assessment_ids_sha256": sha256_bytes(
            rows.surveyId.to_numpy(np.int64)[result["assessment"]].astype("<i8").tobytes()),
        "minimum_assessment_training_distance_km": float(support.min()),
        "v22_v23_assessments_consumed_and_not_reused": True,
        "assessment_used_for_selection": False,
    }
    return result, manifest


def make_deployment_split(rows: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    blocks = spatial_blocks(rows)
    bucket = np.asarray([stable_bucket(f"v24-deploy:{SEEDS['split']}:{block}") for block in blocks])
    selection = bucket < 7
    calibration = (bucket >= 7) & (bucket < 16)
    evaluation = selection | calibration
    candidate_train = ~evaluation
    coordinates = rows[["lat", "lon"]].to_numpy(np.float64)
    distance = nearest_distance_km(coordinates[evaluation], coordinates[candidate_train])
    candidates = np.flatnonzero(candidate_train)
    training = candidates[distance >= 10.0]
    result = {"training": training, "selection": np.flatnonzero(selection),
              "calibration": np.flatnonzero(calibration)}
    if min(map(len, result.values())) < 500:
        raise ValueError("Deployment partitions are unexpectedly small")
    return result, {"seed": SEEDS["split"], "block_size_degrees": 1.0,
                    "selection_bucket_range": [0, 6], "calibration_bucket_range": [7, 15],
                    "training_bucket_range": [16, 99], "training_buffer_km": 10.0,
                    "adaptive_retries": 0,
                    "partition_counts": {name: len(value) for name, value in result.items()}}


def normalization_stats(arrays: dict[str, np.ndarray], indices: np.ndarray
                        ) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, values in arrays.items():
        fit = np.asarray(values[indices], dtype=np.float32)
        fit[~np.isfinite(fit)] = np.nan
        mean = np.nanmean(fit, axis=0).astype(np.float32)
        std = np.nanstd(fit, axis=0).astype(np.float32)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std[std < 1e-5] = 1.0
        result[name] = {"mean": mean, "std": std}
    return result


def normalized_batch(arrays: dict[str, np.ndarray], indices: np.ndarray,
                     stats: dict[str, dict[str, np.ndarray]], device: torch.device
                     ) -> dict[str, torch.Tensor]:
    result = {}
    for name in MODALITIES:
        values = np.asarray(arrays[name][indices], dtype=np.float32)
        values = np.clip(np.nan_to_num((values - stats[name]["mean"]) / stats[name]["std"],
                                       nan=0.0, posinf=0.0, neginf=0.0), -10, 10)
        result[name] = torch.from_numpy(values).to(device, non_blocking=True)
    return result


class ResidualVectorBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.10):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(width * 2, width),
                                     nn.Dropout(dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class MatchedV23Control(nn.Module):
    """Early-fusion refit of the frozen v23 family for new-fold recipe transfer.

    The exact deployed v23 CSV remains the official-test control.  This model is
    deliberately named *matched* rather than *exact*: old v23 assessment folds
    are consumed and exact fold checkpoints were not exported.
    """
    def __init__(self, dims: dict[str, int], labels: int, width: int = 384):
        super().__init__()
        total = sum(dims.values())
        self.network = nn.Sequential(nn.Linear(total, width), nn.GELU(),
                                     ResidualVectorBlock(width), ResidualVectorBlock(width),
                                     nn.LayerNorm(width), nn.Linear(width, labels))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(torch.cat([batch[name] for name in MODALITIES], dim=1))


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(),
                                     ResidualVectorBlock(width), nn.LayerNorm(width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class V24MultimodalRareJSDM(nn.Module):
    def __init__(self, dims: dict[str, int], labels: int, rare_indices: np.ndarray,
                 *, width: int = 160, rank: int = 80):
        super().__init__()
        self.encoders = nn.ModuleDict({name: ModalityEncoder(dims[name], width)
                                       for name in MODALITIES})
        self.modality_embeddings = nn.Parameter(torch.randn(len(MODALITIES), width) * 0.02)
        self.gate = nn.Sequential(nn.Linear(len(MODALITIES) * width, width), nn.GELU(),
                                  nn.Linear(width, len(MODALITIES)))
        self.fusion = nn.Sequential(nn.Linear(len(MODALITIES) * width + width, width * 2),
                                    nn.GELU(), ResidualVectorBlock(width * 2),
                                    nn.Linear(width * 2, width), nn.LayerNorm(width))
        self.independent_head = nn.Linear(width, labels)
        self.joint_projection = nn.Linear(width, rank, bias=False)
        self.species_embedding = nn.Parameter(torch.randn(labels, rank) * 0.02)
        self.joint_scale = nn.Parameter(torch.tensor(-1.5))
        rare = torch.as_tensor(np.asarray(rare_indices, dtype=np.int64))
        self.register_buffer("rare_indices", rare)
        self.rare_head = nn.Linear(width, len(rare)) if len(rare) else None
        self.rare_scale = nn.Parameter(torch.tensor(-1.5))
        self.richness_head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(),
                                           nn.Linear(width // 2, 1))

    def forward_with_aux(self, batch: dict[str, torch.Tensor]
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.stack([self.encoders[name](batch[name]) for name in MODALITIES], dim=1)
        tokens = tokens + self.modality_embeddings.unsqueeze(0)
        flat = tokens.flatten(1)
        weights = torch.softmax(self.gate(flat), dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(1)
        fused = self.fusion(torch.cat([flat, pooled], dim=1))
        logits = self.independent_head(fused)
        joint = self.joint_projection(fused) @ self.species_embedding.T
        logits = logits + torch.sigmoid(self.joint_scale) * joint
        if self.rare_head is not None:
            rare_logits = self.rare_head(fused)
            rare_delta = torch.zeros_like(logits).index_copy(1, self.rare_indices, rare_logits)
            logits = logits + torch.sigmoid(self.rare_scale) * rare_delta
        else:
            rare_logits = logits[:, :0]
        richness = self.richness_head(fused).squeeze(1)
        return logits, richness, weights

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_with_aux(batch)[0]


def frequency_aware_asymmetric_loss(logits: torch.Tensor, targets: torch.Tensor,
                                    positive_weights: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    targets = targets.float()
    probabilities = torch.sigmoid(values)
    positive = -F.logsigmoid(values) * targets * positive_weights.unsqueeze(0)
    clipped = (probabilities - 0.05).clamp_min(0.0)
    negative = -torch.log1p(-clipped.clamp_max(1 - 1e-6)) * (1 - targets) * clipped.pow(4)
    positive_loss = positive.sum() / (targets * positive_weights.unsqueeze(0)).sum().clamp_min(1)
    negative_loss = negative.sum() / (1 - targets).sum().clamp_min(1)
    return positive_loss + negative_loss


def training_sampling_weights(labels: np.ndarray, indices: np.ndarray,
                              frequencies: np.ndarray) -> np.ndarray:
    weights = np.ones(len(indices), dtype=np.float64)
    inverse = np.where(frequencies > 0, 1.0 / np.sqrt(np.maximum(frequencies, 1)), 0.0)
    scale = np.percentile(inverse[inverse > 0], 75) if np.any(inverse > 0) else 1.0
    for begin in range(0, len(indices), 2048):
        batch = np.asarray(labels[indices[begin:begin + 2048]], dtype=np.uint8)
        rarity = (batch * inverse).sum(1) / np.maximum(batch.sum(1), 1)
        weights[begin:begin + len(batch)] += np.clip(rarity / max(scale, 1e-8), 0, 4)
    weights /= weights.sum()
    return weights


def top_rank(probabilities: np.ndarray, maximum: int = 64) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities)
    maximum = min(maximum, probabilities.shape[1])
    indices = np.argpartition(probabilities, -maximum, axis=1)[:, -maximum:]
    values = np.take_along_axis(probabilities, indices, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1), np.take_along_axis(values, order, axis=1)


def f1_from_ranked(targets: np.ndarray, ranked_indices: np.ndarray,
                   counts: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets)
    counts = np.asarray(counts, dtype=np.int64)
    hits = np.take_along_axis(targets, ranked_indices, axis=1).cumsum(1)
    return 2 * hits[np.arange(len(targets)), counts - 1] / np.maximum(
        targets.sum(1) + counts, 1)


def v23_cardinality(distance_km: np.ndarray) -> np.ndarray:
    risk = np.clip(np.log1p(np.asarray(distance_km, dtype=np.float64)) / np.log(201.0), 0, 1)
    return np.where(risk < 0.5, 20, 28).astype(np.int64)


def _checkpoint_score(model: nn.Module, arrays: dict[str, np.ndarray], labels: np.ndarray,
                      indices: np.ndarray, stats: dict[str, dict[str, np.ndarray]],
                      device: torch.device, *, v24: bool, batch_size: int = 512) -> float:
    probabilities, richness, _ = predict_model(model, arrays, indices, stats, device,
                                                v24=v24, batch_size=batch_size)
    ranked, _ = top_rank(probabilities, 32)
    if v24:
        counts = np.clip(np.rint(np.expm1(richness)), 16, 28).astype(np.int64)
    else:
        counts = np.full(len(indices), 20, dtype=np.int64)
    return float(f1_from_ranked(np.asarray(labels[indices]), ranked, counts).mean())


@torch.no_grad()
def predict_model(model: nn.Module, arrays: dict[str, np.ndarray], indices: np.ndarray,
                  stats: dict[str, dict[str, np.ndarray]], device: torch.device, *, v24: bool,
                  batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    label_count = (model.independent_head.out_features if isinstance(model, V24MultimodalRareJSDM)
                   else model.network[-1].out_features)
    probabilities = np.empty((len(indices), label_count), dtype=np.float16)
    richness = np.full(len(indices), np.log1p(20.0), dtype=np.float32)
    modality_weights = np.full((len(indices), len(MODALITIES)), 1 / len(MODALITIES),
                               dtype=np.float32)
    for begin in range(0, len(indices), batch_size):
        take = indices[begin:begin + batch_size]
        batch = normalized_batch(arrays, take, stats, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if v24:
                logits, predicted_richness, weights = model.forward_with_aux(batch)
            else:
                logits, predicted_richness, weights = model(batch), None, None
        size = len(take)
        probabilities[begin:begin + size] = torch.sigmoid(logits).float().cpu().numpy().astype(np.float16)
        if predicted_richness is not None:
            richness[begin:begin + size] = predicted_richness.float().cpu().numpy()
            modality_weights[begin:begin + size] = weights.float().cpu().numpy()
    if not np.isfinite(probabilities).all() or not np.isfinite(richness).all():
        raise FloatingPointError("Non-finite model predictions")
    return probabilities, richness, modality_weights


def train_model(model: nn.Module, arrays: dict[str, np.ndarray], labels: np.ndarray,
                training_indices: np.ndarray, selection_indices: np.ndarray,
                stats: dict[str, dict[str, np.ndarray]], device: torch.device,
                checkpoint: Path, guard: RuntimeGuard, *, seed: int, v24: bool,
                epochs: int, minimum_epochs: int, batch_size: int = 256) -> dict[str, Any]:
    set_seed(seed)
    model.to(device)
    frequencies = _frequency(labels, training_indices).astype(np.float32)
    positive_weights_np = np.where(
        frequencies > 0,
        np.clip(np.sqrt(np.maximum(np.median(frequencies[frequencies > 0]), 1) /
                        np.maximum(frequencies, 1)), 1, 6),
        1,
    ).astype(np.float32)
    positive_weights = torch.from_numpy(positive_weights_np).to(device)
    sampling = training_sampling_weights(labels, training_indices, frequencies) if v24 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4 if v24 else 6e-4,
                                  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        if v24:
            order = rng.choice(training_indices, size=len(training_indices), replace=True, p=sampling)
        else:
            order = rng.permutation(training_indices)
        model.train()
        total, seen = 0.0, 0
        for begin in range(0, len(order), batch_size):
            guard.require(FINAL_RESERVE_SECONDS + 75 * 60, f"training epoch {epoch}")
            take = order[begin:begin + batch_size]
            batch = normalized_batch(arrays, take, stats, device)
            targets = torch.from_numpy(np.asarray(labels[take], dtype=np.float32)).to(
                device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                if v24:
                    logits, richness, _ = model.forward_with_aux(batch)
                    classification = frequency_aware_asymmetric_loss(logits, targets,
                                                                     positive_weights)
                    richness_loss = F.smooth_l1_loss(richness.float(),
                                                      torch.log1p(targets.sum(1)).float())
                    rare_mask = model.rare_indices
                    rare_loss = (frequency_aware_asymmetric_loss(
                        logits[:, rare_mask], targets[:, rare_mask], positive_weights[rare_mask])
                                 if len(rare_mask) else classification.new_zeros(()))
                    loss = classification + 0.20 * rare_loss + 0.08 * richness_loss
                else:
                    logits = model(batch)
                    loss = frequency_aware_asymmetric_loss(logits, targets, positive_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(take)
            seen += len(take)
        scheduler.step()
        score = None
        if epoch >= minimum_epochs and (epoch == minimum_epochs or epoch % 2 == 0 or epoch == epochs):
            score = _checkpoint_score(model, arrays, labels, selection_indices, stats, device,
                                      v24=v24)
            if score > best_score:
                best_score, best_epoch = score, epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "selection_f1": score, "v24": v24}, checkpoint)
        seconds = time.monotonic() - epoch_started
        record = {"epoch": epoch, "loss": total / max(seen, 1), "selection_f1": score,
                  "seconds": seconds, "examples_per_second": seen / max(seconds, 1e-6)}
        history.append(record)
        guard.stamp("train_epoch", model="v24" if v24 else "matched_v23", **record)
        if epoch >= minimum_epochs and guard.remaining_seconds() < FINAL_RESERVE_SECONDS + 75 * 60 + seconds * 1.3:
            break
    if best_epoch == 0 or len(history) < minimum_epochs:
        raise TimeoutError("A required model did not complete its minimum registered epochs")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    return {"best_epoch": best_epoch, "selection_f1": best_score, "history": history,
            "checkpoint_sha256": sha256_file(checkpoint),
            "parameters": sum(parameter.numel() for parameter in model.parameters()
                              if parameter.requires_grad),
            "training_frequency": frequencies.tolist()}


class PASpatialIndex:
    def __init__(self, rows: pd.DataFrame, labels: np.ndarray, reference_indices: np.ndarray):
        self.reference_indices = np.asarray(reference_indices, dtype=np.int64)
        coordinates = rows.iloc[self.reference_indices][["lat", "lon"]].to_numpy(np.float64)
        self.tree = BallTree(np.deg2rad(coordinates), metric="haversine")
        self.labels = labels

    def query(self, coordinates: np.ndarray, *, neighbors: int = 8, radius_km: float = 30.0,
              maximum_candidates: int = 28) -> tuple[list[dict[int, float]], np.ndarray]:
        distances, positions = self.tree.query(np.deg2rad(np.asarray(coordinates, np.float64)),
                                               k=min(neighbors, len(self.reference_indices)))
        distances *= EARTH_RADIUS_KM
        candidates: list[dict[int, float]] = []
        for row_distances, row_positions in zip(distances, positions):
            valid = row_distances <= radius_km
            scores: dict[int, float] = defaultdict(float)
            support: Counter[int] = Counter()
            for distance, position in zip(row_distances[valid], row_positions[valid]):
                columns = np.flatnonzero(self.labels[self.reference_indices[position]])
                weight = math.exp(-float(distance) / 12.0)
                for column in columns:
                    scores[int(column)] += weight
                    support[int(column)] += 1
            eligible = [(column, value) for column, value in scores.items()
                        if support[column] >= 2 or (row_distances[0] <= 2.0 and support[column] >= 1)]
            eligible.sort(key=lambda item: (-item[1], item[0]))
            eligible = eligible[:maximum_candidates]
            scale = max((value for _, value in eligible), default=1.0)
            candidates.append({column: float(value / scale) for column, value in eligible})
        return candidates, distances[:, 0]


class POGridIndex:
    def __init__(self, cells: dict[tuple[int, int], list[tuple[int, float]]],
                 species_ids: np.ndarray, global_counts: np.ndarray, rows_seen: int,
                 rows_retained: int):
        self.cells = cells
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.global_counts = np.asarray(global_counts, dtype=np.int64)
        self.rows_seen = int(rows_seen)
        self.rows_retained = int(rows_retained)

    @classmethod
    def build(cls, metadata_path: Path, species_ids: np.ndarray, pa_coordinates: np.ndarray,
              guard: RuntimeGuard, *, cell_degrees: float = 0.10,
              chunksize: int = 350_000) -> "POGridIndex":
        species_ids = np.asarray(species_ids, dtype=np.int64)
        species_lookup = pd.Index(species_ids)
        pa_tree = BallTree(np.deg2rad(np.asarray(pa_coordinates, np.float64)), metric="haversine")
        accumulated: dict[tuple[int, int, int], int] = defaultdict(int)
        global_counts = np.zeros(len(species_ids), dtype=np.int64)
        seen, retained = 0, 0
        for chunk in pd.read_csv(metadata_path, usecols=["lat", "lon", "speciesId"],
                                 chunksize=chunksize):
            guard.require(FINAL_RESERVE_SECONDS + 5 * 3600, "presence-only aggregation")
            seen += len(chunk)
            chunk = chunk.dropna(subset=["lat", "lon", "speciesId"])
            columns = species_lookup.get_indexer(chunk.speciesId.astype(np.int64))
            valid = columns >= 0
            chunk, columns = chunk.loc[valid].copy(), columns[valid]
            if len(chunk):
                distance, _ = pa_tree.query(np.deg2rad(chunk[["lat", "lon"]].to_numpy(np.float64)),
                                            k=1)
                keep = distance[:, 0] * EARTH_RADIUS_KM > 0.10
                chunk, columns = chunk.loc[keep], columns[keep]
            if len(chunk):
                cell_x = np.floor((chunk.lon.to_numpy(np.float64) + 180) / cell_degrees).astype(int)
                cell_y = np.floor((chunk.lat.to_numpy(np.float64) + 90) / cell_degrees).astype(int)
                local = pd.DataFrame({"x": cell_x, "y": cell_y, "column": columns})
                grouped = local.groupby(["x", "y", "column"], sort=False).size()
                for (x, y, column), count in grouped.items():
                    accumulated[(int(x), int(y), int(column))] += int(count)
                    global_counts[int(column)] += int(count)
                retained += len(chunk)
            if seen % (chunksize * 3) < chunksize:
                guard.stamp("prepare_po_grid", rows_seen=seen, retained=retained,
                            aggregated_entries=len(accumulated))
        raw_cells: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for (x, y, column), count in accumulated.items():
            raw_cells[(x, y)].append((column, count))
        cells: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for cell, values in raw_cells.items():
            scored = [(column, count / max(global_counts[column], 1) ** 0.35)
                      for column, count in values]
            scored.sort(key=lambda item: (-item[1], item[0]))
            selected = scored[:48]
            scale = max((value for _, value in selected), default=1.0)
            cells[cell] = [(column, float(value / scale)) for column, value in selected]
        accumulated.clear()
        raw_cells.clear()
        gc.collect()
        return cls(cells, species_ids, global_counts, seen, retained)

    def query(self, coordinates: np.ndarray, *, cell_degrees: float = 0.10,
              maximum_candidates: int = 28) -> tuple[list[dict[int, float]], np.ndarray]:
        result: list[dict[int, float]] = []
        coverage = np.zeros(len(coordinates), dtype=np.float32)
        for row, (lat, lon) in enumerate(np.asarray(coordinates, np.float64)):
            x = int(math.floor((lon + 180) / cell_degrees))
            y = int(math.floor((lat + 90) / cell_degrees))
            scores: dict[int, float] = defaultdict(float)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell_weight = math.exp(-0.8 * math.hypot(dx, dy))
                    for column, score in self.cells.get((x + dx, y + dy), ()): 
                        scores[column] += cell_weight * score
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:maximum_candidates]
            scale = max((value for _, value in ordered), default=1.0)
            result.append({column: float(value / scale) for column, value in ordered})
            coverage[row] = float(sum(value for _, value in ordered))
        return result, coverage


class CooccurrenceGraph:
    def __init__(self, neighbors: np.ndarray, weights: np.ndarray):
        self.neighbors = np.asarray(neighbors, dtype=np.int32)
        self.weights = np.asarray(weights, dtype=np.float32)

    @classmethod
    def build(cls, labels: np.ndarray, training_indices: np.ndarray, *, top_n: int = 8
              ) -> "CooccurrenceGraph":
        blocks: list[sparse.csr_matrix] = []
        for begin in range(0, len(training_indices), 2048):
            dense = np.asarray(labels[training_indices[begin:begin + 2048]], dtype=np.float32)
            blocks.append(sparse.csr_matrix(dense))
        matrix = sparse.vstack(blocks, format="csr")
        frequencies = np.asarray(matrix.sum(0)).ravel()
        cooccurrence = (matrix.T @ matrix).tocsr()
        neighbors = np.full((matrix.shape[1], top_n), -1, dtype=np.int32)
        weights = np.zeros((matrix.shape[1], top_n), dtype=np.float32)
        for species in range(matrix.shape[1]):
            start, end = cooccurrence.indptr[species:species + 2]
            columns = cooccurrence.indices[start:end]
            counts = cooccurrence.data[start:end]
            keep = (columns != species) & (counts >= 3)
            columns, counts = columns[keep], counts[keep]
            if not len(columns):
                continue
            score = counts / np.sqrt(np.maximum(frequencies[species] * frequencies[columns], 1))
            order = np.argsort(-score, kind="stable")[:top_n]
            chosen, chosen_score = columns[order], score[order]
            scale = max(float(chosen_score[0]), 1e-8)
            neighbors[species, :len(chosen)] = chosen
            weights[species, :len(chosen)] = chosen_score / scale
        return cls(neighbors, weights)

    def digest(self) -> str:
        return sha256_bytes(self.neighbors.astype("<i4").tobytes() +
                            self.weights.astype("<f4").tobytes())


def richness_features(probabilities: np.ndarray, raw_log_richness: np.ndarray,
                      rows: pd.DataFrame, pa_distance: np.ndarray, po_coverage: np.ndarray,
                      country_means: dict[str, float], global_mean: float) -> np.ndarray:
    _, top = top_rank(probabilities, 40)
    month = pd.to_numeric(rows.get("month", 6), errors="coerce").fillna(6).to_numpy(np.float32)
    phase = 2 * np.pi * (month - 1) / 12
    country = rows.get("country", pd.Series(["unknown"] * len(rows))).fillna("unknown").astype(str)
    country_richness = np.asarray([country_means.get(value, global_mean) for value in country],
                                  dtype=np.float32)
    features = np.column_stack([
        raw_log_richness, top[:, 0], top[:, :5].mean(1), top[:, :20].mean(1),
        top.mean(1), top.std(1), top[:, 19] - top[:, 39], np.log1p(pa_distance),
        np.log1p(po_coverage), np.sin(phase), np.cos(phase), np.log1p(country_richness),
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_richness_model(probabilities: np.ndarray, raw_log_richness: np.ndarray,
                       rows: pd.DataFrame, pa_distance: np.ndarray, po_coverage: np.ndarray,
                       target_cardinality: np.ndarray, training_rows: pd.DataFrame,
                       training_cardinality: np.ndarray, *, seed: int) -> tuple[Any, dict[str, Any]]:
    training_cardinality = np.asarray(training_cardinality)
    countries = training_rows.get("country", pd.Series(["unknown"] * len(training_rows))).fillna("unknown")
    table = pd.DataFrame({"country": countries.to_numpy(), "richness": training_cardinality})
    country_means = table.groupby("country").richness.mean().to_dict()
    global_mean = float(training_cardinality.mean())
    features = richness_features(probabilities, raw_log_richness, rows, pa_distance, po_coverage,
                                 country_means, global_mean)
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=70, max_leaf_nodes=15,
                                          learning_rate=0.06, l2_regularization=1.0,
                                          random_state=seed).fit(features, target_cardinality)
    prediction = np.clip(model.predict(features), 12, 32)
    return model, {"country_means": country_means, "global_mean": global_mean,
                   "selection_mae": float(np.mean(np.abs(prediction - target_cardinality))),
                   "feature_names": ["neural_log_richness", "top1", "top5_mean", "top20_mean",
                                     "top40_mean", "top40_std", "rank_margin_20_40",
                                     "log_pa_distance", "log_po_coverage", "month_sin",
                                     "month_cos", "country_training_richness"]}


def predict_richness(model: Any, metadata: dict[str, Any], probabilities: np.ndarray,
                     raw_log_richness: np.ndarray, rows: pd.DataFrame, pa_distance: np.ndarray,
                     po_coverage: np.ndarray) -> np.ndarray:
    features = richness_features(probabilities, raw_log_richness, rows, pa_distance, po_coverage,
                                 metadata["country_means"], metadata["global_mean"])
    return np.clip(model.predict(features), 12, 32)


def ood_risk(pa_distance: np.ndarray, po_coverage: np.ndarray,
             base_lists: list[list[int]], v24_ranked: np.ndarray) -> np.ndarray:
    pa = np.clip(np.log1p(pa_distance) / np.log(201.0), 0, 1)
    po = 1 - np.clip(np.log1p(po_coverage) / np.log(25.0), 0, 1)
    disagreement = np.empty(len(base_lists), dtype=np.float32)
    for row, (base, ranked) in enumerate(zip(base_lists, v24_ranked)):
        a, b = set(base[:20]), set(map(int, ranked[:20]))
        disagreement[row] = 1 - len(a & b) / max(len(a | b), 1)
    return np.clip(0.50 * pa + 0.25 * po + 0.25 * disagreement, 0, 1)


def compose_predictions(base_lists: list[list[int]], v24_probabilities: np.ndarray,
                        predicted_richness: np.ndarray, frequencies: np.ndarray,
                        spatial_candidates: list[dict[int, float]],
                        po_candidates: list[dict[int, float]], graph: CooccurrenceGraph,
                        risk: np.ndarray, policy: dict[str, Any]) -> list[list[int]]:
    v24_ranked, v24_values = top_rank(v24_probabilities, 64)
    result: list[list[int]] = []
    for row, base in enumerate(base_lists):
        base = list(map(int, base))
        alpha = policy["alpha_near"] + (policy["alpha_far"] - policy["alpha_near"]) * risk[row]
        scores: dict[int, float] = {}
        base_denominator = max(len(base) - 1, 1)
        for rank, column in enumerate(base):
            scores[column] = max(scores.get(column, 0.0),
                                 (1 - alpha) * (1.0 - 0.70 * rank / base_denominator))
        for rank, column in enumerate(v24_ranked[row]):
            scores[int(column)] = scores.get(int(column), 0.0) + alpha * (1.0 - 0.85 * rank / 63)
        for column, support in po_candidates[row].items():
            if frequencies[column] <= 25 and support >= 0.12:
                relative = float(v24_probabilities[row, column]) / max(float(v24_values[row, 0]), 1e-6)
                scores[column] = scores.get(column, 0.0) + policy["rare_weight"] * support * (
                    0.35 + 0.65 * min(relative, 1.0))
        for column, support in spatial_candidates[row].items():
            scores[column] = scores.get(column, 0.0) + policy["spatial_weight"] * support
        seeds = list(v24_ranked[row, :12]) + base[:8]
        for seed_rank, seed_column in enumerate(seeds):
            for neighbor, weight in zip(graph.neighbors[int(seed_column)], graph.weights[int(seed_column)]):
                if neighbor >= 0:
                    scores[int(neighbor)] = scores.get(int(neighbor), 0.0) + (
                        policy["cooccurrence_weight"] * float(weight) / (1 + 0.08 * seed_rank))
        base_count = len(base)
        desired = int(round((1 - policy["cardinality_weight"]) * base_count +
                            policy["cardinality_weight"] * predicted_richness[row]))
        desired = int(np.clip(desired, max(16, base_count - 3), min(30, base_count + 3)))
        ordered = [column for column, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        selected: list[int] = []
        new_zero, new_rare = 0, 0
        base_set = set(base)
        for column in ordered:
            if column not in base_set and frequencies[column] == 0:
                if new_zero >= 2 or column not in po_candidates[row]:
                    continue
                new_zero += 1
            elif column not in base_set and frequencies[column] <= 25:
                if new_rare >= 4:
                    continue
                new_rare += 1
            selected.append(column)
            if len(selected) == desired:
                break
        if len(selected) < desired:
            for column in base:
                if column not in selected:
                    selected.append(column)
                if len(selected) == desired:
                    break
        if len(selected) != len(set(selected)) or not 16 <= len(selected) <= 30:
            raise ValueError("Post-processing produced an invalid prediction row")
        result.append(selected)
    return result


def probabilities_to_base_lists(probabilities: np.ndarray, distance_km: np.ndarray
                                ) -> list[list[int]]:
    counts = v23_cardinality(distance_km)
    ranked, _ = top_rank(probabilities, int(counts.max()))
    return [list(map(int, ranked[row, :count])) for row, count in enumerate(counts)]


def score_prediction_lists(targets: np.ndarray, predictions: list[list[int]]) -> np.ndarray:
    scores = np.empty(len(predictions), dtype=np.float64)
    for row, predicted in enumerate(predictions):
        truth_count = int(np.asarray(targets[row]).sum())
        hits = int(np.asarray(targets[row])[predicted].sum())
        scores[row] = 2 * hits / max(truth_count + len(predicted), 1)
    return scores


def species_group_metrics(targets: np.ndarray, predictions: list[list[int]],
                          frequencies: np.ndarray) -> dict[str, Any]:
    result = {}
    for name, mask in (("zero_pa", frequencies == 0),
                       ("rare_1_to_25", (frequencies >= 1) & (frequencies <= 25)),
                       ("common_over_25", frequencies > 25)):
        true_positives = int(np.asarray(targets)[:, mask].sum())
        predicted_positives, hits = 0, 0
        for row, columns in enumerate(predictions):
            group_columns = [column for column in columns if mask[column]]
            predicted_positives += len(group_columns)
            hits += int(np.asarray(targets[row])[group_columns].sum()) if group_columns else 0
        result[name] = {"species": int(mask.sum()), "target_positives": true_positives,
                        "predicted_positives": predicted_positives, "true_positives": hits,
                        "precision": hits / predicted_positives if predicted_positives else None,
                        "recall": hits / true_positives if true_positives else None}
    return result


def _frequency(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    total = np.zeros(labels.shape[1], dtype=np.int64)
    for begin in range(0, len(indices), 2048):
        total += np.asarray(labels[indices[begin:begin + 2048]], dtype=np.uint8).sum(0,
                                                                                   dtype=np.int64)
    return total


def _cardinality(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    total = np.empty(len(indices), dtype=np.int16)
    for begin in range(0, len(indices), 2048):
        batch = np.asarray(labels[indices[begin:begin + 2048]], dtype=np.uint8)
        total[begin:begin + len(batch)] = batch.sum(1, dtype=np.int16)
    return total


def _role_components(rows: pd.DataFrame, role_indices: np.ndarray, spatial: PASpatialIndex,
                     po: POGridIndex) -> tuple[list[dict[int, float]], np.ndarray,
                                               list[dict[int, float]], np.ndarray]:
    coordinates = rows.iloc[role_indices][["lat", "lon"]].to_numpy(np.float64)
    spatial_candidates, pa_distance = spatial.query(coordinates)
    po_candidates, po_coverage = po.query(coordinates)
    return spatial_candidates, pa_distance, po_candidates, po_coverage


def _build_models_for_fold(name: str, split: dict[str, np.ndarray], rows: pd.DataFrame,
                           store: FeatureStore, po: POGridIndex, temporary: Path,
                           guard: RuntimeGuard, device: torch.device, seed: int
                           ) -> tuple[dict[str, Any], dict[str, Any]]:
    guard.stamp("fold_start", fold=name)
    stats = normalization_stats(store.train, split["training"])
    frequencies = _frequency(store.labels, split["training"])
    rare_indices = np.flatnonzero(frequencies <= 25)
    fold_dir = temporary / name
    fold_dir.mkdir(parents=True, exist_ok=True)
    control = MatchedV23Control(store.dims, len(store.species_ids))
    control_record = train_model(
        control, store.train, store.labels, split["training"], split["selection"], stats,
        device, fold_dir / "matched_v23_control.pt", guard, seed=seed + 10, v24=False,
        epochs=6, minimum_epochs=4,
    )
    v24 = V24MultimodalRareJSDM(store.dims, len(store.species_ids), rare_indices)
    v24_record = train_model(
        v24, store.train, store.labels, split["training"], split["selection"], stats,
        device, fold_dir / "v24_multimodal.pt", guard, seed=seed, v24=True,
        epochs=8, minimum_epochs=6,
    )
    spatial_index = PASpatialIndex(rows, store.labels, split["training"])
    graph = CooccurrenceGraph.build(store.labels, split["training"])
    predictions: dict[str, Any] = {}
    role_components: dict[str, Any] = {}
    for role in ("selection", "calibration", "assessment"):
        indices = split[role]
        control_probability, _, _ = predict_model(control, store.train, indices, stats, device,
                                                   v24=False)
        probability, raw_richness, modality_weight = predict_model(
            v24, store.train, indices, stats, device, v24=True)
        spatial_candidates, pa_distance, po_candidates, po_coverage = _role_components(
            rows, indices, spatial_index, po)
        predictions[role] = {"control": control_probability, "v24": probability,
                             "raw_richness": raw_richness,
                             "modality_weight_mean": modality_weight.mean(0)}
        role_components[role] = {"spatial": spatial_candidates, "pa_distance": pa_distance,
                                 "po": po_candidates, "po_coverage": po_coverage}
    selection = split["selection"]
    selection_values = predictions["selection"]
    selection_components = role_components["selection"]
    richness_model, richness_metadata = fit_richness_model(
        selection_values["v24"], selection_values["raw_richness"], rows.iloc[selection],
        selection_components["pa_distance"], selection_components["po_coverage"],
        _cardinality(store.labels, selection), rows.iloc[split["training"]],
        _cardinality(store.labels, split["training"]), seed=seed,
    )
    for role in ("calibration", "assessment"):
        values, components = predictions[role], role_components[role]
        values["predicted_richness"] = predict_richness(
            richness_model, richness_metadata, values["v24"], values["raw_richness"],
            rows.iloc[split[role]], components["pa_distance"], components["po_coverage"])
        values["base_lists"] = probabilities_to_base_lists(values["control"],
                                                            components["pa_distance"])
        ranked, _ = top_rank(values["v24"], 64)
        values["risk"] = ood_risk(components["pa_distance"], components["po_coverage"],
                                  values["base_lists"], ranked)
    calibration_targets = np.asarray(store.labels[split["calibration"]])
    calibration_trials = []
    for policy in POLICIES:
        predicted = compose_predictions(
            predictions["calibration"]["base_lists"], predictions["calibration"]["v24"],
            predictions["calibration"]["predicted_richness"], frequencies,
            role_components["calibration"]["spatial"], role_components["calibration"]["po"],
            graph, predictions["calibration"]["risk"], policy,
        )
        calibration_trials.append({"policy_id": policy["id"],
                                   "sample_f1": float(score_prediction_lists(
                                       calibration_targets, predicted).mean()),
                                   "surveys": len(calibration_targets)})
    training_record = {
        "matched_v23_control": {key: value for key, value in control_record.items()
                                if key != "training_frequency"},
        "v24": {key: value for key, value in v24_record.items()
                if key != "training_frequency"},
        "rare_species": int((frequencies <= 25).sum()),
        "zero_pa_species": int((frequencies == 0).sum()),
        "common_species": int((frequencies > 25).sum()),
        "normalization_fit_on_training_only": True,
        "richness": richness_metadata,
        "cooccurrence_sha256": graph.digest(),
        "calibration_trials": calibration_trials,
    }
    bundle = {"name": name, "split": split, "stats": stats, "frequencies": frequencies,
              "graph": graph, "predictions": predictions, "components": role_components,
              "calibration_trials": calibration_trials}
    del control, v24, richness_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bundle, training_record


def select_global_policy(bundles: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials = []
    for policy in POLICIES:
        records = [next(item for item in bundle["calibration_trials"]
                        if item["policy_id"] == policy["id"]) for bundle in bundles]
        surveys = sum(record["surveys"] for record in records)
        score = sum(record["sample_f1"] * record["surveys"] for record in records) / surveys
        intervention = (policy["alpha_near"] + policy["alpha_far"] + policy["rare_weight"] +
                        policy["spatial_weight"] + policy["cooccurrence_weight"] +
                        policy["cardinality_weight"])
        trials.append({"policy_id": policy["id"], "pooled_calibration_f1": score,
                       "surveys": surveys, "fold_scores": [record["sample_f1"] for record in records],
                       "intervention": intervention})
    selected_record = max(trials, key=lambda item: (item["pooled_calibration_f1"],
                                                     -item["intervention"]))
    selected = next(dict(policy) for policy in POLICIES if policy["id"] == selected_record["policy_id"])
    selected["pooled_calibration_f1"] = selected_record["pooled_calibration_f1"]
    return selected, trials


def _decode_v23_submission(path: Path, template_ids: np.ndarray, species_ids: np.ndarray
                           ) -> list[list[int]]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["surveyId", "predictions"]:
        raise ValueError("Frozen v23 submission schema changed")
    if not np.array_equal(frame.surveyId.to_numpy(np.int64), np.asarray(template_ids, np.int64)):
        raise ValueError("Frozen v23 submission is not in official template order")
    lookup = pd.Index(species_ids)
    result: list[list[int]] = []
    for text in frame.predictions.astype(str):
        ids = np.asarray([int(value) for value in text.split()], dtype=np.int64)
        columns = lookup.get_indexer(ids)
        if (columns < 0).any() or len(columns) != len(np.unique(columns)) or not 20 <= len(columns) <= 28:
            raise ValueError("Frozen v23 prediction row is invalid")
        result.append(list(map(int, columns)))
    return result


def _train_deployment(split: dict[str, np.ndarray], rows: pd.DataFrame, test_rows: pd.DataFrame,
                      store: FeatureStore, po: POGridIndex, base_lists: list[list[int]],
                      policy: dict[str, Any], temporary: Path, guard: RuntimeGuard,
                      device: torch.device) -> tuple[list[list[int]], dict[str, Any]]:
    guard.stamp("deployment_start")
    stats = normalization_stats(store.train, split["training"])
    frequencies = _frequency(store.labels, split["training"])
    rare_indices = np.flatnonzero(frequencies <= 25)
    output = temporary / "deployment"
    output.mkdir(parents=True, exist_ok=True)
    model = V24MultimodalRareJSDM(store.dims, len(store.species_ids), rare_indices)
    training = train_model(
        model, store.train, store.labels, split["training"], split["selection"], stats,
        device, output / "v24_multimodal.pt", guard, seed=SEEDS["deployment"], v24=True,
        epochs=10, minimum_epochs=6,
    )
    spatial = PASpatialIndex(rows, store.labels, split["training"])
    graph = CooccurrenceGraph.build(store.labels, split["training"])
    calibration = split["calibration"]
    calibration_probability, calibration_raw_richness, calibration_modality = predict_model(
        model, store.train, calibration, stats, device, v24=True)
    calibration_spatial, calibration_pa_distance, calibration_po, calibration_po_coverage = (
        _role_components(rows, calibration, spatial, po))
    richness_model, richness_metadata = fit_richness_model(
        calibration_probability, calibration_raw_richness, rows.iloc[calibration],
        calibration_pa_distance, calibration_po_coverage,
        _cardinality(store.labels, calibration), rows.iloc[split["training"]],
        _cardinality(store.labels, split["training"]),
        seed=SEEDS["deployment"],
    )
    test_indices = np.arange(len(store.test_ids), dtype=np.int64)
    test_probability, test_raw_richness, test_modality = predict_model(
        model, store.test, test_indices, stats, device, v24=True)
    test_coordinates = test_rows[["lat", "lon"]].to_numpy(np.float64)
    test_spatial, test_pa_distance = spatial.query(test_coordinates)
    test_po, test_po_coverage = po.query(test_coordinates)
    predicted_richness = predict_richness(
        richness_model, richness_metadata, test_probability, test_raw_richness, test_rows,
        test_pa_distance, test_po_coverage)
    ranked, _ = top_rank(test_probability, 64)
    risk = ood_risk(test_pa_distance, test_po_coverage, base_lists, ranked)
    predictions = compose_predictions(base_lists, test_probability, predicted_richness, frequencies,
                                      test_spatial, test_po, graph, risk, policy)
    record = {
        "training": {key: value for key, value in training.items()
                     if key != "training_frequency"},
        "richness": richness_metadata, "cooccurrence_sha256": graph.digest(),
        "frequency_groups": {"zero_pa": int((frequencies == 0).sum()),
                             "rare_1_to_25": int(((frequencies >= 1) & (frequencies <= 25)).sum()),
                             "common_over_25": int((frequencies > 25).sum())},
        "test": {"pa_distance_km_mean": float(test_pa_distance.mean()),
                 "po_coverage_mean": float(test_po_coverage.mean()),
                 "ood_risk_mean": float(risk.mean()),
                 "predicted_cardinality_min": min(map(len, predictions)),
                 "predicted_cardinality_mean": float(np.mean(list(map(len, predictions)))),
                 "predicted_cardinality_max": max(map(len, predictions)),
                 "modality_weight_mean": dict(zip(MODALITIES, test_modality.mean(0).tolist()))},
        "checkpoint_sha256": sha256_file(output / "v24_multimodal.pt"),
    }
    del model, richness_model, calibration_probability, test_probability
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, record


def write_submission(path: Path, template: pd.DataFrame, test_ids: np.ndarray,
                     predictions: list[list[int]], species_ids: np.ndarray) -> dict[str, Any]:
    if list(template.columns) != ["surveyId", "predictions"]:
        raise ValueError("Official sample submission schema changed")
    if set(map(int, template.surveyId)) != set(map(int, test_ids)):
        raise ValueError("Test IDs do not match the official sample submission")
    by_id = {int(survey_id): " ".join(map(str, species_ids[predicted]))
             for survey_id, predicted in zip(test_ids, predictions)}
    frame = pd.DataFrame({"surveyId": template.surveyId.astype(np.int64),
                          "predictions": [by_id[int(value)] for value in template.surveyId]})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    return validate_submission(path, template, species_ids)


def validate_submission(path: Path, template: pd.DataFrame, species_ids: np.ndarray
                        ) -> dict[str, Any]:
    frame = pd.read_csv(path)
    checks = {"columns": list(frame.columns) == ["surveyId", "predictions"],
              "row_count": len(frame) == len(template) == EXPECTED_TEST_ROWS,
              "row_order": np.array_equal(frame.surveyId.to_numpy(np.int64),
                                           template.surveyId.to_numpy(np.int64)),
              "unique_ids": frame.surveyId.nunique() == len(frame)}
    vocabulary = set(map(int, species_ids))
    counts = []
    valid_rows = True
    for text in frame.predictions.astype(str):
        values = [int(value) for value in text.split()]
        counts.append(len(values))
        valid_rows &= len(values) == len(set(values)) and set(values).issubset(vocabulary)
    checks.update({"vocabulary_and_unique_predictions": bool(valid_rows),
                   "cardinality_bounds": min(counts) >= 16 and max(counts) <= 30})
    if not all(checks.values()):
        raise ValueError(f"Submission validation failed: {checks}")
    return {"checks": checks, "rows": len(frame), "species_vocabulary": len(vocabulary),
            "prediction_count_min": min(counts), "prediction_count_mean": float(np.mean(counts)),
            "prediction_count_max": max(counts), "sha256": sha256_file(path)}


def distance_bucket(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), "200km_plus", dtype="<U20")
    result[values < 200] = "100_to_200km"
    result[values < 100] = "50_to_100km"
    result[values < 50] = "20_to_50km"
    result[values < 20] = "0_to_20km"
    return result


def summarize_by_group(frame: pd.DataFrame, column: str, score_columns: Iterable[str]
                       ) -> dict[str, Any]:
    result = {}
    for value, group in frame.groupby(column, dropna=False):
        result[str(value)] = {"n": len(group), **{name: float(group[name].mean())
                                                  for name in score_columns}}
    return result


def paired_block_bootstrap(delta: np.ndarray, blocks: np.ndarray, *, iterations: int = 500,
                           seed: int = SEEDS["bootstrap"]) -> dict[str, Any]:
    unique = np.unique(blocks)
    block_values = [np.asarray(delta)[blocks == block] for block in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        chosen = rng.integers(0, len(unique), size=len(unique))
        numerator = sum(float(block_values[index].sum()) for index in chosen)
        denominator = sum(len(block_values[index]) for index in chosen)
        estimates[iteration] = numerator / denominator
    return {"mean_difference": float(np.mean(delta)),
            "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "iterations": iterations, "seed": seed, "spatial_blocks": len(unique),
            "unit": "one_degree_spatial_block"}


def assess_bundles(bundles: list[dict[str, Any]], policy: dict[str, Any], rows: pd.DataFrame,
                   labels: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    frames = []
    fold_reports = []
    pooled_targets, pooled_base, pooled_v24, pooled_frequencies = [], [], [], []
    for fold, bundle in enumerate(bundles):
        indices = bundle["split"]["assessment"]
        values = bundle["predictions"]["assessment"]
        components = bundle["components"]["assessment"]
        targets = np.asarray(labels[indices])
        predicted = compose_predictions(
            values["base_lists"], values["v24"], values["predicted_richness"],
            bundle["frequencies"], components["spatial"], components["po"], bundle["graph"],
            values["risk"], policy,
        )
        base_scores = score_prediction_lists(targets, values["base_lists"])
        v24_scores = score_prediction_lists(targets, predicted)
        frequencies = bundle["frequencies"]
        rarity = []
        for target in targets:
            present = np.flatnonzero(target)
            rarity.append(
                f"zero={int((frequencies[present] == 0).sum())};"
                f"rare={int(((frequencies[present] >= 1) & (frequencies[present] <= 25)).sum())};"
                f"common={int((frequencies[present] > 25).sum())}"
            )
        selected_rows = rows.iloc[indices]
        frame = pd.DataFrame({
            "surveyId": selected_rows.surveyId.to_numpy(np.int64), "fold": fold,
            "spatial_block": spatial_blocks(selected_rows),
            "country": selected_rows.country.fillna("unknown").astype(str).to_numpy(),
            "pa_distance_bucket": distance_bucket(components["pa_distance"]),
            "rarity_summary": rarity, "true_cardinality": targets.sum(1).astype(int),
            "predicted_cardinality": np.asarray(list(map(len, predicted)), dtype=int),
            "frozen_v23_f1": base_scores, "v24_f1": v24_scores,
            "delta_f1": v24_scores - base_scores,
        })
        frames.append(frame)
        fold_reports.append({
            "fold": fold, "surveys": len(frame), "spatial_blocks": frame.spatial_block.nunique(),
            "frozen_v23_sample_f1": float(base_scores.mean()),
            "v24_sample_f1": float(v24_scores.mean()),
            "gain": float((v24_scores - base_scores).mean()),
            "cardinality_mae": float(np.mean(np.abs(frame.predicted_cardinality -
                                                     frame.true_cardinality))),
            "frozen_v23_cardinality_mae": float(np.mean(np.abs(
                np.asarray(list(map(len, values["base_lists"]))) - frame.true_cardinality))),
            "frozen_v23_species_groups": species_group_metrics(targets, values["base_lists"],
                                                                frequencies),
            "v24_species_groups": species_group_metrics(targets, predicted, frequencies),
            "modality_weight_mean": dict(zip(MODALITIES,
                                               values["modality_weight_mean"].tolist())),
        })
        pooled_targets.append(targets)
        pooled_base.extend(values["base_lists"])
        pooled_v24.extend(predicted)
        pooled_frequencies.append(frequencies)
    frame = pd.concat(frames, ignore_index=True)
    if frame.surveyId.duplicated().any():
        raise ValueError("The two v24 assessment folds overlap")
    bootstrap = paired_block_bootstrap(frame.delta_f1.to_numpy(), frame.spatial_block.to_numpy())
    country = summarize_by_group(frame, "country", ("frozen_v23_f1", "v24_f1", "delta_f1"))
    distance = summarize_by_group(frame, "pa_distance_bucket",
                                  ("frozen_v23_f1", "v24_f1", "delta_f1"))
    ablations = {}
    for component, fields in {
        "without_multimodal": ("alpha_near", "alpha_far"),
        "without_rare_expert": ("rare_weight",),
        "without_spatial": ("spatial_weight",),
        "without_cooccurrence": ("cooccurrence_weight",),
        "without_richness": ("cardinality_weight",),
    }.items():
        ablated = dict(policy)
        for field in fields:
            ablated[field] = 0.0
        scores = []
        for bundle in bundles:
            values = bundle["predictions"]["assessment"]
            components = bundle["components"]["assessment"]
            predictions = compose_predictions(
                values["base_lists"], values["v24"], values["predicted_richness"],
                bundle["frequencies"], components["spatial"], components["po"],
                bundle["graph"], values["risk"], ablated,
            )
            scores.extend(score_prediction_lists(
                np.asarray(labels[bundle["split"]["assessment"]]), predictions))
        ablations[component] = {"sample_f1": float(np.mean(scores)),
                                "delta_vs_full_v24": float(np.mean(scores) - frame.v24_f1.mean())}
    group_summary = {
        "note": "Rarity is fold-specific; pooled counts are sums of fold metrics.",
        "folds": [{"fold": record["fold"],
                   "frozen_v23": record["frozen_v23_species_groups"],
                   "v24": record["v24_species_groups"]} for record in fold_reports],
    }
    pooled_groups: dict[str, dict[str, Any]] = {}
    for group_name in ("zero_pa", "rare_1_to_25", "common_over_25"):
        pooled_groups[group_name] = {}
        for model_name, record_key in (("frozen_v23", "frozen_v23_species_groups"),
                                       ("v24", "v24_species_groups")):
            records = [fold[record_key][group_name] for fold in fold_reports]
            target_positives = sum(record["target_positives"] for record in records)
            predicted_positives = sum(record["predicted_positives"] for record in records)
            true_positives = sum(record["true_positives"] for record in records)
            pooled_groups[group_name][model_name] = {
                "target_positives": target_positives, "predicted_positives": predicted_positives,
                "true_positives": true_positives,
                "precision": true_positives / predicted_positives if predicted_positives else None,
                "recall": true_positives / target_positives if target_positives else None,
            }
    group_summary["pooled"] = pooled_groups
    report = {
        "surveys": len(frame), "spatial_blocks": frame.spatial_block.nunique(),
        "control_definition": (
            "The exact deployed v23 CSV is frozen for official-test inference. New-fold F1 uses a "
            "matched early-fusion refit with the frozen v23 cardinality rule because the original "
            "v23 assessment is consumed and its fold checkpoints were not exported. This is "
            "recipe-transfer evidence, not evaluation of the exact deployed v23 weights."
        ),
        "frozen_v23_sample_f1": float(frame.frozen_v23_f1.mean()),
        "v24_sample_f1": float(frame.v24_f1.mean()),
        "gain": float(frame.delta_f1.mean()), "folds": fold_reports,
        "spatial_bootstrap": bootstrap, "by_country": country,
        "by_pa_distance": distance, "rarity_groups": group_summary,
        "cardinality": {"v24_mae": float(np.mean(np.abs(frame.predicted_cardinality -
                                                         frame.true_cardinality))),
                        "true_mean": float(frame.true_cardinality.mean()),
                        "predicted_mean": float(frame.predicted_cardinality.mean())},
        "ablations": ablations, "used_for_selection": False, "now_consumed": True,
        "warning": "Matched-recipe spatial cross-fit evidence, not a hidden-test score.",
    }
    common_ok = True
    for fold in fold_reports:
        old = fold["frozen_v23_species_groups"]
        new = fold["v24_species_groups"]
        old_common = old["common_over_25"]["recall"] or 0
        new_common = new["common_over_25"]["recall"] or 0
        common_ok &= new_common >= old_common - 0.005
    pooled_rare = pooled_groups["rare_1_to_25"]
    rare_gain = ((pooled_rare["v24"]["recall"] or 0) >
                 (pooled_rare["frozen_v23"]["recall"] or 0))
    substantial_countries = [value for value in country.values() if value["n"] >= 200]
    gate_components = {
        "pooled_gain_positive": report["gain"] > 0,
        "spatial_ci_lower_positive": bootstrap["ci95"][0] > 0,
        "positive_gain_each_fold": all(record["gain"] > 0 for record in fold_reports),
        "not_one_country_only": sum(value["delta_f1"] > 0 for value in substantial_countries) >= 2,
        "common_recall_protected": common_ok,
        "rare_recall_gain": rare_gain,
        "nonzero_new_component": policy["id"] != "control",
    }
    return frame, report, gate_components


def notebook_self_tests() -> dict[str, Any]:
    values = np.asarray([[0.1, 0.8, 0.4], [0.9, 0.2, 0.3]], dtype=np.float32)
    ranked, _ = top_rank(values, 2)
    if ranked.tolist() != [[1, 2], [0, 2]]:
        raise AssertionError("top_rank self-test failed")
    targets = np.asarray([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    if not np.allclose(f1_from_ranked(targets, ranked, np.asarray([2, 1])), 1.0):
        raise AssertionError("F1 self-test failed")
    if stable_bucket("same") != stable_bucket("same"):
        raise AssertionError("stable split hashing failed")
    model = V24MultimodalRareJSDM({name: 3 for name in MODALITIES}, 7,
                                  np.asarray([1, 3]), width=16, rank=4)
    batch = {name: torch.zeros(2, 3) for name in MODALITIES}
    logits, richness, weights = model.forward_with_aux(batch)
    if logits.shape != (2, 7) or richness.shape != (2,) or weights.shape != (2, 5):
        raise AssertionError("v24 model shape self-test failed")
    if not torch.allclose(weights.sum(1), torch.ones(2), atol=1e-5):
        raise AssertionError("modality gate self-test failed")
    return {"passed": True, "tests": 5}


def _clean_directory(path: Path, allowed_parent: Path) -> None:
    resolved, parent = path.resolve(), allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"Unsafe cleanup target: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run_v24(frozen_v23_payload_b64: str) -> dict[str, Any]:
    guard = RuntimeGuard()
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
    temporary = working / "v24_runtime"
    export = working / "v24_export"
    _clean_directory(temporary, working)
    _clean_directory(export, working)
    temporary.mkdir(parents=True)
    export.mkdir(parents=True)
    failure_path = working / "failure_report.json"
    if failure_path.exists():
        failure_path.unlink()
    try:
        tests_before = notebook_self_tests()
        data_root = discover_data_root()
        v23_dir = temporary / "frozen_v23"
        frozen_v23 = verify_frozen_v23(frozen_v23_payload_b64, v23_dir)
        feature_manifest = prepare_feature_store(data_root, temporary / "features", guard)
        store = FeatureStore(temporary / "features")
        rows, test_rows, pairs = load_rows_and_pairs(data_root, store.train_ids, store.test_ids)
        template = pd.read_csv(data_root / "GLC25_SAMPLE_SUBMISSION.csv")
        if not np.array_equal(template.surveyId.to_numpy(np.int64), store.test_ids):
            test_order = pd.Index(store.test_ids).get_indexer(template.surveyId.to_numpy(np.int64))
            if (test_order < 0).any():
                raise ValueError("Official test/template IDs differ")
            store.test_ids = store.test_ids[test_order]
            store.test = {name: values[test_order] for name, values in store.test.items()}
            test_rows = test_rows.iloc[test_order].reset_index(drop=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("The full v24 notebook requires one Kaggle GPU")
        torch.set_num_threads(min(os.cpu_count() or 2, 6))
        guard.stamp("data_ready", device=torch.cuda.get_device_name(0),
                    train_rows=len(rows), test_rows=len(test_rows))
        po_path = data_root / "GLC25_P0_metadata_train.csv"
        if not po_path.is_file():
            raise FileNotFoundError("Official presence-only metadata GLC25_P0_metadata_train.csv missing")
        po = POGridIndex.build(po_path, store.species_ids,
                               rows[["lat", "lon"]].to_numpy(np.float64), guard)
        outer_bundles, training_records, split_manifests = [], {}, []
        for fold in (0, 1):
            split, split_manifest = make_outer_split(rows, fold)
            bundle, training = _build_models_for_fold(
                f"fold_{fold}", split, rows, store, po, temporary, guard, device,
                SEEDS[f"fold_{fold}"],
            )
            outer_bundles.append(bundle)
            training_records[f"fold_{fold}"] = training
            split_manifests.append(split_manifest)
        selected_policy, policy_trials = select_global_policy(outer_bundles)
        deployment_split, deployment_manifest = make_deployment_split(rows)
        v23_base_lists = _decode_v23_submission(
            v23_dir / "GLC25_PA_submission_v23.csv", template.surveyId.to_numpy(np.int64),
            store.species_ids)
        deployment_predictions, deployment_record = _train_deployment(
            deployment_split, rows, test_rows, store, po, v23_base_lists, selected_policy,
            temporary, guard, device)
        submission_path = export / "GLC25_PA_submission_v24.csv"
        submission = write_submission(submission_path, template, store.test_ids,
                                      deployment_predictions, store.species_ids)
        assessment_predictions_hashes = {}
        for bundle in outer_bundles:
            values = bundle["predictions"]["assessment"]
            components = bundle["components"]["assessment"]
            predicted = compose_predictions(
                values["base_lists"], values["v24"], values["predicted_richness"],
                bundle["frequencies"], components["spatial"], components["po"],
                bundle["graph"], values["risk"], selected_policy,
            )
            encoded = json.dumps(predicted, separators=(",", ":")).encode("utf-8")
            assessment_predictions_hashes[bundle["name"]] = sha256_bytes(encoded)
        pre_assessment_freeze = {
            "assessment_reporting_started": False, "all_models_and_policies_frozen": True,
            "selected_policy": selected_policy, "assessment_prediction_sha256": assessment_predictions_hashes,
            "submission_sha256": submission["sha256"],
            "checkpoint_sha256": {
                str(path.relative_to(temporary)): sha256_file(path)
                for path in sorted(temporary.rglob("*.pt"))},
        }
        guard.stamp("pre_assessment_freeze", submission_sha256=submission["sha256"])
        assessment_frame, assessment, gate_components = assess_bundles(
            outer_bundles, selected_policy, rows, store.labels)
        assessment_path = export / "assessment_per_survey_v24.csv"
        required_columns = ["surveyId", "fold", "spatial_block", "country",
                            "pa_distance_bucket", "rarity_summary", "true_cardinality",
                            "predicted_cardinality", "frozen_v23_f1", "v24_f1", "delta_f1"]
        assessment_frame[required_columns].to_csv(assessment_path, index=False,
                                                  lineterminator="\n")
        tests_after = notebook_self_tests()
        integrity = {
            "frozen_v23_exact": all(frozen_v23["checks"].values()),
            "official_competition_only": feature_manifest["external_data_or_weights"] is False,
            "expected_dimensions": (len(store.species_ids) == EXPECTED_SPECIES and
                                    len(store.test_ids) == EXPECTED_TEST_ROWS),
            "new_spatial_folds": all(item["v22_v23_assessments_consumed_and_not_reused"]
                                     for item in split_manifests),
            "twenty_km_buffer": all(item["minimum_assessment_training_distance_km"] >= 20
                                    for item in split_manifests),
            "selection_calibration_assessment_separate": all(
                not (set(bundle["split"]["selection"]) & set(bundle["split"]["calibration"]) or
                     set(bundle["split"]["selection"]) & set(bundle["split"]["assessment"]) or
                     set(bundle["split"]["calibration"]) & set(bundle["split"]["assessment"]))
                for bundle in outer_bundles),
            "assessment_predictions_frozen": True,
            "submission_unchanged_after_freeze": sha256_file(submission_path) ==
                                                  pre_assessment_freeze["submission_sha256"],
            "submission_schema_valid": all(submission["checks"].values()),
            "notebook_tests_before_and_after": tests_before["passed"] and tests_after["passed"],
            "runtime_within_limit": guard.elapsed_hours() < MAX_TOTAL_HOURS,
            "test_labels_unused": True, "external_pretrained_weights": False,
        }
        gate = {**gate_components, "all_integrity_checks": all(integrity.values())}
        gate["eligible_for_submission"] = all(gate.values())
        assessment_sha = sha256_file(assessment_path)
        report = {
            "experiment": EXPERIMENT, "status": "complete",
            "runtime_hours": guard.elapsed_hours(), "registered_max_total_hours": MAX_TOTAL_HOURS,
            "runtime_plan": {"expected_hours": [5.0, 8.5], "feature_preparation_cap_hours": 2.75,
                             "hard_guard_hours": 11.25, "kaggle_limit_hours": 12.0,
                             "finalization_reserve_minutes": 35,
                             "models_trained_sequentially": 5,
                             "v23_reference_runtime_hours": 6.61616224692927,
                             "vram_estimate_gb": "under 5 on one T4"},
            "frozen_v23_baseline": frozen_v23, "assessment": assessment,
            "training": {**training_records, "deployment": deployment_record},
            "selected_policy": selected_policy, "policy_trials": policy_trials,
            "pre_assessment_freeze": pre_assessment_freeze, "integrity": integrity,
            "submission_gate": gate, "submission": submission,
            "official_submission_made": False, "official_submission_reference": None,
            "official_public_score": None, "official_private_score": None,
            "external_data_or_weights": False, "pretrained_weight_provenance": [],
            "final_file_hashes": {"GLC25_PA_submission_v24.csv": submission["sha256"],
                                  "assessment_per_survey_v24.csv": assessment_sha,
                                  "v24_report.json": None, "v24_manifest.json": None},
            "hash_note": "A file cannot contain its own byte hash; the manifest records the report hash, "
                         "and the notebook prints the manifest hash after finalization.",
        }
        report_path = export / "v24_report.json"
        save_json(report_path, report)
        manifest = {
            "experiment": EXPERIMENT, "source_commit": V24_SOURCE_COMMIT,
            "source_base_commit": V23_COMMIT,
            "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
            "kaggle": {"kernel": "con1los/geolifeclef-risk-aware-sdm-phase-1",
                       "intended_version": 26, "runtime_gpu": torch.cuda.get_device_name(0)},
            "datasets": [{"slug": "geolifeclef-2025", "kind": "competition",
                          "version": "competition snapshot mounted by Kaggle"}],
            "feature_manifest": feature_manifest,
            "split_definitions": {"outer": split_manifests, "deployment": deployment_manifest,
                                  "consumed": {"v22": True, "v23": True}},
            "seeds": SEEDS, "model_configurations": {
                "matched_v23_control": {"kind": "early_fusion_residual", "width": 384,
                                        "epochs": 6, "role": "new-fold recipe-transfer control"},
                "v24": {"modality_encoders": list(MODALITIES), "width": 160,
                        "low_rank_joint_species_head": 80, "rare_threshold": 25,
                        "epochs_outer": 8, "epochs_deployment": 10,
                        "loss": "frequency-aware asymmetric + rare auxiliary + richness"},
                "postprocessing": {"policies": list(POLICIES), "selected": selected_policy,
                                   "max_zero_pa_additions": 2, "max_rare_additions": 4,
                                   "cardinality_bounds": [16, 30], "relative_count_change": 3}},
            "checkpoint_identifiers_and_hashes": pre_assessment_freeze["checkpoint_sha256"],
            "pretrained_weight_provenance": [], "external_data_or_weights": False,
            "runtime_budget": {"expected_hours": [5.0, 8.5], "hard_guard_hours": 11.25,
                               "kaggle_limit_hours": 12.0, "feature_preparation_cap_hours": 2.75,
                               "finalization_reserve_minutes": 35, "single_gpu": True,
                               "models_kept_on_gpu_concurrently": 1},
            "frozen_policies": selected_policy, "pre_assessment_freeze": pre_assessment_freeze,
            "final_file_hashes": {"GLC25_PA_submission_v24.csv": submission["sha256"],
                                  "assessment_per_survey_v24.csv": assessment_sha,
                                  "v24_report.json": sha256_file(report_path),
                                  "v24_manifest.json": None},
            "self_hash_note": "The manifest's own byte hash is emitted by the final notebook cell.",
        }
        manifest_path = export / "v24_manifest.json"
        save_json(manifest_path, manifest)
        final_hashes = {path.name: sha256_file(path) for path in sorted(export.iterdir()) if path.is_file()}
        if set(final_hashes) != {"GLC25_PA_submission_v24.csv", "v24_report.json",
                                "assessment_per_survey_v24.csv", "v24_manifest.json"}:
            raise ValueError(f"Export directory contains unexpected files: {sorted(final_hashes)}")
        guard.stamp("v24_complete", eligible=gate["eligible_for_submission"],
                    hashes=final_hashes)
        _clean_directory(temporary, working)
        return {"status": "complete", "eligible_for_submission": gate["eligible_for_submission"],
                "runtime_hours": guard.elapsed_hours(), "export_directory": str(export),
                "final_hashes": final_hashes, "assessment_gain": assessment["gain"],
                "spatial_ci95": assessment["spatial_bootstrap"]["ci95"],
                "selected_policy": selected_policy["id"],
                "instruction": ("Submit GLC25_PA_submission_v24.csv exactly once only if eligible is true."
                                if gate["eligible_for_submission"] else
                                "DO NOT SUBMIT: keep the candidate for analysis; the frozen v23 remains control.")}
    except Exception as error:
        failure = {"experiment": EXPERIMENT, "status": "failed",
                   "failed_stage": "see traceback", "error_type": type(error).__name__,
                   "error": str(error), "runtime_hours": guard.elapsed_hours(),
                   "safe_restart": "Fix the stated cause and rerun the notebook from the first cell; "
                                   "no competition submission was made.",
                   "traceback": traceback.format_exc()[-12000:]}
        save_json(failure_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
