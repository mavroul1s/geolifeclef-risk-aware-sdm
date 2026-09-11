#!/usr/bin/env python
"""Prepare identical multimodal train/calibration/holdout splits for fair comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LANDSAT_SHAPE = (6, 4, 21)
CLIMATE_SHAPE = (4, 19, 12)


def construct_patch_path(root: Path, survey_id: int) -> Path:
    text = str(int(survey_id))
    return root / text[-2:] / text[-4:-2] / f"{text}.tiff"


def split_survey_ids(
    survey_table: pd.DataFrame,
    *,
    holdout_country: str = "Netherlands",
    seed: int = 2025,
    calibration_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Freeze a country holdout and spatial-block calibration partition."""
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    holdout_mask = survey_table["country"].astype("string").str.strip() == holdout_country
    validation_ids = survey_table.loc[holdout_mask, "surveyId"].to_numpy(dtype=np.int64)
    development = survey_table.loc[~holdout_mask].copy()
    if len(validation_ids) == 0 or len(development) == 0:
        raise ValueError(f"Country holdout {holdout_country!r} produced an empty split")

    lon = pd.to_numeric(development["lon"], errors="coerce").fillna(0).to_numpy()
    lat = pd.to_numeric(development["lat"], errors="coerce").fillna(0).to_numpy()
    block_x = np.floor((lon + 180.0) * 111.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.2) / 10)
    block_y = np.floor((lat + 90.0) * 111.0 / 10)
    block_hash = (
        block_x.astype(np.int64) * 73856093
        ^ block_y.astype(np.int64) * 19349663
        ^ int(seed)
    )
    buckets = np.mod(block_hash, 10_000) / 10_000.0
    calibration_mask = buckets < calibration_fraction
    train_ids = development.loc[~calibration_mask, "surveyId"].to_numpy(dtype=np.int64)
    calibration_ids = development.loc[calibration_mask, "surveyId"].to_numpy(dtype=np.int64)
    if len(train_ids) == 0 or len(calibration_ids) == 0:
        raise ValueError("Spatial-block calibration split produced an empty partition")
    return train_ids, calibration_ids, validation_ids


def deterministic_limit(ids: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or len(ids) <= maximum:
        return ids
    rng = np.random.default_rng(seed)
    return ids[np.sort(rng.choice(len(ids), size=maximum, replace=False))]


def static_features(rows: pd.DataFrame) -> np.ndarray:
    lon = pd.to_numeric(rows["lon"], errors="coerce").to_numpy(dtype=np.float32)
    lat = pd.to_numeric(rows["lat"], errors="coerce").to_numpy(dtype=np.float32)
    year = pd.to_numeric(rows["year"], errors="coerce").to_numpy(dtype=np.float32)
    uncertainty = np.log1p(
        np.maximum(pd.to_numeric(rows["geoUncertaintyInM"], errors="coerce"), 0)
    ).to_numpy(dtype=np.float32)
    area = np.log1p(
        np.maximum(pd.to_numeric(rows["areaInM2"], errors="coerce"), 0)
    ).to_numpy(dtype=np.float32)
    columns = [lon, lat, year, uncertainty, area]
    for frequency in (1, 2, 4, 8):
        columns.extend(
            [
                np.sin(np.deg2rad(lat) * frequency),
                np.cos(np.deg2rad(lat) * frequency),
                np.sin(np.deg2rad(lon) * frequency),
                np.cos(np.deg2rad(lon) * frequency),
            ]
        )
    return np.stack(columns, axis=1).astype(np.float32)


def read_sentinel(path: Path, image_size: int) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        image = dataset.read(
            out_shape=(dataset.count, image_size, image_size),
            out_dtype=np.float32,
            resampling=Resampling.bilinear,
        )
    if image.shape[0] != 4:
        raise ValueError(f"Expected four Sentinel bands at {path}, got {image.shape}")
    normalized = np.zeros_like(image, dtype=np.float32)
    for band_index, band in enumerate(image):
        finite = np.isfinite(band)
        if not finite.any():
            continue
        low, high = np.nanpercentile(band[finite], (2, 98))
        if high > low:
            normalized[band_index] = np.clip((band - low) / (high - low), 0, 1)
    return np.nan_to_num(normalized)


def build_split(
    rows: pd.DataFrame,
    pairs: pd.DataFrame,
    survey_ids: np.ndarray,
    species_ids: np.ndarray,
    data_root: Path,
    image_size: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    row_lookup = rows.set_index("surveyId")
    species_to_column = {int(species_id): index for index, species_id in enumerate(species_ids)}
    landsat_root = data_root / "SateliteTimeSeries-Landsat" / "cubes" / "PA-train"
    climate_root = data_root / "BioclimTimeSeries" / "cubes" / "PA-train"
    sentinel_root = data_root / "SatelitePatches" / "PA-train"
    selected, landsat, climate, sentinel, missing = [], [], [], [], []
    for survey_id_value in survey_ids:
        survey_id = int(survey_id_value)
        landsat_path = landsat_root / f"GLC25-PA-train-landsat-time-series_{survey_id}_cube.pt"
        climate_path = climate_root / f"GLC25-PA-train-bioclimatic_monthly_{survey_id}_cube.pt"
        sentinel_path = construct_patch_path(sentinel_root, survey_id)
        if not landsat_path.is_file() or not climate_path.is_file() or not sentinel_path.is_file():
            missing.append(survey_id)
            continue
        landsat_cube = torch.load(landsat_path, map_location="cpu", weights_only=True)
        climate_cube = torch.load(climate_path, map_location="cpu", weights_only=True)
        if not isinstance(landsat_cube, torch.Tensor) or tuple(landsat_cube.shape) != LANDSAT_SHAPE:
            raise ValueError(f"Unexpected Landsat shape at {landsat_path}")
        if not isinstance(climate_cube, torch.Tensor) or tuple(climate_cube.shape) != CLIMATE_SHAPE:
            raise ValueError(f"Unexpected bioclimatic shape at {climate_path}")
        landsat.append(landsat_cube.permute(2, 1, 0).reshape(84, 6).numpy())
        climate.append(climate_cube.permute(1, 2, 0).reshape(228, 4).numpy())
        sentinel.append(read_sentinel(sentinel_path, image_size))
        selected.append(survey_id)
    if not selected:
        raise ValueError("No surveys had all requested modalities")

    selected_array = np.asarray(selected, dtype=np.int64)
    labels = np.zeros((len(selected), len(species_ids)), dtype=np.uint8)
    row_index = {survey_id: index for index, survey_id in enumerate(selected)}
    selected_pairs = pairs.loc[pairs["surveyId"].isin(selected)]
    for survey_id, species_id in selected_pairs[["surveyId", "speciesId"]].itertuples(index=False):
        labels[row_index[int(survey_id)], species_to_column[int(species_id)]] = 1
    return (
        {
            "sample_id": selected_array,
            "labels": labels,
            "landsat": np.stack(landsat).astype(np.float32),
            "climate": np.stack(climate).astype(np.float32),
            "sentinel": np.stack(sentinel).astype(np.float32),
            "static": static_features(row_lookup.loc[selected].reset_index()),
        },
        missing,
    )


def normalize_splits(splits: list[dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    train = splits[0]
    stats: dict[str, dict[str, list[float]]] = {}
    axes = {"landsat": (0, 1), "climate": (0, 1), "sentinel": (0, 2, 3), "static": (0,)}
    for name, reduction_axes in axes.items():
        mean = np.nanmean(train[name], axis=reduction_axes).astype(np.float32)
        std = np.nanstd(train[name], axis=reduction_axes).astype(np.float32)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std[std < 1e-6] = 1.0
        shape = [1] * train[name].ndim
        if name in ("landsat", "climate"):
            shape[-1] = len(mean)
        elif name == "sentinel":
            shape[1] = len(mean)
        else:
            shape[-1] = len(mean)
        for split in splits:
            values = np.nan_to_num(split[name], nan=0.0, posinf=0.0, neginf=0.0)
            split[name] = ((values - mean.reshape(shape)) / std.reshape(shape)).astype(
                np.float16 if name != "static" else np.float32
            )
        stats[name] = {"mean": mean.tolist(), "std": std.tolist()}
    return stats


def split_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.sort(ids).astype("<i8").tobytes()).hexdigest()


def main() -> None:
    preparation_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/spatial_multimodal"))
    parser.add_argument("--holdout-country", default="Netherlands")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-train-surveys", type=int)
    parser.add_argument("--max-calibration-surveys", type=int)
    parser.add_argument("--max-validation-surveys", type=int)
    args = parser.parse_args()

    metadata = pd.read_csv(args.data_root / "GLC25_PA_metadata_train.csv")
    pairs = metadata[["surveyId", "speciesId"]].dropna().drop_duplicates().copy()
    pairs["surveyId"] = pairs["surveyId"].astype("int64")
    pairs["speciesId"] = pairs["speciesId"].astype("int64")
    rows = metadata.dropna(subset=["surveyId"]).drop_duplicates("surveyId").copy()
    rows["surveyId"] = rows["surveyId"].astype("int64")
    species_ids = np.sort(pairs["speciesId"].unique())
    train_ids, calibration_ids, validation_ids = split_survey_ids(
        rows, holdout_country=args.holdout_country, seed=args.seed
    )
    train_ids = deterministic_limit(train_ids, args.max_train_surveys, args.seed)
    calibration_ids = deterministic_limit(
        calibration_ids, args.max_calibration_surveys, args.seed + 1
    )
    validation_ids = deterministic_limit(validation_ids, args.max_validation_surveys, args.seed + 2)
    split_ids = (train_ids, calibration_ids, validation_ids)
    names = ("train", "calibration", "validation")
    built, missing = [], {}
    for name, ids in zip(names, split_ids):
        values, missing_ids = build_split(
            rows, pairs, ids, species_ids, args.data_root, args.image_size
        )
        built.append(values)
        missing[name] = len(missing_ids)
    normalization = normalize_splits(built)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in zip(names, built):
        np.savez_compressed(args.output_dir / f"{name}.npz", **values)
    manifest = {
        "seed": args.seed,
        "holdout_country": args.holdout_country,
        "calibration_policy": "deterministic 10 km spatial-block hash within non-holdout countries",
        "image_size": args.image_size,
        "species": int(len(species_ids)),
        "species_ids": species_ids.tolist(),
        "samples": {name: int(len(values["sample_id"])) for name, values in zip(names, built)},
        "split_sha256": {
            name: split_digest(values["sample_id"]) for name, values in zip(names, built)
        },
        "missing_all_modalities": missing,
        "normalization": normalization,
        "fairness_constraint": "All compared models use these exact split hashes.",
        "preparation_seconds": time.monotonic() - preparation_started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
