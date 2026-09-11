#!/usr/bin/env python
"""Create compact canonical PA train/validation NPZ splits from Landsat cubes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_pairs(metadata_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(metadata_path, usecols=["surveyId", "speciesId"])
    pairs = metadata.dropna(subset=["surveyId", "speciesId"]).copy()
    pairs["surveyId"] = pairs["surveyId"].astype("int64")
    pairs["speciesId"] = pairs["speciesId"].astype("int64")
    pairs = pairs.drop_duplicates(["surveyId", "speciesId"])
    survey_ids = pairs["surveyId"].unique()
    species_ids = np.sort(pairs["speciesId"].unique())
    return pairs, survey_ids, species_ids


def split_survey_ids(
    survey_ids: np.ndarray, *, seed: int, validation_fraction: float, max_train: int, max_validation: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    shuffled = survey_ids.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_ids, train_ids = shuffled[:validation_count], shuffled[validation_count:]
    return train_ids[:max_train], validation_ids[:max_validation]


def build_split(
    pairs: pd.DataFrame, survey_ids: np.ndarray, species_ids: np.ndarray, cube_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    species_to_column = {int(species_id): index for index, species_id in enumerate(species_ids)}
    selected_ids, features, missing = [], [], []
    for survey_id in survey_ids:
        cube_path = cube_root / f"GLC25-PA-train-landsat-time-series_{int(survey_id)}_cube.pt"
        if not cube_path.is_file():
            missing.append(int(survey_id))
            continue
        cube = torch.load(cube_path, map_location="cpu", weights_only=True)
        if not isinstance(cube, torch.Tensor) or tuple(cube.shape) != (6, 4, 21):
            raise ValueError(f"Unexpected Landsat cube shape at {cube_path}: {getattr(cube, 'shape', None)}")
        # Raw cube is [band, season, year]; the TCN consumes chronological [year*season, band].
        features.append(cube.permute(2, 1, 0).reshape(84, 6).numpy().astype(np.float32, copy=False))
        selected_ids.append(int(survey_id))
    if not selected_ids:
        raise ValueError("No requested survey has a usable Landsat cube")
    selected = np.asarray(selected_ids, dtype=np.int64)
    row_index = {survey_id: index for index, survey_id in enumerate(selected_ids)}
    labels = np.zeros((len(selected), len(species_ids)), dtype=np.uint8)
    split_pairs = pairs.loc[pairs["surveyId"].isin(selected_ids)]
    for survey_id, species_id in split_pairs[["surveyId", "speciesId"]].itertuples(index=False):
        labels[row_index[int(survey_id)], species_to_column[int(species_id)]] = 1
    return selected, np.stack(features), labels, missing


def normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(features), features, mean.reshape(1, 1, -1))
    return ((filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/landsat_pa_smoke_train.npz"))
    parser.add_argument("--val-output", type=Path, default=Path("data/processed/landsat_pa_smoke_val.npz"))
    parser.add_argument("--manifest-path", type=Path, default=Path("data/processed/landsat_pa_smoke_manifest.json"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-surveys", type=int, default=12000)
    parser.add_argument("--max-validation-surveys", type=int, default=3000)
    args = parser.parse_args()
    metadata_path = args.data_root / "GLC25_PA_metadata_train.csv"
    cube_root = args.data_root / "SateliteTimeSeries-Landsat" / "cubes" / "PA-train"
    pairs, all_survey_ids, species_ids = read_pairs(metadata_path)
    train_ids, val_ids = split_survey_ids(
        all_survey_ids,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        max_train=args.max_train_surveys,
        max_validation=args.max_validation_surveys,
    )
    train_sample_ids, train_features, train_labels, train_missing = build_split(pairs, train_ids, species_ids, cube_root)
    val_sample_ids, val_features, val_labels, val_missing = build_split(pairs, val_ids, species_ids, cube_root)
    mean = np.nanmean(train_features, axis=(0, 1)).astype(np.float32)
    std = np.nanstd(train_features, axis=(0, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_features, val_features = normalize(train_features, mean, std), normalize(val_features, mean, std)
    for output in (args.train_output, args.val_output, args.manifest_path):
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.train_output, labels=train_labels, landsat=train_features, sample_id=train_sample_ids)
    np.savez_compressed(args.val_output, labels=val_labels, landsat=val_features, sample_id=val_sample_ids)
    manifest = {
        "seed": args.seed,
        "raw_landsat_shape": [6, 4, 21],
        "canonical_landsat_shape": [84, 6],
        "species_ids": species_ids.tolist(),
        "train_samples": int(len(train_sample_ids)),
        "validation_samples": int(len(val_sample_ids)),
        "species": int(len(species_ids)),
        "missing_train_cubes": len(train_missing),
        "missing_validation_cubes": len(val_missing),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
