#!/usr/bin/env python
"""Prepare every PA training survey and the official unlabeled PA test surveys."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_spatial_multimodal import (
    CLIMATE_SHAPE,
    LANDSAT_SHAPE,
    construct_patch_path,
    normalize_splits,
    read_sentinel,
    static_features,
)


def split_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def feature_paths(data_root: Path, source: str, survey_id: int) -> tuple[Path, Path, Path]:
    if source not in {"PA-train", "PA-test"}:
        raise ValueError(f"Unsupported PA source: {source}")
    source_token = "train" if source == "PA-train" else "test"
    landsat_stem = "landsat-time-series" if source == "PA-train" else "landsat_time_series"
    landsat_path = (
        data_root
        / "SateliteTimeSeries-Landsat"
        / "cubes"
        / source
        / f"GLC25-PA-{source_token}-{landsat_stem}_{survey_id}_cube.pt"
    )
    climate_path = (
        data_root
        / "BioclimTimeSeries"
        / "cubes"
        / source
        / f"GLC25-PA-{source_token}-bioclimatic_monthly_{survey_id}_cube.pt"
    )
    sentinel_path = construct_patch_path(data_root / "SatelitePatches" / source, survey_id)
    return landsat_path, climate_path, sentinel_path


def build_feature_split(
    rows: pd.DataFrame,
    survey_ids: np.ndarray,
    species_ids: np.ndarray,
    data_root: Path,
    *,
    source: str,
    image_size: int,
    pairs: pd.DataFrame | None = None,
) -> tuple[dict[str, np.ndarray], list[int]]:
    if source not in {"PA-train", "PA-test"}:
        raise ValueError(f"Unsupported PA source: {source}")
    row_lookup = rows.set_index("surveyId")
    selected: list[int] = []
    landsat: list[np.ndarray] = []
    climate: list[np.ndarray] = []
    sentinel: list[np.ndarray] = []
    missing: list[int] = []
    for survey_id_value in survey_ids:
        survey_id = int(survey_id_value)
        landsat_path, climate_path, sentinel_path = feature_paths(
            data_root, source, survey_id
        )
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
        raise ValueError(f"No surveys had all requested modalities for {source}")

    selected_array = np.asarray(selected, dtype=np.int64)
    labels = np.zeros((len(selected), len(species_ids)), dtype=np.uint8)
    if pairs is not None:
        species_to_column = {
            int(species_id): index for index, species_id in enumerate(species_ids)
        }
        row_index = {survey_id: index for index, survey_id in enumerate(selected)}
        selected_pairs = pairs.loc[pairs["surveyId"].isin(selected)]
        for survey_id, species_id in selected_pairs[["surveyId", "speciesId"]].itertuples(
            index=False
        ):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()
    started = time.monotonic()

    train_metadata = pd.read_csv(args.data_root / "GLC25_PA_metadata_train.csv")
    test_metadata = pd.read_csv(args.data_root / "GLC25_PA_metadata_test.csv")
    sample_submission = pd.read_csv(args.data_root / "GLC25_SAMPLE_SUBMISSION.csv")
    pairs = train_metadata[["surveyId", "speciesId"]].dropna().drop_duplicates().copy()
    pairs = pairs.astype({"surveyId": "int64", "speciesId": "int64"})
    train_rows = train_metadata.dropna(subset=["surveyId"]).drop_duplicates("surveyId").copy()
    test_rows = test_metadata.dropna(subset=["surveyId"]).drop_duplicates("surveyId").copy()
    train_rows["surveyId"] = train_rows["surveyId"].astype("int64")
    test_rows["surveyId"] = test_rows["surveyId"].astype("int64")
    species_ids = np.sort(pairs["speciesId"].unique())
    train_ids = train_rows["surveyId"].to_numpy(dtype=np.int64)
    test_ids = test_rows["surveyId"].to_numpy(dtype=np.int64)
    template_ids = sample_submission["surveyId"].to_numpy(dtype=np.int64)
    if set(test_ids.tolist()) != set(template_ids.tolist()):
        raise ValueError("PA test metadata and sample submission contain different survey IDs")

    train, train_missing = build_feature_split(
        train_rows,
        train_ids,
        species_ids,
        args.data_root,
        source="PA-train",
        image_size=args.image_size,
        pairs=pairs,
    )
    test, test_missing = build_feature_split(
        test_rows,
        test_ids,
        species_ids,
        args.data_root,
        source="PA-test",
        image_size=args.image_size,
    )
    if train_missing or test_missing:
        raise ValueError(
            f"Official run requires complete modalities; missing train={len(train_missing)}, "
            f"test={len(test_missing)}"
        )
    normalization = normalize_splits([train, test])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "full_train.npz", **train)
    np.savez_compressed(args.output_dir / "official_test.npz", **test)
    manifest = {
        "protocol": "full PA retraining plus official unlabeled PA test inference",
        "train_samples": int(len(train["sample_id"])),
        "test_samples": int(len(test["sample_id"])),
        "species": int(len(species_ids)),
        "species_ids": species_ids.tolist(),
        "train_ids_sha256": split_digest(train["sample_id"]),
        "test_ids_sha256": split_digest(test["sample_id"]),
        "sample_submission_ids_sha256": split_digest(template_ids),
        "test_and_template_sets_match": True,
        "test_and_template_order_match": bool(np.array_equal(test_ids, template_ids)),
        "missing_all_modalities": {"train": 0, "test": 0},
        "normalization": normalization,
        "test_labels_used": False,
        "preparation_seconds": time.monotonic() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
