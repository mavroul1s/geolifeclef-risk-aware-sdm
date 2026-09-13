"""Stream competition PO into deduplicated, sampling-aware geographic pseudo-surveys."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree


PO_FILENAME = "GLC25_P0_metadata_train.csv"
REQUIRED_COLUMNS = (
    "publisher", "year", "month", "day", "lat", "lon", "geoUncertaintyInM",
    "taxonRank", "date", "dayOfYear", "speciesId", "surveyId",
)
CHUNK_SIZE = 250_000
EARTH_RADIUS_KM = 6371.0088
CELL_DEGREES = 0.05
DUPLICATE_RADIUS_KM = 0.1


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("PO preparation exhausted the registered pipeline deadline")


def _coordinates(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("Coordinates must have [rows, 2] shape in latitude/longitude order")
    if not np.isfinite(result).all() or (np.abs(result) > [90, 180]).any():
        raise ValueError("Coordinates must be finite latitude/longitude within geographic bounds")
    return result


def coordinate_features(coords: np.ndarray) -> np.ndarray:
    """Fixed 34-column coordinate encoding; fit no statistics on heldout data."""
    values = _coordinates(coords)
    radians = np.deg2rad(values)
    columns = [values[:, 0] / 90, values[:, 1] / 180]
    for frequency in (1, 2, 4, 8, 16, 32, 64, 128):
        for axis in (0, 1):
            columns.extend((np.sin(radians[:, axis] * frequency), np.cos(radians[:, axis] * frequency)))
    return np.stack(columns, axis=1).astype(np.float32)


def expand_features(
    geo: np.ndarray, environment: np.ndarray, available: bool = True,
) -> np.ndarray:
    """Append matching normalized PA environment or masked PO zeros and a flag."""
    geo = np.asarray(geo, dtype=np.float32)
    environment = np.asarray(environment, dtype=np.float32)
    if geo.ndim != 2 or geo.shape[1] != 34 or environment.ndim != 2 or len(geo) != len(environment):
        raise ValueError("Expected matching 34-column geography and environmental matrices")
    if not isinstance(available, (bool, np.bool_)):
        raise ValueError("Environment availability must be a boolean")
    if not available:
        environment = np.zeros_like(environment)
    if not np.isfinite(geo).all() or not np.isfinite(environment).all():
        raise ValueError("Expert features must be finite")
    indicator = np.full((len(geo), 1), float(available), dtype=np.float32)
    return np.concatenate((geo, environment, indicator), axis=1)


def prepare_po(
    data_root: Path,
    output: Path,
    pa_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    species: np.ndarray,
    deadline: float,
) -> dict:
    """Build binary cell/publisher pseudo-surveys without reading any PA labels.

    The P0 spelling is the verified competition file name. PO environmental
    tables have not been verified, so PO features are geography only. The
    caller adds a zero environmental block and an explicit availability flag.
    Independent PO/PA numeric survey-ID namespaces can coincide: overlap is
    reported but only coordinate proximity excludes potential duplicates.
    """
    started = time.monotonic()
    _check_deadline(deadline)
    data_root, output = Path(data_root), Path(output)
    source = data_root / PO_FILENAME
    header = pd.read_csv(source, nrows=0)
    missing = sorted(set(REQUIRED_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"Competition PO schema missing required columns: {missing}")
    species = np.asarray(species)
    if species.ndim != 1 or not len(species) or not np.issubdtype(species.dtype, np.integer):
        raise ValueError("PA species vocabulary must be a nonempty integer vector")
    if len(np.unique(species)) != len(species) or not np.array_equal(species, np.sort(species)):
        raise ValueError("PA species vocabulary must be sorted and unique")
    species_index = pd.Index(species)
    # This function accesses only PA coordinates and optional IDs, never labels.
    pa_coords = _coordinates(pa_rows[["lat", "lon"]].to_numpy())
    test_coords = _coordinates(test_rows[["lat", "lon"]].to_numpy())
    all_pa_coords = np.unique(np.concatenate((pa_coords, test_coords), axis=0), axis=0)
    if not len(all_pa_coords):
        raise ValueError("At least one PA coordinate is required for PO duplicate exclusion")
    tree = BallTree(np.deg2rad(all_pa_coords), metric="haversine")
    known_ids = pd.Index(np.unique(np.concatenate([
        pd.to_numeric(frame["surveyId"], errors="raise").to_numpy(dtype=np.int64)
        for frame in (pa_rows, test_rows) if "surveyId" in frame.columns
    ]))) if "surveyId" in pa_rows.columns or "surveyId" in test_rows.columns else pd.Index([])
    counts = {
        "raw_rows": 0, "invalid_coordinate_rows": 0, "unknown_or_invalid_species_rows": 0,
        "excess_uncertainty_rows": 0, "valid_rows_before_pa_exclusion": 0,
        "near_pa_coordinate_excluded_rows": 0, "retained_rows_before_deduplication": 0,
        "numeric_survey_id_overlap_rows_diagnostic_only": 0,
    }
    raw_publishers, retained_publishers = Counter(), Counter()
    chunks = []
    schema_dtypes = {}
    used = ["publisher", "lat", "lon", "geoUncertaintyInM", "speciesId", "surveyId"]
    for chunk in pd.read_csv(source, usecols=used, dtype={"publisher": "string"}, chunksize=CHUNK_SIZE):
        _check_deadline(deadline)
        if not schema_dtypes:
            schema_dtypes = {name: str(dtype) for name, dtype in chunk.dtypes.items()}
        counts["raw_rows"] += len(chunk)
        publisher = chunk.publisher.fillna("<missing>").str.strip().replace("", "<missing>")
        raw_publishers.update(publisher.value_counts().to_dict())
        coordinates = chunk[["lat", "lon"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        labels = pd.to_numeric(chunk.speciesId, errors="coerce")
        label_indices = species_index.get_indexer(labels)
        uncertainty = pd.to_numeric(chunk.geoUncertaintyInM, errors="coerce").to_numpy(dtype=np.float64)
        coordinate_valid = np.isfinite(coordinates).all(axis=1) & (np.abs(coordinates) <= [90, 180]).all(axis=1)
        species_valid = label_indices >= 0
        uncertainty_valid = np.isnan(uncertainty) | (uncertainty <= 1000)
        counts["invalid_coordinate_rows"] += int((~coordinate_valid).sum())
        counts["unknown_or_invalid_species_rows"] += int((~species_valid).sum())
        counts["excess_uncertainty_rows"] += int((~uncertainty_valid).sum())
        if len(known_ids):
            counts["numeric_survey_id_overlap_rows_diagnostic_only"] += int(pd.to_numeric(chunk.surveyId, errors="coerce").isin(known_ids).sum())
        keep = coordinate_valid & species_valid & uncertainty_valid
        counts["valid_rows_before_pa_exclusion"] += int(keep.sum())
        selected = np.flatnonzero(keep)
        if not len(selected):
            continue
        distances = tree.query(np.deg2rad(coordinates[selected]), k=1, return_distance=True)[0][:, 0] * EARTH_RADIUS_KM
        _check_deadline(deadline)
        duplicate = distances <= DUPLICATE_RADIUS_KM
        counts["near_pa_coordinate_excluded_rows"] += int(duplicate.sum())
        selected = selected[~duplicate]
        counts["retained_rows_before_deduplication"] += len(selected)
        if not len(selected):
            continue
        # Integer cells are latitude/longitude floor bins, with closed poles
        # assigned to the final valid cell. Centers never use record averages.
        cells = np.floor(coordinates[selected] * 20).astype(np.int32)
        cells[:, 0] = np.clip(cells[:, 0], -1800, 1799)
        cells[:, 1] = np.clip(cells[:, 1], -3600, 3599)
        selected_publishers = publisher.iloc[selected].to_numpy(dtype=str)
        retained_publishers.update(Counter(selected_publishers))
        chunks.append(pd.DataFrame({
            "cell_lat": cells[:, 0], "cell_lon": cells[:, 1],
            "publisher": selected_publishers,
            "species_index": label_indices[selected].astype(np.int32),
        }))
    _check_deadline(deadline)
    if not chunks:
        raise ValueError("No usable PO observations remain after vocabulary and duplicate filtering")
    retained = pd.concat(chunks, ignore_index=True)
    del chunks
    retained["publisher"] = pd.Categorical(retained["publisher"])
    group_columns = ["cell_lat", "cell_lon", "publisher"]
    deduplicated = retained.drop_duplicates(group_columns + ["species_index"])
    # Sorted factorization makes arrays invariant to raw record order/repetition.
    group_codes, group_keys = pd.factorize(pd.MultiIndex.from_frame(deduplicated[group_columns]), sort=True)
    groups = group_keys.to_frame(index=False)
    groups.columns = group_columns
    groups["publisher"] = groups.publisher.astype(str)
    group_count = len(groups)
    labels = sparse.csr_matrix((
        np.ones(len(deduplicated), dtype=np.uint8),
        (group_codes, deduplicated.species_index.to_numpy(dtype=np.int32)),
    ), shape=(group_count, len(species)), dtype=np.uint8)
    labels.sort_indices()
    if labels.nnz != len(deduplicated) or (labels.data != 1).any() or (np.diff(labels.indptr) == 0).any():
        raise ValueError("PO aggregation failed binary, nonempty-label integrity")
    raw_group_codes = group_keys.get_indexer(pd.MultiIndex.from_frame(retained[group_columns]))
    if (raw_group_codes < 0).any():
        raise ValueError("PO support group alignment failed")
    raw_support = np.bincount(raw_group_codes, minlength=group_count)
    cell_pairs = groups[["cell_lat", "cell_lon"]].to_numpy(dtype=np.int32)
    centers = (cell_pairs.astype(np.float64) + 0.5) * CELL_DEGREES
    # All publisher pseudo-surveys in a cell share its total sampling mass.
    cell_keys = pd.MultiIndex.from_arrays(cell_pairs.T)
    _, cell_inverse, cell_group_counts = np.unique(cell_pairs, axis=0, return_inverse=True, return_counts=True)
    del cell_keys
    unique_cells = np.unique(cell_pairs, axis=0)
    unique_blocks = np.floor_divide(unique_cells, 20)
    block_keys, block_counts = np.unique(unique_blocks, axis=0, return_counts=True)
    group_blocks = np.floor_divide(cell_pairs, 20)
    block_lookup = pd.MultiIndex.from_arrays(block_keys.T)
    block_index = block_lookup.get_indexer(pd.MultiIndex.from_arrays(group_blocks.T))
    publishers_in_cell = cell_group_counts[cell_inverse]
    block_cell_count = block_counts[block_index]
    weights = 1.0 / (publishers_in_cell * np.sqrt(block_cell_count))
    weights = weights.astype(np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("PO sampling weights must be finite and positive")
    publisher_names, publisher_codes = np.unique(groups.publisher.to_numpy(dtype=str), return_inverse=True)
    weighted_exposure = np.bincount(publisher_codes, weights=weights, minlength=len(publisher_names))
    weighted_exposure /= weighted_exposure.sum()
    coverage = np.asarray(labels.getnnz(axis=0)).ravel()
    counts.update({
        "deduplicated_cell_publisher_species": int(labels.nnz),
        "removed_repeated_cell_publisher_species_rows": int(len(retained) - labels.nnz),
        "pseudo_surveys": group_count, "unique_spatial_cells": len(unique_cells),
        "unique_one_degree_blocks": len(block_keys), "publishers": len(publisher_names),
    })
    _check_deadline(deadline)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / filename for name, filename in {
        "po_features": "po_features.npy", "po_labels": "po_labels.npz",
        "po_weights": "po_weights.npy", "po_support": "po_support.npz",
    }.items()}
    np.save(paths["po_features"], coordinate_features(centers), allow_pickle=False)
    sparse.save_npz(paths["po_labels"], labels, compressed=True)
    np.save(paths["po_weights"], weights, allow_pickle=False)
    np.savez_compressed(paths["po_support"],
        coordinates=centers, cells=cell_pairs, publisher_codes=publisher_codes.astype(np.int32),
        publisher_names=publisher_names, unique_species=np.diff(labels.indptr), raw_records=raw_support,
        publishers_in_cell=publishers_in_cell, unique_cells_in_block=block_cell_count,
        species_ids=species.astype(np.int64), species_pseudo_survey_support=coverage,
    )
    _check_deadline(deadline)
    report = {
        "source": PO_FILENAME, "schema_columns": list(header.columns), "read_column_dtypes": schema_dtypes,
        "paths": {name: str(path) for name, path in paths.items()}, "counts": counts,
        "coordinate_feature_dimension": 34, "po_environmental_features": "masked; not verified available",
        "cell_degrees": CELL_DEGREES, "representative_coordinates": "fixed cell center",
        "deduplication_key": group_columns + ["speciesId"],
        "maximum_nonmissing_geographic_uncertainty_m": 1000,
        "pa_duplicate_exclusion_radius_m": 100,
        "pa_duplicate_exclusion_reference": "all PA train and test coordinates; no species labels read",
        "numeric_id_overlap_policy": "diagnostic only; independent namespaces not assumed identical",
        "weight_formula": "1 / (publishers_in_cell * sqrt(unique_cells_in_1degree_block))",
        "sampling": "replacement using normalized weights; raw occurrence counts never increase group weight",
        "publisher_is_model_input": False,
        "raw_publisher_record_exposure": {str(k): int(v) for k, v in sorted(raw_publishers.items())},
        "retained_publisher_record_exposure": {str(k): int(v) for k, v in sorted(retained_publishers.items())},
        "weighted_publisher_draw_fraction": {str(k): float(v) for k, v in zip(publisher_names, weighted_exposure)},
        "pa_vocabulary_species": len(species), "po_covered_species": int((coverage > 0).sum()),
        "pa_species_without_po_support": int((coverage == 0).sum()),
        "external_data_or_weights": False, "pa_labels_accessed": False,
        "preparation_seconds": time.monotonic() - started,
    }
    (output / "po_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
