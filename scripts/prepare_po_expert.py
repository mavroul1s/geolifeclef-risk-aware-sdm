"""Stream competition PO into deduplicated, sampling-aware geographic pseudo-surveys."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree

from scripts.prepare_environmental_challenger import aligned_table, discover_environment_pairs


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


def _stream_environment(
    path: Path, requested_ids: np.ndarray, columns: list[str], deadline: float,
    landcover_stratum: bool = False,
) -> tuple[np.ndarray, dict]:
    """Read one PO environmental source and align only requested rows in memory."""
    _check_deadline(deadline)
    header = pd.read_csv(path, nrows=0)
    required = {"surveyId", *columns}
    if not required.issubset(header.columns):
        raise ValueError(f"Environmental schema missing requested columns in {path.name}")
    index = pd.Index(requested_ids)
    if not index.is_unique:
        raise ValueError("Requested environmental IDs must be unique")
    width = 1 if landcover_stratum else len(columns)
    result = np.full((len(requested_ids), width), np.nan, dtype=np.float32)
    seen = np.zeros(len(requested_ids), dtype=bool)
    all_source_ids = []
    source_rows = 0
    for chunk in pd.read_csv(path, usecols=["surveyId", *columns], chunksize=CHUNK_SIZE):
        _check_deadline(deadline)
        numeric_ids = pd.to_numeric(chunk.surveyId, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_ids).all() or (numeric_ids != np.floor(numeric_ids)).any():
            raise ValueError(f"Environmental source has invalid survey IDs: {path.name}")
        source_ids = numeric_ids.astype(np.int64)
        all_source_ids.append(source_ids)
        source_rows += len(chunk)
        destination = index.get_indexer(source_ids)
        relevant = destination >= 0
        if not relevant.any():
            continue
        selected = destination[relevant]
        if seen[selected].any() or len(np.unique(selected)) != len(selected):
            raise ValueError(f"Duplicate environmental survey IDs in {path.name}")
        values = chunk.loc[relevant, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        if landcover_stratum:
            # The verified columns are numerical predictors, not documented
            # class probabilities. Do not interpret their argmax as a habitat.
            available = np.isfinite(values[:, 0])
            strata = np.full(len(values), -1, dtype=np.float32)
            strata[available] = np.floor(values[available, 0] / 5.)
            values = strata[:, None]
        result[selected] = values
        seen[selected] = True
    _check_deadline(deadline)
    concatenated_ids = np.concatenate(all_source_ids) if all_source_ids else np.empty(0, dtype=np.int64)
    if len(np.unique(concatenated_ids)) != source_rows:
        raise ValueError(f"Duplicate environmental survey IDs in {path.name}")
    if not seen.all():
        raise ValueError(f"{int((~seen).sum())} requested survey IDs absent from {path.name}")
    return result, {
        "file": path.name, "schema_columns": list(header.columns), "feature_columns": columns,
        "source_rows": source_rows, "requested_rows": len(requested_ids),
        "requested_missing_values": int(np.isnan(result).sum()),
        "source_ids_unique": True, "requested_ids_complete": True,
    }


def prepare_po(
    data_root: Path,
    output: Path,
    pa_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    species: np.ndarray,
    deadline: float,
) -> dict:
    """Build binary cell/publisher pseudo-surveys without reading any PA labels.

    The P0 spelling is the verified metadata file name; environmental sources
    use PO in their names. Geography is saved separately from raw environmental
    arrays; the caller fits each path's normalization on PA training rows only.
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
        "invalid_survey_id_rows": 0,
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
        survey_ids = pd.to_numeric(chunk.surveyId, errors="coerce").to_numpy(dtype=np.float64)
        id_valid = np.isfinite(survey_ids) & (survey_ids == np.floor(survey_ids))
        coordinate_valid = np.isfinite(coordinates).all(axis=1) & (np.abs(coordinates) <= [90, 180]).all(axis=1)
        species_valid = label_indices >= 0
        uncertainty_valid = np.isnan(uncertainty) | (uncertainty <= 1000)
        counts["invalid_coordinate_rows"] += int((~coordinate_valid).sum())
        counts["unknown_or_invalid_species_rows"] += int((~species_valid).sum())
        counts["excess_uncertainty_rows"] += int((~uncertainty_valid).sum())
        counts["invalid_survey_id_rows"] += int((~id_valid).sum())
        if len(known_ids):
            counts["numeric_survey_id_overlap_rows_diagnostic_only"] += int(pd.to_numeric(chunk.surveyId, errors="coerce").isin(known_ids).sum())
        keep = coordinate_valid & species_valid & uncertainty_valid & id_valid
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
            "survey_id": survey_ids[selected].astype(np.int64),
        }))
    _check_deadline(deadline)
    if not chunks:
        raise ValueError("No usable PO observations remain after vocabulary and duplicate filtering")
    retained = pd.concat(chunks, ignore_index=True)
    del chunks
    retained["publisher"] = pd.Categorical(retained["publisher"])
    # Competition PA/PO file naming was checked against all five real families.
    pa_ids = pd.to_numeric(pa_rows.surveyId, errors="raise").to_numpy(dtype=np.int64)
    test_ids = pd.to_numeric(test_rows.surveyId, errors="raise").to_numpy(dtype=np.int64)
    environment_sources = []
    pa_environment, test_environment = [], []
    environment_columns = []
    for pa_path, test_path in discover_environment_pairs(data_root):
        _check_deadline(deadline)
        po_name = re.sub(r"(?i)PA[-_]train", "PO-train", pa_path.name)
        po_path = pa_path.with_name(po_name)
        if po_path == pa_path or not po_path.is_file():
            raise FileNotFoundError(f"Verified matching PO environmental file not found: {po_name}")
        a, b = aligned_table(pa_path, pa_ids), aligned_table(test_path, test_ids)
        if set(a.columns) != set(b.columns) or not len(a.columns):
            raise ValueError(f"PA environmental train/test schema mismatch: {pa_path.name}")
        columns = list(a.columns)
        b = b[columns]
        pa_environment.append(a.to_numpy(dtype=np.float32))
        test_environment.append(b.to_numpy(dtype=np.float32))
        environment_columns.extend(f"{pa_path.parent.name}/{column}" for column in columns)
        environment_sources.append((pa_path, test_path, po_path, columns))
    landcover_sources = [source for source in environment_sources if any("landcover" in c.lower() for c in source[3])]
    if len(landcover_sources) != 1:
        raise ValueError("Expected exactly one verified landcover environmental source")
    _, _, landcover_path, landcover_columns = landcover_sources[0]
    metadata_ids = np.unique(retained.survey_id.to_numpy(dtype=np.int64))
    landcover, landcover_schema = _stream_environment(
        landcover_path, metadata_ids, landcover_columns, deadline, landcover_stratum=True,
    )
    retained["landcover_stratum"] = landcover[pd.Index(metadata_ids).get_indexer(retained.survey_id), 0].astype(np.int16)
    group_columns = ["cell_lat", "cell_lon", "publisher", "landcover_stratum"]
    # Keep minimum ID per binary presence, then minimum group ID as the
    # environmental representative. Repeating/reordering records has no effect.
    retained = retained.sort_values("survey_id", kind="stable")
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
    representative_ids = np.full(group_count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(representative_ids, group_codes, deduplicated.survey_id.to_numpy(dtype=np.int64))
    # One environmental survey may represent multiple groups if metadata uses
    # that ID for several species; align unique IDs and then restore group rows.
    unique_representatives, representative_inverse = np.unique(representative_ids, return_inverse=True)
    po_environment, environmental_schema = [], []
    for pa_path, test_path, po_path, columns in environment_sources:
        values, schema = _stream_environment(po_path, unique_representatives, columns, deadline)
        po_environment.append(values[representative_inverse])
        schema.update({"pa_train_file": str(pa_path.relative_to(data_root)),
                       "pa_test_file": str(test_path.relative_to(data_root)),
                       "po_file": str(po_path.relative_to(data_root))})
        environmental_schema.append(schema)
    cell_pairs = groups[["cell_lat", "cell_lon"]].to_numpy(dtype=np.int32)
    centers = (cell_pairs.astype(np.float64) + 0.5) * CELL_DEGREES
    # Split cell mass equally over publishers, and each publisher's mass equally
    # over its ecological strata, so strata cannot reinflate publisher exposure.
    cell_publisher_columns = ["cell_lat", "cell_lon", "publisher"]
    cell_publishers = groups[cell_publisher_columns].drop_duplicates()
    cell_publisher_counts = cell_publishers.groupby(["cell_lat", "cell_lon"], observed=True).size()
    cell_publisher_strata = groups.groupby(cell_publisher_columns, observed=True).size()
    publishers_in_cell = cell_publisher_counts.reindex(pd.MultiIndex.from_arrays(cell_pairs.T)).to_numpy(dtype=np.int64)
    strata_in_cell_publisher = cell_publisher_strata.reindex(pd.MultiIndex.from_frame(groups[cell_publisher_columns])).to_numpy(dtype=np.int64)
    unique_cells = np.unique(cell_pairs, axis=0)
    unique_blocks = np.floor_divide(unique_cells, 20)
    block_keys, block_counts = np.unique(unique_blocks, axis=0, return_counts=True)
    group_blocks = np.floor_divide(cell_pairs, 20)
    block_lookup = pd.MultiIndex.from_arrays(block_keys.T)
    block_index = block_lookup.get_indexer(pd.MultiIndex.from_arrays(group_blocks.T))
    block_cell_count = block_counts[block_index]
    weights = 1.0 / (publishers_in_cell * strata_in_cell_publisher * np.sqrt(block_cell_count))
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
        "landcover_strata": int(groups.landcover_stratum.nunique()),
    })
    _check_deadline(deadline)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / filename for name, filename in {
        "po_features": "po_features.npy", "po_labels": "po_labels.npz",
        "po_weights": "po_weights.npy", "po_support": "po_support.npz",
        "pa_environment_raw": "pa_environment_raw.npy",
        "test_environment_raw": "test_environment_raw.npy",
        "po_environment_raw": "po_environment_raw.npy",
    }.items()}
    np.save(paths["po_features"], coordinate_features(centers), allow_pickle=False)
    sparse.save_npz(paths["po_labels"], labels, compressed=True)
    np.save(paths["po_weights"], weights, allow_pickle=False)
    np.save(paths["pa_environment_raw"], np.concatenate(pa_environment, axis=1), allow_pickle=False)
    np.save(paths["test_environment_raw"], np.concatenate(test_environment, axis=1), allow_pickle=False)
    np.save(paths["po_environment_raw"], np.concatenate(po_environment, axis=1), allow_pickle=False)
    np.savez_compressed(paths["po_support"],
        coordinates=centers, cells=cell_pairs, publisher_codes=publisher_codes.astype(np.int32),
        publisher_names=publisher_names, unique_species=np.diff(labels.indptr), raw_records=raw_support,
        publishers_in_cell=publishers_in_cell, unique_cells_in_block=block_cell_count,
        strata_in_cell_publisher=strata_in_cell_publisher,
        landcover_stratum=groups.landcover_stratum.to_numpy(dtype=np.int16),
        representative_survey_ids=representative_ids,
        species_ids=species.astype(np.int64), species_pseudo_survey_support=coverage,
    )
    _check_deadline(deadline)
    report = {
        "source": PO_FILENAME, "schema_columns": list(header.columns), "read_column_dtypes": schema_dtypes,
        "paths": {name: str(path) for name, path in paths.items()}, "counts": counts,
        "coordinate_feature_dimension": 34,
        "environment_columns": environment_columns, "raw_environment_dimension": len(environment_columns),
        "environmental_sources": environmental_schema, "landcover_stratum_source": landcover_schema,
        "po_environmental_features": "verified competition environmental values at minimum-ID representative",
        "cell_degrees": CELL_DEGREES, "representative_coordinates": "fixed cell center",
        "deduplication_key": group_columns + ["speciesId"],
        "environmental_representative": "minimum surveyId after cell/publisher/stratum/species deduplication",
        "landcover_stratum_rule": "floor(first supplied land-cover feature / 5); missing -1; numerical bin, no semantic habitat claim",
        "maximum_nonmissing_geographic_uncertainty_m": 1000,
        "pa_duplicate_exclusion_radius_m": 100,
        "pa_duplicate_exclusion_reference": "all PA train and test coordinates; no species labels read",
        "numeric_id_overlap_policy": "diagnostic only; independent namespaces not assumed identical",
        "weight_formula": "1 / (publishers_in_cell * strata_in_cell_publisher * sqrt(unique_cells_in_1degree_block))",
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
