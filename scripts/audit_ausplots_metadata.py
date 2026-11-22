"""Screen public AusPlots visit metadata for links to previously used SAGE data."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
from rasterio.warp import transform
from sklearn.neighbors import BallTree


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/ausplots_metadata_audit"
OUT = ROOT / "results/ausplots_metadata_audit"
PROTOCOL = ROOT / "docs/ausplots_metadata_audit_protocol.md"
VISIT_COLUMNS = ["site_location_name", "site_location_visit_id",
                 "visit_start_date", "latitude", "longitude"]
COORDINATES = ["longitude", "latitude"]
EARTH_RADIUS_KM = 6371.0088


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_visits(path, headers):
    content_range = re.findall(r"^Content-Range:\s*(\d+)-(\d+)/(\d+)\s*$",
                               headers.read_text(), re.MULTILINE | re.IGNORECASE)
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records or any(
        not isinstance(row, dict) or row.keys() != set(VISIT_COLUMNS) for row in records
    ):
        raise ValueError("Expected the metadata-only site-visit projection")
    if content_range != [("0", str(len(records) - 1), str(len(records)))]:
        raise ValueError("The server count does not confirm a complete site-visit response")
    visits = pd.DataFrame.from_records(records, columns=VISIT_COLUMNS)
    ids = VISIT_COLUMNS[:2]
    if visits[ids].isna().any().any() or visits[ids[0]].eq("").any() or visits.duplicated(ids).any():
        raise ValueError("Missing or duplicate site-visit keys")
    # The question compares field dates. Keep the recorded local date.
    visits["visit_date"] = pd.to_datetime(
        visits.visit_start_date.str.slice(0, 10), format="%Y-%m-%d", errors="coerce",
    )
    return visits


def valid_locations(coordinates):
    return (np.isfinite(coordinates).all(axis=1) & (np.abs(coordinates[:, 0]) <= 180)
            & (np.abs(coordinates[:, 1]) <= 90))


def projected_grid(coordinates):
    x, y = transform("EPSG:4326", "EPSG:6933", coordinates[:, 0], coordinates[:, 1])
    return np.floor(np.column_stack((x, y)) / 1000).astype("int64")


def pair_keys(pairs):
    return (pairs[:, 0].astype("uint64") << np.uint64(32)) | pairs[:, 1].astype("uint32")


def location_bits(coordinates):
    return pair_keys(np.ascontiguousarray(coordinates, dtype="<f4").view("<u4"))


def geographic_distance(queries, references):
    reference_latlon = np.asarray(references[:, ::-1], dtype=np.float64)
    query_latlon = np.asarray(queries[:, ::-1], dtype=np.float64)
    tree = BallTree(np.deg2rad(reference_latlon), metric="haversine")
    distances, _ = tree.query(np.deg2rad(query_latlon))
    return distances[:, 0] * EARTH_RADIUS_KM


def locations(path):
    with h5py.File(path) as handle:
        coords = handle["predictor_data"][:]
    if coords.ndim != 2 or coords.shape[1] != 2 or not valid_locations(coords).all():
        raise ValueError(f"Invalid coordinates in {path.name}")
    return coords


def summarize_stratum(projected):
    grid = projected[["grid_x", "grid_y"]].drop_duplicates().to_numpy()
    return dict(visits=len(projected), sites=projected.site_location_name.nunique(),
                cells_1km=len(grid), blocks_100km=len(np.unique(grid // 100, axis=0)),
                possible_existing_eval_input_rows=int(projected.exact_eval_input_center.sum()),
                possible_existing_train_input_rows=int(projected.exact_train_location_center.sum()))


def audit(visits, metadata, train, evaluation):
    australian = metadata[metadata.Country.eq("Australia")]
    if not len(australian) or not australian.GIVD_ID.eq("AU-AU-002").all():
        raise ValueError("The Australian evaluation source contract changed")
    recorded_dates = pd.to_datetime(australian.Date_of_recording, format="mixed", errors="coerce")
    cutoff = recorded_dates.max().normalize()
    if pd.isna(cutoff):
        raise ValueError("The AEKOS record has no usable dates")
    legacy = australian[["Longitude", "Latitude"]].to_numpy(float)
    valid_legacy = valid_locations(legacy)
    reference = legacy[valid_legacy]
    if not len(reference):
        raise ValueError("No valid AEKOS locations")
    raw = visits[COORDINATES].to_numpy(float)
    valid = valid_locations(raw)
    post_date = visits.visit_date.gt(cutoff).to_numpy()
    late = visits.loc[post_date & valid].copy()
    if late.empty:
        raise ValueError("No valid post-cutoff visits")
    candidate = raw[post_date & valid]
    grid = projected_grid(candidate)
    centers = np.column_stack(transform(
        "EPSG:6933", "EPSG:4326", (grid[:, 0] + .5)*1000, (grid[:, 1] + .5)*1000,
    ))
    key = pair_keys(grid)
    late["grid_x"], late["grid_y"] = grid.T
    late["legacy_same_grid"] = np.isin(key, pair_keys(projected_grid(reference)))
    late["legacy_distance_km"] = geographic_distance(candidate, reference)
    late["train_same_grid"] = np.isin(key, pair_keys(projected_grid(train)))
    late["train_distance_km"] = geographic_distance(candidate, train)
    center_bits = location_bits(centers)
    late["exact_eval_input_center"] = np.isin(center_bits, location_bits(evaluation))
    late["exact_train_location_center"] = np.isin(center_bits, location_bits(train))
    earlier_sites = set(visits.loc[visits.visit_date.le(cutoff), "site_location_name"])
    undated_sites = set(visits.loc[visits.visit_date.isna(), "site_location_name"])
    late["site_has_earlier_visit"] = late.site_location_name.isin(earlier_sites)
    late["site_has_undated_visit"] = late.site_location_name.isin(undated_sites)

    legacy_buffer = late.legacy_distance_km.gt(1) & ~late.legacy_same_grid
    off_training = legacy_buffer & ~late.train_same_grid
    beyond_one = off_training & late.train_distance_km.gt(1)
    new_site = ~(late.site_has_earlier_visit | late.site_has_undated_visit)
    strata = {
        "post_cutoff_valid": late,
        "outside_legacy_grid_and_1km_buffer": late.loc[legacy_buffer],
        "also_outside_training_grid": late.loc[off_training],
        "also_more_than_1km_from_training": late.loc[beyond_one],
        "also_more_than_10km_from_training": late.loc[beyond_one & late.train_distance_km.gt(10)],
        "post_cutoff_site_history_only": late.loc[new_site],
        "outside_legacy_buffer_and_training_grid_with_post_cutoff_site_history": late.loc[off_training & new_site],
    }
    result = {
        "source": "ausplotsR public site API, five metadata fields only",
        "species_outcomes_read": False,
        "server_count_matches_payload": True,
        "metadata_visits": len(visits),
        "metadata_sites": visits.site_location_name.nunique(),
        "unparseable_visit_dates": int(visits.visit_date.isna().sum()),
        "invalid_visit_locations": int((~valid).sum()),
        "late_visits_with_invalid_coordinates": int((post_date & ~valid).sum()),
        "aekos_evaluation_plot_rows": len(australian),
        "aekos_evaluation_cells": australian.cell_row.nunique(),
        "aekos_unparseable_dates": int(recorded_dates.isna().sum()),
        "aekos_invalid_plot_locations": int((~valid_legacy).sum()),
        "aekos_last_parseable_date": str(cutoff.date()),
        "published_train_location_rows": len(train),
        "published_eval_input_location_rows": len(evaluation),
        "candidate_diagnostics": {name: summarize_stratum(frame) for name, frame in strata.items()},
        "post_cutoff_visits_in_legacy_grid": int(late.legacy_same_grid.sum()),
        "post_cutoff_visits_in_training_grid": int(late.train_same_grid.sum()),
        "post_cutoff_visits_at_sites_with_earlier_visits": int(late.site_has_earlier_visit.sum()),
        "median_distance_to_training_km": float(late.train_distance_km.median()),
        "final_cohort_admitted": False,
    }
    return result, late


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUT / "spatial_feasibility.json", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = CACHE / "site_visits.json"
    headers = CACHE / "site_visits.headers"
    legacy = ROOT / "data/independent_data_audit/sage_eval_plot_metadata.csv"
    train = ROOT / "data/neon_bridge/train_location_1km.h5"
    evaluation = ROOT / "data/external_replication_e1/predictors/eval_location.h5"
    visits = read_visits(path, headers)
    metadata = pd.read_csv(legacy, usecols=["Country", "GIVD_ID", "Date_of_recording", "cell_row",
                                           "Longitude", "Latitude"])
    summary, rows = audit(visits, metadata, locations(train), locations(evaluation))
    summary["sha256"] = {str(p.relative_to(ROOT)): sha(p)
                         for p in [path, legacy, train, evaluation, PROTOCOL, Path(__file__).resolve()]}
    date_header = re.findall(r"^Date:\s*([^\n]+)", headers.read_text(), re.MULTILINE | re.IGNORECASE)
    summary["site_response_server_date"] = date_header[0].strip() if date_header else None
    rows.to_csv(CACHE / "post_cutoff_visit_geometry.csv", index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
