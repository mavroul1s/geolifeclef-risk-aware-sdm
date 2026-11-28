"""Screen the second SAGE model's original-resolution presence-only locations."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
from rasterio.warp import transform

from audit_ausplots_metadata import (COORDINATES, EARTH_RADIUS_KM, ROOT, geographic_distance,
                                    pair_keys, projected_grid, sha, valid_locations)
from audit_ausplots_voucher_metadata import scope
from check_ausplots_predictors import DATA, GEOMETRY, OUTPUT, candidate_visits
from retrieve_neon_bridge import range_header
from retrieve_sage_e1 import fetch_member


RELEASE = ROOT / "data/external_replication_e1/zenodo_21297133.json"
ARCHIVE = "sage_po_records_nonaggregated.tar"
MEMBER = "predictors/train_location.h5"
ROWS = 30_138_089
CHUNK_ROWS = 1_000_000


def retrieve_coordinates(record, path):
    item = next(f for f in record["files"] if f["key"] == ARCHIVE)
    url = item["links"]["self"]
    offset = 0
    for header_count in range(1, 65):
        if offset + 512 > item["size"]:
            break
        header = range_header(url, offset)
        member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        if member.name == MEMBER:
            if not member.isfile() or not 0 < member.size <= 256 * 1024**2:
                raise ValueError("Unexpected location-only member size")
            location = {"name": member.name, "size": member.size, "data_offset": offset + 512}
            receipt = fetch_member(record, ARCHIVE, location, path)
            return {**receipt, "header_sha256": hashlib.sha256(header).hexdigest(),
                    "header_bytes_checked": 512 * header_count, "data_offset": offset + 512}
        offset += 512 + ((member.size + 511) // 512) * 512
    raise ValueError("Location-only TAR member not found within the header limit")


def buffered_coords(path, low, high):
    near = []
    with h5py.File(path) as file:
        dataset = file["predictor_data"]
        if dataset.shape != (ROWS, 2):
            raise ValueError("Unexpected nonaggregated coordinate archive shape")
        for start in range(0, ROWS, CHUNK_ROWS):
            batch = dataset[start:start + CHUNK_ROWS]
            if not valid_locations(batch).all():
                raise ValueError("Invalid nonaggregated training location")
            near.append(batch[(batch >= low).all(axis=1) & (batch <= high).all(axis=1)])
    return np.concatenate(near)


def audit(path, visits):
    queries = visits[COORDINATES].to_numpy(dtype=np.float64)
    grid = visits[["grid_x", "grid_y"]].to_numpy(dtype=np.int64)
    if not np.array_equal(projected_grid(queries), grid):
        raise ValueError("Candidate grid changed")

    # Every point outside the lon/lat box has at least a one-degree gap on one axis.
    low, high = queries.min(axis=0) - 1, queries.max(axis=0) + 1
    if low[0] < -179 or high[0] > 179:
        raise ValueError("Buffer crosses the dateline")
    for offset_x, offset_y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        corners = np.column_stack(transform("EPSG:6933", "EPSG:4326",
                                            *(1000 * (grid + [offset_x, offset_y]).T)))
        if not ((corners > low) & (corners < high)).all():
            raise ValueError("Candidate cell extends outside the coordinate buffer")
    training = buffered_coords(path, low, high)
    distances = geographic_distance(queries, training)
    max_abs_lat = np.deg2rad(max(abs(low[1]), abs(high[1])))
    excluded_distance_bound = 2 * EARTH_RADIUS_KM * np.arcsin(np.cos(max_abs_lat) * np.sin(np.deg2rad(.5)))
    if distances.max() >= excluded_distance_bound:
        raise ValueError("The buffered reference cannot establish the exact nearest point")

    training_grid = projected_grid(training)
    same_grid = np.isin(pair_keys(grid), pair_keys(training_grid))
    # Check against direct distances on a fixed spread of candidate visits.
    checked = np.linspace(0, len(queries) - 1, 16, dtype=int)
    lon, lat = np.deg2rad(training.astype(np.float64)).T
    independent = []
    for query in queries[checked]:
        q_lon, q_lat = np.deg2rad(query)
        h = np.sin((lat - q_lat) / 2)**2 + np.cos(lat) * np.cos(q_lat) * np.sin((lon - q_lon) / 2)**2
        independent.append(float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h.min()))))
    max_error = float(np.max(np.abs(distances[checked] - independent)))
    if max_error > 1e-8:
        raise ValueError("Independent nearest-distance check disagrees")

    result = visits.copy()
    result["nonaggregated_same_grid"] = same_grid
    result["nonaggregated_distance_km"] = distances
    both_off_grid = result[~result.nonaggregated_same_grid]
    return result, {
        "original_training_locations_checked": ROWS,
        "locations_inside_search_buffer": len(training),
        "omitted_location_distance_lower_bound_km": float(excluded_distance_bound),
        "max_candidate_nearest_distance_km": float(distances.max()),
        "median_candidate_nearest_distance_km": float(np.median(distances)),
        "visits_in_nonaggregated_training_cell": int(same_grid.sum()),
        "outside_both_presence_training_grids": scope(both_off_grid),
        "outside_1km_of_both_presence_training_sets": scope(both_off_grid[
            both_off_grid.train_distance_km.gt(1) & both_off_grid.nonaggregated_distance_km.gt(1)]),
        "outside_10km_of_both_presence_training_sets": scope(both_off_grid[
            both_off_grid.train_distance_km.gt(10) & both_off_grid.nonaggregated_distance_km.gt(10)]),
        "independent_distance_checks": len(checked), "independent_max_difference_km": max_error,
    }


def main():
    report = OUTPUT / "nonaggregated_training_overlap.json"
    if report.exists():
        raise FileExistsError(report)
    path = DATA / "training_overlap/train_location.h5"
    receipt = retrieve_coordinates(json.loads(RELEASE.read_text()), path)
    result, totals = audit(path, candidate_visits())
    visits = path.parent / "screened_visits.csv"
    result.to_csv(visits, index=False)
    totals.update({"completed_at_utc": datetime.now(timezone.utc).isoformat(), "retrieval": receipt,
                   "species_outcomes_read": False, "screened_visits_sha256": sha(visits),
                   "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [RELEASE, GEOMETRY, Path(__file__)]}})
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as stream:
        stream.write(json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
