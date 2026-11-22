"""Validate the saved outcome-blinded AusPlots screen using independent checks."""
import json
import re

import h5py
import numpy as np
import pandas as pd
from rasterio.warp import transform

from audit_ausplots_metadata import CACHE, OUT, ROOT, sha


def closest_km(point, reference):
    lon, lat = np.deg2rad(point)
    ref_lon, ref_lat = np.deg2rad(reference).T
    a = np.sin((ref_lat - lat) / 2)**2 + np.cos(lat)*np.cos(ref_lat)*np.sin((ref_lon-lon)/2)**2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(np.min(a))))


def main():
    spatial = json.loads((OUT / "spatial_feasibility.json").read_text())
    native = json.loads((OUT / "native_mask_feasibility.json").read_text())
    for report in [spatial, native]:
        assert not report["species_outcomes_read"] and not report["final_cohort_admitted"]
        for path, digest in report["sha256"].items():
            assert sha(ROOT / path) == digest

    headers = (CACHE / "site_visits.headers").read_text()
    records = json.loads((CACHE / "site_visits.json").read_text())
    server_range = re.search(r"^Content-Range: 0-(\d+)/(\d+)$", headers, re.I | re.M)
    assert server_range and int(server_range[1])+1 == int(server_range[2]) == len(records)
    source = pd.DataFrame(records)
    expected_visits = source[source.visit_start_date.str[:10].gt(spatial["aekos_last_parseable_date"])]
    visits = pd.read_csv(CACHE / "post_cutoff_visit_geometry.csv")
    keys = ["site_location_name", "site_location_visit_id"]
    pd.testing.assert_frame_equal(visits[keys].reset_index(drop=True), expected_visits[keys].reset_index(drop=True))

    path = ROOT / "data/independent_data_audit/sage_eval_plot_metadata.csv"
    legacy = pd.read_csv(path, usecols=["Longitude", "Latitude", "Country"])
    legacy = legacy.loc[legacy.Country.eq("Australia"), ["Longitude", "Latitude"]].to_numpy()
    with h5py.File(ROOT / "data/neon_bridge/train_location_1km.h5") as handle:
        training = handle["predictor_data"][:].astype(np.float64)

    sample = visits.iloc[np.linspace(0, len(visits)-1, 8, dtype=int)]
    distances = [
        (closest_km((row.longitude, row.latitude), legacy),
         closest_km((row.longitude, row.latitude), training))
        for row in sample.itertuples()
    ]
    np.testing.assert_allclose(distances, sample[["legacy_distance_km", "train_distance_km"]], atol=1e-8, rtol=0)

    # Rotate long/lat into the published grid, then test membership via joins.
    tx, ty = transform("EPSG:4326", "EPSG:6933", training[:, 0], training[:, 1])
    grid = np.floor(np.column_stack((tx, ty)) / 1000).astype(np.int64)
    candidate_keys = pd.MultiIndex.from_frame(visits[["grid_x", "grid_y"]])
    train_keys = pd.MultiIndex.from_arrays(grid.T)
    np.testing.assert_array_equal(candidate_keys.isin(train_keys), visits.train_same_grid)
    clean = visits.loc[visits.legacy_distance_km.gt(1) & ~visits.legacy_same_grid & ~visits.train_same_grid]
    assert len(clean) == native["spatial_candidate_visits"]
    assert len(clean[["grid_x", "grid_y"]].drop_duplicates()) == native["candidate_cells"]
    assert native["strict_same_region_mask_cells"] + native["polygon_fallback_cells"] + native[
        "candidate_cells_in_unrepresented_or_inconsistent_regions"
    ] == native["candidate_cells"]
    assert native["covariate_feature_vectors_built"] == 0

    result = {
        "source_count_checked": len(records),
        "post_cutoff_visit_keys_checked": len(visits),
        "great_circle_distances_checked": 2*len(sample),
        "training_grid_memberships_checked": len(visits),
        "mask_reference_rows": spatial["published_eval_input_location_rows"],
        "source_and_input_hashes_verified": True,
        "independent_outcome_opened": False,
    }
    output = OUT / "validation.json"
    if output.exists():
        assert json.loads(output.read_text()) == result
    else:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
