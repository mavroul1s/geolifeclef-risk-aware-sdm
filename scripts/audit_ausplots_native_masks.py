"""Check native-mask coverage for the SAGE-disjoint candidate AusPlots cells."""
import json

import h5py
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform
import shapefile
from shapely.geometry import shape

from audit_ausplots_metadata import CACHE, OUT, ROOT, locations, sha
from geolifeclef.native_region_bridge import assign_regions, constant_region_masks


def main():
    spatial_result = OUT / "spatial_feasibility.json"
    spatial = json.loads(spatial_result.read_text())
    visits_path = CACHE / "post_cutoff_visit_geometry.csv"
    visits = pd.read_csv(visits_path)
    eligible = visits[visits.legacy_distance_km.gt(1)
                      & ~visits.legacy_same_grid & ~visits.train_same_grid]
    cells = eligible[["grid_x", "grid_y"]].drop_duplicates().sort_values(["grid_x", "grid_y"])
    reference_count = spatial["candidate_diagnostics"]["also_outside_training_grid"]
    if (len(eligible), len(cells)) != (reference_count["visits"], reference_count["cells_1km"]):
        raise ValueError("Spatial audit cohort changed")
    coords = np.column_stack(transform("EPSG:6933", "EPSG:4326",
                                       (cells.grid_x.to_numpy()+.5)*1000, (cells.grid_y.to_numpy()+.5)*1000))
    coast_shape = ROOT / "data/neon_input_followup/tdwg_level3/level3.shp"
    if CRS.from_wkt(coast_shape.with_suffix(".prj").read_text()).to_epsg() != 4326:
        raise ValueError("Unexpected region CRS")
    with shapefile.Reader(str(coast_shape)) as reader:
        records = list(reader.iterShapeRecords(fields=["LEVEL3_COD"]))
    geometries = [shape(record.shape.__geo_interface__) for record in records]
    codes = [record.record.as_dict()["LEVEL3_COD"] for record in records]

    published = ROOT / "data/external_replication_e1/predictors/eval_location.h5"
    mask_path = ROOT / "data/external_replication_e1/range_masks_1km/eval_range_mask.h5"
    reference_regions, _ = assign_regions(geometries, codes, locations(published))
    with h5py.File(mask_path) as handle:
        masks, inconsistent, counts = constant_region_masks(reference_regions,
            handle["cumulative_lengths"][:], handle["species_index"][:], 5771)
    candidate_regions, fallback = assign_regions(geometries, codes, coords)
    available = np.isin(candidate_regions, list(masks))
    strict = available & ~fallback
    classes = [len(masks[region]) for region in candidate_regions[strict]]
    receipt = {
        "spatial_candidate_visits": len(eligible),
        "candidate_cells": len(cells),
        "strict_same_region_mask_cells": int(strict.sum()),
        "polygon_fallback_cells": int(fallback.sum()),
        "candidate_cells_in_unrepresented_or_inconsistent_regions": int((~available).sum()),
        "candidate_regions": {region: int((candidate_regions == region).sum())
                              for region in np.unique(candidate_regions)},
        "published_eval_masks_per_candidate_region": {region: counts.get(region, 0)
                                                      for region in np.unique(candidate_regions)},
        "inconsistent_regions": inconsistent,
        "strict_native_species_count_range": [min(classes), max(classes)] if classes else None,
        "covariate_feature_vectors_built": 0,
        "species_outcomes_read": False,
        "final_cohort_admitted": False,
    }
    paths = [spatial_result, visits_path, published, mask_path,
             *(coast_shape.with_suffix(ext) for ext in (".shp", ".shx", ".dbf", ".prj"))]
    receipt["sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    with (OUT / "native_mask_feasibility.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
