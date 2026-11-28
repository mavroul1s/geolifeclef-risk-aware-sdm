"""Independently reconcile projections, raster receipts and the combined AusPlots screen."""

import argparse
import hashlib
import json
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
from rasterio.warp import transform
import shapefile
from shapely.geometry import shape

from audit_ausplots_metadata import ROOT, locations, sha
from audit_ausplots_voucher_metadata import BARCODE, POINT, PROJECTIONS, SITE, TRANSECT, VISIT, scope
from check_ausplots_predictors import DATA, OUTPUT, RELEASED, candidate_cells, published_layer_names
from geolifeclef.native_region_bridge import assign_regions, constant_region_masks


KEYS = [SITE, VISIT]
GRIDS = ["grid_x", "grid_y"]
MASK_PATH = ROOT / "data/external_replication_e1/range_masks_1km/eval_range_mask.h5"
POLYGONS = ROOT / "data/neon_input_followup/tdwg_level3/level3.shp"


def read_receipt(path):
    result = json.loads(path.read_text())
    for name, digest in result.get("source_sha256", {}).items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Hash mismatch for {name}")
    return result


def independent_record_screen():
    receipt = read_receipt(OUTPUT / "voucher_metadata.json")
    paths = DATA / "voucher_metadata"
    private = paths / "per_visit.csv"
    if sha(private) != receipt["per_visit_sha256"]:
        raise ValueError("Visit-level voucher receipt changed")
    data = pd.read_csv(private)
    values = {}
    for table, columns in PROJECTIONS.items():
        path = paths / f"{table}.json"
        archived = receipt["retrieval"][table]
        frame = pd.read_json(path, convert_dates=False)
        if (sha(path) != archived["sha256"] or len(frame) != archived["rows"] or set(frame) != set(columns)
                or not pd.MultiIndex.from_frame(frame[KEYS]).isin(pd.MultiIndex.from_frame(data[KEYS])).all()):
            raise ValueError(f"{table} projection differs")
        values[table] = frame
    points, vouchers = values["veg_pi"], values["veg_voucher"]
    if vouchers.duplicated(KEYS + [BARCODE]).any():
        raise ValueError("A projected barcode resolves to multiple vouchers on the same visit")
    if points[[TRANSECT, POINT]].isna().any().any():
        raise ValueError("Missing point key in independent reconstruction")
    transects = points.groupby(KEYS + [TRANSECT])[POINT].agg(["nunique", "min", "max"])
    transects["complete"] = transects["nunique"].eq(101) & transects["min"].isin([0, 1]) & transects["max"].sub(transects["min"]).eq(100)
    effort = transects.groupby(KEYS).complete.agg(["size", "all"])
    complete = effort.index[effort["size"].eq(10) & effort["all"]]
    keys = pd.MultiIndex.from_frame(data[KEYS])
    np.testing.assert_array_equal(keys.isin(complete), data.enhanced_effort)
    with_barcode = points[points[BARCODE].fillna("").str.strip().ne("")]
    join = with_barcode[KEYS + [BARCODE]].merge(vouchers[KEYS + [BARCODE]], on=KEYS + [BARCODE],
                                               validate="many_to_one", how="left", indicator=True)
    unmatched = join[join._merge.eq("left_only")]
    counts = unmatched.drop_duplicates(KEYS + [BARCODE]).groupby(KEYS).size().reindex(keys, fill_value=0)
    np.testing.assert_array_equal(counts, data.point_barcodes_unlinked)
    if len(join) - len(unmatched) != receipt["record_totals"]["point_records_uniquely_linked"]:
        raise ValueError("Same-visit barcode linkage count differs")
    voucher_sizes = vouchers.groupby(KEYS).size().reindex(keys, fill_value=0)
    np.testing.assert_array_equal(voucher_sizes, data.voucher_rows)
    site_sizes = values["site"].set_index(KEYS)["plot_is_100m_by_100m"].reindex(keys)
    np.testing.assert_array_equal(site_sizes, data.plot_is_100m_by_100m)
    strict = (data.plot_is_100m_by_100m.eq(True) & data.enhanced_effort & data.voucher_rows.gt(0)
              & data.blank_voucher_barcodes.eq(0) & data.duplicate_voucher_keys.eq(0)
              & data.voucher_date_disagreements.eq(0) & data.point_barcodes_unlinked.eq(0)
              & data.point_barcodes_ambiguous.eq(0))
    if scope(data[strict]) != receipt["strata"]["strict_record_provenance"]:
        raise ValueError("Strict metadata counts differ")
    return data.assign(strict_record_provenance=strict)


def native_cells(cells):
    with shapefile.Reader(str(POLYGONS)) as reader:
        records = list(reader.iterShapeRecords(fields=["LEVEL3_COD"]))
    polygons = [shape(record.shape.__geo_interface__) for record in records]
    names = [record.record.as_dict()["LEVEL3_COD"] for record in records]
    regions, _ = assign_regions(polygons, names, locations(RELEASED))
    with h5py.File(MASK_PATH) as handle:
        masks, _, _ = constant_region_masks(regions, handle["cumulative_lengths"][:], handle["species_index"][:], 5771)
    geo = np.column_stack(transform("EPSG:6933", "EPSG:4326", *(1000 * (cells.T + .5))))
    region, fallback = assign_regions(polygons, names, geo)
    float_region, float_fallback = assign_regions(polygons, names, geo.astype(np.float32))
    strict = ~fallback & np.isin(region, list(masks))
    if int(strict.sum()) != 113 or not np.array_equal(region, float_region) or not np.array_equal(fallback, float_fallback):
        raise ValueError("Native mask changes relative to the metadata audit or float32 center")
    return strict


def predictor_check(run):
    path = DATA / run / "candidate_features.npz"
    report = read_receipt(OUTPUT / run / "predictors.json")
    if sha(path) != report["candidate_features_sha256"]:
        raise ValueError("Candidate feature archive changed")
    cells, _ = candidate_cells()
    names = [name for group in published_layer_names().values() for name in group]
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["cell_keys"], cells)
        np.testing.assert_array_equal(archive["feature_names"], names)
        features, checked = archive["values"], archive["sampled_columns"]
    if features.shape != (119, 52) or features.dtype != np.float32 or np.isinf(features).any():
        raise ValueError("Unexpected candidate features shape or dtype")
    for pos, result in enumerate(report["layers"]):
        if result["file"] != names[pos] or checked[pos] != (result["status"] == "verified_and_sampled"):
            raise ValueError("Predictor receipt and column order disagree")
        column = features[:, pos].copy()
        if checked[pos]:
            if (result["reference_matches"] != 32 or result["reference_nan_pattern_disagreements"] != 0
                    or result["reference_max_finite_absolute_error"] != 0
                    or result["candidate_value_sha256"] != hashlib.sha256(column.tobytes()).hexdigest()):
                raise ValueError("Claimed sample failed its feature check")
        elif not np.isnan(column).all():
            raise ValueError("Unverified column contains candidate values")
    return cells, checked, int(np.isfinite(features).all(axis=1).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-run", default="predictors_reference_v1")
    args = parser.parse_args()
    output = OUTPUT / "combined_admission.json"
    if output.exists():
        raise FileExistsError(output)
    visits = independent_record_screen()
    cells, predictors, all_finite = predictor_check(args.predictor_run)
    model = read_receipt(OUTPUT / "second_model_reference.json")
    overlap = read_receipt(OUTPUT / "nonaggregated_training_overlap.json")
    screen = DATA / "training_overlap/screened_visits.csv"
    if sha(screen) != overlap["screened_visits_sha256"] or not model["reference_passed"]:
        raise ValueError("Model or training-overlap fixture changed")
    coverage = pd.DataFrame(cells, columns=GRIDS).assign(strict_region=native_cells(cells))
    train = pd.read_csv(screen, usecols=KEYS + ["nonaggregated_same_grid", "nonaggregated_distance_km"])
    merged = visits.merge(train, on=KEYS, validate="one_to_one").merge(coverage, on=GRIDS, validate="many_to_one")
    if len(merged) != len(visits) or merged.strict_region.isna().any():
        raise ValueError("Combined join lost visits")
    min_distance = merged[["train_distance_km", "nonaggregated_distance_km"]].min(axis=1)
    strict = merged[merged.strict_record_provenance & merged.strict_region & ~merged.nonaggregated_same_grid]
    primary = strict[min_distance.loc[strict.index].gt(1)]
    earliest = primary.sort_values(["visit_date", VISIT]).drop_duplicates(SITE)
    beyond_ten = strict[min_distance.loc[strict.index].gt(10)]
    sources = [Path(__file__), MASK_PATH, RELEASED, screen, OUTPUT / "voucher_metadata.json",
               OUTPUT / "second_model_reference.json", OUTPUT / "nonaggregated_training_overlap.json",
               OUTPUT / args.predictor_run / "predictors.json",
               *(POLYGONS.with_suffix(extension) for extension in (".shp", ".shx", ".dbf", ".prj"))]
    summary = {"all_candidate_visits_checked": len(merged), "strict_same_region_and_records": scope(strict),
               "primary_metadata_screen_gt_1km_both_models": scope(primary),
               "earliest_per_site_after_primary_metadata_screen": scope(earliest),
               "strict_gt_10km_both_models": scope(beyond_ten),
               "primary_visits_without_legacy_site_history": scope(primary[~primary.site_has_earlier_visit]),
               "primary_voucher_rows": int(primary.voucher_rows.sum()),
               "primary_vouchers_uncertain": int(primary.voucher_uncertain_true.sum()),
               "primary_vouchers_flag_missing": int(primary.voucher_uncertainty_missing.sum()),
               "predictor_run": args.predictor_run, "verified_sampled_features": int(predictors.sum()),
               "complete_52_finite_vectors": all_finite, "second_model_cpu_reference_passed": True,
               "final_cohort_admitted": False, "species_outcomes_read": False,
               "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources}}
    with output.open("x") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
