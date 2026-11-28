"""Audit AusPlots visit completeness and voucher linkage without downloading taxa."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from audit_ausplots_metadata import ROOT, sha
from check_ausplots_predictors import DATA, GEOMETRY, OUTPUT, PROTOCOL, candidate_visits


ENDPOINT = "http://swarmapi.ausplots.aekos.org.au:80"
SITE, VISIT, BARCODE = "site_location_name", "site_location_visit_id", "veg_barcode"
PLOT_SIZE = "plot_is_100m_by_100m"
UNCERTAIN, DATE = "is_uncertain_determination", "visit_start_date"
TRANSECT, POINT = "transect", "point_number"
PROJECTIONS = {
    "site": (SITE, VISIT, PLOT_SIZE),
    "veg_voucher": (SITE, VISIT, BARCODE, UNCERTAIN, DATE),
    "veg_pi": (SITE, VISIT, TRANSECT, POINT, BARCODE),
}
PAGE = 10000
MAX_RESPONSE_BYTES = 4 * 1024**2


def fetch_projection(table, visit_ids):
    columns = PROJECTIONS[table]
    rows, response_bytes, dates = [], 0, []
    total = None
    for _ in range(4):
        offset = len(rows)
        query = urlencode({VISIT: f"in.({','.join(map(str, visit_ids))})", "select": ",".join(columns),
                           "order": ",".join(columns), "limit": PAGE, "offset": offset})
        request = Request(f"{ENDPOINT}/{table}?{query}",
                          headers={"Accept": "application/json", "Prefer": "count=exact"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=25) as response:
                    if response.headers.get_content_type() != "application/json":
                        raise ValueError("Expected a JSON projection")
                    body = response.read(MAX_RESPONSE_BYTES + 1)
                    range_header = response.headers.get("Content-Range", "")
                    dates.append(response.headers.get("Date"))
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Response byte limit exceeded")
        batch = json.loads(body)
        if not isinstance(batch, list) or any(not isinstance(row, dict) or row.keys() != set(columns)
                                              or row[VISIT] not in visit_ids for row in batch):
            raise ValueError("Response does not match the allowlisted projection and visit keys")
        count = re.fullmatch(r"(?:\*|(\d+)-(\d+))/(\d+)", range_header)
        if count is None or (total is not None and int(count[3]) != total):
            raise ValueError("Missing or changing exact server count")
        total = int(count[3])
        if batch and (count[1], count[2]) != (str(offset), str(offset + len(batch) - 1)):
            raise ValueError("Content-Range does not describe the payload")
        if not batch and (range_header != "*/0" or total != 0):
            raise ValueError("Incomplete pagination")
        response_bytes += len(body)
        rows.extend(batch)
        if len(rows) == total:
            return rows, response_bytes, dates
        if len(rows) > total:
            break
    raise ValueError("Projection did not reconcile with the exact server count within the page limit")


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def point_effort(points):
    transects = defaultdict(set)
    invalid = 0
    for point in points:
        if not nonblank(point[TRANSECT]) or type(point[POINT]) is not int:
            invalid += 1
            continue
        transects[point[TRANSECT]].add(point[POINT])
    sizes = tuple(sorted(len(numbers) for numbers in transects.values()))
    full_sequence = all(numbers in (set(range(101)), set(range(1, 102))) for numbers in transects.values())
    return sizes, invalid, full_sequence


def summarize_visit(visit, site, vouchers, points):
    voucher_keys = Counter(row[BARCODE] for row in vouchers if nonblank(row[BARCODE]))
    point_keys = Counter(row[BARCODE] for row in points if nonblank(row[BARCODE]))
    has_date = lambda row: isinstance(row[DATE], str) and row[DATE][:10] == visit.visit_date
    sizes, invalid_points, full_sequence = point_effort(points)
    return {
        SITE: visit.site_location_name, VISIT: visit.site_location_visit_id,
        "grid_x": visit.grid_x, "grid_y": visit.grid_y, "visit_date": visit.visit_date,
        "train_distance_km": visit.train_distance_km,
        "site_has_earlier_visit": visit.site_has_earlier_visit,
        "plot_is_100m_by_100m": None if site is None else site[PLOT_SIZE],
        "point_records": len(points), "distinct_intercept_points": sum(sizes), "transects": len(sizes),
        "enhanced_effort": sizes == (101,) * 10 and full_sequence and invalid_points == 0,
        "standard_effort": sizes == (101,) * 4 and full_sequence and invalid_points == 0,
        "invalid_point_keys": invalid_points,
        "voucher_rows": len(vouchers), "blank_voucher_barcodes": len(vouchers) - sum(voucher_keys.values()),
        "duplicate_voucher_keys": sum(count > 1 for count in voucher_keys.values()),
        "voucher_date_disagreements": sum(not has_date(row) for row in vouchers),
        "voucher_uncertain_true": sum(row[UNCERTAIN] is True for row in vouchers),
        "voucher_uncertainty_missing": sum(row[UNCERTAIN] is None for row in vouchers),
        "voucher_uncertain_false": sum(row[UNCERTAIN] is False for row in vouchers),
        "point_records_with_barcode": sum(point_keys.values()),
        "point_barcodes_distinct": len(point_keys),
        "point_barcodes_unlinked": sum(key not in voucher_keys for key in point_keys),
        "point_barcodes_ambiguous": sum(voucher_keys[key] > 1 for key in point_keys),
        "point_records_uniquely_linked": sum(count for key, count in point_keys.items() if voucher_keys[key] == 1),
    }


def scope(frame):
    cells = frame[["grid_x", "grid_y"]].to_numpy(dtype=np.int64)
    return {"visits": len(frame), "sites": frame[SITE].nunique(), "cells_1km": len(np.unique(cells, axis=0)),
            "blocks_100km": len(np.unique(cells // 100, axis=0))}


def audit(visits, payloads):
    keys = set(zip(visits[SITE], visits[VISIT]))
    grouped = {table: defaultdict(list) for table in PROJECTIONS}
    for table, rows in payloads.items():
        for row in rows:
            key = (row[SITE], row[VISIT])
            if key not in keys or row.keys() != set(PROJECTIONS[table]):
                raise ValueError("Unexpected site or projected field")
            if table == "veg_voucher" and row[UNCERTAIN] is not None and type(row[UNCERTAIN]) is not bool:
                raise ValueError("Expected a nullable boolean voucher uncertainty flag")
            grouped[table][key].append(row)
    rows = []
    for visit in visits.itertuples(index=False):
        key = (visit.site_location_name, visit.site_location_visit_id)
        site = grouped["site"][key]
        if len(site) != 1 or (site[0][PLOT_SIZE] is not None and type(site[0][PLOT_SIZE]) is not bool):
            raise ValueError("Expected one plot-size flag per candidate visit")
        rows.append(summarize_visit(visit, site[0], grouped["veg_voucher"][key], grouped["veg_pi"][key]))
    frame = pd.DataFrame(rows)
    has_vouchers = frame.voucher_rows.gt(0)
    valid_vouchers = (has_vouchers & frame.blank_voucher_barcodes.eq(0)
                      & frame.duplicate_voucher_keys.eq(0) & frame.voucher_date_disagreements.eq(0))
    linked = frame.point_barcodes_unlinked.eq(0) & frame.point_barcodes_ambiguous.eq(0)
    strict = frame.plot_is_100m_by_100m.eq(True) & frame.enhanced_effort & valid_vouchers & linked
    potential = strict & frame.train_distance_km.gt(1)
    strata = {"all_candidates": frame, "positive_one_hectare_flag": frame[frame.plot_is_100m_by_100m.eq(True)],
              "enhanced_point_effort": frame[frame.enhanced_effort],
              "standard_point_effort": frame[frame.standard_effort], "any_voucher": frame[has_vouchers],
              "valid_voucher_date_and_unique_keys": frame[valid_vouchers],
              "strict_record_provenance": frame[strict], "strict_records_gt_1km": frame[potential],
              "strict_records_gt_10km": frame[strict & frame.train_distance_km.gt(10)],
              "strict_records_gt_1km_earliest_per_site": frame[potential].sort_values(
                  ["visit_date", VISIT]).drop_duplicates(SITE)}
    metric_columns = list(frame.columns[frame.columns.str.startswith(("voucher_", "point_"))])
    metric_columns += ["invalid_point_keys", "blank_voucher_barcodes", "duplicate_voucher_keys"]
    return frame, {"strata": {name: scope(value) for name, value in strata.items()},
                   "record_totals": {column: int(frame[column].sum()) for column in metric_columns},
                   "plot_size_flag_missing_visits": int(frame.plot_is_100m_by_100m.isna().sum()),
                   "visits_without_point_rows": int(frame.point_records.eq(0).sum()),
                   "point_effort_counts": {str(n): int(count) for n, count in
                                           frame.distinct_intercept_points.value_counts().sort_index().items()},
                   "species_outcomes_requested": False, "inventory_absence_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true", help="Recompute a summary from saved projections")
    args = parser.parse_args()
    data = DATA / "voucher_metadata"
    output = OUTPUT / "voucher_metadata.json"
    if output.exists() and not args.cached:
        raise FileExistsError(output)
    visits = candidate_visits()
    visit_ids = sorted(visits[VISIT].tolist())
    if len(set(visit_ids)) != len(visit_ids):
        raise ValueError("Visit IDs are not unique across sites")
    payloads = {}
    retrieval_path = data / "retrieval.json"
    if args.cached:
        retrieval = json.loads(retrieval_path.read_text())
        for table in PROJECTIONS:
            path = data / f"{table}.json"
            if retrieval[table]["sha256"] != sha(path):
                raise ValueError("Cached projection hash differs")
            payloads[table] = json.loads(path.read_text())
    else:
        data.mkdir(parents=True, exist_ok=True)
        retrieval = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            for table in PROJECTIONS:
                size = 4 if table == "veg_pi" else 40
                batches = [visit_ids[start:start + size] for start in range(0, len(visit_ids), size)]
                groups = list(pool.map(lambda ids: fetch_projection(table, ids), batches))
                rows = [row for group, _, _ in groups for row in group]
                path = data / f"{table}.json"
                path.write_text(json.dumps(rows, separators=(",", ":")))
                retrieval[table] = {"rows": len(rows), "bytes": sum(size for _, size, _ in groups),
                                    "response_dates": sorted({date for _, _, dates in groups for date in dates if date}),
                                    "batches": len(batches), "sha256": sha(path),
                                    "columns": list(PROJECTIONS[table])}
                payloads[table] = rows
                print(f"{table}: {len(rows)} projected rows", flush=True)
        retrieval_path.write_text(json.dumps(retrieval, indent=2) + "\n")
    frame, result = audit(visits, payloads)
    per_visit = data / "per_visit.csv"
    frame.to_csv(per_visit, index=False)
    if args.cached:
        expected = json.loads(output.read_text())
        if (any(expected[key] != value for key, value in result.items())
                or expected["per_visit_sha256"] != sha(per_visit)):
            raise ValueError("Voucher audit does not reproduce")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.update({"completed_at_utc": datetime.now(timezone.utc).isoformat(), "retrieval": retrieval,
                       "per_visit_sha256": sha(per_visit), "endpoint": ENDPOINT,
                       "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                         [GEOMETRY, PROTOCOL, Path(__file__)]}})
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["strata"], indent=2), flush=True)


if __name__ == "__main__":
    main()
