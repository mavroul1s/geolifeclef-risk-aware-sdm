"""Reproduce SAGE's environmental inputs on fixed released rows, then sample AusPlots."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

from audit_ausplots_metadata import CACHE as METADATA, OUT as METADATA_RESULTS, ROOT, locations, sha
from geolifeclef.range_io import BoundedHTTPFile
from geolifeclef.raster_bridge import nearest_valid_value


DATA = ROOT / "data/ausplots_followup"
OUTPUT = ROOT / "results/ausplots_followup"
PROTOCOL = ROOT / "docs/ausplots_followup_protocol.md"
BRIDGE = ROOT / "data/neon_bridge"
SOURCE = BRIDGE / "published_source/scripts/data_prep_scripts/predictors.py"
RELEASED = ROOT / "data/external_replication_e1/predictors/eval_location.h5"
SAGE_PLOTS = ROOT / "data/independent_data_audit/sage_eval_plot_metadata.csv"
GEOMETRY = METADATA / "post_cutoff_visit_geometry.csv"
MASK_RECEIPT = METADATA_RESULTS / "native_mask_feasibility.json"

BLOCK_SIZE = 256 * 1024
MAX_BYTES_PER_LAYER = 144 * 1024**2
SAGE_FEATURE_GROUPS = {"chelsa": 19, "soilgrids": 8, "topography": 16, "human_footprint": 9}


def published_layer_names():
    tree = ast.parse(SOURCE.read_text())
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SOILGRIDS_FILENAMES", "TOPO_FILENAMES"}
    }
    human = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "get_human_footprint_paths")
    return {
        "chelsa": [f"CHELSA_bio{index:02d}_1981-2010_V.2.1.tif" for index in range(1, 20)],
        "soilgrids": assignments["SOILGRIDS_FILENAMES"],
        "topography": assignments["TOPO_FILENAMES"],
        "human_footprint": ast.literal_eval(human.body[0].value),
    }


def layer_url(group, name):
    if group == "chelsa":
        return f"https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/bioclim/{name.split('_')[1]}/1981-2010/{name}"
    if group == "soilgrids":
        return f"https://files.isric.org/soilgrids/former/2017-03-10/data/{name}"
    if group == "topography":
        return f"https://data.earthenv.org/topography/{name}"
    return None


def reference_rows(count):
    first = np.load(ROOT / "data/external_replication_e1/test_indices.npy")[:8]
    metadata = pd.read_csv(SAGE_PLOTS, usecols=["GIVD_ID", "cell_row"])
    australian = np.setdiff1d(metadata.loc[metadata.GIVD_ID.eq("AU-AU-002"), "cell_row"], first)
    selected = australian[np.linspace(0, len(australian) - 1, 12, dtype=int)]
    remaining = np.setdiff1d(np.arange(count), np.concatenate([first, selected]))
    return np.concatenate([first, selected, remaining[np.linspace(0, len(remaining) - 1, 12, dtype=int)]])


def candidate_visits():
    expected_hash = json.loads(MASK_RECEIPT.read_text())["sha256"][str(GEOMETRY.relative_to(ROOT))]
    if sha(GEOMETRY) != expected_hash:
        raise ValueError("Geometry differs from the frozen metadata audit")
    frame = pd.read_csv(GEOMETRY)
    visits = frame.loc[frame.legacy_distance_km.gt(1) & ~frame.legacy_same_grid & ~frame.train_same_grid].copy()
    if len(visits) != 151 or visits.duplicated(["site_location_name", "site_location_visit_id"]).any():
        raise ValueError("Unexpected candidate visit keys")
    return visits


def candidate_cells():
    cells = candidate_visits()[["grid_x", "grid_y"]].drop_duplicates().sort_values(["grid_x", "grid_y"])
    cells = cells.to_numpy(dtype=np.int64)
    if len(cells) != 119:
        raise ValueError("Unexpected candidate cell count")
    centers = np.column_stack(transform("EPSG:6933", "EPSG:4326", *(1000 * (cells.T + 0.5))))
    return cells, centers.astype(np.float32)


def sample_raster(source, coords):
    x, y = np.asarray(transform("EPSG:4326", source.crs, *coords.T), dtype=np.float64)
    affine = ~source.transform
    col = np.floor(affine.a * x + affine.b * y + affine.c).astype(np.intp)
    row = np.floor(affine.d * x + affine.e * y + affine.f).astype(np.intp)
    col = np.clip(col, 0, source.width - 1)
    row = np.clip(row, 0, source.height - 1)
    # Sorted pixels reduce block-cache churn on strip-based rasters.
    order = np.lexsort((col, row))
    values = np.empty(len(coords), dtype=np.float32)
    fills = unfilled = 0
    for index in order:
        r, c = int(row[index]), int(col[index])
        value = float(source.read(1, window=Window(c, r, 1, 1))[0, 0])
        if source.nodata is not None and np.isclose(value, source.nodata, atol=0):
            value, _, _ = nearest_valid_value(source, r, c)
            fills += int(np.isfinite(value))
            unfilled += int(np.isnan(value))
        values[index] = value
    return values, {"nearest_fills": fills, "unfilled_nodata": unfilled, "nan_values": int(np.isnan(values).sum())}


def remote_identity(url):
    request = Request(url, headers={"Range": "bytes=0-31", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=25) as response:
        match = re.fullmatch(r"bytes 0-31/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None or len(response.read(33)) != 32:
            raise OSError("Expected a bounded 32-byte range from the source")
        return {"size": int(match[1]), "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified")}


def check_layer(group, name, reference_coords, expected, centers, human_dir):
    url = layer_url(group, name)
    receipt = {"group": group, "file": name, "source_url": url, "status": "source_unavailable"}
    target = None
    streams = []
    try:
        if url:
            identity = remote_identity(url)
            receipt["source_identity"] = identity
            cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]

            def opener(path, mode="rb"):
                if str(path) != name or mode != "rb":
                    raise FileNotFoundError(path)
                remaining_budget = MAX_BYTES_PER_LAYER - sum(stream.fetched_bytes for stream in streams)
                stream = BoundedHTTPFile(url, 0, identity["size"], budget=remaining_budget,
                                         block_size=BLOCK_SIZE, cache_dir=DATA / "rasters" / name / cache_key)
                streams.append(stream)
                return stream

            raster = name
        else:
            if human_dir is None or not (human_dir / name).is_file():
                return receipt, target
            raster, opener = human_dir / name, None
            receipt["local_raster_sha256"] = sha(raster)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO"):
            with rasterio.open(raster, opener=opener) as source:
                actual, diagnostics = sample_raster(source, reference_coords)
                equal = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
                comparable = np.isfinite(actual) & np.isfinite(expected)
                maximum = np.max(np.abs(actual[comparable].astype(np.float64) - expected[comparable]), initial=0)
                receipt.update({"crs": str(source.crs), "shape": [source.height, source.width],
                                "dtype": source.dtypes[0], "reference": diagnostics,
                                "reference_matches": int(equal.sum()), "reference_rows": len(expected),
                                "reference_nan_pattern_disagreements": int((np.isnan(actual) != np.isnan(expected)).sum()),
                                "reference_max_finite_absolute_error": float(maximum),
                                "reference_mismatched_positions": np.flatnonzero(~equal).tolist(),
                                "status": "reference_mismatch"})
                if equal.all():
                    receipt["status"] = "reference_passed_candidate_pending"
                    target, candidate_diagnostics = sample_raster(source, centers)
                    receipt.update({"status": "verified_and_sampled", "candidate": candidate_diagnostics,
                                    "candidate_finite": int(np.isfinite(target).sum()),
                                    "candidate_value_sha256": hashlib.sha256(target.tobytes()).hexdigest()})
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)[:250]}
    finally:
        if streams:
            receipt["range_bytes"] = sum(stream.fetched_bytes for stream in streams)
            receipt["range_requests"] = sum(stream.requests for stream in streams)
            for stream in streams:
                stream.close()
    return receipt, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--human-dir", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.run) or not 1 <= args.workers <= 12:
        raise ValueError("Invalid run name or worker count")
    if args.human_dir is not None and not (args.human_dir / "provenance.json").is_file():
        raise FileNotFoundError("Record source/archive provenance beside the Human Footprint rasters")
    output = OUTPUT / args.run
    output.mkdir(parents=True, exist_ok=False)
    data = DATA / args.run
    data.mkdir(parents=True, exist_ok=True)
    cells, centers = candidate_cells()
    coords = locations(RELEASED)
    ids = reference_rows(len(coords))
    layers = published_layer_names()
    if {key: len(val) for key, val in layers.items()} != SAGE_FEATURE_GROUPS:
        raise ValueError("Published layer cardinalities changed")
    reference_paths = {group: BRIDGE / "reference_predictors" / f"eval_{group}.h5" for group in layers}
    reference = {}
    for group, path in reference_paths.items():
        with h5py.File(path) as handle:
            reference[group] = handle["predictor_data"][:][ids]
    jobs = [(group, name, coords[ids], reference[group][:, column], centers, args.human_dir)
            for group, names in layers.items() for column, name in enumerate(names)]

    def check(job):
        receipt, values = check_layer(*job)
        print(f"{receipt['file']} {receipt['status']} {receipt.get('reference_matches', '-')}", flush=True)
        return receipt, values

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, jobs))
    receipts = [result[0] for result in results]
    vectors = np.column_stack([values if values is not None else np.full(len(cells), np.nan, np.float32)
                               for _, values in results])
    verified = np.array([receipt["status"] == "verified_and_sampled" for receipt in receipts])
    path = data / "candidate_features.npz"
    np.savez_compressed(path, cell_keys=cells, values=vectors, sampled_columns=verified,
                        feature_names=[receipt["file"] for receipt in receipts])
    paths = [GEOMETRY, MASK_RECEIPT, RELEASED, SAGE_PLOTS, SOURCE, PROTOCOL, Path(__file__), *reference_paths.values(),
             ROOT / "data/external_replication_e1/test_indices.npy",
             ROOT / "src/geolifeclef/range_io.py", ROOT / "src/geolifeclef/raster_bridge.py"]
    if args.human_dir is not None:
        paths.append(args.human_dir / "provenance.json")
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "reference_row_ids": ids.tolist(),
              "candidate_cells": len(cells), "verified_sampled_features": int(verified.sum()),
              "full_reference_match_and_sampled": bool(verified.all()),
              "candidate_cells_all_52_finite": int(np.isfinite(vectors).all(axis=1).sum()),
              "species_outcomes_read": False, "candidate_features_sha256": sha(path),
              "range_block_bytes": BLOCK_SIZE, "max_range_bytes_per_layer": MAX_BYTES_PER_LAYER,
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "layers": receipts}
    (output / "predictors.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{verified.sum()}/52 features reproduced and sampled", flush=True)


if __name__ == "__main__":
    main()
