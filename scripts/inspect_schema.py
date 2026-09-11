#!/usr/bin/env python
"""Probe GeoLifeCLEF's raw tables and one cube per modality without loading all data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def describe_value(value: Any) -> dict[str, Any]:
    """Return JSON-safe tensor/container metadata only, never its values."""
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": {str(key): describe_value(item) for key, item in value.items()}}
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "length": len(value), "items": [describe_value(item) for item in value[:8]]}
    return {"kind": type(value).__name__}


def describe_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, nrows=8)
    return {
        "path": str(path),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "sample_rows_read": len(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=Path("data/reports/schema_report.json"))
    args = parser.parse_args()
    root = args.data_root
    if not root.is_dir():
        raise SystemExit(f"Data directory not found: {root}")

    table_paths = [
        root / "GLC25_PA_metadata_train.csv",
        root / "GLC25_PA_metadata_test.csv",
        root / "GLC25_SAMPLE_SUBMISSION.csv",
        root / "SateliteTimeSeries-Landsat" / "values" / "PA-train" / "GLC25-PA-train-landsat_time_series-blue.csv",
        root / "BioclimTimeSeries" / "values" / "GLC25-PA-train-bioclimatic_monthly.csv",
    ]
    tables = [describe_csv(path) for path in table_paths if path.is_file()]
    cube_paths = {
        "landsat": next((root / "SateliteTimeSeries-Landsat" / "cubes" / "PA-train").glob("*.pt"), None),
        "bioclim": next((root / "BioclimTimeSeries" / "cubes" / "PA-train").glob("*.pt"), None),
    }
    cubes: dict[str, Any] = {}
    for modality, path in cube_paths.items():
        if path is not None:
            value = torch.load(path, map_location="cpu", weights_only=True)
            cubes[modality] = {"path": str(path.relative_to(root)), "contents": describe_value(value)}

    report = {"data_root": str(root), "tables": tables, "sample_cubes": cubes}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
