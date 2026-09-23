"""Build the compact, self-contained GeoLifeCLEF v26 Kaggle notebook."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


EXPECTED_V25_HASH = "c450107d5219bb3a99f37741142ea40cf9320c87e851173ec50f1e899ada48c5"
KAGGLE_KERNEL_SOURCE_LIMIT_BYTES = 1_000_000
EXPECTED_ROWS = 14_784
EXPECTED_SPECIES = 5_016


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [""]


def canonical_submission_bytes(path: Path) -> bytes:
    values = path.read_bytes()
    if hashlib.sha256(values).hexdigest() == EXPECTED_V25_HASH:
        return values
    for suffix in (b"\r\n", b"\n"):
        if hashlib.sha256(values + suffix).hexdigest() == EXPECTED_V25_HASH:
            return values + suffix
    raise ValueError("The supplied v25 submission does not match the frozen official artifact")


def packed_v25_submission(submission_path: Path, species_path: Path) -> tuple[str, str, str]:
    submission = canonical_submission_bytes(submission_path)
    frame = pd.read_csv(submission_path)
    species = np.load(species_path, allow_pickle=False).astype(np.int64)
    if list(frame.columns) != ["surveyId", "predictions"] or len(frame) != EXPECTED_ROWS:
        raise ValueError("Unexpected v25 submission schema")
    if len(species) != EXPECTED_SPECIES:
        raise ValueError("Unexpected species vocabulary")
    lookup = pd.Index(species)
    rows = []
    for text in frame.predictions.astype(str):
        columns = lookup.get_indexer(np.fromstring(text, sep=" ", dtype=np.int64))
        if (columns < 0).any() or len(columns) != len(np.unique(columns)) or not 10 <= len(columns) <= 40:
            raise ValueError("Invalid v25 prediction row")
        rows.append(columns.astype("<u2"))
    counts = np.asarray([len(row) for row in rows], dtype=np.uint8)
    raw = counts.tobytes() + np.concatenate(rows).astype("<u2").tobytes()
    packed = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    return (base64.b64encode(packed).decode("ascii"), hashlib.sha256(packed).hexdigest(),
            hashlib.sha256(raw).hexdigest())


def packed_consumed_ids(paths: list[Path]) -> tuple[str, str, int]:
    consumed: set[int] = set()
    for path in paths:
        consumed.update(pd.read_csv(path, usecols=["surveyId"]).surveyId.astype("int64"))
    ordered = np.asarray(sorted(consumed), dtype=np.int64)
    deltas = np.diff(np.r_[np.int64(0), ordered]).astype("<u4")
    packed = lzma.compress(deltas.tobytes(), preset=9 | lzma.PRESET_EXTREME)
    return base64.b64encode(packed).decode("ascii"), hashlib.sha256(packed).hexdigest(), len(ordered)


def make_notebook(core: str, control_b64: str, control_sha256: str, control_raw_sha256: str,
                  consumed_b64: str, consumed_sha256: str, consumed_count: int,
                  source_commit: str | None = None) -> dict:
    source_sha = hashlib.sha256(core.encode("utf-8")).hexdigest()
    source_commit = source_commit or f"notebook-source-sha256:{source_sha}"
    markdown = """# GeoLifeCLEF 2025 — v26 raw-spatial raster ensemble

This complete Kaggle deliverable uses only the official `geolifeclef-2025` input and one GPU.
**Before running:** open the right sidebar and set `Session options -> Accelerator -> GPU T4 x1`,
restart the session, and then choose `Run All`. The first runtime check stops immediately when
CUDA is unavailable, because CPU training cannot finish safely within 12 hours.
The exact scored v25 prediction is embedded in compact binary form as the official-test control.
Every assessment survey used by v21–v25 is embedded as an immutable exclusion set; v26
assessment therefore uses only never-assessed surveys.

The candidate preserves v25's adaptive-count gain but replaces summary-only remote sensing with
small residual CNN encoders over raw Sentinel, Landsat, and bioclimatic tensors. It averages two
deployment seeds and blends their ranking with exact v25. It has a 10.75-hour hard budget and
always deletes temporary rasters/checkpoints. Only four compact files remain in
`/kaggle/working/v26_export`. Submit `GLC25_PA_submission_v26.csv` only when
`eligible_for_submission` is `true`.
"""
    payload_cell = (
        f"NOTEBOOK_SOURCE_SHA256 = {source_sha!r}\n"
        f"V26_SOURCE_COMMIT = {source_commit!r}\n"
        f"FROZEN_V25_PAYLOAD_SHA256 = {control_sha256!r}\n"
        f"FROZEN_V25_RAW_SHA256 = {control_raw_sha256!r}\n"
        f"FROZEN_V25_PAYLOAD_B64 = {control_b64!r}\n"
        f"CONSUMED_ASSESSMENT_IDS_SHA256 = {consumed_sha256!r}\n"
        f"CONSUMED_ASSESSMENT_IDS_COUNT = {consumed_count}\n"
        f"CONSUMED_ASSESSMENT_IDS_B64 = {consumed_b64!r}\n"
        "print({'embedded_v25_bytes': len(FROZEN_V25_PAYLOAD_B64), "
        "'consumed_ids': CONSUMED_ASSESSMENT_IDS_COUNT, 'core_sha256': NOTEBOOK_SOURCE_SHA256})\n"
    )
    run_cell = (
        "SELF_TEST_RESULTS = notebook_self_tests()\n"
        "print({'self_tests': SELF_TEST_RESULTS})\n"
        "V26_RESULT = run_v26(FROZEN_V25_PAYLOAD_B64, CONSUMED_ASSESSMENT_IDS_B64)\n"
        "print(json.dumps(V26_RESULT, indent=2))\n"
        "V26_RESULT\n"
    )
    return {
        "cells": [
            {"cell_type": "markdown", "id": "v26-intro", "metadata": {},
             "source": source_lines(markdown)},
            {"cell_type": "code", "id": "v26-core", "execution_count": None,
             "metadata": {}, "outputs": [], "source": source_lines(core)},
            {"cell_type": "code", "id": "v26-evidence", "execution_count": None,
             "metadata": {}, "outputs": [], "source": source_lines(payload_cell)},
            {"cell_type": "code", "id": "v26-run", "execution_count": None,
             "metadata": {}, "outputs": [], "source": source_lines(run_cell)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "isGpuEnabled": True,
                       "isInternetEnabled": False, "language": "python",
                       "sourceType": "notebook"},
            "glc_v26": {
                "experiment": "v26_raw_spatial_raster_ensemble",
                "source_commit": source_commit,
                "frozen_v25_submission_sha256": EXPECTED_V25_HASH,
                "consumed_assessment_ids": consumed_count,
                "required_input": ["geolifeclef-2025"],
                "accelerator": "NVIDIA T4 x1 or faster", "internet": False,
                "max_total_hours": 10.75,
                "final_output": "/kaggle/working/v26_export/GLC25_PA_submission_v26.csv",
                "submission_performed": False, "notebook_source_sha256": source_sha,
            },
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--v25-output", type=Path,
                        default=repository / "results/v25_kaggle_output")
    parser.add_argument("--species", type=Path,
                        default=repository / "artifacts/v20_frozen_bundle_v21/species_ids.npy")
    parser.add_argument("--output", type=Path,
                        default=repository / "notebooks/geolifeclef_v26_raw_spatial_raster_ensemble.ipynb")
    args = parser.parse_args()
    consumed_paths = [
        repository / "artifacts/v21_review/assessment_per_survey.csv",
        repository / "artifacts/v22_review/assessment_per_survey.csv",
        repository / "artifacts/manual_upload_v23_retry2/user_provided_v25_exports/assessment_per_survey.csv",
        repository / "results/v24_kaggle_output/assessment_per_survey_v24.csv",
        args.v25_output / "assessment_per_survey_v25.csv",
    ]
    control = packed_v25_submission(args.v25_output / "GLC25_PA_submission_v25.csv", args.species)
    consumed = packed_consumed_ids(consumed_paths)
    core_path = repository / "scripts/v26_notebook_core.py"
    core = core_path.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(core.encode("utf-8")).hexdigest()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        source_commit = f"{head};notebook-source-sha256:{source_sha}"
    except (OSError, subprocess.CalledProcessError):
        source_commit = f"notebook-source-sha256:{source_sha}"
    notebook = make_notebook(core, *control, *consumed, source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    size = args.output.stat().st_size
    if size >= KAGGLE_KERNEL_SOURCE_LIMIT_BYTES:
        raise RuntimeError(f"Generated notebook is {size} bytes; Kaggle requires less than 1 MB")
    print(json.dumps({"output": str(args.output), "bytes": size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "cells": len(notebook["cells"]), "consumed_ids": consumed[2]}, indent=2))


if __name__ == "__main__":
    main()
