"""Build the one-file, self-contained Kaggle v24 notebook."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


EXPECTED_V23_HASH = "9da01ce45a3478e8073cd93e22dbf69dde65def0f86b7ef2700e84630f2c30f8"
REQUIRED_EVIDENCE = {
    "assessment_per_survey.csv",
    "frozen_policies.json",
    "GLC25_PA_submission_v23.csv",
    "partition_ids.csv",
    "po_manifests.json",
    "pre_assessment_freeze.json",
    "split_manifest.json",
    "unchanged_v22_submission.csv",
    "v23_report.json",
}


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [""]


def canonical_v23_files(directory: Path) -> dict[str, bytes]:
    found = {path.name for path in directory.iterdir() if path.is_file()}
    missing = REQUIRED_EVIDENCE - found
    if missing:
        raise FileNotFoundError(f"Missing v23 evidence: {sorted(missing)}")
    files = {name: (directory / name).read_bytes() for name in sorted(REQUIRED_EVIDENCE)}
    submission = files["GLC25_PA_submission_v23.csv"]
    if hashlib.sha256(submission).hexdigest() != EXPECTED_V23_HASH:
        if hashlib.sha256(submission + b"\r\n").hexdigest() == EXPECTED_V23_HASH:
            submission += b"\r\n"
        else:
            raise ValueError("The supplied v23 CSV cannot be restored to its frozen SHA256")
    files["GLC25_PA_submission_v23.csv"] = submission
    return files


def payload(files: dict[str, bytes]) -> str:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, values in sorted(files.items()):
            item = tarfile.TarInfo(name=name)
            item.size = len(values)
            item.mtime = 0
            item.mode = 0o644
            archive.addfile(item, io.BytesIO(values))
    packed = gzip.compress(stream.getvalue(), compresslevel=9, mtime=0)
    return base64.b64encode(packed).decode("ascii")


def make_notebook(core: str, payload_b64: str) -> dict:
    source_sha = hashlib.sha256(core.encode("utf-8")).hexdigest()
    markdown = """# GeoLifeCLEF 2025 — v24 multimodal rare-species SDM

This is the complete Kaggle deliverable. It uses only the official `geolifeclef-2025`
competition input and one GPU. The exact v23 submission and its audit evidence are embedded
as the frozen control. Internet and external/pretrained weights are not used.

The notebook has an 11.25-hour hard budget inside Kaggle's 12-hour limit. It trains two new
spatial outer folds plus one deployment model, freezes every decision before assessment,
and writes exactly four files to `/kaggle/working/v24_export`. Submit
`GLC25_PA_submission_v24.csv` only when the printed `eligible_for_submission` value is `true`.
"""
    payload_cell = (
        f"NOTEBOOK_SOURCE_SHA256 = {source_sha!r}\n"
        "# Exact frozen v23 evidence, compressed into this notebook.\n"
        f"FROZEN_V23_PAYLOAD_B64 = {payload_b64!r}\n"
        "print({'embedded_v23_payload_bytes': len(FROZEN_V23_PAYLOAD_B64), "
        "'core_sha256': NOTEBOOK_SOURCE_SHA256})\n"
    )
    run_cell = (
        "SELF_TEST_RESULTS = notebook_self_tests()\n"
        "print({'self_tests': SELF_TEST_RESULTS})\n"
        "V24_RESULT = run_v24(FROZEN_V23_PAYLOAD_B64)\n"
        "print(json.dumps(V24_RESULT, indent=2))\n"
        "V24_RESULT\n"
    )
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": source_lines(markdown)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": source_lines(core)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": source_lines(payload_cell)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": source_lines(run_cell)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "glc_v24": {
                "experiment": "v24_multimodal_rare_species_sdm",
                "frozen_v23_commit": "d307326eb55af13d1bc3b593f17997a8246df644",
                "frozen_v23_submission_sha256": EXPECTED_V23_HASH,
                "intended_kernel": "con1los/geolifeclef-risk-aware-sdm-phase-1",
                "intended_kernel_version": 26,
                "required_input": ["geolifeclef-2025"],
                "accelerator": "NVIDIA T4 x1 or faster",
                "internet": False,
                "max_total_hours": 11.25,
                "final_output": "/kaggle/working/v24_export/GLC25_PA_submission_v24.csv",
                "submission_performed": False,
                "notebook_source_sha256": source_sha,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=repository / "artifacts/manual_upload_v23_retry2/user_provided_v25_exports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "notebooks/geolifeclef_v24_multimodal_rare_species_sdm.ipynb",
    )
    args = parser.parse_args()
    core_path = repository / "scripts/v24_notebook_core.py"
    core = core_path.read_text(encoding="utf-8")
    notebook = make_notebook(core, payload(canonical_v23_files(args.evidence)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "cells": len(notebook["cells"])}, indent=2))


if __name__ == "__main__":
    main()
