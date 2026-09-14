"""Manual-upload v23 launcher with strict inputs and a 10.5-hour parent guard."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXPECTED_COMMIT = os.environ.get("GLC_SOURCE_COMMIT", "")
EXPECTED_KERNEL_VERSION = 23
MAX_SECONDS = int(10.5 * 3600)


def _one(root: Path, marker: str, slug: str) -> Path:
    candidates = [path.parent for path in root.rglob(marker) if slug in str(path.parent)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Exactly one mounted {slug}/1 input is required")
    return candidates[0]


def main():
    os.environ.setdefault("GLC_PIPELINE_STARTED_AT", str(time.time()))
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["GLC_KERNEL_VERSION"] = str(EXPECTED_KERNEL_VERSION)
    started = float(os.environ["GLC_PIPELINE_STARTED_AT"])

    def remaining():
        seconds = MAX_SECONDS - (time.time() - started)
        if seconds <= 0:
            raise TimeoutError("Total v23 budget exceeded 10.5 hours")
        return seconds

    root = Path("/kaggle/input")
    data = next((path for path in (root / "competitions/geolifeclef-2025",
                 root / "geolifeclef-2025") if path.is_dir()), None)
    if data is None:
        raise FileNotFoundError("Mounted GeoLifeCLEF 2025 competition data is required")
    v20 = _one(root, "provenance.json", "geolifeclef-v20-frozen-control")
    v21 = _one(root, "v21_provenance.json", "geolifeclef-v21-frozen-control")
    v22 = _one(root, "provenance.json", "geolifeclef-v22-frozen-control")
    if not EXPECTED_COMMIT or len(EXPECTED_COMMIT) != 40:
        raise ValueError("Embedded source commit was not established")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, timeout=remaining())
    os.environ["GLC_TESTS_BEFORE"] = "1"
    subprocess.run([sys.executable, "-m", "scripts.run_diverse_po_v23",
        "--data-root", str(data), "--frozen-v20-dir", str(v20),
        "--frozen-v21-dir", str(v21), "--frozen-v22-dir", str(v22),
        "--expected-commit", EXPECTED_COMMIT], check=True, timeout=remaining())
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, timeout=remaining())
    report_path = Path("artifacts/diverse_po_v23/v23_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["notebook_tests_after_passed"] = True
    report["total_pipeline_hours"] = (time.time() - started) / 3600
    report["integrity"]["registered_runtime"] = report["total_pipeline_hours"] < 10.5
    report["integrity"]["notebook_tests_before_and_after_passed"] = (
        report.get("notebook_tests_before_passed") is True)
    from scripts.v23_protocol import submission_gate
    report["submission_gate"] = submission_gate(
        report["assessment"], report["integrity"], report["selected_policies"])
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    remaining()
    print(json.dumps({"V23_RUN_COMPLETE": True, "hours": report["total_pipeline_hours"],
                      "manual_submission_gate": report["submission_gate"]["eligible_for_manual_submission"],
                      "output_csv": report["output_csv"]}))


if __name__ == "__main__":
    main()
