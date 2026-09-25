"""Build the standalone v29 notebook; no runtime repository or auxiliary dataset."""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
from pathlib import Path

from scripts.build_v28_notebook import packed_consumed_ids, packed_v27_submission

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/geolifeclef_v29_full_data_multisensor_ensemble.ipynb"
LIMIT = 1_000_000


def consumed_paths(root=ROOT):
    return [root / "artifacts/v21_review/assessment_per_survey.csv",
            root / "artifacts/v22_review/assessment_per_survey.csv",
            root / "artifacts/manual_upload_v23_retry2/user_provided_v25_exports/assessment_per_survey.csv",
            *[root / f"results/v{version}_kaggle_output/assessment_per_survey_v{version}.csv"
              for version in range(24, 29)]]


def make_notebook(core, reference, control, consumed):
    core_hash = hashlib.sha256(core.encode()).hexdigest()
    reference_hash = hashlib.sha256(reference.encode()).hexdigest()
    packed = base64.b64encode(lzma.compress(reference.encode(), preset=9)).decode()
    loader = (
        "import types as _types\n"
        f"LEGACY_SOURCE_HASH = {reference_hash!r}\n"
        f"_reference_source = lzma.decompress(base64.b64decode({packed!r}))\n"
        "assert hashlib.sha256(_reference_source).hexdigest() == LEGACY_SOURCE_HASH\n"
        "legacy = _types.ModuleType('v29_frozen_reference')\n"
        "exec(compile(_reference_source, 'frozen-v27-reference', 'exec'), legacy.__dict__)\n"
        "del _reference_source\n")
    if core.count("import scripts.v27_notebook_core as legacy") != 1:
        raise ValueError("Unexpected legacy import")
    standalone = core.replace("import scripts.v27_notebook_core as legacy", loader)
    payloads = (f"V29_SOURCE_HASH = {core_hash!r}\n"
                f"CONTROL_PAYLOAD_HASH = {control[1]!r}\n"
                f"CONTROL_RAW_HASH = {control[2]!r}\n"
                f"CONTROL_B64 = {control[0]!r}\n"
                f"CONSUMED_HASH = {consumed[1]!r}\n"
                f"CONSUMED_COUNT = {consumed[2]}\n"
                f"CONSUMED_B64 = {consumed[0]!r}\n"
                "print({'source_sha256': V29_SOURCE_HASH, 'consumed_audit_ids': CONSUMED_COUNT})\n")
    intro = """# GeoLifeCLEF v29 — full-data multisensor ensemble

**Experimental candidate, not a demonstrated SOTA result.** v28 reproduced v27's CSV exactly.
This version trains three genuinely different models from competition data: temporal/spatial
attention, chronological convolution, and a geography-free attention expert. It uses 64x64
Sentinel images, all 5,016 species, calibrated probability ensembling and adaptive set sizes.
Winning calibrated candidates are refitted on all 88,987 PA surveys before test inference.
No external data, pretrained weights, Internet, extra dataset, or repository is required.

## Kaggle setup

1. Attach **GeoLifeCLEF25 @ CVPR & LifeCLEF** (`geolifeclef-2025`).
2. Select **GPU T4 x1**, restart the session, and **Run All**.
3. Read the final decision. Submit `v29_export/GLC25_PA_submission_v29.csv` ONLY when
   `eligible_for_submission` is `true`. Files named `DO_NOT_SUBMIT` failed the gate or
   are unchanged v27 predictions. Do not submit them as a new experiment.

The cooperative runtime budget is 10.75 hours, with preflight checks, batch-level guards,
time-bounded development training and full-ensemble refit admission. Completion on an
unmeasured Kaggle session cannot be guaranteed: an exceptionally slow run stops safely.
Temporary rasters and checkpoints are removed; only four small export files remain.

The 86,592 previously assessed IDs are excluded from the one final audit. Its remaining
2,395 surveys are mostly Danish and cannot establish hidden-test performance. Selection
and calibration use previously observed development labels. Full-data refits use all PA
labels; the independent audit measures only the separately held-out development predictor,
not the full-data production weights. The matched v27 reference is a recipe refit, not its
exact deployed weights; exact scored v27 predictions are embedded for the test control.
"""
    run = ("V29_RESULT = run_v29(CONTROL_B64, CONSUMED_B64)\n"
           "print(json.dumps(V29_RESULT, indent=2))\n"
           "print('\\n' + V29_RESULT['message'])\n"
           "V29_RESULT\n")
    cells = [{"cell_type": "markdown", "id": "v29-intro", "metadata": {},
              "source": intro.splitlines(keepends=True)}]
    for name, source in (("core", standalone), ("evidence", payloads), ("run", run)):
        compile(source, f"v29-{name}", "exec")
        cells.append({"cell_type": "code", "id": f"v29-{name}", "metadata": {},
                      "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook"},
        "glc_v29": {"experiment": "v29_full_data_multisensor_ensemble",
                    "core_sha256": core_hash, "reference_sha256": reference_hash,
                    "consumed_assessment_ids": consumed[2], "max_total_hours": 10.75,
                    "required_input": ["geolifeclef-2025"], "external_weights": False,
                    "official_submission_made": False}}}


def build(output=OUTPUT):
    control = packed_v27_submission(ROOT / "results/v27_kaggle_output/GLC25_PA_submission_v27.csv",
                                    ROOT / "artifacts/v20_frozen_bundle_v21/species_ids.npy")
    consumed = packed_consumed_ids(consumed_paths())
    notebook = make_notebook((ROOT / "scripts/v29_notebook_core.py").read_text(encoding="utf-8"),
                             (ROOT / "scripts/v27_notebook_core.py").read_text(encoding="utf-8"),
                             control, consumed)
    serialized = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if len(serialized) >= LIMIT:
        raise ValueError(f"Notebook exceeds Kaggle source limit: {len(serialized)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)
    print(json.dumps({"notebook": str(output), "bytes": len(serialized),
                      "sha256": hashlib.sha256(serialized).hexdigest(), "consumed_ids": consumed[2]}, indent=2))
    return notebook


if __name__ == "__main__":
    build()
