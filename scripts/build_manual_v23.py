"""Build the self-contained notebook and local manual-upload delivery package."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

from scripts.stage_frozen_v20 import sha256_file
from scripts.stage_frozen_v22 import DATASET_SLUG, build as build_v22_bundle
from scripts.v23_protocol import EXPECTED_KERNEL_VERSION


NOTEBOOK_NAME = "geolifeclef_v23_manual.ipynb"
DATASET_ZIP = "frozen_v22_dataset.zip"
OUTPUT_CSV = "GLC25_PA_submission_v23.csv"
REQUIRED_INPUTS = [
    "geolifeclef-2025",
    "con1los/geolifeclef-v20-frozen-control/1",
    "con1los/geolifeclef-v21-frozen-control/1",
    "con1los/geolifeclef-v22-frozen-control/1",
]


def _git(repository: Path, *arguments: str, binary=False):
    result = subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True,
                            text=not binary)
    return result.stdout


def _source_archive(repository: Path, commit: str):
    raw = _git(repository, "archive", "--format=tar", commit, binary=True)
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    manifest = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isfile():
                manifest[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return compressed, manifest


def _notebook(commit: str, archive: bytes, source_manifest: dict) -> dict:
    archive_sha = hashlib.sha256(archive).hexdigest()
    encoded = base64.b64encode(archive).decode("ascii")
    source_json = json.dumps(source_manifest, sort_keys=True, separators=(",", ":"))
    bootstrap = r'''import base64, gzip, hashlib, io, json, os, subprocess, sys, tarfile, time
from pathlib import Path
os.environ.setdefault("GLC_PIPELINE_STARTED_AT", str(time.time()))
if hashlib.sha256(base64.b64decode(SOURCE_ARCHIVE_B64)).hexdigest() != SOURCE_ARCHIVE_SHA256:
    raise RuntimeError("Embedded source archive hash mismatch")
destination = Path("/kaggle/working/geolifeclef_v23_source")
if destination.exists():
    raise RuntimeError("Source destination must be absent at notebook start")
destination.mkdir(parents=True)
raw = gzip.decompress(base64.b64decode(SOURCE_ARCHIVE_B64))
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
            raise RuntimeError("Unsafe embedded source member")
    archive.extractall(destination)
manifest = json.loads(SOURCE_MANIFEST_JSON)
actual = {}
for path in sorted(destination.rglob("*")):
    if path.is_file():
        actual[path.relative_to(destination).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != manifest:
    raise RuntimeError("Extracted source does not match the tested commit manifest")
os.environ["GLC_SOURCE_COMMIT"] = EXPECTED_COMMIT
os.environ["PYTHONPATH"] = str(destination / "src") + os.pathsep + str(destination)
os.chdir(destination)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."],
               check=True, timeout=max(1, int(10.5 * 3600 - (time.time() - float(os.environ["GLC_PIPELINE_STARTED_AT"])))))
from scripts.launch_diverse_po_v23 import main
main()
'''
    return {
        "cells": [
            {"cell_type": "markdown", "id": "v23-design", "metadata": {}, "source": [
                "# GeoLifeCLEF 2025 — v23 diverse PO-initialized ensemble\n",
                "Self-contained manual-upload notebook. It embeds the exact tested Git source, uses only competition PA/PO/provided predictors plus frozen v20/v21/v22 private inputs, and never submits.\n",
            ]},
            {"cell_type": "code", "execution_count": None, "id": "v23-source", "metadata": {},
             "outputs": [], "source": [
                f'EXPECTED_COMMIT = {commit!r}\n',
                f'SOURCE_ARCHIVE_SHA256 = {archive_sha!r}\n',
                f'SOURCE_MANIFEST_JSON = {source_json!r}\n',
                f'SOURCE_ARCHIVE_B64 = {encoded!r}\n',
             ]},
            {"cell_type": "code", "execution_count": None, "id": "v23-run", "metadata": {},
             "outputs": [], "source": bootstrap.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "glc_v23": {"expected_source_commit": commit, "expected_kernel_version": EXPECTED_KERNEL_VERSION,
                         "max_total_hours": 10.5, "accelerator": "NVIDIA T4 x1",
                         "internet": False, "required_inputs": REQUIRED_INPUTS,
                         "output_csv": OUTPUT_CSV, "submission_performed": False},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def _instructions(commit: str, package: Path, tests_passed: int) -> str:
    downloaded = package / "downloaded_kaggle_v23_output"
    return f"""# Χειροκίνητη εκτέλεση GeoLifeCLEF v23 στο Kaggle

## Offline preflight πριν από upload

Από το repository root τρέξε ακριβώς:

```powershell
$env:PYTHONPATH='src;.'
& '.\\.venv\\research\\Scripts\\python.exe' scripts\\preflight_manual_v23.py --package '{package}'
```

Συνέχισε μόνο αν εμφανιστεί `"PREFLIGHT_OK": true`.

## Αρχεία και inputs

1. Ανέβασε το notebook `{NOTEBOOK_NAME}` από τον φάκελο `{package}`.
2. Δημιούργησε **Private Dataset** από τα μεμονωμένα αρχεία μέσα στον φάκελο `frozen_v22_dataset/`. Το `{DATASET_ZIP}` είναι μόνο αντίγραφο μεταφοράς· αν το χρησιμοποιήσεις, αποσυμπίεσέ το πρώτα και ανέβασε τα αρχεία του, όχι το ZIP ως μοναδικό αρχείο. Χρησιμοποίησε ακριβώς τίτλο `GeoLifeCLEF frozen v22 control` και slug `{DATASET_SLUG}`. Η πρώτη έκδοση πρέπει να είναι `{DATASET_SLUG}/1`.
3. Σύνδεσε ακριβώς αυτά τα υπάρχοντα private inputs στις συγκεκριμένες εκδόσεις:
   - `con1los/geolifeclef-v20-frozen-control/1`
   - `con1los/geolifeclef-v21-frozen-control/1`
   - `con1los/geolifeclef-v22-frozen-control/1`
4. Από **Add Input → Competitions**, σύνδεσε το `geolifeclef-2025` competition data. Μην προσθέσεις άλλο dataset.

## Ρυθμίσεις και εκτέλεση

- Accelerator: **GPU → NVIDIA T4 x1**. Ο κώδικας απαιτεί τουλάχιστον μία T4 και χρησιμοποιεί αποκλειστικά την `cuda:0`.
- Internet: **Off**.
- Notebook source commit: `{commit}`.
- Έλεγξε ότι το v22 dataset παραμένει **Private**.
- Πάτησε **Save Version**, επίλεξε **Save & Run All**, και επιβεβαίωσε. Η αναμενόμενη έκδοση του υπάρχοντος kernel είναι η **{EXPECTED_KERNEL_VERSION}**.

## Επιτυχής ολοκλήρωση

Το τελευταίο output πρέπει να περιέχει `"V23_RUN_COMPLETE": true`, χρόνο μικρότερο από 10,5 ώρες, το `manual_submission_gate`, και `"output_csv": "{OUTPUT_CSV}"`. Πρέπει επίσης να έχουν περάσει τα notebook tests πριν και μετά το pipeline.

Κατέβασε ολόκληρο τον φάκελο `geolifeclef_v23_source/artifacts/diverse_po_v23/` από τα Kaggle outputs και αποθήκευσέ τον τοπικά ως `{downloaded}`. Βεβαιώσου ειδικά ότι περιέχει:

- `{OUTPUT_CSV}`
- `unchanged_v22_submission.csv`
- `v23_report.json`
- `assessment_per_survey.csv`
- `frozen_policies.json`
- `pre_assessment_freeze.json`
- `split_manifest.json`
- `partition_ids.csv`
- `po_manifests.json`

Στο επόμενο chat επέστρεψε το πλήρες Kaggle execution log, το `v23_report.json`, το `assessment_per_survey.csv`, το `frozen_policies.json`, το `pre_assessment_freeze.json` και το `{OUTPUT_CSV}`.

## Gate πριν από επίσημη submission

Τρέξε τοπικά:

```powershell
$env:PYTHONPATH='src;.'
& '.\\.venv\\research\\Scripts\\python.exe' scripts\\validate_v23_outputs.py '{downloaded}' --expected-commit {commit} --template artifacts\\v20_frozen\\raw\\GLC25_SAMPLE_SUBMISSION.csv
```

Επίσημη submission επιτρέπεται μόνο αν ο validator επιστρέψει ακριβώς `ELIGIBLE_FOR_MANUAL_SUBMISSION`. Σε κάθε `DO_NOT_SUBMIT`, αποτυχία integrity gate, runtime ≥10,5 ώρες, missing output ή αβέβαιο run, μην υποβάλεις τίποτα. Αν περάσει το gate, το μοναδικό CSV που επιτρέπεται να υποβληθεί χειροκίνητα είναι το `{OUTPUT_CSV}`. Κανένα script του πακέτου δεν κάνει upload, run ή submission.

Το τοπικό package δημιουργήθηκε από καθαρό commit `{commit}` αφού πέρασαν {tests_passed} tests.
"""


def build(args):
    repository = args.repository.resolve()
    package = args.output.resolve()
    status = _git(repository, "status", "--porcelain", "--untracked-files=all").strip()
    if status:
        raise ValueError("Repository must be clean before packaging")
    commit = _git(repository, "rev-parse", "HEAD").strip()
    if len(commit) != 40:
        raise ValueError("Expected a full Git commit")
    if package.exists() and any(package.iterdir()):
        raise ValueError("Manual-upload output must be a fresh empty directory")
    package.mkdir(parents=True, exist_ok=True)
    dataset = package / "frozen_v22_dataset"
    build_v22_bundle(repository / "artifacts/v22_frozen_source",
        repository / "artifacts/v22_review", repository / "artifacts/v20_frozen_bundle_v21",
        repository / "artifacts/v21_frozen_bundle",
        repository / "artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv",
        repository / "artifacts/v20_frozen/raw/GLC25_PA_metadata_test.csv",
        repository / "artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv", dataset)
    archive, source_manifest = _source_archive(repository, commit)
    notebook = _notebook(commit, archive, source_manifest)
    notebook_path = package / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    instructions = package / "MANUAL_KAGGLE_INSTRUCTIONS.md"
    instructions.write_text(_instructions(commit, package, args.tests_passed), encoding="utf-8")
    zip_path = package / DATASET_ZIP
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive_zip:
        for path in sorted(dataset.iterdir()):
            archive_zip.write(path, arcname=path.name)
    hashed_paths = [notebook_path, zip_path, instructions, *sorted(dataset.iterdir())]
    relative_hashes = {path.relative_to(package).as_posix(): sha256_file(path) for path in hashed_paths}
    manifest = {
        "format": "geolifeclef_manual_upload_v23_v1", "created_from_clean_repository": True,
        "repository_clean_when_packaged": True, "expected_git_commit": commit,
        "expected_kernel": "con1los/geolifeclef-risk-aware-sdm-phase-1",
        "expected_kernel_version": EXPECTED_KERNEL_VERSION, "notebook": {"file": NOTEBOOK_NAME,
        "sha256": relative_hashes[NOTEBOOK_NAME]}, "dataset": DATASET_SLUG,
        "dataset_version_to_create": 1, "dataset_zip": {"file": DATASET_ZIP,
        "sha256": relative_hashes[DATASET_ZIP]}, "existing_inputs": REQUIRED_INPUTS[1:3],
        "competition_input": REQUIRED_INPUTS[0], "expected_output_csv": OUTPUT_CSV,
        "max_total_hours": 10.5, "accelerator": "NVIDIA T4 x1", "internet": False,
        "tests_passed_before_packaging": args.tests_passed, "sha256": relative_hashes,
        "no_upload_run_or_submission_performed": True,
    }
    (package / "UPLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(relative_hashes.items()))
    (package / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    return manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "artifacts/manual_upload_v23")
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
