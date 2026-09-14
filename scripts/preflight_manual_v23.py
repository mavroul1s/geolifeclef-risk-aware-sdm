"""Offline validation of the complete v23 manual-upload package."""
from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile

from scripts.stage_frozen_v20 import sha256_file
from scripts.stage_frozen_v22 import DATASET_SLUG, verify_v22


REQUIRED_INPUTS = [
    "geolifeclef-2025",
    "con1los/geolifeclef-v20-frozen-control/1",
    "con1los/geolifeclef-v21-frozen-control/1",
    "con1los/geolifeclef-v22-frozen-control/1",
]
DATASET_FILES = {
    "retained_po_calibration.npy", "retained_po_test.npy", "frozen_policies.json",
    "GLC25_PA_submission.csv", "v22_report.json", "provenance.json",
    "dataset-metadata.json", "hashes.json",
}


def _literal_assignments(source: str) -> dict:
    result = {}
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def _archive_files(data: bytes) -> dict[str, bytes]:
    result = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data)), mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe embedded source member")
            if member.isfile():
                result[member.name] = archive.extractfile(member).read()
    return result


def _scan_sensitive(name: str, data: bytes):
    text = data.decode("utf-8", errors="ignore")
    patterns = (
        r"(?i)x-amz-(?:signature|credential)=",
        r"(?i)https?://[^\s\"']+[?&](?:token|signature|credential)=",
        r'(?i)"key"\s*:\s*"[A-Za-z0-9_\-]{16,}"',
        r"KGAT_[A-Za-z0-9_\-]{16,}",
    )
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError(f"Credential or signed URL pattern detected in {name}")


def validate(args) -> dict:
    package = args.package.resolve()
    notebook_path = package / "geolifeclef_v23_manual.ipynb"
    manifest = json.loads((package / "UPLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("repository_clean_when_packaged") is not True:
        raise ValueError("Package does not attest a clean source repository")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    if None in cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Notebook cell IDs are missing or duplicated")
    assignments = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            compile(source, f"notebook-cell-{index}", "exec")
            assignments.update(_literal_assignments(source))
    expected_commit = assignments.get("EXPECTED_COMMIT")
    archive = base64.b64decode(assignments["SOURCE_ARCHIVE_B64"], validate=True)
    if hashlib.sha256(archive).hexdigest() != assignments.get("SOURCE_ARCHIVE_SHA256"):
        raise ValueError("Embedded source archive hash mismatch")
    files = _archive_files(archive)
    source_manifest = json.loads(assignments["SOURCE_MANIFEST_JSON"])
    if set(files) != set(source_manifest):
        raise ValueError("Embedded source member list differs from its manifest")
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != source_manifest[name]:
            raise ValueError(f"Embedded source hash mismatch: {name}")
        _scan_sensitive(f"embedded:{name}", data)
    metadata = notebook.get("metadata", {}).get("glc_v23", {})
    if metadata.get("expected_source_commit") != expected_commit:
        raise ValueError("Notebook commit metadata mismatch")
    if metadata.get("required_inputs") != REQUIRED_INPUTS or metadata.get("max_total_hours") != 10.5:
        raise ValueError("Notebook requests undeclared inputs or lacks the 10.5-hour guard")
    if metadata.get("expected_kernel_version") != 23 or metadata.get("output_csv") != "GLC25_PA_submission_v23.csv":
        raise ValueError("Notebook kernel/output contract mismatch")
    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repository,
                                    check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=args.repository, check=True, capture_output=True, text=True).stdout.strip()
    if current_commit != expected_commit or status:
        raise ValueError("Current repository is not the clean packaged commit")
    if manifest.get("expected_git_commit") != expected_commit:
        raise ValueError("Upload manifest commit mismatch")
    if sha256_file(notebook_path) != manifest["notebook"]["sha256"]:
        raise ValueError("Notebook package hash mismatch")
    dataset = package / "frozen_v22_dataset"
    if {path.name for path in dataset.iterdir()} != DATASET_FILES:
        raise ValueError("Frozen-v22 dataset allowlist mismatch")
    hashes = json.loads((dataset / "hashes.json").read_text(encoding="utf-8"))
    if set(hashes) != DATASET_FILES - {"hashes.json"}:
        raise ValueError("Frozen-v22 hashes.json coverage mismatch")
    for name, digest in hashes.items():
        if sha256_file(dataset / name) != digest:
            raise ValueError(f"Frozen-v22 declared hash mismatch: {name}")
    dataset_metadata = json.loads((dataset / "dataset-metadata.json").read_text(encoding="utf-8"))
    if dataset_metadata.get("id") != DATASET_SLUG:
        raise ValueError("Private dataset slug mismatch")
    for path in [notebook_path, *dataset.iterdir()]:
        if path.suffix.lower() not in (".npy", ".csv"):
            _scan_sensitive(str(path), path.read_bytes())
    proof = verify_v22(dataset, v20=args.v20, v21=args.v21,
        train_metadata=args.train_metadata, test_metadata=args.test_metadata, template=args.template)
    zip_path = package / "frozen_v22_dataset.zip"
    if sha256_file(zip_path) != manifest["dataset_zip"]["sha256"]:
        raise ValueError("Frozen-v22 ZIP hash mismatch")
    with zipfile.ZipFile(zip_path) as bundle_zip:
        if set(bundle_zip.namelist()) != DATASET_FILES:
            raise ValueError("Frozen-v22 ZIP allowlist mismatch")
        for name in bundle_zip.namelist():
            if hashlib.sha256(bundle_zip.read(name)).hexdigest() != sha256_file(dataset / name):
                raise ValueError(f"Frozen-v22 ZIP content mismatch: {name}")
    sums = {}
    for line in (package / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for name, digest in manifest["sha256"].items():
        if sums.get(name) != digest or sha256_file(package / name) != digest:
            raise ValueError(f"Delivery hash mismatch: {name}")
    launcher = files["scripts/launch_diverse_po_v23.py"].decode("utf-8")
    if "MAX_SECONDS = int(10.5 * 3600)" not in launcher or any(item.split("/")[-2] not in launcher for item in REQUIRED_INPUTS[1:]):
        raise ValueError("Embedded launcher input/runtime contract mismatch")
    return {"PREFLIGHT_OK": True, "expected_git_commit": expected_commit,
            "notebook_code_cells_compiled": True, "unique_cell_ids": True,
            "embedded_source_files": len(files), "dataset_files": len(DATASET_FILES),
            "frozen_v22_exact_rank_parity": proof["parity"]["exact_rank_probability_parity"],
            "official_v22_submission_sha256": proof["official_submission_sha256"],
            "required_inputs": REQUIRED_INPUTS, "max_total_hours": 10.5}


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--package", type=Path, default=repository / "artifacts/manual_upload_v23")
    parser.add_argument("--v20", type=Path, default=repository / "artifacts/v20_frozen_bundle_v21")
    parser.add_argument("--v21", type=Path, default=repository / "artifacts/v21_frozen_bundle")
    parser.add_argument("--train-metadata", type=Path, default=repository / "artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv")
    parser.add_argument("--test-metadata", type=Path, default=repository / "artifacts/v20_frozen/raw/GLC25_PA_metadata_test.csv")
    parser.add_argument("--template", type=Path, default=repository / "artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args), indent=2))
    except Exception as error:
        print(json.dumps({"PREFLIGHT_OK": False, "reason": str(error)}, indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
