"""Verify and stage the minimal private frozen-v22 input; never uploads it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from scripts.ood_po_protocol import mix_probabilities, nearest_training_support
from scripts.prepare_environmental_challenger import spatial_partitions
from scripts.run_retained_po import deployment_partitions as v22_deployment_partitions
from scripts.stage_frozen_v20 import sha256_file, verify_bundle, verify_original_ties
from scripts.stage_frozen_v21 import verify_v21
from scripts.v22_protocol import mix as v22_mix


KERNEL = "con1los/geolifeclef-risk-aware-sdm-phase-1"
SOURCE_VERSION = 22
SOURCE_COMMIT = "519ae6cecadb30e4339d6fcf5354c4f0ed19f479"
OFFICIAL_CSV_SHA256 = "fe4eb33353936c7bc66b6c7451b89b838ee0c8ad55d9440d76da86778b5646de"
DATASET_SLUG = "con1los/geolifeclef-v22-frozen-control"
EXPERT_FILES = ("retained_po_calibration.npy", "retained_po_test.npy")
COPIES = (*EXPERT_FILES, "frozen_policies.json", "GLC25_PA_submission.csv", "v22_report.json")
V21_POLICY = {"alpha": 0.025, "gate": "pa_distance", "k": 20}


def _source_files(source: Path, review: Path) -> dict[str, Path]:
    return {
        "retained_po_calibration.npy": source / "retained_po_calibration.npy",
        "retained_po_test.npy": source / "retained_po_test.npy",
        "frozen_policies.json": review / "frozen_policies.json",
        "GLC25_PA_submission.csv": review / "GLC25_PA_submission.csv",
        "v22_report.json": review / "v22_report.json",
    }


def reconstruct(v20: Path, v21: Path, v22: Path, train_rows: pd.DataFrame,
                test_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verify_bundle(v20, test_ids=test_rows.surveyId.to_numpy())
    verify_v21(v21)
    original = spatial_partitions(train_rows)
    species = np.load(v20 / "species_ids.npy", allow_pickle=False)
    distance = nearest_training_support(train_rows.loc[original == 0], train_rows)["distance_km"]
    test_distance = nearest_training_support(train_rows.loc[original == 0], test_rows)["distance_km"]
    policies = json.loads((v22 / "frozen_policies.json").read_text(encoding="utf-8"))
    policy = policies["deployment"]["retained_po"]["selected"]
    if policy != {"alpha": 0.025, "gate": "uniform", "k": 20, "calibration_f1": policy.get("calibration_f1")}:
        raise ValueError("Unexpected frozen v22 deployment policy")
    v22_production = v22_deployment_partitions(train_rows)
    calibration_rows = np.flatnonzero(v22_production == 2)
    v21_calibration = mix_probabilities(
        np.load(v20 / "v20_calibration_probabilities.npy", mmap_mode="r"),
        np.load(v21 / "deployment_po_calibration.npy", mmap_mode="r"),
        distance[original == 2], V21_POLICY,
    )
    old_calibration_rows = np.flatnonzero(original == 2)
    positions = {row: index for index, row in enumerate(old_calibration_rows)}
    calibration_positions = [positions[row] for row in calibration_rows]
    v22_calibration = v22_mix(
        v21_calibration[calibration_positions],
        np.load(v22 / "retained_po_calibration.npy", mmap_mode="r"),
        distance[calibration_rows], policy,
    )
    v21_test = mix_probabilities(
        np.load(v20 / "v20_test_probabilities.npy", mmap_mode="r"),
        np.load(v21 / "deployment_po_test.npy", mmap_mode="r"),
        test_distance, V21_POLICY,
    )
    v22_test = v22_mix(
        v21_test,
        np.load(v22 / "retained_po_test.npy", mmap_mode="r"),
        test_distance, policy,
    )
    return species, v22_calibration, v22_test


def verify_v22(directory: Path, *, v20: Path | None = None, v21: Path | None = None,
               train_metadata: Path | None = None, test_metadata: Path | None = None,
               template: Path | None = None) -> dict:
    directory = Path(directory)
    proof = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if proof.get("format") != "geolifeclef_frozen_v22_bundle_v1":
        raise ValueError("Invalid frozen-v22 bundle format")
    if proof.get("source_kernel") != KERNEL or proof.get("source_kernel_version") != SOURCE_VERSION:
        raise ValueError("Wrong frozen-v22 source kernel/version")
    if proof.get("source_commit") != SOURCE_COMMIT or proof.get("official_submission_sha256") != OFFICIAL_CSV_SHA256:
        raise ValueError("Wrong officially evaluated frozen-v22 source")
    if set(proof.get("files", {})) != set(COPIES):
        raise ValueError("Unexpected frozen-v22 payload")
    allowed = set(COPIES) | {"provenance.json", "dataset-metadata.json", "hashes.json"}
    if any(not path.is_file() or path.name not in allowed for path in directory.iterdir()):
        raise ValueError("Unexpected file in frozen-v22 bundle")
    for name, record in proof["files"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen-v22 hash mismatch: {name}")
    shapes = {"retained_po_calibration.npy": (4807, 5016), "retained_po_test.npy": (14784, 5016)}
    for name, shape in shapes.items():
        path = directory / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != np.float16 or not np.isfinite(array).all():
            raise ValueError(f"Invalid frozen-v22 probability artifact: {name}")
        if path.stat().st_size != array.offset + array.nbytes:
            raise ValueError(f"Frozen-v22 NPY has trailing or truncated bytes: {name}")
    if sha256_file(directory / "GLC25_PA_submission.csv") != OFFICIAL_CSV_SHA256:
        raise ValueError("Frozen-v22 CSV is not the official v22 submission")
    if all(value is not None for value in (v20, v21, train_metadata, test_metadata, template)):
        train_rows = pd.read_csv(train_metadata).drop_duplicates("surveyId").reset_index(drop=True)
        test_rows = pd.read_csv(test_metadata).drop_duplicates("surveyId").reset_index(drop=True)
        species, _, probabilities = reconstruct(Path(v20), Path(v21), directory, train_rows, test_rows)
        template_ids = pd.read_csv(template).surveyId.to_numpy()
        parity = verify_original_ties(probabilities, species, test_rows.surveyId.to_numpy(), template_ids,
                                      (directory / "GLC25_PA_submission.csv").read_bytes())
        if parity != proof.get("parity"):
            raise ValueError("Frozen-v22 parity proof changed")
    return proof


def build(source: Path, review: Path, v20: Path, v21: Path, train_metadata: Path,
          test_metadata: Path, template: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Frozen-v22 output must be a fresh empty directory")
    output.mkdir(parents=True, exist_ok=True)
    sources = _source_files(Path(source), Path(review))
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen-v22 source artifact: {name}")
        shutil.copyfile(path, output / name)
    train_rows = pd.read_csv(train_metadata).drop_duplicates("surveyId").reset_index(drop=True)
    test_rows = pd.read_csv(test_metadata).drop_duplicates("surveyId").reset_index(drop=True)
    species, calibration, probabilities = reconstruct(v20, v21, output, train_rows, test_rows)
    if calibration.shape != (4807, 5016):
        raise ValueError("Frozen-v22 calibration reconstruction failed")
    parity = verify_original_ties(
        probabilities, species, test_rows.surveyId.to_numpy(),
        pd.read_csv(template).surveyId.to_numpy(), (output / "GLC25_PA_submission.csv").read_bytes(),
    )
    report = json.loads((output / "v22_report.json").read_text(encoding="utf-8"))
    if report.get("source_commit") != SOURCE_COMMIT or report.get("status") != "complete":
        raise ValueError("Frozen-v22 report provenance mismatch")
    proof = {
        "format": "geolifeclef_frozen_v22_bundle_v1",
        "source_kernel": KERNEL,
        "source_kernel_version": SOURCE_VERSION,
        "source_commit": SOURCE_COMMIT,
        "official_submission_ref": "56233226",
        "official_submission_sha256": OFFICIAL_CSV_SHA256,
        "calibration_rows": 4807,
        "test_rows": 14784,
        "species": 5016,
        "v20_v21_inputs_duplicated": False,
        "pa_labels_packaged": False,
        "external_data_or_weights": False,
        "checkpoint_files_packaged": False,
        "parity": parity,
        "files": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                  for name in COPIES},
    }
    (output / "provenance.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": DATASET_SLUG,
        "title": "GeoLifeCLEF frozen v22 control",
        "licenses": [{"name": "other"}],
        "description": "Private competition-derived frozen v22 residual probabilities and provenance. Competition terms apply; no redistribution permission.",
    }
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    hashes = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.name != "hashes.json"}
    (output / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    verify_v22(output, v20=v20, v21=v21, train_metadata=train_metadata,
               test_metadata=test_metadata, template=template)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/v22_frozen_source"))
    parser.add_argument("--review", type=Path, default=Path("artifacts/v22_review"))
    parser.add_argument("--v20", type=Path, default=Path("artifacts/v20_frozen_bundle_v21"))
    parser.add_argument("--v21", type=Path, default=Path("artifacts/v21_frozen_bundle"))
    parser.add_argument("--train-metadata", type=Path, default=Path("artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv"))
    parser.add_argument("--test-metadata", type=Path, default=Path("artifacts/v20_frozen/raw/GLC25_PA_metadata_test.csv"))
    parser.add_argument("--template", type=Path, default=Path("artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/manual_upload_v23/frozen_v22_dataset"))
    args = parser.parse_args()
    proof = build(args.source, args.review, args.v20, args.v21, args.train_metadata,
                  args.test_metadata, args.template, args.output)
    print(json.dumps({"verified": True, "dataset": DATASET_SLUG, "files": len(proof["files"])}))


if __name__ == "__main__":
    main()
