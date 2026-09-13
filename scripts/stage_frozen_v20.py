"""Build a compact, verified v20 input bundle locally; never calls Kaggle.

Only probabilities and small provenance artifacts are copied. The numerical
blend preserves the original float32 arithmetic, with no model reconstruction,
new training, rounded probabilities or reassessment of the old v20 audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from scripts.ood_po_protocol import frozen_v20_blend
from scripts.prepare_environmental_challenger import spatial_partitions
from scripts.run_environmental_challenger import top_rank


MODELS = ("reference_2025", "challenger_2025", "challenger_3407")
KERNEL = "con1los/geolifeclef-risk-aware-sdm-phase-1"
COPIES = {"data_manifest.json": "v20_data_manifest.json", "frozen_policy.json": "v20_frozen_policy.json",
          "GLC25_PA_submission.csv": "v20_original_submission.csv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(ids: np.ndarray) -> str:
    """v20 stored the original row order, not sorted survey IDs."""
    return hashlib.sha256(np.asarray(ids).astype("<i8").tobytes()).hexdigest()


def _ids(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame.surveyId, errors="raise").to_numpy()
    if not np.isfinite(values).all() or np.any(values != values.astype(np.int64)):
        raise ValueError("Finite integer survey IDs required")
    return values.astype(np.int64)


def _original_policy(policy: dict) -> None:
    selected = policy.get("selected", {})
    if selected.get("challenger_weight") != .75 or selected.get("policy") != {"kind": "top_k", "k": 20}:
        raise ValueError("Source policy is not the frozen original v20 75/25 top20 blend")
    if policy.get("further_training") is not False:
        raise ValueError("Source policy does not establish frozen model weights")


def submission_bytes(probabilities: np.ndarray, species: np.ndarray, test_ids: np.ndarray,
                     template_ids: np.ndarray) -> bytes:
    """Reproduce v20 CSV ordering, top50 ranking path, top20 and CRLF bytes."""
    species, test_ids, template_ids = map(np.asarray, (species, test_ids, template_ids))
    if probabilities.shape != (len(test_ids), len(species)):
        raise ValueError("Test probability shape does not match IDs/vocabulary")
    if len(np.unique(test_ids)) != len(test_ids) or len(np.unique(template_ids)) != len(template_ids):
        raise ValueError("Duplicate test/template survey IDs")
    if len(test_ids) != len(template_ids) or set(test_ids) != set(template_ids):
        raise ValueError("Test/template survey IDs differ")
    if len(species) < 20 or len(np.unique(species)) != len(species):
        raise ValueError("Invalid species vocabulary")
    lookup = {int(sid): index for index, sid in enumerate(test_ids)}
    # Batch only the row dimension; top_rank is row independent and remains exact.
    ranks = np.empty((len(test_ids), 20), dtype=np.int64)
    for begin in range(0, len(test_ids), 512):
        values = np.asarray(probabilities[begin:begin + 512])
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Invalid test probabilities")
        ranks[begin:begin + len(values)] = top_rank(values)[0][:, :20]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["surveyId", "predictions"])
    for sid in template_ids:
        predictions = species[ranks[lookup[int(sid)]]]
        writer.writerow([int(sid), " ".join(map(str, predictions))])
    return stream.getvalue().encode("utf-8")


def _combine(source: Path, output: Path, split: str, shape: tuple[int, int]) -> list[dict]:
    paths = [source / f"{model}_{split}_probabilities.npy" for model in MODELS]
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
    for path, values in zip(paths, arrays):
        if values.shape != shape or values.dtype != np.float16:
            raise ValueError(f"Original v20 probability shape/dtype mismatch: {path.name}")
    combined = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=shape)
    for begin in range(0, shape[0], 512):
        combined[begin:begin + 512] = frozen_v20_blend(*(values[begin:begin + 512] for values in arrays))
    combined.flush()
    del combined
    return [{"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
             "shape": list(values.shape), "dtype": str(values.dtype)} for path, values in zip(paths, arrays)]


def stage_bundle(source_dir: Path, output_dir: Path, test_metadata: Path, template: Path,
                 train_metadata: Path | None = None, *, expected_species: int = 5016,
                 expected_test_samples: int = 14784) -> dict:
    """Create a fresh upload-ready folder, failing before provenance on mismatch.

    The CLI fixes the actual competition dimensions. Small alternative expected
    dimensions are an injectable boundary for synthetic integrity tests only.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    manifest = json.loads((source_dir / "data_manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((source_dir / "frozen_policy.json").read_text(encoding="utf-8"))
    _original_policy(policy)
    if manifest.get("protocol") != "environmental_challenger_v20":
        raise ValueError("Unexpected source data protocol")
    if manifest.get("species") != expected_species or manifest.get("test_samples") != expected_test_samples:
        raise ValueError("Source dimensions differ from the registered competition")
    species = np.asarray(manifest["species_ids"], dtype=np.int64)
    if len(species) != expected_species or not np.array_equal(species, np.unique(species)):
        raise ValueError("Original vocabulary must be complete, unique and sorted")
    expected_calibration_hash = manifest["partition_ids_sha256"]["policy_calibration"]
    if len(expected_calibration_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_calibration_hash):
        raise ValueError("Invalid original ordered calibration ID hash")
    calibration_rows = int(manifest["partition_counts"]["policy_calibration"])
    if calibration_rows <= 0:
        raise ValueError("Empty source calibration partition")
    test_frame = pd.read_csv(test_metadata, usecols=["surveyId"]).drop_duplicates("surveyId")
    test_ids = _ids(test_frame)
    template_ids = _ids(pd.read_csv(template, usecols=["surveyId"]))
    if len(test_ids) != expected_test_samples or len(template_ids) != expected_test_samples:
        raise ValueError("Official test/template row counts differ from original v20")
    calibration_ids = None
    if train_metadata is not None:
        # Reading coordinates/IDs suffices; no labels are loaded or evaluated.
        train_frame = pd.read_csv(train_metadata, usecols=["surveyId", "lat", "lon", "country"]).drop_duplicates("surveyId")
        calibration_ids = _ids(train_frame)[spatial_partitions(train_frame) == 2]
        if len(calibration_ids) != calibration_rows or ordered_ids_sha256(calibration_ids) != expected_calibration_hash:
            raise ValueError("Original calibration ID order/hash does not match competition metadata")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Bundle output must be a fresh empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    components = []
    for split, rows in (("calibration", calibration_rows), ("test", expected_test_samples)):
        components.extend(_combine(source_dir, output_dir / f"v20_{split}_probabilities.npy", split, (rows, expected_species)))
    test_probability = np.load(output_dir / "v20_test_probabilities.npy", mmap_mode="r", allow_pickle=False)
    generated = submission_bytes(test_probability, species, test_ids, template_ids)
    original = (source_dir / "GLC25_PA_submission.csv").read_bytes()
    generated_rows = list(csv.reader(io.StringIO(generated.decode("utf-8"))))
    original_rows = list(csv.reader(io.StringIO(original.decode("utf-8-sig"))))
    if generated_rows != original_rows:
        raise ValueError("Recomputed v20 top20 predictions/order differ from original submission")
    if generated != original:
        raise ValueError("Recomputed v20 CSV bytes differ from original submission")
    np.save(output_dir / "species_ids.npy", species, allow_pickle=False)
    np.save(output_dir / "test_ids.npy", test_ids, allow_pickle=False)
    if calibration_ids is not None:
        np.save(output_dir / "calibration_ids.npy", calibration_ids, allow_pickle=False)
    for original_name, destination in COPIES.items():
        shutil.copyfile(source_dir / original_name, output_dir / destination)
    provenance = {
        "format": "geolifeclef_frozen_v20_bundle_v1", "source_kernel": KERNEL, "source_kernel_version": 20,
        "source_commit": "1a2528b", "source_protocol": manifest["protocol"],
        "mixture": {"reference_weight": .25, "challenger_weight": .75, "challenger_seed_mean": [2025, 3407],
                    "arithmetic_dtype": "float32", "saved_dtype": "float32", "policy": {"kind": "top_k", "k": 20}},
        "source_components": components,
        "source_metadata": {name: {"sha256": sha256_file(source_dir / name), "bytes": (source_dir / name).stat().st_size}
                            for name in COPIES},
        "calibration": {"rows": calibration_rows, "ordered_ids_sha256": expected_calibration_hash,
                        "id_order_verified_locally": calibration_ids is not None,
                        "runner_must_verify_against_competition_pa_metadata": True,
                        "original_partition": "policy_calibration", "labels_packaged": False},
        "test": {"rows": expected_test_samples, "ordered_ids_sha256": ordered_ids_sha256(test_ids),
                 "template_ordered_ids_sha256": ordered_ids_sha256(template_ids),
                 "prediction_parity": True, "submission_byte_parity": True,
                 "original_submission_sha256": hashlib.sha256(original).hexdigest()},
        "species": expected_species, "species_ids_sha256": ordered_ids_sha256(species),
        "checkpoint_files_packaged": False, "v20_retraining": False, "external_data_or_weights": False,
        "old_audit_labels_loaded": False, "files": {},
    }
    for path in sorted(output_dir.iterdir()):
        provenance["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    # Presence of provenance marks successful verification. A failed staging
    # directory without this file is never a valid upload input.
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def verify_bundle(bundle_dir: Path, *, calibration_ids: np.ndarray | None = None,
                  test_ids: np.ndarray | None = None) -> dict:
    """Validate mounted immutable files and optional independently rebuilt IDs."""
    bundle_dir = Path(bundle_dir)
    provenance = json.loads((bundle_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("format") != "geolifeclef_frozen_v20_bundle_v1" or provenance.get("source_kernel_version") != 20:
        raise ValueError("Unexpected frozen bundle format/source version")
    if provenance.get("source_kernel") != KERNEL:
        raise ValueError("Unexpected frozen bundle source kernel")
    required = {"v20_calibration_probabilities.npy", "v20_test_probabilities.npy", "species_ids.npy", "test_ids.npy", *COPIES.values()}
    if not required.issubset(provenance.get("files", {})):
        raise ValueError("Frozen bundle is missing mandatory file records")
    for name, record in provenance["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Invalid bundle filename")
        path = bundle_dir / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen bundle file integrity failure: {name}")
    if calibration_ids is not None and ordered_ids_sha256(calibration_ids) != provenance["calibration"]["ordered_ids_sha256"]:
        raise ValueError("Mounted frozen calibration ID order differs from competition metadata")
    if test_ids is not None and ordered_ids_sha256(test_ids) != provenance["test"]["ordered_ids_sha256"]:
        raise ValueError("Mounted frozen test ID order differs from competition metadata")
    for split in ("calibration", "test"):
        array = np.load(bundle_dir / f"v20_{split}_probabilities.npy", mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (provenance[split]["rows"], provenance["species"]):
            raise ValueError("Mounted frozen probability dimensions/dtype differ from provenance")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/v20_frozen"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v20_frozen_bundle"))
    parser.add_argument("--test-metadata", type=Path, default=Path("artifacts/v20_frozen/raw/GLC25_PA_metadata_test.csv"))
    parser.add_argument("--template", type=Path, default=Path("artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv"))
    parser.add_argument("--train-metadata", type=Path)
    args = parser.parse_args()
    report = stage_bundle(args.source_dir, args.output_dir, args.test_metadata, args.template, args.train_metadata)
    print(json.dumps({"bundle": str(args.output_dir), "source_kernel_version": report["source_kernel_version"],
                      "files": len(report["files"]), "species": report["species"],
                      "calibration_id_order_verified": report["calibration"]["id_order_verified_locally"],
                      "test_prediction_parity": report["test"]["prediction_parity"],
                      "submission_byte_parity": report["test"]["submission_byte_parity"],
                      "original_submission_sha256": report["test"]["original_submission_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
