"""Reproduce the released nonaggregated SAGE baseline on the fixed CPU reference."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch
import yaml

from audit_ausplots_metadata import ROOT, sha
from geolifeclef.range_io import BoundedHTTPFile
from geolifeclef.sage_inference import load_network, normalization_from_source, normalize
from retrieve_sage_e1 import fetch_member


MODEL = "baseline_deepsdm_s0"
DATA = ROOT / "data/ausplots_followup/model_reference"
OUTPUT = ROOT / "results/ausplots_followup/second_model_reference.json"
BRIDGE = ROOT / "data/neon_bridge"
EXTERNAL = ROOT / "data/external_replication_e1"
MODEL_RELEASE = EXTERNAL / "zenodo_21335338.json"
CHECKPOINT_INDEX = BRIDGE / "checkpoint_index.json"
PREDICTION_INDEX = ROOT / "data/independent_data_audit/sage_predictions_tar_index.json"
CONFIGS = ROOT / "data/neon_confirmation_feasibility/sage_configs.tar"
NORMALIZATION = BRIDGE / "published_source/src/data_modules/datamodule.py"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(4)
    record = json.loads(MODEL_RELEASE.read_text())
    checkpoint_info = next(m for m in json.loads(CHECKPOINT_INDEX.read_text())["members"]
                           if m["name"] == f"{MODEL}/checkpoints/best.ckpt")
    if not 0 < checkpoint_info["size"] <= 50 * 1024**2:
        raise ValueError("Unexpected checkpoint transfer size")
    checkpoint = DATA / "best.ckpt"
    retrieval = fetch_member(record, "checkpoints.tar", checkpoint_info, checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt checkpoint ZIP")
    with tarfile.open(CONFIGS) as archive:
        config = yaml.safe_load(archive.extractfile(f"{MODEL}/config.yaml"))
    arguments = config["data"]["init_args"]
    if arguments["aggregated"] or config["model"]["init_args"]["net"]["init_args"]["use_location"]:
        raise ValueError("Unexpected published model input or training representation")
    subset = arguments["predictors_subset"]
    ids_path = EXTERNAL / "test_indices.npy"
    ids = np.load(ids_path)[:128]
    paths = [ids_path, checkpoint, CHECKPOINT_INDEX, PREDICTION_INDEX, MODEL_RELEASE, CONFIGS,
             NORMALIZATION, Path(__file__), ROOT / "src/geolifeclef/sage_inference.py",
             ROOT / "docs/ausplots_followup_protocol.md"]
    columns = []
    for group, indices in subset.items():
        if group == "location":
            continue
        path = BRIDGE / "reference_predictors" / f"eval_{group}.h5"
        with h5py.File(path) as handle:
            columns.append(handle["predictor_data"][:][ids][:, indices])
        paths.append(path)
    features = np.concatenate(columns, axis=1)
    means, stds = normalization_from_source(NORMALIZATION, subset)
    model = load_network(checkpoint, config)
    with torch.inference_mode():
        actual = torch.sigmoid(model(normalize(features, means, stds))).numpy()

    info = next(f for f in record["files"] if f["key"] == "predictions.tar")
    member = next(m for m in json.loads(PREDICTION_INDEX.read_text())["members"]
                  if m["name"] == f"{MODEL}/test_predictions.h5")
    remote = BoundedHTTPFile(info["links"]["self"], member["data_offset"], member["size"],
                             budget=32 * 1024**2, cache_dir=DATA / "published_ranges")
    with remote, h5py.File(remote, "r") as handle:
        expected = handle["probabilities"][:128]
    if expected.shape != actual.shape or actual.shape != (128, 5771):
        raise ValueError("Unexpected probability shape")
    error = np.abs(actual.astype(np.float64) - expected)
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite probability error")
    maximum, mean = float(error.max()), float(error.mean())
    passed = maximum <= 1e-5 and mean <= 1e-6
    np.savez_compressed(DATA / "reference.npz", ids=ids, raw=features, means=means, stds=stds,
                        published=expected, cpu=actual)
    receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
               "architecture": type(model).__name__, "training_occurrences_aggregated": False,
               "probabilities_checked": int(actual.size), "device": "CPU", "torch_version": torch.__version__,
               "maximum_absolute_error": maximum, "mean_absolute_error": mean, "reference_passed": passed,
               "maximum_absolute_error_limit": 1e-5, "mean_absolute_error_limit": 1e-6,
               "published_array_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
               "reference_bundle_sha256": sha(DATA / "reference.npz"), "checkpoint": retrieval,
               "new_prediction_range_bytes": remote.fetched_bytes, "prediction_range_requests": remote.requests,
               "outcome_labels_read": False, "training_coordinate_screen_complete": False,
               "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(f"{MODEL}: {'pass' if passed else 'fail'} (max {maximum:.8g}, mean {mean:.8g})", flush=True)


if __name__ == "__main__":
    main()
