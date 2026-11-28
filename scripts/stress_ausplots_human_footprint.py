"""Measure prediction-set drift from omitting the nine Human Footprint inputs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from audit_ausplots_metadata import ROOT, sha
from check_ausplots_predictors import BRIDGE, OUTPUT, SAGE_PLOTS
from check_ausplots_second_model import CONFIGS, DATA, MODEL, NORMALIZATION
from geolifeclef.sage_inference import load_network, normalization_from_source, normalize


FULL_MODEL = "aggregation_s0"
MASKS = ROOT / "data/external_replication_e1/range_masks_1km/eval_range_mask.h5"
DEPTHS = (5, 10, 20)
NCLASSES = 5771
BATCH = 128


def read_predictors(subset, cells):
    arrays, human_columns, paths = [], [], []
    offset = 0
    for group, columns in subset.items():
        if group == "location":
            continue
        path = BRIDGE / "reference_predictors" / f"eval_{group}.h5"
        with h5py.File(path) as handle:
            arrays.append(handle["predictor_data"][:][cells][:, columns])
        paths.append(path)
        if group == "human_footprint":
            human_columns.extend(range(offset, offset + len(columns)))
        offset += len(columns)
    if (offset, len(human_columns)) != (52, 9):
        raise ValueError("The predictor contract changed")
    return np.concatenate(arrays, axis=1), human_columns, paths


def native_mask(cells, ends, indices):
    mask = np.zeros((len(cells), NCLASSES), dtype=bool)
    for row, cell in enumerate(cells):
        mask[row, indices[ends[cell]:ends[cell + 1]]] = True
    return mask


def top_overlap(mask, full, ablated):
    order_full = np.argsort(np.where(mask, -full, np.inf), axis=1, kind="stable")
    order_ablated = np.argsort(np.where(mask, -ablated, np.inf), axis=1, kind="stable")
    size = mask.sum(axis=1)
    scores = {}
    for k in DEPTHS:
        row_size = np.minimum(k, size)
        intersection = np.array([len(np.intersect1d(a[:n], b[:n], assume_unique=True))
                                 for a, b, n in zip(order_full[:, :k], order_ablated[:, :k], row_size)])
        scores[k] = intersection / (2 * row_size - intersection)
    return scores


def compare(model, features, human, means, stds, cells, ends, indices):
    masked_features = features.copy()
    masked_features[:, human] = np.nan
    overlaps = {k: [] for k in DEPTHS}
    seen = 0
    total = maximum = 0.0
    with torch.inference_mode():
        for start in range(0, len(cells), BATCH):
            batch = slice(start, start + BATCH)
            full = torch.sigmoid(model(normalize(features[batch], means, stds))).numpy()
            ablated = torch.sigmoid(model(normalize(masked_features[batch], means, stds))).numpy()
            ranges = native_mask(cells[batch], ends, indices)
            if not ranges.any(axis=1).all() or not np.isfinite(full).all() or not np.isfinite(ablated).all():
                raise ValueError("Missing native mask or invalid probability")
            delta = np.abs(full[ranges].astype(np.float64) - ablated[ranges])
            seen += len(delta)
            total += float(delta.sum())
            maximum = max(maximum, float(delta.max()))
            for k, scores in top_overlap(ranges, full, ablated).items():
                overlaps[k].append(scores)
    stats = {}
    for k, values in overlaps.items():
        scores = np.concatenate(values)
        stats[k] = {"changed_sets": int((scores < 1).sum()), "fraction_changed_sets": float((scores < 1).mean()),
                    "median_jaccard": float(np.median(scores)), "tenth_percentile_jaccard": float(np.quantile(scores, .1))}
    return {"top_k": stats, "native_probability_entries": seen,
            "mean_absolute_probability_change": total / seen, "maximum_absolute_probability_change": maximum}


def main():
    destination = OUTPUT / "human_footprint_stress.json"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(4)
    metadata = pd.read_csv(SAGE_PLOTS, usecols=["GIVD_ID", "cell_row"])
    cells = np.unique(metadata.loc[metadata.GIVD_ID.eq("AU-AU-002"), "cell_row"].to_numpy(dtype=np.int64))
    if len(cells) != 7530:
        raise ValueError("AEKOS released cell count differs")
    with h5py.File(MASKS) as handle:
        ends, indices = handle["cumulative_lengths"][:], handle["species_index"][:]
    if (ends[0] != 0 or ends[-1] != len(indices) or (np.diff(ends) < 0).any()
            or cells[-1] >= len(ends) - 1 or (indices < 0).any() or (indices >= NCLASSES).any()):
        raise ValueError("Invalid published native-mask coordinates")
    paths = [SAGE_PLOTS, MASKS, NORMALIZATION, CONFIGS, Path(__file__),
             ROOT / "docs/ausplots_predictor_stress_protocol.md"]
    runs = {}
    inputs = {FULL_MODEL: (BRIDGE / f"checkpoints/{FULL_MODEL}/best.ckpt",
                           ROOT / "results/neon_bridge/cpu_reference.json"),
              MODEL: (DATA / "best.ckpt", OUTPUT / "second_model_reference.json")}
    with tarfile.open(CONFIGS) as archive:
        for name, (checkpoint, receipt) in inputs.items():
            reference = json.loads(receipt.read_text())
            if name == FULL_MODEL:
                reference = next(run for run in reference["runs"] if run["run"] == name)
                expected = reference["checkpoint_sha256"]
            else:
                expected = reference["checkpoint"]["sha256"]
            if not reference["reference_passed"] or sha(checkpoint) != expected:
                raise ValueError("Frozen model reference failed")
            config = yaml.safe_load(archive.extractfile(f"{name}/config.yaml"))
            subset = config["data"]["init_args"]["predictors_subset"]
            features, human, feature_paths = read_predictors(subset, cells)
            means, stds = normalization_from_source(NORMALIZATION, subset)
            model = load_network(checkpoint, config)
            runs[name] = compare(model, features, human, means, stds, cells, ends, indices)
            paths.extend([checkpoint, receipt, *feature_paths])
            print(name, json.dumps(runs[name]["top_k"][10]), flush=True)
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "AEKOS_cells": len(cells),
              "stress_development_only": True, "target_labels_read": False, "AusPlots_predictions_generated": False,
              "runs": runs, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)}}
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
