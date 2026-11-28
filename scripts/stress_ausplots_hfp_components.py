"""Repeat the fixed SAGE stress test with HFP2009 kept and its eight components missing.

The recovered aggregate is the first Human Footprint feature. Use the same
AEKOS cells, native masks, model checkpoints, normalization and top-k endpoints
as the all-nine omission run. Only the omitted column set changes. Comparing
both passes on released inputs uses no AusPlots outcomes.
"""

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

from stress_ausplots_human_footprint import (BRIDGE, CONFIGS, DATA, FULL_MODEL, MASKS, MODEL,
                                           NORMALIZATION, OUTPUT, ROOT, SAGE_PLOTS, compare,
                                           load_network, normalization_from_source, read_predictors, sha)
from check_ausplots_predictors import published_layer_names
from validate_ausplots_followup import predictor_check, read_receipt


def main():
    previous = OUTPUT / "human_footprint_stress.json"
    destination = OUTPUT / "human_footprint_components_stress.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = read_receipt(previous)
    _, checked, all_finite = predictor_check("predictors_with_wcs")
    if checked.sum() != 44 or not checked[43] or checked[44:].any() or all_finite != 0:
        raise ValueError("Current availability differs from the eight-component omission scenario")
    human = published_layer_names()["human_footprint"]
    if human[0] != "HFP2009.tif":
        raise ValueError("Published HFP order changed")
    torch.set_num_threads(4)
    metadata = pd.read_csv(SAGE_PLOTS, usecols=["GIVD_ID", "cell_row"])
    cells = np.unique(metadata.loc[metadata.GIVD_ID.eq("AU-AU-002"), "cell_row"].to_numpy(dtype=np.int64))
    if len(cells) != report["AEKOS_cells"]:
        raise ValueError("Reference stress-test cells differ")
    with h5py.File(MASKS) as handle:
        ends, indices = handle["cumulative_lengths"][:], handle["species_index"][:]
    checkpoints = {FULL_MODEL: BRIDGE / f"checkpoints/{FULL_MODEL}/best.ckpt", MODEL: DATA / "best.ckpt"}
    runs = {}
    with tarfile.open(CONFIGS) as archive:
        for name, checkpoint in checkpoints.items():
            config = yaml.safe_load(archive.extractfile(f"{name}/config.yaml"))
            subset = config["data"]["init_args"]["predictors_subset"]
            features, columns, _ = read_predictors(subset, cells)
            if subset["human_footprint"] != list(range(9)):
                raise ValueError("Unexpected Human Footprint model feature order")
            means, stds = normalization_from_source(NORMALIZATION, subset)
            model = load_network(checkpoint, config)
            runs[name] = compare(model, features, columns[1:], means, stds, cells, ends, indices)
            print(name, json.dumps(runs[name]["top_k"][10]), flush=True)
    sources = [Path(__file__), previous, OUTPUT / "predictors_with_wcs/predictors.json"]
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "AEKOS_cells": len(cells),
              "retained": human[0], "omitted": human[1:], "stress_development_only": True,
              "target_labels_read": False, "AusPlots_predictions_generated": False, "runs": runs,
              "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources}}
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
