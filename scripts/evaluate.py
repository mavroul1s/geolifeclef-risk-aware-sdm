#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from geolifeclef.data import CanonicalNPZDataset, collate_modalities
from geolifeclef.metrics import brier_score, fit_adaptive_thresholds, multilabel_f1, multilabel_top_k_f1
from geolifeclef.models import TemporalCNN
from geolifeclef.training import predict
from geolifeclef.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--split", required=True, type=Path); parser.add_argument("--calibration-split", type=Path); parser.add_argument("--modality", default="landsat"); parser.add_argument("--channels", type=int, default=32); parser.add_argument("--top-k", type=int); args = parser.parse_args()
    dataset = CanonicalNPZDataset(args.split, args.modality); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = TemporalCNN(dataset.arrays[args.modality].shape[-1], dataset.summary.species, args.channels).to(device); model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)["model_state"])
    targets, probabilities = predict(model, DataLoader(dataset, batch_size=256, collate_fn=collate_modalities), device, args.modality); report = {"sample_f1_fixed_0.5": multilabel_f1(targets, probabilities, average="samples"), "micro_f1_fixed_0.5": multilabel_f1(targets, probabilities), "macro_f1_fixed_0.5": multilabel_f1(targets, probabilities, average="macro"), "brier": brier_score(targets, probabilities)}
    if args.top_k:
        report.update({"top_k": args.top_k, "sample_f1_top_k": multilabel_top_k_f1(targets, probabilities, args.top_k, average="samples"), "micro_f1_top_k": multilabel_top_k_f1(targets, probabilities, args.top_k), "macro_f1_top_k": multilabel_top_k_f1(targets, probabilities, args.top_k, average="macro")})
    if args.calibration_split:
        calibration = CanonicalNPZDataset(args.calibration_split, args.modality); cal_targets, cal_probabilities = predict(model, DataLoader(calibration, batch_size=256, collate_fn=collate_modalities), device, args.modality); thresholds = fit_adaptive_thresholds(cal_targets, cal_probabilities); report["sample_f1_adaptive_threshold"] = multilabel_f1(targets, probabilities, thresholds, average="samples"); report["micro_f1_adaptive_threshold"] = multilabel_f1(targets, probabilities, thresholds); np.save(args.checkpoint.parent / "adaptive_thresholds.npy", thresholds)
    write_json(args.checkpoint.parent / "evaluation.json", report); print(report)


if __name__ == "__main__": main()
