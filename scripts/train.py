#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from geolifeclef.data import CanonicalNPZDataset, collate_modalities
from geolifeclef.metrics import multilabel_f1
from geolifeclef.models import TemporalCNN, count_trainable_parameters
from geolifeclef.training import train_temporal
from geolifeclef.utils import read_config, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); args = parser.parse_args()
    config = read_config(args.config); set_seed(int(config["seed"])); run_dir = Path("artifacts") / config["run_name"]; run_dir.mkdir(parents=True, exist_ok=True); shutil.copy2(args.config, run_dir / "config.yaml")
    modality = config["model"].get("modality"); train_set = CanonicalNPZDataset(config["data"]["train_path"], modality); val_set = CanonicalNPZDataset(config["data"]["val_path"], modality); labels = train_set.arrays["labels"]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["model"]["name"] == "frequency":
        probabilities = np.broadcast_to(labels.mean(0), val_set.arrays["labels"].shape); result = {"model": "frequency", "val_micro_f1": multilabel_f1(val_set.arrays["labels"], probabilities), "species": int(labels.shape[1]), "device": str(device)}; write_json(run_dir / "metrics.json", result); print(result); return
    modality = modality or "landsat"; model = TemporalCNN(train_set.arrays[modality].shape[-1], labels.shape[1], config["model"]["channels"], config["model"]["dropout"]).to(device)
    settings = config["training"]; train_loader = DataLoader(train_set, batch_size=settings["batch_size"], shuffle=True, num_workers=settings.get("num_workers", 0), collate_fn=collate_modalities); val_loader = DataLoader(val_set, batch_size=settings["batch_size"], shuffle=False, num_workers=settings.get("num_workers", 0), collate_fn=collate_modalities)
    positives = torch.as_tensor(labels.sum(axis=0), dtype=torch.float32); negatives = len(labels) - positives; max_weight = float(settings.get("max_pos_weight", 1.0)); pos_weight = torch.sqrt(negatives / positives.clamp_min(1)).clamp(1.0, max_weight) if max_weight > 1.0 else None
    result = train_temporal(model, train_loader, val_loader, device, modality, settings["epochs"], settings["lr"], settings.get("weight_decay", 0.), settings.get("mixed_precision", True), run_dir, settings.get("early_stopping_patience", 6), pos_weight, settings.get("selection_top_k")); result.update({"model": "temporal_cnn", "parameters": count_trainable_parameters(model), "device": str(device), "max_pos_weight": max_weight}); write_json(run_dir / "metrics.json", result); print(result)


if __name__ == "__main__": main()
