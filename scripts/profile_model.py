#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import torch

from geolifeclef.data import CanonicalNPZDataset
from geolifeclef.models import TemporalCNN, count_trainable_parameters
from geolifeclef.utils import read_config


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args(); config = read_config(args.config); modality = config["model"].get("modality", "landsat"); dataset = CanonicalNPZDataset(config["data"]["train_path"], modality); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = TemporalCNN(dataset.archive[modality].shape[-1], dataset.summary.species, config["model"].get("channels", 32), config["model"].get("dropout", .15)).to(device); batch = torch.zeros(min(config["training"].get("batch_size", 128), len(dataset)), *dataset.archive[modality].shape[1:], device=device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    for _ in range(10): model(batch)
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(100): model(batch)
    if device.type == "cuda": torch.cuda.synchronize()
    report = {"parameters": count_trainable_parameters(model), "batch_size": len(batch), "forward_examples_per_second": len(batch) * 100 / (time.perf_counter() - started), "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None, "device": str(device)}; print(report)


if __name__ == "__main__": main()

