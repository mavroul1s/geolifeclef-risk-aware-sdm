from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import multilabel_f1, multilabel_top_k_f1


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, modality: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); probabilities, targets = [], []
    for batch in loader:
        probabilities.append(torch.sigmoid(model(batch[modality].to(device, non_blocking=True))).cpu().numpy())
        targets.append(batch["labels"].numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def train_temporal(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, modality: str, epochs: int, lr: float, weight_decay: float, mixed_precision: bool, output_dir: Path, patience: int = 6, pos_weight: torch.Tensor | None = None, selection_top_k: int | None = None) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay); criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None); scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_dir.mkdir(parents=True, exist_ok=True); history, best_f1, stale = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter(); model.train(); total_loss = 0.; samples = 0
        for batch in train_loader:
            features, labels = batch[modality].to(device), batch["labels"].to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
                loss = criterion(model(features), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total_loss += loss.item() * len(labels); samples += len(labels)
        elapsed = time.perf_counter() - started; targets, probabilities = predict(model, val_loader, device, modality); fixed_micro = multilabel_f1(targets, probabilities); fixed_sample = multilabel_f1(targets, probabilities, average="samples"); score = multilabel_top_k_f1(targets, probabilities, selection_top_k, average="samples") if selection_top_k else fixed_sample
        record = {"epoch": epoch, "train_loss": total_loss / max(samples, 1), "val_micro_f1": fixed_micro, "val_sample_f1": fixed_sample, "val_selection_sample_f1": score, "seconds": elapsed, "examples_per_second": samples / max(elapsed, 1e-6)}; history.append(record)
        if score > best_f1:
            best_f1, stale = score, 0; torch.save({"model_state": model.state_dict(), "val_sample_f1": score}, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience: break
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    return {"best_val_selection_sample_f1": best_f1, "selection_top_k": selection_top_k, "selection_metric": "sample-averaged F1", "epochs_completed": len(history), "last_epoch": history[-1], "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None}
