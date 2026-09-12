"""Bounded PO pretraining and PA adaptation; only checkpoint labels select weights."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from geolifeclef.losses import asymmetric_loss
from geolifeclef.po_expert import POExpert, masked_po_loss
from geolifeclef.utils import set_seed
from scripts.run_environmental_challenger import top_rank, ranked_f1


@torch.no_grad()
def expert_predict(model, features, device, deadline, batch_size=512):
    model.eval()
    output = np.empty((len(features), model.num_labels), dtype=np.float16)
    for begin in range(0, len(features), batch_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("PO inference exceeded the registered deadline")
        x = torch.as_tensor(np.array(features[begin:begin + batch_size], dtype=np.float32), device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probability = model(x).float().sigmoid()
        output[begin:begin + len(x)] = probability.cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite expert probability")
    return output


def fit_expert(pa_features, pa_labels, train_indices, selection_indices,
               po_features, po_labels, po_weights, device, output: Path,
               name: str, deadline: float, use_po: bool,
               pretrain_epochs=8, epochs=28, minimum_epochs=12,
               batch_size=256, maximum_po_draws=200_000):
    """Both controls share width, seed and PA schedule; PO exposure is explicit.

    Fixed PO pretraining has no PA checkpoint decision. PA adaptation alone
    selects the final checkpoint. A weak PO rehearsal term limits forgetting.
    Zero-PO takes the identical PA steps, without a PO forward or gradient.
    """
    if epochs < minimum_epochs or len(train_indices) < batch_size:
        raise ValueError("Insufficient registered PA training exposure")
    set_seed(20250921)
    rng = np.random.default_rng(20250921)
    pa_rng = np.random.default_rng(20250921)
    model = POExpert(pa_labels.shape[1], pa_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{name}_best.pt"
    probability_weights = np.asarray(po_weights, dtype=np.float64)
    probability_weights /= probability_weights.sum()

    def batch(features, labels, indices):
        x = torch.as_tensor(np.array(features[indices], dtype=np.float32), device=device)
        selected = labels[indices]
        y = selected.toarray() if hasattr(selected, "toarray") else np.array(selected)
        return x, torch.as_tensor(y, dtype=torch.float32, device=device)

    def update(x, y, po=False, rehearsal=False):
        if time.monotonic() > deadline - 90:
            raise TimeoutError(f"Insufficient remaining expert budget: {name}")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        loss = masked_po_loss(logits.float(), y) if po else asymmetric_loss(logits.float(), y)
        if rehearsal:
            sample = rng.choice(len(po_features), min(batch_size, len(po_features)), replace=True, p=probability_weights)
            px, py = batch(po_features, po_labels, sample)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                auxiliary = model(px)
            # ASL is averaged over 5,016 outputs. Match scales explicitly.
            loss = loss + 0.00005 * masked_po_loss(auxiliary.float(), py)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite expert loss: {name}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    if use_po:
        for epoch in range(pretrain_epochs):
            started = time.monotonic()
            model.train()
            count = min(maximum_po_draws, len(po_features))
            draws = rng.choice(len(po_features), count, replace=True, p=probability_weights)
            losses = []
            for begin in range(0, count, batch_size):
                x, y = batch(po_features, po_labels, draws[begin:begin + batch_size])
                losses.append(update(x, y, po=True))
            item = {"model": name, "stage": "po_pretraining", "epoch": epoch + 1,
                    "draws": count, "loss": float(np.mean(losses)), "seconds": time.monotonic() - started}
            history.append(item)
            print(json.dumps(item), flush=True)
    # Identical PA optimizer/schedule regardless of pretrained state.
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    best, best_epoch = -1., 0
    selection_labels = np.array(pa_labels[selection_indices])
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        indices = pa_rng.permutation(train_indices)
        losses = []
        for step, begin in enumerate(range(0, len(indices), batch_size)):
            x, y = batch(pa_features, pa_labels, indices[begin:begin + batch_size])
            losses.append(update(x, y, rehearsal=use_po and step % 3 == 0))
        scheduler.step()
        probabilities = expert_predict(model, pa_features[selection_indices], device, deadline)
        ranks, _ = top_rank(probabilities)
        score = float(ranked_f1(selection_labels, ranks, np.full(len(ranks), min(20, ranks.shape[1]))).mean())
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "selection_top20_f1": score, "input_dim": pa_features.shape[1],
                        "num_labels": pa_labels.shape[1]}, checkpoint)
        item = {"model": name, "stage": "pa_adaptation", "epoch": epoch,
                "loss": float(np.mean(losses)), "selection_top20_f1": score,
                "seconds": time.monotonic() - started}
        history.append(item)
        print(json.dumps(item), flush=True)
        (output / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch >= minimum_epochs and (epoch - best_epoch >= 8 or time.monotonic() + 1.3 * item["seconds"] + 90 > deadline):
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state"])
    model.eval()
    return model, {"name": name, "use_po": use_po, "seed": 20250921,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "best_epoch": best_epoch, "pa_epochs": epoch,
                   "po_epochs": pretrain_epochs if use_po else 0,
                   "selection_top20_f1": best, "history": history}
