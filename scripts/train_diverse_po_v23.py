"""v23 deeper multi-seed PO initialization with fixed learning-curve checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geolifeclef.losses import asymmetric_loss
from geolifeclef.po_expert import masked_po_loss
from geolifeclef.utils import set_seed
from scripts.v23_protocol import CHECKPOINT_EPOCHS, checkpoint_score


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(width * 2, width), nn.Dropout(0.1),
        )

    def forward(self, values):
        return values + self.layers(values)


class DeepPOExpert(nn.Module):
    def __init__(self, num_labels: int, input_dim: int, width: int = 512, blocks: int = 3):
        super().__init__()
        if min(num_labels, input_dim, width, blocks) < 1:
            raise ValueError("Positive v23 model dimensions required")
        self.num_labels, self.input_dim, self.width, self.blocks = num_labels, input_dim, width, blocks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, width), nn.GELU(),
            *(ResidualBlock(width) for _ in range(blocks)), nn.LayerNorm(width),
        )
        self.classifier = nn.Linear(width, num_labels)

    def forward(self, values):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"Expected [batch, {self.input_dim}] v23 features")
        return self.classifier(self.encoder(values))


class AdaptedExpert(nn.Module):
    def __init__(self, initial: DeepPOExpert):
        super().__init__()
        self.encoder = copy.deepcopy(initial.encoder)
        self.classifier = copy.deepcopy(initial.classifier)
        self.po_classifier = copy.deepcopy(initial.classifier)
        self.num_labels, self.input_dim = initial.num_labels, initial.input_dim
        self.width, self.blocks = initial.width, initial.blocks

    def forward(self, values):
        return self.classifier(self.encoder(values))


def _tensors(features, labels, indices, device):
    x = torch.as_tensor(np.asarray(features[indices], dtype=np.float32), device=device)
    y = labels[indices]
    y = y.toarray() if hasattr(y, "toarray") else np.asarray(y)
    return x, torch.as_tensor(y, dtype=torch.float32, device=device)


def _step(loss, model, optimizer, scaler):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite v23 loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    scaler.step(optimizer)
    scaler.update()


@torch.no_grad()
def predict(model, features, device, deadline):
    model.eval()
    result = np.empty((len(features), model.num_labels), dtype=np.float16)
    for start in range(0, len(features), 512):
        if time.monotonic() >= deadline:
            raise TimeoutError("v23 inference deadline")
        x = torch.as_tensor(np.asarray(features[start:start + 512], dtype=np.float32), device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probability = model(x).float().sigmoid()
        result[start:start + len(x)] = probability.cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Invalid v23 probability")
    return result


def _po_groups(count: int, seed: int):
    permutation = np.random.default_rng(seed).permutation(count)
    heldout = min(10000, max(1, count // 20))
    return permutation[heldout:], permutation[:heldout]


@torch.no_grad()
def po_probe(model, features, labels, indices, device, deadline):
    model.eval()
    values = []
    for start in range(0, len(indices), 256):
        if time.monotonic() >= deadline:
            raise TimeoutError("v23 PO probe deadline")
        x, y = _tensors(features, labels, indices[start:start + 256], device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model.po_classifier(model.encoder(x)) if isinstance(model, AdaptedExpert) else model(x)
        values.append((len(x), float(masked_po_loss(logits.float(), y))))
    return sum(count * loss for count, loss in values) / sum(count for count, _ in values)


def pretrain(features, labels, weights, device, output: Path, deadline, *, seed: int,
             width: int = 512, blocks: int = 3, epochs: int = 24, draws: int = 350000):
    set_seed(seed)
    model = DeepPOExpert(labels.shape[1], features.shape[1], width, blocks).to(device)
    train, probe = _po_groups(len(features), seed)
    sampling = np.asarray(weights, dtype=np.float64)[train]
    sampling /= sampling.sum()
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.monotonic()
        losses = []
        samples = rng.choice(train, min(draws, len(train)), replace=True, p=sampling)
        for start in range(0, len(samples), 256):
            if time.monotonic() > deadline - 120:
                raise TimeoutError("v23 PO pretraining deadline")
            x, y = _tensors(features, labels, samples[start:start + 256], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = masked_po_loss(logits.float(), y)
            _step(loss, model, optimizer, scaler)
            losses.append(float(loss.detach()))
        scheduler.step()
        item = {"epoch": epoch, "train_loss": float(np.mean(losses)), "draws": len(samples),
                "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started}
        if epoch in (8, 16, 24) or epoch == epochs:
            item["po_probe_loss_diagnostic"] = po_probe(model, features, labels, probe, device, deadline)
        history.append(item)
        print(json.dumps({"stage": "v23_po_pretrain", "seed": seed, **item}), flush=True)
    path = output / f"po_pretrained_seed_{seed}_w{width}_d{blocks}.pt"
    torch.save({"state_dict": model.state_dict(), "seed": seed, "width": width, "blocks": blocks,
                "input_dim": features.shape[1], "num_labels": labels.shape[1]}, path)
    return model, {"seed": seed, "width": width, "blocks": blocks, "history": history,
                   "probe_groups": len(probe), "probe_used_for_selection": False}


def fit_adaptation(pretrained: DeepPOExpert | None, arm: str, pa_features, labels,
                   train_indices, selection_indices, base_selection, ensemble_prefix,
                   po_features, po_labels, po_weights, device, output: Path, deadline, *,
                   seed: int, width: int = 512, blocks: int = 3, epochs: int = 72,
                   checkpoint_epochs=CHECKPOINT_EPOCHS):
    if arm not in ("single_head", "retained_po", "zero_po"):
        raise ValueError("Unknown v23 adaptation arm")
    checkpoint_epochs = tuple(int(value) for value in checkpoint_epochs)
    if not checkpoint_epochs or checkpoint_epochs[-1] > epochs or any(value < 1 for value in checkpoint_epochs):
        raise ValueError("Invalid fixed v23 checkpoint schedule")
    set_seed(seed)
    initial = pretrained if pretrained is not None else DeepPOExpert(labels.shape[1], pa_features.shape[1], width, blocks).to(device)
    if initial.width != width or initial.blocks != blocks:
        raise ValueError("Pretraining/adaptation architecture mismatch")
    model = AdaptedExpert(initial).to(device)
    teacher = copy.deepcopy(initial.encoder).eval() if arm == "retained_po" else None
    if teacher is not None:
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    for parameter in model.po_classifier.parameters():
        parameter.requires_grad_(arm == "retained_po")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    po_train, po_test = _po_groups(len(po_features), seed)
    sampling = np.asarray(po_weights, dtype=np.float64)[po_train]
    sampling /= sampling.sum()
    po_rng, pa_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    targets = np.asarray(labels[selection_indices])
    history, best_score, best_epoch = [], -1.0, None
    checkpoint = output / f"{arm}_seed_{seed}_w{width}_d{blocks}_best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.monotonic()
        losses = []
        order = pa_rng.permutation(train_indices)
        for start in range(0, len(order), 256):
            if time.monotonic() > deadline - 180:
                raise TimeoutError("v23 PA adaptation deadline")
            x, y = _tensors(pa_features, labels, order[start:start + 256], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            pa_loss = asymmetric_loss(logits.float(), y)
            auxiliary = pa_loss.new_zeros(())
            retention = pa_loss.new_zeros(())
            if arm == "retained_po":
                sample = po_rng.choice(po_train, min(256, len(po_train)), replace=True, p=sampling)
                px, py = _tensors(po_features, po_labels, sample, device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    encoded = model.encoder(px)
                    po_logits = model.po_classifier(encoded)
                    with torch.no_grad():
                        anchor = teacher(px)
                auxiliary = masked_po_loss(po_logits.float(), py)
                retention = F.mse_loss(F.normalize(encoded.float(), dim=1), F.normalize(anchor.float(), dim=1))
            loss = pa_loss + 0.001 * auxiliary + 0.01 * retention
            _step(loss, model, optimizer, scaler)
            losses.append((float(pa_loss.detach()), float(auxiliary.detach()), float(retention.detach())))
        scheduler.step()
        item = {"epoch": epoch, "loss_components": np.mean(losses, axis=0).tolist(),
                "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started,
                "checkpoint_evaluated": epoch in checkpoint_epochs}
        if epoch in checkpoint_epochs:
            values = predict(model, pa_features[selection_indices], device, deadline)
            ensemble = np.mean([*ensemble_prefix, values], axis=0, dtype=np.float32)
            item["selection_complement_f1"] = checkpoint_score(targets, base_selection, ensemble)
            if item["selection_complement_f1"] > best_score:
                best_score, best_epoch = item["selection_complement_f1"], epoch
                torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                            "epoch": epoch, "selection_complement_f1": best_score,
                            "width": width, "blocks": blocks, "input_dim": pa_features.shape[1],
                            "num_labels": labels.shape[1]}, checkpoint)
        history.append(item)
        print(json.dumps({"stage": "v23_pa_adaptation", "arm": arm, "seed": seed, **item}), flush=True)
    if best_epoch is None:
        raise AssertionError("No v23 checkpoint was evaluated")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    probe = po_probe(model, po_features, po_labels, po_test, device, deadline) if arm != "zero_po" else None
    (output / f"{arm}_seed_{seed}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return model, {"arm": arm, "seed": seed, "width": width, "blocks": blocks,
                   "fixed_checkpoint_epochs": list(checkpoint_epochs), "selected_epoch": best_epoch,
                   "selection_complement_f1": best_score, "po_probe_loss_after_adaptation": probe,
                   "history": history, "early_stopping_used": False, "learning_rate": 2e-4,
                   "checkpoint_sha256": digest}
