from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

KNOWN_MODALITIES = ("landsat", "climate", "sentinel", "static")


@dataclass(frozen=True)
class SplitSummary:
    path: str
    samples: int
    species: int
    keys: list[str]
    shapes: dict[str, list[int]]


def inspect_npz(path: str | Path) -> SplitSummary:
    source = Path(path)
    with np.load(source, mmap_mode="r", allow_pickle=False) as archive:
        if "labels" not in archive.files:
            raise ValueError(f"{source} is missing labels")
        labels = archive["labels"]
        if labels.ndim != 2:
            raise ValueError("labels must have shape [samples, species]")
        for name in KNOWN_MODALITIES:
            if name in archive.files and archive[name].shape[0] != labels.shape[0]:
                raise ValueError(f"{name} sample count differs from labels")
        return SplitSummary(str(source), labels.shape[0], labels.shape[1], sorted(archive.files), {key: list(archive[key].shape) for key in archive.files})


class CanonicalNPZDataset(Dataset[dict[str, torch.Tensor]]):
    """Loads a canonical split once; compressed NPZ members must not be reopened per item."""
    def __init__(self, path: str | Path, required_modality: str | None = None) -> None:
        self.path = Path(path); self.summary = inspect_npz(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        if required_modality and required_modality not in self.arrays:
            raise ValueError(f"{self.path} lacks required modality '{required_modality}'")

    def __len__(self) -> int:
        return self.summary.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"labels": torch.from_numpy(np.nan_to_num(self.arrays["labels"][index]).astype(np.float32))}
        for name in KNOWN_MODALITIES:
            if name in self.arrays:
                item[name] = torch.from_numpy(np.nan_to_num(self.arrays[name][index]).astype(np.float32))
        return item

    def close(self) -> None:
        self.arrays.clear()


def collate_modalities(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in batch]) for key in batch[0]}
