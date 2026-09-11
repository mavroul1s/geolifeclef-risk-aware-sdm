from __future__ import annotations

import numpy as np


def binarize(probabilities: np.ndarray, thresholds: float | np.ndarray = 0.5) -> np.ndarray:
    return (np.asarray(probabilities) >= np.asarray(thresholds)).astype(np.int8)


def multilabel_f1(targets: np.ndarray, probabilities: np.ndarray, thresholds: float | np.ndarray = 0.5, average: str = "micro") -> float:
    predicted, truth = binarize(probabilities, thresholds).astype(bool), np.asarray(targets).astype(bool)
    if average == "micro":
        tp, fp, fn = (predicted & truth).sum(), (predicted & ~truth).sum(), (~predicted & truth).sum()
        return float(2 * tp / max(2 * tp + fp + fn, 1))
    if average == "macro":
        tp, fp, fn = (predicted & truth).sum(0), (predicted & ~truth).sum(0), (~predicted & truth).sum(0)
        return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)))
    raise ValueError("average must be 'micro' or 'macro'")


def top_k_predictions(probabilities: np.ndarray, k: int) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or not 1 <= k <= probabilities.shape[1]:
        raise ValueError("probabilities must be [samples, labels] and k must be within its label dimension")
    indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    predicted = np.zeros_like(probabilities, dtype=np.int8)
    predicted[np.arange(len(probabilities))[:, None], indices] = 1
    return predicted


def multilabel_top_k_f1(targets: np.ndarray, probabilities: np.ndarray, k: int, average: str = "micro") -> float:
    return multilabel_f1(targets, top_k_predictions(probabilities, k), average=average)


def fit_adaptive_thresholds(targets: np.ndarray, probabilities: np.ndarray, minimum_positives: int = 5, candidates: np.ndarray | None = None, fallback: float = 0.5) -> np.ndarray:
    candidates = np.linspace(.05, .95, 19) if candidates is None else candidates; targets, probabilities = np.asarray(targets), np.asarray(probabilities); thresholds = np.full(targets.shape[1], fallback, dtype=np.float32)
    for label in range(targets.shape[1]):
        if targets[:, label].sum() >= minimum_positives:
            scores = [multilabel_f1(targets[:, [label]], probabilities[:, [label]], candidate) for candidate in candidates]
            thresholds[label] = candidates[int(np.argmax(scores))]
    return thresholds


def brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(targets) - np.asarray(probabilities)) ** 2))
