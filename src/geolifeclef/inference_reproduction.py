"""Numerical audit for float16 top-k boundaries; no outcomes enter this check."""
import numpy as np


def reference_topk_gap_ulps(probability, reference):
    p, selected = np.asarray(probability), np.asarray(reference, int)
    if p.dtype != np.float16 or p.ndim != 2 or selected.ndim != 2 or len(p) != len(selected):
        raise ValueError('Aligned float16 probabilities and reference top-k indices required')
    k = selected.shape[1]
    if not 0 < k <= p.shape[1] or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid top-k probability matrix')
    if np.any(selected < 0) or np.any(selected >= p.shape[1]) or np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
        raise ValueError('Reference species sets must contain distinct valid indices')
    boundary = np.partition(p, -k, axis=1)[:, -k]
    minimum = np.take_along_axis(p, selected, axis=1).min(1)
    spacing = np.spacing(boundary).astype(np.float32)
    return np.maximum(boundary.astype(np.float32)-minimum.astype(np.float32), 0)/spacing
