"""Bounded nearest-valid sampling equivalent within the published 50-pixel radius."""
import numpy as np
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt


def nearest_valid_value(src, row, col, maximum_radius=50):
    """Expand only as far as needed; all nearer pixels are inside the window.

    Use the publisher's nodata comparison, EDT tie order and finite-value check.
    A valid point farther than the admitted radius remains missing.
    """
    for radius in [r for r in [1, 2, 4, 8, 16, 32, 50] if r <= maximum_radius] + [maximum_radius]:
        r0, r1 = max(0, row - radius), min(src.height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(src.width, col + radius + 1)
        values = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
        valid = ~np.isclose(values, src.nodata, atol=0)
        if not valid.any():
            continue
        distance, indices = distance_transform_edt(~valid, return_distances=True, return_indices=True)
        rr, cc = row - r0, col - c0
        d = distance[rr, cc]
        value = values[indices[0, rr, cc], indices[1, rr, cc]]
        if d <= radius:
            return (float(value) if np.isfinite(value) and not np.isclose(value, src.nodata, atol=0) else np.nan,
                    float(d), radius)
    return np.nan, None, maximum_radius
