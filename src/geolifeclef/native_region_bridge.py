"""Published TDWG-region assignment and verified region-constant mask reuse."""
import numpy as np
from shapely import points
from shapely.strtree import STRtree


def assign_regions(geometries, region_codes, coordinates):
    """Match the publisher: within, then nearest polygon for unmatched points."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError('Finite longitude, latitude pairs required')
    tree = STRtree(geometries)
    locations = points(coordinates[:, 0], coordinates[:, 1])
    indices = np.full(len(locations), -1, dtype=np.int32)
    point_index, polygon_index = tree.query(locations, predicate='within')
    indices[point_index] = polygon_index
    missing = indices == -1
    if missing.any():
        indices[missing] = tree.nearest(locations[missing])
    return np.asarray(region_codes)[indices], missing


def constant_region_masks(codes, offsets, species, num_classes):
    """A region can be reused only if every published row has the same mask."""
    if len(offsets) != len(codes) + 1 or offsets[0] != 0 or offsets[-1] != len(species) or np.any(np.diff(offsets) < 0):
        raise ValueError('Invalid published sparse mask offsets')
    if np.any(species < 0) or np.any(species >= num_classes):
        raise ValueError('Published species index outside class order')
    masks, inconsistent, counts = {}, set(), {}
    for i, code in enumerate(codes):
        ids = np.unique(species[offsets[i]:offsets[i+1]])
        counts[code] = counts.get(code, 0) + 1
        if code not in masks:
            masks[code] = ids
        elif not np.array_equal(ids, masks[code]):
            inconsistent.add(code)
    return {code: ids for code, ids in masks.items() if code not in inconsistent}, sorted(inconsistent), counts
