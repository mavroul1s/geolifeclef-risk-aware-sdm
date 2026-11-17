"""Audit spatial training support without reading either source's species labels."""
import hashlib
import json
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
from rasterio.warp import transform
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    data, out = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'
    candidates = pd.read_csv(data / 'full_plot_metadata_candidates.csv')
    with h5py.File(data / 'train_location_1km.h5') as f:
        training = f['predictor_data'][:]
    if training.ndim != 2 or training.shape[1] != 2 or not np.isfinite(training).all():
        raise ValueError('Training coordinate contract mismatch')
    tree = BallTree(np.deg2rad(training[:, [1, 0]].astype(np.float64)), metric='haversine')
    plot_distance, _ = tree.query(np.deg2rad(candidates[['lat', 'lon']]))
    lon, lat = transform('EPSG:6933', 'EPSG:4326', (candidates.grid_x + .5) * 1000, (candidates.grid_y + .5) * 1000)
    center_distance, nearest_row = tree.query(np.deg2rad(np.column_stack([lat, lon])))
    nearest_row = nearest_row[:, 0]
    tx, ty = transform('EPSG:4326', 'EPSG:6933', training[nearest_row, 0], training[nearest_row, 1])
    same_grid = ((np.floor(np.array(tx) / 1000).astype(np.int64) == candidates.grid_x)
                 & (np.floor(np.array(ty) / 1000).astype(np.int64) == candidates.grid_y))
    candidates['nearest_any_training_location_km'] = plot_distance[:, 0] * 6371.0088
    candidates['nearest_training_row_to_cell_center'] = nearest_row
    candidates['cell_center_nearest_training_distance_m'] = center_distance[:, 0] * 6371008.8
    candidates['same_grid_any_published_PO_training_location'] = same_grid
    path = data / 'candidate_training_overlap.csv'
    candidates.to_csv(path, index=False)
    result = {'status': 'geographic_training_support_audited', 'published_training_locations': len(training),
              'candidate_plots': len(candidates), 'candidate_sites': int(candidates.site.nunique()),
              'candidate_1km_cells': len(candidates[['grid_x', 'grid_y']].drop_duplicates()),
              'plots_same_training_grid': int(same_grid.sum()),
              'same_training_grid_distinct_cells': len(candidates[same_grid][['grid_x', 'grid_y']].drop_duplicates()),
              'plots_more_than_1km_from_any_training_location': int(candidates.nearest_any_training_location_km.gt(1).sum()),
              'plots_more_than_10km_from_any_training_location': int(candidates.nearest_any_training_location_km.gt(10).sum()),
              'plot_nearest_training_distance_median_km': float(candidates.nearest_any_training_location_km.median()),
              'site_counts_with_any_same_grid': candidates[same_grid].groupby('site').size().astype(int).to_dict(),
              'interpretation': 'The released aggregation model uses this PO training-location universe. Spatial support overlap does not prove that a given species label or NEON record was used in training; no unseen-training-location claim is justified for overlapping cells.',
              'NEON_or_training_species_labels_read': False,
              'candidate_metadata_sha256': hashlib.sha256((data / 'full_plot_metadata_candidates.csv').read_bytes()).hexdigest(),
              'training_coordinates_sha256': hashlib.sha256((data / 'train_location_1km.h5').read_bytes()).hexdigest(),
              'overlap_table_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / 'training_overlap_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
