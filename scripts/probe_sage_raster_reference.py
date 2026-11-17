"""Bounded source-raster checks against eight fixed released input rows."""
import hashlib
import json
from pathlib import Path
import sys

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geolifeclef.range_io import BoundedHTTPFile
from geolifeclef.raster_bridge import nearest_valid_value


def main():
    data, out = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'
    access = json.loads((out / 'raster_access_probe.json').read_text())
    ids = np.load(data / 'cpu_reference.npz')['cell_ids'][:8]
    with h5py.File(ROOT / 'data/external_replication_e1/predictors/eval_location.h5') as f:
        locations = f['predictor_data'][:][ids]
    reports = []
    for probe, group in zip(access['results'], ['chelsa', 'soilgrids', 'topography']):
        if not probe['exact_64KiB_range']:
            reports.append({'source': probe['source'], 'status': 'access_failed'})
            continue
        size = int(probe['headers']['Content-Range'].split('/')[-1])
        handles = []
        name = probe['source'] + '.tif'

        def opener(path, mode='rb'):
            if str(path) != name:
                raise FileNotFoundError(path)
            stream = BoundedHTTPFile(probe['url'], 0, size, budget=16 * 1024**2,
                                     block_size=64 * 1024, cache_dir=data / 'raster_cache' / probe['source'])
            handles.append(stream)
            return stream

        with h5py.File(data / 'reference_predictors' / f'eval_{group}.h5') as f:
            reference = f['predictor_data'][:][ids, 0]
        result = {'source': probe['source'], 'predictor_group': group, 'feature_index': 0,
                  'reference_cell_ids': ids.tolist(), 'per_handle_network_byte_cap': 16 * 1024**2}
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_PAM_ENABLED='NO'):
                with rasterio.open(name, opener=opener) as src:
                    x, y = transform('EPSG:4326', src.crs, locations[:, 0], locations[:, 1])
                    values, indices = [], []
                    for px, py in zip(x, y):
                        row, col = src.index(px, py)
                        row, col = np.clip(row, 0, src.height - 1), np.clip(col, 0, src.width - 1)
                        values.append(float(src.read(1, window=Window(col, row, 1, 1))[0, 0]))
                        indices.append((int(row), int(col)))
                    values = np.array(values)
                    nodata = np.isclose(values, src.nodata, atol=0) if src.nodata is not None else np.zeros(8, dtype=bool)
                    result.update({'raster_shape': [src.height, src.width], 'CRS': str(src.crs),
                                   'nodata': src.nodata, 'block_shapes': src.block_shapes,
                                   'nodata_samples': int(nodata.sum()), 'raw_values': values.tolist(),
                                   'published_input_values': reference.tolist()})
                    fills = []
                    for index in np.flatnonzero(nodata):
                        value, distance, radius = nearest_valid_value(src, *indices[index])
                        values[index] = value
                        fills.append({'sample_index': int(index), 'value': value, 'distance_pixels': distance, 'read_radius': radius})
                    result['nearest_valid_fills'] = fills
                    result['values_after_published_fill_rule'] = values.tolist()
                    match = bool(np.array_equal(values, reference, equal_nan=True))
                    result['status'] = 'eight_samples_match' if match else 'not_reproduced'
            result['new_network_bytes'] = sum(h.fetched_bytes for h in handles)
            result['new_range_requests'] = sum(h.requests for h in handles)
        except Exception as error:
            result.update({'status': 'probe_failed', 'error_type': type(error).__name__,
                           'diagnostic': str(error)[:300]})
        reports.append(result)
        print(json.dumps(result), flush=True)
    result = {'scope': 'Three of 52 feature columns, eight fixed input rows, including the observed nodata fill; no full feature-vector or NEON-input equivalence claim.',
              'all_three_sampled_columns_match': all(r['status'] == 'eight_samples_match' for r in reports),
              'results': reports, 'NEON_observations_or_predictions_read': False}
    (out / 'raster_reference_probe.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
