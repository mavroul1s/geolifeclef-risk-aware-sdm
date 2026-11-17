"""Describe metadata-only full-inventory candidates and spatial overlap."""
import hashlib
import json
from pathlib import Path
import sys

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
from rasterio.warp import transform
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geolifeclef.neon_survey import full_plot_completeness


def nearest(coordinates, reference):
    return BallTree(np.deg2rad(np.asarray(reference, dtype=np.float64)), metric='haversine').query(
        np.deg2rad(np.asarray(coordinates, dtype=np.float64)))[0][:, 0] * 6371.0088


def main():
    data, out = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'
    receipt = json.loads((out / 'survey_metadata_retrieval.json').read_text())
    frames = []
    for file in receipt['files']:
        path = ROOT / file['metadata_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != file['metadata_sha256']:
            raise ValueError('Projected metadata changed')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame['table'] = file['table']
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True).drop_duplicates()
    records = []
    for (site, plot, bout), frame in metadata.groupby(['siteID', 'plotID', 'boutNumber'], sort=True):
        coordinates = frame[['decimalLatitude', 'decimalLongitude']].drop_duplicates()
        coherent = len(coordinates) == 1
        row = {'site': site, 'plot_id': plot, 'bout': int(bout), 'first_date': frame.endDate.min(),
               'last_date': frame.endDate.max(), 'lat': float(coordinates.iloc[0, 0]),
               'lon': float(coordinates.iloc[0, 1]), 'coherent_plot_coordinates': coherent,
               'max_coordinate_uncertainty_m': float(pd.to_numeric(frame.coordinateUncertainty, errors='coerce').max()),
               'WGS84': bool(frame.geodeticDatum.eq('WGS84').all()),
               **full_plot_completeness(frame.to_dict('records'))}
        row['metadata_candidate'] = bool(row['full_plot_eligible'] and coherent and row['WGS84']
                                         and np.isfinite(row['max_coordinate_uncertainty_m'])
                                         and 0 < row['max_coordinate_uncertainty_m'] <= 100)
        records.append(row)
    surveys = pd.DataFrame(records)
    locations = surveys[['lat', 'lon']].to_numpy()
    with h5py.File(ROOT / 'data/external_replication_e1/predictors/eval_location.h5') as f:
        sage = f['predictor_data'][:][:, [1, 0]]
    consumed = pd.concat([pd.read_csv(ROOT / 'data/external_replication_e1/cohort_roles.csv')[['lat', 'lon']],
                          pd.read_csv(ROOT / 'data/confirmation_e3/locked_cohort.csv')[['lat', 'lon']]])
    glc = pd.concat([pd.read_csv(ROOT / 'data/raw' / file, usecols=['lat', 'lon'])
                     for file in ['GLC25_PA_metadata_train.csv', 'GLC25_PA_metadata_test.csv']])
    surveys['nearest_published_SAGE_eval_km'] = nearest(locations, sage)
    surveys['nearest_consumed_SAGE_km'] = nearest(locations, consumed[['lat', 'lon']])
    surveys['nearest_GLC_PA_km'] = nearest(locations, glc[['lat', 'lon']])
    x, y = transform('EPSG:4326', 'EPSG:6933', surveys.lon, surveys.lat)
    surveys['grid_x'] = np.floor(np.array(x) / 1000).astype(np.int64)
    surveys['grid_y'] = np.floor(np.array(y) / 1000).astype(np.int64)
    sx, sy = transform('EPSG:4326', 'EPSG:6933', sage[:, 1], sage[:, 0])
    published_keys = set(zip(np.floor(np.array(sx) / 1000).astype(np.int64), np.floor(np.array(sy) / 1000).astype(np.int64)))
    surveys['same_grid_cell_any_SAGE_eval'] = [key in published_keys for key in zip(surveys.grid_x, surveys.grid_y)]
    surveys['spatial_candidate'] = (surveys.metadata_candidate & surveys.nearest_consumed_SAGE_km.ge(100)
                                     & surveys.nearest_GLC_PA_km.ge(500) & ~surveys.same_grid_cell_any_SAGE_eval)
    # Count a plot once, using the earliest eligible bout; no outcome-driven visit choice.
    candidates = surveys[surveys.spatial_candidate].sort_values(['plot_id', 'first_date', 'bout']).drop_duplicates('plot_id')
    path = data / 'plot_bout_metadata_audit.csv'
    surveys.to_csv(path, index=False)
    candidates.to_csv(data / 'full_plot_metadata_candidates.csv', index=False)
    result = {'status': 'metadata_only_feasibility_completed', 'year': 2023, 'release': 'RELEASE-2026',
              'sites': int(surveys.site.nunique()), 'distinct_plots': int(surveys.plot_id.nunique()),
              'plot_bouts': len(surveys), 'full_nested_unit_plot_bouts': int(surveys.nested_units_present.sum()),
              'full_units_and_quality_plot_bouts': int(surveys.full_plot_eligible.sum()),
              'full_units_quality_coordinates_plot_bouts': int(surveys.metadata_candidate.sum()),
              'spatial_candidate_plot_bouts': int(surveys.spatial_candidate.sum()),
              'candidate_distinct_plots': len(candidates), 'candidate_sites': int(candidates.site.nunique()),
              'candidate_1km_cells': len(candidates[['grid_x', 'grid_y']].drop_duplicates()),
              'candidate_site_counts': {k: int(v) for k, v in candidates.site.value_counts().items()},
              'all_plot_bouts_same_SAGE_eval_grid': int(surveys.same_grid_cell_any_SAGE_eval.sum()),
              'candidate_minimum_distance_consumed_SAGE_km': float(candidates.nearest_consumed_SAGE_km.min()) if len(candidates) else None,
              'candidate_minimum_distance_GLC_PA_km': float(candidates.nearest_GLC_PA_km.min()) if len(candidates) else None,
              'area_semantics': 'Nested union is 400 m2; do not sum 400+60+6 or repeated visits.',
              'selection_scope': 'Feasibility thresholds, not a frozen scientific evaluation cohort or source-role allocation.',
              'NEON_composition_values_inspected': False, 'original_model_PO_training_overlap_audited': False,
              'native_species_masks_verified': False, 'new_location_predictor_bridge_verified': False,
              'metadata_audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'no_independent_training_or_sample_size_claim': True}
    (out / 'survey_metadata_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
