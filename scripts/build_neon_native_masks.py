"""Transfer native masks only through verified identical TDWG regions."""
import hashlib
import json
from pathlib import Path
import sys

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import shape
from rasterio.crs import CRS
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geolifeclef.native_region_bridge import assign_regions, constant_region_masks


def main():
    data, out = ROOT/'data/neon_input_followup', ROOT/'results/neon_input_followup'
    geometry_path = data/'tdwg_level3/level3'
    if CRS.from_wkt(geometry_path.with_suffix('.prj').read_text()).to_epsg() != 4326:
        raise ValueError('TDWG geometry CRS changed')
    reader = shapefile.Reader(str(geometry_path))
    # Only the ASCII region code is needed; do not decode legacy DBF place names.
    records = list(reader.iterShapeRecords(fields=['LEVEL3_COD']))
    geometries = [shape(r.shape.__geo_interface__) for r in records]
    codes = [r.record.as_dict()['LEVEL3_COD'] for r in records]
    with h5py.File(ROOT/'data/external_replication_e1/predictors/eval_location.h5') as f:
        locations = f['predictor_data'][:]
    regions, fallback = assign_regions(geometries, codes, locations)
    mask_path = ROOT/'data/external_replication_e1/range_masks_1km/eval_range_mask.h5'
    with h5py.File(mask_path) as f:
        offsets, species = f['cumulative_lengths'][:], f['species_index'][:]
    lookup, inconsistent, counts = constant_region_masks(regions, offsets, species, 5771)
    plots = pd.read_csv(ROOT/'data/neon_bridge/full_plot_metadata_candidates.csv')
    cells = plots.groupby(['grid_x','grid_y'], as_index=False).agg(site=('site','first'), n_plots=('plot_id','nunique'))
    if plots.groupby(['grid_x','grid_y']).site.nunique().max() != 1:
        raise ValueError('A candidate grid cell spans multiple sites')
    lon, lat = transform('EPSG:6933','EPSG:4326',(cells.grid_x+.5)*1000,(cells.grid_y+.5)*1000)
    cells['lon'], cells['lat'] = lon, lat
    cells['region'], cells['nearest_polygon_fallback'] = assign_regions(geometries,codes,cells[['lon','lat']])
    cells['native_mask_available'] = cells.region.isin(lookup)
    native = np.zeros((len(cells),5771),dtype=bool)
    for i, code in enumerate(cells.region):
        if code in lookup:
            native[i,lookup[code]] = True
    # Missing masks remain explicitly invalid; an all-false row is not an absence claim.
    np.savez_compressed(data/'candidate_native_masks.npz', native=native,
                        valid=cells.native_mask_available.to_numpy(), grid_x=cells.grid_x.to_numpy(),grid_y=cells.grid_y.to_numpy())
    cells.to_csv(data/'candidate_cells.csv',index=False)
    result = {'status':'all_candidate_masks_verified' if cells.native_mask_available.all() else 'partial',
              'method':'TDWG region equality plus verification of every released eval mask within region; no nearest-mask substitution',
              'reference_eval_cells':len(regions),'represented_regions':len(counts),'consistent_regions':len(lookup),
              'inconsistent_regions':inconsistent,'reference_polygon_fallbacks':int(fallback.sum()),
              'candidate_cells':len(cells),'candidate_mask_valid_cells':int(cells.native_mask_available.sum()),
              'candidate_polygon_fallbacks':int(cells.nearest_polygon_fallback.sum()),
              'candidate_regions':cells.groupby('region').size().astype(int).to_dict(),
              'unavailable_candidate_regions':sorted(set(cells.loc[~cells.native_mask_available,'region'])),
              'native_species_per_candidate_cell_range':[int(native[cells.native_mask_available].sum(1).min()),int(native.sum(1).max())] if cells.native_mask_available.any() else None,
              'published_reference_counts_per_candidate_region':{k:counts.get(k,0) for k in sorted(set(cells.region))},
              'native_mask_sha256':hashlib.sha256((data/'candidate_native_masks.npz').read_bytes()).hexdigest(),
              'NEON_composition_values_read':False,'fresh_predictor_or_scientific_evaluation_admission':False}
    (out/'native_mask_bridge.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
