"""Reuse released inputs only at identical stored 1 km grid centers."""
import hashlib
import json
from pathlib import Path
import sys

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geolifeclef.range_io import BoundedHTTPFile


def main():
    data,out=ROOT/'data/neon_input_followup',ROOT/'results/neon_input_followup'
    config=json.loads((ROOT/'configs/neon_input_followup.json').read_text())
    cells=pd.read_csv(data/'candidate_cells.csv')
    plots=pd.read_csv(ROOT/'data/neon_bridge/candidate_training_overlap.csv')
    keys=['grid_x','grid_y']
    grouped=plots.groupby(keys)
    if grouped.nearest_training_row_to_cell_center.nunique().max()!=1 or grouped.same_grid_any_published_PO_training_location.nunique().max()!=1:
        raise ValueError('Cell-level cached-input identity inconsistent')
    lookup=grouped.first().reset_index()[keys+['nearest_training_row_to_cell_center','same_grid_any_published_PO_training_location']]
    cells=cells.merge(lookup,on=keys,how='left',validate='one_to_one')
    valid=cells.same_grid_any_published_PO_training_location.to_numpy(bool)
    rows=cells.loc[valid,'nearest_training_row_to_cell_center'].to_numpy(np.int64)
    with h5py.File(ROOT/'data/neon_bridge/train_location_1km.h5') as f:
        locations=f['predictor_data'][:]
    expected=cells.loc[valid,['lon','lat']].to_numpy(np.float32)
    if not np.array_equal(locations[rows],expected):
        raise ValueError('Same grid must also have identical published float32 center coordinates')
    record=json.loads((ROOT/'data/external_replication_e1/zenodo_21297133.json').read_text())
    url=next(f for f in record['files'] if f['key']=='sage_benchmark_data.tar')['links']['self']
    inventory=json.loads((ROOT/'data/independent_data_audit/sage_sage_benchmark_data_tar_index.json').read_text())
    results=[];parts=[];order=np.argsort(rows)
    for group,width in [('chelsa',19),('soilgrids',8),('topography',16),('human_footprint',9)]:
        member=next(m for m in inventory['members'] if m['name']==f'predictors/train_{group}_1km.h5')
        with BoundedHTTPFile(url,member['data_offset'],member['size'],budget=config['single_remote_raster_network_cap_bytes'],
                             block_size=256*1024,cache_dir=data/'training_predictor_cache'/group) as stream:
            with h5py.File(stream) as f:
                dataset=f['predictor_data'];values=np.empty((len(rows),width),dtype=np.float32)
                if dataset.shape!=(len(locations),width):raise ValueError('Training feature identity mismatch')
                values[order]=dataset[rows[order]]
            item={'group':group,'member':member['name'],'rows':len(rows),'width':width,
                  'new_network_bytes':stream.fetched_bytes,'new_range_requests':stream.requests,
                  'array_sha256':hashlib.sha256(values.tobytes()).hexdigest()}
        parts.append(values);results.append(item);print(json.dumps(item),flush=True)
    raw=np.full((len(cells),52),np.nan,dtype=np.float32);raw[valid]=np.concatenate(parts,axis=1)
    np.savez_compressed(data/'exact_cached_inputs.npz',raw=raw,valid=valid,
                        grid_x=cells.grid_x.to_numpy(),grid_y=cells.grid_y.to_numpy(),training_row=cells.nearest_training_row_to_cell_center.to_numpy(np.int64))
    cells['input_status']=np.where(valid,'exact_published_coordinate_cache','no_exact_published_coordinate_cache')
    cells.to_csv(data/'candidate_input_availability.csv',index=False)
    result={'status':'partial_full_candidate_inputs','candidate_cells':len(cells),'exact_all_52_feature_cells':int(valid.sum()),
            'candidate_plots_with_exact_inputs':int(cells.loc[valid,'n_plots'].sum()),
            'sites_with_exact_inputs':int(cells.loc[valid,'site'].nunique()),'remaining_cells':int((~valid).sum()),
            'raw_nan_values_in_valid_vectors':int(np.isnan(raw[valid]).sum()),
            'raw_infinite_values_in_valid_vectors':int(np.isinf(raw[valid]).sum()),
            'identical_float32_coordinates_verified':True,
            'method':'exact released SAGE training input rows at identical canonical EPSG:6933 cell centers; not nearest-point sampling',
            'groups':results,'bundle_sha256':hashlib.sha256((data/'exact_cached_inputs.npz').read_bytes()).hexdigest(),
            'NEON_species_outcomes_read':False,'ecological_study_admitted':False,
            'restriction':'The exact-cache subset is a training-support stratum, not a new-location generalization test; full 209-cell inputs remain incomplete.'}
    (out/'exact_input_bridge.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
