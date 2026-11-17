"""Extend region masks using same-region published training-mask probes."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geolifeclef.range_io import BoundedHTTPFile
from geolifeclef.native_region_bridge import assign_regions


def main():
    data,out=ROOT/'data/neon_input_followup',ROOT/'results/neon_input_followup'
    reader=shapefile.Reader(str(data/'tdwg_level3/level3'))
    shapes=list(reader.iterShapeRecords(fields=['LEVEL3_COD']))
    geometries=[shape(r.shape.__geo_interface__) for r in shapes];codes=[r.record.as_dict()['LEVEL3_COD'] for r in shapes]
    cells=pd.read_csv(data/'candidate_cells.csv');plots=pd.read_csv(ROOT/'data/neon_bridge/candidate_training_overlap.csv')
    probes=np.sort(plots.nearest_training_row_to_cell_center.unique().astype(np.int64))
    with h5py.File(ROOT/'data/neon_bridge/train_location_1km.h5') as f:
        location_count=f['predictor_data'].shape[0]
        locations=f['predictor_data'][probes]
    regions,fallback=assign_regions(geometries,codes,locations)
    inventory=json.loads((ROOT/'data/independent_data_audit/sage_sage_benchmark_data_tar_index.json').read_text())
    member=next(m for m in inventory['members'] if m['name']=='range_masks_1km/train_range_mask.h5')
    record=json.loads((ROOT/'data/external_replication_e1/zenodo_21297133.json').read_text())
    url=next(f for f in record['files'] if f['key']=='sage_benchmark_data.tar')['links']['self']
    lookup={};inconsistent=set();counts={}
    with BoundedHTTPFile(url,member['data_offset'],member['size'],budget=96*1024**2,block_size=256*1024,cache_dir=data/'training_mask_cache') as stream:
        with h5py.File(stream) as f:
            if f['cumulative_lengths'].shape[0]!=location_count+1:raise ValueError('Published training mask/locations differ')
            offset_rows=np.unique(np.r_[probes,probes+1]);offset_values=f['cumulative_lengths'][offset_rows]
            offsets=dict(zip(offset_rows,offset_values))
            for row,region in zip(probes,regions):
                ids=np.unique(f['species_index'][offsets[row]:offsets[row+1]])
                if np.any(ids<0) or np.any(ids>=5771):raise ValueError('Training mask class-order mismatch')
                counts[region]=counts.get(region,0)+1
                if region in lookup and not np.array_equal(ids,lookup[region]):inconsistent.add(region)
                lookup[region]=ids
        transferred=stream.fetched_bytes;requests=stream.requests
    if inconsistent:raise ValueError('Training mask is not constant for queried region: '+str(sorted(inconsistent)))
    prior=np.load(data/'candidate_native_masks.npz');mask=prior['native'].copy();valid=prior['valid'].copy()
    agreement=0
    for i,code in enumerate(cells.region):
        if code not in lookup:continue
        candidate=np.zeros(5771,dtype=bool);candidate[lookup[code]]=True
        if valid[i]:
            if not np.array_equal(mask[i],candidate):raise ValueError('Training and evaluation masks disagree in same TDWG region')
            agreement+=1
        mask[i]=candidate;valid[i]=True
    cells['native_mask_available']=valid
    cells['mask_reference_source']=np.where(prior['valid'],'all_same_region_eval_rows_plus_training_probes','same_region_training_probes')
    np.savez_compressed(data/'candidate_native_masks_full.npz',native=mask,valid=valid,
                        grid_x=cells.grid_x.to_numpy(),grid_y=cells.grid_y.to_numpy())
    cells.to_csv(data/'candidate_cells_native.csv',index=False)
    result={'status':'all_candidate_masks_verified' if valid.all() else 'partial','candidate_cells':len(cells),
            'valid_native_mask_cells':int(valid.sum()),'same_region_training_probe_rows':len(probes),
            'training_probes_per_region':{str(k):int(v) for k,v in counts.items()},
            'training_probe_polygon_fallbacks':int(fallback.sum()),'training_eval_mask_agreement_cells':agreement,
            'training_regions_inconsistent':sorted(inconsistent),'new_network_bytes':transferred,'new_range_requests':requests,
            'method':'published region-constant mask construction, all 53336 eval rows verified; same-region training probes extend uncovered regions and cross-check overlapping regions',
            'no_nearest_mask_substitution':True,'new_native_polygons_reconstructed_from_POWO':False,
            'scope_limit':'Uses released native-mask artifacts and verified TDWG region identity; not a fresh botanical range validation.',
            'mask_bundle_sha256':hashlib.sha256((data/'candidate_native_masks_full.npz').read_bytes()).hexdigest(),
            'NEON_composition_values_read':False,'ecological_study_admitted':False}
    (out/'native_mask_extension.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
