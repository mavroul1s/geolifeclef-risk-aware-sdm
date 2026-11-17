"""Prepare outcome-free E4 inputs, masks, roles and GPU payload."""
import hashlib
import json
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT=Path(__file__).resolve().parents[1]


def main():
    data,out=ROOT/'data/confirmation_e4',ROOT/'results/confirmation_e4'
    data.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    if (out/'protocol_freeze.json').exists():raise ValueError('E4 already frozen; do not rebuild inputs')
    cells=pd.read_csv(ROOT/'data/neon_input_followup/candidate_input_availability.csv')
    inputs=np.load(ROOT/'data/neon_input_followup/exact_cached_inputs.npz');masks=np.load(ROOT/'data/neon_input_followup/candidate_native_masks_full.npz')
    valid=inputs['valid'];np.testing.assert_array_equal(inputs['grid_x'],masks['grid_x']);np.testing.assert_array_equal(inputs['grid_y'],masks['grid_y'])
    assert masks['valid'].all() and valid.sum()==163
    neon=cells.loc[valid].copy().reset_index(drop=True)
    source=pd.read_csv(ROOT/'data/external_replication_e1/cohort_roles.csv')
    keep=source.role.ne('assessment');source=source.loc[keep].copy().reset_index(drop=True)
    eligible=np.load(ROOT/'data/external_replication_e1/eligible_species.npy',mmap_mode='r')[keep]
    tax=pd.read_csv(ROOT/'data/neon_bridge/neon_to_sage_taxonomy.csv')
    universe=np.zeros(5771,dtype=bool);universe[tax.sage_index.dropna().astype(int).unique()]=True
    source_eligible=eligible & universe
    native=masks['native'][valid]&universe
    source_raw=[]
    for group in ['chelsa','soilgrids','topography','human_footprint']:
        with h5py.File(ROOT/'data/neon_bridge/reference_predictors'/f'eval_{group}.h5') as f:
            source_raw.append(f['predictor_data'][:][source.cell_id.to_numpy(int)])
    raw=np.concatenate(source_raw,axis=1)
    plot=pd.read_csv(ROOT/'data/neon_bridge/full_plot_metadata_candidates.csv')
    uncertainty=plot.groupby(['grid_x','grid_y']).max_coordinate_uncertainty_m.max()
    neon['geo_uncertainty']=[uncertainty.loc[(r.grid_x,r.grid_y)] for r in neon.itertuples()]
    neon['area']=400*neon.n_plots;neon['year']=2023
    probability=source[source.role.eq('probability_calibration')]
    distance=BallTree(np.deg2rad(probability[['lat','lon']].to_numpy(float)),metric='haversine').query(np.deg2rad(neon[['lat','lon']].to_numpy(float)))[0][:,0]*6371.0088
    neon['log_area']=np.log1p(neon.area);neon['log_geo_uncertainty']=np.log1p(neon.geo_uncertainty)
    neon['log_calibration_distance']=np.log1p(distance);neon['cell_id']=[f'E4:{r.grid_x}:{r.grid_y}' for r in neon.itertuples()]
    neon['role']='assessment';neon['area_group']=np.where(neon.n_plots.eq(1),'one_plot','multiple_plots')
    source.to_csv(data/'source_calibration_roles.csv',index=False);neon.to_csv(data/'neon_assessment_roles.csv',index=False)
    selected=plot.merge(neon[['grid_x','grid_y','cell_id']],on=['grid_x','grid_y'],how='inner',validate='many_to_one')
    assert len(selected)==301 and selected.plot_id.nunique()==301
    selected.to_csv(data/'selected_plots.csv',index=False)
    reference=np.load(ROOT/'data/neon_bridge/cpu_reference.npz')
    np.savez_compressed(data/'input_bundle.npz',reference=reference['aggregation_s0__raw'],source=raw,
                        neon=inputs['raw'][valid],means=reference['aggregation_s0__means'],stds=reference['aggregation_s0__stds'])
    np.savez_compressed(data/'label_contract.npz',source_eligible=source_eligible,neon_eligible=native,universe=universe,
                        rare=np.load(ROOT/'data/external_replication_e1/rare_species_mask.npy')&universe)
    result={'status':'inputs_and_roles_prepared_outcomes_closed','source_roles':source.role.value_counts().astype(int).to_dict(),
            'NEON_assessment_cells':len(neon),'NEON_plots':len(selected),'NEON_sites':int(neon.site.nunique()),
            'minimum_distance_probability_calibration_km':float(distance.min()),'label_universe_classes':int(universe.sum()),
            'native_eligible_class_range':[int(native.sum(1).min()),int(native.sum(1).max())],
            'source_assessment_rows_included':False,'NEON_composition_values_read':False,
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir()) if p.is_file()}}
    (out/'input_preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
