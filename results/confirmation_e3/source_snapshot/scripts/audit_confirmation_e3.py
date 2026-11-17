"""Find uninspected source/domain cohorts using metadata only, without PA targets."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geolifeclef.external_reliability import read_rows

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    config=json.loads((ROOT/'configs/confirmation_e3_audit.json').read_text())
    source=ROOT/'data/external_replication_e1';data=ROOT/'data/confirmation_e3';out=ROOT/'results/confirmation_e3'
    data.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    columns=['cell_row','PlotObservationID','GIVD_ID','Country','Continent','Date_of_recording',
        'Latitude','Longitude','Releve_area','Location_uncertainty']
    metadata_path=ROOT/'data/independent_data_audit/sage_eval_plot_metadata.csv'
    frame=pd.read_csv(metadata_path,usecols=columns)
    consumed=pd.read_csv(source/'cohort_roles.csv')
    seen=frame[frame.cell_row.isin(consumed.cell_id)]
    consumed_sources=set(seen.GIVD_ID.dropna());consumed_plots=set(seen.PlotObservationID.dropna())
    old=pd.concat([pd.read_csv(ROOT/'data/raw'/name,usecols=['lat','lon']) for name in
        ['GLC25_PA_metadata_train.csv','GLC25_PA_metadata_test.csv']])
    finite=np.isfinite(frame[['Latitude','Longitude']]).all(1)
    distance=np.full(len(frame),np.nan)
    distance[finite]=BallTree(np.deg2rad(old[['lat','lon']]),metric='haversine').query(
        np.deg2rad(frame.loc[finite,['Latitude','Longitude']]))[0][:,0]*6371.0088
    common=(finite & frame.Continent.notna() & frame.Continent.ne('Europe') & frame.Country.notna() &
        frame.GIVD_ID.notna() & ~frame.GIVD_ID.astype(str).str.startswith('EU-') &
        frame.PlotObservationID.notna() & ~frame.PlotObservationID.isin(consumed_plots) &
        frame.Location_uncertainty.between(0,config['maximum_plot_uncertainty_m']) &
        (distance>=config['minimum_existing_GLC_PA_distance_km']))
    source_valid=common & frame.Releve_area.between(*config['source_cohort_plot_area_m2']) & ~frame.GIVD_ID.isin(consumed_sources)
    domain_valid=common & frame.Releve_area.gt(config['large_plot_domain_area_m2'][0]) & frame.Releve_area.le(config['large_plot_domain_area_m2'][1])
    frame['year']=pd.to_numeric(frame.Date_of_recording.astype(str).str.extract(r'(\d{4})')[0],errors='coerce')
    grouped=frame.groupby('cell_row')
    table=grouped.agg(country=('Country','first'),continent=('Continent','first'),area=('Releve_area','sum'),
        geo_uncertainty=('Location_uncertainty','max'),year=('year','median'),
        plots=('PlotObservationID','nunique'),sources=('GIVD_ID',lambda x:'|'.join(sorted(x.dropna().unique()))),
        minimum_plot_area=('Releve_area','min'),maximum_plot_area=('Releve_area','max'))
    table['new_source']=source_valid.groupby(frame.cell_row).all()
    table['large_plot_domain']=domain_valid.groupby(frame.cell_row).all()
    table['coherent']=grouped.Country.nunique().eq(1)&grouped.Continent.nunique().eq(1)
    test=np.load(source/'test_indices.npy',allow_pickle=False)
    lookup=pd.Series(np.arange(len(test)),index=test)
    table=table[table.coherent & table.index.isin(test) & ~table.index.isin(consumed.cell_id) &
        (table.new_source|table.large_plot_domain)].sort_index().copy()
    table['prediction_row']=lookup.reindex(table.index).to_numpy()
    with h5py.File(source/'predictors/eval_location.h5') as f:locations=f['predictor_data'][:]
    table['lon']=locations[table.index,0];table['lat']=locations[table.index,1]
    table['spatial_block']=[f'{int(np.floor(lat/5))}:{int(np.floor(lon/5))}' for lat,lon in table[['lat','lon']].to_numpy()]
    distance_old=BallTree(np.deg2rad(consumed[['lat','lon']]),metric='haversine').query(
        np.deg2rad(table[['lat','lon']]))[0][:,0]*6371.0088
    table['distance_consumed_km']=distance_old
    table['unseen_block']=~table.spatial_block.isin(set(consumed.spatial_block))
    # Locations and effort are covariates. Do not index the targets dataset.
    with h5py.File(source/'sealed_eval_test_targets.h5') as f:
        published_area=read_rows(f['sampled_area_m2'],table.prediction_row.to_numpy(int))
        published_location=read_rows(f['locations'],table.prediction_row.to_numpy(int))
    assert np.allclose(published_location,table[['lon','lat']],atol=1e-5,rtol=0)
    table['area_consistent']=np.isclose(published_area,table.area,atol=config['area_consistency_atol_m2'],rtol=config['area_consistency_rtol'])
    before=len(table)
    table.to_csv(data/'candidate_covariate_audit_all.csv',index_label='cell_id')
    admitted=table[table.area_consistent & table.unseen_block & table.distance_consumed_km.ge(config['minimum_consumed_SAGE_distance_km'])].copy()
    admitted.to_csv(data/'candidate_cohorts.csv',index_label='cell_id')
    summary={}
    for name,mask in [('new_source',admitted.new_source),('large_plot_domain',admitted.large_plot_domain)]:
        cohort=admitted.loc[mask]
        summary[name]={'cells':len(cohort),'blocks':int(cohort.spatial_block.nunique()),
            'sources':int(cohort.sources.nunique()),'countries':cohort.country.value_counts().to_dict(),
            'continents':cohort.continent.value_counts().to_dict(),
            'source_counts':cohort.groupby('sources').size().sort_values(ascending=False).to_dict(),
            'minimum_distance_consumed_km':float(cohort.distance_consumed_km.min()) if len(cohort) else None,
            'cell_ID_sha256':hashlib.sha256(np.asarray(cohort.index,'<i8').tobytes()).hexdigest()}
    result={'status':'metadata_only_feasibility_completed','at_utc':datetime.now(timezone.utc).isoformat(),
        'config':config,'metadata_sha256':digest(metadata_path),'script_sha256':digest(Path(__file__)),
        'consumed_cells':len(consumed),'consumed_sources':len(consumed_sources),
        'pre_distance_area_block_admission_cells':before,'post_admission_cells':len(admitted),
        'outcome_rows_requested':0,'prediction_values_read':False,'cohorts':summary,
        'scope':'uninspected analysis subset within published SAGE, not a new independent data release; published backbone overlap unchanged'}
    (out/'metadata_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
