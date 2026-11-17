"""Audit E3 outcome separation, locked policies, primary numbers and source provenance."""
from datetime import datetime,timezone
import json
import shutil
import xml.etree.ElementTree as ET

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from run_confirmation_e3 import OUT,ROOT,SOURCE,DATA,freeze,paired,save,sha
from geolifeclef.confirmation import cell_digest,confirmation_success
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.external_reliability import read_rows


def main():
    freeze();lock=json.loads((OUT/'protocol_lock.json').read_text())
    for name,digest in lock['source_hashes'].items():assert sha(OUT/'source_snapshot'/name)==digest
    admission=json.loads((OUT/'evaluation_admission.json').read_text())
    access=json.loads((OUT/'assessment_access.json').read_text())
    assert access['admission_sha256']==sha(OUT/'evaluation_admission.json')
    assert admission['protocol_lock_sha256']==sha(OUT/'protocol_lock.json')
    assert lock['locked_at_utc']<admission['admitted_at_utc']<access['first_E3_target_access_utc']
    cohort=pd.read_csv(DATA/'locked_cohort.csv',dtype={'area_bin':str,'new_area_bin':str})
    consumed=pd.read_csv(SOURCE/'cohort_roles.csv')
    assert cell_digest(cohort.cell_id)==admission['cohort_ID_sha256']
    assert not set(cohort.cell_id)&set(consumed.cell_id)
    assert not set(cohort.spatial_block)&set(consumed.spatial_block)
    distance=BallTree(np.deg2rad(consumed[['lat','lon']]),metric='haversine').query(
        np.deg2rad(cohort[['lat','lon']]))[0].min()*6371.0088
    assert distance>=100
    metadata=pd.read_csv(ROOT/'data/independent_data_audit/sage_eval_plot_metadata.csv',
        usecols=['cell_row','PlotObservationID','Latitude','Longitude','Releve_area','Location_uncertainty','Continent','GIVD_ID'])
    new=metadata[metadata.cell_row.isin(cohort.cell_id)]
    old=metadata[metadata.cell_row.isin(consumed.cell_id)]
    assert not set(new.PlotObservationID)&set(old.PlotObservationID)
    assert new.Releve_area.gt(400).all() and new.Releve_area.le(10000).all()
    assert new.Location_uncertainty.between(0,1000).all()
    assert new.Continent.notna().all() and new.Continent.ne('Europe').all()
    assert new.GIVD_ID.notna().all() and not new.GIVD_ID.str.startswith('EU-').any()
    reference=pd.concat([pd.read_csv(ROOT/'data/raw'/name,usecols=['lat','lon']) for name in
        ['GLC25_PA_metadata_train.csv','GLC25_PA_metadata_test.csv']])
    nearest=BallTree(np.deg2rad(reference[['lat','lon']]),metric='haversine').query(np.deg2rad(new[['Latitude','Longitude']]))[0].min()*6371.0088
    assert nearest>=500
    with h5py.File(SOURCE/'sealed_eval_test_targets.h5') as f:
        locations=read_rows(f['locations'],cohort.prediction_row.to_numpy())
        areas=read_rows(f['sampled_area_m2'],cohort.prediction_row.to_numpy())
    assert np.allclose(locations,cohort[['lon','lat']],atol=1e-5,rtol=0)
    assert np.allclose(areas,cohort.area,atol=.1,rtol=1e-4)
    config=json.loads((ROOT/'configs/confirmation_e3.json').read_text())
    q=[];p=[];r=[];a=[];b=[];settings=0
    with np.load(OUT/'GPU_replay_inputs.npz',allow_pickle=False) as inp:
        for model,hashes in admission['policy_hashes'].items():
            for extension,digest in hashes.items():assert sha(OUT/'policies'/f'{model}.{extension}')==digest
            detail=json.loads((OUT/'policies'/f'{model}.json').read_text())
            assert detail['frozen_at_utc']<admission['admitted_at_utc']
        for model in config['models']:
            with np.load(OUT/'policies'/f'{model}.npz',allow_pickle=False) as policy:
                assert np.array_equal(policy['cell_id'],cohort.cell_id)
                keys=policy['keys'].tolist()
                for key,accept in zip(keys,policy['accept']):
                    scheme,coverage,method=key.split('|')
                    areas=cohort.area_bin if scheme=='primary' else cohort.new_area_bin
                    selector=QuotaSelection(cohort.cell_id,float(coverage),cohort.country.astype(str)+'|'+areas.astype(str))
                    for take,quota in selector.groups:assert int(accept[take].sum())==quota
                    settings+=1
                q.append(inp[model+'__test_quality']);p.append(policy['affine_quality']);r.append(policy['HGB_quality'])
                a.append(policy['accept'][keys.index('primary|0.8|SDC')]);b.append(policy['accept'][keys.index('primary|0.8|HGB')])
    report=json.loads((OUT/'report.json').read_text());result=paired(config,q,p,r,a,b,cohort.spatial_block)
    for name in ['MSE_gain','F1_gain','MSE_interval','F1_interval']:
        np.testing.assert_allclose(result[name],report['primary_comparison'][name],atol=1e-14)
    success,checks=confirmation_success(result,config)
    assert success==report['primary_success'] and checks==report['primary_checks']
    old_manifests={}
    for dirname in ['external_replication_e1','ranking_study_e2']:
        files=json.loads((ROOT/'results'/dirname/'artifact_manifest.json').read_text())['files']
        for name,item in files.items():assert sha(ROOT/'results'/dirname/name)==item['sha256']
        old_manifests[dirname]=len(files)
    tests=ET.parse(OUT/'focused_tests.xml').getroot().find('testsuite').attrib
    assert tests['failures']==tests['errors']=='0'
    for name in ['scripts/diagnose_confirmation_e3.py','scripts/summarize_confirmation_e3.py',
                 'src/geolifeclef/calibration_diagnostics.py','scripts/validate_confirmation_e3.py']:
        path=OUT/'posthoc_source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    save(OUT/'validation.json',{'status':'passed','at_utc':datetime.now(timezone.utc).isoformat(),
        'tests':{k:tests[k] for k in ['tests','failures','errors','skipped']},
        'policies_frozen_before_outcomes':True,'source_and_cohort_hashes_verified':True,
        'no_consumed_cells_plots_or_blocks':True,'minimum_distance_consumed_SAGE_km':float(distance),
        'minimum_linked_plot_distance_existing_GLC_PA_km':float(nearest),
        'target_coordinate_and_area_alignment':True,'native_target_mask_check_passed_at_assessment':True,
        'verified_policy_settings':settings,'all_group_quotas_exact':True,
        'primary_numbers_exactly_reproduced':True,'previous_manifests_unchanged':old_manifests,
        'scientific_primary_success':success,'GPU_job_launched':False,'new_neural_backbone_fits':0})
    save(OUT/'artifact_manifest.json',{'files':{str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}})
    # NEON readiness is deliberately separate from successful metadata retrieval.
    neon=ROOT/'results/neon_confirmation_feasibility';npdata=ROOT/'data/neon_confirmation_feasibility'
    audit=json.loads((neon/'audit.json').read_text());sites=pd.read_csv(npdata/'site_metadata_projection.csv')
    assert len(sites)==sites.siteCode.nunique()==audit['sites']==47
    assert sites.siteLatitude.between(-90,90).all() and sites.siteLongitude.between(-180,180).all()
    assert sha(npdata/'plant_product.json')==audit['product_metadata_sha256']
    assert sha(npdata/'site_metadata_projection.csv')==audit['site_projection_sha256']
    save(neon/'validation.json',{'status':'metadata_only_passed','sites':47,'outcomes_read':False,
        'prediction_ready':False,'plot_level_nonoverlap_established':False,
        'metadata_hashes_match':True,'required_next_step':'plot/taxonomy/environmental predictor bridge and reference inference tests'})
    print(json.dumps({'status':'passed','tests':tests['tests'],'policy_settings':settings,
        'primary_success':success,'NEON_metadata_sites':47,'NEON_prediction_ready':False}))


if __name__=='__main__':main()
