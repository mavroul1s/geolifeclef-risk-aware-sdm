"""Freeze reliability heads, then evaluate the uninspected E3 domain exactly once."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import BallTree

from run_external_e1 import check_mask,save,sha,summarize
from geolifeclef.confirmation import cell_digest,confirmation_success,confirmation_targets,paired_confirmation
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.external_reliability import (META_FEATURES,RISK_FEATURES,apply_calibration,
    decision_features,decision_outcomes,decode_counts,fit_calibration,guarded_target_rows,prediction_context,read_rows)
from geolifeclef.ranking_selection import monotone_affine
from geolifeclef.reliability import r1_hash

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/external_replication_e1'
DATA=ROOT/'data/confirmation_e3'
OUT=ROOT/'results/confirmation_e3'
METHODS=['SDC','HGB','mean_confidence','entropy','random']
SOURCES=['configs/confirmation_e3.json','configs/confirmation_e3_audit.json',
    'configs/external_reliability_e1.json','docs/confirmation_e3_protocol.md',
    'scripts/run_confirmation_e3.py','scripts/audit_confirmation_e3.py',
    'scripts/run_external_e1.py','src/geolifeclef/confirmation.py',
    'src/geolifeclef/external_reliability.py','src/geolifeclef/ecological_selection.py',
    'src/geolifeclef/ranking_selection.py','src/geolifeclef/reliability.py']


def now():return datetime.now(timezone.utc).isoformat()


def prepare(config):
    table=pd.read_csv(ROOT/config['cohort_file'])
    table=table.loc[table[config['cohort_flag']]].sort_values('cell_id').reset_index(drop=True)
    consumed=pd.read_csv(SOURCE/'cohort_roles.csv')
    if len(table)<config['minimum_cells'] or table.spatial_block.nunique()<config['minimum_spatial_blocks']:
        raise ValueError('Covariate admission failed')
    if set(table.cell_id)&set(consumed.cell_id) or set(table.spatial_block)&set(consumed.spatial_block):
        raise ValueError('Consumed confirmation identity/block')
    if not table.distance_consumed_km.ge(config['minimum_prior_SAGE_distance_km']).all():
        raise ValueError('Spatial distance admission failed')
    test_indices=np.load(SOURCE/'test_indices.npy',allow_pickle=False)
    if not np.array_equal(test_indices[table.prediction_row],table.cell_id):raise ValueError('Test row identity mismatch')
    prob=consumed[consumed.role.eq('probability_calibration')]
    edges=json.loads((ROOT/'results/external_replication_e1/cohort_design.json').read_text())['area_edges_log1p']
    table['log_area']=np.log1p(table.area)
    table['log_geo_uncertainty']=np.log1p(table.geo_uncertainty)
    table['area_bin']=np.searchsorted(edges,table.log_area,side='right').astype(str)
    new_edges=np.unique(np.quantile(table.log_area,[.2,.4,.6,.8]))
    table['new_area_bin']=np.searchsorted(new_edges,table.log_area,side='right').astype(str)
    tree=BallTree(np.deg2rad(prob[['lat','lon']]),metric='haversine')
    distance=tree.query(np.deg2rad(table[['lat','lon']]))[0][:,0]*6371.0088
    table['log_calibration_distance']=np.log1p(distance)
    table.to_csv(DATA/'locked_cohort.csv',index=False)
    rare=np.load(SOURCE/'rare_species_mask.npy',allow_pickle=False)
    with h5py.File(SOURCE/'range_masks_1km/eval_range_mask.h5') as f:
        offsets=f['cumulative_lengths'][:];species=f['species_index'][:]
    mask=np.zeros((len(table),len(rare)),bool)
    for i,cell in enumerate(table.cell_id):mask[i,species[offsets[cell]:offsets[cell+1]]]=True
    np.save(DATA/'eligible_species.npy',mask,allow_pickle=False)
    save(OUT/'cohort_design.json',{'prepared_at_utc':now(),'cells':len(table),
        'blocks':int(table.spatial_block.nunique()),'cohort_ID_sha256':cell_digest(table.cell_id),
        'countries':table.country.value_counts().to_dict(),'sources':table.sources.value_counts().to_dict(),
        'original_area_bin_counts':table.area_bin.value_counts().to_dict(),
        'secondary_area_edges_log1p':new_edges.tolist(),'minimum_distance_consumed_km':float(table.distance_consumed_km.min()),
        'outcome_values_read':False,'prediction_values_read':False})
    return table


def freeze():
    hashes={name:sha(ROOT/name) for name in SOURCES}
    inputs={name:sha(DATA/name) for name in ['locked_cohort.csv','eligible_species.npy']}
    prior={name:sha(SOURCE/name) for name in ['cohort_roles.csv','eligible_species.npy',
        'rare_species_mask.npy','sealed_eval_test_targets.h5']}
    path=OUT/'protocol_lock.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['source_hashes']!=hashes or old['input_hashes']!=inputs or old['prior_input_hashes']!=prior:
            raise RuntimeError('E3 lock changed; preserve original and explicitly amend before continuation')
        return
    for name in SOURCES:
        target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    save(path,{'locked_at_utc':now(),'source_hashes':hashes,'input_hashes':inputs,
        'prior_input_hashes':prior,'E3_outcomes_accessed':False,
        'metadata_audit_sha256':sha(OUT/'metadata_audit.json')})


def fit(config):
    if (OUT/'evaluation_admission.json').exists():raise ValueError('E3 policies already frozen; no refitting')
    table=prepare(config) if not (DATA/'locked_cohort.csv').exists() else pd.read_csv(DATA/'locked_cohort.csv',dtype={'area_bin':str,'new_area_bin':str})
    freeze()
    meta=pd.read_csv(SOURCE/'cohort_roles.csv')
    cal=np.flatnonzero(meta.role.eq('probability_calibration'));risk=np.flatnonzero(meta.role.eq('risk_calibration'))
    old_mask=np.load(SOURCE/'eligible_species.npy',allow_pickle=False)
    new_mask=np.load(DATA/'eligible_species.npy',allow_pickle=False)
    rare=np.load(SOURCE/'rare_species_mask.npy',allow_pickle=False)
    with h5py.File(SOURCE/'sealed_eval_test_targets.h5') as f:
        yc=guarded_target_rows(f['targets'],meta,'probability_calibration','fit')
        yr=guarded_target_rows(f['targets'],meta,'risk_calibration','fit')
    check_mask(yc,old_mask[cal]);check_mask(yr,old_mask[risk])
    prior=json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    masks=np.concatenate([old_mask[risk],new_mask])
    prediction_rows=np.concatenate([meta.iloc[cal].prediction_row,meta.iloc[risk].prediction_row,table.prediction_row]).astype(int)
    policies={}
    for model in config['models']:
        start=time.monotonic();folder=OUT/'policies';folder.mkdir(exist_ok=True)
        path=folder/f'{model}.npz';description=path.with_suffix('.json')
        if path.exists() and description.exists():
            policies[model]={'npz':sha(path),'json':sha(description)};continue
        with h5py.File(SOURCE/model/'test_predictions.h5') as f:
            raw=read_rows(f['probabilities'],prediction_rows)
        transform=fit_calibration(raw[:len(cal)],yc,old_mask[cal],prior['calibration_bounds'])
        expected=json.loads((ROOT/'results/external_replication_e1/policies'/f'{model}.json').read_text())['calibration']
        for name in ['slope','intercept']:
            if not np.isclose(transform[name],expected[name],atol=1e-10,rtol=1e-10):raise ValueError('Original Platt transform failed reproduction')
        probability=apply_calibration(raw[len(cal):],transform,masks)
        rctx=prediction_context(probability[:len(risk)],old_mask[risk])
        ctx=prediction_context(probability[len(risk):],new_mask)
        rc=decode_counts(rctx,config['decoder']);counts=decode_counts(ctx,config['decoder'])
        rf=decision_features(rctx,rc,meta.iloc[risk][META_FEATURES[:-1]],rare)
        tf=decision_features(ctx,counts,table[META_FEATURES[:-1]],rare)
        ro=decision_outcomes(yr,old_mask[risk],rctx['rank'],rc,rare)
        reg=HistGradientBoostingRegressor(**prior['risk_regressor']).fit(rf[RISK_FEATURES],ro.loss)
        hgb=1-reg.predict(tf[RISK_FEATURES])
        affine,audit=monotone_affine(rf.SDC,ro.sample_F1,tf.SDC)
        scores={'SDC':tf.SDC.to_numpy(),'HGB':hgb,'mean_confidence':tf.mean_set_confidence.to_numpy(),
            'entropy':-tf.set_entropy.to_numpy(),'random':np.array([r1_hash('E3:'+str(i))/2**64 for i in table.cell_id])}
        keys=[];accepted=[]
        for scheme,areas,coverages in [('primary',table.area_bin,[1.,.9,.8,.6]),('new_area',table.new_area_bin,[.8])]:
            for coverage in coverages:
                selector=QuotaSelection(table.cell_id,coverage,table.country.astype(str)+'|'+areas.astype(str))
                for method,score in scores.items():
                    take=selector.accept(-score)
                    for ix,quota in selector.groups:assert int(take[ix].sum())==quota
                    keys.append(f'{scheme}|{coverage}|{method}');accepted.append(take)
        np.savez_compressed(path,cell_id=table.cell_id.to_numpy(),rank=ctx['rank'].astype(np.int16),counts=counts,
            SDC=scores['SDC'],affine_quality_raw=affine,HGB_quality_raw=hgb,
            affine_quality=np.clip(affine,0,1),HGB_quality=np.clip(hgb,0,1),
            train_SDC=rf.SDC.to_numpy(),train_quality=ro.sample_F1.to_numpy(),
            keys=np.array(keys),accept=np.array(accepted))
        save(description,{'model':model,'frozen_at_utc':now(),'probability_calibration':transform,
            'affine_quality_calibration':audit,'HGB_parameters':prior['risk_regressor'],
            'risk_features':RISK_FEATURES,'training_cell_ID_sha256':cell_digest(meta.iloc[risk].cell_id),
            'E3_outcomes_accessed':False,'seconds':time.monotonic()-start})
        policies[model]={'npz':sha(path),'json':sha(description)}
        print(json.dumps({'model':model,'stage':'policy_frozen','seconds':time.monotonic()-start}),flush=True)
    save(OUT/'evaluation_admission.json',{'admitted_at_utc':now(),'all_policies_frozen':True,
        'policy_hashes':policies,'cohort_ID_sha256':cell_digest(table.cell_id),
        'protocol_lock_sha256':sha(OUT/'protocol_lock.json'),'E3_outcomes_accessed':False})
    print('All four E3 policies frozen; E3 target values remain closed.',flush=True)


def paired(config,q,p,r,a,b,blocks):
    boot=config['bootstrap']
    return paired_confirmation(q,p,r,a,b,blocks,draws=boot['draws'],
        confidence=boot['confidence_each_endpoint'],seed=boot['seed'])


def assess(config):
    freeze()
    if (OUT/'report.json').exists():raise ValueError('E3 assessment complete; do not tune/retest')
    admission=json.loads((OUT/'evaluation_admission.json').read_text())
    if sha(OUT/'protocol_lock.json')!=admission['protocol_lock_sha256']:raise ValueError('Protocol changed')
    for model,hashes in admission['policy_hashes'].items():
        for extension,digest in hashes.items():
            if sha(OUT/'policies'/f'{model}.{extension}')!=digest:raise ValueError('Frozen policy changed')
    table=pd.read_csv(DATA/'locked_cohort.csv',dtype={'area_bin':str,'new_area_bin':str})
    consumed=pd.read_csv(SOURCE/'cohort_roles.csv')
    access=OUT/'assessment_access.json'
    if not access.exists():save(access,{'first_E3_target_access_utc':now(),
        'all_policies_previously_frozen':True,'admission_sha256':sha(OUT/'evaluation_admission.json')})
    with h5py.File(SOURCE/'sealed_eval_test_targets.h5') as f:
        y=confirmation_targets(f['targets'],table,consumed.cell_id,admission,config['models'])
    eligible=np.load(DATA/'eligible_species.npy',allow_pickle=False);check_mask(y,eligible)
    rare=np.load(SOURCE/'rare_species_mask.npy',allow_pickle=False)
    rare_total=int((y[:,rare]>0).sum())
    qualities=[];candidate=[];baseline=[];pa=[];pb=[];raw_pa=[];raw_pb=[];records=[];regions=[]
    secondary_a=[];secondary_b=[];gpu={}
    for model in config['models']:
        with np.load(OUT/'policies'/f'{model}.npz',allow_pickle=False) as policy:
            if not np.array_equal(policy['cell_id'],table.cell_id):raise ValueError('Policy row identity changed')
            outcome=decision_outcomes(y,eligible,policy['rank'],policy['counts'],rare)
            keys=policy['keys'].tolist()
            q=outcome.sample_F1.to_numpy()
            a=policy['accept'][keys.index('primary|0.8|SDC')];b=policy['accept'][keys.index('primary|0.8|HGB')]
            qualities.append(q);candidate.append(a);baseline.append(b)
            pa.append(policy['affine_quality']);pb.append(policy['HGB_quality'])
            raw_pa.append(policy['affine_quality_raw']);raw_pb.append(policy['HGB_quality_raw'])
            secondary_a.append(policy['accept'][keys.index('new_area|0.8|SDC')])
            secondary_b.append(policy['accept'][keys.index('new_area|0.8|HGB')])
            for key,accepted in zip(keys,policy['accept']):
                scheme,coverage,method=key.split('|')
                summary=summarize(outcome,policy['counts'],accepted,rare_total)
                estimate=policy['affine_quality'] if method=='SDC' else policy['HGB_quality'] if method=='HGB' else None
                records.append({'model':model,'quota_scheme':scheme,'coverage':float(coverage),'method':method,
                    'all_cell_quality_MSE':float(np.mean((estimate-q)**2)) if estimate is not None else None,
                    'raw_SDC_quality_MSE':float(np.mean((policy['SDC']-q)**2)),**summary})
            for dim in ['country','continent','sources']:
                for name,ix in table.groupby(dim).groups.items():
                    ix=np.asarray(ix);sub=outcome.iloc[ix].reset_index(drop=True)
                    for method,accept,estimate in [('SDC',a,policy['affine_quality']),('HGB',b,policy['HGB_quality'])]:
                        take=accept[ix]
                        metric=(summarize(sub,policy['counts'][ix],take,int(sub.rare_truth.sum())) if take.any()
                            else {'retained':0,'sample_F1':None})
                        regions.append({'model':model,'dimension':dim,'name':name,'cells':len(ix),
                            'blocks':int(table.iloc[ix].spatial_block.nunique()),'method':method,
                            'quality_MSE':float(np.mean((estimate[ix]-q[ix])**2)),**metric})
            gpu[model]={'train_SDC':policy['train_SDC'].copy(),'train_quality':policy['train_quality'].copy(),
                'test_SDC':policy['SDC'].copy(),'test_quality':q,'test_HGB_quality':policy['HGB_quality'].copy(),
                'candidate_accept':a,'baseline_accept':b}
    qualities=np.array(qualities);pa=np.array(pa);pb=np.array(pb);candidate=np.array(candidate);baseline=np.array(baseline)
    primary=paired(config,qualities,pa,pb,candidate,baseline,table.spatial_block)
    success,checks=confirmation_success(primary,config)
    secondary={'new_area_quotas':paired(config,qualities,pa,pb,secondary_a,secondary_b,table.spatial_block),
        'unclipped_magnitude':paired(config,qualities,raw_pa,raw_pb,candidate,baseline,table.spatial_block)}
    for name,mask in [('non_Brazil',table.country.ne('Brazil').to_numpy()),('Brazil',table.country.eq('Brazil').to_numpy())]:
        if mask.sum() and candidate[:,mask].sum(1).all() and baseline[:,mask].sum(1).all():
            value=paired(config,qualities[:,mask],pa[:,mask],pb[:,mask],candidate[:,mask],baseline[:,mask],table.spatial_block[mask])
            value['too_few_blocks_for_source_population_inference']=value['clusters']<10
            secondary[name]=value
    pd.DataFrame(records).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(regions).to_csv(OUT/'regional_summary.csv',index=False)
    arrays={f'{model}__{key}':value for model,items in gpu.items() for key,value in items.items()}
    np.savez_compressed(OUT/'GPU_replay_inputs.npz',**arrays)
    tests=ET.parse(OUT/'focused_tests.xml').getroot().find('testsuite').attrib
    correct=tests['failures']==tests['errors']=='0'
    save(OUT/'report.json',{'status':'completed_locked_domain_confirmation','completed_at_utc':now(),
        'primary_success':success,'primary_checks':checks,'primary_comparison':primary,
        'secondary_comparisons':secondary,'assessment_cells':len(table),'rare_occurrences':rare_total,
        'evidence_scope':config['evidence_scope'],'correctness_tests_passed':correct,
        'GPU_scientifically_admitted':bool(success and correct),'GPU_job_launched':False,
        'user_GPU_authorization_recorded':True,'new_neural_backbone_fits':0,
        'E3_outcomes_now_consumed':True,'original_E1_E2_assessment_results_modified':False,
        'limitations':['Published benchmark subset, not a new independent data release',
            'Some GIVD source IDs overlap calibration; pretrained PO backbone overlap unchanged',
            'Brazil contributes many cells in only two spatial blocks',
            'Primary intervals condition on fixed models and heads; secondary findings cannot replace primary']})
    print(json.dumps({'primary_success':success,'checks':checks,'primary':primary,
        'GPU_scientifically_admitted':bool(success and correct)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['fit','assess'])
    args=parser.parse_args();DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/confirmation_e3.json').read_text())
    (fit if args.phase=='fit' else assess)(config)


if __name__=='__main__':main()
