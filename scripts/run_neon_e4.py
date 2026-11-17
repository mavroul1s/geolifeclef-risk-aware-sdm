"""Locally frozen NEON source-transfer confirmation; no assessment-driven refits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geolifeclef.external_reliability import META_FEATURES,RISK_FEATURES,fit_calibration,apply_calibration,prediction_context,decode_counts,decision_features,decision_outcomes,guarded_target_rows
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.reliability import r1_hash
from geolifeclef.neon_assessment import selected_csv_rows,TaxonomicAssessment,score_fixed_sets,site_primary_comparison

DATA,OUT=ROOT/'data/confirmation_e4',ROOT/'results/confirmation_e4'
SOURCES=['configs/confirmation_e4.json','docs/confirmation_e4_protocol.md','scripts/prepare_neon_e4.py',
         'scripts/build_neon_e4_gpu.py','scripts/run_neon_e4.py','src/geolifeclef/neon_assessment.py',
         'src/geolifeclef/external_reliability.py','src/geolifeclef/ecological_selection.py',
         'src/geolifeclef/reliability.py','src/geolifeclef/sage_inference.py','notebooks/neon_e4_inference.ipynb']


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n')


def freeze():
    inputs=['input_bundle.npz','label_contract.npz','source_calibration_roles.csv','neon_assessment_roles.csv','selected_plots.csv']
    identity={'sources':{n:sha(ROOT/n) for n in SOURCES},'inputs':{n:sha(DATA/n) for n in inputs},
              'external_inputs':{n:sha(ROOT/n) for n in ['data/neon_bridge/neon_to_sage_taxonomy.csv',
                  'data/neon_bridge/species_names.csv','data/neon_bridge/neon_2023_manifest.json',
                  'results/neon_bridge/artifact_manifest.json','data/external_replication_e1/sealed_eval_test_targets.h5']}}
    path=OUT/'protocol_freeze.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['identity']!=identity:raise ValueError('E4 frozen inputs/source changed')
        return old
    for name in SOURCES:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    lock={'locked_at_utc':now(),'identity':identity,'NEON_outcomes_accessed':False,'scope':'local prospective lock, one-model training-support stratum, not external registration'}
    save(path,lock);return lock


def verify_gpu():
    state=json.loads((OUT/'kaggle_state.json').read_text());base=ROOT/'artifacts/neon_e4_gpu'
    assert state['status']=='complete' and state['private'] and state['GPU_requested']
    assert state['notebook_sha256']==sha(ROOT/'notebooks/neon_e4_inference.ipynb')
    for file in state['output_files']:
        p=base/file['name'];assert p.stat().st_size==file['bytes'] and sha(p)==file['sha256']
    report=json.loads((base/'e4_gpu_report.json').read_text())
    assert report['input_bundle_sha256']==sha(DATA/'input_bundle.npz') and report['elapsed_seconds']<=1200
    assert not report['training_performed'] and not report['NEON_credentials_or_labels_included']
    p=np.load(base/'e4_probabilities.npz');reference=np.load(ROOT/'data/neon_bridge/cpu_reference.npz')
    checks={}
    for name in ['cpu','published']:
        error=np.abs(p['reference'].astype(float)-reference['aggregation_s0__'+name])
        checks[name]={'maximum_absolute_error':float(error.max()),'mean_absolute_error':float(error.mean())}
        assert error.max()<=1e-5 and error.mean()<=1e-6
    assert p['source'].shape==(2226,5771) and p['neon'].shape==(163,5771)
    save(OUT/'GPU_validation.json',{'status':'passed','reference_comparisons':checks,'GPU_report':report,'output_hashes_verified':True})
    return p


def fit(config):
    freeze()
    if (OUT/'policy_freeze.json').exists():raise ValueError('E4 policies already frozen; no refit')
    p=verify_gpu();source=pd.read_csv(DATA/'source_calibration_roles.csv');neon=pd.read_csv(DATA/'neon_assessment_roles.csv')
    label=np.load(DATA/'label_contract.npz');eligible=np.concatenate([label['source_eligible'],label['neon_eligible']])
    cal=np.flatnonzero(source.role.eq('probability_calibration'));risk=np.flatnonzero(source.role.eq('risk_calibration'))
    save(OUT/'source_target_access.json',{'at_utc':now(),'roles_requested':['probability_calibration','risk_calibration'],'E1_assessment_requested':False,'NEON_outcomes_accessed':False})
    with h5py.File(ROOT/'data/external_replication_e1/sealed_eval_test_targets.h5') as f:
        yc=guarded_target_rows(f['targets'],source,'probability_calibration','fit')
        yr=guarded_target_rows(f['targets'],source,'risk_calibration','fit')
    yc=np.where(label['source_eligible'][cal],yc,-1);yr=np.where(label['source_eligible'][risk],yr,-1)
    calibration=fit_calibration(p['source'][cal],yc,eligible[cal],config['probability_calibration_bounds'])
    probability=apply_calibration(np.concatenate([p['source'],p['neon']]),calibration,eligible)
    context=prediction_context(probability,eligible);count=decode_counts(context,config['decoder'])
    metadata=pd.concat([source[META_FEATURES[:-1]],neon[META_FEATURES[:-1]]],ignore_index=True)
    features=decision_features(context,count,metadata,label['rare'])
    outcome=decision_outcomes(yr,eligible[risk],context['rank'][risk],count[risk],label['rare'])
    quality=outcome.sample_F1.to_numpy();model=HistGradientBoostingRegressor(**config['risk_regressor'])
    model.fit(features.iloc[risk][RISK_FEATURES],quality)
    sdc=features.iloc[risk].SDC.to_numpy(float);variance=float(sdc.var())
    slope=max(float(np.mean((sdc-sdc.mean())*(quality-quality.mean()))/variance),1e-9) if variance>1e-15 else 1e-9
    intercept=float(quality.mean()-slope*sdc.mean());start=len(source);f=features.iloc[start:]
    arrays={'rank':context['rank'][start:].astype(np.int16),'count':count[start:],
            'HGB_quality':np.clip(model.predict(f[RISK_FEATURES]),0,1),'SDC':f.SDC.to_numpy(),
            'affine_SDC_quality':np.clip(slope*f.SDC.to_numpy()+intercept,0,1),
            'mean_quality':np.repeat(quality.mean(),len(neon)),'mean_confidence':f.mean_set_confidence.to_numpy(),
            'entropy':f.set_entropy.to_numpy(),'random':np.array([r1_hash('E4:'+x)/2**64 for x in neon.cell_id])}
    np.savez_compressed(DATA/'frozen_policy.npz',**arrays)
    f.to_csv(DATA/'frozen_neon_features.csv',index=False)
    parameters={'probability_calibration':calibration,'source_riskcal_mean_quality':float(quality.mean()),
                'positive_affine_SDC':{'slope':slope,'intercept':intercept},'HGB_config':config['risk_regressor']}
    save(OUT/'fit_parameters.json',parameters)
    lock={'at_utc':now(),'NEON_outcomes_accessed':False,'policy_sha256':sha(DATA/'frozen_policy.npz'),
          'features_sha256':sha(DATA/'frozen_neon_features.csv'),'fit_parameters_sha256':sha(OUT/'fit_parameters.json'),
          'protocol_freeze_sha256':sha(OUT/'protocol_freeze.json')}
    save(OUT/'policy_freeze.json',lock);print(json.dumps({'status':'policies_frozen_before_NEON_outcomes','parameters':parameters}))


def assess(config):
    freeze();lock=json.loads((OUT/'policy_freeze.json').read_text())
    assert lock['policy_sha256']==sha(DATA/'frozen_policy.npz') and lock['features_sha256']==sha(DATA/'frozen_neon_features.csv')
    assert lock['fit_parameters_sha256']==sha(OUT/'fit_parameters.json')
    if (OUT/'primary_result.json').exists():raise ValueError('E4 assessment already complete; do not overwrite')
    neon=pd.read_csv(DATA/'neon_assessment_roles.csv');plots=pd.read_csv(DATA/'selected_plots.csv')
    selected=dict(zip(plots.plot_id,plots.bout.astype(str)));cell_for_plot=dict(zip(plots.plot_id,plots.cell_id))
    cell_rows={key:i for i,key in enumerate(neon.cell_id)}
    retrieval=json.loads((ROOT/'results/neon_bridge/taxonomy_retrieval.json').read_text())
    taxa={r['taxonID']:r for item in retrieval['files'] for r in json.loads((ROOT/item['file']).read_text())['data']}
    mapped=pd.read_csv(ROOT/'data/neon_bridge/neon_to_sage_taxonomy.csv')
    mapping={r.neon_taxon_id:int(r.sage_index) for r in mapped[mapped.sage_index.notna()].itertuples()}
    names=pd.read_csv(ROOT/'data/neon_bridge/species_names.csv')['Species Name'].tolist()
    contract=np.load(DATA/'label_contract.npz');reducer=TaxonomicAssessment(contract['neon_eligible'],taxa,mapping,names)
    manifest=json.loads((ROOT/'data/neon_bridge/neon_2023_manifest.json').read_text())['data']
    files=[]
    for release in manifest['releases']:
        assert release['release']=='RELEASE-2026'
        for package in release['packages']:
            if package['siteCode'] not in set(neon.site):continue
            for file in package['files']:
                for table in ['div_1m2Data','div_10m2Data100m2Data']:
                    if '.'+table+'.' in file['name']:files.append({**file,'table':table})
    assert sum(f['size'] for f in files)<=80_000_000
    access_path=OUT/'assessment_access.json'
    if access_path.exists():
        assert json.loads(access_path.read_text())['policy_freeze_sha256']==sha(OUT/'policy_freeze.json')
    else:
        save(access_path,{'first_request_at_utc':now(),'policy_freeze_sha256':sha(OUT/'policy_freeze.json'),
                         'selected_cells':len(neon),'selected_plots':len(plots),'raw_files_retained':False,
                         'unselected_plot_or_bout_taxon_values_decoded':False})
    def retrieve(file):
        with urllib.request.urlopen(file['url'],timeout=40) as response:body=response.read(file['size']+1)
        if len(body)!=file['size'] or hashlib.md5(body).hexdigest()!=file['md5']:raise ValueError('NEON published size/MD5 mismatch')
        return file,body
    allowed=['plotID','boutNumber','taxonID','targetTaxaPresent','divDataType','identificationQualifier','dataQF']
    receipts=[];seen_plots=set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for file,body in pool.map(retrieve,files):
            for row in selected_csv_rows(body,selected,allowed):
                seen_plots.add(row['plotID']);reducer.add(cell_rows[cell_for_plot[row['plotID']]],row,file['table'])
            receipts.append({'name':file['name'],'bytes':len(body),'MD5_verified':file['md5']})
    if seen_plots!=set(selected):raise ValueError('Selected plot identities absent from outcome files')
    truth,known,fraction,counts=reducer.finish()
    np.savez_compressed(DATA/'assessment_labels.npz',truth=truth,known=known,known_fraction=fraction)
    policy=np.load(DATA/'frozen_policy.npz');scored=score_fixed_sets(truth,known,policy['rank'],policy['count'])
    valid=(fraction>=config['minimum_known_label_fraction'])&known.any(1)
    save(OUT/'target_mapping_audit.json',{'record_counts':counts,'input_cells':len(neon),'scorable_cells':int(valid.sum()),
        'scorable_sites':int(neon.loc[valid,'site'].nunique()),'minimum_known_fraction':float(fraction.min()),
        'median_known_fraction':float(np.median(fraction)),'MD5_receipts':receipts,
        'per_site_scorable_cells':neon.loc[valid].groupby('site').size().astype(int).to_dict()})
    rows=neon.copy();rows['known_fraction']=fraction;rows['scorable']=valid
    for key,value in scored.items():rows[key]=value
    for key in policy.files:
        if policy[key].ndim==1:rows['policy_'+key]=policy[key]
    admitted=bool(valid.sum()>=config['minimum_scored_cells'] and neon.loc[valid,'site'].nunique()>=config['minimum_scored_sites'])
    if not valid.any():
        save(OUT/'primary_result.json',{'status':'no_scorable_cells','scientific_admission':False,'joint_primary_success':False})
        rows.to_csv(OUT/'cell_results.csv',index=False);print('No scorable cells; no hypothesis confirmation.');return
    metadata=neon.loc[valid];groups=(metadata.site.astype(str)+'|'+metadata.area_group.astype(str)).to_numpy()
    selector=QuotaSelection(metadata.cell_id.to_numpy(),config['coverage'],groups)
    risks={'HGB':1-policy['HGB_quality'][valid],'SDC':1-policy['SDC'][valid],
           'mean_confidence':1-policy['mean_confidence'][valid],'entropy':policy['entropy'][valid],'random':policy['random'][valid]}
    chosen={name:selector.accept(risk) for name,risk in risks.items()}
    for name,mask in chosen.items():
        for indices,quota in selector.groups:assert mask[indices].sum()==quota
        rows['accept_'+name]=False;rows.loc[valid,'accept_'+name]=mask
    q=scored['F1'][valid];comparison=site_primary_comparison(q,policy['HGB_quality'][valid],float(policy['mean_quality'][0]),
        chosen['HGB'],chosen['SDC'],metadata.site,**{k:v for k,v in config['bootstrap'].items() if k!='cluster'})
    success=bool(admitted and comparison['MSE_gain_mean_minus_HGB']>=.001 and comparison['MSE_gain_interval'][0]>0
                 and comparison['HGB_minus_SDC_selective_F1']<=-.002 and comparison['F1_difference_interval'][1]<0)
    metrics={name:float(np.mean((policy[key][valid]-q)**2)) for name,key in [('training_mean','mean_quality'),('affine_SDC','affine_SDC_quality'),('HGB','HGB_quality'),('raw_SDC','SDC')]}
    selected_f1={name:float(q[mask].mean()) for name,mask in chosen.items()}
    rare=contract['rare'];rare_truth=(truth[:,rare]>0).sum(1)[valid]
    rare_retention={name:float(rare_truth[mask].sum()/rare_truth.sum()) if rare_truth.sum() else None for name,mask in chosen.items()}
    result={'status':'assessment_complete','scientific_admission':admitted,'joint_primary_success':success,
            'primary':comparison,'quality_MSE':metrics,'selective_F1':selected_f1,'rare_occurrence_retention_secondary':rare_retention,
            'full_coverage_F1':float(q.mean()),'full_coverage_F1_zero_empty':float(scored['F1_zero_empty'][valid].mean()),
            'input_cells':len(neon),'scored_cells':int(valid.sum()),'sites':int(metadata.site.nunique()),
            'scored_known_truth_occurrences':int((truth[valid]>0).sum()),'neural_training_fits':0,
            'scope':'one fixed pretrained MLP, training-support/input-availability stratum, partially observed modeled-native-set F1, site-conditioned inference; no unseen-training-location or new-algorithm claim'}
    rows.to_csv(OUT/'cell_results.csv',index=False);save(OUT/'primary_result.json',result);print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','fit','assess']);args=parser.parse_args()
    config=json.loads((ROOT/'configs/confirmation_e4.json').read_text())
    if args.phase=='freeze':print(json.dumps(freeze()))
    elif args.phase=='fit':fit(config)
    else:assess(config)


if __name__=='__main__':main()
