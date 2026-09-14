"""One compute-bounded v22: two outer folds, retained PO arms, frozen v21 deployment."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from scripts.prepare_environmental_challenger import prepare, spatial_partitions
from scripts.prepare_po_expert import prepare_po, coordinate_features, expand_features
from scripts.run_ood_po_expert import normalized_environment, matched_control, save_json, validate_submission
from scripts.train_po_expert import fit_expert, expert_predict
from scripts.train_retained_po import pretrain, fit_arm, predict
from scripts.stage_frozen_v20 import verify_bundle, sha256_file, submission_bytes, verify_original_ties
from scripts.stage_frozen_v21 import verify_v21, CSV_HASH
from scripts.ood_po_protocol import (nearest_training_support, mix_probabilities,
    assessment_metrics, paired_block_bootstrap, spatial_block_ids)
from scripts.v22_protocol import crossfit_partitions, hashed_blocks, SEED, mix, select_policy

ARMS=('retained_po','single_head','zero_po')
V21_POLICY={'alpha':.025,'gate':'pa_distance','k':20}


def prepare_labels(raw,rows,species,path):
    matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(len(rows),len(species)))
    matrix[:]=0
    pairs=raw[['surveyId','speciesId']].dropna().drop_duplicates()
    ri=pd.Index(rows.surveyId).get_indexer(pairs.surveyId)
    ci=pd.Index(species).get_indexer(pairs.speciesId)
    if (ri<0).any() or (ci<0).any(): raise ValueError('PA label join failed')
    matrix[ri,ci]=1; matrix.flush()
    return matrix


def features(pa,test,po,fit_ix,rows,test_rows,output,name):
    a,b,c,normalization=normalized_environment(pa,test,po,fit_ix)
    save_json(output/f'{name}_normalization.json',normalization)
    return (expand_features(coordinate_features(rows[['lat','lon']].to_numpy()),a),
            expand_features(coordinate_features(test_rows[['lat','lon']].to_numpy()),b),c)


def load_po(directory):
    return {'pa':np.load(directory/'pa_environment_raw.npy',mmap_mode='r'),
            'test':np.load(directory/'test_environment_raw.npy',mmap_mode='r'),
            'po':np.load(directory/'po_environment_raw.npy',mmap_mode='r'),
            'geo':np.load(directory/'po_features.npy',mmap_mode='r'),
            'labels':sparse.load_npz(directory/'po_labels.npz'),
            'weights':np.load(directory/'po_weights.npy')}


def expert_features(pool,fit_ix,rows,test_rows,output,name):
    pa,test,po=features(pool['pa'],pool['test'],pool['po'],fit_ix,rows,test_rows,output,name)
    return pa,test,expand_features(pool['geo'],po)


def fit_v21_control(rows,test_rows,labels,split,distance,pool,data_dir,output,device,deadline):
    control=matched_control(data_dir,output,split,labels,device,deadline,2,24,include_selection=True)
    ix=[np.flatnonzero(split==i) for i in range(4)]
    pa,_,po=expert_features(pool,ix[0],rows,test_rows,output,'v21_control')
    model,history=fit_expert(pa,labels,ix[0],ix[1],po,pool['labels'],pool['weights'],device,
        output,'legacy_po',min(deadline-3600,time.monotonic()+3600),True)
    for index,name in ((1,'selection'),(2,'calibration'),(3,'assessment')):
        expert=expert_predict(model,pa[ix[index]],device,deadline)
        base=np.load(output/f'frozen_control_{name}.npy',mmap_mode='r')
        value=mix_probabilities(base,expert,distance[ix[index]],V21_POLICY)
        np.save(output/f'frozen_v21_{name}.npy',value,allow_pickle=False)
    del model,pa,po; gc.collect(); torch.cuda.empty_cache()
    return {'multimodal':control,'legacy_po':history,'fixed_v21_policy':V21_POLICY}


def fit_new_arms(rows,test_rows,labels,split,distance,pool,base_selection,base_calibration,
                 output,device,deadline,deployment=False):
    ix=[np.flatnonzero(split==i) for i in range(4)]
    pa,test,po=expert_features(pool,ix[0],rows,test_rows,output,'v22_expert')
    pretrained,pretraining=pretrain(po,pool['labels'],pool['weights'],device,output,min(deadline-2400,time.monotonic()+3600))
    histories,policies={},{}
    for arm in ARMS:
        model,history=fit_arm(pretrained,arm,pa,labels,ix[0],ix[1],base_selection,distance,
            po,pool['labels'],pool['weights'],device,output,min(deadline-1200,time.monotonic()+3600))
        histories[arm]=history
        calibration=predict(model,pa[ix[2]],device,deadline)
        np.save(output/f'{arm}_calibration.npy',calibration,allow_pickle=False)
        selected,trials=select_policy(np.array(labels[ix[2]]),base_calibration,calibration,distance[ix[2]])
        policies[arm]={'selected':selected,'trials':trials,'checkpoint_sha256':sha256_file(output/f'{arm}_best.pt')}
        if deployment and arm=='retained_po':
            np.save(output/'retained_po_test.npy',predict(model,test,device,deadline),allow_pickle=False)
        elif not deployment:
            np.save(output/f'{arm}_assessment.npy',predict(model,pa[ix[3]],device,deadline),allow_pickle=False)
        del model,calibration; gc.collect(); torch.cuda.empty_cache()
    del pretrained,pa,test,po; gc.collect(); torch.cuda.empty_cache()
    return policies,{'pretraining':pretraining,'arms':histories}


def deployment_partitions(rows):
    original=spatial_partitions(rows)
    split=np.full(len(rows),-1,dtype=np.int8); split[original==0]=0
    bucket=hashed_blocks(rows,SEED+200)
    split[(original==2)&(bucket<50)]=1
    split[(original==2)&(bucket>=50)]=2
    if min(int((split==i).sum()) for i in range(3))<100: raise ValueError('Insufficient production split')
    return split


def gate_from_assessment(assessment,integrity,policies):
    comparisons=assessment['comparisons']
    checks={'positive_spatial_ci_over_v21':comparisons['frozen_v21']['ci95'][0]>0,
        'positive_gain_over_v20':comparisons['frozen_v20']['mean_difference']>0,
        'positive_gain_over_zero_po':comparisons['zero_po']['mean_difference']>0,
        'positive_v21_gain_each_fold':all(f['retained_po']>f['frozen_v21'] for f in assessment['fold_sample_f1']),
        'nonzero_po_each_fold':all(policies[f'fold_{i}']['retained_po']['selected']['alpha']>0 for i in (0,1)),
        'nonzero_production_po':policies['deployment']['retained_po']['selected']['alpha']>0,
        'all_integrity_checks_pass':bool(integrity) and all(v is True for v in integrity.values())}
    checks['eligible_for_official_submission']=all(checks.values())
    return checks


def run(args):
    started=float(os.environ.get('GLC_PIPELINE_STARTED_AT',time.time()))
    deadline=time.monotonic()+10.5*3600-(time.time()-started)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0): raise RuntimeError('Registered T4 required')
    device=torch.device('cuda'); torch.set_num_threads(min(os.cpu_count() or 2,4))
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    cache=Path('data/processed/retained_po_v22'); cache.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.data_root/'GLC25_PA_metadata_train.csv')
    rows=raw.drop_duplicates('surveyId').reset_index(drop=True)
    test_rows=pd.read_csv(args.data_root/'GLC25_PA_metadata_test.csv').drop_duplicates('surveyId').reset_index(drop=True)
    species=np.sort(raw.speciesId.dropna().unique().astype(np.int64))
    if len(species)!=5016 or len(test_rows)!=14784: raise ValueError('Competition dimensions changed')
    labels=prepare_labels(raw,rows,species,cache/'pa_labels.npy'); del raw
    folds=crossfit_partitions(rows)
    save_json(out/'split_manifest.json',{'folds':[m for _,_,m in folds],'assessment_used_for_selection':False})
    pd.DataFrame({'surveyId':rows.surveyId,'fold_0':folds[0][0],'fold_1':folds[1][0]}).to_csv(out/'partition_ids.csv',index=False)
    print(json.dumps({'stage':'v22_geography','folds':[m for _,_,m in folds]}),flush=True)
    original=spatial_partitions(rows)
    v20=verify_bundle(args.frozen_v20_dir,calibration_ids=rows.surveyId.to_numpy()[original==2],test_ids=test_rows.surveyId.to_numpy())
    v21=verify_v21(args.frozen_v21_dir)
    if not np.array_equal(species,np.load(args.frozen_v20_dir/'species_ids.npy')): raise ValueError('Species order mismatch')
    save_json(out/'frozen_v21_provenance.json',v21)
    manifests={}
    for mode in ('v21','v22'):
        manifests[mode]=prepare_po(args.data_root,cache/mode,rows,test_rows,species,deadline-7*3600,mode=mode)
    save_json(out/'po_manifests.json',manifests)
    old_pool,new_pool=load_po(cache/'v21'),load_po(cache/'v22')
    policies,training={},{}
    for fold,(split,distance,manifest) in enumerate(folds):
        name=f'fold_{fold}'; directory=out/name; directory.mkdir()
        data_dir=cache/f'{name}_modalities'
        prepare(args.data_root,data_dir,workers=6,deadline=deadline-4*3600,partition_override=split)
        training[name]={'control':fit_v21_control(rows,test_rows,labels,split,distance,old_pool,data_dir,directory,device,deadline)}
        base_selection=np.load(directory/'frozen_v21_selection.npy',mmap_mode='r')
        base_calibration=np.load(directory/'frozen_v21_calibration.npy',mmap_mode='r')
        policies[name],training[name]['new_experts']=fit_new_arms(rows,test_rows,labels,split,distance,new_pool,
            base_selection,base_calibration,directory,device,deadline)
        save_json(out/'frozen_policies.json',policies)
        del base_selection,base_calibration; gc.collect()
        if not data_dir.resolve().is_relative_to(cache.resolve()): raise ValueError('Unsafe cache cleanup')
        shutil.rmtree(data_dir)
    # Separate deployment fitting uses no outer score and cannot influence any
    # outer predictor, checkpoint or policy. Its training may include outer rows.
    production=deployment_partitions(rows)
    distance=nearest_training_support(rows.loc[original==0],rows)['distance_km']
    test_distance=nearest_training_support(rows.loc[original==0],test_rows)['distance_km']
    base_cal=mix_probabilities(np.load(args.frozen_v20_dir/'v20_calibration_probabilities.npy',mmap_mode='r'),
        np.load(args.frozen_v21_dir/'deployment_po_calibration.npy',mmap_mode='r'),distance[original==2],V21_POLICY)
    calibration_ids=np.flatnonzero(original==2)
    positions={i:j for j,i in enumerate(calibration_ids)}
    selection_positions=[positions[i] for i in np.flatnonzero(production==1)]
    calibration_positions=[positions[i] for i in np.flatnonzero(production==2)]
    directory=out/'deployment'; directory.mkdir()
    policies['deployment'],training['deployment']=fit_new_arms(rows,test_rows,labels,production,distance,new_pool,
        base_cal[selection_positions],base_cal[calibration_positions],directory,device,deadline,True)
    save_json(out/'frozen_policies.json',policies)
    base_test=mix_probabilities(np.load(args.frozen_v20_dir/'v20_test_probabilities.npy',mmap_mode='r'),
        np.load(args.frozen_v21_dir/'deployment_po_test.npy',mmap_mode='r'),test_distance,V21_POLICY)
    template=pd.read_csv(args.data_root/'GLC25_SAMPLE_SUBMISSION.csv').surveyId.to_numpy()
    verify_original_ties(base_test,species,test_rows.surveyId.to_numpy(),template,(args.frozen_v21_dir/'GLC25_PA_submission.csv').read_bytes())
    shutil.copyfile(args.frozen_v21_dir/'GLC25_PA_submission.csv',out/'unchanged_v21_submission.csv')
    policy=policies['deployment']['retained_po']['selected']
    destination=out/'GLC25_PA_submission.csv'
    if policy['alpha']==0: shutil.copyfile(out/'unchanged_v21_submission.csv',destination)
    else:
        value=mix(base_test,np.load(directory/'retained_po_test.npy',mmap_mode='r'),test_distance,policy)
        destination.write_bytes(submission_bytes(value,species,test_rows.surveyId.to_numpy(),template)); del value
    submission=validate_submission(destination,template,species)
    freeze={'policies_sha256':sha256_file(out/'frozen_policies.json'),
        'submission_sha256':sha256_file(destination),'assessment_reporting_started':False,
        'production_rows':{str(i):int((production==i).sum()) for i in range(3)}}
    save_json(out/'pre_assessment_freeze.json',freeze)
    del base_cal,base_test; gc.collect()
    # First and only outer assessment: both folds, all arms and production frozen.
    frames=[]; metrics={}; fold_scores=[]
    for fold,(split,distance,manifest) in enumerate(folds):
        directory=out/f'fold_{fold}'; ix=np.flatnonzero(split==3)
        targets=np.array(labels[ix]); countries=rows.country.fillna('unknown').to_numpy(dtype=str)[ix]
        counts=np.asarray(labels[np.flatnonzero(split==0)]).sum(axis=0)
        base=np.load(directory/'frozen_v21_assessment.npy',mmap_mode='r')
        predictions={'frozen_v20':np.load(directory/'frozen_control_assessment.npy',mmap_mode='r'),'frozen_v21':base}
        for arm in ARMS:
            predictions[arm]=mix(base,np.load(directory/f'{arm}_assessment.npy',mmap_mode='r'),distance[ix],policies[f'fold_{fold}'][arm]['selected'])
        scores={}; fold_metrics={}
        for name,value in predictions.items():
            fold_metrics[name],scores[name]=assessment_metrics(targets,value,countries,distance[ix],counts)
        metrics[f'fold_{fold}']=fold_metrics
        fold_scores.append({name:float(score.mean()) for name,score in scores.items()})
        frames.append(pd.DataFrame({'surveyId':rows.surveyId.to_numpy()[ix],'fold':fold,'country':countries,
            'block':spatial_block_ids(rows.iloc[ix]),'pa_distance_km':distance[ix],**scores}))
        del predictions,base,targets; gc.collect()
    frame=pd.concat(frames,ignore_index=True)
    if frame.surveyId.duplicated().any(): raise ValueError('Outer folds overlap')
    frame.to_csv(out/'assessment_per_survey.csv',index=False)
    comparisons={name:paired_block_bootstrap(frame.retained_po.to_numpy(),frame[name].to_numpy(),frame.block.to_numpy())
                 for name in ('frozen_v21','frozen_v20','zero_po','single_head')}
    assessment={'sample_f1':{name:float(frame[name].mean()) for name in ('frozen_v21','frozen_v20',*ARMS)},
        'fold_sample_f1':fold_scores,'metrics':metrics,'comparisons':comparisons,'used_for_selection':False,
        'surveys':len(frame),'blocks':int(frame.block.nunique()),
        'warning':'Recipe transfer only; shared cross-fit training induces dependence not fully captured by block bootstrap. Old audits are consumed.'}
    integrity={'all_5016_species':len(species)==5016,'both_folds_fresh':all(m['assessment_v21_training_only'] for _,_,m in folds),
        'twenty_km_buffer':all(m['minimum_evaluation_distance_km']>=20-1e-7 for _,_,m in folds),
        'unchanged_v21_csv_exact':sha256_file(out/'unchanged_v21_submission.csv')==CSV_HASH,
        'policies_unchanged':sha256_file(out/'frozen_policies.json')==freeze['policies_sha256'],
        'submission_unchanged':sha256_file(destination)==freeze['submission_sha256'],
        'competition_only':True,'test_labels_unused':True,'no_post_assessment_training':True,
        'registered_runtime':time.time()-started<10.5*3600}
    report={'experiment':'v22_retained_po_crossfit','status':'complete','source_commit':os.environ.get('GLC_SOURCE_COMMIT','local'),
        'split':[m for _,_,m in folds],'po':manifests,'training':training,'selected_policies':policies,
        'assessment':assessment,'integrity':integrity,'submission_gate':gate_from_assessment(assessment,integrity,policies),
        'submission':submission,'frozen_v21_source_version':21,'original_v21_retrained_to_recover_outputs':False,
        'external_data_or_weights':False,'test_labels_used':False,'registered_max_total_hours':10.5,
        'total_pipeline_hours':(time.time()-started)/3600,'official_public_score':None,'official_private_score':None}
    save_json(out/'v22_report.json',report)
    print(json.dumps({'stage':'v22_complete','scores':assessment['sample_f1'],'gate':report['submission_gate'],
                      'hours':report['total_pipeline_hours']}),flush=True)
    del labels,old_pool,new_pool; gc.collect()
    if cache.resolve().is_relative_to(Path.cwd().resolve()/'data'/'processed'): shutil.rmtree(cache)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--frozen-v20-dir',type=Path,required=True)
    parser.add_argument('--frozen-v21-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/retained_po_v22'))
    args=parser.parse_args()
    try: run(args)
    except Exception as error:
        save_json(args.output_dir/'failure.json',{'status':'failed','error_type':type(error).__name__,'message':str(error)})
        raise
