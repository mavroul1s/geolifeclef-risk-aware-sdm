"""Nested, buffered spatial ranking and source-exclusion development studies."""
from datetime import datetime, timezone
import hashlib
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

from run_external_e1 import save, sha
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.external_reliability import (META_FEATURES,RISK_FEATURES,apply_calibration,
    decision_features,decision_outcomes,decode_counts,fit_calibration,
    guarded_target_rows,joint_ecological_comparison,prediction_context,read_rows)
from geolifeclef.ranking_selection import (buffered_training,choose_penalty,fit_anchored,
    monotone_affine,spatial_splits)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/external_replication_e1'
DATA = ROOT/'data/ranking_study_e2'
OUT = ROOT/'results/ranking_study_e2'
MODELS = ['SDC','mean_confidence','HGB','affine_SDC','mse_residual','pairwise']


def fingerprint_ids(ids):
    return hashlib.sha256(np.sort(np.asarray(ids,'<i8')).tobytes()).hexdigest()


def lock(config):
    names = ['configs/ranking_study_e2.json','docs/ranking_study_e2_protocol.md',
        'scripts/study_ranking_e2.py','src/geolifeclef/ranking_selection.py',
        'src/geolifeclef/external_reliability.py','src/geolifeclef/ecological_selection.py']
    hashes = {n:sha(ROOT/n) for n in names}
    path = OUT/'development_lock.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['source_hashes'] != hashes:
            raise ValueError('E2 source changed after lock: document an amendment and preserve old outputs')
        return
    snap = OUT/'source_snapshot'
    for name in names:
        target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    save(path,{'locked_at_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':hashes,
        'E1_assessment_targets_read_in_E2':False,'evidence_scope':config['evidence_scope'],
        'E1_cohort_sha256':sha(SOURCE/'cohort_roles.csv'),
        'E1_input_hashes':{name:sha(SOURCE/name) for name in ['rare_species_mask.npy',
            'eligible_species.npy','sealed_eval_test_targets.h5']}})


def calibration_only(cohort):
    """Project before any target-row read; assessment is structurally excluded."""
    take = cohort.role.isin(['probability_calibration','risk_calibration'])
    result = cohort.loc[take].reset_index(drop=True)
    if result.empty or result.role.eq('assessment').any():
        raise ValueError('Calibration-only projection failed')
    return result,np.flatnonzero(take)


def load_inputs(config):
    whole = pd.read_csv(SOURCE/'cohort_roles.csv',dtype={'area_bin':str})
    meta,indices = calibration_only(whole)
    cache_identity=DATA/'prediction_cache_identity.json'
    identity={'cell_ids_sha256':fingerprint_ids(meta.cell_id),'rows':len(meta),
        'cohort_sha256':sha(SOURCE/'cohort_roles.csv')}
    if cache_identity.exists() and json.loads(cache_identity.read_text())!=identity:
        raise ValueError('Cached predictions belong to a different cohort')
    save(cache_identity,identity)
    metadata = pd.read_csv(ROOT/'data/independent_data_audit/sage_eval_plot_metadata.csv',
        usecols=['cell_row','GIVD_ID']).drop_duplicates()
    sources = metadata.groupby('cell_row').GIVD_ID.agg(lambda x:'|'.join(sorted(x.dropna().unique())))
    meta['source'] = sources.reindex(meta.cell_id).to_numpy()
    eligible = np.load(SOURCE/'eligible_species.npy',allow_pickle=False)[indices]
    rare = np.load(SOURCE/'rare_species_mask.npy',allow_pickle=False)
    cal = np.flatnonzero(meta.role.eq('probability_calibration'))
    risk = np.flatnonzero(meta.role.eq('risk_calibration'))
    targets = np.zeros(eligible.shape,np.int8)
    with h5py.File(SOURCE/'sealed_eval_test_targets.h5') as f:
        for role,ix in [('probability_calibration',cal),('risk_calibration',risk)]:
            targets[ix] = guarded_target_rows(f['targets'],meta,role,'fit')
    if not np.array_equal(targets>=0,eligible):
        raise ValueError('Calibration-only native mask alignment failed')
    raw = {}
    for model in config['models']:
        path=DATA/f'{model}_calibration.npy'
        if path.exists():
            p=np.load(path,allow_pickle=False,mmap_mode='r')
            if p.shape != eligible.shape:raise ValueError('Invalid cached prediction rows')
        else:
            with h5py.File(SOURCE/model/'test_predictions.h5') as f:
                p=read_rows(f['probabilities'],meta.prediction_row.to_numpy())
            np.save(path,p,allow_pickle=False)
        raw[model]=p
    return meta,eligible,rare,targets,raw,cal,risk


def get_features(config,meta,eligible,rare,targets,raw,cal,risk):
    old_config=json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    riskmeta=meta.iloc[risk].reset_index(drop=True).copy()
    edges=np.unique(np.quantile(meta.iloc[cal].log_area,[.2,.4,.6,.8]))
    riskmeta['area_bin']=np.searchsorted(edges,riskmeta.log_area,side='right').astype(str)
    tree=BallTree(np.deg2rad(meta.iloc[cal][['lat','lon']]),metric='haversine')
    distance=tree.query(np.deg2rad(riskmeta[['lat','lon']]))[0][:,0]*6371.0088
    riskmeta['log_calibration_distance']=np.log1p(distance)
    result={};audit={}
    for model in config['models']:
        calibration=fit_calibration(raw[model][cal],targets[cal],eligible[cal],old_config['calibration_bounds'])
        p=apply_calibration(raw[model][risk],calibration,eligible[risk])
        ctx=prediction_context(p,eligible[risk]);counts=decode_counts(ctx,config['decoder'])
        features=decision_features(ctx,counts,riskmeta[META_FEATURES[:-1]],rare)
        outcome=decision_outcomes(targets[risk],eligible[risk],ctx['rank'],counts,rare)
        result[model]={'X':features[config['features']].to_numpy(),
            'base':features.SDC.to_numpy(),'mean_confidence':features.mean_set_confidence.to_numpy(),
            'quality':outcome.sample_F1.to_numpy(),'rare':outcome.rare_truth.to_numpy()}
        audit[model]={'calibration':calibration,'decoder':config['decoder']}
    return riskmeta,result,audit


def fit_one(config,data,meta,train,penalty,objective):
    group=meta.iloc[train].country.astype(str)+'|'+meta.iloc[train].area_bin.astype(str)
    return fit_anchored(data['X'][train],data['base'][train],data['quality'][train],
        meta.iloc[train].cell_id.to_numpy(),group.to_numpy(),penalty,objective,
        config['maximum_pairs_per_group'],config['pair_seed'],config['pair_temperature'])


def select(config,features,meta,train):
    local=meta.iloc[train].reset_index(drop=True)
    splits=spatial_splits(local,config['inner_spatial_folds'],config['buffer_km'])
    totals={name:[] for name in ['baseline']+[f'{o}:{p}' for o in ['pairwise','mse'] for p in config['pairwise_penalties']]}
    audits=[]
    for inner,(t,v) in enumerate(splits):
        tr,va=train[t],train[v]
        if len(tr)<config['minimum_training_cells']:
            return {'pairwise':None,'mse':None},{'fallback':'inner buffered sample below admission',
                'minimum_inner_training_cells':len(tr),'audits':audits}
        vm=meta.iloc[va]
        groups=vm.country.astype(str)+'|'+vm.area_bin.astype(str)
        selector=QuotaSelection(vm.cell_id,config['coverage'],groups)
        for model,data in features.items():
            base=selector.accept(-data['base'][va])
            totals['baseline'].append((float(data['quality'][va][base].sum()),int(base.sum())))
            for objective in ['pairwise','mse']:
                for penalty in config['pairwise_penalties']:
                    fitted=fit_one(config,data,meta,tr,penalty,objective)
                    score=fitted.predict(data['X'][va],data['base'][va])
                    accepted=selector.accept(-score)
                    totals[f'{objective}:{penalty}'].append((float(data['quality'][va][accepted].sum()),int(accepted.sum())))
        audits.append({'inner':inner,'training':len(tr),'validation':len(va),
            'train_ids_sha256':fingerprint_ids(meta.iloc[tr].cell_id),
            'validation_ids_sha256':fingerprint_ids(meta.iloc[va].cell_id)})
    values={name:sum(x for x,n in rows)/sum(n for x,n in rows) for name,rows in totals.items()}
    chosen={objective:choose_penalty(values['baseline'],
        {p:values[f'{objective}:{p}'] for p in config['pairwise_penalties']},.001) for objective in ['pairwise','mse']}
    return chosen,{'inner_values':values,'chosen':chosen,'audits':audits}


def evaluate(config,features,meta,train,val,choice,study,fold):
    old_config=json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    evaluation=meta.iloc[val]
    selector=QuotaSelection(evaluation.cell_id,config['coverage'],
        evaluation.country.astype(str)+'|'+evaluation.area_bin.astype(str))
    records=[];fits=[]
    for model,data in features.items():
        scores={'SDC':data['base'][val],'mean_confidence':data['mean_confidence'][val]}
        scores['affine_SDC'],affine=monotone_affine(data['base'][train],data['quality'][train],data['base'][val])
        reg=HistGradientBoostingRegressor(**old_config['risk_regressor']).fit(data['X'][train],1-data['quality'][train])
        scores['HGB']=1-reg.predict(data['X'][val])
        for objective,name in [('pairwise','pairwise'),('mse','mse_residual')]:
            penalty=choice[objective]
            if penalty is None:
                scores[name]=data['base'][val].copy();audit={'fallback':'unchanged SDC selected by inner CV'}
            else:
                fitted=fit_one(config,data,meta,train,penalty,objective)
                scores[name]=fitted.predict(data['X'][val],data['base'][val]);audit=fitted.audit
            fits.append({'model':model,'study':study,'fold':fold,'objective':objective,**audit})
        baseline=selector.accept(-scores['SDC'])
        for method,score in scores.items():
            accepted=selector.accept(-score)
            for group,quota in selector.groups:
                assert accepted[group].sum()==quota
            if method=='affine_SDC' and not np.array_equal(accepted,baseline):
                raise ValueError('Monotone calibration changed the acceptance policy')
            for local,ix in enumerate(val):
                records.append({'study':study,'fold':str(fold),'model':model,'method':method,
                    'cell_id':int(meta.iloc[ix].cell_id),'spatial_block':meta.iloc[ix].spatial_block,
                    'source':meta.iloc[ix].source,'quality':float(data['quality'][ix]),
                    'rare_truth':int(data['rare'][ix]),'accepted':bool(accepted[local]),
                    'score':float(score[local])})
    return records,fits


def comparison(config,frame,method):
    subset=frame[frame.method.isin([method,'SDC'])]
    qualities=[];rares=[];candidate=[];baseline=[]
    identity=None;blocks=None
    for model in config['models']:
        a=subset[(subset.model==model)&(subset.method==method)].sort_values('cell_id')
        b=subset[(subset.model==model)&(subset.method=='SDC')].sort_values('cell_id')
        if not np.array_equal(a.cell_id,b.cell_id) or a.cell_id.duplicated().any():
            raise ValueError('Misaligned held-out contrasts')
        if identity is None:identity=a.cell_id.to_numpy();blocks=a.spatial_block.to_numpy()
        if not np.array_equal(a.cell_id,identity):raise ValueError('Models do not share held-out cells')
        qualities.append(a.quality.to_numpy());rares.append(a.rare_truth.to_numpy())
        candidate.append(a.accepted.to_numpy());baseline.append(b.accepted.to_numpy())
    return joint_ecological_comparison(qualities,rares,candidate,baseline,blocks,confidence=.95)


def summarize(config,records,split_audits):
    frame=pd.DataFrame(records);frame.to_csv(OUT/'heldout_development_rows.csv',index=False)
    comparisons={};summary=[]
    for study,subset in frame.groupby('study'):
        comparisons[study]={name:comparison(config,subset,name) for name in MODELS if name!='SDC'}
        for (model,method),g in subset.groupby(['model','method']):
            retained=g[g.accepted]
            summary.append({'study':study,'model':model,'method':method,'cells':len(g),
                'retained':len(retained),'F1':float(retained.quality.mean()),
                'rare_occurrence_coverage':float(retained.rare_truth.sum()/g.rare_truth.sum()),
                'raw_quality_score_MSE':float(np.mean((g.score-g.quality)**2)) if method!='pairwise' else None})
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    gate=config['gpu_gate'];checks={}
    for study in ['spatial','source']:
        result=comparisons[study]['pairwise']
        checks[study+'_practical_F1_gain']=result['F1_gain']>=gate['minimum_mean_F1_gain_each_study']
        checks[study+'_F1_interval_positive']=result['F1_interval'][0]>gate['minimum_F1_interval_lower_each_study']
        checks[study+'_all_model_F1_guard']=min(result['model_F1_gains'])>=gate['minimum_model_F1_gain_each_study']
        checks[study+'_rare_interval_guard']=result['rare_coverage_interval'][0]>=gate['minimum_rare_coverage_interval_lower_each_study']
        checks[study+'_all_model_rare_guard']=min(result['model_rare_coverage_gains'])>=gate['minimum_model_rare_coverage_gain_each_study']
    by_source={}
    for source,subset in frame[frame.study.eq('source')].groupby('fold'):
        by_source[source]=comparison(config,subset,'pairwise')
    checks['supported_source_guard']=all(r['F1_gain']>=gate['minimum_supported_source_F1_gain'] for r in by_source.values())
    tests=ET.parse(OUT/'focused_tests.xml').getroot().find('testsuite').attrib
    checks['correctness_tests']=int(tests['failures'])==0 and int(tests['errors'])==0
    checks['all_required_folds_admitted']=len(split_audits)==8 and all(
        'fallback' not in audit['inner'] for audit in split_audits)
    checks={name:bool(value) for name,value in checks.items()}
    report={'status':'completed_development_studies','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'evidence_scope':config['evidence_scope'],'E1_assessment_targets_read_in_E2':False,
        'gpu_scientifically_admitted':all(checks.values()),'gpu_checks':checks,
        'user_GPU_authorization_recorded':True,'primary_comparisons':{s:r['pairwise'] for s,r in comparisons.items()},
        'all_comparisons':comparisons,'source_primary_comparisons':by_source,
        'splits':split_audits,'new_neural_backbone_fits':0,
        'multiple_model_settings_are_independent_datasets':False}
    save(OUT/'report.json',report)
    print(json.dumps({'status':report['status'],'GPU_admitted':report['gpu_scientifically_admitted'],
        'checks':checks,'primary_comparisons':report['primary_comparisons']}),flush=True)


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/ranking_study_e2.json').read_text());lock(config)
    if (OUT/'report.json').exists():raise ValueError('E2 results already exist; preserve completed study')
    meta,eligible,rare,targets,raw,cal,risk=load_inputs(config)
    rm,features,cal_audit=get_features(config,meta,eligible,rare,targets,raw,cal,risk)
    splits=spatial_splits(rm,config['outer_spatial_folds'],config['buffer_km'])
    source_counts=rm[~rm.source.str.contains('|',regex=False)].groupby('source').agg(
        cells=('cell_id','size'),blocks=('spatial_block','nunique')).sort_values(['cells'],ascending=False)
    criteria=config['source_admission']
    selected=source_counts[(source_counts.cells>=criteria['minimum_riskcal_cells'])&
        (source_counts.blocks>=criteria['minimum_riskcal_blocks'])].head(criteria['maximum_sources'])
    save(OUT/'metadata_split_design.json',{'selected_sources':selected.reset_index().to_dict('records'),
        'spatial_folds':[{'fold':i,'training':len(t),'validation':len(v),
            'training_ids_sha256':fingerprint_ids(rm.iloc[t].cell_id),
            'validation_ids_sha256':fingerprint_ids(rm.iloc[v].cell_id)} for i,(t,v) in enumerate(splits)],
        'calibration_cells':len(cal),'riskcal_cells':len(risk),'assessment_target_rows_requested':0})
    save(OUT/'spatial_probability_calibration.json',cal_audit)
    records=[];audits=[];fit_audits=[]
    for fold,(train,val) in enumerate(splits):
        if len(train)<config['minimum_training_cells']:raise ValueError('Outer spatial admission failed')
        choice,inner=select(config,features,rm,train)
        rows,fits=evaluate(config,features,rm,train,val,choice,'spatial',fold)
        records.extend(rows);fit_audits.extend(fits)
        audits.append({'study':'spatial','fold':fold,'train_cells':len(train),'validation_cells':len(val),
            'training_ids_sha256':fingerprint_ids(rm.iloc[train].cell_id),
            'validation_ids_sha256':fingerprint_ids(rm.iloc[val].cell_id),'inner':inner})
        print(json.dumps({'study':'spatial','fold':fold,'selected':choice,'seconds':time.monotonic()-started}),flush=True)
    for source in selected.index:
        val=np.flatnonzero(rm.source.eq(source));remaining=np.flatnonzero(~rm.source.eq(source))
        train=buffered_training(rm,remaining,val,config['buffer_km'])
        allowed_cal=cal[~meta.iloc[cal].source.eq(source).to_numpy()]
        allowed_cal=buffered_training(meta,allowed_cal,risk[val],config['buffer_km'])
        if len(allowed_cal)<config['minimum_probability_calibration_cells'] or len(train)<config['minimum_training_cells']:
            raise ValueError('Source admission failed: '+source)
        sm,sfeatures,audit=get_features(config,meta,eligible,rare,targets,raw,allowed_cal,risk)
        choice,inner=select(config,sfeatures,sm,train)
        rows,fits=evaluate(config,sfeatures,sm,train,val,choice,'source',source)
        records.extend(rows);fit_audits.extend(fits)
        audits.append({'study':'source','fold':source,'train_cells':len(train),'validation_cells':len(val),
            'probability_calibration_cells':len(allowed_cal),'excluded_source_from_both_calibration_roles':True,
            'training_ids_sha256':fingerprint_ids(sm.iloc[train].cell_id),
            'validation_ids_sha256':fingerprint_ids(sm.iloc[val].cell_id),
            'probability_calibration_ids_sha256':fingerprint_ids(meta.iloc[allowed_cal].cell_id),
            'probability_calibration':audit,'inner':inner})
        print(json.dumps({'study':'source','held_out':source,'selected':choice,'seconds':time.monotonic()-started}),flush=True)
    save(OUT/'fit_audits.json',fit_audits);summarize(config,records,audits)


if __name__=='__main__':
    main()
