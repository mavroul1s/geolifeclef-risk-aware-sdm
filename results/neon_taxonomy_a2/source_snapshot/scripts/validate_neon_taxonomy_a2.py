"""Independent fixed-matrix scores, selector budgets and positive provenance."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_neon_taxonomy_a2 import DATA,OUT,freeze,metadata,cached,sha,save,now
from geolifeclef.neon_record_contract import (NAMED_RANKS,public_rank,current_resolution,combine_current_record,rank_matches)
from run_taxonomic_resolution_t1 import verify_history


def number(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:15],16)


def main():
    freeze();lock=json.loads((OUT/'labels_freeze.json').read_text())
    for name,digest in lock['files'].items():assert sha(ROOT/name)==digest
    times=[json.loads((OUT/name).read_text())[field] for name,field in [
        ('protocol_freeze.json','at_utc'),('record_access.json','at_utc'),('labels_freeze.json','at_utc'),('score_access.json','at_utc')]]
    assert times==sorted(times) and len(set(times))==4
    cells=pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv')
    plots=pd.read_csv(ROOT/'data/confirmation_e4/selected_plots.csv')
    selected=dict(zip(plots.plot_id,plots.bout.astype(str)));cell_for_plot=dict(zip(plots.plot_id,plots.cell_id))
    row_for_cell={v:i for i,v in enumerate(cells.cell_id)}
    observations=cached('selected_observations.json.gz')
    assert all(r['plotID'] in selected and r['boutNumber']==selected[r['plotID']] for _,r in observations)
    morpho={tuple(key):rows for key,rows in cached('selected_morphospecies.json.gz')}
    taxa,old,names,decisions,mapping=metadata()
    resolutions={key:current_resolution(rows) for key,rows in morpho.items()}
    linked={}
    for table,r in observations:
        if table=='div_1m2Data' and r.get('divDataType','').casefold()!='plantspecies':continue
        if r.get('identificationHistoryID','').strip():
            linked.setdefault((r['siteID'],r['identificationHistoryID']),[]).append(r)
    safe={}
    for key,rows in linked.items():
        evidence=list(rows)
        for row in rows:
            resolution,status,conflicts=resolutions.get((row['siteID'],row.get('morphospeciesID','')),(None,'no_dated_resolution',[]))
            evidence.extend([resolution] if resolution is not None else conflicts)
        safe[key]=all(public_rank(row,taxa) in NAMED_RANKS for row in evidence)
    histories=cached('selected_history.json.gz')
    for r in histories:
        key=(r['siteID'],r['identificationHistoryID']);assert key in linked
        assert not {'identifiedBy','measuredBy','recordedBy','remarks','identificationRemarks'}&set(r)
        if not safe[key]:assert 'taxonID' not in r and 'acceptedTaxonID' not in r
    label=np.load(DATA/'revised_labels.npz');baseline=np.load(DATA/'baseline_labels.npz')
    gold=np.load(ROOT/'data/confirmation_e4/posthoc_morphospecies_labels.npz')
    for key in ['truth','known','known_fraction']:np.testing.assert_array_equal(baseline[key],gold[key])
    contract=np.load(ROOT/'data/confirmation_e4/label_contract.npz');eligible=contract['neon_eligible']
    y,k=label['truth'],label['known'];assert not np.any(k&~eligible) and not np.any((y>0)&~k)
    assert np.isin(y[k],[0,1]).all() and np.all(y[~k]==-1)
    np.testing.assert_array_equal(label['known_fraction'],k.sum(1)/eligible.sum(1))
    # Every positive requires a current-record witness; no history identity enters.
    witness=np.zeros_like(eligible)
    for table,row in observations:
        if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
        resolution,status,conflicts=resolutions.get((row['siteID'],row.get('morphospeciesID','')),(None,'no_dated_resolution',[]))
        new,reason,evidence=combine_current_record(row,resolution,status,conflicts,taxa)
        if evidence or not rank_matches(row,taxa) or not rank_matches(new,taxa):continue
        if new.get('identificationQualifier','').strip().casefold() not in {'','na','none','no qualifier'}:continue
        if new.get('dataQF','').strip() not in {'','0','0.0'}:continue
        code=new.get('taxonID','').strip();index=mapping.get(code)
        if index is None:continue
        current=taxa[code]
        if current.get('dwc:taxonRank')!='species':
            seen=set();terminal=current
            while terminal['acceptedTaxonID']!=terminal['taxonID']:
                assert terminal['taxonID'] not in seen;seen.add(terminal['taxonID']);terminal=taxa[terminal['acceptedTaxonID']]
            assert terminal['dwc:taxonRank']=='species'
        cell=row_for_cell[cell_for_plot[row['plotID']]]
        if eligible[cell,index]:witness[cell,index]=True
    np.testing.assert_array_equal(witness,y>0)
    p=np.load(ROOT/'data/confirmation_e4/frozen_policy.npz');n,classes=y.shape
    pred=np.zeros_like(eligible)
    for i in range(n):
        indices=p['rank'][i,:p['count'][i]];assert len(indices)==len(set(indices));pred[i,indices]=True
    assert not np.any(pred&~eligible)
    hits=(pred&(y>0)).sum(1);positive=(y>0).sum(1);den=(pred&k).sum(1)+positive
    score=np.divide(2*hits,den,out=np.ones(n),where=den>0)
    unknown_pred=(pred&~k).sum(1);unknown_other=(eligible&~k&~pred).sum(1)
    low_den=pred.sum(1)+positive+unknown_other;high_den=pred.sum(1)+positive+unknown_pred
    lower=np.divide(2*hits,low_den,out=np.ones(n),where=low_den>0)
    upper=np.divide(2*(hits+unknown_pred),high_den,out=np.ones(n),where=high_den>0)
    rows=pd.read_csv(OUT/'cell_results.csv');assert rows.cell_id.tolist()==cells.cell_id.tolist()
    for key,value in [('F1',score),('lower',lower),('upper',upper)]:np.testing.assert_allclose(rows[key],value,atol=1e-12)
    table=pd.read_csv(OUT/'cohort_cells.csv');cfg=json.loads((ROOT/'configs/neon_taxonomy_a2.json').read_text())
    summary=json.loads((OUT/'score_summary.json').read_text())
    selectors=0
    for cohort,frame in table.groupby('cohort',sort=False):
        indices=np.array([row_for_cell[v] for v in frame.cell_id]);meta=cells.iloc[indices]
        groups=(meta.site.astype(str)+'|'+meta.area_group.astype(str)).to_numpy();group_names,sizes=np.unique(groups,return_counts=True)
        total=int(np.ceil(len(frame)*cfg['coverage']));exact=sizes*total/len(frame);quotas=np.floor(exact).astype(int)
        order=sorted(range(len(group_names)),key=lambda i:(-(exact[i]-quotas[i]),number(group_names[i])))
        for i in order[:total-int(quotas.sum())]:quotas[i]+=1
        chosen={}
        for name,key in [('HGB','HGB_quality'),('SDC','SDC'),('mean_confidence','mean_confidence'),('entropy','entropy'),('random','random')]:
            risk=p[key][indices] if name in ['entropy','random'] else 1-p[key][indices]
            accept=np.zeros(len(frame),bool)
            for group,quota in zip(group_names,quotas):
                take=np.flatnonzero(groups==group)
                ranking=sorted(take,key=lambda i:(float(risk[i]),number('20261113:'+str(frame.iloc[i].cell_id))))
                accept[ranking[:quota]]=True
            np.testing.assert_array_equal(accept,frame['accept_'+name].to_numpy(bool));chosen[name]=accept;selectors+=1
            assert np.isclose(score[indices][accept].mean(),summary['cohorts'][cohort]['selective_F1'][name])
        w=chosen['HGB']/chosen['HGB'].sum()-chosen['SDC']/chosen['SDC'].sum()
        lo=np.where(w>=0,w*lower[indices],w*upper[indices]).sum();hi=np.where(w>=0,w*upper[indices],w*lower[indices]).sum()
        np.testing.assert_allclose([lo,hi],summary['cohorts'][cohort]['HGB_minus_SDC_identification_range'],atol=1e-12)
    history=verify_history();extra={}
    for name in ['taxonomic_resolution_t1','neon_taxonomy_a1']:
        m=json.loads((ROOT/'results'/name/'artifact_manifest.json').read_text())
        for file,d in m['files'].items():
            path=ROOT/file;assert path.stat().st_size==d['bytes'] and sha(path)==d['sha256']
        extra[name]=len(m['files'])
    save(OUT/'validation.json',{'passed':True,'at_utc':now(),'baseline_bit_identical':True,
        'independent_confusion_rows':n,'current_positive_witnesses':int(witness.sum()),
        'independent_selector_rankings':selectors,'selected_scope_and_history_identity_guards':True,
        'archived_taxon_labels_used_as_replacements':False,'chronology_and_frozen_inputs_verified':True,
        'historical_files_verified':{name:row['files_verified'] for name,row in history.items()},'additional_frozen_files_verified':extra})
    print(json.dumps({'passed':True,'independent_cells':n,'current_positive_witnesses':int(witness.sum()),'selector_checks':selectors}),flush=True)


if __name__=='__main__':main()
