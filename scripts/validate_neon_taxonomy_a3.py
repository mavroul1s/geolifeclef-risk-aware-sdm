"""Independent alias witnesses, label evidence, scores and group budgets."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_neon_taxonomy_a3 import DATA,OUT,freeze
from run_neon_taxonomy_a2 import metadata,cached,sha,save,now
from geolifeclef.taxonomy_bridge import canonical_name,normalized_name
from geolifeclef.neon_record_contract import NAMED_RANKS,current_resolution,combine_current_record,rank_matches


def terminal(code,taxa):
    seen=set()
    while code not in seen and code in taxa:
        seen.add(code);r=taxa[code]
        if r['acceptedTaxonID']==code:return r
        code=r['acceptedTaxonID']
    return None


def number(text):return int(hashlib.sha256(str(text).encode()).hexdigest()[:15],16)


def main():
    freeze();lock=json.loads((OUT/'labels_freeze.json').read_text());assert sha(DATA/'revised_labels.npz')==lock['labels_sha256']
    assert sha(OUT/'dictionary_alias_extensions.csv')==lock['dictionary_links_sha256']
    times=[json.loads((OUT/n).read_text())['at_utc'] for n in ['protocol_freeze.json','labels_freeze.json','score_access.json']]
    assert times==sorted(times) and len(set(times))==3
    taxa,old,names,decisions,mapping=metadata();name_to_class=defaultdict(set)
    for i,name in enumerate(names):name_to_class[normalized_name(name)].add(i)
    groups=defaultdict(set)
    for code,t in taxa.items():
        if t['dwc:taxonRank']!='species':continue
        end=terminal(code,taxa)
        if end is not None and end['dwc:taxonRank']=='species':groups[end['taxonID']].update(name_to_class[canonical_name(t)])
    added=pd.read_csv(OUT/'dictionary_alias_extensions.csv');new_mapping=dict(mapping)
    for r in added.itertuples():
        assert r.neon_taxon_id not in mapping and taxa[r.neon_taxon_id]['dwc:taxonRank'] in NAMED_RANKS
        end=terminal(r.neon_taxon_id,taxa);assert end['dwc:taxonRank']=='species' and end['taxonID']==r.accepted_species_id
        assert groups[end['taxonID']]=={int(r.sage_index)};new_mapping[r.neon_taxon_id]=int(r.sage_index)
    cells=pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv');plots=pd.read_csv(ROOT/'data/confirmation_e4/selected_plots.csv')
    row_for_cell={v:i for i,v in enumerate(cells.cell_id)};cell_for_plot=dict(zip(plots.plot_id,plots.cell_id))
    eligible=np.load(ROOT/'data/confirmation_e4/label_contract.npz')['neon_eligible'];witness=np.zeros_like(eligible)
    morpho={tuple(k):current_resolution(v) for k,v in cached('selected_morphospecies.json.gz')}
    for table,row in cached('selected_observations.json.gz'):
        if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
        res,status,conflicts=morpho.get((row['siteID'],row.get('morphospeciesID','')),(None,'no_dated_resolution',[]))
        chosen,reason,evidence=combine_current_record(row,res,status,conflicts,taxa)
        if evidence or not rank_matches(row,taxa) or not rank_matches(chosen,taxa):continue
        if chosen.get('identificationQualifier','').strip().casefold() not in {'','na','none','no qualifier'} or chosen.get('dataQF','').strip() not in {'','0','0.0'}:continue
        index=new_mapping.get(chosen.get('taxonID','').strip());cell=row_for_cell[cell_for_plot[row['plotID']]]
        if index is not None and eligible[cell,index]:witness[cell,index]=True
    label=np.load(DATA/'revised_labels.npz');y,k=label['truth'],label['known'];previous=np.load(ROOT/'data/neon_taxonomy_a2/revised_labels.npz')
    np.testing.assert_array_equal(witness,y>0);assert not np.any((previous['truth']>0)&~witness)
    assert not np.any(k&~eligible) and np.isin(y[k],[0,1]).all() and (y[~k]==-1).all()
    np.testing.assert_array_equal(label['known_fraction'],k.sum(1)/eligible.sum(1))
    p=np.load(ROOT/'data/confirmation_e4/frozen_policy.npz');pred=np.zeros_like(eligible)
    for i in range(len(cells)):pred[i,p['rank'][i,:p['count'][i]]]=True
    hits=(pred&(y>0)).sum(1);positive=(y>0).sum(1);den=(pred&k).sum(1)+positive
    q=np.divide(2*hits,den,out=np.ones(len(cells)),where=den>0)
    unknown_pred=(pred&~k).sum(1);unknown_other=(eligible&~k&~pred).sum(1)
    ld=pred.sum(1)+positive+unknown_other;ud=pred.sum(1)+positive+unknown_pred
    lo=np.divide(2*hits,ld,out=np.ones(len(cells)),where=ld>0);hi=np.divide(2*(hits+unknown_pred),ud,out=np.ones(len(cells)),where=ud>0)
    rows=pd.read_csv(OUT/'cell_results.csv')
    for key,value in [('F1',q),('lower',lo),('upper',hi)]:np.testing.assert_allclose(rows[key],value,atol=1e-12)
    cohorts=pd.read_csv(OUT/'cohort_cells.csv');summary=json.loads((OUT/'score_summary.json').read_text());checks=0
    for name,frame in cohorts.groupby('cohort',sort=False):
        ids=np.array([row_for_cell[v] for v in frame.cell_id]);meta=cells.iloc[ids]
        groups=(meta.site.astype(str)+'|'+meta.area_group.astype(str)).to_numpy();gn,sizes=np.unique(groups,return_counts=True)
        total=int(np.ceil(.8*len(frame)));exact=sizes*total/len(frame);quotas=np.floor(exact).astype(int)
        order=sorted(range(len(gn)),key=lambda i:(-(exact[i]-quotas[i]),number(gn[i])))
        for i in order[:total-int(quotas.sum())]:quotas[i]+=1
        chosen={}
        for selector,column in [('HGB','HGB_quality'),('SDC','SDC'),('random','random'),('mean_confidence','mean_confidence'),('entropy','entropy')]:
            risk=p[column][ids] if selector in ['random','entropy'] else 1-p[column][ids];accept=np.zeros(len(frame),bool)
            for group,quota in zip(gn,quotas):
                take=np.flatnonzero(groups==group);ranking=sorted(take,key=lambda i:(float(risk[i]),number('20261113:'+str(frame.iloc[i].cell_id))))
                accept[ranking[:quota]]=True
            np.testing.assert_array_equal(accept,frame['accept_'+selector].to_numpy(bool));chosen[selector]=accept;checks+=1
            assert np.isclose(q[ids][accept].mean(),summary['cohorts'][name]['selective_F1'][selector])
        w=chosen['HGB']/chosen['HGB'].sum()-chosen['SDC']/chosen['SDC'].sum()
        bounds=[np.where(w>=0,w*lo[ids],w*hi[ids]).sum(),np.where(w>=0,w*hi[ids],w*lo[ids]).sum()]
        np.testing.assert_allclose(bounds,summary['cohorts'][name]['HGB_minus_SDC_identification_range'],atol=1e-12)
    save(OUT/'validation.json',{'passed':True,'at_utc':now(),'alias_links_independently_verified':len(added),
        'independent_confusion_rows':len(cells),'independent_selector_rankings':checks,'positive_current_record_witnesses':int(witness.sum()),
        'existing_A2_positives_preserved':True,'global_uniqueness_before_eligibility_filtering':True,'chronology_and_frozen_inputs_verified':True})
    print(json.dumps({'passed':True,'alias_links_verified':len(added),'current_positive_pairs':int(witness.sum()),'selector_checks':checks}),flush=True)


if __name__=='__main__':main()
