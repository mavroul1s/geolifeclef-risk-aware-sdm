"""One exact-synonym extension on A2's cached consumed records; no refits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_neon_taxonomy_a2 import metadata,cached,sha,save,now
from geolifeclef.neon_record_contract import AcceptedConceptAssessment,current_resolution,combine_current_record,rank_matches
from geolifeclef.taxonomic_equivalence import accepted_alias_classes,extend_exact_alias_mapping
from geolifeclef.taxonomy_bridge import normalized_name
from geolifeclef.taxonomy_contract import accepted_concept
from geolifeclef.neon_assessment import score_fixed_sets,site_primary_comparison
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.partial_set_bounds import fixed_set_f1_bounds,linear_contrast_bounds

DATA,OUT=ROOT/'data/neon_taxonomy_a3',ROOT/'results/neon_taxonomy_a3'
SOURCES=['configs/neon_taxonomy_a3.json','docs/neon_taxonomy_a3_protocol.md','scripts/run_neon_taxonomy_a3.py',
    'scripts/validate_neon_taxonomy_a3.py','src/geolifeclef/taxonomic_equivalence.py','tests/test_taxonomic_equivalence.py']
INPUTS=['results/neon_taxonomy_a2/protocol_freeze.json','results/neon_taxonomy_a2/labels_freeze.json',
    'results/neon_taxonomy_a2/validation.json','data/neon_taxonomy_a2/selected_observations.json.gz',
    'data/neon_taxonomy_a2/selected_morphospecies.json.gz','data/neon_taxonomy_a2/revised_labels.npz',
    'scripts/run_neon_taxonomy_a2.py','src/geolifeclef/neon_record_contract.py','src/geolifeclef/taxonomy_contract.py',
    'data/confirmation_e4/label_contract.npz','data/confirmation_e4/frozen_policy.npz',
    'data/confirmation_e4/neon_assessment_roles.csv','data/confirmation_e4/selected_plots.csv',
    'data/confirmation_e4/assessment_labels.npz','data/confirmation_e4/posthoc_morphospecies_labels.npz',
    'data/neon_bridge/neon_to_sage_taxonomy.csv','data/neon_bridge/species_names.csv','results/neon_bridge/taxonomy_retrieval.json']


def freeze():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    identity={name:sha(ROOT/name) for name in SOURCES+INPUTS}
    receipt=json.loads((ROOT/'results/neon_bridge/taxonomy_retrieval.json').read_text())
    for f in receipt['files']:
        identity[f['file']]=sha(ROOT/f['file']);assert identity[f['file']]==f['sha256']
    path=OUT/'protocol_freeze.json'
    if path.exists():
        assert json.loads(path.read_text())['identity']==identity;return
    for name in SOURCES:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    save(path,{'at_utc':now(),'identity':identity,'role':'adaptive development correction after A2 metadata diagnosis','A3_labels_not_generated':True})


def labels():
    freeze()
    if (OUT/'labels_freeze.json').exists():raise ValueError('A3 labels already frozen')
    taxa,old,names,decisions,base_mapping=metadata();sage={}
    for i,name in enumerate(names):sage.setdefault(normalized_name(name),[]).append(i)
    groups=accepted_alias_classes(taxa,sage);mapping,audit=extend_exact_alias_mapping(taxa,base_mapping,groups)
    new_links=[]
    for code,index in mapping.items():
        if code not in base_mapping:
            terminal,_=accepted_concept(code,taxa)
            new_links.append({'neon_taxon_id':code,'accepted_species_id':terminal['taxonID'],'sage_index':index,
                              'model_species_name':names[index],'evidence':'one exact model alias in explicit accepted-species group'})
    pd.DataFrame(new_links).to_csv(OUT/'dictionary_alias_extensions.csv',index=False)
    observations=cached('selected_observations.json.gz');morpho={tuple(k):v for k,v in cached('selected_morphospecies.json.gz')}
    resolutions={key:current_resolution(rows) for key,rows in morpho.items()}
    cells=pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv');plots=pd.read_csv(ROOT/'data/confirmation_e4/selected_plots.csv')
    cell_for_plot=dict(zip(plots.plot_id,plots.cell_id));row_for_cell={v:i for i,v in enumerate(cells.cell_id)}
    eligible=np.load(ROOT/'data/confirmation_e4/label_contract.npz')['neon_eligible']
    reducer=AcceptedConceptAssessment(eligible,taxa,mapping,names);new_record_status=Counter()
    for table,row in observations:
        if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
        cell=row_for_cell[cell_for_plot[row['plotID']]]
        resolution,status,conflicts=resolutions.get((row['siteID'],row.get('morphospeciesID','')),(None,'no_dated_resolution',[]))
        chosen,reason,evidence=combine_current_record(row,resolution,status,conflicts,taxa)
        if evidence:
            for item in evidence:reducer.uncertain[cell,reducer.compatible(taxa.get(item.get('taxonID')))]=True
            continue
        if not rank_matches(row,taxa) or not rank_matches(chosen,taxa):
            reducer.uncertain[cell,reducer.compatible(taxa.get(chosen.get('taxonID')))]=True;continue
        code=chosen.get('taxonID','')
        if code not in base_mapping and code in mapping:
            clean=chosen.get('identificationQualifier','').strip().casefold() in {'','na','none','no qualifier'} and chosen.get('dataQF','').strip() in {'','0','0.0'}
            new_record_status['clean_new_alias_records' if clean else 'qualified_new_alias_records']+=1
            if clean:new_record_status['inside_fixed_eligible' if eligible[cell,mapping[code]] else 'outside_fixed_eligible']+=1
        reducer.add(cell,chosen,table)
    y,k,f,counts=reducer.finish();previous=np.load(ROOT/'data/neon_taxonomy_a2/revised_labels.npz')
    assert not np.any((previous['truth']>0)&~(y>0))
    np.savez_compressed(DATA/'revised_labels.npz',truth=y,known=k,known_fraction=f)
    save(OUT/'mapping_audit.json',{'at_utc':now(),'dictionary_added_taxon_links':len(new_links),
        'dictionary_extension_status':dict(Counter(audit.values())),'selected_record_extension':dict(new_record_status),
        'reducer_record_counts':counts,'cells':len(cells),'scorable_cells':int((f>=.8).sum()),
        'scorable_sites':int(cells.loc[f>=.8,'site'].nunique()),'known_fraction_median':float(np.median(f)),
        'newly_known_pairs':int((k&~previous['known']).sum()),'newly_unknown_pairs':int((previous['known']&~k).sum()),
        'added_positive_pairs':int(((y>0)&~(previous['truth']>0)).sum()),'removed_positive_pairs':0,
        'lexical_parents_applied':0,'new_HTTP_composition_requests':0,'new_model_fits':0,'new_GPU_jobs':0,
        'A3_labels_not_yet_scored':True,'existing_exact_mappings_retained':True})
    save(OUT/'labels_freeze.json',{'at_utc':now(),'labels_sha256':sha(DATA/'revised_labels.npz'),
        'dictionary_links_sha256':sha(OUT/'dictionary_alias_extensions.csv'),'protocol_freeze_sha256':sha(OUT/'protocol_freeze.json'),
        'A3_F1_not_evaluated':True})
    print((OUT/'mapping_audit.json').read_text(),flush=True)


def score(config):
    freeze();lock=json.loads((OUT/'labels_freeze.json').read_text())
    assert lock['labels_sha256']==sha(DATA/'revised_labels.npz') and lock['dictionary_links_sha256']==sha(OUT/'dictionary_alias_extensions.csv')
    if (OUT/'score_summary.json').exists():raise ValueError('A3 score already complete')
    save(OUT/'score_access.json',{'at_utc':now(),'labels_freeze_sha256':sha(OUT/'labels_freeze.json')})
    cells=pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv');p=np.load(ROOT/'data/confirmation_e4/frozen_policy.npz')
    eligible=np.load(ROOT/'data/confirmation_e4/label_contract.npz')['neon_eligible']
    previous=np.load(ROOT/'data/neon_taxonomy_a2/revised_labels.npz');new=np.load(DATA/'revised_labels.npz')
    original=np.load(ROOT/'data/confirmation_e4/assessment_labels.npz');morpho=np.load(ROOT/'data/confirmation_e4/posthoc_morphospecies_labels.npz')
    y,k,f=new['truth'],new['known'],new['known_fraction']
    q0=score_fixed_sets(previous['truth'],previous['known'],p['rank'],p['count'])['F1'];q=score_fixed_sets(y,k,p['rank'],p['count'])['F1']
    lower,upper=fixed_set_f1_bounds(y,k,eligible,p['rank'],p['count'])
    cohorts={'original_65':original['known_fraction']>=.8,'previous_morphospecies_77':morpho['known_fraction']>=.8,
        'all_163':np.ones(len(cells),bool),'A2_fixed31':previous['known_fraction']>=.8,'A3_80pct_known':f>=.8}
    result={};records=[]
    for name,take in cohorts.items():
        if not take.any():continue
        meta=cells.loc[take];groups=(meta.site.astype(str)+'|'+meta.area_group.astype(str)).to_numpy();selector=QuotaSelection(meta.cell_id.to_numpy(),config['coverage'],groups)
        risks={'HGB':1-p['HGB_quality'][take],'SDC':1-p['SDC'][take],'mean_confidence':1-p['mean_confidence'][take],'entropy':p['entropy'][take],'random':p['random'][take]}
        chosen={key:selector.accept(value) for key,value in risks.items()};a,b=chosen['HGB'],chosen['SDC']
        comparison=site_primary_comparison(q[take],p['HGB_quality'][take],float(p['mean_quality'][0]),a,b,meta.site,
            **{key:value for key,value in config['bootstrap'].items() if key!='cluster'})
        result[name]={'cells':int(take.sum()),'sites':int(meta.site.nunique()),'known_F1':float(q[take].mean()),
            'A2_known_F1_same_cells':float(q0[take].mean()),'score_change_from_A2_same_cells':float((q-q0)[take].mean()),
            'mean_known_fraction':float(f[take].mean()),'mean_F1_identification_width':float((upper-lower)[take].mean()),
            'F1_identification_range':[float(lower[take].mean()),float(upper[take].mean())],
            'HGB_minus_SDC_identification_range':linear_contrast_bounds(lower[take],upper[take],a/a.sum()-b/b.sum()),
            'descriptive_comparison':comparison,
            'quality_MSE':{key:float(np.mean((p[column][take]-q[take])**2)) for key,column in [('training_mean','mean_quality'),('HGB','HGB_quality'),('affine_SDC','affine_SDC_quality'),('raw_SDC','SDC')]},
            'selective_F1':{key:float(q[take][mask].mean()) for key,mask in chosen.items()},
            'meets_historical_100cell_12site_reference':bool(take.sum()>=100 and meta.site.nunique()>=12)}
        for j,index in enumerate(np.flatnonzero(take)):
            records.append({'cohort':name,'cell_id':cells.iloc[index].cell_id,'site':cells.iloc[index].site,
                'F1':q[index],'A2_F1':q0[index],'lower':lower[index],'upper':upper[index],**{'accept_'+key:bool(mask[j]) for key,mask in chosen.items()}})
    pd.DataFrame(records).to_csv(OUT/'cohort_cells.csv',index=False)
    pd.DataFrame({'cell_id':cells.cell_id,'site':cells.site,'known_fraction':f,'A2_known_fraction':previous['known_fraction'],
                  'F1':q,'A2_F1':q0,'lower':lower,'upper':upper}).to_csv(OUT/'cell_results.csv',index=False)
    save(OUT/'score_summary.json',{'status':'exact_alias_development_replay_complete','at_utc':now(),'cohorts':result,
        'new_model_fits':0,'new_GPU_jobs':0,'original_E4_primary_changed':False,'fresh_confirmation':False,
        'scope':'exact name-equivalence extension; no merged ontology, complete-field oracle or new-method success claim'})
    print(json.dumps({'status':'scoring_complete','cohorts':result}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','labels','score']);args=parser.parse_args()
    if args.phase=='freeze':freeze();print('A3 development protocol frozen')
    elif args.phase=='labels':labels()
    else:score(json.loads((ROOT/'configs/neon_taxonomy_a3.json').read_text()))


if __name__=='__main__':main()
