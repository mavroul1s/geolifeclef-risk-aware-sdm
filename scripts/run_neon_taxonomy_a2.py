"""Selected-record audit and fixed-policy development replay; preserve E4."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geolifeclef.neon_assessment import selected_csv_rows, TaxonomicAssessment, score_fixed_sets, site_primary_comparison
from geolifeclef.neon_morphospecies import latest_resolution
from geolifeclef.neon_record_contract import (NAMED_RANKS, AcceptedConceptAssessment, selected_key_rows, accepted_species_mapping,
    public_rank, current_resolution, combine_current_record, rank_matches, history_table_matches)
from geolifeclef.taxonomy_bridge import normalized_name
from geolifeclef.taxonomy_contract import INFRASPECIFIC, parent_candidates, species_name_index
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.partial_set_bounds import fixed_set_f1_bounds, linear_contrast_bounds

DATA, OUT = ROOT/'data/neon_taxonomy_a2', ROOT/'results/neon_taxonomy_a2'
SOURCES = ['configs/neon_taxonomy_a2.json','docs/neon_taxonomy_a2_protocol.md',
    'scripts/run_neon_taxonomy_a2.py','scripts/validate_neon_taxonomy_a2.py',
    'src/geolifeclef/neon_record_contract.py','tests/test_neon_record_contract.py']
INPUTS = ['data/confirmation_e4/selected_plots.csv','data/confirmation_e4/neon_assessment_roles.csv',
    'data/confirmation_e4/label_contract.npz','data/confirmation_e4/frozen_policy.npz',
    'data/confirmation_e4/assessment_labels.npz','data/confirmation_e4/posthoc_morphospecies_labels.npz',
    'data/confirmation_e4/posthoc_expanded_manifest.json','data/neon_bridge/neon_2023_manifest.json',
    'data/neon_bridge/neon_to_sage_taxonomy.csv','data/neon_bridge/species_names.csv',
    'results/neon_bridge/taxonomy_retrieval.json','results/confirmation_e4/target_mapping_audit.json',
    'results/confirmation_e4/primary_result.json','results/confirmation_e4/posthoc_morphospecies/summary.json',
    'src/geolifeclef/neon_assessment.py','src/geolifeclef/neon_morphospecies.py',
    'src/geolifeclef/taxonomy_contract.py','src/geolifeclef/taxonomy_bridge.py',
    'src/geolifeclef/ecological_selection.py','src/geolifeclef/partial_set_bounds.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')


def cache(name, value):
    with gzip.open(DATA/name,'wt',encoding='utf-8') as f:
        json.dump(value,f)


def cached(name):
    with gzip.open(DATA/name,'rt',encoding='utf-8') as f:
        return json.load(f)


def freeze():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    taxonomy=json.loads((ROOT/'results/neon_bridge/taxonomy_retrieval.json').read_text())
    identity={name:sha(ROOT/name) for name in SOURCES+INPUTS}
    identity.update({f['file']:sha(ROOT/f['file']) for f in taxonomy['files']})
    assert all(identity[f['file']]==f['sha256'] for f in taxonomy['files'])
    path=OUT/'protocol_freeze.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['identity']==identity,'A2 frozen source/input changed'
        return old
    for name in SOURCES:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    value={'at_utc':now(),'identity':identity,'selected_records_retrieved_by_A2':False,
           'role':'development on the consumed E4 cohort, not external registration or fresh confirmation'}
    save(path,value);return value


def metadata():
    receipt=json.loads((ROOT/'results/neon_bridge/taxonomy_retrieval.json').read_text())
    taxa={r['taxonID']:r for f in receipt['files'] for r in json.loads((ROOT/f['file']).read_text())['data']}
    frame=pd.read_csv(ROOT/'data/neon_bridge/neon_to_sage_taxonomy.csv')
    old={r.neon_taxon_id:int(r.sage_index) for r in frame[frame.sage_index.notna()].itertuples()}
    names=pd.read_csv(ROOT/'data/neon_bridge/species_names.csv')['Species Name'].tolist()
    sage={}
    for i,name in enumerate(names):sage.setdefault(normalized_name(name),[]).append(i)
    decisions={code:accepted_species_mapping(t,taxa,sage) for code,t in taxa.items()}
    mapping={code:r['sage_index'] for code,r in decisions.items() if r['sage_index'] is not None}
    return taxa,old,names,decisions,mapping


def audit(config):
    freeze()
    if (OUT/'labels_freeze.json').exists():raise ValueError('A2 labels already frozen; no overwrite')
    plots=pd.read_csv(ROOT/'data/confirmation_e4/selected_plots.csv')
    cells=pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv')
    selected=dict(zip(plots.plot_id,plots.bout.astype(str)));cell_for_plot=dict(zip(plots.plot_id,plots.cell_id))
    row_for_cell={v:i for i,v in enumerate(cells.cell_id)}
    assert len(selected)==301 and len(cells)==163
    manifest=json.loads((ROOT/'data/neon_bridge/neon_2023_manifest.json').read_text())['data']
    receipt=json.loads((ROOT/'results/confirmation_e4/target_mapping_audit.json').read_text())
    wanted={f['name']:f for f in receipt['MD5_receipts']}
    obs=[f for r in manifest['releases'] for p in r['packages'] for f in p['files'] if f['name'] in wanted]
    expanded=json.loads((ROOT/'data/confirmation_e4/posthoc_expanded_manifest.json').read_text())['data']
    extra={f['url']:f for r in expanded['releases'] for p in r['packages'] for f in p['files']}
    morpho=[f for f in extra.values() if '.div_morphospecies.' in f['name']]
    history=[f for f in extra.values() if '.div_identificationHistory.' in f['name']]
    assert all(r['release']==config['release'] for r in expanded['releases'])
    assert sum(f['size'] for f in obs+morpho+history)<=config['download_max_bytes']
    receipts=[]
    def fetch(file):
        with urllib.request.urlopen(file['url'],timeout=30) as response:body=response.read(file['size']+1)
        assert len(body)==file['size'] and hashlib.md5(body).hexdigest()==file['md5'],'Published NEON file identity mismatch'
        return file,body
    def receipt_for(file,body):
        receipts.append({'file':file['name'],'bytes':len(body),'md5_verified':file['md5'],'sha256':hashlib.sha256(body).hexdigest()})
    save(OUT/'record_access.json',{'at_utc':now(),'protocol_freeze_sha256':sha(OUT/'protocol_freeze.json'),
         'plots':len(selected),'cells':len(cells),'new_plot_or_bout_outcomes':False,'raw_source_tables_retained':False})
    observed=[];morpho_keys=set()
    allowed=['plotID','siteID','boutNumber','morphospeciesID','taxonID','targetTaxaPresent','divDataType',
             'identificationQualifier','dataQF','identificationHistoryID','taxonRank','scientificName','eventID','endDate']
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        for file,body in pool.map(fetch,obs):
            table='div_1m2Data' if '.div_1m2Data.' in file['name'] else 'div_10m2Data100m2Data'
            for row in selected_csv_rows(body,selected,allowed):
                observed.append([table,row])
                if row.get('morphospeciesID'):morpho_keys.add((row['siteID'],row['morphospeciesID']))
            receipt_for(file,body)
    assert {r['plotID'] for _,r in observed}==set(selected)
    cache('selected_observations.json.gz',observed)
    print(json.dumps({'selected_observation_rows':len(observed),'morphospecies_keys':len(morpho_keys)}),flush=True)
    matches=defaultdict(list)
    morpho_fields=['morphospeciesResolvedDate','taxonID','taxonRank','scientificName','identificationQualifier','dataQF','identificationHistoryID']
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        for file,body in pool.map(fetch,morpho):
            for row in selected_key_rows(body,morpho_keys,['siteID','morphospeciesID'],morpho_fields):
                matches[(row['siteID'],row['morphospeciesID'])].append(row)
            receipt_for(file,body)
    resolutions={key:current_resolution(matches.get(key,[])) for key in morpho_keys}
    cache('selected_morphospecies.json.gz',list(matches.items()))
    taxa,old,names,decisions,mapping=metadata()
    contract=np.load(ROOT/'data/confirmation_e4/label_contract.npz');eligible=contract['neon_eligible']
    # Reproduce the historical post-hoc control without changing its original files.
    baseline=TaxonomicAssessment(eligible,taxa,old,names)
    for table,row in observed:
        cell=row_for_cell[cell_for_plot[row['plotID']]];new=dict(row)
        resolution,_=latest_resolution(matches.get((row['siteID'],row.get('morphospeciesID','')),[]))
        if resolution is not None:
            a,b=row.get('taxonID',''),resolution.get('taxonID','')
            if a in old and b in old and old[a]!=old[b]:
                for code in [a,b]:baseline.uncertain[cell,baseline.compatible(taxa.get(code))]=True
                continue
            for field in ['taxonID','identificationQualifier','dataQF']:new[field]=resolution.get(field,'')
        baseline.add(cell,new,table)
    by, bk, bf, bc=baseline.finish();original=np.load(ROOT/'data/confirmation_e4/posthoc_morphospecies_labels.npz')
    np.testing.assert_array_equal(by,original['truth']);np.testing.assert_array_equal(bk,original['known'])
    np.testing.assert_array_equal(bf,original['known_fraction'])
    np.savez_compressed(DATA/'baseline_labels.npz',truth=by,known=bk,known_fraction=bf)
    # Only already selected observation-history IDs are followed, with site guard.
    linked=defaultdict(list);public_identity={}
    for table,row in observed:
        if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
        code=row.get('identificationHistoryID','').strip()
        if code:linked[(row['siteID'],code)].append([table,row])
    for key,rows in linked.items():
        evidence=[]
        for _,row in rows:
            evidence.append(row)
            r,status,conflicts=resolutions.get((row['siteID'],row.get('morphospeciesID','')),(None,'no_dated_resolution',[]))
            evidence.extend([r] if r is not None else conflicts)
        public_identity[key]=all(public_rank(r,taxa) in NAMED_RANKS for r in evidence)
    histories=[]
    with ThreadPoolExecutor(max_workers=config['download_workers']) as pool:
        for file,body in pool.map(fetch,history):
            histories.extend(selected_key_rows(body,set(linked),['siteID','identificationHistoryID'],
                ['identificationTable','identifiedDate','endDate','identificationQualifier','dataQF'],
                {k for k,v in public_identity.items() if v},['taxonID','acceptedTaxonID']))
            receipt_for(file,body)
    histories=list({json.dumps(r,sort_keys=True):r for r in histories}.values())
    cache('selected_history.json.gz',histories)
    history_by_key=defaultdict(list)
    for row in histories:history_by_key[(row['siteID'],row['identificationHistoryID'])].append(row)
    history_counts=Counter();past_same=past_different=0
    for key,rows in linked.items():
        history_counts['keys_with_history_rows' if history_by_key[key] else 'keys_without_history_rows']+=1
        for table,row in rows:
            matched=[h for h in history_by_key[key] if history_table_matches(h,table)]
            history_counts['current_rows_with_table_matched_history' if matched else 'current_rows_without_table_matched_history']+=1
            for h in matched:
                if 'taxonID' in h:
                    if h['taxonID']==row.get('taxonID',''):past_same+=1
                    else:past_different+=1
    revised=AcceptedConceptAssessment(eligible,taxa,mapping,names)
    reasons=Counter();mapping_counts=Counter();parent_counts=Counter();events=defaultdict(set)
    name_index=species_name_index(taxa)
    for table,row in observed:
        if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
        cell=row_for_cell[cell_for_plot[row['plotID']]]
        if row.get('eventID'):events[row['plotID']].add(row['eventID'])
        resolution,status,conflicts=resolutions.get((row['siteID'],row.get('morphospeciesID','')),(None,'no_dated_resolution',[]))
        new,reason,evidence=combine_current_record(row,resolution,status,conflicts,taxa)
        reasons[reason]+=1
        if evidence:
            for item in evidence:revised.uncertain[cell,revised.compatible(taxa.get(item.get('taxonID')))]=True
            continue
        if not rank_matches(row,taxa) or not rank_matches(new,taxa):
            reasons['declared_dictionary_rank_mismatch']+=1
            revised.uncertain[cell,revised.compatible(taxa.get(new.get('taxonID')))]=True
            continue
        code=new.get('taxonID','');decision=decisions.get(code,{'sage_index':None,'status':'missing_dictionary_identity'})
        mapping_counts[decision['status']]+=1
        t=taxa.get(code)
        if t and t.get('dwc:taxonRank') in INFRASPECIFIC:
            parent_counts[parent_candidates(t,taxa,name_index)['status']]+=1
        revised.add(cell,new,table)
    y,k,f,counts=revised.finish()
    np.savez_compressed(DATA/'revised_labels.npz',truth=y,known=k,known_fraction=f)
    np.savez_compressed(DATA/'label_changes.npz',newly_known=k&~bk,newly_unknown=bk&~k,
                        added_positives=(y>0)&~(by>0),removed_positives=(by>0)&~(y>0))
    save(OUT/'file_receipts.json',receipts)
    result={'status':'selected_record_audit_complete','at_utc':now(),'baseline_reproduced_bit_for_bit':True,
        'selected_plots':len(plots),'cells':len(cells),'selected_rows':len(observed),
        'morphospecies_keys':len(morpho_keys),'morphospecies_resolution_status':dict(Counter(r[1] for r in resolutions.values())),
        'history_linked_keys':len(linked),'history_unique_selected_rows':len(histories),'history_linkage':dict(history_counts),
        'history_keys_restricted_to_metadata':sum(not x for x in public_identity.values()),
        'history_keys_permitting_identity_audit':sum(public_identity.values()),
        'past_current_same_ID_pairs':past_same,'past_current_different_ID_pairs':past_different,
        'archived_labels_used_as_replacements':0,'unselected_taxon_values_decoded':False,
        'current_resolution_decisions':dict(reasons),'current_mapping_status':dict(mapping_counts),
        'infraspecific_current_record_status':dict(parent_counts),'reducer_record_counts':counts,
        'plots_with_multiple_event_IDs':sum(len(v)>1 for v in events.values()),
        'baseline_scorable_cells':int((bf>=.8).sum()),'revised_scorable_cells':int((f>=.8).sum()),
        'revised_scorable_sites':int(cells.loc[f>=.8,'site'].nunique()),
        'baseline_known_fraction_median':float(np.median(bf)),'revised_known_fraction_median':float(np.median(f)),
        'newly_known_pairs':int((k&~bk).sum()),'newly_unknown_pairs':int((bk&~k).sum()),
        'added_modeled_native_positives':int(((y>0)&~(by>0)).sum()),'removed_modeled_native_positives':int(((by>0)&~(y>0)).sum()),
        'lexical_parent_candidates_applied':0,'field_positive_genus_constraints_applied':0,
        'new_model_fits':0,'new_GPU_jobs':0,'new_plot_or_bout_outcomes':False,'fresh_confirmation':False,
        'labels_not_yet_scored_by_A2':True}
    save(OUT/'audit.json',result)
    save(OUT/'labels_freeze.json',{'at_utc':now(),'protocol_freeze_sha256':sha(OUT/'protocol_freeze.json'),
        'files':{str(p.relative_to(ROOT)):sha(p) for p in DATA.iterdir() if p.is_file()},'A2_F1_not_evaluated':True})
    print(json.dumps(result),flush=True)


def score(config):
    freeze()
    if (OUT/'score_summary.json').exists():raise ValueError('A2 score already complete')
    lock=json.loads((OUT/'labels_freeze.json').read_text())
    for name,digest in lock['files'].items():assert sha(ROOT/name)==digest
    save(OUT/'score_access.json',{'at_utc':now(),'labels_freeze_sha256':sha(OUT/'labels_freeze.json')})
    cells=pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv')
    labels=np.load(DATA/'revised_labels.npz');baseline=np.load(DATA/'baseline_labels.npz')
    original=np.load(ROOT/'data/confirmation_e4/assessment_labels.npz')
    policy=np.load(ROOT/'data/confirmation_e4/frozen_policy.npz');contract=np.load(ROOT/'data/confirmation_e4/label_contract.npz')
    y,k,f=labels['truth'],labels['known'],labels['known_fraction']
    before=score_fixed_sets(baseline['truth'],baseline['known'],policy['rank'],policy['count'])['F1']
    scored=score_fixed_sets(y,k,policy['rank'],policy['count']);q=scored['F1']
    lower,upper=fixed_set_f1_bounds(y,k,contract['neon_eligible'],policy['rank'],policy['count'])
    cohorts={'original_65':original['known_fraction']>=.8,'previous_morphospecies_77':baseline['known_fraction']>=.8,
             'all_163':np.ones(len(cells),bool),'revised_80pct_known':f>=config['minimum_known_fraction']}
    assert cohorts['original_65'].sum()==65 and cohorts['previous_morphospecies_77'].sum()==77
    records=[];cohort_results={}
    for name,take in cohorts.items():
        if not take.any():continue
        meta=cells.loc[take];groups=(meta.site.astype(str)+'|'+meta.area_group.astype(str)).to_numpy()
        selector=QuotaSelection(meta.cell_id.to_numpy(),config['coverage'],groups)
        risks={'HGB':1-policy['HGB_quality'][take],'SDC':1-policy['SDC'][take],
               'mean_confidence':1-policy['mean_confidence'][take],'entropy':policy['entropy'][take],'random':policy['random'][take]}
        chosen={key:selector.accept(value) for key,value in risks.items()}
        for mask in chosen.values():
            for indices,quota in selector.groups:assert mask[indices].sum()==quota
        a,b=chosen['HGB'],chosen['SDC']
        comparison=site_primary_comparison(q[take],policy['HGB_quality'][take],float(policy['mean_quality'][0]),a,b,
            meta.site,**{key:value for key,value in config['bootstrap'].items() if key!='cluster'})
        weights=a/a.sum()-b/b.sum()
        result={'cells':int(take.sum()),'sites':int(meta.site.nunique()),'known_F1':float(q[take].mean()),
            'previous_contract_known_F1_same_cells':float(before[take].mean()),
            'known_score_change_same_cells':float((q-before)[take].mean()),
            'mean_known_fraction':float(f[take].mean()),'mean_F1_identification_width':float((upper-lower)[take].mean()),
            'mean_F1_identification_range':[float(lower[take].mean()),float(upper[take].mean())],
            'HGB_minus_SDC_identification_range':linear_contrast_bounds(lower[take],upper[take],weights),
            'descriptive_comparison':comparison,
            'quality_MSE':{key:float(np.mean((policy[column][take]-q[take])**2)) for key,column in
                [('training_mean','mean_quality'),('HGB','HGB_quality'),('affine_SDC','affine_SDC_quality'),('raw_SDC','SDC')]},
            'selective_F1':{key:float(q[take][mask].mean()) for key,mask in chosen.items()},
            'previous_contract_selective_F1_same_cells_and_acceptance':{key:float(before[take][mask].mean()) for key,mask in chosen.items()},
            'meets_historical_100cell_12site_reference':bool(take.sum()>=100 and meta.site.nunique()>=12)}
        cohort_results[name]=result
        for j,index in enumerate(np.flatnonzero(take)):
            records.append({'cohort':name,'cell_id':cells.iloc[index].cell_id,'site':cells.iloc[index].site,
                'F1':q[index],'previous_F1':before[index],'lower':lower[index],'upper':upper[index],
                **{'accept_'+key:bool(mask[j]) for key,mask in chosen.items()}})
    pd.DataFrame(records).to_csv(OUT/'cohort_cells.csv',index=False)
    pd.DataFrame({'cell_id':cells.cell_id,'site':cells.site,'known_fraction':f,'previous_known_fraction':baseline['known_fraction'],
                  'F1':q,'previous_F1':before,'lower':lower,'upper':upper}).to_csv(OUT/'cell_results.csv',index=False)
    save(OUT/'score_summary.json',{'status':'corrected_contract_development_replay_complete','at_utc':now(),
        'cohorts':cohort_results,'new_model_fits':0,'new_GPU_jobs':0,'fresh_confirmation':False,
        'original_E4_primary_changed':False,'new_method_success_established':False,
        'interpretation':'known-label scores change their observed-label estimand; no complete-field oracle or rescued E4 confirmation'})
    print(json.dumps({'status':'scoring_complete','cohorts':cohort_results}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','audit','score']);args=parser.parse_args()
    config=json.loads((ROOT/'configs/neon_taxonomy_a2.json').read_text())
    if args.phase=='freeze':print(json.dumps(freeze()))
    elif args.phase=='audit':audit(config)
    else:score(config)


if __name__=='__main__':main()
