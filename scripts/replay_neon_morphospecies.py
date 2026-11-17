"""Post-hoc documented morphospecies join; original E4 failure stays immutable."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geolifeclef.neon_assessment import selected_csv_rows,TaxonomicAssessment,score_fixed_sets,site_primary_comparison
from geolifeclef.neon_morphospecies import selected_morpho_rows,latest_resolution
from geolifeclef.ecological_selection import QuotaSelection


def main():
    data,base=ROOT/'data/confirmation_e4',ROOT/'results/confirmation_e4';out=base/'posthoc_morphospecies';out.mkdir(exist_ok=True)
    original_hash=hashlib.sha256((base/'primary_result.json').read_bytes()).hexdigest()
    plots=pd.read_csv(data/'selected_plots.csv');cells=pd.read_csv(data/'neon_assessment_roles.csv')
    selected=dict(zip(plots.plot_id,plots.bout.astype(str)));cell_for_plot=dict(zip(plots.plot_id,plots.cell_id));row_for_cell={k:i for i,k in enumerate(cells.cell_id)}
    manifest=json.loads((ROOT/'data/neon_bridge/neon_2023_manifest.json').read_text())['data']
    receipt=json.loads((base/'target_mapping_audit.json').read_text());wanted={f['name'] for f in receipt['MD5_receipts']}
    obs=[f for r in manifest['releases'] for p in r['packages'] for f in p['files'] if f['name'] in wanted]
    def fetch(f):
        with urllib.request.urlopen(f['url'],timeout=40) as response:body=response.read(f['size']+1)
        assert len(body)==f['size'] and hashlib.md5(body).hexdigest()==f['md5']
        return f,body
    observed=[];keys=set()
    allowed=['plotID','siteID','boutNumber','morphospeciesID','taxonID','targetTaxaPresent','divDataType','identificationQualifier','dataQF']
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f,body in pool.map(fetch,obs):
            table='div_1m2Data' if '.div_1m2Data.' in f['name'] else 'div_10m2Data100m2Data'
            for row in selected_csv_rows(body,selected,allowed):
                observed.append((table,row))
                if row.get('morphospeciesID',''):keys.add((row['siteID'],row['morphospeciesID']))
    expanded=json.loads((data/'posthoc_expanded_manifest.json').read_text())['data']
    morpho={f['url']:f for r in expanded['releases'] for p in r['packages'] for f in p['files'] if '.div_morphospecies.' in f['name']}
    matches=defaultdict(list)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _,body in pool.map(fetch,morpho.values()):
            for row in selected_morpho_rows(body,keys):matches[(row['siteID'],row['morphospeciesID'])].append(row)
    resolutions={};statuses=Counter()
    for key in keys:
        row,status=latest_resolution(matches.get(key,[]));statuses[status]+=1
        if row is not None:resolutions[key]=row
    taxonomy=json.loads((ROOT/'results/neon_bridge/taxonomy_retrieval.json').read_text())
    taxa={r['taxonID']:r for f in taxonomy['files'] for r in json.loads((ROOT/f['file']).read_text())['data']}
    table=pd.read_csv(ROOT/'data/neon_bridge/neon_to_sage_taxonomy.csv')
    mapping={r.neon_taxon_id:int(r.sage_index) for r in table[table.sage_index.notna()].itertuples()}
    names=pd.read_csv(ROOT/'data/neon_bridge/species_names.csv')['Species Name'].tolist();contract=np.load(data/'label_contract.npz')
    reducer=TaxonomicAssessment(contract['neon_eligible'],taxa,mapping,names)
    changed=conflicts=0
    for table,row in observed:
        index=row_for_cell[cell_for_plot[row['plotID']]];new=dict(row)
        resolution=resolutions.get((row['siteID'],row.get('morphospeciesID','')))
        if resolution is not None:
            old_id,new_id=row.get('taxonID',''),resolution.get('taxonID','')
            if old_id in mapping and new_id in mapping and mapping[old_id]!=mapping[new_id]:
                conflicts+=1
                for code in [old_id,new_id]:reducer.uncertain[index,reducer.compatible(taxa.get(code))]=True
                continue
            for field in ['taxonID','identificationQualifier','dataQF']:new[field]=resolution.get(field,'')
            changed+=int(any(new.get(k,'')!=row.get(k,'') for k in ['taxonID','identificationQualifier','dataQF']))
        reducer.add(index,new,table)
    truth,known,fraction,counts=reducer.finish();policy=np.load(data/'frozen_policy.npz')
    scored=score_fixed_sets(truth,known,policy['rank'],policy['count']);valid=(fraction>=.8)&known.any(1)
    result={'status':'posthoc_replay_complete','original_primary_sha256':original_hash,'original_primary_changed':False,
            'fresh_confirmation_claim':False,'new_model_or_policy_fit':False,
            'selected_morphospecies_keys':len(keys),'resolution_status_counts':dict(statuses),'observation_records_changed_by_join':changed,
            'conflicting_resolved_species_records_left_unknown':conflicts,'known_fraction_median':float(np.median(fraction)),
            'scorable_cells':int(valid.sum()),'scorable_sites':int(cells.loc[valid,'site'].nunique()),'record_counts':counts,
            'source':'Official NEON quick-start join by siteID,morphospeciesID; latest dated resolution, conflicts not picked for favorable mapping.',
            'qualification':'Recommended join was omitted in the conservative first pass. This replay corrects that limitation but is post-hoc and cannot rescue original confirmation or establish intrinsic taxonomic unobservability.'}
    if valid.any():
        meta=cells.loc[valid];groups=(meta.site+'|'+meta.area_group).to_numpy();selector=QuotaSelection(meta.cell_id.to_numpy(),.8,groups)
        a=selector.accept(1-policy['HGB_quality'][valid]);b=selector.accept(1-policy['SDC'][valid]);q=scored['F1'][valid]
        result['descriptive_comparison']=site_primary_comparison(q,policy['HGB_quality'][valid],float(policy['mean_quality'][0]),a,b,meta.site)
        result['quality_MSE']={name:float(np.mean((policy[key][valid]-q)**2)) for name,key in [('training_mean','mean_quality'),('HGB','HGB_quality'),('affine_SDC','affine_SDC_quality'),('raw_SDC','SDC')]}
    frame=cells[['cell_id','site']].copy();frame['known_fraction']=fraction;frame['scorable']=valid
    for k,v in scored.items():frame[k]=v
    frame.to_csv(out/'cell_results.csv',index=False)
    np.savez_compressed(data/'posthoc_morphospecies_labels.npz',truth=truth,known=known,known_fraction=fraction)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    assert hashlib.sha256((base/'primary_result.json').read_bytes()).hexdigest()==original_hash
    print(json.dumps(result))


if __name__=='__main__':main()
