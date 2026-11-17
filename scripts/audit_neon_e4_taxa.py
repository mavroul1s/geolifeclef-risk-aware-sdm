"""Post-primary taxonomy attrition audit on the same consumed plot/bout records."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import urllib.request
import hashlib

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geolifeclef.neon_assessment import selected_csv_rows,TaxonomicAssessment


def main():
    data,out=ROOT/'data/confirmation_e4',ROOT/'results/confirmation_e4'
    audit=json.loads((out/'target_mapping_audit.json').read_text())
    wanted={r['name']:r for r in audit['MD5_receipts']}
    manifest=json.loads((ROOT/'data/neon_bridge/neon_2023_manifest.json').read_text())['data']
    files=[f for release in manifest['releases'] for p in release['packages'] for f in p['files'] if f['name'] in wanted]
    plots=pd.read_csv(data/'selected_plots.csv');cells=pd.read_csv(data/'neon_assessment_roles.csv')
    selected=dict(zip(plots.plot_id,plots.bout.astype(str)));cell_for_plot=dict(zip(plots.plot_id,plots.cell_id))
    row_for_cell={v:i for i,v in enumerate(cells.cell_id)}
    receipt=json.loads((ROOT/'results/neon_bridge/taxonomy_retrieval.json').read_text())
    taxa={r['taxonID']:r for f in receipt['files'] for r in json.loads((ROOT/f['file']).read_text())['data']}
    frame=pd.read_csv(ROOT/'data/neon_bridge/neon_to_sage_taxonomy.csv')
    mapping={r.neon_taxon_id:int(r.sage_index) for r in frame[frame.sage_index.notna()].itertuples()}
    names=pd.read_csv(ROOT/'data/neon_bridge/species_names.csv')['Species Name'].tolist()
    label=np.load(data/'label_contract.npz');eligible=label['neon_eligible']
    reducer=TaxonomicAssessment(eligible,taxa,mapping,names)
    records=Counter();qualifiers=Counter();flags=Counter();ranks=Counter();masks={}
    distinct=Counter();seen=set()
    def fetch(f):
        with urllib.request.urlopen(f['url'],timeout=40) as r:b=r.read(f['size']+1)
        assert len(b)==f['size'] and hashlib.md5(b).hexdigest()==wanted[f['name']]['MD5_verified']
        return f,b
    allowed=['plotID','boutNumber','taxonID','targetTaxaPresent','divDataType','identificationQualifier','dataQF']
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f,b in pool.map(fetch,files):
            table='div_1m2Data' if '.div_1m2Data.' in f['name'] else 'div_10m2Data100m2Data'
            for row in selected_csv_rows(b,selected,allowed):
                cell=row_for_cell[cell_for_plot[row['plotID']]];reducer.add(cell,row,table)
                if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
                code=row.get('taxonID','').strip();present=row.get('targetTaxaPresent','').strip().casefold()
                if not code and present not in {'yes','y','true','1'}:continue
                qualifier=row.get('identificationQualifier','').strip().casefold();flag=row.get('dataQF','').strip()
                qualifiers[qualifier or '(blank)']+=1;flags[flag or '(blank)']+=1
                taxon=taxa.get(code);rank=taxon.get('dwc:taxonRank','missing') if taxon else 'missing';ranks[rank]+=1
                if not code:reason='positive_without_taxon_ID'
                elif qualifier not in {'','na','none','no qualifier'}:reason='identification_qualifier'
                elif flag not in {'','0','0.0'}:reason='quality_flag'
                elif code in mapping:reason='resolved_mapped'
                elif taxon is None:reason='taxon_ID_absent_from_dictionary'
                elif rank in {'subspecies','variety','form','subvariety'}:reason='identified_infraspecific_rank'
                elif rank=='species':reason='unmapped_species_name'
                else:reason='unresolved_higher_rank'
                records[reason]+=1
                key=(cell,code,qualifier,flag)
                if key not in seen:distinct[reason]+=1;seen.add(key)
                if reason!='resolved_mapped':
                    if reason not in masks:masks[reason]=np.zeros_like(eligible)
                    ids=reducer.compatible(taxon);masks[reason][cell,ids]=True
    truth,known,fraction,counts=reducer.finish();original=np.load(data/'assessment_labels.npz')
    np.testing.assert_array_equal(truth,original['truth']);np.testing.assert_array_equal(known,original['known'])
    assert counts==audit['record_counts']
    summaries=[]
    for reason,mask in masks.items():
        missing=mask&eligible&~(truth>0)
        summaries.append({'reason':reason,'records':records[reason],'distinct_cell_taxon_qualifier_keys':distinct[reason],
                          'cells_with_possible_unknown_labels':int(missing.any(1).sum()),
                          'unknown_eligible_label_pairs_potentially_affected':int(missing.sum())})
    pd.DataFrame(summaries).to_csv(out/'posthoc_taxonomy_attrition_reasons.csv',index=False)
    result={'status':'post_primary_taxonomy_diagnostic_complete','same_assessment_labels_reproduced_bit_for_bit':True,
            'record_reason_counts':dict(records),'distinct_cell_taxon_key_reason_counts':dict(distinct),
            'rank_record_counts':dict(ranks),'identification_qualifier_record_counts':dict(qualifiers),'data_quality_flag_record_counts':dict(flags),
            'potential_missingness_by_reason':summaries,'reason_masks_overlap_and_are_not_additive':True,
            'scientific_primary_or_thresholds_changed':False,'new_plots_or_bouts_exposed':False,
            'interpretation':'Attrition belongs to this conservative taxonomic bridge and missing-label rule; it is not proof that NEON itself lacks useful plant identifications. Identified infraspecific taxa need an explicit rank-aware species mapping in future work, not a post-hoc rescue of E4.'}
    (out/'posthoc_taxonomy_attrition.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
