"""Post-audit component ablation; never selects a preferred contract by F1."""
from collections import defaultdict
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from run_neon_taxonomy_a2 import DATA,OUT,cached,metadata,save,sha,now
from geolifeclef.neon_assessment import TaxonomicAssessment,score_fixed_sets
from geolifeclef.neon_morphospecies import latest_resolution
from geolifeclef.neon_record_contract import AcceptedConceptAssessment,current_resolution,combine_current_record,rank_matches
from geolifeclef.taxonomy_contract import accepted_concept
from geolifeclef.taxonomy_bridge import canonical_name,normalized_name


def main():
    target=OUT/'posthoc';target.mkdir(exist_ok=True)
    if (target/'diagnosis.json').exists():raise ValueError('A2 diagnosis already complete')
    assert json.loads((OUT/'validation.json').read_text())['passed']
    original={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'audit.json',OUT/'score_summary.json',DATA/'revised_labels.npz',DATA/'baseline_labels.npz']}
    save(target/'analysis_lock.json',{'at_utc':now(),'explicitly_posthoc':True,'source_sha256':sha(__file__),'primary_identities':original})
    dest=target/'source_snapshot/scripts/diagnose_neon_taxonomy_a2.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,dest)
    observed=cached('selected_observations.json.gz');matches={tuple(key):rows for key,rows in cached('selected_morphospecies.json.gz')}
    resolutions={key:current_resolution(rows) for key,rows in matches.items()}
    taxa,old,names,decisions,new_mapping=metadata()
    cells=pd.read_csv(ROOT/'data/confirmation_e4/neon_assessment_roles.csv');plots=pd.read_csv(ROOT/'data/confirmation_e4/selected_plots.csv')
    cell_for_plot=dict(zip(plots.plot_id,plots.cell_id));row_for_cell={v:i for i,v in enumerate(cells.cell_id)}
    eligible=np.load(ROOT/'data/confirmation_e4/label_contract.npz')['neon_eligible']
    baseline=np.load(DATA/'baseline_labels.npz');revised=np.load(DATA/'revised_labels.npz')
    p=np.load(ROOT/'data/confirmation_e4/frozen_policy.npz');fixed77=baseline['known_fraction']>=.8
    rows=[];all_labels={}
    for strict,masks,expanded in itertools.product([False,True],repeat=3):
        reducer=(AcceptedConceptAssessment if masks else TaxonomicAssessment)(eligible,taxa,new_mapping if expanded else old,names)
        for table,row in observed:
            if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
            cell=row_for_cell[cell_for_plot[row['plotID']]];key=(row['siteID'],row.get('morphospeciesID',''))
            if strict:
                resolution,status,conflicts=resolutions.get(key,(None,'no_dated_resolution',[]))
                chosen,reason,evidence=combine_current_record(row,resolution,status,conflicts,taxa)
                if evidence:
                    for item in evidence:reducer.uncertain[cell,reducer.compatible(taxa.get(item.get('taxonID')))]=True
                    continue
                if not rank_matches(row,taxa) or not rank_matches(chosen,taxa):
                    reducer.uncertain[cell,reducer.compatible(taxa.get(chosen.get('taxonID')))]=True;continue
            else:
                chosen=dict(row);resolution,_=latest_resolution(matches.get(key,[]))
                if resolution is not None:
                    a,b=row.get('taxonID',''),resolution.get('taxonID','')
                    if a in old and b in old and old[a]!=old[b]:
                        for code in [a,b]:reducer.uncertain[cell,reducer.compatible(taxa.get(code))]=True
                        continue
                    for field in ['taxonID','identificationQualifier','dataQF']:chosen[field]=resolution.get(field,'')
            reducer.add(cell,chosen,table)
        y,k,f,_=reducer.finish();valid=f>=.8;q=score_fixed_sets(y,k,p['rank'],p['count'])['F1']
        key=f'{int(strict)}{int(masks)}{int(expanded)}';all_labels[key]=(y,k,f)
        rows.append({'strict_current_record_rules':strict,'full_chain_and_fallback_masks':masks,'expanded_accepted_mapping':expanded,
            'scorable_cells':int(valid.sum()),'scorable_sites':int(cells.loc[valid,'site'].nunique()),
            'known_fraction_median':float(np.median(f)),'known_F1_on_fixed77':float(q[fixed77].mean()),
            'known_F1_all163':float(q.mean()),'positive_pairs':int((y>0).sum()),
            'new_unknown_pairs_vs_baseline':int((baseline['known']&~k).sum())})
        print(json.dumps({'component_arm':key,'scorable_cells':int(valid.sum())}),flush=True)
    for key,source in [('000',baseline),('111',revised)]:
        for value,name in zip(all_labels[key],['truth','known','known_fraction']):np.testing.assert_array_equal(value,source[name])
    # Report how masks change the scoring denominator on unchanged predictions.
    pred=np.zeros_like(eligible)
    for i in range(len(cells)):pred[i,p['rank'][i,:p['count'][i]]]=True
    removed=baseline['known']&~revised['known'];assert not np.any(removed&(baseline['truth']>0))
    q0=score_fixed_sets(baseline['truth'],baseline['known'],p['rank'],p['count'])['F1']
    q1=score_fixed_sets(revised['truth'],revised['known'],p['rank'],p['count'])['F1']
    assert np.all(q1>=q0-1e-12)
    # Metadata-only next-step screen: reverse accepted-name equivalence classes.
    # This does not add labels or scores, and ambiguous model-class groups remain candidates.
    sage={}
    for i,name in enumerate(names):sage.setdefault(normalized_name(name),[]).append(i)
    classes=defaultdict(set)
    for code,taxon in taxa.items():
        if taxon.get('dwc:taxonRank')!='species':continue
        terminal,_=accepted_concept(code,taxa)
        if terminal is not None and terminal.get('dwc:taxonRank')=='species':classes[terminal['taxonID']].update(sage.get(canonical_name(taxon),[]))
    possible=set();ambiguous=set();unmatched_records=possible_records=ambiguous_records=0
    for table,row in observed:
        if table=='div_1m2Data' and row.get('divDataType','').casefold()!='plantspecies':continue
        resolution,status,conflicts=resolutions.get((row['siteID'],row.get('morphospeciesID','')),(None,'no_dated_resolution',[]))
        chosen,reason,evidence=combine_current_record(row,resolution,status,conflicts,taxa)
        if evidence or not rank_matches(row,taxa) or not rank_matches(chosen,taxa):continue
        if chosen.get('identificationQualifier','').strip().casefold() not in {'','na','none','no qualifier'} or chosen.get('dataQF','').strip() not in {'','0','0.0'}:continue
        code=chosen.get('taxonID','')
        if code not in taxa or code in new_mapping:continue
        terminal,_=accepted_concept(code,taxa)
        if terminal is None or terminal.get('dwc:taxonRank')!='species':continue
        unmatched_records+=1;candidates=classes.get(terminal['taxonID'],set())
        cell=row_for_cell[cell_for_plot[row['plotID']]]
        if len(candidates)==1:
            possible_records+=1;possible.add((cell,terminal['taxonID']))
        elif len(candidates)>1:
            ambiguous_records+=1;ambiguous.add((cell,terminal['taxonID']))
    pd.DataFrame(rows).to_csv(target/'component_ablation.csv',index=False)
    result={'at_utc':now(),'posthoc':True,'all_eight_components_reported_no_contract_chosen_by_F1':True,
        'original_and_A2_label_endpoints_reproduced':True,'newly_unknown_pairs':int(removed.sum()),
        'newly_unscored_previously_negative_predictions':int((removed&pred).sum()),
        'newly_unscored_negative_predictions_on_fixed77':int((removed&pred)[fixed77].sum()),
        'positive_labels_and_prediction_sets_unchanged':bool(np.array_equal(baseline['truth']>0,revised['truth']>0)),
        'cells_with_strict_known_F1_increase':int((q1>q0+1e-12).sum()),
        'mean_F1_change_fixed77':float((q1-q0)[fixed77].mean()),
        'reverse_alias_metadata_screen':{'unmatched_current_species_records_with_terminal_species':unmatched_records,
            'records_with_one_model_class_alias_candidate':possible_records,'distinct_cell_concept_candidates':len(possible),
            'records_with_multiple_model_aliases':ambiguous_records,'distinct_ambiguous_cell_concepts':len(ambiguous),
            'candidates_applied_to_labels':0,'scope':'dictionary-equivalence development screen; requires a new conflict-aware contract, not guessed genus parents'}}
    save(target/'diagnosis.json',result)
    for name,digest in original.items():assert sha(ROOT/name)==digest
    plot(cells,baseline,revised,rows,q0,q1)
    print(json.dumps(result),flush=True)


def plot(cells,baseline,revised,arms,q0,q1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.6),constrained_layout=True)
    before,after=baseline['known_fraction'],revised['known_fraction'];lost=(before>=.8)&(after<.8)
    axes[0].scatter(before[~lost],after[~lost],s=18,alpha=.65,color='#245da8',label='Other cells')
    axes[0].scatter(before[lost],after[lost],s=25,alpha=.8,color='#bc493e',label='Lose >=80% eligibility')
    axes[0].plot([0,1],[0,1],color='grey',linewidth=.8);axes[0].axvline(.8,color='black',linestyle='--',linewidth=.8);axes[0].axhline(.8,color='black',linestyle='--',linewidth=.8)
    axes[0].set(xlabel='Previous known-label fraction',ylabel='A2 known-label fraction',title='Observability decreases: 77 → 31 cells',xlim=(-.02,1.02),ylim=(-.02,1.02));axes[0].legend(fontsize=8)
    subsets=[baseline['known_fraction']>=.8,np.ones(len(cells),bool)]
    for i,(mask,name) in enumerate(zip(subsets,['Fixed previous 77','All 163'])):
        axes[1].plot([i-.15,i+.15],[q0[mask].mean(),q1[mask].mean()],color='#777777',linewidth=2)
        axes[1].scatter(i-.15,q0[mask].mean(),color='#245da8',s=55,label='Previous contract' if i==0 else None)
        axes[1].scatter(i+.15,q1[mask].mean(),color='#bc493e',s=55,label='A2 contract' if i==0 else None)
    axes[1].set_xticks([0,1],['Fixed previous 77','All 163']);axes[1].set(ylabel='Mean known-label F1',title='Observed scores rise without new predictions',ylim=(.37,.56));axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('A2 development replay: unchanged positive labels and prediction sets\nDifferent unknown-label masks change the scoring denominator',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(OUT/('observation_contract_a2.'+suffix),dpi=190)
    plt.close(fig)


if __name__=='__main__':main()
