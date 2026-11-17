"""Independent E4 provenance, scoring and fixed-policy validation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT,DATA=ROOT/'results/confirmation_e4',ROOT/'data/confirmation_e4'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check_manifest(base,root_relative=False):
    m=json.loads((base/'artifact_manifest.json').read_text())
    for name,item in m['files'].items():
        p=(ROOT if root_relative else base)/name
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    return len(m['files'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    earlier={name:check_manifest(ROOT/'results'/name) for name in ['external_replication_e1','ranking_study_e2','confirmation_e3','neon_confirmation_feasibility']}
    earlier['neon_bridge']=check_manifest(ROOT/'results/neon_bridge',True)
    lock=json.loads((OUT/'protocol_freeze.json').read_text())
    for group,prefix in [('sources',ROOT),('inputs',DATA),('external_inputs',ROOT)]:
        for name,digest in lock['identity'][group].items():assert sha(prefix/name)==digest,name
    policy_lock=json.loads((OUT/'policy_freeze.json').read_text());access=json.loads((OUT/'assessment_access.json').read_text())
    assert datetime.fromisoformat(lock['locked_at_utc'])<datetime.fromisoformat(policy_lock['at_utc'])<datetime.fromisoformat(access['first_request_at_utc'])
    assert access['policy_freeze_sha256']==sha(OUT/'policy_freeze.json') and policy_lock['protocol_freeze_sha256']==sha(OUT/'protocol_freeze.json')
    assert policy_lock['policy_sha256']==sha(DATA/'frozen_policy.npz') and policy_lock['fit_parameters_sha256']==sha(OUT/'fit_parameters.json')
    assert not access['unselected_plot_or_bout_taxon_values_decoded']
    p=np.load(DATA/'frozen_policy.npz');y=np.load(DATA/'assessment_labels.npz');c=np.load(DATA/'label_contract.npz')
    rows=pd.read_csv(OUT/'cell_results.csv');config=json.loads((ROOT/'configs/confirmation_e4.json').read_text())
    assert y['truth'].shape==(163,5771) and rows.cell_id.nunique()==163 and rows.site.nunique()==16
    assert np.isin(y['truth'],[-1,0,1]).all() and not np.any(y['known']&~c['neon_eligible'])
    assert np.array_equal(y['truth']>=0,y['known'])
    pred=np.zeros_like(y['known'])
    for i,count in enumerate(p['count']):pred[i,p['rank'][i,:count]]=True
    assert (pred.sum(1)==p['count']).all() and not np.any(pred&~c['neon_eligible'])
    hits=(pred&(y['truth']>0)).sum(1);npred=(pred&y['known']).sum(1);ntruth=(y['truth']>0).sum(1)
    den=npred+ntruth;expected=np.divide(2*hits,den,out=np.ones(len(rows)),where=den>0)
    np.testing.assert_allclose(rows.F1,expected,rtol=0,atol=1e-12)
    fraction=y['known'].sum(1)/c['neon_eligible'].sum(1)
    np.testing.assert_allclose(rows.known_fraction,fraction,rtol=0,atol=1e-12)
    valid=(fraction>=config['minimum_known_label_fraction'])&y['known'].any(1)
    assert np.array_equal(rows.scorable,valid) and valid.sum()==65 and rows.loc[valid,'site'].nunique()==10
    groups=rows.loc[valid].groupby(['site','area_group'])
    for _,f in groups:
        budgets=[int(f['accept_'+method].sum()) for method in ['HGB','SDC','mean_confidence','entropy','random']]
        assert len(set(budgets))==1
    primary=json.loads((OUT/'primary_result.json').read_text());q=expected[valid]
    assert not primary['scientific_admission'] and not primary['joint_primary_success']
    for name,key in [('training_mean','mean_quality'),('HGB','HGB_quality'),('affine_SDC','affine_SDC_quality'),('raw_SDC','SDC')]:
        assert np.isclose(np.mean((p[key][valid]-q)**2),primary['quality_MSE'][name],atol=1e-12)
    for method in ['HGB','SDC','mean_confidence','entropy','random']:
        assert np.isclose(rows.loc[valid&rows['accept_'+method],'F1'].mean(),primary['selective_F1'][method],atol=1e-12)
    diagnostic=json.loads((OUT/'posthoc_taxonomy_attrition.json').read_text())
    assert diagnostic['same_assessment_labels_reproduced_bit_for_bit'] and not diagnostic['new_plots_or_bouts_exposed']
    joined=json.loads((OUT/'posthoc_morphospecies/summary.json').read_text())
    assert joined['original_primary_sha256']==sha(OUT/'primary_result.json') and not joined['fresh_confirmation_claim']
    assert joined['scorable_cells']==77 and joined['scorable_sites']==11
    assert joined['scorable_cells']<config['minimum_scored_cells'] and joined['scorable_sites']<config['minimum_scored_sites']
    gpu=json.loads((OUT/'GPU_validation.json').read_text());assert gpu['status']=='passed'
    state=json.loads((OUT/'kaggle_state.json').read_text());assert state['private'] and not state['competition_submission_made']
    for item in state['output_files']:
        assert sha(ROOT/'artifacts/neon_e4_gpu'/item['name'])==item['sha256']
    xml=ET.parse(OUT/'validation_focused_tests.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in xml)==34
    assert all(int(s.attrib.get('failures',0))==int(s.attrib.get('errors',0))==0 for s in xml)
    key=json.loads((ROOT.parent/'kaggle_api_key/kaggle.json').read_text())['key'];token=(ROOT.parent/'neon_api_key/neon_api_key').read_text().strip()
    paths=[ROOT/'notebooks/neon_e4_inference.ipynb']
    paths+=[p for base in [OUT,ROOT/'results/neon_input_followup',ROOT/'docs',ROOT/'scripts',ROOT/'src',ROOT/'configs'] for p in base.rglob('*') if p.is_file() and p.suffix in ['.json','.py','.md','.xml']]
    for path in paths:
        content=path.read_bytes();assert key.encode() not in content and token.encode() not in content,'Credential found'
    result={'status':'validated','earlier_manifests_unchanged':earlier,'focused_tests_after_diagnostics':34,
            'protocol_and_policy_before_target_access':True,'independent_confusion_matrix_scoring_matches':True,
            'all_five_selectors_have_identical_group_budgets':True,'assessment_reconstructed_bit_for_bit':True,
            'original_primary_admission_and_success':False,'original_scorable_cells_sites':[65,10],
            'posthoc_documented_morphospecies_join_cells_sites':[77,11],'posthoc_replay_is_not_fresh_confirmation':True,
            'GPU_reference_and_export_hashes_verified':True,'credential_scan_passed':True,'new_SDM_training_fits':0,
            'NEON_assessment_cells_now_consumed':163,'scientific_scope':'one fixed model, training-support stratum; partially observed modeled-native-set F1; failed admission'}
    manifest_path=OUT/'artifact_manifest.json'
    if args.freeze:
        if manifest_path.exists():raise ValueError('Refusing to replace E4 artifact manifest')
        (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
        files=[p for base in [OUT,DATA,ROOT/'data/neon_input_followup',ROOT/'results/neon_input_followup',ROOT/'artifacts/neon_e4_gpu'] for p in base.rglob('*') if p.is_file()]
        files += [ROOT/n for n in ['configs/neon_input_followup.json','configs/confirmation_e4.json','docs/confirmation_e4_results.md','docs/taxonomic_resolution_study_plan.md','scripts/validate_neon_e4.py','scripts/diagnose_neon_e4.py','scripts/audit_neon_e4_taxa.py','scripts/replay_neon_morphospecies.py','src/geolifeclef/native_region_bridge.py','src/geolifeclef/neon_assessment.py','src/geolifeclef/partial_set_bounds.py','src/geolifeclef/neon_morphospecies.py','scripts/build_neon_native_masks.py','scripts/extend_neon_native_masks.py','scripts/retrieve_exact_neon_inputs.py']]
        files += [ROOT/'tests'/n for n in ['test_native_region_bridge.py','test_neon_assessment.py','test_partial_set_bounds.py','test_neon_morphospecies.py']]
        content={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))}
        manifest_path.write_text(json.dumps({'scope':'E4 sources/inputs/results plus followup engineering; scientific source lock is separate and predates targets','files':content},indent=2)+'\n')
    if manifest_path.exists():result['new_manifest_files_verified']=check_manifest(OUT,True)
    print(json.dumps(result))


if __name__=='__main__':main()
