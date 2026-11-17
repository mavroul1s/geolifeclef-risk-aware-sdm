"""Verify E1 locked sources, policies, budgets, provenance and repair invariance."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from run_external_e1 import DATA, OUT, ROOT, METHODS, freeze, save, sha, validate_metadata


def main():
    freeze()
    admission = json.loads((OUT/'evaluation_admission.json').read_text())
    access = json.loads((OUT/'assessment_access.json').read_text())
    assert admission['protocol_sha256'] == sha(OUT/'protocol_freeze.json')
    assert access['admission_sha256'] == sha(OUT/'evaluation_admission.json')
    assert admission['admitted_at_utc'] < access['first_assessment_target_access_utc']
    assert admission['secondary_rare_mask_sha256'] == sha(DATA/'secondary_rare_mask.npy')
    cohort = pd.read_csv(DATA/'cohort_roles.csv', dtype={'area_bin':str})
    metadata = validate_metadata(cohort)
    original = json.loads((OUT/'protocol_freeze.json').read_text())
    for name, digest in original['source_hashes'].items():
        assert sha(OUT/'source_snapshot'/name) == digest
    target_record = json.loads((DATA/'zenodo_21335338.json').read_text())
    target_file = next(f for f in target_record['files'] if f['key'] == 'eval_test_targets.h5')
    with (DATA/'sealed_eval_test_targets.h5').open('rb') as stream:
        assert 'md5:'+hashlib.file_digest(stream,'md5').hexdigest() == target_file['checksum']
    prediction_receipt = json.loads((OUT/'retrieval_predictions.json').read_text())
    for item in prediction_receipt['files']:
        assert sha(ROOT/item['file']) == item['sha256']
    assessment = cohort[cohort.role.eq('assessment')].reset_index(drop=True)
    groups = assessment.country+'|'+assessment.area_bin
    config = json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    verified_policies = 0
    for model,checks in admission['policy_hashes'].items():
        for extension, digest in checks.items():
            assert sha(OUT/'policies'/f'{model}.{extension}') == digest
        with np.load(OUT/'policies'/f'{model}.npz',allow_pickle=False) as p:
            assert np.array_equal(p['cell_id'],assessment.cell_id)
            for decoder in config['decoders']:
                for coverage in config['coverage_grid']:
                    keys = p['keys'].tolist()
                    base = p['accept'][keys.index(f'{decoder}|{coverage}|SDC')]
                    for method in METHODS:
                        accepted = p['accept'][keys.index(f'{decoder}|{coverage}|{method}')]
                        assert int(accepted.sum()) == round(coverage*len(assessment))
                        for group in groups.unique():
                            ix = groups.eq(group).to_numpy()
                            assert accepted[ix].sum() == base[ix].sum()
                        verified_policies += 1
    amendment = json.loads((OUT/'serialization_amendment.json').read_text())
    invariance = {}
    for name,key in [('summary.csv','pre_repair_summary_sha256'),
                     ('regional_summary.csv','pre_repair_regional_summary_sha256')]:
        invariance[name] = sha(OUT/name) == amendment[key]
        assert invariance[name]
    # Additional geographic integrity check on the final cell centroids.
    reference = pd.concat([pd.read_csv(ROOT/'data/raw'/name,usecols=['lat','lon'])
        for name in ['GLC25_PA_metadata_train.csv','GLC25_PA_metadata_test.csv']])
    distance = BallTree(np.deg2rad(reference[['lat','lon']]),metric='haversine').query(
        np.deg2rad(cohort[['lat','lon']]))[0].min()*6371.0088
    assert distance >= 500
    tests = ET.parse(OUT/'focused_tests.xml').getroot().find('testsuite').attrib
    assert tests['errors'] == '0' and tests['failures'] == '0'
    table = pd.read_csv(OUT/'summary.csv')
    assert len(table) == verified_policies == 512
    report = json.loads((OUT/'report.json').read_text())
    assert report['primary_success'] == all(report['primary_checks'].values())
    versions = {name: importlib.metadata.version(name) for name in
        ['numpy','pandas','scipy','scikit-learn','h5py','hdf5plugin','matplotlib','pytest']}
    save(OUT/'validation.json', {'validated_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','source_snapshot_and_explicit_serialization_amendment_verified':True,
        'policies_frozen_before_assessment_target_access':True,'verified_policy_settings':verified_policies,
        'all_group_budgets_equal':True,'target_published_MD5_matches':True,
        'prediction_local_hashes_match_retrieval':True,'whole_prediction_TAR_MD5_recomputed':False,
        'serialization_repair_summary_invariance':invariance,'metadata_validation':metadata,
        'minimum_final_centroid_distance_existing_GLC_PA_km':float(distance),
        'focused_tests':{k:tests[k] for k in ['tests','errors','failures','skipped']},
        'python':platform.python_version(),'package_versions':versions,
        'primary_success':report['primary_success'],'new_neural_fits':0,
        'scope':'technical validation does not convert a failed scientific hypothesis into success'})
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    save(OUT/'artifact_manifest.json', {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':{str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}})
    print(json.dumps({'status':'passed','policy_settings':verified_policies,'tests':tests['tests'],
        'minimum_centroid_distance_km':float(distance)}))


if __name__ == '__main__':
    main()
