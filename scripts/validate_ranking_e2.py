"""Audit E2 geographic/source exclusions and unchanged E1 artifacts."""
from datetime import datetime,timezone
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from study_ranking_e2 import (ROOT,OUT,SOURCE,DATA,calibration_only,fingerprint_ids,save,sha)
from geolifeclef.ranking_selection import buffered_training,spatial_splits


def assert_buffer(meta,train,val):
    assert not set(train)&set(val)
    d=BallTree(np.deg2rad(meta.iloc[val][['lat','lon']]),metric='haversine').query(
        np.deg2rad(meta.iloc[train][['lat','lon']]))[0].min()*6371.0088
    assert d>=50
    return float(d)


def main():
    config=json.loads((ROOT/'configs/ranking_study_e2.json').read_text())
    lock=json.loads((OUT/'development_lock.json').read_text())
    for name,digest in lock['source_hashes'].items():
        assert sha(ROOT/name)==sha(OUT/'source_snapshot'/name)==digest
    assert sha(SOURCE/'cohort_roles.csv')==lock['E1_cohort_sha256']
    for name,digest in lock['E1_input_hashes'].items():assert sha(SOURCE/name)==digest
    for name,item in json.loads((ROOT/'results/external_replication_e1/artifact_manifest.json').read_text())['files'].items():
        assert sha(ROOT/'results/external_replication_e1'/name)==item['sha256']
    meta,_=calibration_only(pd.read_csv(SOURCE/'cohort_roles.csv',dtype={'area_bin':str}))
    metadata=pd.read_csv(ROOT/'data/independent_data_audit/sage_eval_plot_metadata.csv',
        usecols=['cell_row','GIVD_ID']).drop_duplicates()
    source=metadata.groupby('cell_row').GIVD_ID.agg(lambda x:'|'.join(sorted(x.dropna().unique())))
    meta['source']=source.reindex(meta.cell_id).to_numpy()
    risk=np.flatnonzero(meta.role.eq('risk_calibration'));cal=np.flatnonzero(meta.role.eq('probability_calibration'))
    rm=meta.iloc[risk].reset_index(drop=True)
    report=json.loads((OUT/'report.json').read_text())
    spatial=spatial_splits(rm,5,50);buffer_checks=[]
    for audit in report['splits']:
        if audit['study']=='spatial':
            train,val=spatial[audit['fold']]
        else:
            val=np.flatnonzero(rm.source.eq(audit['fold']))
            train=buffered_training(rm,np.flatnonzero(~rm.source.eq(audit['fold'])),val,50)
            allowed=cal[~meta.iloc[cal].source.eq(audit['fold']).to_numpy()]
            allowed=buffered_training(meta,allowed,risk[val],50)
            assert not meta.iloc[allowed].source.eq(audit['fold']).any()
            assert not rm.iloc[train].source.eq(audit['fold']).any()
            assert fingerprint_ids(meta.iloc[allowed].cell_id)==audit['probability_calibration_ids_sha256']
            buffer_checks.append(assert_buffer(meta,allowed,risk[val]))
        assert fingerprint_ids(rm.iloc[train].cell_id)==audit['training_ids_sha256']
        assert fingerprint_ids(rm.iloc[val].cell_id)==audit['validation_ids_sha256']
        buffer_checks.append(assert_buffer(rm,train,val))
        inner=spatial_splits(rm.iloc[train].reset_index(drop=True),3,50)
        for (tr,va),row in zip(inner,audit['inner']['audits']):
            assert fingerprint_ids(rm.iloc[train[tr]].cell_id)==row['train_ids_sha256']
            assert fingerprint_ids(rm.iloc[train[va]].cell_id)==row['validation_ids_sha256']
            buffer_checks.append(assert_buffer(rm,train[tr],train[va]))
    frame=pd.read_csv(OUT/'heldout_development_rows.csv')
    assert not set(frame.cell_id)-set(rm.cell_id)
    counts=frame.groupby(['study','fold','model','method']).accepted.sum().unstack('method')
    assert counts.eq(counts.SDC,axis=0).all().all()
    assert report['E1_assessment_targets_read_in_E2'] is False
    assert report['gpu_scientifically_admitted']==all(report['gpu_checks'].values())
    tests=ET.parse(OUT/'focused_tests.xml').getroot().find('testsuite').attrib
    assert tests['failures']==tests['errors']=='0'
    save(OUT/'validation.json',{'validated_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','E1_artifact_manifest_unchanged':True,'locked_sources_and_inputs_match':True,
        'outer_spatial_folds':5,'outer_source_folds':3,'inner_folds':24,
        'source_removed_from_both_reliability_calibration_roles':True,
        'minimum_audited_crossfold_distance_km':min(buffer_checks),
        'buffer_checks':len(buffer_checks),'only_original_riskcal_rows_evaluated':True,
        'assessment_targets_requested':0,'all_policy_counts_equal':True,
        'tests':{k:tests[k] for k in ['tests','failures','errors','skipped']},
        'scientific_GPU_admission':report['gpu_scientifically_admitted'],'GPU_job_launched':False,
        'helper_source_hashes':{n:sha(ROOT/n) for n in ['scripts/run_external_e1.py','src/geolifeclef/reliability.py']}})
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json')
    save(OUT/'artifact_manifest.json',{'files':{str(p.relative_to(OUT)):{'bytes':p.stat().st_size,
        'sha256':sha(p)} for p in files}})
    print(json.dumps({'status':'passed','tests':tests['tests'],'buffer_checks':len(buffer_checks),
        'minimum_distance_km':min(buffer_checks),'GPU_launched':False}))


if __name__=='__main__':main()
