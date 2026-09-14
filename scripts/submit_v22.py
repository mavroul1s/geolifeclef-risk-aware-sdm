"""Independently validate v22, make at most one authorized submission, retrieve scores."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.kaggle_artifacts import KaggleReader,ROOT,COMPETITION
from scripts.submit_v21 import official_client,matched_scores
from scripts.stage_frozen_v20 import sha256_file
from scripts.stage_frozen_v21 import CSV_HASH
from scripts.run_ood_po_expert import validate_submission
from scripts.run_retained_po import gate_from_assessment
from scripts.ood_po_protocol import paired_block_bootstrap
from scripts.v22_protocol import ALPHAS

REQUIRED_INTEGRITY={'all_5016_species','both_folds_fresh','twenty_km_buffer','unchanged_v21_csv_exact',
    'policies_unchanged','submission_unchanged','competition_only','test_labels_unused','no_post_assessment_training','registered_runtime'}


def validate_assessment(report,frame,expected_commit):
    if report.get('experiment')!='v22_retained_po_crossfit' or report.get('status')!='complete' or report.get('source_commit')!=expected_commit:
        raise ValueError('Wrong completed source run')
    if report.get('notebook_tests_before_and_after_passed') is not True: raise ValueError('Notebook tests missing')
    hours=report.get('total_pipeline_hours',float('nan'))
    if not math.isfinite(hours) or not 0<hours<10.5 or report.get('registered_max_total_hours')!=10.5: raise ValueError('Invalid runtime')
    for key in ('external_data_or_weights','test_labels_used','original_v21_retrained_to_recover_outputs'):
        if report.get(key) is not False: raise ValueError('Invalid integrity declaration')
    if report.get('frozen_v21_source_version')!=21: raise ValueError('Wrong frozen baseline')
    integrity=report['integrity']
    if not REQUIRED_INTEGRITY.issubset(integrity) or not all(v is True for v in integrity.values()): raise ValueError('Integrity gate failed')
    assessment=report['assessment']
    if assessment.get('used_for_selection') is not False: raise ValueError('Assessment was used for selection')
    if len(frame)!=14479 or frame.surveyId.duplicated().any() or set(frame.fold)!={0,1} or frame.block.nunique()!=64:
        raise ValueError('Outer assessment dimensions changed')
    if len(frame)!=assessment['surveys'] or frame.block.nunique()!=assessment['blocks'] or not np.isfinite(frame.pa_distance_km).all() or (frame.pa_distance_km<20-1e-7).any():
        raise ValueError('Assessment report or buffer mismatch')
    for fold,expected_n in ((0,7435),(1,7044)):
        subset=frame.loc[frame.fold==fold]
        manifest=report['split'][fold]
        digest=hashlib.sha256(np.sort(subset.surveyId.to_numpy()).astype('<i8').tobytes()).hexdigest()
        if len(subset)!=expected_n or digest!=manifest['assessment_ids_sha256'] or manifest['assessment_v21_training_only'] is not True:
            raise ValueError('Assessment IDs or ancestry differ')
        for name in assessment['sample_f1']:
            if not math.isclose(subset[name].mean(),assessment['fold_sample_f1'][fold][name],abs_tol=1e-12,rel_tol=0):
                raise ValueError('Per-fold F1 mismatch')
    for name,score in assessment['sample_f1'].items():
        if not math.isclose(frame[name].mean(),score,abs_tol=1e-12,rel_tol=0): raise ValueError('Pooled F1 mismatch')
    for control,saved in assessment['comparisons'].items():
        actual=paired_block_bootstrap(frame.retained_po.to_numpy(),frame[control].to_numpy(),frame.block.to_numpy())
        if not np.allclose(actual['ci95'],saved['ci95'],rtol=0,atol=1e-12) or not math.isclose(actual['mean_difference'],saved['mean_difference'],abs_tol=1e-12,rel_tol=0):
            raise ValueError('Spatial interval/gain mismatch')
    for path in ('fold_0','fold_1','deployment'):
        for arm in ('retained_po','single_head','zero_po'):
            p=report['selected_policies'][path][arm]['selected']
            if p.get('alpha') not in ALPHAS or p.get('gate') not in ('uniform','pa_distance') or p.get('k')!=20:
                raise ValueError('Unregistered calibration policy')
    gate=gate_from_assessment(assessment,integrity,report['selected_policies'])
    if gate!=report['submission_gate'] or not gate['eligible_for_official_submission']: raise ValueError('Registered submission gate failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['validate','submit','scores'])
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--version',type=int,default=22)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/v22_review')
    args=parser.parse_args(); dest=args.output_dir; dest.mkdir(parents=True,exist_ok=True)
    receipt_path=dest/'official_submission_receipt.json'
    message=f'v22 retained PO {args.source_commit[:12]}: frozen crossfit gate'
    if args.action=='scores':
        if not receipt_path.is_file(): raise ValueError('No recorded submission')
        with official_client() as api: scores=matched_scores(api,message)
        result={'message':message,'matching_submissions':scores}
        (dest/'official_scores.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result)); return
    reader=KaggleReader()
    if reader.status()['status']!='complete': raise ValueError('Run is not complete')
    output=reader.output(version=args.version)
    entries={e['fileName']:e for e in output['files']}
    prefix='geolifeclef-risk-aware-sdm/artifacts/retained_po_v22/'
    for name in ('v22_report.json','GLC25_PA_submission.csv','assessment_per_survey.csv','frozen_policies.json',
                 'pre_assessment_freeze.json','unchanged_v21_submission.csv','split_manifest.json','po_manifests.json'):
        reader.download_url(entries[prefix+name]['url'],dest/name)
    report=json.loads((dest/'v22_report.json').read_text())
    validate_assessment(report,pd.read_csv(dest/'assessment_per_survey.csv'),args.source_commit)
    freeze=json.loads((dest/'pre_assessment_freeze.json').read_text())
    if freeze['assessment_reporting_started'] is not False or sha256_file(dest/'frozen_policies.json')!=freeze['policies_sha256']:
        raise ValueError('Policy freeze mismatch')
    if json.loads((dest/'frozen_policies.json').read_text())!=report['selected_policies']: raise ValueError('Report policies mismatch')
    species=np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    template=pd.read_csv(ROOT/'artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv').surveyId.to_numpy()
    submission=validate_submission(dest/'GLC25_PA_submission.csv',template,species)
    digest=submission['submission_sha256']
    if digest!=freeze['submission_sha256'] or digest!=report['submission']['submission_sha256'] or digest==CSV_HASH:
        raise ValueError('Production CSV digest mismatch or unchanged control')
    if sha256_file(dest/'unchanged_v21_submission.csv')!=CSV_HASH: raise ValueError('Original v21 control mismatch')
    validation={'validated':True,'version':args.version,'source_commit':args.source_commit,'submission_sha256':digest,
                'independent_per_survey_means_and_spatial_intervals':True}
    (dest/'independent_validation.json').write_text(json.dumps(validation,indent=2))
    if args.action=='validate': print(json.dumps(validation)); return
    if receipt_path.exists(): raise ValueError('Existing receipt: retrieve scores, never resubmit')
    with official_client() as api:
        if matched_scores(api,message): raise ValueError('Official submission already exists')
        receipt={'status':'attempt_started','message':message,'kernel_version':args.version,'source_commit':args.source_commit,
            'started_at_utc':datetime.now(timezone.utc).isoformat(),'submission_sha256':digest,'automatic_retry_allowed':False}
        with receipt_path.open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2)
        response=api.competition_submit(str(dest/'GLC25_PA_submission.csv'),message,COMPETITION,quiet=True)
        if isinstance(response,str): raise ValueError('Submission response uncertain; inspect scores')
        receipt['status']='request_returned'; receipt_path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':
    try: main()
    except Exception:
        print('v22 validation/Kaggle operation failed safely; inspect stored reports. Never retry a recorded submission.'); raise SystemExit(2)
