"""Preserve and privately stage exact v21 residual outputs; no training or submission."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scripts.stage_frozen_v20 import sha256_file, verify_original_ties, verify_bundle
from scripts.ood_po_protocol import mix_probabilities, nearest_training_support
from scripts.prepare_environmental_challenger import spatial_partitions
from scripts.kaggle_artifacts import ROOT

REPORT_HASH = '60858bca1e00cbc7fcd52641785975cb6c43a950e6dbd410b2f7e02a55effce2'
CSV_HASH = '93c1d03258f87131eaaf3000dc9df01fe66b2dd61771e0fb4642a1a4a98847a1'
COPIES = ('deployment_po_calibration.npy','deployment_po_test.npy','frozen_policies.json','GLC25_PA_submission.csv','v21_report.json')


def verify_v21(directory):
    directory=Path(directory)
    proof=json.loads((directory/'v21_provenance.json').read_text())
    if proof.get('source_version') != 21 or proof.get('original_rank_probability_parity') is not True:
        raise ValueError('Invalid v21 provenance')
    if set(proof['files']) != set(COPIES):
        raise ValueError('Unexpected frozen v21 files')
    for name,digest in proof['files'].items():
        if sha256_file(directory/name) != digest:
            raise ValueError('Frozen v21 hash mismatch')
    if proof['files']['v21_report.json'] != REPORT_HASH or proof['files']['GLC25_PA_submission.csv'] != CSV_HASH:
        raise ValueError('Wrong officially evaluated v21 source')
    for split,n in (('calibration',7343),('test',14784)):
        value=np.load(directory/f'deployment_po_{split}.npy',mmap_mode='r')
        if value.shape != (n,5016) or value.dtype != np.float16 or not np.isfinite(value).all() or (value<0).any() or (value>1).any():
            raise ValueError('Invalid v21 residual probabilities')
    return proof


def build(source, v20, metadata, test_metadata, template, output):
    verify_bundle(v20)
    output.mkdir(parents=True,exist_ok=True)
    for name in COPIES:
        shutil.copyfile(source/name,output/name)
    if sha256_file(output/'v21_report.json') != REPORT_HASH or sha256_file(output/'GLC25_PA_submission.csv') != CSV_HASH:
        raise ValueError('Original v21 report/CSV mismatch')
    rows=pd.read_csv(metadata).drop_duplicates('surveyId').reset_index(drop=True)
    test=pd.read_csv(test_metadata).drop_duplicates('surveyId').reset_index(drop=True)
    old=spatial_partitions(rows)
    policy=json.loads((output/'frozen_policies.json').read_text())['deployment']['po']['selected']
    if policy['alpha'] != .025 or policy['gate'] != 'pa_distance':
        raise ValueError('Unexpected v21 deployment policy')
    distance=nearest_training_support(rows.loc[old==0],test)['distance_km']
    combined=mix_probabilities(np.load(v20/'v20_test_probabilities.npy',mmap_mode='r'),
        np.load(output/'deployment_po_test.npy',mmap_mode='r'),distance,policy)
    parity=verify_original_ties(combined,np.load(v20/'species_ids.npy'),test.surveyId.to_numpy(),
        pd.read_csv(template).surveyId.to_numpy(),(output/'GLC25_PA_submission.csv').read_bytes())
    proof={'source_version':21,'source_commit':'c04358425500da006acc0572b786f23f0fd9f4a2',
        'original_rank_probability_parity':True,'parity':parity,
        'files':{name:sha256_file(output/name) for name in COPIES},
        'private_competition_derived_only':True,'no_external_data_or_weights':True}
    (output/'v21_provenance.json').write_text(json.dumps(proof,indent=2))
    verify_v21(output)
    return proof


def upload(directory):
    from scripts.submit_v21 import official_client
    verify_v21(directory)
    allowed=set(COPIES)|{'v21_provenance.json','dataset-metadata.json'}
    if any(not p.is_file() or p.name not in allowed for p in directory.iterdir()):
        raise ValueError('Unexpected private bundle payload')
    if not directory.resolve().is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('Bundle must be inside repository artifacts')
    credential=json.loads((ROOT/'api_key/kaggle_2.json').read_text(encoding='utf-8-sig'))
    owner=credential['username']
    dataset=owner+'/geolifeclef-v21-frozen-control'
    (directory/'dataset-metadata.json').write_text(json.dumps({'id':dataset,
        'title':'GeoLifeCLEF frozen v21 control','licenses':[{'name':'other'}],
        'description':'Private competition-derived original v21 residual probabilities and provenance. Competition terms apply; no redistribution permission.'}))
    with official_client() as api:
        response=api.dataset_create_new(str(directory),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
        if str(getattr(response,'status','')).lower()!='ok':
            raise ValueError('Creation uncertain; inspect dataset before any retry')
    return {'dataset':dataset,'private':True,'status':'creation_requested'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','upload','verify'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/v21_frozen_bundle')
    args=parser.parse_args()
    try:
        if args.action=='build':
            result=build(ROOT/'artifacts/v21_frozen_source',ROOT/'artifacts/v20_frozen_bundle_v21',
                ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv',ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_test.csv',
                ROOT/'artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv',args.output)
        else:
            result=upload(args.output) if args.action=='upload' else verify_v21(args.output)
        print(json.dumps(result))
    except Exception:
        print('Frozen v21 operation failed safely; sensitive details withheld. Inspect state before retrying uploads.')
        raise SystemExit(2)
