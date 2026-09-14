import copy
import hashlib
import numpy as np
import pandas as pd
import pytest
from scripts.submit_v22 import validate_assessment,REQUIRED_INTEGRITY
from scripts.ood_po_protocol import paired_block_bootstrap
from scripts.run_retained_po import gate_from_assessment


def fixture():
    n=14479; folds=np.array([0]*7435+[1]*7044)
    block=np.array([f'a{i%28}' for i in range(7435)]+[f'b{i%36}' for i in range(7044)])
    scores={'retained_po':.32,'single_head':.319,'zero_po':.31,'frozen_v21':.30,'frozen_v20':.299}
    frame=pd.DataFrame({'surveyId':np.arange(n),'fold':folds,'block':block,'pa_distance_km':30.,**scores})
    assessment={'used_for_selection':False,'surveys':n,'blocks':64,'sample_f1':scores,
        'fold_sample_f1':[scores.copy(),scores.copy()],
        'comparisons':{name:paired_block_bootstrap(frame.retained_po.to_numpy(),frame[name].to_numpy(),block)
                       for name in ('single_head','zero_po','frozen_v21','frozen_v20')}}
    policies={path:{arm:{'selected':{'alpha':.1,'gate':'pa_distance','k':20}}
              for arm in ('retained_po','single_head','zero_po')} for path in ('fold_0','fold_1','deployment')}
    integrity={name:True for name in REQUIRED_INTEGRITY}
    report={'experiment':'v22_retained_po_crossfit','status':'complete','source_commit':'test',
        'notebook_tests_before_and_after_passed':True,'total_pipeline_hours':2.,'registered_max_total_hours':10.5,
        'external_data_or_weights':False,'test_labels_used':False,'original_v21_retrained_to_recover_outputs':False,
        'frozen_v21_source_version':21,'assessment':assessment,'selected_policies':policies,'integrity':integrity,
        'split':[{'assessment_ids_sha256':hashlib.sha256(np.sort(frame.loc[frame.fold==i,'surveyId'].to_numpy()).astype('<i8').tobytes()).hexdigest(),
                  'assessment_v21_training_only':True} for i in (0,1)]}
    report['submission_gate']=gate_from_assessment(assessment,integrity,policies)
    return report,frame


def test_real_per_survey_submission_validation():
    report,frame=fixture()
    validate_assessment(report,frame,'test')
    frame.loc[0,'retained_po']=0.
    with pytest.raises(ValueError,match='F1'):validate_assessment(report,frame,'test')


@pytest.mark.parametrize('change',['interval','commit','runtime','integrity','policy'])
def test_submission_rejects_false_report_claims(change):
    report,frame=fixture()
    if change=='interval':report['assessment']['comparisons']['frozen_v21']['ci95']=[.1,.2]
    elif change=='commit':report['source_commit']='other'
    elif change=='runtime':report['total_pipeline_hours']=10.6
    elif change=='integrity':report['integrity']['both_folds_fresh']=False
    else:report['selected_policies']['deployment']['retained_po']['selected']['alpha']=.9
    with pytest.raises(ValueError):validate_assessment(report,frame,'test')
