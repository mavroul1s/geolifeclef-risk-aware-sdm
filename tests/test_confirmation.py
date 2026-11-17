import numpy as np
import pandas as pd
import pytest

from geolifeclef.confirmation import cell_digest,confirmation_targets,paired_confirmation,confirmation_success


class TargetTrap:
    shape=(30,2)
    dtype=np.dtype('int8')
    chunks=None
    def __init__(self):self.requested=[]
    def __getitem__(self,rows):
        self.requested.extend(rows.tolist())
        return np.ones((len(rows),2),np.int8)


def case():
    c=pd.DataFrame({'cell_id':[10,20],'prediction_row':[2,5]})
    a={'all_policies_frozen':True,'policy_hashes':{'a':'hash','b':'hash'},'cohort_ID_sha256':cell_digest(c.cell_id)}
    return c,a


def test_target_barrier_reads_only_exact_locked_unconsumed_rows():
    c,a=case();dataset=TargetTrap()
    y=confirmation_targets(dataset,c,[1,3],a,['a','b'])
    assert y.shape==(2,2) and dataset.requested==[2,5]


@pytest.mark.parametrize('failure',['not_frozen','missing_model','consumed','changed_cohort'])
def test_invalid_confirmation_admission_fails_before_any_dataset_indexing(failure):
    c,a=case();dataset=TargetTrap();consumed=[1,3]
    if failure=='not_frozen':a['all_policies_frozen']=False
    if failure=='missing_model':a['policy_hashes'].pop('b')
    if failure=='consumed':consumed=[10]
    if failure=='changed_cohort':c.loc[0,'cell_id']=11
    with pytest.raises(PermissionError):confirmation_targets(dataset,c,consumed,a,['a','b'])
    assert dataset.requested==[]


def test_joint_endpoints_have_correct_direction_and_duplicate_models_do_not_shrink_intervals():
    q=np.array([[.1,.9,.2,.8,.3,.7]])
    a=np.array([[0,1,0,1,0,1]],bool);b=~a
    p=q.copy();r=np.full_like(q,.5);blocks=['a','a','b','b','c','c']
    single=paired_confirmation(q,p,r,a,b,blocks,draws=200)
    repeated=paired_confirmation(*(np.tile(x,(4,1)) for x in [q,p,r,a,b]),blocks,draws=200)
    for name in ['F1_gain','MSE_gain','F1_interval','MSE_interval']:
        np.testing.assert_allclose(single[name],repeated[name])
    assert single['F1_gain']>0 and single['MSE_gain']>0 and single['clusters']==3


def test_same_policy_still_allows_magnitude_improvement_but_fails_joint_success():
    q=np.array([[.1,.8,.2,.7]]);a=np.array([[1,1,0,0]],bool)
    result=paired_confirmation(q,q,np.full_like(q,.5),a,a,['a','a','b','b'],draws=100)
    assert result['F1_gain']==0 and result['MSE_gain']>0
    config={'minimum_practical_gain_each_endpoint':.002,'each_endpoint_interval_lower_must_exceed':0,
        'minimum_individual_model_F1_gain':-.001,'minimum_individual_model_MSE_gain':0}
    success,checks=confirmation_success(result,config)
    assert not success and not checks['practical_F1_gain'] and checks['practical_MSE_gain']


def test_nonfinite_or_unaligned_inputs_do_not_yield_misleading_success():
    q=np.array([[.2,.3]])
    with pytest.raises(ValueError):paired_confirmation(q,q,q,np.ones_like(q,bool),np.ones_like(q,bool),['a'])
