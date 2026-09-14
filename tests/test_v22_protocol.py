import numpy as np
import pandas as pd
import pytest
from scripts.v22_protocol import cap_publisher_weights,crossfit_partitions,mix,select_policy
from scripts.ood_po_protocol import make_partitions,spatial_block_ids
from scripts.run_retained_po import deployment_partitions,gate_from_assessment


def test_publisher_cap_preserves_within_publisher_relative_sampling():
    weights=np.array([60.,30.,3.,3.,2.,2.]); codes=np.array([0,0,1,2,3,4])
    actual=cap_publisher_weights(weights,codes)
    shares=np.bincount(codes,weights=actual)/actual.sum()
    assert shares.max()<=.3+1e-12
    assert actual[0]/actual[1]==2
    assert actual.sum()==pytest.approx(1.)
    assert cap_publisher_weights(np.ones(3),np.arange(3)).sum()==pytest.approx(1.)
    with pytest.raises(ValueError): cap_publisher_weights(np.array([0.]),np.array([0]))


def geography():
    rng=np.random.default_rng(811)
    n=6000
    return pd.DataFrame({'surveyId':np.arange(n),'lat':rng.uniform(38,58,n),'lon':rng.uniform(-5,25,n),'country':['France']*n})


def test_crossfit_excludes_consumed_assessments_and_all_selection_overlap():
    rows=geography(); previous,_,_=make_partitions(rows,minimum_partition_size=10)
    folds=crossfit_partitions(rows,minimum=10)
    outer=(folds[0][0]==3)|(folds[1][0]==3)
    assert not np.any((folds[0][0]==3)&(folds[1][0]==3))
    for split,distance,manifest in folds:
        assert np.all(previous[split==3]==0)
        assert np.all(distance[split>0]>=20-1e-7)
        assert not np.any(outer&np.isin(split,[1,2]))
        blocks=spatial_block_ids(rows)
        assert not set(blocks[split==3])&set(blocks[split==0])
        assert manifest['assessment_v21_training_only']
    changed=rows.assign(speciesId=np.arange(len(rows)),surveyId=np.arange(len(rows))+900000)
    for old,new in zip(folds,crossfit_partitions(changed,minimum=10)):
        np.testing.assert_array_equal(old[0],new[0])


def test_v22_mixture_identity_boundedness_and_calibration_tie():
    base=np.full((3,25),.2,dtype=np.float16); expert=np.full_like(base,.8)
    p={'alpha':0.,'gate':'uniform','k':20}
    assert mix(base,expert,np.zeros(3),p) is base
    actual=mix(base,expert,np.array([0.,10.,100.]),{'alpha':.5,'gate':'pa_distance','k':20})
    np.testing.assert_array_equal(actual[0],base[0])
    assert (actual>=base).all() and (actual<=expert).all()
    selected,trials=select_policy(np.ones_like(base),base,base,np.ones(3))
    assert selected['alpha']==0 and len(trials)==14
    with pytest.raises(ValueError): mix(base,expert,np.ones(3),{'alpha':.9,'gate':'uniform','k':20})


def test_submission_gate_rejects_one_bad_fold_or_zero_production_weight():
    policies={name:{'retained_po':{'selected':{'alpha':.1}}} for name in ('fold_0','fold_1','deployment')}
    assessment={'comparisons':{name:{'ci95':[.001,.003],'mean_difference':.002} for name in ('frozen_v21','frozen_v20','zero_po')},
                'fold_sample_f1':[{'retained_po':.3,'frozen_v21':.29}]*2}
    assert gate_from_assessment(assessment,{'ok':True},policies)['eligible_for_official_submission']
    assessment['fold_sample_f1'][1]={'retained_po':.28,'frozen_v21':.29}
    assert not gate_from_assessment(assessment,{'ok':True},policies)['eligible_for_official_submission']
    assessment['fold_sample_f1'][1]={'retained_po':.3,'frozen_v21':.29}
    policies['deployment']['retained_po']['selected']['alpha']=0
    assert not gate_from_assessment(assessment,{'ok':True},policies)['eligible_for_official_submission']
