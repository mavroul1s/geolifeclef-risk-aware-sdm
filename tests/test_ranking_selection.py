import numpy as np
import pandas as pd
import pytest

from geolifeclef.ranking_selection import (RobustScaling,training_pairs,pairwise_objective,
    fit_anchored,monotone_affine,choose_penalty,spatial_splits)


def test_pairwise_gradient_matches_finite_difference():
    rng=np.random.default_rng(12);x=rng.normal(size=(20,3));b=rng.normal(size=20)
    w=rng.random(20);w/=w.sum();coef=rng.normal(size=3)
    value,gradient=pairwise_objective(coef,x,b,w,.3)
    numerical=[]
    for i in range(3):
        step=np.zeros(3);step[i]=1e-6
        numerical.append((pairwise_objective(coef+step,x,b,w,.3)[0]-pairwise_objective(coef-step,x,b,w,.3)[0])/2e-6)
    np.testing.assert_allclose(gradient,numerical,rtol=1e-5,atol=1e-6)


def test_pairs_are_within_groups_oriented_by_quality_and_permutation_invariant():
    ids=np.array([30,10,20,60,50,40]);g=np.array(['a']*3+['b']*3);y=np.array([0,.8,.5,.9,.1,.2])
    hi,lo,w=training_pairs(ids,g,y,2,19)
    assert np.all(g[hi]==g[lo]) and np.all(y[hi]>y[lo]) and w.sum()==pytest.approx(1)
    perm=np.array([4,3,1,5,0,2]);a,b,v=training_pairs(ids[perm],g[perm],y[perm],2,19)
    assert list(zip(ids[hi],ids[lo]))==list(zip(ids[perm][a],ids[perm][b]))
    np.testing.assert_allclose(w,v)


def test_scaling_uses_training_values_only_and_handles_missing_constant_columns():
    x=np.array([[1.,np.nan,3],[2,np.nan,3],[3,np.nan,3]])
    scaling=RobustScaling.fit(x);old=scaling.center.copy()
    transformed=scaling.transform([[1e9,5.,np.nan]])
    assert np.isfinite(transformed).all() and abs(transformed).max()<=5
    np.testing.assert_array_equal(scaling.center,old)


def test_ranker_learns_quality_order_when_baseline_is_constant():
    x=np.arange(60)[:,None]/60;y=x[:,0];base=np.ones(60)*.4
    ranker=fit_anchored(x,base,y,np.arange(60),['a']*60,.1)
    score=ranker.predict(x,base)
    assert np.corrcoef(score,y)[0,1]>.99 and ranker.audit['pairs']>0


def test_no_comparable_pairs_falls_back_to_unchanged_baseline():
    x=np.array([[1],[2]]);base=np.array([.4,.8])
    fit=fit_anchored(x,base,[.1,.9],[1,2],['a','b'],1.)
    np.testing.assert_array_equal(fit.predict(x,base),base)


def test_monotone_calibration_preserves_decisions_even_when_empirical_slope_is_negative():
    b=np.linspace(.1,.9,8);q=b[::-1]
    score,audit=monotone_affine(b,q,b)
    assert audit['slope']>0
    np.testing.assert_array_equal(np.argsort(score),np.argsort(b))


def test_inner_selection_requires_practical_gain_and_prefers_shrinkage_on_ties():
    assert choose_penalty(.4,{.1:.4001,1.:.4009}) is None
    assert choose_penalty(.4,{.1:.405,1.:.405,10.:.405})==10.


def test_geographic_splits_remove_nearby_training_rows():
    meta=pd.DataFrame({'lat':[0,0.01,10,20,30,40],'lon':[0]*6,
        'spatial_block':['a','b','c','d','e','f']})
    splits=spatial_splits(meta,3,50)
    seen=[]
    for train,val in splits:
        seen.extend(val)
        assert not set(train)&set(val)
        if 0 in val:assert 1 not in train
        if 1 in val:assert 0 not in train
    assert sorted(seen)==list(range(6))


def test_e2_projects_out_assessment_before_target_access():
    from scripts.study_ranking_e2 import calibration_only
    frame=pd.DataFrame({'role':['assessment','probability_calibration','risk_calibration','assessment'],
        'cell_id':[10,20,30,40]})
    result,take=calibration_only(frame)
    assert result.cell_id.tolist()==[20,30] and take.tolist()==[1,2]


def test_mse_residual_includes_bias_and_shrinks_toward_baseline():
    x=np.linspace(-1,1,80)[:,None];base=np.full(80,.3);quality=.5+.1*x[:,0]
    fitted=fit_anchored(x,base,quality,np.arange(80),['a']*80,.1,'mse')
    pred=fitted.predict(x,base)
    assert np.mean((pred-quality)**2)<np.mean((base-quality)**2)/10
    assert fitted.intercept==pytest.approx(.2)


def test_magnitude_bootstrap_uses_shared_sites_for_model_average():
    from scripts.summarize_ranking_e2 import mse_comparison
    rows=[]
    for model in ['a','b','c','d']:
        for i in range(6):
            for method,score in [('HGB',1.),('affine_SDC',0.)]:
                rows.append({'model':model,'method':method,'cell_id':i,'spatial_block':str(i//2),
                    'quality':0.,'score':score})
    result=mse_comparison(pd.DataFrame(rows),100)
    assert result['mean_MSE_difference']==1.
    np.testing.assert_array_equal(result['descriptive_95_interval'],[1,1])
    assert result['clusters']==3
