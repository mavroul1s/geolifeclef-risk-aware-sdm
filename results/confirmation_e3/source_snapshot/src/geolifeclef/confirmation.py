"""Outcome barrier and shared-model paired endpoints for E3 confirmation."""
import hashlib
import numpy as np

from geolifeclef.external_reliability import read_rows


def cell_digest(ids):
    return hashlib.sha256(np.sort(np.asarray(ids,'<i8')).tobytes()).hexdigest()


def confirmation_targets(dataset,cohort,consumed_ids,admission,expected_models):
    """Reject the entire read before indexing if identity/admission is invalid."""
    ids=cohort.cell_id.to_numpy()
    if not admission.get('all_policies_frozen') or set(admission.get('policy_hashes',{}))!=set(expected_models):
        raise PermissionError('All declared policies must be frozen before confirmation targets')
    if len(np.unique(ids))!=len(ids) or cohort.prediction_row.duplicated().any():
        raise ValueError('Confirmation identities must be unique')
    if np.intersect1d(ids,np.asarray(consumed_ids)).size:
        raise PermissionError('Previously consumed target IDs cannot enter confirmation')
    if cell_digest(ids)!=admission.get('cohort_ID_sha256'):
        raise PermissionError('Target request differs from the locked confirmation cohort')
    return read_rows(dataset,cohort.prediction_row.to_numpy(int))


def paired_confirmation(quality,candidate_quality,baseline_quality,candidate_accept,
                        baseline_accept,blocks,draws=5000,confidence=.975,seed=20261116):
    """Co-primary MSE and selective F1, with shared spatial/model draws."""
    q=np.asarray(quality,float);a=np.asarray(candidate_accept,bool);b=np.asarray(baseline_accept,bool)
    p=np.asarray(candidate_quality,float);r=np.asarray(baseline_quality,float)
    if q.ndim!=2 or any(x.shape!=q.shape for x in [a,b,p,r]):
        raise ValueError('Aligned models by cells required')
    if any(not np.isfinite(x).all() for x in [q,p,r]) or np.any((q<0)|(q>1)):
        raise ValueError('Finite quality estimates and valid F1 required')
    if len(blocks)!=q.shape[1] or not a.sum(1).all() or not b.sum(1).all():
        raise ValueError('Aligned blocks and nonempty accepted sets required')
    mse=(r-q)**2-(p-q)**2
    _,inverse=np.unique(np.asarray(blocks).astype(str),return_inverse=True);n=int(inverse.max()+1)
    columns=[a,b,q*a,q*b,np.ones_like(q),mse]
    sums=np.array([[np.bincount(inverse,weights=row,minlength=n) for row in col] for col in columns])
    samples=np.random.default_rng(seed).integers(n,size=(draws,n))
    boot=sums[:,:,samples].sum(3)
    valid=((boot[0]>0)&(boot[1]>0)&(boot[4]>0)).all(0)
    selected=boot[:,:,valid]
    f=(selected[2]/selected[0]-selected[3]/selected[1]).mean(0)
    m=(selected[5]/selected[4]).mean(0)
    points_f=[float(row[x].mean()-row[y].mean()) for row,x,y in zip(q,a,b)]
    points_m=mse.mean(1).tolist();tail=(1-confidence)/2
    return {'F1_gain':float(np.mean(points_f)),'MSE_gain':float(np.mean(points_m)),
        'F1_interval':np.quantile(f,[tail,1-tail]).tolist() if valid.any() else None,
        'MSE_interval':np.quantile(m,[tail,1-tail]).tolist() if valid.any() else None,
        'model_F1_gains':points_f,'model_MSE_gains':points_m,'models':q.shape[0],
        'cells':q.shape[1],'clusters':n,'confidence_each_endpoint':confidence,
        'valid_draws':int(valid.sum()),'draws':draws,
        'scope':'fixed cohort, predictors and heads; shared spatial resampling, not independent model trials'}


def confirmation_success(comparison,config):
    minimum=config['minimum_practical_gain_each_endpoint'];lower=config['each_endpoint_interval_lower_must_exceed']
    checks={'practical_F1_gain':comparison['F1_gain']>=minimum,
        'practical_MSE_gain':comparison['MSE_gain']>=minimum,
        'F1_interval_positive':comparison['F1_interval'] is not None and comparison['F1_interval'][0]>lower,
        'MSE_interval_positive':comparison['MSE_interval'] is not None and comparison['MSE_interval'][0]>lower,
        'model_F1_guard':min(comparison['model_F1_gains'])>=config['minimum_individual_model_F1_gain'],
        'model_MSE_guard':min(comparison['model_MSE_gains'])>=config['minimum_individual_model_MSE_gain']}
    checks={name:bool(value) for name,value in checks.items()}
    return all(checks.values()),checks
