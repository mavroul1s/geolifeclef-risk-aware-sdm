"""SDC-anchored ranking and magnitude controls for development study E2."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree


@dataclass
class RobustScaling:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features):
        x = np.asarray(features, float)
        if x.ndim != 2 or not len(x):
            raise ValueError('Nonempty training feature matrix required')
        medians, scales = [], []
        for column in x.T:
            finite = column[np.isfinite(column)]
            center = float(np.median(finite)) if len(finite) else 0.
            spread = float(np.subtract(*np.quantile(finite, [.75,.25]))) if len(finite) else 0.
            medians.append(center);scales.append(spread if spread > 1e-8 else 1.)
        return cls(np.array(medians),np.array(scales))

    def transform(self, features):
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != len(self.center):
            raise ValueError('Feature columns changed')
        return np.clip((np.where(np.isfinite(x),x,self.center)-self.center)/self.scale,-5,5)


def buffered_training(metadata, train, validation, buffer_km=50.):
    train, validation = np.asarray(train,int),np.asarray(validation,int)
    if np.intersect1d(train,validation).size or not len(validation):
        raise ValueError('Disjoint training/validation rows required')
    if not len(train):
        return train
    tree = BallTree(np.deg2rad(metadata.iloc[validation][['lat','lon']]),metric='haversine')
    d = tree.query(np.deg2rad(metadata.iloc[train][['lat','lon']]))[0][:,0]*6371.0088
    return train[d >= buffer_km]


def spatial_splits(metadata, folds, buffer_km=50.):
    if metadata.spatial_block.nunique() < folds:
        raise ValueError('Insufficient spatial blocks')
    result = []
    for train,val in GroupKFold(folds).split(metadata,groups=metadata.spatial_block):
        kept = buffered_training(metadata,train,val,buffer_km)
        result.append((kept,val))
    return result


def training_pairs(ids, groups, quality, maximum_per_group=10000, seed=20261116):
    ids,groups,y = np.asarray(ids),np.asarray(groups).astype(str),np.asarray(quality,float)
    if len(np.unique(ids)) != len(ids) or groups.shape != ids.shape or y.shape != ids.shape:
        raise ValueError('Unique aligned training identities required')
    rng = np.random.default_rng(seed)
    winners,losers,weights = [],[],[]
    for group in np.unique(groups):
        take = np.flatnonzero(groups == group)
        take = take[np.argsort(ids[take],kind='stable')]
        if len(take) < 2:
            continue
        left,right = np.triu_indices(len(take),1)
        difference = y[take[left]]-y[take[right]]
        nonzero = np.flatnonzero(np.abs(difference) > 1e-12)
        if len(nonzero) > maximum_per_group:
            nonzero = np.sort(rng.choice(nonzero,maximum_per_group,replace=False))
        if not len(nonzero):
            continue
        l,r,d = take[left[nonzero]],take[right[nonzero]],difference[nonzero]
        winners.extend(np.where(d > 0,l,r));losers.extend(np.where(d > 0,r,l))
        weights.extend(np.abs(d)*len(take)/len(nonzero))
    w = np.asarray(weights,float)
    if len(w):w /= w.sum()
    return np.array(winners,int),np.array(losers,int),w


def pairwise_objective(coef, differences, base_difference, weight, penalty, temperature=10.):
    margin = temperature*(base_difference+differences@coef)
    loss = float(np.dot(weight,np.logaddexp(0,-margin))+.5*penalty*np.dot(coef,coef))
    gradient = differences.T@(-temperature*weight*expit(-margin))+penalty*coef
    return loss,gradient


@dataclass
class AnchoredLinear:
    scaling: RobustScaling
    coefficient: np.ndarray
    intercept: float
    audit: dict

    def predict(self, features, base):
        base = np.asarray(base,float)
        z = self.scaling.transform(features)
        if base.shape != (len(z),) or not np.isfinite(base).all():
            raise ValueError('Aligned finite baseline required')
        return base+z@self.coefficient+self.intercept


def fit_anchored(features, base, quality, ids, groups, penalty, objective='pairwise',
                 maximum_pairs_per_group=10000, seed=20261116, temperature=10.):
    if penalty <= 0 or objective not in ['pairwise','mse']:
        raise ValueError('Positive penalty and declared objective required')
    x,b,y = np.asarray(features,float),np.asarray(base,float),np.asarray(quality,float)
    if b.shape != (len(x),) or y.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(y).all():
        raise ValueError('Aligned finite training quality/baseline required')
    scaling = RobustScaling.fit(x);z = scaling.transform(x)
    if objective == 'mse':
        centered = z-z.mean(0);residual = y-b
        coefficient = np.linalg.solve(centered.T@centered/len(z)+penalty*np.eye(z.shape[1]),
            centered.T@(residual-residual.mean())/len(z))
        intercept = float(residual.mean()-z.mean(0)@coefficient)
        return AnchoredLinear(scaling,coefficient,intercept,{'objective':'mse','rows':len(z),'penalty':penalty})
    winner,loser,weight = training_pairs(ids,groups,y,maximum_pairs_per_group,seed)
    if not len(weight):
        return AnchoredLinear(scaling,np.zeros(z.shape[1]),0.,
            {'objective':'pairwise','rows':len(z),'pairs':0,'fallback':'no within-group quality differences'})
    differences = z[winner]-z[loser]
    result = minimize(pairwise_objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',
        args=(differences,b[winner]-b[loser],weight,penalty,temperature),
        options={'maxiter':300,'ftol':1e-11,'gtol':1e-7})
    if not result.success:
        raise RuntimeError('Pairwise optimization failed: '+str(result.message))
    return AnchoredLinear(scaling,result.x,0.,{'objective':'pairwise','rows':len(z),
        'pairs':len(weight),'penalty':penalty,'optimizer_iterations':int(result.nit),
        'loss':float(result.fun),'coefficient_norm':float(np.linalg.norm(result.x))})


def monotone_affine(base, quality, evaluation):
    b,y = np.asarray(base,float),np.asarray(quality,float)
    slope = max(float(np.dot(b-b.mean(),y-y.mean())/max(np.dot(b-b.mean(),b-b.mean()),1e-12)),1e-9)
    intercept = float(y.mean()-slope*b.mean())
    return slope*np.asarray(evaluation)+intercept,{'slope':slope,'intercept':intercept}


def choose_penalty(baseline_F1, candidate_F1, minimum_gain=.001):
    """Inner validation only; deterministic baseline fallback and shrinkage tie rule."""
    admissible = [(value,float(penalty)) for penalty,value in candidate_F1.items()
                  if value-baseline_F1 >= minimum_gain]
    return max(admissible)[1] if admissible else None
