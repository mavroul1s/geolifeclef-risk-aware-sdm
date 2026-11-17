"""Elementary identification bounds for a fixed set with unknown binary labels.

These are an evaluation diagnostic, not a new theorem or a confidence interval.
Unknown labels vary independently; extra taxonomic constraints could tighten them.
"""
import numpy as np


def fixed_set_f1_bounds(truth,known,eligible,rank,count):
    y=np.asarray(truth);known=np.asarray(known,bool);eligible=np.asarray(eligible,bool)
    rank,count=np.asarray(rank,int),np.asarray(count,int)
    if y.shape!=known.shape or y.shape!=eligible.shape or np.any(known&~eligible):raise ValueError('Label-space mismatch')
    chosen=np.arange(rank.shape[1])[None,:]<count[:,None]
    pred_eligible=np.take_along_axis(eligible,rank,axis=1)
    if np.any(chosen&~pred_eligible):raise ValueError('Predicted class outside fixed eligible universe')
    pred_known=np.take_along_axis(known,rank,axis=1)
    hit=(chosen&pred_known&np.take_along_axis(y>0,rank,axis=1)).sum(1)
    positives=(y>0).sum(1);unknown_pred=(chosen&~pred_known).sum(1)
    unknown_total=(eligible&~known).sum(1);unknown_other=unknown_total-unknown_pred
    low_den=count+positives+unknown_other;high_den=count+positives+unknown_pred
    low=np.divide(2*hit,low_den,out=np.ones(len(y),dtype=float),where=low_den>0)
    high=np.divide(2*(hit+unknown_pred),high_den,out=np.ones(len(y),dtype=float),where=high_den>0)
    if np.any(low>high+1e-12):raise ArithmeticError('Inverted F1 identification interval')
    return low,high


def linear_contrast_bounds(lower,upper,weights):
    lower,upper,weights=np.asarray(lower,float),np.asarray(upper,float),np.asarray(weights,float)
    if lower.shape!=upper.shape or lower.shape!=weights.shape:raise ValueError('Contrast identity mismatch')
    low=np.sum(np.where(weights>=0,weights*lower,weights*upper))
    high=np.sum(np.where(weights>=0,weights*upper,weights*lower))
    return [float(low),float(high)]


def mse_gain_bounds(lower,upper,mean_quality,prediction):
    prediction=np.asarray(prediction,float)
    constant=mean_quality**2-prediction**2;coefficient=2*(prediction-mean_quality)
    low=constant+np.where(coefficient>=0,coefficient*lower,coefficient*upper)
    high=constant+np.where(coefficient>=0,coefficient*upper,coefficient*lower)
    return [float(low.mean()),float(high.mean())]
