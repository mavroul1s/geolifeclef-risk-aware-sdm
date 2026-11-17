"""Descriptive identities for magnitude calibration; no policy fitting."""
import numpy as np


def mse_decomposition(prediction,quality):
    p,y=np.asarray(prediction,float),np.asarray(quality,float)
    if p.ndim!=1 or p.shape!=y.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Aligned finite one-dimensional predictions and outcomes required')
    bias=float(p.mean()-y.mean())
    covariance=float(np.mean((p-p.mean())*(y-y.mean())))
    result={'prediction_mean':float(p.mean()),'quality_mean':float(y.mean()),
        'mean_bias':bias,'squared_bias':bias*bias,'prediction_variance':float(p.var()),
        'quality_variance':float(y.var()),'minus_twice_covariance':-2*covariance,
        'MSE':float(np.mean((p-y)**2))}
    reconstructed=result['squared_bias']+result['prediction_variance']+result['quality_variance']+result['minus_twice_covariance']
    if not np.isclose(reconstructed,result['MSE'],atol=1e-12,rtol=1e-10):
        raise ArithmeticError('MSE identity reconstruction failed')
    return result
