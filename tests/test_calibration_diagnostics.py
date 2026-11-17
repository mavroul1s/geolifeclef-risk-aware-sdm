import numpy as np
import pytest

from geolifeclef.calibration_diagnostics import mse_decomposition


def test_mse_decomposition_reconstructs_shifted_noisy_prediction():
    y=np.linspace(0,1,40);p=.3+.2*y
    result=mse_decomposition(p,y)
    assert result['MSE']==pytest.approx(np.mean((p-y)**2))
    assert result['squared_bias']+result['prediction_variance']+result['quality_variance']+result['minus_twice_covariance']==pytest.approx(result['MSE'])


def test_training_mean_constant_loses_under_new_domain_mean_shift():
    y=np.array([.3,.4,.5]);result=mse_decomposition(np.full(3,.1),y)
    assert result['prediction_variance']==pytest.approx(0,abs=1e-30)
    assert result['minus_twice_covariance']==pytest.approx(0,abs=1e-30)
    assert result['MSE']==pytest.approx(.09+y.var())
