import itertools
import numpy as np

from geolifeclef.partial_set_bounds import fixed_set_f1_bounds,linear_contrast_bounds,mse_gain_bounds


def test_f1_bounds_equal_exhaustive_binary_completions_including_empty_sets():
    for known_values in [[-1,-1,-1], [1,-1,0], [0,-1,0], [1,0,1]]:
        y=np.array([known_values]);known=y>=0;eligible=np.ones_like(known);rank=np.array([[0,1,2]])
        for count in range(4):
            low,high=fixed_set_f1_bounds(y,known,eligible,rank,np.array([count]));values=[]
            missing=np.flatnonzero(~known[0])
            for fill in itertools.product([0,1],repeat=len(missing)):
                truth=y[0].copy();truth[missing]=fill;den=count+truth.sum()
                values.append(2*truth[:count].sum()/den if den else 1.)
            assert np.isclose(low[0],min(values)) and np.isclose(high[0],max(values))


def test_paired_contrast_cancels_shared_accepted_cells():
    lower=np.array([.1,.2,.3]);upper=np.array([.9,.8,.7]);weight=np.array([.5,0,-.5])
    actual=linear_contrast_bounds(lower,upper,weight)
    assert np.allclose(actual,[-.3,.3])
    gain=mse_gain_bounds(lower,upper,.4,np.array([.2,.5,.6]))
    all_values=[np.mean((.4-np.array(y))**2-(np.array([.2,.5,.6])-np.array(y))**2) for y in itertools.product(*zip(lower,upper))]
    assert np.allclose(gain,[min(all_values),max(all_values)])
