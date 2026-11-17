import numpy as np
import pytest
from shapely.geometry import box

from geolifeclef.native_region_bridge import assign_regions, constant_region_masks


def test_region_assignment_uses_containment_then_documented_nearest_fallback():
    codes, fallback = assign_regions([box(0,0,1,1), box(3,0,4,1)], ['A','B'], [[.5,.5],[3.5,.5],[4.1,.5]])
    assert codes.tolist() == ['A','B','B'] and fallback.tolist() == [False,False,True]


def test_only_constant_region_masks_can_be_transferred():
    codes = np.array(['A','A','B','B','C'])
    offsets = np.array([0,2,4,5,7,8])
    species = np.array([2,0,0,2,1,1,3,3])
    masks, inconsistent, counts = constant_region_masks(codes, offsets, species, 4)
    assert inconsistent == ['B'] and counts == {'A':2,'B':2,'C':1}
    np.testing.assert_array_equal(masks['A'], [0,2])
    assert 'B' not in masks


def test_invalid_sparse_mask_cannot_become_empty_absence_mask():
    with pytest.raises(ValueError):
        constant_region_masks(['A'], np.array([0,2]), np.array([8]), 4)
