import numpy as np
import pytest

from geolifeclef.inference_reproduction import reference_topk_gap_ulps


def test_tied_species_sets_are_rank_equivalent_without_using_labels():
    p = np.array([[.7, .4, .4, .1]], np.float16)
    assert reference_topk_gap_ulps(p, [[0, 1]]).tolist() == [0]
    assert reference_topk_gap_ulps(p, [[0, 2]]).tolist() == [0]


def test_boundary_gap_is_measured_in_actual_float16_units():
    boundary = np.float16(.25)
    unit = np.spacing(boundary)
    p = np.array([[.8, boundary, boundary-2*unit, boundary-3*unit]], np.float16)
    assert reference_topk_gap_ulps(p, [[0, 2]])[0] == 2
    assert reference_topk_gap_ulps(p, [[0, 3]])[0] == 3


def test_invalid_reference_indices_are_rejected():
    with pytest.raises(ValueError):
        reference_topk_gap_ulps(np.array([[.8, .2]], np.float16), [[0, 0]])
