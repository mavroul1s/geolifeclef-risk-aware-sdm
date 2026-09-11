import numpy as np
import pandas as pd

from scripts.prepare_spatial_multimodal import split_survey_ids, static_features


def test_country_holdout_and_calibration_are_disjoint():
    rows = pd.DataFrame(
        {
            "surveyId": np.arange(100),
            "country": ["Netherlands"] * 20 + ["France"] * 80,
            "lon": np.linspace(0, 20, 100),
            "lat": np.linspace(40, 55, 100),
        }
    )
    train, calibration, validation = split_survey_ids(rows, calibration_fraction=0.2)
    assert set(train).isdisjoint(calibration)
    assert set(train).isdisjoint(validation)
    assert set(calibration).isdisjoint(validation)
    assert set(validation) == set(range(20))


def test_static_features_are_finite_and_have_registered_width():
    rows = pd.DataFrame(
        {
            "lon": [4.5],
            "lat": [52.0],
            "year": [2020],
            "geoUncertaintyInM": [10],
            "areaInM2": [100],
        }
    )
    values = static_features(rows)
    assert values.shape == (1, 21)
    assert np.isfinite(values).all()
