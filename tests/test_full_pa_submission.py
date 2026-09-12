import numpy as np
import pandas as pd

from scripts.prepare_official_pa import feature_paths
from scripts.train_full_pa_ensemble import top_k_species_ids, write_submission


def test_top_k_species_ids_preserves_probability_order():
    probabilities = np.array([[0.1, 0.9, 0.4]], dtype=np.float32)
    assert top_k_species_ids(probabilities, np.array([10, 20, 30]), 2) == [[20, 30]]


def test_official_test_uses_registered_landsat_underscore_name(tmp_path):
    landsat, climate, sentinel = feature_paths(tmp_path, "PA-test", 1001507)
    assert landsat.name == "GLC25-PA-test-landsat_time_series_1001507_cube.pt"
    assert climate.name == "GLC25-PA-test-bioclimatic_monthly_1001507_cube.pt"
    assert sentinel.as_posix().endswith("PA-test/07/15/1001507.tiff")


def test_submission_uses_official_template_order(tmp_path):
    template = tmp_path / "sample.csv"
    output = tmp_path / "submission.csv"
    pd.DataFrame({"surveyId": [2, 1], "predictions": ["", ""]}).to_csv(
        template, index=False
    )
    report = write_submission(
        output,
        template,
        np.array([1, 2]),
        np.array([[0.9, 0.1, 0.3], [0.2, 0.8, 0.7]], dtype=np.float32),
        np.array([10, 20, 30]),
        k=2,
    )
    submission = pd.read_csv(output)
    assert submission["surveyId"].tolist() == [2, 1]
    assert submission["predictions"].tolist() == ["20 30", "10 30"]
    assert report["template_order_preserved"] is True
