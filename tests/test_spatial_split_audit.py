from pathlib import Path

import pandas as pd

from scripts.audit_spatial_split import audit_spatial_splits


def test_spatial_audit_recommends_eligible_country(tmp_path: Path) -> None:
    rows = []
    for survey_id in range(10):
        country = "held-out" if survey_id < 2 else "train"
        region = "west" if survey_id < 2 else "east"
        for species_id in (1, 2):
            rows.append(
                {
                    "surveyId": survey_id,
                    "speciesId": species_id,
                    "lon": float(survey_id),
                    "lat": 40.0,
                    "region": region,
                    "country": country,
                }
            )
    path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    report = audit_spatial_splits(path)

    assert report["surveys"] == 10
    assert report["recommended_holdout"]["validation_fraction"] == 0.2
    assert report["recommended_holdout"]["unseen_validation_label_fraction"] == 0.0
    assert report["survey_metadata_conflicts"] == {
        "lon": 0,
        "lat": 0,
        "country": 0,
        "region": 0,
    }
