import pandas as pd

from scripts.run_frequency_baseline import evaluate_frequency_baseline


def test_frequency_baseline_uses_training_prevalence_only(tmp_path):
    metadata = pd.DataFrame(
        {
            "surveyId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "speciesId": [10, 11, 10, 12, 10, 13, 14, 15, 10, 16],
        }
    )
    path = tmp_path / "metadata.csv"
    metadata.to_csv(path, index=False)
    result = evaluate_frequency_baseline(path, seed=7, validation_fraction=0.4)
    assert result["train_surveys"] == 3
    assert result["validation_surveys"] == 2
    assert result["predicted_species_per_survey"] >= 1
    assert 0.0 <= result["micro_f1"] <= 1.0
    assert 0.0 <= result["macro_f1_observed_validation_species"] <= 1.0
