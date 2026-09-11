import numpy as np
import pandas as pd
import torch

from scripts.prepare_landsat_pa import build_split, read_pairs, split_survey_ids


def test_prepare_landsat_split_preserves_chronological_shape(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"surveyId": [1, 1, 2, 3], "speciesId": [10, 11, 10, 12]}).to_csv(metadata_path, index=False)
    cube_root = tmp_path / "cubes"
    cube_root.mkdir()
    for survey_id in (1, 2, 3):
        torch.save(torch.ones(6, 4, 21) * survey_id, cube_root / f"GLC25-PA-train-landsat-time-series_{survey_id}_cube.pt")
    pairs, surveys, species = read_pairs(metadata_path)
    train_ids, val_ids = split_survey_ids(surveys, seed=3, validation_fraction=0.34, max_train=10, max_validation=10)
    sample_ids, landsat, labels, missing = build_split(pairs, np.concatenate((train_ids, val_ids)), species, cube_root)
    assert landsat.shape == (3, 84, 6)
    assert labels.shape == (3, 3)
    assert sample_ids.shape == (3,)
    assert missing == []
