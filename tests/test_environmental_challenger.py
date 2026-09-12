import json
import time

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from geolifeclef.environmental_model import EnvironmentalChallenger
from scripts.prepare_environmental_challenger import (
    aligned_table, discover_environment_pairs, normalize_columns, spatial_partitions,
)
from scripts.run_environmental_challenger import (
    MappedDataset, audit_metrics, policy_counts, ranked_f1, select_policy, top_rank, train_one,
)


def test_spatial_blocks_and_country_audit_ignore_ids_and_labels():
    frame = pd.DataFrame({"lat": [55.11, 55.12, 42.1, 46.1], "lon": [10.11, 10.12, 12.1, 8.1], "country": ["Denmark", "Denmark", "Italy", "Switzerland"], "surveyId": [1, 2, 3, 4]})
    result = spatial_partitions(frame)
    assert result[0] == result[1]
    assert result[2:].tolist() == [3, 3]
    frame.surveyId += 9000
    assert np.array_equal(result, spatial_partitions(frame))
    frame.loc[0, "lat"] = np.nan
    with pytest.raises(ValueError, match="Finite coordinates"):
        spatial_partitions(frame)


def test_normalization_training_only_and_missing_to_zero():
    train = np.array([[1, 2], [3, 4], [10000, np.nan]], dtype=np.float32)
    a, b, stats = normalize_columns(train, np.array([[2, np.nan]], dtype=np.float32), np.array([True, True, False]))
    assert stats["mean"] == [2, 3]
    assert a[2, 1] == 0
    assert b.tolist() == [[0, 0]]


def test_environment_pair_discovery_and_id_alignment(tmp_path):
    root = tmp_path / "EnvironmentalValues" / "Climate"
    root.mkdir(parents=True)
    train = root / "GLC25-PA-train-climate.csv"
    test = root / "GLC25-PA-test-climate.csv"
    pd.DataFrame({"surveyId": [20, 10], "temperature": [2, 1]}).to_csv(train, index=False)
    pd.DataFrame({"surveyId": [30], "temperature": [3]}).to_csv(test, index=False)
    pd.DataFrame({"surveyId": [99]}).to_csv(root / "GLC25-PO-train-climate.csv", index=False)
    assert discover_environment_pairs(tmp_path) == [(train, test)]
    assert aligned_table(train, np.array([10, 20])).temperature.tolist() == [1, 2]
    with pytest.raises(ValueError, match="absent"):
        aligned_table(train, np.array([99]))
    pd.DataFrame({"surveyId": [20, 20], "temperature": [2, 1]}).to_csv(train, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        aligned_table(train, np.array([20]))


def test_ranked_f1_matches_explicit_per_sample_sets():
    probabilities = np.array([[.9, .3, .4], [.2, .8, .9]])
    labels = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    indices, values = top_rank(probabilities)
    counts = policy_counts(values, {"kind": "threshold_min_k", "threshold": .5, "minimum_k": 1})
    assert counts.tolist() == [1, 2]
    assert ranked_f1(labels, indices, counts).tolist() == [1, 1]


def test_zero_challenger_fallback_is_available():
    # At least 50 labels to exercise the genuine bounded top-k policies.
    reference = np.zeros((2, 60), dtype=np.float32)
    reference[:, :8] = .9
    challenger = np.zeros_like(reference)
    challenger[:, 40:] = .9
    targets = np.zeros_like(reference, dtype=np.uint8)
    targets[:, :8] = 1
    best, trials = select_policy(targets, reference, challenger)
    assert best["calibration_f1"] == 1
    assert best["challenger_weight"] == 0
    assert len(trials) == 125


def test_challenger_backward_and_auxiliary_heads():
    torch.set_num_threads(2)
    model = EnvironmentalChallenger(60, environment_features=17, width=64)
    batch = {"landsat": torch.randn(2, 84, 6), "climate": torch.randn(2, 228, 4), "sentinel": torch.rand(2, 4, 32, 32), "static": torch.randn(2, 21), "environment": torch.randn(2, 17)}
    logits, auxiliary = model.forward_with_aux(batch)
    assert logits.shape == (2, 60)
    assert len(auxiliary) == 2
    loss = logits.square().mean() + sum(p.square().mean() for p in auxiliary)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_prepare_and_training_integration(tmp_path, monkeypatch):
    from scripts import prepare_environmental_challenger as preprocessing
    root, prepared, artifacts = tmp_path / "source", tmp_path / "prepared", tmp_path / "artifacts"
    root.mkdir()
    artifacts.mkdir()
    n = 400
    rows = pd.DataFrame({"surveyId": np.arange(1, n + 1), "speciesId": np.arange(n) % 60, "lat": np.full(n, 55.1), "lon": np.full(n, 10.1), "country": ["Denmark"] * n, "year": [2020] * n, "geoUncertaintyInM": [10] * n, "areaInM2": [25] * n})
    rows.to_csv(root / "GLC25_PA_metadata_train.csv", index=False)
    test = rows.iloc[:3].drop(columns="speciesId").copy()
    test["surveyId"] += 1000
    test.to_csv(root / "GLC25_PA_metadata_test.csv", index=False)
    pd.DataFrame({"surveyId": test.surveyId.iloc[::-1], "predictions": [""] * 3}).to_csv(root / "GLC25_SAMPLE_SUBMISSION.csv", index=False)
    env = root / "EnvironmentalValues"
    env.mkdir()
    pd.DataFrame({"surveyId": rows.surveyId.iloc[::-1], "temperature": np.arange(n)}).to_csv(env / "GLC25-PA-train-climate.csv", index=False)
    pd.DataFrame({"surveyId": test.surveyId, "temperature": [1, 2, 3]}).to_csv(env / "GLC25-PA-test-climate.csv", index=False)
    monkeypatch.setattr(preprocessing, "spatial_partitions", lambda frame: np.repeat(np.arange(4, dtype=np.int8), 100))
    monkeypatch.setattr(preprocessing, "read_modalities", lambda task: (np.ones((84, 6), np.float32), np.ones((228, 4), np.float32), np.ones((4, 32, 32), np.float16)))
    manifest = preprocessing.prepare(root, prepared, workers=1)
    assert manifest["test_labels_used"] is False
    assert manifest["partition_counts"]["training"] == 100
    dataset = MappedDataset(prepared, "train", np.arange(4))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    targets = np.array(dataset.arrays["labels"][:4])
    model = EnvironmentalChallenger(60, environment_features=1, width=64)
    result = train_one(model, loader, loader, targets, torch.device("cpu"), artifacts, "smoke", 1, time.monotonic() + 600, minimum_epochs=1)
    assert result["epochs_completed"] == 1
    assert (artifacts / "smoke_best.pt").is_file()
    json.loads((prepared / "manifest.json").read_text())


def test_audit_country_metric_is_report_only():
    labels = np.eye(3, dtype=np.uint8)
    metric, _ = audit_metrics(labels, labels.astype(np.float32), {"kind": "top_k", "k": 1}, np.array(["Italy", "Switzerland", "Denmark"]))
    assert metric["sample_f1"] == 1
    assert metric["untouched_country_ood_f1"] == 1


def test_master_notebook_and_api_embedded_script_compile():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/geolifeclef_research_pipeline.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "master_notebook", "exec")
    script = (root / "scripts/push_kaggle_kernel.ps1").read_text(encoding="utf-8")
    embedded = script.split('$notebookText = @"', 1)[1].split('"@', 1)[0]
    compile(embedded, "kaggle_bootstrap", "exec")
