import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.build_v26_notebook import (
    EXPECTED_V25_HASH,
    KAGGLE_KERNEL_SOURCE_LIMIT_BYTES,
    make_notebook,
    packed_consumed_ids,
    packed_v25_submission,
)
import scripts.v26_notebook_core as core
from scripts.v26_notebook_core import (
    MODALITIES,
    POLICIES,
    RASTER_MODALITIES,
    RASTER_SHAPES,
    REMOTE_DIMS,
    CooccurrenceGraph,
    POGridIndex,
    RuntimeGuard,
    SpatialRasterJSDM,
    V24MultimodalRareJSDM,
    _build_models_for_fold,
    _channel_summary,
    _clean_raw_tensor,
    compose_predictions,
    discover_data_root,
    make_outer_split,
    notebook_self_tests,
    oracle_f1_counts,
    richness_features,
    validate_submission,
    write_submission,
)


ROOT = Path(__file__).resolve().parents[1]
V25 = ROOT / "results/v25_kaggle_output"
SPECIES = ROOT / "artifacts/v20_frozen_bundle_v21/species_ids.npy"
CONSUMED_PATHS = [
    ROOT / "artifacts/v21_review/assessment_per_survey.csv",
    ROOT / "artifacts/v22_review/assessment_per_survey.csv",
    ROOT / "artifacts/manual_upload_v23_retry2/user_provided_v25_exports/assessment_per_survey.csv",
    ROOT / "results/v24_kaggle_output/assessment_per_survey_v24.csv",
    V25 / "assessment_per_survey_v25.csv",
]


def test_remote_summary_dimensions_are_frozen():
    landsat = _channel_summary(np.zeros((6, 84), np.float32), 12)
    bioclim = _channel_summary(np.zeros((4, 228), np.float32), 12)
    sentinel_band = _channel_summary(np.zeros((4, 256), np.float32), 16)
    ndvi = _channel_summary(np.zeros((1, 256), np.float32), 16)
    assert len(landsat) == REMOTE_DIMS["landsat"]
    assert len(bioclim) == REMOTE_DIMS["bioclim"]
    assert len(np.concatenate([sentinel_band, ndvi])) == REMOTE_DIMS["sentinel"]


def test_official_float32_fill_values_are_safe_for_float16_cache():
    values = np.asarray([0.0, 12_345.0, np.inf, -np.inf, np.nan, 3.4e38], dtype=np.float32)
    cleaned = _clean_raw_tensor(values)
    cached = cleaned.astype(np.float16)
    assert np.isfinite(cached).all()
    assert cached.tolist() == [0.0, 12344.0, 0.0, 0.0, 0.0, 0.0]


def test_self_tests_cover_model_and_adaptive_count_contracts():
    assert notebook_self_tests() == {"passed": True, "tests": 9}
    model = V24MultimodalRareJSDM({name: 3 for name in MODALITIES}, 7, np.array([1, 3]),
                                  width=16, rank=4)
    assert model.independent_head.out_features == 7
    spatial = SpatialRasterJSDM({name: 3 for name in MODALITIES}, 7, np.ones(7, dtype=bool),
                                raster_width=8, vector_width=16, fusion_width=32, rank=4)
    assert spatial.independent_head.out_features == 7
    probabilities = np.asarray([[0.9, 0.8, 0.7, 0.1]], dtype=np.float32)
    targets = np.asarray([[1, 0, 1, 0]], dtype=np.uint8)
    assert oracle_f1_counts(probabilities, targets, minimum=1, maximum=4).tolist() == [3]


def test_gpu_preflight_fails_before_feature_preparation(monkeypatch):
    monkeypatch.setattr(core.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Session options.*Accelerator.*GPU"):
        core.require_gpu()


def test_richness_features_accept_official_metadata_without_month():
    features = richness_features(
        np.full((3, 40), 0.5, dtype=np.float32), np.zeros(3, dtype=np.float32),
        pd.DataFrame({"year": [2018, 2019, 2020], "country": ["FR", "DE", "ES"]}),
        np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32),
        {"FR": 20.0}, 18.0,
    )
    assert features.shape == (3, 12)
    assert np.isfinite(features).all()


def test_control_policy_is_an_exact_noop():
    base = [list(range(20)), list(range(5, 25))]
    probability = np.random.default_rng(7).random((2, 30), dtype=np.float32)
    graph = CooccurrenceGraph(np.full((30, 2), -1), np.zeros((30, 2)))
    result = compose_predictions(base, probability, np.array([12, 32]), np.full(30, 100),
                                 [{}, {}], [{}, {}], graph, np.zeros(2), dict(POLICIES[0]))
    assert result == base


def test_new_outer_folds_are_fresh_disjoint_and_buffered():
    coordinates = np.array([(lat, lon) for lat in np.arange(-60, 61)
                            for lon in np.arange(-170, 171)], dtype=float)
    rows = pd.DataFrame({"surveyId": np.arange(len(coordinates)),
                         "lat": coordinates[:, 0] + 0.25,
                         "lon": coordinates[:, 1] + 0.25})
    consumed = rows.surveyId.to_numpy(np.int64)[::2]
    fold_zero, manifest_zero = make_outer_split(rows, 0, consumed)
    fold_one, manifest_one = make_outer_split(rows, 1, consumed)
    assert set(fold_zero["assessment"]).isdisjoint(fold_one["assessment"])
    assert manifest_zero["minimum_assessment_training_distance_km"] >= 20
    assert manifest_one["minimum_assessment_training_distance_km"] >= 20
    for split in (fold_zero, fold_one):
        ids = rows.surveyId.to_numpy()[split["assessment"]]
        assert np.intersect1d(ids, consumed).size == 0
        assert set(split["selection"]).isdisjoint(split["calibration"])
        assert set(split["assessment"]).isdisjoint(split["training"])


def test_data_root_supports_nested_kaggle_competition_mount(tmp_path):
    root = tmp_path / "input"
    competition = root / "competitions" / "geolifeclef-2025"
    competition.mkdir(parents=True)
    for name in ("GLC25_PA_metadata_train.csv", "GLC25_PA_metadata_test.csv",
                 "GLC25_SAMPLE_SUBMISSION.csv"):
        (competition / name).write_text("surveyId\n", encoding="utf-8")
    assert discover_data_root([root]) == competition.resolve()


def test_embedded_v25_and_consumed_union_roundtrip_exactly(tmp_path, monkeypatch):
    control_b64, control_sha, raw_sha = packed_v25_submission(
        V25 / "GLC25_PA_submission_v25.csv", SPECIES)
    consumed_b64, consumed_sha, consumed_count = packed_consumed_ids(CONSUMED_PATHS)
    monkeypatch.setattr(core, "FROZEN_V25_PAYLOAD_SHA256", control_sha, raising=False)
    monkeypatch.setattr(core, "FROZEN_V25_RAW_SHA256", raw_sha, raising=False)
    monkeypatch.setattr(core, "CONSUMED_ASSESSMENT_IDS_SHA256", consumed_sha, raising=False)
    monkeypatch.setattr(core, "CONSUMED_ASSESSMENT_IDS_COUNT", consumed_count, raising=False)
    template = pd.read_csv(ROOT / "artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv")
    species = np.load(SPECIES, allow_pickle=False)
    predictions, proof = core.decode_v25_submission(
        control_b64, template.surveyId.to_numpy(), species)
    output = tmp_path / "roundtrip.csv"
    report = write_submission(output, template, template.surveyId.to_numpy(), predictions, species)
    assert report["sha256"] == EXPECTED_V25_HASH
    assert proof["private_score"] == 0.20503
    assert validate_submission(output, template, species)["checks"]["row_order"]
    decoded_consumed = core.decode_consumed_ids(consumed_b64)
    assert len(decoded_consumed) == 69_630
    assert np.all(np.diff(decoded_consumed) > 0)


def test_generated_notebook_has_no_repository_runtime_dependency():
    core_text = (ROOT / "scripts/v26_notebook_core.py").read_text(encoding="utf-8")
    control = packed_v25_submission(V25 / "GLC25_PA_submission_v25.csv", SPECIES)
    consumed = packed_consumed_ids(CONSUMED_PATHS)
    notebook = make_notebook(core_text, *control, *consumed)
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 4
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")
    assert "from scripts." not in code
    assert "from geolifeclef" not in code
    assert notebook["metadata"]["glc_v26"]["required_input"] == ["geolifeclef-2025"]
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"] is True
    path = ROOT / "notebooks/geolifeclef_v26_raw_spatial_raster_ensemble.ipynb"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert path.stat().st_size < KAGGLE_KERNEL_SOURCE_LIMIT_BYTES
    assert on_disk["metadata"]["glc_v26"]["frozen_v25_submission_sha256"] == EXPECTED_V25_HASH
    scope = {}
    exec(compile("".join(on_disk["cells"][1]["source"]), "v26-notebook-core", "exec"), scope)
    exec(compile("".join(on_disk["cells"][2]["source"]), "v26-notebook-payload", "exec"), scope)
    assert scope["notebook_self_tests"]()["passed"] is True


def test_tiny_fold_runs_end_to_end_on_cpu(tmp_path):
    rng = np.random.default_rng(25)
    rows_count, species_count = 120, 40
    arrays = {name: rng.normal(size=(rows_count, 3)).astype(np.float32) for name in MODALITIES}
    rasters = {name: rng.normal(size=(rows_count, *RASTER_SHAPES[name])).astype(np.float16)
               for name in RASTER_MODALITIES}
    labels = np.zeros((rows_count, species_count), dtype=np.uint8)
    for row in range(rows_count):
        labels[row, rng.choice(species_count, size=6, replace=False)] = 1
    rows = pd.DataFrame({"surveyId": np.arange(rows_count),
                         "lat": 35 + np.arange(rows_count) * 0.001,
                         "lon": 20 + np.arange(rows_count) * 0.001,
                         "country": ["synthetic"] * rows_count,
                         "year": 2000 + (np.arange(rows_count) % 21)})
    store = SimpleNamespace(train=arrays, raster_train=rasters, labels=labels,
                            species_ids=np.arange(species_count),
                            dims={name: 3 for name in MODALITIES})
    split = {"training": np.arange(60), "selection": np.arange(60, 80),
             "calibration": np.arange(80, 100), "assessment": np.arange(100, 120)}
    po = POGridIndex({}, np.arange(species_count), np.zeros(species_count), 0, 0)
    bundle, record = _build_models_for_fold("smoke", split, rows, store, po, tmp_path,
                                            RuntimeGuard(3), __import__("torch").device("cpu"), 25)
    assert len(bundle["calibration_trials"]) == len(POLICIES)
    assert len(record["matched_v25_candidate_seeds"]) == 2
    assert record["v26_spatial_raster"]["best_epoch"] >= 1
    assert record["matched_v24"]["best_epoch"] >= 6
