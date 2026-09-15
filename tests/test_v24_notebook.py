import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.build_v24_notebook import (
    EXPECTED_V23_HASH,
    canonical_v23_files,
    make_notebook,
    payload,
)
from scripts.v24_notebook_core import (
    MODALITIES,
    REMOTE_DIMS,
    CooccurrenceGraph,
    POGridIndex,
    RuntimeGuard,
    V24MultimodalRareJSDM,
    _channel_summary,
    _build_models_for_fold,
    _decode_v23_submission,
    compose_predictions,
    make_outer_split,
    notebook_self_tests,
    sha256_file,
    validate_submission,
    verify_frozen_v23,
    write_submission,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/manual_upload_v23_retry2/user_provided_v25_exports"


def test_remote_summary_dimensions_are_frozen():
    landsat = _channel_summary(np.zeros((6, 84), np.float32), 12)
    bioclim = _channel_summary(np.zeros((4, 228), np.float32), 12)
    sentinel_band = _channel_summary(np.zeros((4, 256), np.float32), 16)
    ndvi = _channel_summary(np.zeros((1, 256), np.float32), 16)
    assert len(landsat) == REMOTE_DIMS["landsat"]
    assert len(bioclim) == REMOTE_DIMS["bioclim"]
    assert len(np.concatenate([sentinel_band, ndvi])) == REMOTE_DIMS["sentinel"]


def test_self_tests_cover_model_contract():
    assert notebook_self_tests() == {"passed": True, "tests": 5}
    model = V24MultimodalRareJSDM({name: 3 for name in MODALITIES}, 7, np.array([1, 3]),
                                  width=16, rank=4)
    assert model.independent_head.out_features == 7


def test_control_policy_is_an_exact_noop():
    base = [list(range(20)), list(range(5, 25))]
    probability = np.random.default_rng(7).random((2, 30), dtype=np.float32)
    graph = CooccurrenceGraph(np.full((30, 2), -1), np.zeros((30, 2)))
    policy = {"id": "control", "alpha_near": 0.0, "alpha_far": 0.0,
              "rare_weight": 0.0, "spatial_weight": 0.0,
              "cooccurrence_weight": 0.0, "cardinality_weight": 0.0}
    result = compose_predictions(base, probability, np.array([20, 20]), np.full(30, 100),
                                 [{}, {}], [{}, {}], graph, np.zeros(2), policy)
    assert result == base


def test_new_outer_folds_are_disjoint_and_buffered():
    latitudes = np.arange(-60, 61)
    longitudes = np.arange(-170, 171)
    coordinates = np.array([(lat, lon) for lat in latitudes for lon in longitudes], dtype=float)
    coordinates = coordinates[:18000]
    rows = pd.DataFrame({"surveyId": np.arange(len(coordinates)),
                         "lat": coordinates[:, 0] + 0.25,
                         "lon": coordinates[:, 1] + 0.25})
    fold_zero, manifest_zero = make_outer_split(rows, 0)
    fold_one, manifest_one = make_outer_split(rows, 1)
    assert set(fold_zero["assessment"]).isdisjoint(fold_one["assessment"])
    assert manifest_zero["minimum_assessment_training_distance_km"] >= 20
    assert manifest_one["minimum_assessment_training_distance_km"] >= 20
    for split in (fold_zero, fold_one):
        assert set(split["selection"]).isdisjoint(split["calibration"])
        assert set(split["assessment"]).isdisjoint(split["training"])


def test_embedded_v23_is_exact_and_csv_roundtrips(tmp_path):
    files = canonical_v23_files(EVIDENCE)
    assert sha256_file(EVIDENCE / "GLC25_PA_submission_v23.csv") != EXPECTED_V23_HASH
    destination = tmp_path / "v23"
    proof = verify_frozen_v23(payload(files), destination)
    assert proof["submission_sha256"] == EXPECTED_V23_HASH
    template = pd.read_csv(ROOT / "artifacts/v20_frozen/raw/GLC25_SAMPLE_SUBMISSION.csv")
    species = np.load(ROOT / "artifacts/v20_frozen_bundle_v21/species_ids.npy")
    predictions = _decode_v23_submission(destination / "GLC25_PA_submission_v23.csv",
                                         template.surveyId.to_numpy(), species)
    output = tmp_path / "roundtrip.csv"
    report = write_submission(output, template, template.surveyId.to_numpy(), predictions, species)
    assert report["sha256"] == EXPECTED_V23_HASH
    assert validate_submission(output, template, species)["checks"]["row_order"]


def test_generated_notebook_has_no_repository_runtime_dependency():
    core = (ROOT / "scripts/v24_notebook_core.py").read_text(encoding="utf-8")
    notebook = make_notebook(core, payload(canonical_v23_files(EVIDENCE)))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 4
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")
    assert "from scripts." not in code
    assert "from geolifeclef" not in code
    assert notebook["metadata"]["glc_v24"]["required_input"] == ["geolifeclef-2025"]
    on_disk = json.loads((ROOT / "notebooks/geolifeclef_v24_multimodal_rare_species_sdm.ipynb")
                         .read_text(encoding="utf-8"))
    assert on_disk["metadata"]["glc_v24"]["frozen_v23_submission_sha256"] == EXPECTED_V23_HASH
    scope = {}
    exec(compile("".join(on_disk["cells"][1]["source"]), "v24-notebook-core", "exec"), scope)
    exec(compile("".join(on_disk["cells"][2]["source"]), "v24-notebook-payload", "exec"), scope)
    assert scope["notebook_self_tests"]()["passed"] is True


def test_tiny_fold_runs_end_to_end_on_cpu(tmp_path):
    rng = np.random.default_rng(24)
    rows_count, species_count = 120, 40
    arrays = {name: rng.normal(size=(rows_count, 3)).astype(np.float32) for name in MODALITIES}
    labels = np.zeros((rows_count, species_count), dtype=np.uint8)
    for row in range(rows_count):
        labels[row, rng.choice(species_count, size=6, replace=False)] = 1
    rows = pd.DataFrame({"surveyId": np.arange(rows_count),
                         "lat": 35 + np.arange(rows_count) * 0.001,
                         "lon": 20 + np.arange(rows_count) * 0.001,
                         "country": ["synthetic"] * rows_count,
                         "month": (np.arange(rows_count) % 12) + 1})
    store = SimpleNamespace(train=arrays, labels=labels, species_ids=np.arange(species_count),
                            dims={name: 3 for name in MODALITIES})
    split = {"training": np.arange(60), "selection": np.arange(60, 80),
             "calibration": np.arange(80, 100), "assessment": np.arange(100, 120)}
    po = POGridIndex({}, np.arange(species_count), np.zeros(species_count), 0, 0)
    bundle, record = _build_models_for_fold("smoke", split, rows, store, po, tmp_path,
                                            RuntimeGuard(3), __import__("torch").device("cpu"), 24)
    assert len(bundle["calibration_trials"]) > 1
    assert record["v24"]["best_epoch"] >= 6
