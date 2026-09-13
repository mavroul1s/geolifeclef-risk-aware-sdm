import json

import numpy as np
import pandas as pd
import pytest

from scripts.ood_po_protocol import frozen_v20_blend
from scripts.prepare_environmental_challenger import spatial_partitions
from scripts.stage_frozen_v20 import (
    MODELS, ordered_ids_sha256, stage_bundle, submission_bytes, verify_bundle, verify_original_ties,
)


def source_fixture(tmp_path):
    source, output = tmp_path / "source", tmp_path / "bundle"
    source.mkdir()
    # Select coordinates through the original protocol, without hard-coding its
    # arithmetic. The two calibration IDs deliberately arrive out of order.
    rows = pd.DataFrame({"lat": np.arange(100) * .1 + 45, "lon": np.full(100, 3.1), "country": "France", "surveyId": np.arange(100) + 100})
    rows = rows.loc[spatial_partitions(rows) == 2].iloc[:3].iloc[::-1]
    assert len(rows) >= 2
    rows.to_csv(source / "train.csv", index=False)
    test_ids = np.array([25, 12, 97, 3])
    template_ids = test_ids[::-1]
    pd.DataFrame({"surveyId": test_ids}).to_csv(source / "test.csv", index=False)
    pd.DataFrame({"surveyId": template_ids}).to_csv(source / "template.csv", index=False)
    species = np.arange(5016, dtype=np.int64) + 10
    manifest = {"protocol": "environmental_challenger_v20", "species": len(species), "species_ids": species.tolist(), "test_samples": len(test_ids),
                "partition_counts": {"policy_calibration": len(rows)}, "partition_ids_sha256": {"policy_calibration": ordered_ids_sha256(rows.surveyId.to_numpy())}}
    (source / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "frozen_policy.json").write_text(json.dumps({"selected": {"challenger_weight": .75, "policy": {"kind": "top_k", "k": 20}}, "further_training": False}), encoding="utf-8")
    rng = np.random.default_rng(91)
    for split, count in (("calibration", len(rows)), ("test", len(test_ids))):
        for model in MODELS:
            np.save(source / f"{model}_{split}_probabilities.npy", rng.random((count, len(species))).astype(np.float16), allow_pickle=False)
    test_components = [np.load(source / f"{model}_test_probabilities.npy") for model in MODELS]
    original = submission_bytes(frozen_v20_blend(*test_components), species, test_ids, template_ids)
    (source / "GLC25_PA_submission.csv").write_bytes(original)
    return source, output, rows.surveyId.to_numpy(), test_ids, template_ids


def stage_fixture(source, output, **kwargs):
    return stage_bundle(source, output, source / "test.csv", source / "template.csv", source / "train.csv", expected_test_samples=4, **kwargs)


def test_bundle_preserves_float32_arithmetic_all_species_order_hashes_and_submission_bytes(tmp_path):
    source, output, calibration_ids, test_ids, _ = source_fixture(tmp_path)
    report = stage_fixture(source, output)
    expected = frozen_v20_blend(*(np.load(source / f"{model}_test_probabilities.npy") for model in MODELS))
    actual = np.load(output / "v20_test_probabilities.npy", allow_pickle=False)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert np.load(output / "species_ids.npy").shape == (5016,)
    assert (output / "v20_original_submission.csv").read_bytes() == (source / "GLC25_PA_submission.csv").read_bytes()
    assert report["test"]["submission_byte_parity"] is True
    assert report["calibration"]["id_order_verified_locally"] is True
    assert report["calibration"]["ordered_ids_sha256"] != ordered_ids_sha256(np.sort(calibration_ids))
    assert len(report["source_components"]) == 6
    assert not any(path.suffix == ".pt" for path in output.iterdir())
    verify_bundle(output, calibration_ids=calibration_ids, test_ids=test_ids)
    with pytest.raises(ValueError, match="fresh empty"):
        stage_fixture(source, output)


def test_staging_rejects_probability_dimension_or_nonfinite_value(tmp_path):
    source, output, _, _, _ = source_fixture(tmp_path)
    path = source / "reference_2025_test_probabilities.npy"
    original = np.load(path)
    np.save(path, original[:, :-1])
    with pytest.raises(ValueError, match="shape/dtype"):
        stage_fixture(source, output)
    assert not (output / "provenance.json").exists()
    # Use a fresh directory since partially staged arrays are intentionally not
    # silently accepted or mixed with files from a later retry.
    original[0, 0] = np.nan
    np.save(path, original)
    with pytest.raises(ValueError, match="finite probability"):
        stage_fixture(source, tmp_path / "other")


def test_staging_refuses_changed_top20_predictions(tmp_path):
    source, output, _, _, _ = source_fixture(tmp_path)
    path = source / "GLC25_PA_submission.csv"
    lines = path.read_bytes().splitlines(keepends=True)
    first = lines[1].decode().split(",", 1)
    predictions = first[1].split()
    predictions[0] = "999999"
    lines[1] = (first[0] + "," + " ".join(predictions) + "\r\n").encode()
    path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="predictions/order"):
        stage_fixture(source, output)
    assert not (output / "provenance.json").exists()


def test_staging_requires_byte_parity_and_original_calibration_order(tmp_path):
    source, output, _, _, _ = source_fixture(tmp_path)
    path = source / "GLC25_PA_submission.csv"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="CSV bytes"):
        stage_fixture(source, output)
    rows = pd.read_csv(source / "train.csv").iloc[::-1]
    rows.to_csv(source / "train.csv", index=False)
    with pytest.raises(ValueError, match="calibration ID order/hash"):
        stage_fixture(source, tmp_path / "other")


def test_mounted_bundle_detects_file_tampering_and_independent_id_reordering(tmp_path):
    source, output, calibration_ids, test_ids, _ = source_fixture(tmp_path)
    stage_fixture(source, output)
    with pytest.raises(ValueError, match="calibration ID order"):
        verify_bundle(output, calibration_ids=calibration_ids[::-1])
    with pytest.raises(ValueError, match="test ID order"):
        verify_bundle(output, test_ids=test_ids[::-1])
    with (output / "species_ids.npy").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="file integrity"):
        verify_bundle(output)


def test_submission_reconstruction_rejects_duplicate_template_ids(tmp_path):
    _, _, _, test_ids, template_ids = source_fixture(tmp_path)
    template_ids[0] = template_ids[1]
    with pytest.raises(ValueError, match="Duplicate"):
        submission_bytes(np.ones((4, 5016), np.float32), np.arange(5016), test_ids, template_ids)


def test_original_boundary_tie_is_preserved_but_unequal_score_is_rejected():
    probabilities = np.full((1, 60), .5, dtype=np.float32)
    original = ('surveyId,predictions\r\n1,' + ' '.join(map(str, range(20))) + '\r\n').encode()
    proof = verify_original_ties(probabilities, np.arange(60), np.array([1]), np.array([1]), original)
    assert proof['exact_rank_probability_parity']
    probabilities[0, 59] = .6
    with pytest.raises(ValueError, match='beyond exact probability ties'):
        verify_original_ties(probabilities, np.arange(60), np.array([1]), np.array([1]), original)
