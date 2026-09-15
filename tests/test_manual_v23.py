import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.build_manual_v23 import _notebook
from scripts.preflight_manual_v23 import REQUIRED_INPUTS
import scripts.validate_v23_outputs as postrun


def test_manual_notebook_is_self_contained_and_all_cells_compile():
    notebook = _notebook("a" * 40, b"archive", {"README.md": "b" * 64})
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    metadata = notebook["metadata"]["glc_v23"]
    assert metadata["required_inputs"] == REQUIRED_INPUTS
    assert metadata["max_total_hours"] == 10.5
    assert metadata["submission_performed"] is False


def test_postrun_validator_contains_no_submission_client():
    source = Path("scripts/validate_v23_outputs.py").read_text(encoding="utf-8")
    assert "competition_submit" not in source
    assert "ELIGIBLE_FOR_MANUAL_SUBMISSION" in source
    assert "DO_NOT_SUBMIT" in source


def test_frozen_v22_verifier_rejects_trailing_npy_bytes():
    source = Path("scripts/stage_frozen_v22.py").read_text(encoding="utf-8")
    assert "path.stat().st_size != array.offset + array.nbytes" in source


def test_postrun_validator_recomputes_gate_and_accepts_consistent_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(postrun, "EXPECTED_SPECIES", 24)
    monkeypatch.setattr(postrun, "EXPECTED_TEST_ROWS", 4)
    monkeypatch.setattr(postrun, "EXPECTED_ASSESSMENT_ROWS", 4)
    commit = "a" * 40
    unchanged = tmp_path / "unchanged_v22_submission.csv"
    candidate = tmp_path / postrun.EXPECTED_OUTPUT
    ids = [10, 11, 12, 13]
    old_rows = pd.DataFrame({"surveyId": ids, "predictions": [" ".join(map(str, range(20)))] * 4})
    new_rows = pd.DataFrame({"surveyId": ids, "predictions": [" ".join(map(str, range(1, 21)))] * 4})
    old_rows.to_csv(unchanged, index=False)
    new_rows.to_csv(candidate, index=False)
    monkeypatch.setattr(postrun, "OFFICIAL_CSV_SHA256", hashlib.sha256(unchanged.read_bytes()).hexdigest())
    policies = {**{f"fold_{fold}": {"single_head_ensemble": {"selected": {"alpha": 0.1}}}
                   for fold in (0, 1)},
                "deployment": {"single_head_ensemble": {"selected": {"alpha": 0.1}}}}
    assessment = pd.DataFrame({"surveyId": [1, 2, 3, 4], "fold": [0, 0, 1, 1],
        "block": ["a", "a", "b", "b"], "frozen_v22": [0.1] * 4,
        "single_head_ensemble": [0.4] * 4, "retained_po": [0.3] * 4,
        "zero_po": [0.2] * 4, "large_single_head": [0.35] * 4})
    assessment.to_csv(tmp_path / "assessment_per_survey.csv", index=False)
    comparisons = {}
    for name in ("frozen_v22", "retained_po", "zero_po", "large_single_head"):
        mean, interval = postrun._bootstrap(assessment.single_head_ensemble, assessment[name], assessment.block)
        comparisons[name] = {"mean_difference": mean, "ci95": interval.tolist()}
    report = {"source_commit": commit, "kernel_version": postrun.EXPECTED_KERNEL_VERSION,
        "frozen_v22_source_version": 22, "frozen_v22_source_commit": postrun.SOURCE_COMMIT,
        "notebook_tests_before_passed": True, "notebook_tests_after_passed": True,
        "total_pipeline_hours": 1.0, "split": [{"assessment_previously_consumed": False}] * 2,
        "species": 24, "test_rows": 4, "official_submission_made": False,
        "selected_policies": policies, "integrity": {"ok": True},
        "assessment": {"used_for_selection": False,
            "sample_f1": {name: float(assessment[name].mean()) for name in
                ("frozen_v22", "single_head_ensemble", "retained_po", "zero_po", "large_single_head")},
            "comparisons": comparisons,
            "fold_sample_f1": [{"single_head_ensemble": 0.4, "frozen_v22": 0.1}] * 2},
        "submission": {"template_order_verified": True},
        "submission_gate": {"eligible_for_manual_submission": True}}
    (tmp_path / "v23_report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "frozen_policies.json").write_text(json.dumps(policies), encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    freeze = {"policies_sha256": postrun.sha256_file(tmp_path / "frozen_policies.json"),
        "submission_sha256": postrun.sha256_file(candidate),
        "unchanged_v22_sha256": postrun.sha256_file(unchanged),
        "checkpoint_sha256": {"model.pt": postrun.sha256_file(checkpoint)}}
    (tmp_path / "pre_assessment_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    template = tmp_path / "template.csv"
    pd.DataFrame({"surveyId": ids}).to_csv(template, index=False)
    status, reasons, _ = postrun.validate(tmp_path, commit, template)
    assert status == "ELIGIBLE_FOR_MANUAL_SUBMISSION"
    assert reasons == []
