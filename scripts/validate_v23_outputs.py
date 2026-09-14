"""Fail-closed offline validator for downloaded Kaggle v23 outputs; never submits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.stage_frozen_v20 import sha256_file
from scripts.stage_frozen_v22 import OFFICIAL_CSV_SHA256, SOURCE_COMMIT


EXPECTED_KERNEL_VERSION = 23
EXPECTED_SPECIES = 5016
EXPECTED_TEST_ROWS = 14784
EXPECTED_ASSESSMENT_ROWS = 11032
EXPECTED_OUTPUT = "GLC25_PA_submission_v23.csv"


def _one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one downloaded {name}, found {len(matches)}")
    return matches[0]


def _bootstrap(candidate, control, blocks, iterations=500, seed=20250921):
    unique, inverse = np.unique(np.asarray(blocks, dtype=str), return_inverse=True)
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    sums = np.bincount(inverse, weights=differences)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for iteration in range(iterations):
        draw = rng.integers(0, len(unique), size=len(unique))
        estimates[iteration] = sums[draw].sum() / sizes[draw].sum()
    return float(differences.mean()), np.quantile(estimates, [0.025, 0.975])


def validate(root: Path, expected_commit: str, template: Path | None) -> tuple[str, list[str], dict]:
    reasons = []
    report_path = _one(root, "v23_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir = report_path.parent
    checks = {
        "source_commit": report.get("source_commit") == expected_commit,
        "kernel_version": report.get("kernel_version") == EXPECTED_KERNEL_VERSION,
        "frozen_v22_source": report.get("frozen_v22_source_version") == 22 and report.get("frozen_v22_source_commit") == SOURCE_COMMIT,
        "tests_before": report.get("notebook_tests_before_passed") is True,
        "tests_after": report.get("notebook_tests_after_passed") is True,
        "runtime": 0 < report.get("total_pipeline_hours", 99) < 10.5,
        "split_separation": all(item.get("assessment_previously_consumed") is False for item in report.get("split", [])),
        "species": report.get("species") == EXPECTED_SPECIES,
        "test_rows": report.get("test_rows") == EXPECTED_TEST_ROWS,
        "no_submission": report.get("official_submission_made") is False,
    }
    freeze = json.loads((output_dir / "pre_assessment_freeze.json").read_text(encoding="utf-8"))
    checks["frozen_policies"] = sha256_file(output_dir / "frozen_policies.json") == freeze.get("policies_sha256")
    checks["frozen_csv"] = sha256_file(output_dir / EXPECTED_OUTPUT) == freeze.get("submission_sha256")
    checks["unchanged_v22"] = (sha256_file(output_dir / "unchanged_v22_submission.csv") ==
                                freeze.get("unchanged_v22_sha256") == OFFICIAL_CSV_SHA256)
    integrity = report.get("integrity", {})
    checks["integrity"] = bool(integrity) and all(value is True for value in integrity.values())
    assessment = pd.read_csv(output_dir / "assessment_per_survey.csv")
    required = {"surveyId", "fold", "block", "frozen_v22", "single_head_ensemble",
                "retained_po", "zero_po", "large_single_head"}
    checks["assessment_rows"] = (len(assessment) == EXPECTED_ASSESSMENT_ROWS and
                                  not assessment.surveyId.duplicated().any() and required.issubset(assessment))
    expected_assessment = report.get("assessment", {})
    checks["assessment_not_selection"] = expected_assessment.get("used_for_selection") is False
    for name in ("frozen_v22", "single_head_ensemble", "retained_po", "zero_po", "large_single_head"):
        actual = float(assessment[name].mean())
        expected = expected_assessment.get("sample_f1", {}).get(name, np.nan)
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            reasons.append(f"assessment mean mismatch: {name}")
    for name in ("frozen_v22", "retained_po", "zero_po", "large_single_head"):
        mean, interval = _bootstrap(assessment.single_head_ensemble, assessment[name], assessment.block)
        expected = expected_assessment.get("comparisons", {}).get(name, {})
        if (not np.isclose(mean, expected.get("mean_difference", np.nan), rtol=0, atol=1e-12)
                or not np.allclose(interval, expected.get("ci95", [np.nan, np.nan]), rtol=0, atol=1e-12)):
            reasons.append(f"assessment interval mismatch: {name}")
    csv_path = output_dir / EXPECTED_OUTPUT
    submission = pd.read_csv(csv_path, dtype={"predictions": str})
    counts = submission.predictions.str.split().str.len().to_numpy()
    checks["submission_rows"] = len(submission) == EXPECTED_TEST_ROWS and not submission.surveyId.duplicated().any()
    checks["cardinality"] = counts.min() >= 16 and counts.max() <= 28
    if template is not None:
        template_ids = pd.read_csv(template).surveyId.to_numpy()
        checks["template_order"] = np.array_equal(submission.surveyId.to_numpy(), template_ids)
    else:
        checks["template_order"] = report.get("submission", {}).get("template_order_verified") is True
    policy = report.get("selected_policies", {}).get("deployment", {}).get("single_head_ensemble", {}).get("selected", {})
    comparisons = expected_assessment.get("comparisons", {})
    folds = expected_assessment.get("fold_sample_f1", [])
    gate = {
        "positive_spatial_ci_over_frozen_v22": comparisons.get("frozen_v22", {}).get("ci95", [-1])[0] > 0,
        "positive_gain_over_zero_po": comparisons.get("zero_po", {}).get("mean_difference", -1) > 0,
        "primary_beats_retained_po": comparisons.get("retained_po", {}).get("mean_difference", -1) > 0,
        "primary_beats_v22_each_fold": len(folds) == 2 and all(item["single_head_ensemble"] > item["frozen_v22"] for item in folds),
        "nonzero_production_weight": policy.get("alpha", 0) > 0,
        "all_integrity_checks_pass": checks["integrity"],
    }
    checks["csv_differs_from_v22"] = sha256_file(csv_path) != OFFICIAL_CSV_SHA256
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    for name, passed in gate.items():
        if not passed:
            reasons.append(f"submission_gate:{name}")
    saved_gate = report.get("submission_gate", {}).get("eligible_for_manual_submission") is True
    eligible = not reasons and all(gate.values()) and saved_gate
    return ("ELIGIBLE_FOR_MANUAL_SUBMISSION" if eligible else "DO_NOT_SUBMIT"), sorted(set(reasons)), checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    try:
        status, reasons, checks = validate(args.outputs, args.expected_commit, args.template)
    except Exception as error:
        status, reasons, checks = "DO_NOT_SUBMIT", [str(error)], {}
    print(json.dumps({"status": status, "reasons": reasons, "checks": checks}, indent=2))
    print(status)
    raise SystemExit(0 if status == "ELIGIBLE_FOR_MANUAL_SUBMISSION" else 2)


if __name__ == "__main__":
    main()
