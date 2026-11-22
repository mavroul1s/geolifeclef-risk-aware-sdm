"""Seal or reverify the completed T5/T6 scientific artifacts."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from run_observation_replication_t5 import DATA, OUT as T5, ROOT, now, save, sha
from run_paired_completion_t6 import OUT as T6

MANIFEST = T5 / "artifact_manifest.json"


def receipt(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def verify():
    manifest = json.loads(MANIFEST.read_text())
    for name, recorded in manifest["files"].items():
        if receipt(ROOT / name) != recorded:
            raise ValueError(f"T5/T6 artifact differs from manifest: {name}")
    print(f"Verified {len(manifest['files'])} T5/T6 artifacts")
    return manifest


def seal():
    if MANIFEST.exists():
        return verify()
    full_report = ElementTree.parse(T5 / "test_report.xml").getroot()
    suites = list(full_report) if full_report.tag == "testsuites" else [full_report]
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ("tests", "failures", "errors", "skipped")}
    assert tests == {"tests": 426, "failures": 0, "errors": 0, "skipped": 0}
    t6_report = ElementTree.parse(T6 / "focused_tests.xml").getroot()
    t6_tests = t6_report[0] if t6_report.tag == "testsuites" else t6_report
    assert t6_tests.attrib["tests"] == "2" and t6_tests.attrib["failures"] == "0"
    assert t6_tests.attrib["errors"] == "0" and t6_tests.attrib["skipped"] == "0"

    frozen = {}
    for lock in [T5 / "scores_freeze.json", T5 / "verification_freeze.json", T6 / "freeze.json"]:
        frozen.update(json.loads(lock.read_text())["identity"])
    for name, digest in frozen.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen input or source changed: {name}")

    scores = json.loads((T5 / "scores_complete.json").read_text())
    cost = json.loads((T5 / "verification_complete.json").read_text())
    t6 = json.loads((T6 / "summary.json").read_text())
    assert scores["distinct_masks"] == 60 and t6["case_count"] == 240
    assert cost["executed_shared"] == 480 and cost["executed_local"] == 48
    assert cost["reprices"] == 96 and cost["unchecked_paths"] == 48
    assert cost["guarded_incorrect_endpoint_labels"] == cost["guarded_wrong_practical_signs"] == 0

    t6_data = pd.read_csv(T6 / "endpoints.csv")
    assert t6_data.shared_width.le(t6_data.separate_width + 1e-12).all()
    assert t6_data.shared_lower.le(t6_data.oracle_gap + 1e-12).all()
    assert t6_data.shared_upper.ge(t6_data.oracle_gap - 1e-12).all()
    np.testing.assert_allclose(t6_data.shared_width.mean(), t6["mean_shared_width"], atol=1e-14)
    tests["additional_T6_exhaustive_tests"] = 2
    files = sorted(p for folder in [T5, T6] for p in folder.rglob("*") if p.is_file())
    files += [DATA / "fixed_predictions.npz", DATA / "verification_orders.npz",
              ROOT / "scripts/summarize_observation_replication_t5.py",
              ROOT / "scripts/plot_paired_completion_t6.py", Path(__file__).resolve(),
              ROOT / "docs/observation_replication_t5_results.md",
              ROOT / "docs/paired_completion_t6_results.md",
              ROOT / "docs/external_confirmation_tern_feasibility.md"]
    save(MANIFEST, {"at_utc": now(), "tests": tests,
        "files": {str(path.relative_to(ROOT)): receipt(path) for path in files}})
    return verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true", help="Seal the completed artifacts once")
    if parser.parse_args().seal:
        seal()
    else:
        verify()
