import ast
import json
from pathlib import Path

from scripts.build_manual_v23 import _notebook
from scripts.preflight_manual_v23 import REQUIRED_INPUTS


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
