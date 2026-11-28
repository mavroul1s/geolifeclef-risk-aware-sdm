"""Package and independently verify one private GPU replay of public SAGE inputs."""

import argparse
import ast
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tarfile

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import yaml

from audit_ausplots_metadata import ROOT, sha
from check_ausplots_second_model import BRIDGE, CHECKPOINT_INDEX, CONFIGS, DATA, MODEL, MODEL_RELEASE, NORMALIZATION
from check_ausplots_second_model import OUTPUT as CPU_REPORT
from geolifeclef.sage_inference import normalization_from_source


TEMPLATE = ROOT / "notebooks/sage_gpu_reference.ipynb"
JOB = DATA / "private_gpu_job"
NOTEBOOK = JOB / "reference.ipynb"
RESULT = CPU_REPORT.parent / "second_model_gpu_reference.json"
CONTRACT = JOB / "contract.json"
SLUG = "sage-ausplots-reference-20261128"


def replacement_source(notebook, replacements):
    code_cell, = (cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code = "".join(code_cell["source"])
    assignments = {node.targets[0].id: node for node in ast.parse(code).body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id in replacements}
    if assignments.keys() != replacements.keys():
        raise ValueError("Unexpected public reference notebook structure")
    lines = code.splitlines(keepends=True)
    for key, node in sorted(assignments.items(), key=lambda item: item[1].lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [replacements[key] + "\n"]
    compile("".join(lines), "<public_sage_reference>", "exec")
    code_cell["source"] = lines
    return json.dumps(notebook, indent=1) + "\n"


def build():
    receipt = json.loads(CPU_REPORT.read_text())
    source = DATA / "reference.npz"
    if not receipt["reference_passed"] or sha(source) != receipt["reference_bundle_sha256"]:
        raise ValueError("CPU reference admission failed")
    checkpoint = DATA / "best.ckpt"
    if sha(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint digest differs")
    with tarfile.open(CONFIGS) as archive:
        config = yaml.safe_load(archive.extractfile(f"{MODEL}/config.yaml"))
    subset = config["data"]["init_args"]["predictors_subset"]
    network = config["model"]["init_args"]["net"]
    minimal = {MODEL: {"model": {"init_args": {"net": network}}}}
    archive = next(f for f in json.loads(MODEL_RELEASE.read_text())["files"] if f["key"] == "checkpoints.tar")
    index = json.loads(CHECKPOINT_INDEX.read_text())
    name = f"{MODEL}/checkpoints/best.ckpt"
    member = next(record for record in index["members"] if record["name"] == name)
    members = {MODEL: {**member, "sha256": receipt["checkpoint"]["sha256"]}}
    ids = np.load(ROOT / "data/external_replication_e1/test_indices.npy")[:128]
    public_columns = []
    for group, columns in subset.items():
        if group != "location":
            with h5py.File(BRIDGE / "reference_predictors" / f"eval_{group}.h5") as file:
                public_columns.append(file["predictor_data"][:][ids][:, columns])
    means, stds = normalization_from_source(NORMALIZATION, subset)
    probes = np.linspace(0, 5770, 256, dtype=int)
    with np.load(source) as cpu:
        np.testing.assert_array_equal(cpu["ids"], ids)
        np.testing.assert_array_equal(cpu["raw"], np.concatenate(public_columns, axis=1))
        np.testing.assert_array_equal(cpu["means"], means)
        np.testing.assert_array_equal(cpu["stds"], stds)
        # Use directly published probabilities for the notebook's smaller check.
        arrays = {"cell_ids": ids, "probe_columns": probes, f"{MODEL}__raw": cpu["raw"],
                  f"{MODEL}__means": means, f"{MODEL}__stds": stds,
                  f"{MODEL}__probe": cpu["published"][:, probes]}
    content = io.BytesIO()
    np.savez_compressed(content, **arrays)
    bundle = base64.b64encode(content.getvalue()).decode()
    source = replacement_source(json.loads(TEMPLATE.read_text()), {
        "bundle": f"bundle = np.load(io.BytesIO(base64.b64decode({bundle!r})))",
        "members": f"members = {members!r}", "configs": f"configs = json.loads({json.dumps(minimal)!r})",
        "archive_url": f"archive_url = {archive['links']['self']!r}",
    })
    private = ROOT / ".env"
    account = json.loads((private / "kaggle_api_key/kaggle.json").read_text())
    for secret in (account["key"], (private / "neon_api_key/neon_api_key").read_text().strip()):
        if secret and secret in source:
            raise ValueError("Detected a credential in the upload")
    JOB.mkdir(exist_ok=False)
    NOTEBOOK.write_text(source)
    (JOB / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{account['username']}/{SLUG}", "title": "SAGE AusPlots Reference 20261128",
        "code_file": NOTEBOOK.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2) + "\n")
    CONTRACT.write_text(json.dumps({"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "job_limit": 1, "notebook_time_limit_seconds": 1200,
                                    "notebook_sha256": sha(NOTEBOOK), "public_bundle_checked": True,
                                    "template_sha256": sha(TEMPLATE), "cpu_report_sha256": sha(CPU_REPORT),
                                    "new_field_coordinates_covariates_or_labels_included": False}, indent=2) + "\n")
    print("Private reference notebook prepared from public released arrays")


def verify(output):
    contract = json.loads(CONTRACT.read_text())
    cpu_receipt = json.loads(CPU_REPORT.read_text())
    if (sha(NOTEBOOK) != contract["notebook_sha256"] or sha(CPU_REPORT) != contract["cpu_report_sha256"]
            or sha(DATA / "reference.npz") != cpu_receipt["reference_bundle_sha256"]):
        raise ValueError("Frozen reference inputs differ")
    report_path = output / "gpu_report.json"
    probability_path = output / "gpu_probabilities.npz"
    report = json.loads(report_path.read_text())
    run, = report["runs"]
    if (run["run"] != MODEL or not run["probe_passed"] or not report["GPU_name"]
            or report["neural_training_performed"] or report["target_labels_read"]
            or report["elapsed_seconds"] > contract["notebook_time_limit_seconds"]
            or run["checkpoint_sha256"] != cpu_receipt["checkpoint"]["sha256"]):
        raise ValueError("Private run does not match the execution contract")
    with np.load(probability_path) as gpu, np.load(DATA / "reference.npz") as cpu:
        np.testing.assert_array_equal(gpu["cell_ids"], cpu["ids"])
        values = gpu[MODEL]
        if values.shape != (128, 5771) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid GPU probabilities")
        comparisons = {}
        for name in ("cpu", "published"):
            delta = np.abs(values.astype(np.float64) - cpu[name])
            maximum, mean = float(delta.max()), float(delta.mean())
            comparisons[name] = {"maximum_absolute_error": maximum, "mean_absolute_error": mean,
                                 "passed": maximum <= 1e-5 and mean <= 1e-6}
    passed = all(result["passed"] for result in comparisons.values())
    receipt = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
               "passed": passed, "probabilities_compared": int(values.size), "comparisons": comparisons,
               "gpu": report["GPU_name"], "torch_version": report["torch_version"],
               "cuda_version": report["CUDA_version"], "notebook_seconds": report["elapsed_seconds"],
               "inference_seconds": run["inference_seconds"], "frozen_execution": contract,
               "local_output_sha256": {path.name: sha(path) for path in (report_path, probability_path)},
               "public_reference_inputs_only": True, "source_sha256": sha(Path(__file__))}
    with RESULT.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("build")
    command.add_parser("verify").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
