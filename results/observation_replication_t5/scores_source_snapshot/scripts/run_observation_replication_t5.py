"""Development replay of fixed plant assemblage predictions under censored labels."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import time

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.external_reliability import (
    apply_calibration, decode_counts, prediction_context, read_rows,
)
from geolifeclef.taxonomic_censoring import censor_genera, paired_spatial_interval
from run_taxonomic_resolution_t1 import now, save, sha

NAME = "observation_replication_t5"
OUT = ROOT / "results" / NAME
DATA = ROOT / "data" / NAME
E1 = ROOT / "data/external_replication_e1"
T1 = ROOT / "data/taxonomic_resolution_t1"
T2 = ROOT / "data/taxonomic_verification_t2"
E1_RESULTS = ROOT / "results/external_replication_e1"
METADATA = ROOT / "data/independent_data_audit/sage_eval_plot_metadata.csv"
CONFIG = json.loads((ROOT / f"configs/{NAME}.json").read_text())


def load_npz(path):
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def paths(phase):
    sources = [
        ROOT / f"configs/{NAME}.json",
        ROOT / f"docs/{NAME}_protocol.md",
        Path(__file__).resolve(),
        ROOT / "scripts/run_taxonomic_resolution_t1.py",
    ]
    inputs = [
        T1 / "assessment_roles.csv", T1 / "fixed_policy.npz", T1 / "contract.npz",
        E1 / "cohort_roles.csv", E1 / "eligible_species.npy",
        E1 / "sealed_eval_test_targets.h5", E1_RESULTS / "retrieval_predictions.json",
        METADATA, T2 / "oracle_contract.npz",
        ROOT / "results/adaptive_baselines_t4/artifact_manifest.json",
    ]
    modules = ["taxonomic_censoring", "external_reliability", "ecological_selection", "reliability"]
    if phase == "scores":
        for model in CONFIG["models"][1:]:
            inputs.extend([
                E1 / model / "test_predictions.h5",
                E1_RESULTS / "policies" / f"{model}.json",
                E1_RESULTS / "policies" / f"{model}.npz",
            ])
    else:
        sources.extend([
            ROOT / "scripts/run_verification_replication_t5.py",
            ROOT / "scripts/validate_linkage_robustness_t3.py",
            ROOT / "scripts/validate_taxonomic_verification_t2.py",
        ])
        inputs.extend([DATA / "fixed_predictions.npz", OUT / "scores_freeze.json",
                       OUT / "prediction_validation.json"])
        modules.extend(["adaptive_verification", "taxonomic_queries", "linkage_verification"])
    sources.extend(ROOT / f"src/geolifeclef/{name}.py" for name in modules)
    return sources, inputs


def freeze(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    sources, inputs = paths(phase)
    identity = {str(path.relative_to(ROOT)): sha(path) for path in sources + inputs}
    lock = OUT / f"{phase}_freeze.json"
    if lock.exists():
        if json.loads(lock.read_text())["identity"] != identity:
            raise ValueError(f"Frozen {phase} source or input changed")
        return lock
    for path in sources:
        copy = OUT / f"{phase}_source_snapshot" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
    save(lock, {"at_utc": now(), "identity": identity, "design": "consumed E1 development"})
    return lock


def start(phase):
    freeze(phase)
    marker = OUT / f"{phase}_access.json"
    with marker.open("x") as handle:
        json.dump({"at_utc": now(), "lock_sha256": sha(OUT / f"{phase}_freeze.json"), "previously_consumed": True},
                  handle, indent=2)


def score_set(prediction, truth, known, empty_value, projected):
    observed_truth = truth & known
    scored_prediction = prediction & known if projected else prediction
    denominator = scored_prediction.sum(1) + observed_truth.sum(1)
    hits = (scored_prediction & observed_truth).sum(1)
    f1 = np.divide(2 * hits, denominator, out=np.full(len(hits), float(empty_value)),
                   where=denominator > 0)
    return f1, (denominator == 0) & truth.any(1) & (empty_value == 1)


def dense_prediction(rank, count, eligible):
    rows, slots = np.nonzero(np.arange(rank.shape[1])[None, :] < count[:, None])
    mask = np.zeros_like(eligible)
    mask[rows, rank[rows, slots]] = True
    assert not np.any(mask & ~eligible)
    np.testing.assert_array_equal(mask.sum(1), count)
    return mask


def prepare_metadata():
    roles = pd.read_csv(T1 / "assessment_roles.csv")
    original = pd.read_csv(E1 / "cohort_roles.csv")
    assessment = original.role.eq("assessment").to_numpy()
    np.testing.assert_array_equal(roles.cell_id, original.loc[assessment, "cell_id"])
    np.testing.assert_array_equal(roles.prediction_row, original.loc[assessment, "prediction_row"])
    plots = pd.read_csv(METADATA, usecols=["cell_row", "PlotObservationID", "GIVD_ID", "Country"])
    plots = plots[plots.cell_row.isin(roles.cell_id)]
    assert plots.GIVD_ID.notna().all()
    metadata = plots.groupby("cell_row").agg(
        source=("GIVD_ID", "first"), source_count=("GIVD_ID", "nunique"),
        plot_count=("PlotObservationID", "nunique"), country=("Country", "first"),
    ).loc[roles.cell_id]
    for field in ["source_count", "plot_count", "country"]:
        np.testing.assert_array_equal(roles[field], metadata[field])
    roles["source"] = metadata.source.where(metadata.source_count == 1, "").to_numpy()
    counts = roles[roles.source.ne("")].groupby("source").agg(
        cells=("cell_id", "size"), blocks=("spatial_block", "nunique"),
    )
    admitted = counts.index[(counts.cells >= CONFIG["minimum_source_cells"]) &
                            (counts.blocks >= CONFIG["minimum_source_blocks"])]
    assert set(admitted) == set(CONFIG["sources"])
    cohorts = {"all": np.ones(len(roles), bool)}
    cohorts.update({source: roles.source.eq(source).to_numpy() for source in CONFIG["sources"]})
    frozen = load_npz(T1 / "fixed_policy.npz")
    policies = {}
    for source, take in cohorts.items():
        selected = roles.loc[take]
        quotas = selected.country.astype(str) + "|" + selected.area_bin.astype(str)
        selector = QuotaSelection(selected.cell_id, CONFIG["coverage"], quotas)
        a = selector.accept(1 - frozen["HGB_quality"][take])
        b = selector.accept(1 - frozen["SDC"][take])
        assert a.sum() == b.sum() > 0
        if source == "all":
            np.testing.assert_array_equal(a, frozen["accept_HGB"])
            np.testing.assert_array_equal(b, frozen["accept_SDC"])
        policies[source] = (a, b)
    return roles, assessment, cohorts, policies, frozen


def load_inventory(roles, assessment):
    native = np.load(E1 / "eligible_species.npy")[assessment]
    contract = load_npz(T1 / "contract.npz")
    with h5py.File(E1 / "sealed_eval_test_targets.h5") as handle:
        inventory = read_rows(handle["targets"], roles.prediction_row.to_numpy(int))
    np.testing.assert_array_equal(inventory == -1, ~native)
    np.testing.assert_array_equal(native & contract["universe"], contract["eligible"])
    assert np.isin(inventory[native], [0, 1]).all()
    return native, contract, inventory


def prepare_predictions(roles, frozen, native, eligible):
    models = CONFIG["models"]
    predictions = {models[0]: dense_prediction(frozen["rank"], frozen["count"], eligible)}
    verification = {models[0]: {"saved_T1_prediction_exact": True}}
    published = load_npz(T2 / "oracle_contract.npz")
    np.testing.assert_array_equal(predictions[models[0]], published["prediction"])
    receipts = json.loads((E1_RESULTS / "retrieval_predictions.json").read_text())["files"]
    claims = {row["file"]: row["sha256"] for row in receipts}
    lock = json.loads((OUT / "scores_freeze.json").read_text())["identity"]
    decoder = "expected_F1_0_80"
    for model in models[1:]:
        path = E1 / model / "test_predictions.h5"
        key = str(path.relative_to(ROOT))
        policy_path = E1_RESULTS / "policies" / model
        policy = json.loads(policy_path.with_suffix(".json").read_text())
        assert lock[key] == claims[key] == policy["prediction_sha256"]
        before = load_npz(policy_path.with_suffix(".npz"))
        np.testing.assert_array_equal(before["cell_id"], roles.cell_id)
        with h5py.File(path) as handle:
            raw = read_rows(handle["probabilities"], roles.prediction_row.to_numpy(int))
        for use, mask in [("original", native), ("study", eligible)]:
            ordered = prediction_context(apply_calibration(raw, policy["calibration"], mask), mask)
            count = decode_counts(ordered, decoder)
            if use == "original":
                np.testing.assert_array_equal(ordered["rank"], before["rank"])
                np.testing.assert_array_equal(count, before["counts"][list(before["decoders"]).index(decoder)])
        predictions[model] = dense_prediction(ordered["rank"], count, eligible)
        verification[model] = {"published_receipt_match": True, "original_rank_and_count_exact": True}
    np.savez_compressed(DATA / "fixed_predictions.npz",
                        **{name: np.packbits(mask, axis=1) for name, mask in predictions.items()})
    save(OUT / "prediction_validation.json", verification)
    return predictions


def score_study():
    start("scores")
    started = time.monotonic()
    roles, assessment, cohorts, policies, frozen = prepare_metadata()
    native, contract, inventory = load_inventory(roles, assessment)
    eligible = contract["eligible"]
    predictions = prepare_predictions(roles, frozen, native, eligible)
    truth = (inventory > 0) & eligible
    reference = {(model, empty): score_set(pred, truth, eligible, empty, False)[0]
                 for model, pred in predictions.items() for empty in [0, 1]}
    np.testing.assert_array_equal(reference[(CONFIG["models"][0], 1)],
                                  load_npz(T2 / "oracle_contract.npz")["quality"])
    rows, ordering, differences = [], [], {}
    genus = contract["genus_codes"]
    for mechanism, weights in [("uniform", np.ones_like(contract["genus_weights"])),
                               ("rarity_weighted", contract["genus_weights"])]:
        for dose in CONFIG["censor_doses"]:
            for seed in CONFIG["censor_seeds"]:
                if time.monotonic() - started > CONFIG["maximum_seconds"]:
                    raise TimeoutError("T5 measurement budget expired")
                known, _, _ = censor_genera(truth, eligible, genus, weights, roles.cell_id, dose, seed)
                hidden_by_row = (truth & ~known).sum(1)
                known_by_row, eligible_by_row = known.sum(1), eligible.sum(1)
                true_by_row = truth.sum(1)
                for empty in [1, 0]:
                    for projected in [True, False]:
                        treatment = "known_only" if projected else "unknown_as_absence"
                        scores = {model: score_set(pred, truth, known, empty, projected)
                                  for model, pred in predictions.items()}
                        for source, take in cohorts.items():
                            a, b = policies[source]
                            weights_contrast = a / a.sum() - b / b.sum()
                            fields = dict(source=source, mechanism=mechanism, dose=dose, seed=seed,
                                          empty_value=empty, treatment=treatment)
                            rates = dict(known_fraction=known_by_row[take].sum() / eligible_by_row[take].sum(),
                                         hidden_positive_fraction=hidden_by_row[take].sum() / true_by_row[take].sum())
                            for model, (score, false_perfect) in scores.items():
                                full = reference[(model, empty)][take]
                                delta = score[take] - full
                                old_gap = weights_contrast @ full
                                new_gap = weights_contrast @ score[take]
                                key = (source, model, mechanism, dose, empty, treatment)
                                if key not in differences:
                                    differences[key] = np.zeros_like(delta)
                                differences[key] += delta / len(CONFIG["censor_seeds"])
                                rows.append({**fields, **rates, "model": model,
                                             "oracle_f1": full.mean(), "mean_bias": delta.mean(),
                                             "false_perfect": int(false_perfect[take].sum()),
                                             "oracle_policy_gap": old_gap, "observed_policy_gap": new_gap,
                                             "policy_gap_error": new_gap - old_gap,
                                             "policy_reversal": abs(old_gap) >= .002 and old_gap * new_gap < 0})
                            for first, second in combinations(predictions, 2):
                                old_gap = (reference[(first, empty)][take] -
                                           reference[(second, empty)][take]).mean()
                                new_gap = (scores[first][0][take] - scores[second][0][take]).mean()
                                ordering.append({**fields, "first_model": first, "second_model": second,
                                                 "oracle_gap": old_gap, "observed_gap": new_gap,
                                                 "reversal": abs(old_gap) >= .002 and old_gap * new_gap < 0})
            print(f"score replay complete: {mechanism} dose={dose}", flush=True)
    stats = []
    for (source, model, mechanism, dose, empty, treatment), errors in differences.items():
        take = cohorts[source]
        settings = dict(repetitions=CONFIG["bootstrap_draws"], seed=CONFIG["bootstrap_seed"])
        interval = paired_spatial_interval(errors, roles.spatial_block[take], **settings)
        a, b = policies[source]
        contrast = paired_spatial_interval(errors, roles.spatial_block[take], a, b, **settings)
        stats.append(dict(source=source, model=model, mechanism=mechanism, dose=dose,
                          empty_value=empty, treatment=treatment, cells=int(take.sum()),
                          clusters=interval["clusters"], mean_bias=interval["point"],
                          bias_lo=interval["interval"][0], bias_hi=interval["interval"][1],
                          policy_gap_error=contrast["point"], policy_gap_lo=contrast["interval"][0],
                          policy_gap_hi=contrast["interval"][1]))
    pd.DataFrame(rows).to_csv(OUT / "score_masks.csv", index=False)
    pd.DataFrame(stats).to_csv(OUT / "score_intervals.csv", index=False)
    pd.DataFrame(ordering).to_csv(OUT / "model_ordering.csv", index=False)
    save(OUT / "scores_complete.json", {"at_utc": now(), "seconds": time.monotonic() - started,
        "score_rows": len(rows), "ordering_rows": len(ordering), "interval_rows": len(stats),
        "distinct_masks": len(CONFIG["censor_seeds"]) * len(CONFIG["censor_doses"]) * 2,
        "fresh_confirmation": False, "new_fits": 0, "new_inference": 0, "new_gpu_jobs": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["admission", "scores"])
    args = parser.parse_args()
    if args.phase == "admission":
        roles, _, cohorts, policies, _ = prepare_metadata()
        for source, take in cohorts.items():
            a, b = policies[source]
            print(source, int(take.sum()), roles.spatial_block[take].nunique(),
                  int(a.sum()), int((a ^ b).sum()))
    else:
        score_study()
