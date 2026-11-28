"""Replay frozen T3 query orders under noisy single and double compatibility review."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from geolifeclef.linkage_verification import record_keys
from geolifeclef.taxonomic_queries import PublicKeys, QueryState, contrast_interval
from run_linkage_robustness_t3 import DATA as CASES, OUT as T3, ROOT, T2, now, resident
from run_taxonomic_resolution_t1 import save, sha
from validate_linkage_robustness_t3 import dense, dense_bounds, effective_order, operations


OUT = ROOT / "results/imperfect_audit_sensitivity"
DOSES = (.01, .05)
SEEDS = tuple(range(202611190, 202611200))
REVIEWS = {"lower_error": (.02, .02, .05), "higher_error": (.10, .05, .10)}
FRACTIONS = (0, .05, .25, .5, .75, 1)
PRICE_FIELDS = {"cost_at_0_1": .1, "cost_at_0_5": .5}
MARGIN, WIDTH, EPS = .001, .002, 1e-10
BUDGET_SECONDS = 1200
REVIEW_SEED_OFFSET = 58411


def lock():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "docs/imperfect_audit_protocol.md",
             ROOT / "src/geolifeclef/taxonomic_queries.py", ROOT / "src/geolifeclef/linkage_verification.py",
             ROOT / "scripts/validate_linkage_robustness_t3.py", ROOT / "scripts/validate_taxonomic_verification_t2.py",
             T3 / "artifact_manifest.json", T3 / "query_events.csv", T2 / "oracle_contract.npz"]
    paths.extend(T2 / f"initial_{seed}.npz" for seed in SEEDS)
    paths.extend(CASES / f"case_{seed}_{int(round(1000 * dose))}.npz" for seed in SEEDS for dose in DOSES)
    identity = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    save(OUT / "freeze.json", {"at_utc": now(), "identity": identity, "historically_consumed_data": True,
         "masks": SEEDS, "nominal_linkage_errors": DOSES, "review_profiles": REVIEWS,
         "common_draw_probability": .5, "review_seed_offset": REVIEW_SEED_OFFSET, "prices": PRICE_FIELDS,
         "max_seconds": BUDGET_SECONDS, "width_tolerance": WIDTH, "practical_margin": MARGIN,
         "fixed_order": "T3 paired_width comparison-support groups", "checkpoints": FRACTIONS})


def review_decisions(draw, compatibility, rates):
    false_compatible, false_incompatible, abstain = rates
    threshold = abstain + np.where(compatibility, false_incompatible, false_compatible)
    return np.where(draw < abstain, -1, np.where(draw < threshold, ~compatibility, compatibility))


def group_plan(case, order):
    groups = []
    for group in order:
        records = np.arange(case["offsets"][group], case["offsets"][group + 1])
        groups.append(records[case["support"][records]])
    sizes = np.array([len(group) for group in groups])
    if not len(sizes) or not (sizes > 0).all():
        raise ValueError("Empty group in fixed comparison scope")
    offsets = np.r_[0, np.cumsum(sizes)]
    flat = np.concatenate(groups)
    representative = np.array([group[np.argmin(case["cells"][group])] for group in groups])
    refs = np.repeat(representative, sizes)
    actual = case["private_answers"][flat]
    ref_answer = case["private_answers"][refs]
    return offsets, flat, flat != refs, actual, ref_answer


def audit_actions(case, plan, mode, reviews=None):
    offsets, records, follower, actual, reference = plan
    if mode == "direct_local":
        first = np.zeros_like(follower)
        second = first
        share = first
    elif mode == "ideal_compatibility":
        first = follower
        second = np.zeros_like(follower)
        share = follower & (actual == reference)
    else:
        first = follower
        first_pass = follower & (reviews[0] == 1)
        second = first_pass if mode == "double_review" else np.zeros_like(follower)
        share = first_pass & ((reviews[1] == 1) | ~second)
    resolved = case["private_answers"].copy()
    resolved[records] = np.where(share, reference, actual)
    operations_by_record = {
        "determinations": ~share, "audit_calls": first.astype(np.int8) + second,
        "second_audits": second, "wrong_propagations": share & (actual != reference),
        "abstentions": np.zeros_like(first) if reviews is None else
                       (first & (reviews[0] == -1)).astype(np.int8) + (second & (reviews[1] == -1)),
        "false_incompatible_calls": np.zeros_like(first) if reviews is None else
                       (actual == reference) * ((first & (reviews[0] == 0)).astype(np.int8) + (second & (reviews[1] == 0))),
    }
    return resolved, {key: np.r_[0, np.cumsum(value)][offsets] for key, value in operations_by_record.items()}


def price(counts, prefix):
    values = {key: int(value[prefix]) for key, value in counts.items()}
    return {**values, **{name: float(values["determinations"] + unit * values["audit_calls"])
                         for name, unit in PRICE_FIELDS.items()}}


def sign(lo, hi):
    return int(lo > MARGIN) - int(hi < -MARGIN)


def replay(state, case, plan, order, known_answers, counts, fields, base, initial, started, independent):
    eligible, truth, genera, weights, prediction, quality = base
    oracle = float(weights @ quality)
    offsets, records = plan[:2]
    checkpoints = {fraction: int(np.ceil(fraction * len(order))) for fraction in FRACTIONS}
    lo, hi = contrast_interval(*state.bounds(), weights)
    endpoints, points, dense_checks = {}, [], 0

    def capture(prefix):
        low, high = state.bounds()
        np.testing.assert_allclose([lo, hi], contrast_interval(low, high, weights), rtol=0, atol=EPS)
        direction = sign(lo, hi)
        return {"groups": prefix, "lower": lo, "upper": hi, "width": hi - lo, "oracle": oracle,
                "contains_oracle": lo - EPS <= oracle <= hi + EPS,
                "practical_sign": direction, "wrong_practical_sign": bool(direction and direction * oracle <= MARGIN),
                "incorrect_known_labels": int((state.known & (state.positive != truth)).sum()),
                "cell_exclusions": int(((low > quality + EPS) | (high < quality - EPS)).sum()), **price(counts, prefix)}

    for n in range(len(order) + 1):
        if time.monotonic() - started > BUDGET_SECONDS:
            raise TimeoutError("Frozen audit sensitivity budget exceeded")
        if n:
            members = records[offsets[n - 1]:offsets[n]]
            cells = case["cells"][members]
            before = contrast_interval(*state.bounds(cells), weights[cells])
            for member in members:
                state.query(member, known_answers[member])
            after = contrast_interval(*state.bounds(cells), weights[cells])
            if after[0] < before[0] - EPS or after[1] > before[1] + EPS:
                raise ValueError("A resolved group widened its former completion interval")
            lo += after[0] - before[0]
            hi += after[1] - before[1]
        new_endpoint = []
        if "precision" not in endpoints and hi - lo <= WIDTH + 1e-12:
            new_endpoint.append("precision")
        if "sign" not in endpoints and sign(lo, hi):
            new_endpoint.append("sign")
        reach = [fraction for fraction, at in checkpoints.items() if n == at]
        if reach or new_endpoint:
            row = capture(n)
            points.extend({**fields, **row, "group_fraction": fraction} for fraction in reach)
            for endpoint in new_endpoint:
                endpoints[endpoint] = {**fields, **row, "endpoint": endpoint}

    if independent:
        substitute = {**case, "private_answers": known_answers}
        check = {row["groups"]: row for row in [*points, *endpoints.values()]}
        for n, row in check.items():
            pos, known, _ = dense(substitute, order, "audit_support", n, eligible, initial, genera)
            low, high = dense_bounds(eligible, pos, known, prediction)
            np.testing.assert_allclose(contrast_interval(low, high, weights), [row["lower"], row["upper"]], rtol=0, atol=EPS)
            if int((known & (pos != truth)).sum()) != row["incorrect_known_labels"]:
                raise ValueError("Independent dense known-label reconstruction differs")
            dense_checks += 1
        for kind, row in endpoints.items():
            prev = row["groups"] - 1
            if prev < 0:
                continue
            pos, known, _ = dense(substitute, order, "audit_support", prev, eligible, initial, genera)
            a, b = contrast_interval(*dense_bounds(eligible, pos, known, prediction), weights)
            if (kind == "precision" and b - a <= WIDTH + 1e-12) or (kind == "sign" and sign(a, b)):
                raise ValueError("Apparent endpoint was reached at the previous prefix")
            dense_checks += 1
    for name in ("precision", "sign"):
        if name not in endpoints:
            endpoints[name] = {**fields, "endpoint": name, "groups": None}
    return points, list(endpoints.values()), dense_checks


def main():
    lock()
    started = time.monotonic()
    save(OUT / "access.json", {"at_utc": now(), "protocol_sha256": sha(OUT / "freeze.json"), "new_field_outcomes": False})
    oracle = resident(T2 / "oracle_contract.npz")
    base = [oracle[key] for key in ["eligible", "truth", "genera", "weights", "prediction", "quality"]]
    eligible, truth, genera, weights, prediction, quality = base
    t3 = pd.read_csv(T3 / "query_events.csv")
    frozen = t3[t3["mode"].eq("audit_support") & t3.policy.eq("paired_width")].set_index(["seed", "error_rate"])
    all_points, all_ends, diagnostics = [], [], Counter()
    for seed in SEEDS:
        initial = np.unpackbits(resident(T2 / f"initial_{seed}.npz")["positive"], axis=1, count=eligible.shape[1]).astype(bool)
        for dose in DOSES:
            dose_key = int(round(1000 * dose))
            case = resident(CASES / f"case_{seed}_{dose_key}.npz")
            order = effective_order(case, "paired_width", "audit_support")
            frozen_trace = frozen.loc[(seed, dose)]
            if hashlib.sha256(np.asarray(order, "<i8").tobytes()).hexdigest() != frozen_trace.order_sha256:
                raise ValueError("T3 public order differs")
            plan = group_plan(case, order)
            offsets, _records, _followers, actual, reference = plan
            groups = tuple(case["cells"][a:b] for a, b in zip(case["offsets"][:-1], case["offsets"][1:]))
            keys = record_keys(PublicKeys(case["genus"], groups))
            rng = np.random.default_rng(seed + REVIEW_SEED_OFFSET + dose_key)
            iid = rng.random((2, len(actual)))
            use_common = rng.random(len(order)) < .5
            common = rng.random(len(order)).repeat(np.diff(offsets))
            correlated = np.where(use_common.repeat(np.diff(offsets)), common, iid)
            arms = [("direct_local", "perfect", "none", None), ("ideal_compatibility", "perfect", "none", None)]
            for name, rates in REVIEWS.items():
                for dependence, draws in (("independent", iid), ("correlated", correlated)):
                    decisions = review_decisions(draws, actual == reference, rates)
                    arms.extend((mode, name, dependence, decisions) for mode in ("single_review", "double_review"))
            for mode, profile, dependence, decisions in arms:
                answers, counts = audit_actions(case, plan, mode, decisions)
                state = QueryState(eligible, initial, genera, prediction, keys)
                fields = {"seed": seed, "nominal_linkage_error": dose, "mode": mode,
                          "review_profile": profile, "dependence": dependence}
                points, ends, check_count = replay(state, case, plan, order, answers, counts, fields, base,
                                                   initial, started, seed == SEEDS[0])
                diagnostics["dense_prefix_checks"] += check_count
                if mode in {"direct_local", "ideal_compatibility"}:
                    if any(not row["contains_oracle"] or row["incorrect_known_labels"] for row in points + ends):
                        raise ValueError("Exact-identification control failed support or containment")
                if mode == "ideal_compatibility":
                    for row in ends:
                        endpoint = "sign" if row["endpoint"] == "sign" else "tolerance"
                        n = row["groups"]
                        actual_counts = operations(case, order, "audit_support", n)
                        if (n != frozen_trace[f"{endpoint}_groups"] or
                                (row["determinations"], row["audit_calls"]) != actual_counts or
                                tuple(actual_counts) != (frozen_trace[f"{endpoint}_resolutions"], frozen_trace[f"{endpoint}_audits"])):
                            raise ValueError("Perfect arm did not reproduce T3 stopping and charges")
                        diagnostics["perfect_endpoints_reproduced"] += 1
                all_points.extend(points)
                all_ends.extend(ends)
            print(json.dumps({"mask": seed, "linkage_error": dose, "completed_paths": len(all_ends) // 2}), flush=True)

    points, ends = pd.DataFrame(all_points), pd.DataFrame(all_ends)
    points.to_csv(OUT / "checkpoints.csv", index=False)
    ends.to_csv(OUT / "endpoints.csv", index=False)
    if (len(points), len(ends)) != (1200, 400):
        raise ValueError("Incomplete stress grid")
    save(OUT / "complete.json", {"at_utc": now(), "seconds": time.monotonic() - started,
         "trajectories": len(ends) // 2, "checkpoints": len(points), "endpoint_rows": len(ends),
         **dict(diagnostics), "new_model_fits": 0, "new_field_outcomes": False,
         "output_sha256": {name: sha(OUT / name) for name in ("checkpoints.csv", "endpoints.csv")}})


if __name__ == "__main__":
    main()
