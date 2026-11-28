"""Independently reconcile audit actions and summarize valid and apparent outcomes."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_imperfect_audit_sensitivity import (CASES, DOSES, OUT, PRICE_FIELDS, REVIEW_SEED_OFFSET,
                                            REVIEWS, ROOT, SEEDS, effective_order, resident, save, sha)


def independent_charges(case, order, seed, dose_key, mode, profile, dependence):
    groups = []
    for key in order:
        members = [record for record in range(case["offsets"][key], case["offsets"][key + 1])
                   if case["support"][record]]
        groups.append(members)
    rng = np.random.default_rng(seed + REVIEW_SEED_OFFSET + dose_key)
    separate = rng.random((2, sum(map(len, groups))))
    shared_groups = rng.random(len(groups)) < .5
    shared_value = rng.random(len(groups))
    expected = [Counter()]
    k = 0
    for index, members in enumerate(groups):
        cumulative = expected[-1].copy()
        representative = min(members, key=lambda record: case["cells"][record])
        for record in members:
            actual_match = case["private_answers"][record] == case["private_answers"][representative]
            if record == representative or mode == "direct_local":
                cumulative["determinations"] += 1
            elif mode == "ideal_compatibility":
                cumulative["audit_calls"] += 1
                cumulative["determinations"] += int(not actual_match)
            else:
                wrong_accept, wrong_reject, abstain = REVIEWS[profile]
                for review in range(2 if mode == "double_review" else 1):
                    cumulative["audit_calls"] += 1
                    cumulative["second_audits"] += int(review == 1)
                    draw = (shared_value[index] if dependence == "correlated" and shared_groups[index]
                            else separate[review, k])
                    if draw < abstain:
                        cumulative["abstentions"] += 1
                        cumulative["determinations"] += 1
                        break
                    if actual_match and draw < abstain + wrong_reject:
                        cumulative["false_incompatible_calls"] += 1
                        cumulative["determinations"] += 1
                        break
                    if not actual_match and draw >= abstain + wrong_accept:
                        cumulative["determinations"] += 1
                        break
                else:
                    cumulative["wrong_propagations"] += int(not actual_match)
            k += 1
        expected.append(cumulative)
    return expected


def summarize(checkpoints, endpoints):
    keys = ["nominal_linkage_error", "mode", "review_profile", "dependence"]
    rows = []
    local = endpoints[endpoints["mode"].eq("direct_local")].set_index(["seed", "nominal_linkage_error", "endpoint"])
    for key, points in checkpoints.groupby(keys, sort=True):
        condition = np.logical_and.reduce([endpoints[field].eq(value) for field, value in zip(keys, key)])
        ends = endpoints.loc[condition]
        terminal = points[points.group_fraction.eq(1)]
        row = dict(zip(keys, key))
        row.update({"trajectories": points.seed.nunique(), "any_exclusion": points.loc[~points.contains_oracle, "seed"].nunique(),
                    "checkpoint_wrong_sign_paths": points.loc[points.wrong_practical_sign, "seed"].nunique(),
                    "terminal_wrong_propagations": int(terminal.wrong_propagations.sum()),
                    "terminal_abstentions": int(terminal.abstentions.sum())})
        for endpoint, group in ends.groupby("endpoint"):
            reached = group[group.groups.notna()]
            trusted = reached[reached.contains_oracle.eq(True)]
            row[f"{endpoint}_reached"] = len(reached)
            row[f"{endpoint}_contained"] = len(trusted)
            row[f"{endpoint}_wrong_sign"] = int(reached.wrong_practical_sign.eq(True).sum())
            for price in PRICE_FIELDS:
                reference = local.loc[list(zip(trusted.seed, trusted.nominal_linkage_error, trusted.endpoint)), price]
                relative = trusted[price].to_numpy() / reference.to_numpy()
                row[f"{endpoint}_median_{price}_fraction_of_matched_local"] = (
                    float(np.median(relative)) if len(relative) else None)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    destination = OUT / "validation.json"
    if destination.exists():
        raise FileExistsError(destination)
    freeze, access, complete = [json.loads((OUT / name).read_text())
                                for name in ("freeze.json", "access.json", "complete.json")]
    chronology = [stage["at_utc"] for stage in (freeze, access, complete)]
    if (sorted(set(chronology)) != chronology or access["protocol_sha256"] != sha(OUT / "freeze.json")
            or any(sha(ROOT / path) != value for path, value in freeze["identity"].items())
            or any(sha(OUT / name) != value for name, value in complete["output_sha256"].items())):
        raise ValueError("Frozen inputs, outputs or chronology differ")
    points, ends = [pd.read_csv(OUT / name) for name in ("checkpoints.csv", "endpoints.csv")]
    if (len(points), len(ends), complete["perfect_endpoints_reproduced"]) != (1200, 400, 40):
        raise ValueError("Missing checkpoints or controls")
    charged = pd.concat([points, ends[ends.groups.notna()]], ignore_index=True)
    for price, unit in PRICE_FIELDS.items():
        np.testing.assert_allclose(charged[price], charged.determinations + unit * charged.audit_calls, rtol=0, atol=1e-10)
    first = SEEDS[0]
    counts = 0
    for dose in DOSES:
        dose_key = int(round(1000 * dose))
        case = resident(CASES / f"case_{first}_{dose_key}.npz")
        order = effective_order(case, "paired_width", "audit_support")
        rows = charged[charged.seed.eq(first) & charged.nominal_linkage_error.eq(dose)]
        for (mode, profile, dependence), arm in rows.groupby(["mode", "review_profile", "dependence"]):
            actions = independent_charges(case, order, first, dose_key, mode, profile, dependence)
            for row in arm.itertuples():
                correct = actions[int(row.groups)]
                for field in ("determinations", "audit_calls", "second_audits", "wrong_propagations",
                              "abstentions", "false_incompatible_calls"):
                    if getattr(row, field) != correct[field]:
                        raise ValueError(f"Independent operation count differs for {field}")
                counts += 1
    summary = summarize(points, ends)
    summary.to_csv(OUT / "summary.csv", index=False)
    save(destination, {"passed": True, "freeze_and_output_hashes_checked": True, "all_priced_rows_checked": len(charged),
                       "independent_charge_checkpoints": counts, "dense_prefix_checks": complete["dense_prefix_checks"],
                       "perfect_endpoints_reproduced": 40, "summary_rows": len(summary),
                       "summary_sha256": sha(OUT / "summary.csv"), "source_sha256": sha(Path(__file__))})
    print(f"{counts} independently reconstructed operation checkpoints. {len(summary)} summary rows.")


if __name__ == "__main__":
    main()
