"""Scope-matched verification cost and validity under fixed SAGE predictions."""
from collections import Counter
import hashlib
import time

import numpy as np
import pandas as pd

from run_observation_replication_t5 import (
    CONFIG, DATA, OUT, load_inventory, load_npz, now, prepare_metadata,
    save, score_set, sha, start,
)
from geolifeclef.adaptive_verification import POLICIES, ScopeCredit, choose_batch, priorities
from geolifeclef.linkage_verification import (
    IdentityOracle, comparison_members, corrupt_memberships, record_keys,
    resolve_group, verification_cost,
)
from geolifeclef.taxonomic_queries import (
    QueryState, contrast_interval, simulate_keys,
)
from validate_linkage_robustness_t3 import dense, dense_bounds, operations, scope_from_initial

WIDTH_EPSILON = 1e-12
CHECKPOINTS = (0, .25, .5, .75, 1)
REUSABLE = ("static_scope_width", "adaptive_scope_width")


def case_arrays(keys, offsets, answers, eligible, initial, genera, weights):
    # Scope depends only on observable information. Any eligible-shaped prediction
    # can be used here because scope_from_initial never conditions on predictions.
    public_state = QueryState(eligible, initial, genera, np.zeros_like(eligible), keys)
    support = comparison_members(eligible, public_state.known, genera, weights, keys)
    case = dict(genus=keys.genus, offsets=offsets, cells=np.concatenate(keys.cells),
                private_answers=answers,
                support=np.concatenate([np.isin(cells, s) for cells, s in zip(keys.cells, support)]))
    return case, support, public_state.known


def records(keys, answers):
    return Counter((int(cell), int(genus), int(answer))
                   for genus, cells, answer in zip(keys.genus, keys.cells, answers) for cell in cells)


def expanded_records(case):
    return Counter(zip(case["cells"], np.repeat(case["genus"], np.diff(case["offsets"])),
                       case["private_answers"]))


def practical(lower, upper):
    margin = CONFIG["practical_margin"]
    return int(lower > margin) - int(upper < -margin)


def endpoint_reached(lower, upper, kind):
    if kind == "width":
        return upper - lower <= CONFIG["width_tolerance"] + WIDTH_EPSILON
    return bool(practical(lower, upper))


def checked_range(eligible, initial, genera, prediction, weights, truth, quality, case, order, n):
    positive, known, affected = dense(case, order, "audit_support", n, eligible, initial, genera)
    low, high = dense_bounds(eligible, positive, known, prediction)
    assert not np.any(known & (positive != truth))
    assert np.all(low <= quality + 1e-12) and np.all(quality <= high + 1e-12)
    return contrast_interval(low, high, weights), affected


def execute(keys, case, scope, eligible, initial, genera, prediction, weights, truth, rare,
            quality, policy, price, seed, started):
    state = QueryState(eligible, initial, genera, prediction, record_keys(keys))
    graph = ScopeCredit(state, keys, case["offsets"], case["support"])
    oracle = IdentityOracle(case["private_answers"])
    queried = np.zeros(len(keys.cells), bool)
    affected = np.zeros(len(eligible), bool)
    tie = np.random.default_rng(seed + 101).random(len(keys.cells))
    true_gap = float(weights @ quality)
    order = []
    endpoints = {}
    lower, upper = contrast_interval(*state.bounds(), weights)

    def capture(kind):
        low, high = state.bounds()
        np.testing.assert_allclose([lower, upper], contrast_interval(low, high, weights),
                                   rtol=0, atol=1e-10)
        assert not np.any(state.known & (state.positive != truth))
        assert np.all(low <= quality + 1e-12) and np.all(quality <= high + 1e-12)
        if kind == "sign":
            assert practical(lower, upper) * true_gap > CONFIG["practical_margin"]
        result = dict(groups=len(order), resolutions=oracle.resolutions, audits=oracle.audits,
                      lower=lower, upper=upper, affected_cells=int(affected.sum()),
                      recovered_rare_labels=int((state.positive[:, rare] & ~initial[:, rare]).sum()))
        endpoints[kind] = result

    for kind in ("width", "sign"):
        if endpoint_reached(lower, upper, kind):
            capture(kind)

    while len(endpoints) < 2:
        if time.monotonic() - started > CONFIG["maximum_seconds"]:
            raise TimeoutError("T5 verification budget expired")
        score, _, _ = priorities(graph, state, policy, price, len(order), oracle.resolutions,
                                 oracle.audits, **CONFIG["mismatch_prior"])
        batch = choose_batch(score, graph.members, queried, tie, CONFIG["batch_size"])
        if not len(batch):
            break
        for group in batch:
            rows = scope[group]
            before = contrast_interval(*state.bounds(rows), weights[rows])
            resolve_group(state, int(group), keys, case["offsets"], oracle, "audit_support", scope)
            graph.observe(state, rows, int(keys.genus[group]))
            queried[group] = True
            affected[rows] = True
            order.append(int(group))
            after = contrast_interval(*state.bounds(rows), weights[rows])
            assert after[0] >= before[0] - 1e-12 and after[1] <= before[1] + 1e-12
            lower += after[0] - before[0]
            upper += after[1] - before[1]
            for kind in ("width", "sign"):
                if kind not in endpoints and endpoint_reached(lower, upper, kind):
                    capture(kind)
            if len(endpoints) == 2:
                break

    assert "width" in endpoints
    if "sign" not in endpoints:
        assert abs(true_gap) <= CONFIG["practical_margin"] + 1e-10
        np.testing.assert_allclose([lower, upper], [true_gap, true_gap], rtol=0, atol=1e-10)
    validation = dict(operation_checks=0, dense_prefixes=0)
    for kind, point in endpoints.items():
        n = point["groups"]
        independent = operations(case, order, "audit_support", n)
        assert independent == (point["resolutions"], point["audits"])
        validation["operation_checks"] += 1
        if seed == CONFIG["mask_seeds"][0]:
            for prefix in sorted({n, max(0, n-1)}):
                interval, visited = checked_range(eligible, initial, genera, prediction, weights,
                                                  truth, quality, case, order, prefix)
                assert endpoint_reached(*interval, kind) == (prefix == n)
                if prefix == n:
                    assert int(visited.sum()) == point["affected_cells"]
                    np.testing.assert_allclose(interval, [point["lower"], point["upper"]],
                                               atol=1e-10, rtol=0)
                validation["dense_prefixes"] += 1
    return order, endpoints, validation


def unchecked_checkpoints(keys, case, scope, eligible, initial, genera, prediction, weights,
                          truth, quality, seed):
    state = QueryState(eligible, initial, genera, prediction, record_keys(keys))
    graph = ScopeCredit(state, keys, case["offsets"], case["support"])
    tie = np.random.default_rng(seed+101).random(len(keys.cells))
    order = choose_batch(graph.initial_credit, graph.members, state.queried[:len(keys.cells)],
                         tie, len(keys.cells))
    answer = case["private_answers"].copy()
    for group in order:
        start, end = case["offsets"][group:group+2]
        members = np.flatnonzero(case["support"][start:end]) + start
        rep = members[np.argmin(case["cells"][members])]
        answer[members] = answer[rep]
    unchecked = {**case, "private_answers": answer}
    oracle = float(weights @ quality)
    for fraction in CHECKPOINTS:
        n = int(np.ceil(fraction * len(order)))
        positive, known, _ = dense(unchecked, order, "audit_support", n, eligible, initial, genera)
        lo, hi = dense_bounds(eligible, positive, known, prediction)
        lower, upper = contrast_interval(lo, hi, weights)
        direction = practical(lower, upper)
        yield dict(fraction=fraction, groups=n, lower=lower, upper=upper, oracle_gap=oracle,
                   contained=lower - 1e-10 <= oracle <= upper + 1e-10,
                   incorrect_known_labels=int((known & (positive != truth)).sum()),
                   wrong_practical_sign=bool(direction and direction * oracle <= CONFIG["practical_margin"]))


def event_row(fields, unit, policy, price, result, reprice=False):
    order, endpoints, _ = result
    row = {**fields, "unit": unit, "policy": policy, "price": price, "reprice": reprice,
           "executed_groups": len(order),
           "order_sha256": hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest(),
           "sign_identified": "sign" in endpoints}
    for kind, point in endpoints.items():
        row.update({f"{kind}_{key}": value for key, value in point.items()})
        row[f"{kind}_cost"] = verification_cost(point["resolutions"], point["audits"], price)
    return row


def verify_study():
    start("verification")
    started = time.monotonic()
    roles, assessment, cohorts, policies, _ = prepare_metadata()
    all_native, contract, inventory = load_inventory(roles, assessment)
    prediction_bits = load_npz(DATA / "fixed_predictions.npz")
    events, unchecked, receipts, admissions = [], [], [], []
    checks = Counter()
    paths = {}
    models = CONFIG["models"]
    for source, take in cohorts.items():
        census = roles.loc[take]
        a, b = policies[source]
        weights = a / a.sum() - b / b.sum()
        support_n = int((a ^ b).sum())
        admitted = support_n >= CONFIG["minimum_disagreement_cells"]
        admissions.append(dict(source=source, cells=len(census), clusters=census.spatial_block.nunique(),
                               accepted=int(a.sum()), disagreement_cells=support_n, admitted=admitted))
        if not admitted:
            continue
        native = all_native[take]
        eligible = contract["eligible"][take]
        genera, rare = contract["genus_codes"], contract["rare"]
        full = inventory[take]
        truth = (full > 0) & eligible
        predictions = {}
        for model in models if source == "all" else models[:1]:
            pred = np.unpackbits(prediction_bits[model][take], axis=1,
                                 count=eligible.shape[1]).astype(bool)
            quality = score_set(pred, truth, eligible, 1, False)[0]
            predictions[model] = pred, quality
        blocks = census.spatial_block.to_numpy()
        for seed in CONFIG["mask_seeds"]:
            initial, problems = simulate_keys(full, native, contract["universe"], genera, blocks,
                                              seed, CONFIG["hidden_probability"])
            assert not np.any(initial & ~truth)
            local_keys, local_answers = problems["cell_local"]
            local_offsets = np.arange(len(local_keys.cells)+1)
            local, local_scope, known = case_arrays(local_keys, local_offsets, local_answers,
                                                   eligible, initial, genera, weights)
            original_keys, original_answers = problems["block_shared"]
            original_records = records(original_keys, original_answers)
            assert original_records == expanded_records(local)
            if seed == CONFIG["mask_seeds"][0]:
                scope_from_initial(local, eligible, initial, genera, weights)
            shared_cases = []
            for error in CONFIG["error_rates"]:
                keys, answers, offsets, receipt = corrupt_memberships(
                    original_keys, original_answers, blocks, error,
                    seed + CONFIG["corruption_seed_offset"], max(CONFIG["error_rates"]),
                )
                case, scope, start_known = case_arrays(keys, offsets, answers, eligible, initial, genera, weights)
                assert original_records == expanded_records(case)
                np.testing.assert_array_equal(start_known, known)
                if seed == CONFIG["mask_seeds"][0]:
                    scope_from_initial(case, eligible, initial, genera, weights)
                receipts.append(dict(source=source, seed=seed, **receipt))
                shared_cases.append((error, keys, case, scope))
            for model, (prediction, quality) in predictions.items():
                fields = dict(source=source, seed=seed, model=model, oracle_gap=float(weights @ quality))
                replay_inputs = (eligible, initial, genera, prediction, weights, truth, rare, quality)
                for policy in REUSABLE:
                    result = execute(local_keys, local, local_scope, *replay_inputs,
                                     policy, 0, seed, started)
                    checks.update(result[2])
                    events.append(event_row({**fields, "error_rate": 0}, "local", policy, 0, result))
                    paths[f"{source}_{seed}_{model}_local_{policy}"] = np.asarray(result[0], dtype=np.int32)
                for error, keys, case, scope in shared_cases:
                    scenario = {**fields, "error_rate": error}
                    unchecked.extend({**scenario, **values} for values in unchecked_checkpoints(
                        keys, case, scope, eligible, initial, genera, prediction, weights, truth, quality, seed,
                    ))
                    for policy in POLICIES:
                        for price_index, price in enumerate(CONFIG["audit_prices"]):
                            reuse = bool(price_index and policy in REUSABLE)
                            if not reuse:
                                result = execute(keys, case, scope, *replay_inputs, policy, price,
                                                 seed, started)
                                checks.update(result[2])
                                tag = f"{source}_{seed}_{model}_{error}_{policy}_{price}"
                                paths[tag] = np.asarray(result[0], dtype=np.int32)
                            events.append(event_row(scenario, "audited_shared", policy, price, result, reuse))
            pd.DataFrame(events).to_csv(OUT / "verification_endpoints.csv", index=False)
            pd.DataFrame(unchecked).to_csv(OUT / "unchecked_checkpoints.csv", index=False)
            pd.DataFrame(receipts).to_csv(OUT / "corruption_receipts.csv", index=False)
            print(f"{source} seed={seed} endpoints={len(events)} seconds={time.monotonic()-started:.1f}",
                  flush=True)
    np.savez_compressed(DATA / "verification_orders.npz", **paths)
    pd.DataFrame(admissions).to_csv(OUT / "cohort_admission.csv", index=False)
    frame = pd.DataFrame(events)
    prefixes = pd.DataFrame(unchecked)
    assert len(paths) == int((~frame.reprice).sum())
    save(OUT / "verification_complete.json", {
        "at_utc": now(), "seconds": time.monotonic() - started,
        "executed_local": int(frame.unit.eq("local").sum()),
        "executed_shared": int((frame.unit.eq("audited_shared") & ~frame.reprice).sum()),
        "reprices": int(frame.reprice.sum()),
        "unchecked_paths": len(prefixes) // len(CHECKPOINTS),
        "unchecked_checkpoints": len(prefixes),
        "unchecked_contrast_exclusions": int((~prefixes.contained).sum()),
        "unchecked_wrong_practical_signs": int(prefixes.wrong_practical_sign.sum()),
        "guarded_incorrect_endpoint_labels": 0, "guarded_wrong_practical_signs": 0,
        **checks, "orders_sha256": sha(DATA / "verification_orders.npz"),
        "fresh_confirmation": False, "new_fits": 0, "new_inference": 0, "new_gpu_jobs": 0,
    })


if __name__ == "__main__":
    verify_study()
