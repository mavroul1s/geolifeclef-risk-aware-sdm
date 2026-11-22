"""Summarize frozen T5 endpoints and plot score and verification sensitivities."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_observation_replication_t5 import CONFIG, OUT, ROOT, freeze, load_npz, DATA, sha


def validate_and_summarize():
    for phase in ("scores", "verification"):
        freeze(phase)
        lock = json.loads((OUT / f"{phase}_freeze.json").read_text())
        access = json.loads((OUT / f"{phase}_access.json").read_text())
        assert access["at_utc"] > lock["at_utc"]
        assert access["lock_sha256"] == sha(OUT / f"{phase}_freeze.json")

    scores = pd.read_csv(OUT / "score_masks.csv")
    ordering = pd.read_csv(OUT / "model_ordering.csv")
    assert len(scores) == len(ordering) == 2880
    assert scores[scores.dose.eq(0)][["mean_bias", "policy_gap_error", "false_perfect"]].eq(0).all().all()
    assert not ordering.loc[ordering.dose.eq(0), "reversal"].any()

    events = pd.read_csv(OUT / "verification_endpoints.csv")
    shared = events[events.unit.eq("audited_shared")]
    local = events[events.unit.eq("local")]
    assert len(shared) == 576 and len(local) == 48
    assert int(shared.reprice.sum()) == 96
    assert events.width_groups.le(events.executed_groups).all()
    assert events.width_upper.sub(events.width_lower).le(CONFIG["width_tolerance"] + 1e-11).all()
    for kind in ("width", "sign"):
        valid = events[f"{kind}_groups"].notna()
        np.testing.assert_allclose(events.loc[valid, f"{kind}_cost"],
            events.loc[valid, f"{kind}_resolutions"] + events.loc[valid, "price"]
            * events.loc[valid, f"{kind}_audits"], atol=1e-10, rtol=0)
    assert local.width_audits.eq(0).all()
    assert len(load_npz(DATA / "verification_orders.npz")) == 528
    assert events.loc[~events.sign_identified, "oracle_gap"].abs().le(CONFIG["practical_margin"]).all()

    old = pd.read_csv(ROOT / "results/adaptive_baselines_t4/query_events.csv")
    replayed = shared[shared.source.eq("all") & shared.model.eq(CONFIG["models"][0])]
    matches = replayed.merge(old, left_on=["seed", "error_rate", "price", "policy"],
                             right_on=["seed", "error_rate", "audit_price", "policy"])
    assert len(matches) == 96
    np.testing.assert_allclose(matches.sign_cost_x, matches.sign_cost_y, rtol=0, atol=1e-9)
    np.testing.assert_allclose(matches.width_cost, matches.tolerance_cost, rtol=0, atol=1e-9)

    keys = ["source", "model", "seed"]
    local_best = local.groupby(keys).width_cost.min().rename("local_best_width_cost").reset_index()
    table = shared.merge(local_best, on=keys, validate="many_to_one")
    table["relative_saving"] = 1 - table.width_cost / table.local_best_width_cost
    table["break_even_price"] = (table.local_best_width_cost - table.width_resolutions) / table.width_audits
    table["share_cheaper"] = table.width_cost < table.local_best_width_cost
    summary = table.groupby(["source", "model", "error_rate", "price", "policy"], as_index=False).agg(
        shared_mean=("width_cost", "mean"), local_mean=("local_best_width_cost", "mean"),
        saving_mean=("relative_saving", "mean"), masks_cheaper=("share_cheaper", "sum"),
        masks=("seed", "size"), median_break_even_price=("break_even_price", "median"),
        mean_rare_labels=("width_recovered_rare_labels", "mean"),
        mean_affected_cells=("width_affected_cells", "mean"),
        sign_identified=("sign_identified", "sum"),
    )
    summary.to_csv(OUT / "verification_summary.csv", index=False)
    return summary


def plot(summary):
    intervals = pd.read_csv(OUT / "score_intervals.csv")
    primary = intervals[intervals.dose.eq(.2) & intervals.empty_value.eq(1) &
                        intervals.treatment.eq("known_only")]
    groups = [
        ("all", "aggregation_s0", "All cells · MLP"),
        ("all", "optimized_deepsdm_s0", "All cells · DeepSDM"),
        ("all", "randomforest_s0", "All cells · Random forest"),
        ("AU-AU-002", "aggregation_s0", "AEKOS · MLP"),
        ("NA-US-002", "aggregation_s0", "VegBank · MLP"),
        ("AS-RU-002", "aggregation_s0", "Siberia · MLP"),
    ]
    fig, (ax, cost) = plt.subplots(ncols=2, sharey=True, figsize=(13, 5.1), layout="constrained")
    palette = ("#1a6684", "#b5572c")
    y = np.arange(len(groups))
    for j, (mechanism, name) in enumerate([("uniform", "Uniform"), ("rarity_weighted", "Rarity-weighted")]):
        data = primary[primary.mechanism.eq(mechanism)].set_index(["source", "model"])
        rows = data.loc[[(source, model) for source, model, _ in groups]]
        point = rows.mean_bias.to_numpy() * 100
        ax.errorbar(point, y + (j-.5)*.22, xerr=[point - rows.bias_lo*100, rows.bias_hi*100 - point],
                    fmt="o", markersize=5, color=palette[j], capsize=3, label=name)
    ax.axvline(0, lw=1, color="#474747")
    ax.set_yticks(y, labels=[item[2] for item in groups])
    ax.invert_yaxis()
    ax.set_title("A   Score bias after censoring", loc="left", fontweight="bold")
    ax.set_xlabel("Known-only F1 minus recorded-inventory F1 (points)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False)
    ax.grid(axis="x", color="#e6e6e6")
    chosen = summary[summary.error_rate.eq(.05) & summary.policy.eq("static_scope_width")]
    for j, price in enumerate(CONFIG["audit_prices"]):
        data = chosen[chosen.price.eq(price)].set_index(["source", "model"])
        rows = data.loc[[(source, model) for source, model, _ in groups]]
        ratio = rows.shared_mean / rows.local_mean
        cost.scatter(ratio, y + (j-.5)*.22, color=palette[j], s=35, label=f"Audit price {price:.1f}")
    cost.axvline(1, lw=1, color="#474747")
    cost.set_xlabel("Mean audited shared cost / mean local cost")
    cost.set_title("B   Verification under 5% nominal link error", loc="left", fontweight="bold")
    cost.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False)
    cost.grid(axis="x", color="#e6e6e6")
    for axes in (ax, cost):
        axes.set_axisbelow(True)
        axes.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "observation_and_verification.png", dpi=230)


if __name__ == "__main__":
    plot(validate_and_summarize())
