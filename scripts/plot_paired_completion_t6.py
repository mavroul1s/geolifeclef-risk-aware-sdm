"""Illustrate the T6 shared bounds for the VegBank MLP–DeepSDM reversal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_paired_completion_t6 import OUT


def main():
    results = pd.read_csv(OUT / "endpoints.csv")
    data = results[
        results.source.eq("NA-US-002") & results.mechanism.eq("rarity_weighted") &
        results.first_model.eq("aggregation_s0") & results.second_model.eq("optimized_deepsdm_s0")
    ].sort_values("seed")
    assert len(data) == 10
    labels = np.arange(1, len(data)+1)
    fig, (bounds, detail) = plt.subplots(ncols=2, sharey=True, figsize=(10.5, 4.4), layout="constrained")
    for axes in (bounds, detail):
        axes.axvline(0, color="#777777", lw=1)
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="x", color="#e6e6e6")
        axes.set_axisbelow(True)
    bounds.hlines(labels, data.separate_lower*100, data.separate_upper*100, color="#b8bbbc",
                  lw=3, label="Separate score ranges")
    bounds.hlines(labels, data.shared_lower*100, data.shared_upper*100, color="#1a6684",
                  lw=3, label="Shared inventory")
    bounds.set_title("A   F1 contrasts allowed by unresolved labels", loc="left", fontsize=11,
                     fontweight="bold")
    bounds.set_ylabel("Censoring mask")
    bounds.set_yticks(labels)
    bounds.invert_yaxis()
    bounds.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2)
    detail.scatter(data.oracle_gap*100, labels, marker="D", s=21, color="#323232",
                   label="Recorded inventory")
    detail.scatter(data.known_only_gap*100, labels, s=29, color="#b5572c",
                   label="Known-only score")
    detail.set_title("B   Point estimates on a finer scale", loc="left", fontsize=11, fontweight="bold")
    detail.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2)
    for axes in (bounds, detail):
        axes.set_xlabel("MLP minus DeepSDM mean F1 (points)")
    fig.savefig(OUT / "vegbank_ordering_under_missing_labels.png", dpi=230)


if __name__ == "__main__":
    main()
