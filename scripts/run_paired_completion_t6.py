"""Compare shared and separate F1 difference ranges on T5 recorded inventories."""
from itertools import combinations
import json
import shutil
import time

import numpy as np
import pandas as pd

from run_observation_replication_t5 import (
    CONFIG, DATA, ROOT, OUT as T5, freeze, load_inventory,
    load_npz, now, prepare_metadata, save, score_set, sha,
)
from geolifeclef.paired_completion import shared_f1_difference_bounds
from geolifeclef.taxonomic_censoring import censor_genera
from validate_taxonomic_verification_t2 import dense_bounds

OUT = ROOT / "results/paired_completion_t6"
DOSE = .2
MARGIN = .001


def direction(lower, upper):
    return int(lower > MARGIN) - int(upper < -MARGIN)


def main():
    freeze("scores")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "freeze.json").exists():
        raise FileExistsError("Preserve the previously locked T6 run")
    source_paths = [
        "docs/paired_completion_t6_protocol.md", "scripts/run_paired_completion_t6.py",
        "src/geolifeclef/paired_completion.py", "tests/test_paired_completion.py",
    ]
    inputs = [
        T5 / "scores_freeze.json", T5 / "model_ordering.csv", T5 / "scores_complete.json",
        DATA / "fixed_predictions.npz", ROOT / "scripts/validate_taxonomic_verification_t2.py",
    ]
    identity = {str(p.relative_to(ROOT)): sha(p) for p in [*(ROOT / p for p in source_paths), *inputs]}
    for path in source_paths:
        copy = OUT / "source_snapshot" / path
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, copy)
    save(OUT / "freeze.json", {"at_utc": now(), "identity": identity,
         "dose": DOSE, "margin": MARGIN, "historically_consumed": True})
    started = time.monotonic()
    roles, assessment, cohorts, _, _ = prepare_metadata()
    _, contract, inventory = load_inventory(roles, assessment)
    eligible = contract["eligible"]
    truth = (inventory > 0) & eligible
    predictions = {name: np.unpackbits(bits, axis=1, count=eligible.shape[1]).astype(bool)
                   for name, bits in load_npz(DATA / "fixed_predictions.npz").items()}
    scores = {name: score_set(pred, truth, eligible, 1, False)[0]
              for name, pred in predictions.items()}
    t5 = pd.read_csv(T5 / "model_ordering.csv")
    t5 = t5[t5.dose.eq(DOSE) & t5.empty_value.eq(1) & t5.treatment.eq("known_only")]
    t5 = t5.set_index(["source", "mechanism", "seed", "first_model", "second_model"])
    rows = []
    mechanisms = [("uniform", np.ones_like(contract["genus_weights"])),
                  ("rarity_weighted", contract["genus_weights"])]
    for mechanism, weights in mechanisms:
        for seed in CONFIG["censor_seeds"]:
            known, _, _ = censor_genera(truth, eligible, contract["genus_codes"], weights,
                                       roles.cell_id, DOSE, seed)
            positive = truth & known
            single = {model: dense_bounds(eligible, positive, known, pred)
                      for model, pred in predictions.items()}
            for first, second in combinations(predictions, 2):
                low_a, high_a = single[first]
                low_b, high_b = single[second]
                separate_lo, separate_hi = low_a - high_b, high_a - low_b
                lower, upper = shared_f1_difference_bounds(
                    predictions[first], predictions[second], truth, known, eligible,
                )
                oracle = scores[first] - scores[second]
                assert np.all(separate_lo - 1e-12 <= lower)
                assert np.all(lower - 1e-12 <= oracle)
                assert np.all(oracle <= upper + 1e-12)
                assert np.all(upper <= separate_hi + 1e-12)
                for source, take in cohorts.items():
                    a, b = float(lower[take].mean()), float(upper[take].mean())
                    c, d = float(separate_lo[take].mean()), float(separate_hi[take].mean())
                    result = t5.loc[(source, mechanism, seed, first, second)]
                    oracle_gap = float(oracle[take].mean())
                    np.testing.assert_allclose(oracle_gap, result.oracle_gap, atol=1e-14, rtol=0)
                    rows.append(dict(source=source, mechanism=mechanism, seed=seed,
                                     first_model=first, second_model=second, cells=int(take.sum()),
                                     oracle_gap=oracle_gap, known_only_gap=result.observed_gap,
                                     shared_lower=a, shared_upper=b, shared_width=b-a,
                                     separate_lower=c, separate_upper=d, separate_width=d-c,
                                     shared_sign=direction(a, b), separate_sign=direction(c, d),
                                     known_only_sign=direction(result.observed_gap, result.observed_gap),
                                     known_only_reversal=bool(result.reversal)))
    frame = pd.DataFrame(rows)
    assert len(frame) == len(t5) == 240
    frame.to_csv(OUT / "endpoints.csv", index=False)
    save(OUT / "summary.json", {
        "at_utc": now(), "seconds": time.monotonic() - started,
        "case_count": len(frame), "recorded_oracle_exclusions": 0,
        "mean_shared_width": float(frame.shared_width.mean()),
        "mean_separate_width": float(frame.separate_width.mean()),
        "relative_mean_width_reduction": float(1 - frame.shared_width.mean()/frame.separate_width.mean()),
        "shared_sign_identified": int(frame.shared_sign.ne(0).sum()),
        "separate_sign_identified": int(frame.separate_sign.ne(0).sum()),
        "T5_reversals_included": int(frame.known_only_reversal.sum()),
        "T5_reversals_with_both_signs_feasible": int((frame.known_only_reversal
            & frame.shared_lower.lt(0) & frame.shared_upper.gt(0)).sum()),
        "new_fits": 0, "fresh_confirmation": False,
    })


if __name__ == "__main__":
    main()
