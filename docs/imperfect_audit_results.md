# Imperfect voucher audit sensitivity — 28 November 2026

In 200 replayed trajectories, repeated compatibility review reduced the number
of paths that at some checkpoint excluded the recorded benchmark contrast.
Single review excluded it in **74/80** paths and double review in **44/80**.
Both exact-identification controls retained the benchmark at every checkpoint.
The higher-error, correlated cases illustrate the limits of a second review.
[Frozen protocol](imperfect_audit_protocol.md) ·
[Complete summary](../results/imperfect_audit_sensitivity/summary.csv).

We reused ten T3 inventories and its comparison-support query order at 1% and
5% nominal wrong-link doses. Within each matched scenario, the same group
representative and simulated decisions drive both arms. Two-review arms perform
a second check after a compatible answer. Abstention, rejection or disagreement
triggers full resolution with an ideal recorded identity. The outcome is the
fixed T3 contrast between two selectors that share one model. It is development
evidence on consumed SAGE inventories.

## Range validity and cost

At the 5% wrong-link dose, a higher-error independent second review reduces
wrongly propagated answers from **663 to 64** across the ten paths. With the
correlated reviewer draws, the reduction is **662 to 341**. The median costs
below compare the first width of at most .002 with direct resolution of the
same support in the same query-group order.

| 5% wrong links, higher reviewer error | Paths excluding benchmark at any checkpoint | Cost ratio at audit price .1 | Cost ratio at audit price .5 |
|---|---:|---:|---:|
| One review, independent | 10/10 | .464 | .788 |
| Two reviews, independent | 8/10 | .626 | 1.198 |
| One review, correlated | 10/10 | .461 | .787 |
| Two reviews, correlated | 10/10 | .574 | 1.147 |

All 200 first precision endpoints and all 200 first practical-sign endpoints
contained the recorded contrast. Practical signs were correct at all recorded
checkpoints where a sign was declared. The invalid intervals emerged at later
checkpoints as additional noisy decisions narrowed the ranges. At the higher
audit price, double review cost 6.6–19.8% more than matched local resolution
across the eight noisy strata. At the lower price its costs were 37.4–55.2%
lower. These ratios use normalized operation prices. Actual botanist time,
error rates and the best separately optimized local schedule remain open.

## Validation and reproduction

The run generated 1,200 checkpoints and 400 first-endpoint records in 236
seconds on CPU. The ideal arm reproduces all 40 matched T3 prefixes and charges. The
validators reconstruct 200 dense prefixes and 160 audit-operation checkpoints
independently, and verify the price formula for all 1,600 recorded rows. The
original T3 and T5/T6 artifacts retain their frozen hashes. Replay uses fixed
predictions and existing labels. [Validation receipt](../results/imperfect_audit_sensitivity/validation.json).

With the T3 inputs restored in the local ignored data directories, reproduce
the study in a clean results location with the source and inputs named in the
[freeze receipt](../results/imperfect_audit_sensitivity/freeze.json).

```bash
export PYTHONPATH=src:scripts
.venv/bin/python scripts/run_imperfect_audit_sensitivity.py
.venv/bin/python scripts/summarize_imperfect_audit_sensitivity.py
```
