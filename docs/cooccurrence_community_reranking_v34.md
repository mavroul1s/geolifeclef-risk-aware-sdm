# v34: species-community reranking

## Why this candidate

The current scored best remains v32: **0.24225 public / 0.21514 private**.
V33 completed in 5.0072 hours but selected its control; the submitted CSV was
byte-identical to v32. All 24 alternatives failed calibration selection. The new
convolution member's fixed-four-swap gains were -0.005215/-0.005274 on the repeated
assessment folds. We do not lower the gate or add more such models.

Before implementing v34, two inexpensive hypotheses were inspected locally:

1. A histogram-boosted single-edit-F1 regressor, with bounded cardinality edits.
   Its pilot separated survey IDs and 1-degree blocks with a 20-km buffer.
   None of its alternatives had positive pooled/country-macro/outside-core gains
   together. It is **rejected**, not included in the notebook. Its code and full
   results remain in `scripts/pilot_v34_editor*` and
   `results/v34_exploratory_pilot.json`.
2. A shrunk co-occurrence correction showed a small positive signal. The strongest
   exploratory policy (alpha .25, residual weight 2, four swaps) gained .0004093
   per deduplicated survey, with both fold-view gains positive and outside-DK/NL
   gain .000668. `results/v34_cooccurrence_pilot.json` records all 24 candidates,
   including the negative ones. These observations are **consumed calibration**,
   not independent validation, and do not lock the notebook's selected policy.

`scripts/replay_v34_archived_predictions.py` checks the released implementation
against the same stored predictions, using stable sorting and batched inference.
Its results in `results/v34_archived_replay.json` supersede the approximate pilot
for implementation measurements, not for claims of generalization.

## Model and fixed search

Refit the unchanged eleven-member v32 development pipeline in each existing
buffered fold. Average the three seed-bag calibrated probabilities and the two
asymmetric-specialist calibrated probabilities separately, retaining their top128.
No additional neural architecture, seed, loss, epoch or dataset is introduced.

For graph-training surveys only, compute binary-label pair counts C(i,j), counts
c(i), and smoothed marginal q(j)=(c(j)+.5)/(n+1). Then:

```
A(i,j) = clip(log((C(i,j)+100*q(j))/(c(i)+100)/q(j)), -2, 2)
A(i,i) = 0
p(j) = 1e-7 + .5*bag_top128(j) + .5*specialist_top128(j)
context(j) = sum_i normalized_p(i)*A(i,j)
ranking_score(j) = log(p(j)) + alpha*context(j)
```

The anchors i are the first min(12, ceil(.6*base_count)) species of the matched
v32 list. These are predictions, never the query's true labels. No species is
removed from the vocabulary or graph based on rarity. Missing top128 probabilities
are truncated evidence, not an assertion of biological absence.

Use the frozen residual decoder to protect the leading 60% and **every row's
species count**. Calibration selects one of 17 fixed policies: control, or alpha
0/.25/.5/1, residual weight 1/2, maximum 2/4 swaps. Alpha zero controls for
probability reranking without label-dependency information. Strong graph weights
were not automatically better in the pilot. All tied rankings use deterministic
species-column order.

Label dependencies are an established multi-label modeling approach; see
[Chen et al., CVPR 2019](https://arxiv.org/abs/1904.03582). Our lightweight shrunk
statistical reranker is **not** their GCN architecture or a novelty/SOTA claim.
Co-occurrence need not represent causal ecological interaction: geography,
survey size, sampling and observation bias are serious confounders.

## Separation and production

Training labels fit neural models and graph. Selection labels calibrate neural
scores. Calibration selects the fixed policy. Repeated assessment evaluates it
after freeze; its no-graph ablation cannot switch policies. Require positive gains
in both folds, positive country-macro and outside-Denmark/Netherlands gains, and a
positive block-bootstrap lower bound, with the unchanged 20-km buffer integrity
checks. All **88,987 PA surveys have previously been assessed**; repeated gates and
bootstrap intervals are not fresh evidence after adaptive experimentation.

Production refits the same five v32 models, same seeds/epochs, on all PA surveys.
The graph uses all PA labels, never test labels. Use the exact numerical
calibration coefficients from scored v32 production, not a new production holdout
that discards rows. The base submission is the embedded byte-exact v32 CSV; model
weights are refitted and are not claimed to be byte-exact. A wrong predicted anchor
set may propagate mistakes; bounded swaps reduce but cannot eliminate that risk.

## Operational contract

- One offline notebook, less than 1,000,000 bytes; official competition input only.
- T4 x1 required, checked before expensive feature loading.
- 22 development and at most 5 production neural fits, same counts as v32's
  measured 5.7681-hour T4 run. Graph overhead is extra; full v34 time is unmeasured.
- Cooperative 10.75-hour runtime guard; remaining-plan admission uses the whole
  first fold with a 1.25x margin and production training speeds with a 1.5x margin,
  plus overhead/reserves. This cannot guarantee against stalled I/O/hardware.
- Exactly five final files, at most 16 MB: submission **or NO_SUBMISSION.json**,
  report, manifest, per-survey diagnostic CSV, sampled six-expert NPZ. The NPZ's
  `score` is not necessarily a probability; it includes graph ranking scores.
- Rejected or unchanged candidates export **no submission-shaped CSV**. This
  prevents repeating the accidental v33 duplicate submission. Diagnostics CSVs
  are never competition submissions.
- Failure revokes eligibility, renames an already published CSV to a non-CSV
  failure artifact, invalidates the manifest, and clears temporary checkpoints.

Run: GPU T4 x1, attach `geolifeclef-2025`, restart and Run All. Submit only
`v34_export/GLC25_PA_submission_v34.csv` when `eligible_for_submission` is true.
Keep v32 until an actual hidden-test result exceeds it. Winning is not guaranteed.
