# v31 — full-data bagging with habitat residuals

Frozen candidate specification, 2026-09-26. **Not a demonstrated SOTA result.**

## Evidence motivating the change

V30 scored 0.23451 public / 0.20942 private, below the scored v29 control
(0.23900 / 0.21093). The user's screenshot and all five original output files are
preserved, with hashes, in `results/v30_summary.json`. V29 remains the official best.

V30's repeated development gain was +0.003523, but outside Denmark/Netherlands it
was -0.000744 over 8,653 surveys; equal-country mean gain among countries with >=30
surveys was -0.000386. Its production training set fell from 88,987 to 72,063 PA
surveys and its mean submitted cardinality fell from 26.863 to 23.303. These are
observations, **not causal identification**: architecture, epochs and calibration
also changed simultaneously. The new geographic gates below respond to this result
and are consequently development choices, not an independent audit.

## Method and locked choices

The exact scored v29 CSV (SHA256
`9ec7afb25eb1651b161b0f8c6ecaa64fd553d17dd24b4808b1f07446b85f8194`) is embedded in
the notebook. Every test row retains its count and the leading ceil(60%) species.
At most 2 or 4 unprotected species can be replaced; no cardinality tuning is done.

1. Refit the three original v29 architectures: temporal attention, multisensor CNN,
   and ecology attention. Retain original losses, input features, country-balanced
   sampling, EMA and two-view inference. Epochs are fixed at 12/15/18, the scored
   v29 production schedule, for **both** development groups and final training.
2. Compare original-seed and independent-seed groups on the same training rows.
   Reference seeds are original+1000*fold; replica seeds add 31000. Development
   calibration is fitted on each model's selection partition, not its calibration
   or regression labels. The reference is a matched fixed-epoch recipe refit,
   not reconstructed original model weights.
3. Fit an environmental analogue expert using only the training rows' official
   environmental descriptors and missingness indicators. Train-only mean/std,
   clipping to +/-6, PCA up to 32 components, and partial whitening by variance^0.25
   define the metric. No coordinates, IDs or countries enter this expert's metric.
   It blends locally weighted 32- and 128-neighbour label distributions 50/50.
   Exact batched neighbour search and sparse label multiplication bound memory.
4. Combine ranked evidence with reciprocal ranks 1/(10+rank). Original-list weight
   is 1; neural weights are 0.5/1 and habitat weights 0/0.25/0.5. Also test
   habitat-only weight 1. Cross these seven expert settings with 2/4 swaps, plus
   the unchanged control: **15 policies total**. Zero-probability expert entries
   cannot nominate species. Strict score improvement is required for a swap.

These are rank-list residuals: the new models supplement the information retained
in v29's lists, not a probability ensemble with unavailable original test logits.
This bounded strategy may be too conservative for a large improvement. It is not
a claim to reproduce a winning system. The competition
[working note](https://ceur-ws.org/Vol-4038/paper_261.pdf) is background on multimodal
prediction and geographic transfer, not evidence for the efficacy of this implementation.

## Repeated spatial development, not a fresh holdout

Reuse v30's label-blind one-degree block allocation and 20km training buffers.
There are no split retries or new claims of unused labels. All 88,987 PA survey
IDs have already been assessed in prior experiments.

| Role | Fold 0 | Fold 1 |
|---|---:|---:|
| Training | 35,811 | 37,487 |
| Selection / probability calibration | 8,816 | 8,816 |
| Ranking-policy calibration | 8,691 | 8,691 |
| Repeated regression | 15,792 | 15,778 |

Selection, calibration, regression and training use disjoint blocks within each
fold. Regression survey IDs do not overlap across folds. The same calibration
IDs occur in both folds; country-support counts deduplicate those IDs instead of
pretending they are twice as many independent observations.

Calibration policy eligibility requires positive gain in both folds, positive
pooled gain, positive country-macro gain among countries with >=30 unique surveys,
and positive gain outside Denmark/Netherlands with >=30 unique surveys. Eligible
policies maximize 0.50*pooled + 0.25*country-macro + 0.25*outside-core gain; ties
prefer fewer swaps and lighter expert weights. Control is always available.

After freezing the policy, require those same positive pooled/country/outside
gains, positive gain in each regression fold, and positive lower 95% paired
one-degree-block bootstrap bound. Assessment ablations remove each expert at the
**already frozen** settings and are descriptive only; they do not select a new
policy. Repeated checks and interval estimates are potentially optimistic because
of prior development. Do not use them as hidden-test score forecasts or SOTA proof.

## Production

If the gate passes, fit the needed experts on **all 88,987 PA rows**. There is no
production-anchor exclusion. Neural replicas use original seeds+31000 and the
same fixed 12/15/18 epochs. The three Platt coefficient pairs are frozen from the
scored v29 report, checked by tests, and transferred across seeds. This transfer
is an explicit limitation; it avoids the v30 full-data exclusion tradeoff but is
not production-specific held-out calibration. The residual decoder never changes
the exact v29 row counts. Habitat-only selection skips the three neural refits.

A failed development gate returns the exact v29 CSV marked
`unchanged_v29_DO_NOT_SUBMIT.csv`. Even with a passed gate, an unchanged CSV is not
eligible. An exception after prediction export removes any ready-to-submit filename.

## Single-notebook delivery and runtime

`notebooks/geolifeclef_v31_full_data_bagged_habitat_residual.ipynb` includes verified
frozen v27 readers, v29 models, v30 split code, new v31 code and exact v29 lists.
It is below 1,000,000 bytes, requires only the official `geolifeclef-2025` input,
one Kaggle GPU (T4 x1), and no Internet or external weights. Missing GPU and invalid
data/splits are checked before expensive feature preparation.

There are 12 fixed-epoch development fits and at most three full-data production
fits. V30 took 3.799h on a T4. V31's fixed development total is 180 epochs across
two folds versus v30's 174 actual epochs, but model mix, additional inference and
larger production fits differ: this is **not a measured runtime prediction**.
Full v31 runtime has not been measured. A 10.75h cooperative budget checks training,
inference and residual decoding; whole-production admission uses the slowest
per-row development epoch estimate per replica, scales to all PA rows, applies
1.4x safety and adds 45 minutes. It can stop safely on a slow session. It is not
an operating-system hard timeout or a guarantee of completion within 12 hours.

Successful output is exactly five files, <=16MB total:

- `GLC25_PA_submission_v31.csv` (or a clearly named non-submittable control).
- `regression_per_survey_v31.csv`.
- `v31_report.json` with all gates, recipes, histories and limitations.
- `v31_manifest.json` with source hashes, splits and output hashes.
- `calibration_top128_v31.npz`: at most 1,500 geographically reweighted samples
  per fold, two experts' top128 ranks/probabilities, reference lists and sparse
  truth. These are truncated calibration diagnostics, not fresh evidence.

Only submit `GLC25_PA_submission_v31.csv` when `eligible_for_submission` is true.
Do not submit the notebook as predictions, ZIP, NPZ, JSON, diagnostics or any file
containing `DO_NOT_SUBMIT`. Temporary feature caches and checkpoints are deleted
only within the version-specific runtime directory. The original scored versions
and downloaded archive remain untouched.

## Verification

See `results/v31_local_validation.json`. Tests cover the standalone notebook,
source hashes, exact v29 byte round trip, train-only habitat fitting, independent
manual nearest-neighbour averaging, 128-neighbour/5,016-species shapes, constant
and missing inputs, residual invariants, repeated-ID support counting, actual v30
geographic-gate rejection, GPU and budget preflight, and miniature real CPU fits
through all three full-data refits and compact export. Miniature statistical gates
are controlled fixtures, not evidence of model quality. Full GPU validation and
an official score remain outstanding.
