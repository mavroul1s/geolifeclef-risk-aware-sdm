# v35: full-set, checkpoint-selected balanced attention

Design frozen on 2026-10-05, after inspecting v34. No Kaggle run or external
submission has been made by the assistant. This is a candidate, not proven SOTA.

## Postmortem and decision

V34 completed in 5.24679 hours on T4 and scored **0.24203 public / 0.21483 private**,
below v32's **0.24225 / 0.21514**. It changed 6,143 test sets, keeping all original
counts. Its repeated spatial gain was +0.00043004, with block-bootstrap interval
[0.0000004268, 0.00071510]. No input/schema/runtime defect was found. A tiny positive
lower bound was not evidence strong enough to trust under the actual test shift.
The exact outputs, screenshot and hashes are archived in `results/v34_kaggle_output/`
and `results/v34_summary.json`.

Denmark/Netherlands dominate PA training (~72%) but form only ~16% of official
test surveys. Bulgaria, Ukraine and Switzerland are much more prominent in test;
many test countries are absent or barely represented in PA. This is an observed
distribution difference, not proof of the precise cause of v34's small regression.

V35 does **not** repeat the graph correction or loosen gates to force a submission.
It changes the full predictive model and removes the permanent species-list/count
lock inherited from v29. It reuses verified historical *development scores* to
avoid spending time on 22 old reference fits, allocating that budget to six new
development fits and at most three full-PA production fits.

## Models and selection

Three width-192, three-layer multimodal attention models cover all 5,016 species:

- `long_geo`: successful geographic asymmetric-loss attention recipe, up to 48
  epochs, negative exponent 4, clip .02, no extra rare head or ranking auxiliary.
- `balanced_ecology`: geography-free sensor-balanced pooling, rare residual head,
  exponent 2, clip .02, ranking auxiliary weight .08, up to 48 epochs.
- `balanced_geo`: same balanced pooling with geography, rare head, exponent 4,
  clip .02, ranking weight .08, up to 48 epochs.

The original attention readout concatenates CLS with the mean of all sensor
tokens. That mean weights 64 image tokens much more heavily than the single
environment token. The new readout separately pools image, Landsat, climate,
environment and static tokens, concatenates those with CLS, and learns a fusion
projection. This changes representation weighting, not the official inputs.

The auxiliary loss is `-sum_j (y_j/sum(y))*log_softmax(logits)_j`, averaged over
surveys. It gives each nonempty survey equal total positive ranking mass and
supports soft mixup labels. It is an additional ranking surrogate, not a calibrated
probability likelihood or an exact F1 objective. The existing all-species asymmetric
loss, EMA, mixup, sensor dropout and partial country-balanced sampling remain.

Select EMA checkpoints at epochs 12/24/36/48 using **selection rows only**. The
checkpoint score averages sample F1 at top12/18/24/30, then weights pooled,
country-macro and outside-DK/NL scores .5/.25/.25. A shared Platt calibration is
then fitted on selection rows. Calibration policy search uses neither assessment
labels nor assessment scores to choose a checkpoint/model/count mode.

There are 15 policy candidates: equal weights (1,1,1), ecology-lean (1,2,1), or the
balanced pair (0,1,1), crossed with expected-F1-surrogate scales .7/1/1.3 or fixed
top18/top24. The surrogate chooses k in 8..40 to maximize
`2*cumsum(top_probabilities)/(k + scale*sum(all_probabilities))`. It is a
ratio of expectations, **not** exact expected F1. The distinction is discussed in
[Dembczynski et al., ICML 2013](https://proceedings.mlr.press/v28/dembczynski13.html).
Complete rankings and cardinalities can change; this is no longer bounded swapping.

Pooling, loss, schedule and decoding change together. The ordinary long-geo model
and per-member diagnostics help interpretation, but this is not a full causal
factorial ablation. No accuracy claim follows from local synthetic tests.

## Cached comparator integrity

The cache embeds **matched v32 recipe** calibration predictions and assessment F1
from the v34 run, NOT v34 graph predictions and NOT hidden test labels. V34 refit
the unchanged v32 recipe before its graph stage; we verified the archived manifest,
output hashes and source hash. This relies on the user-provided archive's provenance,
not an independent Kaggle API attestation.

The notebook checks, before comparison:

1. Compressed and raw payload SHA-256.
2. Exact official training-metadata file SHA-256, before predictor preparation.
3. Every training/selection/calibration/assessment survey-ID hash in both spatial
   folds, preserving the original 20-km exclusion buffers.
4. Sorted species vocabulary and the byte hash of **all** binary PA labels.
5. Cached calibration F1 recomputed from the official labels and cached predictions.
6. Complete, unique assessment coverage and valid score ranges.

The all-label hash is an integrity check, not a training signal. Only matching
training indices are passed to model fitting and normalization. A mismatch fails
closed; cached scores are never silently compared against a changed split/dataset.

Calibration uses 1,500 saved rows per fold, not the original full 8,691-row pool.
Some IDs repeat across views; country-support diagnostics average them by survey.
Assessment uses all 31,570 original spatial-check surveys. Cached baseline F1 is
enough for paired score comparisons; it does not reconstruct old assessment label
rankings or baseline micro/macro metrics. Only the new model's such metrics are
reported. Do not invent unavailable baseline analyses from the cached scalar F1.

Every PA survey has been assessed in prior work. These remain **repeated adaptive
development checks**, not fresh independent validation. The score cache changes
compute cost, not that statistical limitation.

## Promotion and production

Both calibration and assessment must show pooled gain >=.001 and outside-core gain
>=.0005, positive country-macro and positive gains in both folds. Assessment's
block-bootstrap lower bound must be >=.00025. These practical margins were chosen
after v34's failure and are safeguards, not calibrated confidence guarantees.

Choose the best eligible calibration policy, or the best attempted candidate if
none qualifies. Freeze the policy and per-model median selected epoch counts before
assessment. A rejected attempt retains its **actual** predictions and diagnostics;
it does not hide failure behind a zero-delta identical control.

If all gates pass, train active ensemble members on **all PA rows**, with no held-out
production anchors. Production stops at the median fold-selected epoch, while
keeping the original 48-epoch cosine schedule prefix. Calibration transfers the
mean fold coefficients; this remains an explicit full-data transfer assumption.
If rejected, do not emit a submission-shaped CSV; output `NO_SUBMISSION.json` and
keep the scored v32. A new actual hidden-test score is required to supersede v32.

## Resource and delivery contract

- One offline notebook below 1 MB; official `geolifeclef-2025` input only, no extra
  uploaded dataset, checkpoint bundle, downloads or external pretraining.
- One T4. GPU preflight precedes predictor preparation. Six new development fits
  and at most three production fits; no frozen reference neural models are refit.
- Cooperative 10.75-hour guard within the requested 12-hour session. Remaining-plan
  admission uses the complete first-fold wall time, next-fold training size, a
  1.25x development margin, maximum production epochs, a 1.5x training-speed margin,
  and overhead reserves. Production admission uses selected epochs. Full T4 duration
  is unmeasured; the guard can stop a slow run, not preempt stalled hardware/I/O.
- Exactly five files, <=16 MB total: eligible CSV **or** `NO_SUBMISSION.json`, report,
  manifest, per-survey diagnostic CSV, and sampled predictions NPZ. The latter
  includes each member's top128, **full** probability mass, truth and cached reference
  F1 for all 3,000 calibration views and 1,000 sampled assessment views.
- Temporary features/checkpoints are removed. Failure after CSV creation revokes
  eligibility and renames the failed prediction to a non-CSV artifact; late manifest
  failure also invalidates the report/manifest.

Run **T4 x1 → attach GeoLifeCLEF25 → restart → Run All**. Submit only
`v35_export/GLC25_PA_submission_v35.csv` when `eligible_for_submission: true`.
No guarantee of monotonic leaderboard improvement or SOTA is made.
