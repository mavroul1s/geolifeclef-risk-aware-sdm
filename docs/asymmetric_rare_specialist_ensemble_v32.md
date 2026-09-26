# v32: competition-score experiment

Recorded 2026-09-26, before running v32 on Kaggle. This iteration focuses only on
improving the hidden-test score, not on additional paper contributions.

## Evidence that motivates this run

The exact five v31 outputs are archived in `results/v31_kaggle_output/`, with the
user's leaderboard screenshot in `results/v31_kaggle_scores.png`. V31 reached
0.24072 public / 0.21301 private, improving the prior best v29 by 0.00172 / 0.00208.
The winning targets remain 0.27142 public / 0.23021 private, respectively. V31 is
not ahead of the winner. These scores are user-provided screenshot evidence;
the original notebook report correctly contains no post-submission scores.

V31 took 3.1845 hours on a Tesla T4. It changed 7,306 test rows, preserved every
v29 row's species count, and passed its repeated geographic development gates.
The development gain was only 0.0003201, with spatial CI [0.00010645, 0.00065869].
Removing habitat left about 0.00030555 gain; removing the neural replicas left
about -0.00001595. This supports testing more neural diversity. It does not establish
that removing habitat from the final scored submission would improve its score.

V30 changed data coverage, counts, epochs and model composition simultaneously
and regressed. We therefore keep v31's counts and all-PA production training here.
No new production calibration anchors are removed. Pure-seed and specialist-only
policies separate the new ensemble groups; this is not a full factorial causal
ablation of all internal architecture/loss changes.

## Competitor evidence and the adaptation

The [organizers' overview](https://www.dei.unipd.it/~faggioli/temp/clef2025/paper_234.pdf)
describes winner webmaking as combining all-species and rare-species classifiers,
GeoCLIP, a species-count regressor and spatial post-processing. The search-indexed
primary-source text was available even though direct PDF retrieval returned 404.
We verified this description, not the winner's complete implementation or weights.

[Tighnari v2](https://ceur-ws.org/Vol-4038/paper_246.pdf) uses asymmetric loss to
address imbalanced, noisy multi-label targets as part of a much larger system.
The adaptation below is a new lightweight hypothesis using our own architecture.
It is neither an exact reproduction nor an assertion that these components caused
the competitors' gains. We do not import external data or pretrained weights.

## Frozen candidate before the next Kaggle run

- Reference: exact v31 official-test CSV, SHA-256
  `da070a8ac5708ef5d7fb38cdbecf862aa9d036e4d246bc1e4b364572bb7fe367`.
- Development reference: frozen v31 recipe with its already-selected policy
  `seed1_habitat0.25_swap2`, refit inside each training partition. No re-selection
  of v31's policy and no claim that refit weights equal historical weights.
- New seed bag: three original v29 architectures, new seeds (+62000), widths 128,
  fixed epochs 12/15/18, two-view inference and the original production calibration.
- Specialist group: two width-192 attention models, fixed 20/24 epochs. Both use
  asymmetric loss, positive exponent zero, negative clipping 0.02 and negative
  exponents 4/2. Negative focal weights are detached; positive and negative terms
  are computed separately to support soft mixup labels correctly.
- The first specialist uses geographic features. The second excludes geography
  and adds a nonlinear residual head for training-only taxa with count >=5 and
  prevalence <=0.005. It is zero-initialized to preserve the ordinary initial head.
  The main classifier and loss still cover all 5,016 taxa, including rarer ones.
- Training: EMA, mixed precision, mixup, original sensor augmentation and partial
  country-balanced sampling. New specialists use batches of 48 on GPU. All models
  run serially; there is no multi-GPU or downloaded-weight dependency.
- Specialist probabilities are calibrated on selection rows only. Production
  transfers the mean fold coefficients. This can be inaccurate under shift and
  increased full-data training size; it is explicitly reported, not hidden.
- 22 policies: unchanged control, plus seven fixed seed/specialist weight pairs
  crossed with max 2/4/8 replacements. Reciprocal-rank fusion keeps every row's
  exact v31 count and its leading 60%. No count predictor is introduced in v32.

## Selection, comparisons and limits

Reuse the fixed two 20-km-buffered spatial folds from v30/v31. All 88,987 official
PA IDs have already been assessed in earlier experiments. These are **repeated
development comparisons**, not new independent audits.

Calibrate each member using selection labels; select the fusion policy on
calibration labels only; freeze it before inspecting the two assessment subsets.
Both calibration-fold gains and overall/country-macro/outside-DK-NL gains must be
positive. The selected policy must then have positive gains on both regression
folds, positive macro-country/outside-core gains and a positive block-bootstrap
lower bound against matched v31. The bootstrap is not corrected for the history
of adaptive experimentation and must not be read as independent significance.

Report selected-policy ablations without each new expert, micro/macro F1,
frequency strata, per-country and per-survey changes. Do not retune after seeing
these regression scores in the same run. If rejected, keep the exact v31 test CSV
under an explicit DO_NOT_SUBMIT filename. Passing all gates does not guarantee
leaderboard improvement. After each user-run experiment, archive results and use
the actual score to decide whether it replaces the best submission.

## Execution and handoff contract

Single notebook: `notebooks/geolifeclef_v32_asymmetric_rare_specialist_ensemble.ipynb`.
Only the official `geolifeclef-2025` competition input; T4 x1; internet not needed.
GPU and full official split sizes are checked before feature extraction. All
frozen source modules and the embedded CSV are hash-checked; no repository import
or auxiliary uploaded dataset is required by the generated notebook.

There are 22 development fits (12 frozen-reference fits and 10 new fits), at most
five new production fits. V32 full runtime is **unmeasured**. V31's 3.18h does not
by itself establish that this heavier run fits. The 10.75h cooperative guard leaves
headroom within 12h. After fold 0, admission of the remaining plan uses measured
training speed with margins. Production admission includes all active members,
1.5x per-epoch scaling to the full training set, and 45 minutes of overhead. A slow
session may stop, and no guarantee is made against a stalled driver/I/O operation.

Exactly five successful exports, capped at 16 MB combined:

1. `GLC25_PA_submission_v32.csv` **only if eligible**, otherwise DO_NOT_SUBMIT CSV.
2. `v32_report.json`: policy trials, member histories, calibration and diagnostics.
3. `v32_manifest.json`: source/output hashes, inputs, splits and configurations.
4. `regression_per_survey_v32.csv`: per-survey matched-v31/new scores and swaps.
5. `calibration_top128_v32.npz`: 1,500 sampled calibration rows per fold, two
   ensemble rankings, probabilities, truth and matched-v31 lists. This is not a
   checkpoint, complete validation probability matrix or fresh holdout.

Temporary caches/checkpoints are deleted only in the version-owned runtime
directory. Failure after CSV creation revokes the ready-to-submit filename.
Run-time cleanup intentionally means this is a compact output for diagnosing
the next run, not a model-weight recovery package. There is no automatic submission.

Local smoke tests use synthetic data and shortened epochs/widths. They validate
code paths and file contracts, not accuracy or full T4 runtime. See
`results/v32_local_validation.json` for the completed local validation evidence.
