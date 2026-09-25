# v29: full-data multisensor ensemble

Registered 2026-09-25 after inspecting the completed v28 output, before running v29.
The external target remains the historical 2025 winning private score, 0.23021.
Current best: v27, 0.23339 public / 0.20831 private. No SOTA claim is made.

## Why v28 stayed flat

V28 selected `control`. Its submission hash is identical to v27:
`32cd02788d910abe0cd18715da00f201a52a371a2135d17399301dd7e532f710`.
The observed leaderboard equality therefore is not rounding. Its 4,010-row fresh audit
changed zero predictions, gained zero F1 and failed submission eligibility. Three PO
policies regressed on calibration; the most restrictive policy tied control. The
experiment completed in 1.6213 hours. The previous output naming was unsafe for manual
handoff: a failed-gate control was still called a submission. V29 fixes that distinction.

## Research basis and scope

The [second-place team's working note](https://ceur-ws.org/Vol-4038/paper_261.pdf)
documents severe train/test geographic imbalance, diverse model ensembling and
probability-based set-size optimization. It also reports difficulties transferring PO
augmentation to deep-feature models. These motivate testing genuine model diversity
instead of another PO tail-swap policy; they do not establish that our implementation
will improve. Their method used external predictors/foundation models and more compute;
this notebook does not reproduce it.

The [GeoLifeCLEF overview](https://www.dei.unipd.it/~faggioli/temp/clef2025/paper_234.pdf)
describes four-band RGB+NIR TIFF inputs. The new reader canonicalizes explicit TIFF
band/color metadata to RGB-NIR and otherwise uses the published RGB-NIR convention.
Conflicting partial metadata is rejected rather than silently guessed. The frozen
reference reader remains untouched, including its historical band-index convention.

Constraints: one GPU T4 or faster, official `geolifeclef-2025` competition only,
random initialization, offline runtime, one self-contained notebook below 1,000,000 bytes.

## Frozen candidate

1. Read the original 32px features for the matched reference and an additional native
   64px Sentinel tensor for new models. Derive NDVI/NDWI; standardize sensor channels
   using development-training rows only. Use soil/environmental predictors as supplied.
2. Train three distinct all-5,016-species models: (a) CNN spatial tokens plus chronological
   Landsat/climate tokens with a three-layer transformer, (b) multi-scale spatial CNN
   plus chronological 1D convolutions, and (c) an attention expert without coordinates
   or country identifiers. Geography-aware models use explicit country categories and
   coordinate Fourier features, not country hashing.
3. Use inverse-country-frequency sampling mixed with uniform sampling, mixup, rotated /
   flipped imagery, modality dropout in attention models, AdamW, cosine learning rates,
   EMA weights and early stopping. Two models use weighted label-wise BCE; one adds
   asymmetric negative focusing. Rare outputs are not masked out. Seeds are set before
   model construction. Maximum development epochs: 48; first checkpoint selection is
   at epoch 1 and then every 3, with bounded per-fit wall time.
4. Fit shared Platt slope/intercept on checkpoint-selection predictions using all species.
   Combine calibrated probabilities equally across the three members. Two image views
   are used for calibration, assessment and official-test inference.
5. Calibrate 13 fixed policies: unchanged control, or model weight 0.25/0.5/0.75/1.0
   crossed with set-size scale 0.8/1.0/1.2. Weight 1.0 is a standalone new ensemble,
   not a small repair to old predictions. The set-size objective is the
   **ratio-of-expectations surrogate** `2*cumulative_probability/(k+sum_probability*scale)`
   over k=8..40; it is not claimed to equal exact expected F1. Intermediate weights
   combine ranked lists and interpolate cardinality with the frozen/matched control.
6. Choose a policy using 70% pooled F1 gain, 15% equal-country gain (countries with at
   least 30 calibration samples) and 15% equal-block gain. Non-control policies must
   have positive pooled and weighted gain; ties favor less model weight.
7. If a new policy wins, refit all three members on **all 88,987 PA surveys**, using
   their selected epoch counts, separate fixed seeds and full-PA normalization. No test
   labels are used. Probability-calibration transfer to refitted models is an assumption.
   The exact scored v27 test list remains the test control.

The fixed internal reference is the v27 recipe with its registered final policy. Its
last pyramid has one seed, whereas the historical deployed v27 used three; this is a
matched recipe approximation, not identical model weights or a hidden-test estimate.

## Development and the last fresh audit

The union of v21-v28 assessments contains **86,592 IDs**. All remaining **2,395** PA
surveys are reserved for one final audit, and their whole one-degree blocks are excluded
from all development roles. On other blocks, `stable_bucket('v29-dev:20260925:'+block)`
assigns <10 to checkpoint selection, 10..19 to calibration and >=20 to candidate
training. Training additionally excludes points within 20 km of any held-out role.
The protocol has no score-driven split retries.

Metadata-only preflight on the local official PA file yields:

| Role | Surveys |
| --- | ---: |
| Training | 42,840 |
| Checkpoint selection / Platt fit | 18,753 |
| Policy calibration | 8,045 |
| Final audit | 2,395 |

The audit contains 24 one-degree blocks, including 1,776 Danish and 373 French surveys.
It cannot demonstrate generalization to Ukraine/Bulgaria or establish SOTA. Development
labels were already observed by previous experiments and are explicitly not fresh.

Policy and prediction lists are frozen before audit scoring. Full-data production
refits use all PA labels, **including these audit rows**, under the already locked
recipe. The audit only evaluates the separate development predictor, which never trained
on audit rows. It is not an evaluation of production weights. The gate needs a nonzero
new-model weight, positive audit gain, positive lower 95% paired spatial-block bootstrap
bound (1,000 repetitions), all integrity checks and a CSV different from v27.
Once this run is inspected, no fresh PA audit pool remains; future validation must be
labeled repeated development/cross-validation rather than claiming untouched assessment.

## Runtime, output and verification

The 10.75-hour guard is **cooperative, not a hard operating-system deadline**. GPU and
split checks run before expensive feature reading. Preparation reserves seven hours;
training and inference check time every batch. Development receives per-model time
budgets, and a conservative measured estimate admits the entire production ensemble
before any refit starts. A slow session may stop with a clear failure report instead of
finishing a model. Full T4 runtime has not yet been measured; do not promise completion.

Only `/kaggle/working/v29_export` remains after cleanup. A successful full execution
contains exactly four small files, normally a few MB:

- `GLC25_PA_submission_v29.csv` **only if eligible**; otherwise
  `candidate_DO_NOT_SUBMIT.csv` or `unchanged_v27_DO_NOT_SUBMIT.csv`.
- `assessment_per_survey_v29.csv`.
- `v29_report.json`: decision, audit, calibration trials, per-member diagnostics,
  training histories, parameters, peak allocated GPU memory, times and limitations.
- `v29_manifest.json`: source and output hashes, split, features and configuration.

Large caches/checkpoints are temporary and excluded from the final output. A failed
execution writes `failure_report.json`. The notebook never submits automatically.

Rebuild with `.venv/research/Scripts/python.exe -m scripts.build_v29_notebook` from the
repository root. Test with `PYTHONPATH=src` and the same interpreter's pytest module.
Regression tests cover embedded-source execution, exact v27 CSV round-trip, real CPU
backpropagation for every new architecture, rare-output updates, fixed-epoch refits,
complete miniature pipeline, calibration, official split disjointness, sensor band
indices, budget admission and failed/no-op output naming. Synthetic CPU tests do not
measure the official score or full Kaggle performance.
