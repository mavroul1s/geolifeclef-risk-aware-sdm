# v30: calibrated multiscale attention

Registered 2026-09-25, after inspecting v29 and before any full v30 Kaggle run.
Current best: **0.23900 public / 0.21093 private**. The historical competition target
is 0.23021 private; the gap is 0.01928. This notebook is an experimental candidate,
not a demonstrated SOTA result. Only official competition data, random initialization,
one T4 GPU, offline runtime and a single notebook below 1 MB are in scope.

## Evidence and hypotheses

V29 completed on a Tesla T4 in 3.0142 hours. It selected 100% new-ensemble predictions,
with count scale 0.8; its CSV passed the gate and improved private F1 by 0.00262 over
v27/v28. The fresh internal gain was much larger, 0.03223, on a small Denmark-heavy
audit. The hidden-test geography remains a major transfer limitation.

The independent-member calibration scores were 0.37789 for temporal attention,
0.33983 for convolution and 0.38038 for geography-free attention. The selected ensemble
reached 0.38591. Thus **a weak standalone CNN is not proof that removing it helps**:
complementary errors can make it useful. V30 tests both attention-only and low-CNN-weight
ensembles against a matched v29 reference.

V29 transferred probability-calibration parameters from development to different
full-data weights, while changing seed, training-set size and learning-rate trajectory.
Its mean prediction length was 19.63 on assessment but 26.86 on official test. This
is a clue to investigate, not proof of overprediction: the actual test distribution
may have different richness. V30 directly calibrates each deployed model on spatially
held-out PA anchors instead of assuming this calibration transfers.

The [GeoLifeCLEF working note on joint/deep models](https://ceur-ws.org/Vol-4038/paper_261.pdf)
discusses probability-based list-size selection, diverse ensembles and the mismatch
between internal/public selection and private outcomes. Our learned-count model and
two-scale architecture are distinct implementations, not a reproduction of that paper.
No paper's score or method makes a performance guarantee for this notebook.

## Models, policies and training

Retain the exact scored v29 reader, normalization, augmentation and training code,
verified by source hashes. Do not edit v29 or the v27 reader module. Cache original
32px reference rasters and canonical RGB-NIR 64px Sentinel inputs, with NDVI/NDWI.

Development fits four models per fold:

- V29 temporal attention, geography-aware, width 128.
- V29 multisensor convolution, width 128.
- V29 ecology attention, no coordinate/country features, width 128.
- New ecology attention, width 192, with shared-CNN tokens from the entire 64px scene
  and its central 32px habitat crop, plus direct environmental/static feature paths.

All models predict all 5,016 species. Mixup, image augmentation, country-balanced /
uniform sampling, EMA, checkpoint selection and early stopping follow the frozen v29
routine, with at most 48 development epochs and explicit wall-time allocations.
Each fold calibrates probabilities on its model-unseen checkpoint-selection rows.
This reuses those rows for selection and calibration fitting and is development, not
an unbiased evaluation. Two Sentinel views are used for final predictions.

The matched reference is the equally averaged three calibrated v29 members, decoded
with its frozen 0.8 count-scale policy. It is a recipe refit, not the historical deployed
weights. The exact deployed v29 CSV is used only as the official-test control.

Candidate weight tuples, ordered temporal/CNN/ecology/multiscale:

| Ensemble | Weights |
| --- | --- |
| Attention pair | 0.5 / 0 / 0.5 / 0 |
| Balanced attention | 1/3 / 0 / 1/3 / 1/3 |
| Geography-leaning | 0.6 / 0 / 0.2 / 0.2 |
| Ecology-leaning | 0.2 / 0 / 0.4 / 0.4 |
| Diverse, low CNN | 0.3 / 0.1 / 0.3 / 0.3 |

For each mixture, a small regularized gradient-boosted regressor learns the oracle
top-k (k=8..40) from model-unseen selection predictions. Its inputs contain probability
mass, ranking summaries, nearest-training distance, survey area and year, but no country
label averages or species labels at inference. Policies use 0%, 50% or 100% learned
counts, combined with the original probability-based count surrogate. Model rank/list
weight is 0.75 or 1.0 around the control, for **30 candidates plus unchanged control**.

Select one policy on calibration rows using 60% pooled F1 gain, 20% country-average gain
(at least 30 distinct calibration surveys per country), and 20% block-average gain.
A candidate must improve calibration in both folds; ties favor less intervention.
Policy parameters are frozen before the outer regression scores are read. No public
or private leaderboard score is used to select among v30 policies inside the notebook.

## Validation: no fresh holdout remains

The union of assessed IDs from v21 through v29 now covers **all 88,987 PA surveys**.
Do not invent a new untouched audit or silently lower a fresh-split size threshold.
V30 instead performs a transparent repeated spatial development regression check.
Previous results have influenced the method design, so the check may be optimistic.

One-degree blocks are deterministically allocated by a label-blind greedy size/country
balancer, targeting roles 18%/18%/10%/10%/44%. Sorting and tie breaking use seed 20260926.
The two 18% roles are disjoint outer regression folds; selection and calibration are
shared across folds. Each model's training excludes its own regression fold, selection,
calibration, and points within 20 km of those roles. The other outer fold can enter
training under ordinary cross-fitting. No score-based split retries are performed.

Metadata-only preflight yields:

| Role | Fold 0 | Fold 1 |
| --- | ---: | ---: |
| Training after 20 km buffer | 35,811 | 37,487 |
| Checkpoint selection / calibration fitting | 8,816 | 8,816 |
| Policy calibration | 8,691 | 8,691 |
| Outer regression check | 15,792 | 15,778 |

Measured minimum evaluation distances are 20.00266 / 20.00039 km. The calibration
rows occur twice as predictions of different fitted models; they are not 17,382 unique
surveys. Outer regression IDs are unique. The gate needs a non-control policy, positive
gain in both outer folds, positive lower paired one-degree-block bootstrap bound
(1,000 repetitions), intact controls/buffers, and runtime within budget. These are
development safeguards, not independent evidence of SOTA or a hidden-test estimate.

## Production-specific calibration

Production uses a separate fixed labels-blind hash of whole one-degree blocks:
`stable_bucket('v30-production:20260926:'+block) < 8` marks calibration anchors.
The anchors and their 20 km training buffer yield **72,063 training / 8,302 anchor rows**,
with a measured minimum distance of 20.00377 km. Anchors never train production weights,
choose checkpoints, or fit normalization. They have been observed in prior experiments
and may participate in development; they are model-held-out, not historically untouched.

Only active ensemble members are refitted, with median selected development epoch
counts and fixed production seeds. Production probabilities are calibrated on anchor
predictions from these exact weights; count correction is also fitted on anchors.
No calibration is transferred from development weights. The model/weight/count policy
is already locked. Inference uses only test predictors and the exact scored v29 lists.

This explicitly trades some PA training data for calibrated deployment: unlike v29,
the production models do not fit all 88,987 rows. Whether that tradeoff improves hidden
F1 must be measured, not assumed.

## Runtime, compact evidence and handoff

The cooperative guard is 10.75 hours, not an operating-system deadline. GPU/split
preflight occurs before remote-data preparation; development allocates wall time per
fit; the entire production ensemble is admitted using measured epoch times, row-count
ratios, a safety factor and finalization reserve. Batch checks stop a slow run safely.
The prior v29 measured 3.0142 hours, but **v30 full T4 runtime is not yet measured**.
No guarantee of SOTA or completion within 12 hours is made without that run.

Delete only this version's bounded temporary directory. Keep five files under
`/kaggle/working/v30_export`, with a 16 MB combined cap:

- `GLC25_PA_submission_v30.csv` if eligible; otherwise `unchanged_v29_DO_NOT_SUBMIT.csv`
  or `candidate_DO_NOT_SUBMIT.csv`.
- `regression_per_survey_v30.csv`, explicitly repeated-development results.
- `v30_report.json`: chosen policy, per-model/country results, timing, gates and limitations.
- `v30_manifest.json`: hashes, folds, features and production-anchor contract.
- `calibration_top64_v30.npz`: at most 2,000 sampled calibration rows per fold, four
  members' top-64 species/probabilities and total probability mass, true-label sparse
  indices, vocabulary, country and fold. This is truncated diagnostic evidence, not a
  full probability matrix or training checkpoint. It enables compact post-run review.

**Only the eligible `GLC25_PA_submission_v30.csv` is submitted to Kaggle.** Never upload
the ZIP, report, NPZ or per-survey diagnostic CSV as the competition submission.
The notebook makes no automatic submission. Its final message repeats this distinction.

Rebuild with `.venv/research/Scripts/python.exe -m scripts.build_v30_notebook`; run
the test suite with `PYTHONPATH=src`. Tests exercise actual CPU backpropagation, the
complete miniature training/calibration/refit/export flow, held-out production anchors,
official metadata split counts, source/artifact hashes, no-op naming and budget admission.
These tests do not substitute for full official-data GPU execution.
