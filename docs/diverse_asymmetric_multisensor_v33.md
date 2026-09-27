# v33: diversify the successful specialist ensemble

Recorded 2026-09-27, before running v33 on Kaggle. The goal remains competition
score improvement, not additional paper contributions. V33 is an unscored candidate.

## What the v32 output actually established

The user-provided Kaggle screenshot shows **0.24225 public / 0.21514 private**:
+0.00153 / +0.00213 over v31. The original five outputs are preserved, unmodified,
in `results/v32_kaggle_output/`; screenshot and hashes are recorded in
`results/v32_summary.json`. All output hashes and the scored source hash match.
The notebook itself did not submit; the post-run score comes from the screenshot.

V32 completed in **5.7681 hours on Tesla T4**, used every PA training row and
exported **7,159,911 bytes**, not gigabytes. It selected `bag1_specialist1_swap4`,
changing 10,387 test rows with 23,118 replacements and preserving all v31 counts.

The repeated spatial comparison, against a matched v31 refit, gained 0.00169160
(fold gains 0.00160979 / 0.00177349; block-bootstrap CI [0.00124181, 0.00235836]).
Outside Denmark/Netherlands the gain was 0.00239122, and country-macro gain was
0.00186442. These are not the hidden-test metric or a fresh independent audit.

Diagnostic ablations at the selected policy:

- Remove the seed bag, keep specialists: gain 0.00102918.
- Remove the specialists, keep seed bag: gain 0.00014605.
- Keep both: gain 0.00169160.

This supports adding specialist diversity instead of another ordinary three-seed
bag. It does **not** isolate whether asymmetric loss, model width, epoch schedule
or the rare head caused the benefit: these changed together in v32. V32 retained
only aggregate specialist ranks, so its two members cannot be separated afterward.
V33 explicitly records individual-member predictions and diagnostics to fix this
observability limitation without expanding output beyond the existing 16 MB cap.

Some small country groups regressed (Croatia, Montenegro, Slovakia, Bulgaria).
We retain macro-country and outside-core checks, not post-hoc country blacklists.
Four swaps narrowly beat eight on calibration; therefore eight is an option, not
a forced increase. We continue to freeze counts after v30's confounded regression.

## Frozen experiment

Control: exact scored v32 CSV, SHA-256
`71c863d9cb05f3efc63ac54dd7faa05efd2fc137a11da9adcd2064c82c88dfee`.
Development comparator: frozen v32 recipe and its selected `bag1_specialist1_swap4`
policy, refit inside each partition. Historical development weights are not saved
and cannot be claimed identical. The two-fold spatial assignment is unchanged.

New candidates, all trained from scratch with official competition inputs only:

1. Independent-seed replica of `asymmetric_geo`, width 192, 20 epochs, clipped ASL
   negative gamma 4, clip 0.02, geographic features, no rare residual branch.
2. Independent-seed replica of `rare_ecology`, width 192, 24 epochs, negative gamma
   2, clip 0.02, no geographic features, training-defined rare residual branch.
3. `asymmetric_pyramid_conv`, width 192, 24 epochs, negative gamma 2, clip 0.02,
   geographic features and rare residual branch. This uses the existing pyramid
   raster encoder, chronological temporal convolutions and environmental/static
   vector fusion, rather than a Transformer. It tests a complementary architecture
   under the successful specialist loss, not an external pretrained model.

The rare branch definition is unchanged: at least five training occurrences and
training prevalence <=0.005. It is identity-initialized. All 5,016 species remain
in the main output and loss, including those below the branch's minimum support.
Normalization, class priors and branch membership use training rows only.
The unchanged v32 trainer supplies mixup, EMA, augmentations, partial country
balancing, mixed precision and batch size 48 on GPU. Two-view inference is retained.

Each new model is calibrated on selection rows. The two attention models form
one probability-averaged expert, and the single convolution model forms another.
There are 25 predetermined policies: unchanged control plus eight attention/conv
weight pairs crossed with 2/4/8 maximum tail swaps. All row counts and the leading
60% of v32 species are protected. Zero evidence cannot promote a candidate.

Select only on calibration labels. Freeze the policy, then evaluate regression
folds, equal-country macro, outside-DK/NL gain and spatial-bootstrap lower bound.
Report individual-member fixed-count predictions and a fixed four-swap residual
diagnostic on both calibration and regression subsets. These diagnostics do not
select members or retune the policy during the run. Group-removal ablations use
the frozen selected policy. On rejection, return unchanged v32 as DO_NOT_SUBMIT.

All production fits use **all 88,987 PA rows**. Selection-only fold calibration
coefficients are averaged and transferred to production, an assumption under
distribution shift. Attention replicas retain the proved epoch schedule; the
new convolution schedule is an untested 24-epoch hypothesis. No count regressor,
PO pseudo-labeling, test labels, external dataset or downloaded weights are added.

## Time and reproducibility

Single deliverable: `notebooks/geolifeclef_v33_diverse_asymmetric_multisensor.ipynb`.
Attach `geolifeclef-2025`, select T4 x1, restart and Run All. The notebook embeds
hash-verified frozen source modules and the exact best CSV; no repo imports or
extra uploaded dataset are needed. The source must remain below 1,000,000 bytes.

There are **28 development fits** (22 frozen-reference fits and six new fits) and
at most **three production fits**, all serial. Full v33 T4 runtime is unmeasured;
v32's measured 5.77h is context, not a guarantee for this heavier development phase.
GPU and official split-size preflight happens before expensive feature loading.

The 10.75h cooperative guard leaves headroom inside the requested 12h. After the
first fold, remaining development admission uses its whole wall time, scaled for
the next fold's training size and multiplied by 1.25. Thus all eleven reference
fits, inference, calibration and diagnostics are counted. Whole-production
admission uses the slower observed per-row epoch time per active model, scaled to
all PA rows, multiplied by 1.5, plus 45 minutes. Additional policy overhead is
reserved. A slow run can stop instead of completing; a blocked driver/I/O call
cannot be preempted by a cooperative guard.

## Output for the next iteration

Exactly five successful files, total <=16,000,000 bytes:

- Eligible `GLC25_PA_submission_v33.csv`, otherwise a DO_NOT_SUBMIT CSV.
- `v33_report.json`: policies, each new model's history/calibration, individual
  model diagnostics, group ablations, per-country and species-group results.
- `v33_manifest.json`: source/output hashes, configurations and spatial splits.
- `regression_per_survey_v33.csv`: matched-v32/new scores, cardinality and swaps.
- `calibration_top128_v33.npz`: 1,500 sampled calibration rows per fold, truth,
  matched-v32 lists, rankings/probabilities for both groups AND all three members.

Temporary caches/checkpoints are removed only under `v33_runtime`. A failure after
CSV creation revokes its ready filename. A late failure after report creation also
marks that report ineligible and renames the now-invalid manifest. A failure report
preserves completed-fold histories for diagnosis. There is no automatic submission.

## Limits

All 88,987 PA labels were consumed in earlier development. Repeated regression
gates and bootstrap intervals are not independent proof after adaptive iteration.
No Kaggle improvement, win, universal SOTA, or full-runtime result is claimed before
the run. Best-score replacement requires an actual better official result. Current
target gaps remain 0.02917 public / 0.01507 private, against the recorded winning
scores. Local synthetic tests check software and export contracts, not accuracy.
