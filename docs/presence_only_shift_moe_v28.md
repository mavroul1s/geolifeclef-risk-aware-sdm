# v28: presence-only shift mixture-of-experts

Preregistered 2026-09-24 after recording the completed v27 result and before any v28 assessment labels are inspected. One self-contained Kaggle notebook, one GPU, official GeoLifeCLEF 2025 data only, random initialization, and all 5,016 PA species. The exact scored v27 submission is the official-test control. The target remains 0.23021 private; v27 reached 0.20831, so the gap is 0.02190 and SOTA is not claimed.

## Diagnosis and hypothesis

V27 is a real improvement: 0.23339 public / 0.20831 private, up 0.00287 / 0.00138 from v26. Its fresh matched assessment gained 0.01294 with spatial-block bootstrap CI [0.00311, 0.02143]. Candidate ranking and adaptive cardinality both contributed. However, 5,633 of 9,095 assessment surveys were Dutch, while the official test is dominated by Bulgaria, Ukraine and Switzerland. V27 changes a large fraction of each list, but this internal gain transferred only weakly to private F1. The next experiment therefore targets geography rather than increasing neural blend strength again.

The v28 hypothesis is that multi-scale competition PO occurrence support can safely repair a small number of low-ranked v27 species in PA-sparse countries. Unlike v21/v22's globally mixed PO neural expert, this layer preserves the exact v27 cardinality and makes at most four confidence-gated tail swaps. This is a deliberately different and lower-variance use of PO evidence.

## Frozen method

Stream `GLC25_P0_metadata_train.csv`, keep only the fixed PA vocabulary, remove PO records within 100 m of any PA coordinate without consulting labels, and frequency-correct each species by global PO count. Build a 0.1-degree local index, then compact 0.5-degree landscape and 2-degree regional indices from its bounded sketch. Query the surrounding cells at each scale and reward species supported coherently across scales. PO non-observation is never treated as absence.

Compute a labels-blind shift risk from nearest PA distance, PA training-country support, PO coverage and cross-scale agreement. Starting from the exact/matched v27 list, protect its upper half. Candidate policies make zero, one, two, three or four tail swaps only when risk, PO confidence and the PO advantage over the removed species exceed frozen thresholds. Cardinality is unchanged.

Select one policy on calibration data using an equally preregistered objective: 50% pooled sample F1, 25% equal-country macro F1 for countries with at least 20 calibration surveys, and 25% equal-quarter-degree-block macro F1. This prevents the largest country from fully determining the intervention. Ties prefer the less invasive policy.

## Fresh assessment and leakage controls

The immutable union of v21-v27 assessments contains 82,582 surveys and is excluded. Only 6,405 never-assessed PA surveys remain, heavily concentrated in Denmark. Whole 0.25-degree blocks are used because one remaining one-degree Danish cell alone contains 4,524 surveys. The fixed labels-blind seed 20261095 was chosen using coordinates and country names only, before reading species labels, to balance fold size and cover countries present in test metadata.

Fold 0 has 1,845 assessment surveys in 31 blocks; fold 1 has 2,165 in 45 blocks. Both retain a measured 20 km minimum distance from their own fitted PA training rows. The other outer fold may enter training under ordinary cross-fitting. Checkpoint selection, policy calibration and current-fold assessment remain separate. The remaining unseen pool is small and still not representative of Ukraine or the full test distribution; the official score remains decisive.

The complete matched v27 recipe is refit for each outer fold. All policies and predictions are frozen before assessment. Submission eligibility requires positive pooled gain, a positive lower spatial-bootstrap bound, positive gain in each fold, positive gain in at least two substantial countries, exact cardinality preservation, a nonzero bounded intervention and all integrity/runtime checks.

## Runtime and deliverables

The official deployment reuses the embedded scored v27 list and trains no new deployment neural model. Outer scientific evaluation refits 12 models sequentially; only one resides on the GPU at a time. The hard guard is 10.75 hours within Kaggle's 12-hour limit, with a 35-minute finalization reserve. The prior, heavier v27 completed in 2.07 hours; the expected v28 range is 2-7 hours.

The notebook source must remain below 1,000,000 bytes and leave exactly four compact files in `/kaggle/working/v28_export`: the submission CSV, per-survey assessment CSV, report JSON and manifest JSON. Temporary rasters and checkpoints are always deleted. The notebook never submits automatically.
