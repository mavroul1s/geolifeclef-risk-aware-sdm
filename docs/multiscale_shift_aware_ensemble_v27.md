# v27 multi-scale shift-aware ensemble

## Decision and objective

The scored v26 run is the new official best at **0.23052 public / 0.20693 private**, but it does
not reach the 0.23021 private target. The remaining private gap is 0.02328. v27 is therefore a
new registered candidate; it does not claim SOTA before a like-for-like hidden-test result.

The v26 evidence is positive and coherent: its fresh 3,857-survey assessment gained 0.003940
sample-F1 over the matched v25 recipe, with a spatial block-bootstrap 95% interval of
[0.002316, 0.005748], and its official score improved on both leaderboard splits. Its selected
policy was also the most aggressive candidate in the registered v26 grid. v27 keeps the exact
scored v26 predictions as its official-test control and tests a more diverse visual model plus a
wider, still calibrated, list-fusion range.

## Frozen data and evaluation protocol

- Use only the official `geolifeclef-2025` competition input; internet, external data, and
  pretrained weights are forbidden.
- Embed the exact v26 submission identified by SHA-256
  `67937cf92f0b4626b7b3c643ababbd0c7a4f39801909e4bd5646b532371c21a9`.
- Embed and exclude the union of all 73,487 survey IDs assessed by v21 through v26.
- Use two preregistered fresh assessment block ranges: 0–19 and 20–39 under the v27 stable
  spatial hash. Neither assessment set may affect checkpoints, calibration, or policy choice.
- The original split seed `20260925` left fold 1 with only 444 fresh surveys. Before any
  assessment labels were read, a labels-blind search over retries 0–99 maximized the smaller
  fresh fold subject to spatial-block and development/training-size constraints. This froze
  retry 29 (`20260954`), producing 4,879 and 4,216 fresh surveys across 16 and 15 spatial blocks.
  The notebook now validates both outer folds and the deployment split before feature extraction
  or training.
- Draw checkpoint-selection and policy-calibration surveys from already-consumed IDs in distinct
  ranges 40–49 and 50–59. Candidate training uses blocks 60–99 and a 20 km buffer from all
  evaluation surveys.
- Freeze checkpoints, the pooled calibration-selected policy, assessment prediction hashes, and
  the official-test submission hash before reading assessment labels.

## Candidate

The v27 challenger uses one deeper model per outer fold and three independent deployment seeds:

- depthwise residual feature pyramids with squeeze-excite for raw Sentinel, Landsat, and
  bioclimatic tensors;
- seven Sentinel channels: four normalized official bands plus derived NDVI, NDWI, and EVI;
- a separate vector-feature branch and learned four-modality attention;
- independent and low-rank joint 5,016-species heads plus an auxiliary richness head;
- random Sentinel flips/quarter rotations in training and deterministic four-view Sentinel TTA;
- a selection-only gradient-boosted oracle-count regressor;
- calibrated rank-list fusion around the matched/exact v26 prediction, including stronger
  rank/count policies because v26 selected the boundary of its previous grid.

The `control` policy is an exact no-op. On the hidden test it reproduces the embedded v26 list;
on fresh folds it reproduces a matched v26 recipe fitted without assessment labels.

## Submission gate

The notebook marks v27 eligible only if all integrity checks pass and the frozen assessment has:

- positive pooled gain and positive lower spatial-bootstrap bound;
- positive gain in both folds and in at least two substantial countries;
- no material common-species or cardinality-MAE regression;
- no severe rare-species collapse; and
- a non-control policy selected without assessment labels.

If the gate fails, the exact v26 artifact remains the official best and the generated v27 CSV
must not be submitted.

## Runtime and deliverables

One Kaggle GPU is required. Models run sequentially, only one is resident on the GPU, and the
hard wall-clock guard is 10.75 hours with 35 minutes reserved for finalization. The v26 reference
runtime was 1.7864 hours; the registered v27 range is 3.0–9.5 hours. Large feature memmaps and
checkpoints are deleted even after late failure.

The self-contained notebook is
`notebooks/geolifeclef_v27_multiscale_shift_aware_ensemble.ipynb`. It is below Kaggle's 1 MB
source limit and leaves exactly four compact files in `/kaggle/working/v27_export`:

- `GLC25_PA_submission_v27.csv`
- `assessment_per_survey_v27.csv`
- `v27_report.json`
- `v27_manifest.json`
