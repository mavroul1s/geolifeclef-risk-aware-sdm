# v26: raw-spatial raster ensemble

## Objective

v25 is the current official best at 0.22812 public / 0.20503 private, but its internal
ablation shows that adaptive cardinality produced almost all of the gain. The summary-only
satellite representation did not supply enough independent ranking information. v26 tests a
larger architectural change while preserving the exact scored v25 prediction as the official-test
control.

This is a candidate for the competition target of 0.23021 private F1, not a claim that the target
has already been reached. Only the official hidden-test score can establish that result.

## Registered candidate

- Inputs: only the official `geolifeclef-2025` competition data; no external data or pretrained
  weights.
- Raw encoders: residual CNNs over Sentinel `4x32x32`, Landsat `6x4x21`, and monthly bioclim
  `4x19x12` arrays.
- Vector branch: the existing environmental, static, and remote-sensing summary features.
- Output: an independent 5,016-species head plus a rank-96 joint-species head and richness head.
- Training support: the raw-raster candidate learns species with more than five occurrences in
  the fold training partition. Rare v25 predictions are pinned during list fusion, so the new
  model cannot erase them merely because its own head was filtered.
- Ensemble: calibration selects among exact-control, 10/20/30% spatial rank blends, and two
  conservative count-aware blends. Deployment averages two independently seeded raw-raster
  models.

The design follows the strongest transferable result in the official working notes: successful
teams trained directly on spatial Sentinel/Landsat/climate tensors and benefited from diverse
list-level ensembling. It deliberately does not copy external Prithvi, WorldCover, or CHELSA
inputs used by some teams.

## Leakage control

The notebook embeds the immutable union of 69,630 survey IDs assessed by v21 through v25.
Neither v26 assessment fold may contain any of them. With the official 88,987 unique PA training
surveys, 19,357 never-assessed surveys remain.

The fixed split produces:

| Fold | Fresh assessment | Assessment blocks | Training | Selection | Calibration | Minimum buffer |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1,236 | 21 | 39,990 | 3,297 | 6,336 | 20.008 km |
| 1 | 2,621 | 21 | 38,651 | 3,297 | 6,336 | 20.002 km |

The candidate policy is selected only on calibration rows. All policies and predictions are
frozen before either assessment label set is scored. The exact v25 submission is embedded as a
losslessly compressed payload and verified against SHA-256
`c450107d5219bb3a99f37741142ea40cf9320c87e851173ec50f1e899ada48c5`.

## Submission gate

The notebook marks the candidate eligible only when:

- pooled assessment gain and the lower 95% spatial-block bootstrap bound are positive;
- both folds improve and gains are not confined to one country;
- common-species F1 is protected and rare-species F1 does not severely collapse;
- cardinality MAE is not materially worse than matched v25;
- every integrity, schema, hash, separation, and runtime check passes; and
- calibration selected a non-control spatial policy.

If the gate is false, the exact v25 result remains the control and the v26 CSV should not be
submitted.

## Runtime and output contract

The notebook has a 10.75-hour hard guard for a 12-hour Kaggle session. Expected runtime is
3–9.5 hours on one T4. Models are trained sequentially, peak VRAM is designed to remain below
6 GB, feature preparation is capped at 2.75 hours, and 35 minutes are reserved for validation
and export.

Temporary raster caches, probability arrays, and checkpoints are deleted even after a late
failure. A successful run leaves exactly four compact files in `/kaggle/working/v26_export`:

- `GLC25_PA_submission_v26.csv`
- `assessment_per_survey_v26.csv`
- `v26_report.json`
- `v26_manifest.json`

The generated notebook is `notebooks/geolifeclef_v26_raw_spatial_raster_ensemble.ipynb` and is
677,705 bytes, below Kaggle's 1,000,000-byte kernel-source limit. Kaggle must be started with
`Session options -> Accelerator -> GPU T4 x1`; the notebook now checks this before reading any
competition rasters and stops immediately with corrective instructions if CUDA is unavailable.

## Evidence used for the architectural change

- Official second-place working note: <https://ceur-ws.org/Vol-4038/paper_261.pdf>
- Official raw-raster/Swin working note: <https://ceur-ws.org/Vol-4038/paper_255.pdf>
- Official GeoLifeCLEF 2025 overview: <https://ceur-ws.org/Vol-4038/paper_234.pdf>
