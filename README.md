# GeoLifeCLEF Risk-Aware SDM

Reproducible PyTorch research code for multi-label plant-species presence prediction using **only** the GeoLifeCLEF 2025 Kaggle competition data. The repository includes diagnostics, spatial validation, and a compute-aware multimodal Sentinel/Landsat/bioclimatic model. It does not claim any score is state of the art without like-for-like hidden-test evaluation.

## Research protocol

The primary external target is the GeoLifeCLEF 2025 winning private-leaderboard sample-averaged F1 of 0.23021. Internal scores are candidate-selection evidence only and are never compared numerically with the hidden-test result. Report sample-averaged F1 first, plus micro/macro F1, prediction policy, common/rare strata, calibration, predicted-set size, parameters, memory, throughput, and wall-clock time. Rare species remain in aggregate metrics.

Use a spatially blocked validation split when coordinates, tiles, regions, or habitats permit it. Otherwise use the official split and explicitly record why spatial blocking was unavailable.

## Setup

Python 3.10--3.12 is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Kaggle authentication and minimal download

The downloader checks authentication without printing values. It prefers `KAGGLE_API_TOKEN`, then `KAGGLE_USERNAME` plus `KAGGLE_KEY`, then user-local `~/.kaggle/kaggle.json`. Never place credentials in this repository.

Accept the rules while signed in on the [GeoLifeCLEF 2025 competition page](https://www.kaggle.com/competitions/geolifeclef-2025). Then list files, review the conservative selection, and download only the needed Presence--Absence and predictor files:

```powershell
python scripts/download_data.py --list-only
python scripts/download_data.py --dry-run
python scripts/download_data.py
```

Files land in ignored `data/raw/`. If access is unavailable, the downloader stops rather than trying to bypass authentication or rules.

### Direct Kaggle execution

To create/update a private Kaggle script kernel and start a Phase-1 run directly through Kaggle's API (without browser interaction), run:

```powershell
.\scripts\push_kaggle_kernel.ps1
.\scripts\push_kaggle_kernel.ps1 -RunMode schema
.\scripts\push_kaggle_kernel.ps1 -RunMode frequency
.\scripts\push_kaggle_kernel.ps1 -RunMode landsat_smoke
.\scripts\push_kaggle_kernel.ps1 -RunMode sota_spatial_smoke
.\scripts\push_kaggle_kernel.ps1 -RunMode sota_spatial_full
.\scripts\push_kaggle_kernel.ps1 -RunMode sota_single
.\scripts\push_kaggle_kernel.ps1 -RunMode environmental_challenger
.\scripts\push_kaggle_kernel.ps1 -RunMode retained_po_v22
```

The script keeps credentials in memory only, uploads a Git archive of **HEAD** (commit intended source changes first), and attaches `geolifeclef-2025`. It also supports the explicitly user-authorized ignored `api_key/kaggle_2.json`. Audit/schema/frequency runs are CPU-only; neural modes request a T4.

The current official best is v32: **0.24225 public / 0.21514 private**. It gained
**0.00153 public / 0.00213 private** over v31 and completed in **5.7681 hours on T4**.
The remaining private-score gap to the competition winner is **0.01507**, so SOTA
is not established. Its five original outputs total **7.16 MB** and are preserved
under `results/v32_kaggle_output/`, with screenshot, hashes and diagnostics in
`results/v32_summary.json` and the history in `results/experiment_registry.json`.

V28 did **not** improve: it selected `control` and emitted the exact same CSV as v27,
with the same public/private scores. Its report correctly said
`eligible_for_submission: false`. Evidence is in `results/v28_summary.json` and
`results/v28_kaggle_output/`; the four original outputs and screenshot are preserved.

V30 regressed to **0.23451 public / 0.20942 private**, despite positive pooled internal
gain. Outside Denmark/Netherlands its internal gain was negative. The cause cannot be
isolated because training-data coverage, model mix, epochs and species counts changed
together. Its five original outputs and screenshot are preserved in
`results/v30_kaggle_output/`, `results/v30_kaggle_scores.png` and `results/v30_summary.json`.

V33 completed in **5.0072 hours**, but all new policies were rejected. Its returned
CSV was byte-identical to v32, explaining the identical public/private scores.
The new convolution specialist consistently hurt the repeated development checks.
The five original outputs and screenshot are preserved in `results/v33_kaggle_output/`,
`results/v33_kaggle_scores.png` and `results/v33_summary.json`.

V34 regressed to **0.24203 public / 0.21483 private** after a 5.2468-hour T4 run.
Its graph policy changed 6,143 test sets and passed all repeated internal gates,
but the internal gain was only 0.0004300 and the bootstrap lower bound almost zero.
Its original 12.62 MB output and screenshot are preserved in `results/v34_kaggle_output/`,
`results/v34_kaggle_scores.png` and `results/v34_summary.json`. No runtime/schema
defect was established; small development gains did not transfer to hidden tests.

The current candidate is
`notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb`. It trains three
attention models for up to **48 epochs**, with selection-only checkpoint choice,
sensor-balanced pooling, geographic/non-geographic diversity and an auxiliary
per-survey ranking objective. It predicts **complete species sets**, removing the
old fixed-count/protected-prefix ceiling. Calibration chooses among 15 ensemble
and cardinality policies. Counts remain in 8..40 and all 5,016 species stay eligible.

The matched-v32 **development scores** are embedded in a small cache from the v34
run, with metadata, labels, species and every split-role ID hash checked. No old
reference neural models are refit. The budget goes to six new development fits and
at most three full-PA production fits instead. This cache contains no hidden-test
labels or external weights and needs no separate uploaded dataset.

Promotion now requires at least **0.001 pooled gain**, **0.0005 outside-DK/NL gain**,
positive country-macro and both folds, and a bootstrap lower bound >=0.00025.
All 88,987 PA IDs were previously assessed: these are repeated development safeguards,
NOT independent evidence or guaranteed improvement. The new full-set model can
regress; v32 remains best until an actual hidden-test score exceeds it.

The single offline notebook is **396,956 bytes**, uses T4 x1 and only official
competition data. It has a **10.75-hour cooperative guard** and five-file output
cap of **16 MB**. Full v35 runtime is unmeasured; conservative remaining-plan
admission may stop a slow session rather than complete the experiment.

Submit **only** eligible `v35_export/GLC25_PA_submission_v35.csv`. If rejected,
`NO_SUBMISSION.json` replaces the submission CSV, while actual failed-candidate
diagnostics are retained. Never submit ZIP, NPZ or diagnostic CSV files.
See `docs/full_set_balanced_attention_v35.md`, `results/v35_reference_validation.json`
and `results/v35_local_validation.json`.

## Audit and canonical data contract

```powershell
python scripts/audit_data.py --data-root data/raw
```

The audit creates `data/reports/data_card.{json,md}` with inventory, inferred modalities, and actual NPY/NPZ dimensions without loading whole arrays. After inspecting the competition schema, create canonical split files at `data/processed/train.npz` and `data/processed/val.npz`:

```text
labels       float32 [samples, species]          required multi-hot labels
landsat      float32 [samples, timesteps, bands] required for Landsat CNN
climate      float32 [samples, timesteps, vars]  optional
static       float32 [samples, features]         optional
sample_id    string   [samples]                  optional
coordinates  float32 [samples, 2]                optional
```

This adapter boundary prevents unsupported assumptions about the raw competition archive. Version and document each conversion.

## Train, evaluate, and verify

```powershell
python scripts/train.py --config configs/frequency.yaml
python scripts/train.py --config configs/landsat_tcn.yaml
python scripts/evaluate.py --checkpoint artifacts/landsat_tcn/best.pt --split data/processed/val.npz
python scripts/profile_model.py --config configs/landsat_tcn.yaml
pytest
```

Runs save their config, epoch CSV, checkpoint, and JSON metrics under ignored `artifacts/`. Adaptive thresholds may only be fit with `--calibration-split`; never use the final test split for threshold fitting.

## Layout

`configs/` holds experimental choices; `docs/` holds the protocol; `scripts/` provides entry points; `src/geolifeclef/` holds reusable code; `tests/` contains synthetic smoke tests; `notebooks/` is reserved for EDA and final figures.
