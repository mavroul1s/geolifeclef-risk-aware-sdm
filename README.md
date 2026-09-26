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

The current official best is v29: **0.23900 public / 0.21093 private**. It gained
**0.00561 public / 0.00262 private** over v27 and completed in 3.0142 hours on a T4. The
remaining private-score gap is **0.01928**, so SOTA is not established. Exact v29
outputs, hashes, assessment diagnostics and leaderboard evidence are preserved under
`results/`; see `results/v29_summary.json` and `results/experiment_registry.json`.

V28 did **not** improve: it selected `control` and emitted the exact same CSV as v27,
with the same public/private scores. Its report correctly said
`eligible_for_submission: false`. Evidence is in `results/v28_summary.json` and
`results/v28_kaggle_output/`; the four original outputs and screenshot are preserved.

V30 regressed to **0.23451 public / 0.20942 private**, despite positive pooled internal
gain. Outside Denmark/Netherlands its internal gain was negative. The cause cannot be
isolated because training-data coverage, model mix, epochs and species counts changed
together. Its five original outputs and screenshot are preserved in
`results/v30_kaggle_output/`, `results/v30_kaggle_scores.png` and `results/v30_summary.json`.

The current candidate is
`notebooks/geolifeclef_v31_full_data_bagged_habitat_residual.ipynb`. It restores all-PA
production training, adds independent-seed v29 replicas and an environmental-neighbour
expert, and preserves **every scored-v29 row count and its leading 60% species**.
At most 2/4 tail substitutions are allowed. It requires positive country-macro and
outside-Denmark/Netherlands gains in addition to the two-fold/spatial-bootstrap gates.
All 88,987 PA IDs have been assessed previously: these remain repeated development
checks, NOT a fresh holdout or proof of SOTA. Production calibration transferred from
v29 to new seeds is an explicit limitation. V29 remains the best actual submission.

The offline single notebook stays below 1 MB, uses one GPU and only official competition
data, has a 10.75-hour cooperative guard, and exports five compact files capped at 16 MB.
V30 took 3.799h on T4; full v31 runtime is **not yet measured**, and its guard may stop
a slow session safely. Submit **only** the eligible `v31_export/GLC25_PA_submission_v31.csv`,
never the ZIP, reports, NPZ or `DO_NOT_SUBMIT` files. If the gate fails, keep v29.
See `docs/full_data_bagged_habitat_residual_v31.md` and `results/v31_local_validation.json`.

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
