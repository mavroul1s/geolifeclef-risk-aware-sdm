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

The current official best is v26: **0.23052 public / 0.20693 private**. It gained
**0.00240 public / 0.00190 private** over v25 and completed in 1.7864 hours. The
remaining private-score gap is **0.02328**, so SOTA is not established. Exact v26
outputs, hashes, assessment diagnostics and leaderboard evidence are preserved under
`results/`; see `results/v26_summary.json` and `results/experiment_registry.json`.

The current candidate notebook is
`notebooks/geolifeclef_v27_multiscale_shift_aware_ensemble.ipynb`. It embeds the exact
scored v26 predictions and excludes all 73,487 survey IDs assessed by v21-v26 from two
new assessment folds. The candidate adds deeper multi-scale raster pyramids, derived
Sentinel spectral indices, modality attention, four-view test-time augmentation, and a
three-seed deployment ensemble around the exact v26 list. The notebook is self-contained,
695,825 bytes, uses only the official competition input, deletes its large temporary cache,
and has a 10.75-hour guard. See `docs/multiscale_shift_aware_ensemble_v27.md`.

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
