# GeoLifeCLEF Risk-Aware SDM

Reproducible PyTorch research code for multi-label plant-species presence prediction using **only** the GeoLifeCLEF 2025 Kaggle competition data. Phase 1 supplies an audit, a prevalence baseline, and a Landsat temporal-CNN baseline. It does not claim any score is state of the art without independent evaluation.

## Research protocol

The published single-modality Landsat reference is approximately F1 0.144. This repository treats a reproducible, documented score above that as the initial target; 0.16--0.18 is a realistic research target, not a result. Report micro-F1 and macro-F1, threshold policy, common/rare strata, calibration, predicted-set size, parameters, memory, throughput, and wall-clock time. Rare species remain in aggregate metrics.

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

