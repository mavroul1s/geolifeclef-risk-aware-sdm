# Experiment plan

## Hypotheses

1. A per-species frequency baseline exposes the long-tail difficulty.
2. A compact Landsat seasonal temporal CNN can beat static pooling at low compute.
3. Climate and static soil/elevation/land-cover/human-footprint features add ecological complementarity.
4. Gated fusion can improve F1 per parameter relative to naive concatenation.
5. Calibration plus validation-only adaptive thresholds can make small field-survey species sets more useful.
6. Masked reconstruction or contrastive pretraining must use only available competition predictors and be compared against an equal supervised budget.

## Sequence

1. Audit and raw-schema probe: PA labels are long-format `surveyId`/`speciesId`; Landsat cubes are `[band=6, season=4, year=21]` and bioclimatic cubes are `[variable=4, year=19, month=12]`.
2. Frequency diagnostic: deterministic 80/20 survey-level split; rank species by unique training-survey prevalence; predict the top-k species for each validation survey, where k is the rounded mean training label cardinality. Record the official sample-averaged F1 first, plus micro-F1, macro-F1 over observed validation species, precision, recall and unseen validation species. This deliberately has no feature input.
3. Landsat-only temporal CNN.
4. Climate-only and static-only baselines.
5. Lightweight gated fusion; optional image patches only as ablation.
6. Calibration/adaptive thresholds and uncertainty analysis.
7. Self-supervised pretraining plus fine-tuning.
8. Compute and ablation study on the 2x T4 16 GB envelope.

## SOTA-oriented spatial comparison

Freeze Netherlands as the untouched geographic holdout. Within all non-Netherlands surveys,
reserve deterministic 10 km spatial blocks for calibration. The Landsat reference and competitive
fusion model must share the exact SHA-256 split identifiers, asymmetric loss family, checkpoint
selection metric, and calibration-only prediction-policy search. The candidate combines a dilated
Landsat encoder, dilated bioclimatic encoder, compact ConvNeXt-style Sentinel encoder, Fourier
geographic/static encoder, and cross-modal Transformer fusion. A smoke run must pass before the
full run. The full job has a 10.5-hour internal guard inside the 12-hour Kaggle allowance.

## Reporting and risks

Fix and save seeds/configs; retain checkpoint, CSV/JSON history, package versions, parameter count, peak GPU memory, throughput, and training time. Report macro/micro F1, prevalence strata, precision/recall, calibration error/Brier score, and candidate-set size. Do not claim conformal guarantees unless assumptions are verified.

Key risks: spatial leakage (use blocks where metadata permits), class imbalance (class-balanced/asymmetric losses plus transparent strata), missing time steps (audit and document imputation/masks), and threshold overfitting (separate calibration set).

## Target and confirmed diagnostic result

The primary target is to exceed the GeoLifeCLEF 2025 winning private-leaderboard sample-averaged F1 of 0.2302 under the official evaluation protocol. Internal validation is used to select credible candidates, but only a like-for-like competition submission can establish whether that target has been met.

With seed 2025, 71,190 training surveys, 17,797 validation surveys, 5,016 species and no missing Landsat cubes, the 169,368-parameter Landsat TCN reached top-16 micro-F1 0.2678696. The matched frequency diagnostic was 0.1570004. These historical values use global micro-F1, not the official sample-averaged F1, so they prove only that the pipeline learns useful Landsat signal on an internal random split; they cannot be compared numerically with 0.2302.

Before any further model comparison, select and freeze a country/region holdout using the
registered spatial-audit rule: 8%-30% of surveys, no more than 20% unseen validation labels,
then closest to a 20% validation share. Do not revise this choice after inspecting model F1.
