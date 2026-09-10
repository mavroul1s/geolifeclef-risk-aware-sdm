# Experiment plan

## Hypotheses

1. A per-species frequency baseline exposes the long-tail difficulty.
2. A compact Landsat seasonal temporal CNN can beat static pooling at low compute.
3. Climate and static soil/elevation/land-cover/human-footprint features add ecological complementarity.
4. Gated fusion can improve F1 per parameter relative to naive concatenation.
5. Calibration plus validation-only adaptive thresholds can make small field-survey species sets more useful.
6. Masked reconstruction or contrastive pretraining must use only available competition predictors and be compared against an equal supervised budget.

## Sequence

1. Audit and data card.
2. Frequency baseline.
3. Landsat-only temporal CNN.
4. Climate-only and static-only baselines.
5. Lightweight gated fusion; optional image patches only as ablation.
6. Calibration/adaptive thresholds and uncertainty analysis.
7. Self-supervised pretraining plus fine-tuning.
8. Compute and ablation study on the 2x T4 16 GB envelope.

## Reporting and risks

Fix and save seeds/configs; retain checkpoint, CSV/JSON history, package versions, parameter count, peak GPU memory, throughput, and training time. Report macro/micro F1, prevalence strata, precision/recall, calibration error/Brier score, and candidate-set size. Do not claim conformal guarantees unless assumptions are verified.

Key risks: spatial leakage (use blocks where metadata permits), class imbalance (class-balanced/asymmetric losses plus transparent strata), missing time steps (audit and document imputation/masks), and threshold overfitting (separate calibration set).

