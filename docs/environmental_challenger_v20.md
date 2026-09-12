# Environmental challenger: preregistered run

Date: 2026-09-12. One notebook and the existing private Kaggle kernel; API mode `environmental_challenger`.

## Motivation and evidence

Our previous best official scores were 0.21594 public / 0.18900 private (v18). Version 19 scored 0.19891 / 0.17516. Its policy was calibrated before a subsequent full-data fine-tune, and it lacked an untouched assessment of the chosen policy. Those are protocol weaknesses, not proof of which change caused the regression.

The public [2025 Model GeoLifeCLEF notebook](https://www.kaggle.com/code/lonansyayf/2025-model-geolifeclef) and [working note](https://ceur-ws.org/Vol-4038/paper_255.pdf) are a concrete comparator, not a winning notebook. Its ImageNet initialization is not imported because this repository's competition-only initialization constraint remains in force.

[PredComX](https://ceur-ws.org/Vol-4038/paper_261.pdf) uses soil/elevation/land-cover covariates and complementary neural/statistical models; its best ensemble scores 0.22153 private. It also uses external covariates/pretrained representations and a much larger compute budget, which this run does not reproduce. [Tighnari](https://arxiv.org/abs/2602.08282) combines modalities and PO/PA training, reaching 0.21689 private. The official 2025 winning target is 0.2302; exceeding it would establish a win over that competition result, not universal current SOTA across different datasets/protocols.

## Fixed design

- All PA training surveys are partitioned using a fixed half-degree coordinate-block hash: nominally 85% training, 5% checkpoint selection, 5% policy calibration, 5% final audit. Italy and Switzerland are assigned wholly to audit. Actual counts and survey-ID hashes are saved. This is a spatial-block protocol, not a minimum-distance exclusion buffer.
- All observed PA species remain in the output vocabulary. Neither audit nor official test labels train the model or normalization. The PA label vocabulary is defined from the supplied training file; some species can have no positives in the actual model-training partition.
- Every EnvironmentalValues train/test CSV pair is aligned by surveyId with duplicate/missing-ID checks. Constant/all-missing features are filtered on training only; missing indicators and normalization use training only. No PO source is accidentally joined as PA features.
- Shared inputs: Landsat 84x6, climate 228x4, Sentinel 4x32x32, metadata. Time-series normalization is position-specific; Sentinel uses fixed reflectance scaling, preserving magnitude rather than per-image per-band stretching.
- Reference: previous CompetitiveFusionSDM architecture, but with this run's common preprocessing/splits. This is **not an exact v18 reproduction**.
- Challenger: environmental residual MLP, position-preserving temporal residual MLP, compact Sentinel encoder, fused classifier and two classification-only specialist heads. No cardinality loss or unvalidated neighbor boost. Two fixed seeds: 2025, 3407.
- Each run requests 24 epochs with the same optimizer schedule and checkpoint selection by fixed top-18 sample F1. A model can stop after at least eight epochs for registered patience/time limits; actual epochs and parameters are reported. One-seed vs one-seed comparison is reported separately from the ensemble.
- Frozen-checkpoint calibration searches 125 predetermined combinations (five ensemble weights, 25 output policies), including zero challenger weight and top-18. No final audit feedback selects a candidate. No training follows calibration.
- Audit reports per-survey F1, per-country F1, country macro F1, Italy/Switzerland OOD F1, and the selected-minus-reference difference. This one split does not establish statistical robustness or novel contribution.
- Exactly the calibrated frozen predictors generate official test outputs. CSV rows follow the official template, with nonempty unique species lists of at most 50 species. A SHA-256 digest accompanies the output.

## Runtime and artifacts

One GPU T4 request; batch 128; mixed-precision neural operations with float32 loss. Prepared arrays are disk-backed rather than eagerly copied into each worker. The 10.5-hour limit includes source setup, tests, preparation, training and inference. A parent process timeout is a backstop; stage checks reserve final inference time. Failure to finish eight epochs for a model is an error, not a silently accepted partial experiment.

Outputs live in `artifacts/environmental_challenger`: three best checkpoints; per-model histories and calibration/audit/test probabilities; `data_manifest.json`; `frozen_policy.json`; per-survey audit; `challenger_report.json`; `GLC25_PA_submission.csv`. Prepared array caches are removed only after final outputs are written; inference probabilities remain for independent audit.

No automatic leaderboard submission occurs in this notebook. Check completion, untouched audit, IDs, digest and frozen policy before the next authorized official evaluation. Expected wall time is uncertain until Kaggle throughput is measured. SOTA is not promised, and PO representation training remains a potential later experiment rather than an untested claim about this one.

## Completed result

Kaggle kernel `con1los/geolifeclef-risk-aware-sdm-phase-1`, version 20, completed on 2026-09-12 in 0.8682 hours. The frozen 75% challenger / 25% reference ensemble with top-20 output scored **0.21601 public / 0.19360 private**. This is the repository's best official private result so far: +0.00460 absolute (+2.43% relative) over v18's 0.18900, but 0.03660 below the 0.2302 winner target. The official result confirms the direction of the independent audit improvement, though the hidden-test gain was smaller.

The standalone single-seed challenger underperformed the matched reference on the untouched internal audit (0.30724 versus 0.31413); only the frozen ensemble won (0.32342). Therefore the result supports the ensemble, not a claim that the challenger architecture alone is superior. The v20 audit is now observed and must not be reused as an untouched v21 model-selection set.
