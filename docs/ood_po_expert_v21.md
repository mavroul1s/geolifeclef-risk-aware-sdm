# v21: preregistered competition-only PO/OOD expert

Date: 2026-09-13. Continue the completed v20 in the existing private Kaggle kernel and the single master notebook, `notebooks/geolifeclef_research_pipeline.ipynb`. This document is frozen with the source commit before execution and before inspecting the new final assessment.

## Objective and scope

The current official best remains v20: **0.21601 public / 0.19360 private**. The recorded winning private target is 0.2302, a gap of 0.03660. Internal sample F1 values are not hidden-test scores, and this experiment does not promise SOTA.

Learn a small environmental/location expert from the competition PO observations and combine it conservatively with v20. The intended benefit is transfer to locations with little PA training support, including Bulgaria, Ukraine and Switzerland. Use only GeoLifeCLEF 2025 PA, PO and competition-provided predictors; no external data or pretrained weights. Retain the sorted vocabulary of all **5,016 PA species**, including species with no positives in a particular training partition.

Inspect the actual Kaggle metadata and predictor schemas before finalizing the executable architecture. Save a schema report containing file names, row counts, column names, missingness, ID uniqueness and PA/PO predictor compatibility. Do not infer publisher, ecological-stratum or time information from a column that is not actually present. The committed executable configuration is the authority for the exact expert dimensions, sampling caps, epoch limits and small calibration grid; record all of them in the run report.

## Why two compatible training paths are necessary

The exact v20 checkpoints have already trained on the old training partition and used the old checkpoint-selection and calibration partitions. The complete old audit has also been examined. Rehashing these rows cannot make predictions from those checkpoints an untouched assessment.

Use a **new matched-v20 recipe control** for the new assessment: train the same reference and two challenger architectures with their original seeds, the new training-only preprocessing, and the new checkpoint-selection partition. Freeze their mixture at the v20 weights: 25% reference, 37.5% challenger seed 2025 and 37.5% challenger seed 3407, with top-20 predictions. This scientifically necessary refit establishes a compatible new control; it is not a retraining merely to recover missing artifacts and is not the exact version-20 predictor.

For production, retrieve only the necessary version-20 artifacts and reuse the exact frozen original predictor. Train the production expert using the original v20 training partition and select its checkpoint and calibration policy on the corresponding original heldouts. Those heldouts are now development data. Preserve the original v20 preprocessing and model weights, verify artifact hashes and survey/species order, and verify that zero expert weight reproduces the unchanged-v20 control.

The evaluation and production experts follow the same preregistered training and calibration procedure, but have separate compatible training partitions, normalization statistics, checkpoint selection and calibration predictions. Freeze **both** paths, their selected policies and production test predictions before opening the new final assessment. No model change, fine-tuning, gate change or recalibration follows that assessment. The resulting gate assesses the training/selection procedure against a matched frozen control; it does not directly measure the exact deployed checkpoint ensemble on unseen labels. This transfer limitation must remain explicit in the results and handoff.

## New geographic partition and separation

Only surveys from the original v20 **training** partition are eligible for the new evaluation protocol. Thus the old v20 audit, including all of its Switzerland and Italy observations, is absent from the new final assessment. No old audit subset is relabeled as untouched.

Assign entire one-degree latitude/longitude cells using a fixed salted hash, seed 20250921, into nominal 70% training, 10% checkpoint selection, 10% calibration and 10% final assessment. Use finite coordinates and a stable integer cell/hash implementation; sort survey IDs when computing partition digests. Partition decisions may use coordinates and the old partition assignment, never species labels or scores. Do not retry split seeds after seeing performance.

Remove candidate PA training surveys within **20 km great-circle distance** of any checkpoint-selection, calibration or final-assessment PA survey. These buffer rows are excluded, not reassigned to a heldout role. Save the partition and excluded-row IDs, hashes, counts, independent cell counts and measured nearest-training distances. A hash of geographic cells by itself is not a distance buffer.

Checkpoint selection is the only PA heldout score read while fitting each model. Calibration is read only after the relevant checkpoints are frozen. New final-assessment targets are scored only after both evaluation and production policies and production predictions have been persisted. Failure of disjointness, the distance buffer, finite-data checks, full vocabulary preservation or sufficient usable samples is a failed run, not permission to silently change the design.

The new assessment measures sparse-PA geographic transfer across its actual countries and distance strata. Since Switzerland was wholly assigned to the consumed v20 audit, it provides no new untouched Swiss estimate. Bulgaria and Ukraine may have too few or no labeled eligible surveys. Save exact counts and report these limitations rather than presenting small-country diagnostics as a reliable assessment of those countries.

## PO construction and training

Stream the approximately five million competition PO records, restrict labels to the complete PA vocabulary, and deduplicate repeated occurrences before constructing pseudo-surveys or a masked representation. Group records using the available location and environmental information. Cap or reweight publisher and local sampling dominance where those metadata exist; save raw, valid, deduplicated and retained counts, species coverage, group support and sampling weights. Sampling must be deterministic from the committed seed.

Unobserved PO species are not reliable absences. Use a masked or explicitly down-weighted unobserved-label objective, with its weights saved in the report. A compact environmental/location network is preferred to loading millions of imagery patches. The PO expert may learn from independent competition PO records in PA-heldout geographic regions: this is the intended scarce-PA transfer setting. It must never train on heldout PA labels. Detect source-ID overlap when the real schema establishes a shared namespace and reject direct duplicated PA observations without consulting heldout species labels; document the implemented duplicate-exclusion rule and counts. Report remaining uncertainty about shared source observations.

The v19 nearest-96-raw-record heuristic is not used. Fit every PA-derived normalizer, frequency statistic and training-support feature on its own training partition. Any PO normalization or support statistic uses only the permitted PO training pool. All gates use features available both for validation and for the official test, such as distance to the matching PA training coordinates and PO support; no test labels or evaluation outcomes influence them.

Train a zero-PO expert with the same architecture and PA training/checkpoint-selection procedure, omitting the PO contribution. Record compute and exposure for both; a PO benefit is a data/training benefit, not an isolated architecture claim. Save the standalone expert diagnostics as well as the mixtures.

## Calibration, controls and the one-submission gate

Use the fixed mixture-strength grid **0, 0.025, 0.05, 0.10, 0.15, 0.20**, with uniform and logarithmic PA-distance gates. The logarithmic distance transformation is fixed in the committed implementation; it is not fitted to assessment scores. All candidates keep **top-20** output, including exactly unchanged v20 at zero expert weight. Choose policies solely from the appropriate calibration partition, with deterministic tie-breaking that prefers less deviation from v20. Persist the full candidate list and selected policies before assessment. Evaluation and production calibration may select different policies, as they use separately fitted compatible predictors; report the difference transparently.

Required controls are the frozen matched-v20 recipe, the unchanged exact v20 production predictor, and the separately trained zero-PO expert/mixture. The exact v20 predictor must not be scored on its own former training rows and described as the untouched control. A zero-weight candidate is the control and cannot count as a new method beating it.

Report new assessment sample-averaged F1 first, paired gain over the compatible frozen control, zero-PO comparison, country and PA-distance strata, rare/common species performance, prediction cardinality, runtime and uncertainty. Use spatial-cell resampling for paired uncertainty and disclose that one partition does not establish robustness across independent geographic folds. Do not select a candidate from assessment country or distance results.

Make at most **one official submission**, only if the already-frozen PO candidate's paired sample-F1 gain over the compatible frozen-v20 control has a **strictly positive lower 95% spatial-cell-bootstrap confidence bound**, its mean sample F1 also exceeds the calibrated zero-PO mixture, and all source, split, artifact, output and runtime checks pass. This is deliberately stricter than a positive point estimate. If the PO component is zero, the assessment is invalid, any integrity check fails, or either comparison fails, save the outcome and do not submit. Do not make a second attempt based on the resulting leaderboard score.

Verify the official output against the template: exactly 14,784 unique survey IDs in template order, all predictions within the unchanged 5,016-species vocabulary, nonempty unique species lists of at most 50 entries, finite underlying probabilities and a saved SHA-256 digest. Retrieve public/private scores if Kaggle exposes them and save the actual values and availability status; never infer a private score from the public score.

## Budget and completion

One Kaggle T4; a hard **10.5-hour total** deadline includes notebook setup, tests, preparation, matched-control training, evaluation/production expert fitting, inference, assessment and artifact writing. The planning target is about 3.5 hours, not a runtime guarantee. Use bounded streaming/memory maps, enforce the remaining stage budget and reserve time for final inference and integrity checks. A parent-process timeout is the backstop. Failure to meet registered minimum training requirements fails the run.

Use `apply_patch` for repository edits, run meaningful local tests and notebook tests, and commit the tested source before the existing API push because it uploads Git HEAD. Keep one master notebook; supporting scripts are allowed. Credentials are read only in memory and never printed, committed, copied into notebook source or exposed through signed URLs.

Save schema and split manifests, PO support/duplicate diagnostics, all checkpoint and calibration histories, frozen evaluation/production policies, assessment predictions and per-survey results, controls, artifact digests, runtime and the submission decision. Update `results/experiment_registry.json`, the v21 summary and handoff with the completed outcome. Monitoring continues until the Kaggle run completes or fails; an executable notebook alone is not evidence of improvement.
