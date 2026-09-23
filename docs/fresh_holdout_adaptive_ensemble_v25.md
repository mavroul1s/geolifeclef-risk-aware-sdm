# v25 fresh-holdout adaptive ensemble

## Status and objective

This is the preregistered successor to the scored v24 submission (public 0.22397,
private 0.20094). The competition target is 0.23021 private F1. v25 must improve
on a genuinely fresh spatial assessment before its CSV is eligible for submission.

## Evidence motivating the change

- v24 improved the official private score by 0.00364 over v23.
- Its internal predictions had mean cardinality 25.14 versus a true mean of 17.01,
  because the old 28-species rule could change by at most three.
- Removing the v24 richness adjustment lost 0.00341 F1, making count selection the
  largest measured component and the clearest target for correction.
- The union of v21-v24 assessment sets contains 55,325 survey IDs. v25 embeds this
  exact union and excludes every one from assessment, leaving 33,662 never-assessed
  official training surveys.
- The official GeoLifeCLEF working note reports useful results from multimodal
  Sentinel-2, bioclimatic and Landsat inputs plus threshold-based inference. v25
  therefore evaluates fixed probability thresholds alongside learned top-k counts:
  <https://ceur-ws.org/Vol-4038/paper_255.pdf>.

## Frozen protocol

1. The exact scored v24 predictions are embedded losslessly and reconstructed to
   submission SHA-256
   `31ce8fcc93d5831f1ecfdffb255c5eec14f0b8a40981f2f16f2ab6cf4b45a111`.
2. Two outer assessment folds use only never-assessed survey IDs, disjoint one-degree
   spatial blocks, and a minimum 20 km training buffer.
3. Previously consumed IDs may be used for checkpoint selection and policy calibration;
   no v25 assessment labels may be used for either.
4. The matched v24 recipe is refit on each fresh fold to provide the internal control.
   This is recipe-transfer evidence; only the official-test control is bit-exact v24.
5. The v25 candidate averages two independently seeded, wider multimodal models trained
   sequentially. Peak VRAM does not contain both trainable models at once.
6. A gradient-boosted count model learns the per-survey top-k that maximizes sample F1
   on the selection partition. Calibration compares that model with thresholds 0.10,
   0.15 and 0.20 and with conservative intervention policies.
7. The selected policy and all predictions are frozen before assessment is opened.

## Submission gate

The notebook marks the candidate eligible only if all integrity checks pass, pooled
gain and the spatial-bootstrap lower bound are positive, each fold gains, at least two
substantial countries gain, common- and rare-species group F1 remain within their
registered tolerances, and cardinality MAE improves. A failed gate preserves v24 as
the control.

## Runtime and outputs

The notebook requires one Kaggle GPU, has a 10.75-hour hard guard inside Kaggle's
12-hour allowance, and is expected to finish in roughly 2-5 hours. v24 completed in
0.98 hours; v25 trains twice as many models with moderately wider hidden layers.

Temporary feature arrays and checkpoints are deleted in `finally`. The only successful
outputs are:

- `GLC25_PA_submission_v25.csv`
- `assessment_per_survey_v25.csv`
- `v25_report.json`
- `v25_manifest.json`
