# Follow-up D1–D3 — 14 November 2026

Authorized after the R1 evaluation. This is an exploratory development follow-up
designed after R1 assessment inspection, not independent preregistration. The
configuration and inference notebook were recorded before new diagnostic results
in `results/reliability_followup_d1_d3/protocol_freeze.json`.

## Scope

Complete the first three studies in [the proposed budget](next_experiment_budget.md)
using existing artifacts and six immutable R1 checkpoints. **Zero new neural
training.** A private inference-only Kaggle notebook reconstructs training-fold
preprocessing, loads the saved weights and saved Platt coefficients, exports
per-member decision diagnostics, normalization objects and float32 ensemble means,
and checks reproduction of R1 outcomes. No test predictors or submission are used.

The fixed configuration is `configs/reliability_followup_d1_d3.json`. The private
job is [checkpoint diagnostics](https://www.kaggle.com/code/nmavros/glc25-r1-checkpoint-diagnostics).
It attaches the completed private R1 output as a kernel input. All checkpoint and
input digests and role ID hashes are verified. A synthetic 5,016-label integration
test exercises saved-checkpoint loading, normalization reconstruction and output
reproduction with training replaced by a function that raises an error.

Inference uses the unchanged v29 prediction path, including its float16 raw
network probabilities, followed by the saved probability calibration and float32
ensemble accumulation. This restores the precision lost by R1's final float16
ensemble export; it is not a newly trained or higher-precision neural model.
Tie handling is explicitly descending probability then ascending species column.
Legacy member-set reproduction is checked separately using the original rule.
Admission for analysis requires at least 99.9% per-survey F1 reproduction and
absolute mean difference <=0.0001, plus >=99.9% original member-set matches.
Any numerical difference is recorded rather than called bitwise reproduction.

**Technical amendment:** the first inference attempt completed in 36.49 minutes
but failed the literal member-set guard (about 99.4–99.6% set agreement), despite
over 99.9% ensemble per-survey F1 agreement and mean differences around 1e-5.
Its failed status is preserved. Before inspecting new D1/D2 reliability contrasts,
one aligned rerun restored the original inference-role batch boundaries and
frozen-EMA parameter flags, reused the exported normalization objects, and read
imagery only for the 49,077 inference-role surveys. The scientific configuration
and all F1/regional success margins stayed fixed. See
`configs/reliability_inference_alignment.json` and `alignment_amendment.json`.

The rerun completed in 24.88 minutes. Literal member-set equality still fails;
**it is not relabeled as an exact reproduction pass**. A separate label-free check
found >=99.9469% of reference sets within two raw-float16 units at the top-18
boundary, passing the recorded numerical-sensitivity criterion. Ensemble F1
matches on >=99.9113% of assessment surveys, with fold mean changes under
0.000014. Analyses of regenerated scores are explicitly exploratory with numerical
tie sensitivity. D2's main full-coverage averaging comparison uses immutable
original R1 outcomes and original member sets, independent of regeneration.
This technical amendment is post-R1 and does not create independent confirmation.

## D1: decision confidence

Compare fixed top-18, one global calibrated threshold with sizes bounded to 8–40,
and the existing expected-F1 ratio surrogate with sizes 8–40. The threshold is
selected from the declared 19-value grid using only the probability-calibration
role. It can share that role with Platt calibration because the separate
risk-calibration role fits the selector and assessment remains separate. This
does not undo historical data reuse.

For each decoder, compare seeded random acceptance, mean predicted-set confidence,
predicted-set entropy, metadata/distance regression and a lean error regressor.
The lean regressor uses metadata, distance, total probability mass, maximum
probability, mean predicted-set confidence, an expected-F1 surrogate and predicted
set size. Each decoder has its own calibration loss and confidence features.
Hyperparameters are the fixed R1 HistGradientBoosting values. True richness and
rare-species outcomes never enter these prediction features.

Report full-coverage F1, count MAE, risk/coverage, entropy/confidence correlations,
rare-occurrence coverage and rare recovery. The 80% reporting comparison uses
exactly the same fold/country/area quotas. Decoder changes are compared on the same
surveys; selector changes within a decoder use the same fixed predictor.

## D2: averaging versus disagreement

Evaluate every individual seed and the ensemble under the same decoders and
calibration/assessment roles. The main averaging diagnostic compares ensemble
full-coverage F1 with the arithmetic mean of the three individual-model F1 values,
including a paired spatial interval. This is expected utility if an individual
member were chosen uniformly, not the F1 of another prediction set. Do not choose
the best member using assessment labels and present it as an available policy.

The mean ensemble is one three-member ensemble per fold. Members are not three
independent ensemble trials. Separate this averaging question from R1's
post-hoc finding that disagreement features add no clear value once confidence
features are present. Inference time includes recomputing all six members and
feature preparation, not a direct isolated per-member production benchmark.

## D3: regional and ecological utility

Use the exact saved R1 top-18 outcomes and role confidence tables. Compare four
fixed variants: pooled lean risk, inverse-square-root country-weighted lean risk,
country-local model blending and biogeographic-region model blending. Local
models require at least 100 risk-calibration surveys and are shrunk toward the
pooled model with weight n/(n+200). Sparse/unknown groups use the pooled model.
These are simple diagnostic baselines, not asserted novel algorithms.

Choose one variant using three geographic inner folds of risk-calibration data,
maximizing the average supported-country macro selective F1. Geographic blocks
remain intact, ties choose the declared simpler ordering, and no assessment
outcomes choose a model or parameter. The risk-calibration IDs are shared across
the two outer model folds; these are two prediction views of the same calibration
surveys, not twice as many independent observations. Final fits use all assigned
risk-calibration labels.

Report every variant and the calibration-chosen one against pooled lean and
simple confidence. Keep R1's +0.005 utility and 0.005 supported-country regression
margins for context; the failed R1 result is not rewritten. Country-specific
constant risk offsets cannot alter acceptance ranks within fixed country/area
quotas; an explicit negative control verifies this.

Rare taxa use 1–25 positives in the neural training fold. Report accepted rare
occurrences, true positives, conditional recall and recovery relative to the full
cohort. Referral costs 0/0.25/0.5/1 are hypothetical units of sample prediction
loss, assuming zero residual prediction loss after referral. They are sensitivity
analyses, not measured field costs or conservation benefits.

## Independent-data workstream

Audit metadata and provenance before inspecting any external species outcomes.
Current GeoPlant test variants were already shown to reuse old test IDs. Inspect
public sPlotOpen/SAGE survey metadata, compare geography against all existing PA
train/test locations, preserve source identities and cell aggregation, and verify
available predictors and release inventories. Candidate selection uses covariates
only. Species labels and released predictions remain unopened until a specific
external protocol is frozen.

SAGE is relevant prior work rather than evidence of our novelty. Its species-level
sampling-aware framework and plot-size sensitivity overlap the broad proposed
story. Our specific candidate question concerns site-level species-set decisions,
selective reliability and retained geographic/ecological coverage. A missing
keyword in a paper is not a sufficient novelty check.

## Execution records

`results/reliability_followup_d1_d3/` contains the protocol snapshot, actual job
state, regional analysis and independent-data audit. Large regenerated matrices
and role tables are ignored under `inference_output/`; compact reports, manifest
hashes and derived tables are retained separately. Current state is recorded in
`README.md` and the maintained Obsidian thesis roadmap; planned stages are not
reported as completed.

The local analysis environment uses Python 3.12.16 under ignored project storage
`artifacts/python-runtime/`, with the existing `.venv` linked to it. This replaces
a vanished temporary interpreter without changing system Python. The NumPy,
PyTorch, pandas and scikit-learn versions remain 1.26.4, 2.5.1+cpu, 2.2.3 and 1.5.2
for local analysis. Kaggle inference uses its recorded T4/PyTorch 2.11.0 image.
