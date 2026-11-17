# Controlled reliability study R1

Protocol/config frozen at **2026-11-13 17:02:12 UTC**, before new GPU accuracy
results. Technical probe and smoke outcomes cannot change scientific comparisons.
An amendment at **17:22:41 UTC**, before inspecting any new GPU accuracy results,
isolated the primary uncertainty contrast from environmental novelty. The initial
configuration and both digests are retained in `protocol_amendment_1.json` and
`protocol_initial_config.json`. Neural settings and data roles did not change.
Implementation bug fixes are recorded separately. Configuration:
`configs/reliability_study_r1.json`; digest: `results/reliability_study_r1/protocol_freeze.json`.

## Question and interpretation

Does model uncertainty improve survey-level reliability beyond plot area,
metadata and geographic training support, at matched geographic/survey coverage?
This is an initial controlled development experiment on previously consumed PA
labels. It is neither a fresh holdout nor an independent paper confirmation.

## Fixed predictive model

One existing v29 SensorAttention architecture: width 128, three transformer layers,
all 5,016 PA species, geography enabled, positive-weighted BCE (`gamma=0`), EMA,
mixup, Sentinel rotations and whole-sensor dropout. Exactly 18 epochs per fit,
cosine schedule over those same 18 epochs. Last EMA checkpoint; no checkpoint
selection, stopping rule or architecture tuning based on validation outcomes.

Three seeds (2026111301/02/03) in each of the existing two buffered spatial folds:
six full neural fits. No old competitive reference chain is refit. The final
prediction is fixed top-18 from the mean of individually calibrated members.
The old v32 production score is historical context, not a matched comparator.

Original roles are mapped as follows:

| Original role | R1 role | Labels used for |
|---|---|---|
| Training | Training | Neural weights, normalization, feature metric |
| Selection | Probability calibration | Shared Platt slope/intercept for each member |
| Calibration | Risk calibration | Regressors predicting `1 - sample F1` |
| Assessment | Assessment | Final development comparisons only |

This mapping is possible because the neural epoch count/checkpoint is fixed.
It avoids using one set of labels to both calibrate probabilities and fit the
error predictor. Original block hashes and the 20 km buffer remain auditable.
Country/area quotas use assessment covariates, never assessment labels.

## Inputs and exports

Use only official competition PA training predictors: Sentinel 64px, Landsat,
bioclimatic time series and numeric environmental tables, plus allowed survey
metadata. The study does not load hidden-test predictors, train a PO model or
make a competition submission. Environmental missing indicators cover every
provided numeric column. Scaling, imputation/PCA and the nearest environmental
training-neighbor metric are fitted on fold training rows only.

Export full mean calibrated probabilities for risk-calibration and assessment
rows, member top-18 sets, true labels with species vocabulary, complete confidence
and metadata rows, role hashes, training-frequency strata, fit histories,
calibration coefficients and full-run checkpoints. Temporary feature/moment
caches are deleted, including on failure. Output envelope: 1.5 GB; cooperative
full-run guard: 5.5 hours. This larger envelope supports scientific analysis,
replacing the earlier competitive notebook's compact-export constraint for this
new study only.

Entropy, variance, mean confidence, expected-F1 surrogate and pairwise set
disagreement are computed from calibrated ensemble predictions. Disagreement
also includes variation in member-specific probability calibration; it is not a
pure or assumption-free estimate of epistemic uncertainty. All-label entropy and
Brier scores can be dominated by thousands of negative labels, so predicted-set
signals and prevalence-stratified results accompany them.

## Comparisons and primary contrast

Evaluate random ordering, raw PA distance, area-only/metadata/metadata-plus-distance
error regressors, mean predicted-set confidence, predicted-set/all-label entropy,
ensemble variance/information, member set disagreement, environmental novelty,
model-signals-only risk, metadata-plus-model risk and a full risk regressor.
All error regressors use the same frozen HistGradientBoosting parameters and
risk-calibration labels. No assessment outcomes select features or parameters.

Every method is evaluated under the same three acceptance policies:

1. Unconstrained global ordering.
2. Country quotas.
3. Country crossed with plot-area bins, whose edges use training covariates.

At each retained fraction, largest-remainder quotas allocate exactly the same
integer budget to each group for every method. Within each quota, accept lowest
estimated risk. This is a simple constrained baseline and provides no error
guarantee. Prespecified coverage grid: 100/90/80/60/40/20%.

**Primary contrast:** metadata-plus-distance-plus-model-signals risk versus the
same metadata-plus-distance risk at
80% retention under identical country-by-area quotas. Practical target: at least
0.005 absolute retained-survey F1 gain, with supported-country F1 regressions no
worse than 0.005. Report gain and all country checks separately; passing the gain
margin alone is not success. Report both folds and spatial paired-bootstrap
intervals; they are descriptive under historical adaptive reuse.

The full-risk/environment-novelty combination is a secondary ablation; it cannot
establish a benefit from model uncertainty by itself. Secondary outputs include
full-coverage sample/micro/macro F1, rare-label recovery,
country retention, Brier error and a frozen expected-F1 count-surrogate comparison.
Secondary analyses are exploratory and cannot replace the primary contrast.

## Admission sequence

- Technical GPU probe: actual tensors/image band order, GPU identity/memory and
  finite backpropagation for a 5,016-label model. No accuracy claim.
- Smoke: fold 0, deterministic label-blind subsets of 2,048 training, 384
  probability-calibration, 512 risk-calibration and 768 assessment surveys; two
  seeds, two epochs. Verify the complete data/training/export/evaluation path.
  Promotion depends on numerical and integrity checks, not F1 gains.
- Full study: two folds, three fixed seeds, 18 epochs. Estimate total runtime from
  smoke measurements with conservative margins before launch. A further admission
  check uses measured first-fold wall time. Runtime failure preserves a partial
  record; it must not be reported as a completed comparison.

The fixed benchmark architecture is sufficient to test the first uncertainty
question. Any later pooling/loss improvement needs a separate matched ablation.

## Independent-data workstream

A candidate is **sPlotOpen**, which provides open vegetation-composition and
survey-design metadata. It remains a candidate, not verified independent
confirmation: source/spatial overlap, survey completeness, positional accuracy,
licenses and predictor alignment must be audited first. GeoPlant's species IDs
are deliberately anonymized, so external named taxa cannot simply be joined to
the current classifier. Do not attempt to recover protected taxon identities.
A separate dataset-specific model evaluation can test the method without such a
join. A new dataset name alone does not establish sample independence.

The official current GeoPlant metadata were subsequently checked using predictor
columns and IDs only. IID has 4,716 IDs and OOD 10,068, with zero intersection;
their union exactly matches the existing 14,784 GLC25 test IDs. None of the three
test metadata files has a species-label column. These protocol names therefore do
not supply a new labeled confirmation set. Raw metadata are ignored under
`data/external_overlap_audit/`; hashes and set checks are recorded in
`results/reliability_study_r1/external_overlap_audit.json`.

Primary sources: [sPlotOpen project](https://www.idiv.de/research/projects/splot/splotopen-splot/),
[data paper](https://doi.org/10.1111/geb.13346),
[GeoPlant dataset documentation](https://plantnet.github.io/GeoPlant/dataset/).

## Validation and live status

**Completed:** all six full fits and all 18 manifest-listed exports were retrieved
and verified. See [R1 results and paper decision](reliability_study_r1_results.md).
The primary gain margin passes, but the supported-country non-regression safeguard
fails. The complete intended reliability claim therefore did not pass. Separate
post-hoc CPU ablations are labeled exploratory; no R1 hypothesis was rewritten.

Mathematical tests verify exact quota budgets/optimal within-group allocation,
permutation invariance, known ensemble uncertainty values, training-only area-bin
edges and invariance to adversarial assessment label columns. A synthetic
integration run checks the actual neural fit, calibration, signal export and
evaluation path. Remote probe/smoke/full states and downloaded artifacts live in
`results/reliability_study_r1/`; only those records establish actual execution.
