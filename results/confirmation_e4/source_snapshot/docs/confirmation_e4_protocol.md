# E4: NEON reliability transfer in an exact-input stratum

Locally specified before inspecting NEON composition. This is not third-party
registration. R1/E1–E3 and dictionary/input feasibility findings inform the design.

The question is whether a learned quality predictor can improve error-magnitude
MSE over a training-mean control while giving worse quota-constrained F1 selection
than established SDC. Both outcomes must pass the joint rule in
`configs/confirmation_e4.json`; a failure cannot be replaced by a favorable
secondary. This tests an application-specific transport hypothesis, not a new
score, theorem or algorithm. Novelty remains to be assessed.

## Cohort and evidence roles

The fixed input-availability stratum contains 163 cells, 301 quality-screened
400 m² plots and 16 sites. Exact released predictor vectors exist at identical
float32 EPSG:6933 grid-center coordinates, covering all 52 features. The other 46
candidate cells are excluded for input availability before labels, not model
performance. This conditioning and source/site counts must accompany results.

The fixed MLP is `aggregation_s0`, already verified on CPU/T4 against published
probabilities. ResNet reproduction fails and it is excluded before NEON scores.
All assessment cells have original PO training support. Holdout concerns the
reliability/probability calibration, not unseen backbone-training locations.
Do not claim that NEON as a whole was absent from the original data sources.

Probability and risk calibration use only their original E1 calibration roles,
whose labels are historically consumed development data. E1 assessment labels
are not read for E4. Every NEON cell is an assessment cell: no fitting or tuning
on NEON outcomes. The old E1 geographic buffers remain in force, and candidate
cells are >=100 km from consumed SAGE locations. Native masks use the published
TDWG region rule with reference checks, not nearest-mask substitution.

## Taxonomic and observation semantics

Both source and target use the 2,453 SAGE classes with unambiguous dictionary
matches, intersected with native masks. The target is this modeled native subset,
not all biodiversity. Presence is the union across the nested 1/10/100 m² tables,
then across distinct plots in a cell. Nested area is 400 m² per unique plot;
repeated bouts are not added. The earliest metadata-eligible bout was fixed
before composition. Taxon IDs of unselected plots/bouts are not decoded.

Only plant-species rows contribute. A resolved, unqualified, acceptable-quality
mapped record establishes a positive class. Unmapped/qualified/uncertain records
make compatible model genera unknown; if genus is unavailable, use family, then
all classes. Positive observations remain known even within an uncertain genus.
Conservative handling can remove many labels or cells. It must not be relaxed
after outcomes to increase sample size. Report occurrence mapping, ambiguity,
known-label fractions and attrition by site. Protection-obfuscated taxa remain
unresolved; do not try to identify them.

Unknown labels are excluded only by the evaluator: frozen predicted sets and
confidence features do not see outcome-derived masks. F1's prediction denominator
counts predicted classes that remain known; known-positive truth forms its truth
denominator. Empty/empty F1=1, with zero-empty sensitivity. This is partially
observed-set F1; do not call it full-community occupancy accuracy. A cell needs
>=80% known prior-eligible classes. At least 100 cells and 12 sites must remain
for the prespecified confirmation to be admitted. Report eligibility-conditioned
results and attrition, not a claim about all candidate sites or discarded cells.

## Fixed fitting, comparisons and inference

Fit the inherited pooled Platt transform on E1 probability calibration only.
Decode with the fixed expected-F1 0–80 surrogate. Fit HGB quality, positive-slope
affine SDC quality and training-mean quality on E1 risk calibration only. Use
the inherited fixed HGB hyperparameters; no new sweep. Quality predictions are
clipped to [0,1]. Raw SDC, mean selected probability, selected-set entropy and
deterministic random ordering are controls. The rare definition is inherited
from E1 training-prevalence metadata and is secondary.

At 80% coverage, every method receives the same site × single/multiple-plot
budgets by largest remainder. Predeclared scorable-cell eligibility is applied
before budget allocation; no rule is chosen by F1. Compute paired site-cluster
bootstrap intervals, 4,000 draws and 97.5% confidence for the two primary
contrasts. The joint success rule requires positive MSE skill by >=.001 and
negative HGB-minus-SDC F1 by <=-.002 with both intervals excluding zero in the
specified directions, plus sample-size admission. These are fixed-model,
eligibility-conditioned site intervals, not independent model replications.

Freeze source/config/input identities before fitting; freeze transforms,
predictions and heads before requesting NEON composition. Keep a first-access
receipt. One authorized <=20-minute private GPU inference job uses public model
weights and embedded covariates; no token or label enters it. No new SDM fit or
competition submission. Retain previous failures and all earlier manifests.
