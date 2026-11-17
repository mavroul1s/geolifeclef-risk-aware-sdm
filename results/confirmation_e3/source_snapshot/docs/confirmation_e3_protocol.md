# E3: locked larger-plot domain confirmation

Specified 16 November 2026 after E1/E2 development and a metadata-only feasibility
audit, before inspecting E3 prediction values or species outcomes. This is a
local prospective lock, not third-party preregistration. The user has authorized
execution, including a bounded GPU test if correctness and scientific criteria
pass; another permission request is unnecessary.

## Question and domain

Does unchanged Soft Dice Confidence selection with scalar magnitude calibration
outperform the fixed E1 learned error regressor on both error estimation and
matched-quota F1 in an uninspected larger-plot domain?

The audit admits 654 released SAGE test cells in 19 five-degree blocks, with every
linked plot >400 and <=10,000 m². All are non-European with known country/source,
uncertainty <=1 km, linked coordinates >=500 km from existing GLC PA locations,
no previously consumed plot or cell IDs, new spatial blocks, and centroids >=100
km from every consumed SAGE cell. Released effort must match reconstructed linked
plot area. Require >=500 cells and >=15 blocks before outcome access. This is a
feasibility threshold, not a guarantee of 80% statistical power or effect transfer.

A wholly new-source subset has 364 cells but only three blocks, dominated by
Brazilian forest plots. It does not support broad source-population inference.
The full cohort is a new **plot-area/geographic domain**, not a new independent
dataset. It includes sources seen in earlier reliability roles. The exact cohort
and all exclusions are determined by covariates, not outcomes. Brazil and
non-Brazil results are secondary and cannot replace the full-cohort primary.

Published SAGE pretrained presence-only training can overlap evaluation cells.
The holdout concerns our direct assessment-outcome access and reliability data
roles, not spatially disjoint backbone training. The outcomes are publicly
available to others. No secret-test or causal ecological generalization claim.
Use immutable [data](https://zenodo.org/records/21297133) and
[prediction](https://zenodo.org/records/21335338) records already retrieved and
hashed in E1. Area consistency alone does not prove metadata completeness.

## Fixed methods

Reuse optimized DeepSDM s0/s1 and random forest s0/s1. Original E1 probability-
calibration rows fit the shared Platt transform. Original E1 risk-calibration
rows fit the exact fixed HGB loss model and monotone-affine SDC quality model.
No E2-selected penalty or new regressor tuning is used. Decoder: expected-F1
surrogate over 0–80; native mask applied before decoding. Empty/empty F1 is 1,
with the zero-divisor-zero convention reported secondarily.

Calibrated-SDC acceptance uses unchanged SDC ordering. Both its affine quality
estimate and HGB quality estimate are clipped to [0,1] for all-cell MSE, since
the target F1 lies there. Report unclipped sensitivity. Selection retains the
original ordinal scores, preventing clipping from changing the established
ranking. Controls: mean confidence, predicted-set entropy, seeded random and
raw SDC magnitude. Exact country/area quotas at 80%; bins come from original
probcal covariates. New large plots may all occupy the original top area bin;
therefore freeze an additional, secondary country × new-domain covariate area-
quintile quota sensitivity before outcomes. It cannot replace the primary.
Other coverages are descriptive only.

## Primary success and uncertainty

Co-primary endpoints, averaged equally across the four fixed model settings:

1. All-cell MSE improvement: HGB MSE minus calibrated-SDC MSE.
2. Retained-cell F1 improvement: SDC selective F1 minus HGB selective F1.

Require each point improvement >=.002 and each 97.5% paired spatial interval
lower bound >0. Also require each model's MSE improvement >=0 and F1 difference
>=-.001. Both endpoints and guards must pass. Use 5,000 shared five-degree block
resamples, seed 20261116. Equal-model averaging is not independent-model
replication. These intervals condition on the fixed pretrained predictors and
fitted calibration heads; 19 blocks and source concentration limit inference.
This is an intentionally narrow empirical confirmation, not an algorithm claim.

Record rare-occurrence service and predictive recall using E1's PO rarity, but
do not add a post-hoc conservation claim. Report country/source/continent counts,
Brazil versus other countries and empty-event sensitivity. Do not promote a
secondary subgroup or endpoint if the primary fails. Do not tune and retest E3.

## Outcome barrier and GPU stage

Hash configuration, code, cohort and input provenance before fitting. Fit/freeze
all four policies before materializing any E3 target values. Record admission
and first-target-access timestamps, enforce that requested IDs are disjoint from
all E1/E2 consumed cells, and verify native-mask/row/area alignment. Retain E1/E2
artifacts unchanged. Synthetic tests cover the target barrier and paired metrics.

If primary success and correctness checks pass, automatically run one private
Kaggle CUDA replay of the frozen scalar magnitude heads and their fixed operating
decisions, capped at 20 minutes. This tests numerical execution of the accepted
postprocessing method; it is not new backbone training or another fresh
confirmation. If primary success fails, report the failure and keep the user's
authorization for future qualifying candidates.
