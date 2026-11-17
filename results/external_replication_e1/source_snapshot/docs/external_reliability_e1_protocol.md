# E1: ecological service under selective species-set prediction

Specified 16 November 2026 after the reused-GeoLifeCLEF pilot, before inspecting
external species outcomes. This is a local prospective analysis lock, not a
third-party preregistration. Configuration and implementation hashes, cohort
identities and policy artifacts are sealed in two stages by the runner.

The question is whether preserving predicted rare-occurrence mass can improve
realized rare-occurrence service at a fixed referral budget, relative to strong
Soft Dice Confidence (SDC) selection, without a practically large F1 loss.
Occurrence service is the fraction of all recorded rare presences located in
accepted cells. It is not rare-species recall, threatened-species protection or
causal conservation benefit. Report rare true positives and precision/recall too.

SDC is established prior work: [Borges et al., 2026](https://link.springer.com/article/10.1007/s10994-026-07096-w).
Our positive-set surrogate has the same ratio, so it is a baseline rather than
a new confidence score. Empty-set confidence uses a declared independence
plug-in. The constrained selector is a Lagrangian heuristic under exact group
quotas; no optimization theorem, finite-sample risk guarantee or true ecological
coverage guarantee is claimed. The contribution candidate is empirical ecological
service evaluation and a useful policy tradeoff, if it replicates.

## Data and admission

Reuse frozen predictions from two released SAGE model families, two seeds each:
optimized DeepSDM s0/s1 and random forest s0/s1. No new neural training, GPU job or
competition submission. Cap downloads at 3 GB. Sources are immutable Zenodo
records [data](https://zenodo.org/records/21297133) and
[models](https://zenodo.org/records/21335338), with source-archive checksums,
range/header verification and local member SHA256. Whole multi-GB TAR checksums
are metadata, not recomputed checksums. The standalone target file is MD5 checked.

SAGE already studies sampling-aware species evaluation; see its
[primary preprint](https://arxiv.org/abs/2609.31082). Do not claim generic novelty
for sampling adjustment or plot-area sensitivity. Published model PO training
can include evaluation cells. Spatial holdout here applies to reliability
calibration versus assessment, not pretrained backbone training.

The metadata-only audit filters every contributing plot: non-European continent,
known non-EU-prefixed source, distance >=500 km from all existing GLC PA train/test
locations, known uncertainty <=1 km, and individual plot area 10–400 m². Require
consistent country/continent per 1 km cell and released test predictions.
Assign whole 5-degree blocks using covariate-only, country-balanced allocation,
then enforce 50 km buffers between the three roles. Assessment stays fixed.
Initial metadata-only allocation had 5,595 assessment cells. An additional area
provenance check, still before species outcomes or prediction-value inspection,
found differing reconstructed/released areas in 489 of those 7,991 retained cells.
The [upstream area code](https://github.com/earens/sage-sdm-benchmark/blob/main/scripts/data_prep_scripts/predictors.py)
uses cleaned plots before species filtering; linked plot metadata need not
represent the identical effort. This does not establish an upstream error.
Exclude area disagreements (0.1 m² absolute / 0.0001 relative tolerance) before
recomputing label-free roles: 530 exclusions before buffering, 708 buffer
exclusions, final 1,165 probability-calibration cells (29 blocks), 1,061
risk-calibration cells (30 blocks), and 5,251 assessment cells (73 blocks).
Area equality does not prove plot-metadata completeness. Minimum role admission
is 600/600/3000 cells. No species outcomes informed this amendment.

Targets outside native range are missing (-1), never negatives. Use the released
native-range mask before calibration, decisions and utility. Check row locations,
species-column indexing, sampled area, and target/mask alignment. Rarity is the
bottom quintile of positive finite PO relative prevalence, including ties:
1,155 of 5,771 species. Only species_id, relative_prevalence, n_species_obs and
sampling_effort are read from species properties; test-derived columns are
excluded. A secondary definition uses 1–25 probcal occurrences, without tuning.

## Fixed analysis

Fit one shared Platt slope/intercept per model on probability-calibration rows.
Bounds and 512-bin numerical approximation are in the configuration. Threshold
decoding selects from a fixed 19-value grid on these rows only. Evaluate top-18,
threshold 0–80, surrogate expected F1 over 8–40, and expected F1 over 0–80.
Empty/empty F1 is 1; report zero-divisor-zero sensitivity and empty rates.

At coverages 1, .9, .8 and .6, all policies receive identical country × plot-area
bin budgets; area cutpoints come from probcal covariates. Largest remainder fixes
the exact counts. Controls: seeded random, mean confidence, predicted-set entropy,
SDC, fixed-hyperparameter learned error regression, raw rare-probability mass
constraint, and supervised total-richness constraint. The candidate constrains
supervised predicted rare-count retention to at least the survey coverage.
Regressors fit risk-calibration rows only. The same outcome-free assessment
features and declared fixed model parameters serve all settings. If the proxy
target is infeasible under quotas, return maximum proxy-mass ordering and report
failure. Never relax geography based on assessment outcomes.

Primary: candidate vs SDC, expected-F1 0–80 decoder, 80% survey retention, averaged
over the four fixed model settings. Require >=200 riskcal rare occurrences.
Joint paired 5-degree block bootstrap: 2,000 draws, seed 20261116, same assessment
block resamples across all models. Use 97.5% intervals for each of two primary
endpoints; other analyses are descriptive. The four fits are not four independent
datasets. Success requires mean rare-coverage gain >=.03 and positive interval
lower bound; F1-gain interval lower bound >=-.005; every model's point F1 gain
>=-.005 and realized rare-occurrence coverage >=.78. This deliberately demands
both practical service and accuracy. Other decoders, countries, sources and
secondary rarity cannot rescue a failed primary claim.

First lock config, source snapshot, cohort/mask hashes and target provenance.
Only then read probability/risk calibration outcomes. Generate and hash all
assessment policies before admitting assessment target reads. Record admission
and first-access timestamps. Assessment never selects methods or hyperparameters.
Public benchmark outcomes are available to others; this is a newly locked
analysis of a published subset, not a secret or universally untouched test.
