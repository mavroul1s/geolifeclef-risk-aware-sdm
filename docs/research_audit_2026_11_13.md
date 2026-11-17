# Thesis-to-paper evidence audit

13 November 2026. Base revision: `a1c84d9a4dd2ec1abd5fe76ff06774c406aee277`,
verified against GitHub's default branch. The existing local clone was reused.
This document distinguishes archived results, new exploratory measurements and
proposed research. No new competitive accuracy or ecological discovery is claimed.

## Assessment

A paper is a credible objective. Beating the 2025 winner remains possible, but
the evidence does not justify making it the paper's acceptance criterion. The
strongest route is a focused study of **reliability, survey design and geographic
coverage in multimodal community prediction**, with competitive F1 as a secondary
outcome. More seeds and small tail edits have diminishing returns. Neither the
current score trajectory nor a failed architecture proves a performance ceiling.

V32's recorded private F1 is **0.21514**, versus the recorded winner's **0.23021**:
an absolute gap of **0.01507**, requiring **7.0047% relative improvement**. The
improvement from v18's 0.18900 to v32 is **13.8307%**. The task overview independently
confirms the sample-averaged F1 metric and the rounded winning score 0.2302.
Public scores, random-split micro-F1 and country-holdout F1 cannot substitute for
that private-test comparison. The five-decimal historical scores remain supported
by the repository registry/screenshots rather than a new score-API attestation.
[Official task overview](https://ceur-ws.org/Vol-4038/paper_234.pdf).

## How the implementation works

The reusable `src/geolifeclef/` package contains the original temporal CNN,
gated fusion, multimodal residual architecture, PO expert, training loop and
metrics. The recent competition systems primarily live in versioned
`scripts/v*_notebook_core.py` files. Builders embed frozen predecessor source,
compact scored predictions and integrity metadata into offline notebooks.

The current dependency chain is v35 -> v32 -> v31 -> v30 -> v29 -> v27. These
imports preserve old recipes; they do not imply every predecessor model is refit
in v35. Notebook definitions are executed separately from the final training cell.

1. Official data readers reconcile PA survey IDs, multi-hot species labels,
   Sentinel imagery, Landsat `[6,4,21]` and bioclimatic `[4,19,12]` cubes, and
   environmental tables. Recent candidates use 64-pixel imagery with derived
   spectral channels. Cached arrays and streaming statistics limit memory use.
2. The v30-and-later development design assigns indivisible one-degree spatial
   blocks to training, checkpoint selection, policy calibration and assessment.
   A 20 km buffer separates training from the other roles. The two outer
   assessment partitions total 31,570 surveys. This audit reconstructed all role
   hashes from the official metadata and matched the archived v34 manifest.
3. SensorAttention produces 64 image tokens, 21 Landsat-year tokens, 19 climate-year
   tokens, environmental/static tokens and CLS. A three-layer transformer feeds
   label and richness heads. EMA, mixup, whole-sensor dropout and partly
   country-balanced sampling support training.
4. Later recipes add asymmetric negative loss and a training-defined rare-species
   residual head. These scores are not automatically calibrated probabilities.
   Shared Platt calibration and subsequent set decoding have separate roles.
5. V31-v34 preserve inherited per-survey counts and leading species while testing
   bounded tail changes. V32 combines seed bags and asymmetric specialists. V33
   rejects its additions and returns the v32 control; v34 adds co-occurrence
   reranking and regresses slightly on the private test.
6. V35 changes sensor pooling and complete set prediction, with checkpoint
   selection up to 48 epochs. Its three models change several ingredients together:
   pooling, geography, loss, rare head and training duration. It is a joint
   hypothesis, not a causal ablation. Its decoder still restricts counts to 8..40.
7. Full-data production joins predictions to the template by survey ID and checks
   cardinalities, vocabulary, output sizes and hashes. Compact archives omit most
   checkpoints/full probabilities, limiting later analysis.

All **88,987 PA surveys have influenced earlier assessment** by v29. A role-separated
refit is useful but does not erase that adaptive history. The repeated-bootstrap
intervals below describe saved development results; they do not establish new
independent significance or quantify uncertainty in the hidden-test score.

## External progress and development transfer

| Version | Private F1 | Interpretation |
|---|---:|---|
| v18 | 0.18900 | Initial full-PA ensemble |
| v24 | 0.20094 | Multimodal rare-aware model |
| v27 | 0.20831 | Multiscale ensemble |
| v29 | 0.21093 | Full-data heterogeneous ensemble |
| v30 | 0.20942 | Regression despite pooled development gain |
| v31 | 0.21301 | Bounded residual edits |
| v32 | **0.21514** | Best recorded result; 5.7681 T4 hours |
| v33 | 0.21514 | Identical v32 CSV, not independent replication |
| v34 | 0.21483 | Small private regression |
| v35 | unavailable | Locally checked; no completed GPU result in the archive |

The new audit recomputed paired changes from the original per-survey CSVs:

| Version | Pooled gain | DK/NL gain | Outside DK/NL gain |
|---|---:|---:|---:|
| v30 vs its v29 recipe refit | +0.003523 | +0.005134 | **-0.000744** |
| v31 vs its v29 recipe refit | +0.000320 | +0.000118 | +0.000857 |
| v32 vs its v31 recipe refit | +0.001692 | +0.001427 | +0.002391 |
| v34 vs its v32 recipe refit | +0.000430 | +0.000490 | +0.000271 |

These are matched within-run comparisons; their refits differ across runs. Do not
add the gains or treat their absolute scores as identical controls. DK/NL account
for 72.59% of this assessment. Its largest one-degree block contains **18.86%** of
assessment surveys. Sensitivity to block size and country weighting belongs in the
paper; 31,570 rows are not 31,570 independent geographic replicates.

## New metadata measurements

After the user accepted the competition rules, the supplied Kaggle credential
downloaded only the two official PA metadata files into ignored `data/raw/`.
No credentials were printed, copied into the repository or uploaded.

| Country | PA training surveys | Unlabeled test surveys |
|---|---:|---:|
| Bulgaria | 81 | 3,273 |
| Ukraine | 0 | 2,911 |
| Switzerland | 155 | 1,982 |
| Denmark | 49,344 | 1,453 |
| Netherlands | 15,024 | 846 |
| United Kingdom | 0 | 843 |

DK/NL represent **72.33% of PA training but 15.55% of test**. **30.34% of test
surveys belong to countries absent from PA training**; 30.41% belong to countries
absent from the repeated assessment. Using haversine distance to all PA training
sites, **26.71% of test surveys are over 100 km away**; the median is 31.74 km and
the 90th percentile 419.26 km. These are predictor/support measurements, not test
accuracy estimates. Absent countries do not by themselves prove disjoint
environmental support, and country reweighting cannot recover unobserved outcomes.

The classifier vocabulary contains 5,016 PA species. Other task papers describe a
larger PO vocabulary. Before expanding outputs, verify the evaluated label space
and measure PO-only support on appropriate proxies; this audit does not show that
PA-unseen species explain the hidden-test gap. In the existing outer folds, only
0.44%-0.55% of positive assessment label mass lacks a training positive, suggesting
those folds provide little stress on that particular failure mode.

## New local pilot: richness accuracy versus F1

Four HistGradientBoosting regressors were fitted on the original fold-training
rows, using metadata to predict log observed richness. All neural rankings were
frozen. Evaluation uses 4,000 historically consumed v30 calibration views,
averaged by 3,504 unique survey IDs. No hyperparameter search was performed.

For the archived temporal-attention member:

| Count decision | Sample F1 | Count MAE | Mean predicted count |
|---|---:|---:|---:|
| Fixed 18 | 0.316285 | 8.05765 | 18.000 |
| Expected-F1 surrogate, 8..40 | 0.322876 | 8.71461 | 19.943 |
| Metadata richness, with coordinates | 0.313359 | **7.06150** | 14.573 |

The count regressor reduces MAE by **12.36%** versus fixed 18, while F1 decreases.
The F1 surrogate improves F1 while increasing MAE for this member. This isolates a
decision-objective tradeoff with the ranking held fixed. It does not show a new
Bayes-optimal decoder, or prove the same result on test. F1 optimization already
has an established structured-decision literature; the current ratio-of-expectations
decoder is a surrogate, not exact expected F1.
[Dembczynski et al., ICML 2013](https://proceedings.mlr.press/v28/dembczynski13.html).

Label-using prefix oracles also show headroom, but are deliberately unattainable
diagnostics. For the v34 probability-only community ranking, the 2,719 unique
calibration surveys yield fixed-18 F1 0.32513, best true-label-selected prefix in
8..40 of 0.40637, and in 1..128 of 0.42825. These are **not expected achievable
improvements**. Finite exported rankings, calibration sampling and oracle label
access limit interpretation; a large oracle gap does not imply a learnable decoder.

## New local pilot: selective reliability and geographic coverage

Six error regressors were fitted only to the cached v34 calibration views, using
the matched-v32 recipe's per-survey loss `1 - F1`. Features are plot area, survey
year, coordinate uncertainty and same-fold distance to PA training. Assessment
uses the exact saved 31,570 disjoint spatial-check rows. No assessment labels
choose a ranking or tune parameters. The labels remain adaptively reused from
earlier work, so this is exploratory evidence.

| Selector at 80% overall coverage | Retained-survey F1 | Bulgarian surveys retained / 31 |
|---|---:|---:|
| Seeded random order | 0.36143 | 24 |
| Nearest-training distance | 0.37090 | 14 |
| Plot area alone | 0.38754 | 18 |
| Metadata without distance | 0.39016 | **0** |
| Metadata plus distance | 0.38806 | **0** |
| Metadata plus distance, country-balanced ordering | 0.36932 | **25** |

At full coverage, all selectors describe the same predictor at F1 0.36095.
Unconstrained metadata selection improves the pooled retained score while
discarding every assessed Bulgarian survey. Bulgaria represents 22.14% of the
official test metadata, but has only 31 development assessment surveys here.
Plot area alone approaches the pooled gain, and adding distance does not improve
over metadata alone. These ablations challenge an interpretation based purely on
epistemic/spatial uncertainty; they do not establish a causal survey-protocol effect.

The country-balanced baseline uses within-country risk percentiles. It preserves
approximately equal retention, with rounding for small groups, at an accuracy
cost. It is a simple baseline, not a novel algorithm or an error guarantee. The
general danger of selective prediction amplifying group disparities is already
known. The potential contribution is a controlled, ecologically relevant account
of survey-design confounding, geographic coverage and multi-label community
decisions, followed by a method that improves their tradeoff.
[Jones et al., ICLR 2021](https://openreview.net/pdf?id=N0M_4BkQ05i).

These retained-survey scores are not Kaggle scores: abstention changes the evaluated
population. No confidence guarantee under arbitrary spatial shift follows from
these curves. Use observed survey richness and anonymized species IDs accurately;
do not interpret them as true occupancy, detectability or identified species ecology.

## Novelty boundaries

Multimodal fusion, asymmetric rare-label losses, PO aggregation, geographic expert
routing and co-occurrence modeling have direct task precedents. Tighnari v2 already
uses PA/PO proximity routing; PredComX integrates deep features with HMSC.
[Tighnari v2](https://ceur-ws.org/Vol-4038/paper_246.pdf),
[PredComX](https://ceur-ws.org/Vol-4038/paper_261.pdf).

"Apply conformal prediction to SDMs" is also insufficient novelty: an SDM-specific
study was published in 2026. Exchangeability/shift assumptions and observed
miscoverage must be stated rather than assuming a generic spatial guarantee.
[Poisot, 2026](https://www.sciencedirect.com/science/article/pii/S0065250426000073).

The focused claim to investigate is: **pooled reliability gains can conceal
geographic coverage loss and survey-design selection; preserving those dimensions
changes model and decoder conclusions.** This is a hypothesis supported by initial
examples, not a demonstrated universal law or a claim of being first.

## Validation, artifacts and reproducibility limits

The original full suite produced **227 passed, 22 failed, 2 skipped** in 222.12 s
under an isolated Python 3.12.16 / PyTorch 2.5.1 CPU environment. Failures concerned
missing ignored historical fixtures and archived CSV bytes normalized by Git.
After restoring the downloaded metadata and archived species vocabulary to the
expected ignored fixture locations, the focused v35 plus initial audit suite
passed **18 tests**. After adding the fourth audit integrity test, the final
focused suite passed **19 tests** in 17.56 s. The complete historical suite has
not been declared repaired or green.

The audit verified 23 manifest-listed outputs: 17 match exact SHA-256; six submission
CSVs recover the recorded digest after CRLF restoration only. The audit records
both checkout and original hashes, rejects changed values and leaves archives
unmodified. Missing v21-v23 user exports remain missing; no substitute raw results
were invented. Historical notebook regeneration still needs those fixtures.

The new CPU notebook reuses the already verified v35 definition/evidence cells,
recomputes calibration scores from official metadata and checks every role hash.
Its completed local execution matched the standalone pilot. The new
[private Kaggle CPU run](https://www.kaggle.com/code/nmavros/glc25-reliability-audit-20261113)
also **completed**, and its downloaded outputs matched all **126** local summary
rows exactly (maximum absolute F1 difference 0). It ran six risk fits in 33.26 s
inside the notebook, excluding queue/startup time, using Python 3.13.18,
scikit-learn 1.6.1 and CPU PyTorch 2.11.0. This is reproduction on the same reused
data, not independent confirmation. No GPU training or competition submission was
made. Three compact outputs, their hashes and environment details are preserved in
`results/research_audit_2026_11_13/kaggle_cpu_output/` and `kaggle_execution.json`.

Reproduction commands from the repository root:

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/glc-mpl .venv/bin/python scripts/audit_paper_readiness.py
PYTHONPATH=src:. OMP_NUM_THREADS=2 .venv/bin/python scripts/pilot_cardinality_decisions.py
PYTHONPATH=src:. OMP_NUM_THREADS=2 .venv/bin/python scripts/pilot_selective_prediction.py
MPLCONFIGDIR=/tmp/glc-mpl .venv/bin/python scripts/plot_selective_pilot.py
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_research_audit.py
.venv/bin/python scripts/build_reliability_audit_notebook.py
```

Read `results/research_audit_2026_11_13/validation.json` for library versions and
analysis-source hashes; `findings.json`, the pilot JSONs and CSVs contain the
machine-readable measurements. PNG/SVG figures accompany the results.
The next research protocol is `docs/paper_research_protocol.md`.
