# Proposed paper protocol: reliability without geographic exclusion

**Latest 16 November update, E3:** the locally locked confirmation on uninspected
larger-plot/geographic cells is complete and does not confirm joint scalar-SDC
superiority over HGB. See [results](confirmation_e3_results.md). E2 magnitude
interpretation is also weakened by a post-primary training-mean-only reference.
Preserve the failed hypotheses: no successful algorithm or global comparator
winner is established. A NEON field-source metadata/model-contract audit is
complete; the compatible data/model path and neutral GPU inference remain pending.

**E2 update, later 16 November:** the bounded ranking and source-exclusion
development studies are completed. Inner validation selects unchanged SDC in
all eight outer procedures. A monotone-affine confidence calibration, preserving
SDC ordering, outperforms HGB error MSE in all four source-study model settings.
Use this stronger control when interpreting E1's raw-SDC MSE diagnostic; see
[E2 results](ranking_study_e2_results.md). A concrete claim/evidence matrix and
methods/results draft are available. The user authorizes GPU tests for candidates
that pass the recorded implementation/scientific criteria, without asking again;
the E2 candidate does not pass scientific promotion and no GPU job is launched.

**Current update, 16 November:** [E1 external results](external_reliability_e1_results.md)
are complete under their separately frozen protocol. The ecological constraint
failed its three-percentage-point rare-service target; 97.5% interval
[-0.357,+1.388] points. Secondary comparisons favor strong SDC over the tested
learned risk selector, and random selection over the tested predicted-set entropy.
A spatial-CV diagnostic already finds weak learned ranking inside calibration.
The published [Soft Dice Confidence paper](https://link.springer.com/article/10.1007/s10994-026-07096-w)
defines our existing positive-set confidence surrogate: do not claim it as new.
This updates the evidence; it does not retroactively revise the dated plans below.
Future method claims require a specific practical improvement and new locked
confirmation. No additional neural-training sweep is currently admitted.

Drafted 13 November 2026 **after** the exploratory audit and pilots. This is a
prospective protocol for new work, not preregistration of already observed results.
See `research_audit_2026_11_13.md` for implementation, measurements and limitations.

**R1 update, 13 November:** the subsequent controlled six-fit GPU study is complete;
see [results and paper decision](reliability_study_r1_results.md). Model signals
improve F1 over metadata/distance under matched coverage, but the frozen regional
safeguard fails. Gain over simple mean confidence is +0.004455, and a post-hoc lean
decision-confidence regressor matches the full uncertainty bundle. These findings
narrow future work toward decision alignment, regional risk and rare-species
service; they do not validate every proposed contribution below. This section
records observed results without retroactively changing the R1 protocol.

**14 November follow-up:** D1–D3 are now authorized under a separate recorded
[development protocol](reliability_followup_d1_d3.md). Initial D3 regional
weighting/blending fails to improve pooled lean confidence and regional utility.
The [external-data/related-work audit](independent_reliability_data_audit.md)
identified SAGE (September 2026), which already addresses species-level
sampling-aware evaluation and plot-area sensitivity. A generic version of that
framing is therefore insufficient novelty. Focus the prospective claim on
site-level species-set decisions and selective geographic/ecological service;
independent confirmation remains pending.

**Completed follow-up findings:** [D1–D3 results](reliability_followup_d1_d3_results.md)
now show that lean error prediction adds only 0.000885 F1 over simple confidence
with expected-F1 decoding (interval spans zero). Original immutable R1 outputs
show a 0.006932 averaging benefit over mean individual-model F1. Regional
extensions remain negative. Regenerated outputs carry numerical tie sensitivity;
the failed literal-set guard is preserved. This favors locked empirical
replication and external artifact reuse before more neural training.

## Research objective

Working title: **Reliable Plant Community Prediction under Geographic and Survey
Shift: Accuracy, Species-Set Decisions and Coverage**.

Investigate whether uncertainty and set-decoding methods improve biodiversity
prediction under geographic transfer after controlling for survey design and
which places receive predictions. A competitive benchmark result supports the
study; exceeding the historical winner is an optional secondary objective.

The first pilot suggests that a selector can improve retained-survey F1 by choosing
favorable plots and excluding a poorly represented country. The paper must test
how widely this occurs, separate mechanisms, and offer a useful improvement.
A single dataset, two reused folds and one retrospective example are insufficient
for a strong general methodological or ecological conclusion.

## Contributions to earn

| Contribution | Scientific question | Evidence needed |
|---|---|---|
| Survey-design-aware reliability evaluation | Does confidence identify uncertain ecology beyond plot area, survey effort proxies and species-set size? | Area/year/uncertainty-only baselines; matched survey-design strata; multiple model families and spatial domains; retained species/region coverage |
| Selection preserving coverage | Can a selector lower prediction risk while maintaining service to remote regions and different survey types? | Same backbone/decoder; equal overall coverage; geographic and plot-area coverage constraints; comparisons to simple quotas, distance, confidence and existing selective methods |
| Community-decision diagnosis | How much transfer failure concerns ranking, count decisions and calibration? | Frozen-logit decoder comparisons; proper probability diagnostics; count MAE/bias alongside F1 and rare-species recovery; support-aware oracle diagnostics clearly separated from deployable methods |

Choose the first two as the main story. Keep the decoder study as a mechanism
analysis unless it yields an independently validated methodological improvement.
Avoid accumulating unrelated model additions into a paper with no central claim.

## Hypotheses and falsification

**H1: Pooled selective gains partly reflect survey design and geography.** Compare
confidence-based selection against plot-area-only and metadata-only selection.
Then compare inside fixed plot-area, region and richness-reporting strata.
True richness is an analysis stratum, never an input to a deployment selector.
Report the remaining gain after matching distributions and retaining poorly
represented countries. Reject a broad claim if the pattern fails across models
or if the controlled comparison removes it inconsistently.

**H2: Coverage constraints improve the reliability/coverage tradeoff in underserved
domains.** Freeze a selector estimating per-survey error from ensemble disagreement,
calibrated score summaries, environmental novelty and training support. Compare
against its feature ablations and simple within-country rank normalization.
At a fixed overall acceptance budget, impose minimum coverage for geographic
groups and plot-area groups. Evaluate risk at the same coverage constraints;
comparing an unconstrained curve to a constrained method alone would be unfair.

A possible implementation minimizes summed estimated loss of accepted surveys
subject to an overall budget and minimum group retention. This is a standard
constrained selection formulation; novelty must come from the ecological setup,
well-supported mechanism or an actual improvement over strong existing methods.
Country groups are an audit starting point, not biological regions. Test spatial
cells, provided biogeographic regions and survey-design bins as well.

**H3: Richness accuracy and F1-optimal set selection diverge under transfer.** On
exactly the same frozen scores compare fixed-k, global threshold, shared Platt
calibration, the current expected-F1 surrogate and a learned cardinality method.
Include a structured label/cardinality baseline if proposing a new decoder.
Reject the interpretation if differences disappear once ranking and calibration
are controlled. Do not call the ratio-of-expectations surrogate Bayes optimal.

Suggested practical decision margins, to freeze before new assessment: at least
0.005 absolute selective-F1 improvement at 80% coverage over the strongest
constraint-matched simple baseline; no more than 0.005 F1 regression in supported
geographic groups; and group coverage within five percentage points of the target
where finite sample size permits. These are **utility thresholds**, not assumed
effects, power calculations or statistical guarantees. Revisit feasibility using
training/calibration evidence only and document any revision before assessment.

## Data and evaluation

1. Label all existing PA results as exploratory/reused. A newly named split of
   those same inspected labels is not an untouched test. Reconstruct old split
   hashes and compare source-file/label digests before using cached comparators.
2. For systematic development, freeze several outer geographic or provided
   biogeographic domains without using their outcomes for assignment. Separate
   model fitting, checkpoint selection, probability calibration, selector fitting
   and final assessment. Use buffered metric-distance blocks and report results
   under more than one buffer/block scale. All new fits must exclude outer labels
   and nearby PA/PO records under a declared rule.
3. For independent confirmation, secure a legitimately new, permitted PA survey
   source or a future benchmark with nonoverlapping samples; let the supervisor
   hold the labels until methods and analysis are locked. Audit spatial, temporal
   and source overlap, including PO records. Different year/competition names
   alone do not prove independence. If this cannot be obtained, retain the more
   limited exploratory framing throughout the paper.
4. Report absence of support rather than extrapolating an accuracy claim for
   Ukraine or another unlabeled region. Country prevalence reweighting cannot
   estimate conditional accuracy where the relevant observations are missing.
   Separate geographic novelty from environmental novelty and label support.
5. Once a final candidate is frozen, allow one documented late Kaggle submission
   for external corroboration if available. The 2025 private scores have already
   guided development, so a new late submission is not a pristine blind test.
   Preserve all failed candidates and the exact submission count.

## Baselines and controlled comparisons

Freeze one compact Landsat baseline, one multimodal attention baseline and one
competitive v32/v35-family model. Use the same training data, seeds and evaluation
roles for each meaningful comparison. Refit matched controls under the new
protocol; cached old scalar F1 does not supply unavailable confidence or per-label
diagnostics. Keep an exact scored v32 CSV as the external historical reference.

The reliability baseline set includes seeded random acceptance, nearest PA/PO
distance, plot-area-only, all allowed metadata, max/mean predicted probability,
entropy, ensemble disagreement, environmental novelty, and a calibrated error
regressor. Include simple group quotas and relevant fair/selective methods.
Count-based and distance-based selection must not use the assessment's true
richness or errors. Calibration selectors may use their assigned labels only.

When testing sensor-balanced pooling, match loss, width, parameter/compute budget,
training duration, seed and decoder. Run pooling-only, ranking-loss-only and their
combination before attributing a benefit to either. Do the same for any rare head.
The existing v35 three-member ensemble does not isolate those factors.

For PO supervision, distinguish sampling intensity from occupancy. Unrecorded PO
species are not verified negatives. A coordinate/distance expert, simple positive
aggregation and the existing masked PO loss are required controls. PO-first
pretraining and proximity routing already have task precedents. Verify species
vocabulary and the official evaluated label space before asserting that expanding
from PA species solves the gap.

## Metrics and inference

- Primary: sample-averaged F1 at full coverage and prespecified retained coverage;
  selective loss `1-F1`; coverage of every supported geographic/survey group.
- Secondary: macro/micro F1, per-species precision/recall in training-defined
  prevalence bins, species-set count MAE/bias, and recovery of rare taxa.
- Reliability: AURC plus fixed-coverage risk, calibration slope/intercept, proper
  scoring rules and reliability diagrams. Report positives/rare labels separately
  because average Brier error across thousands of absent labels can be misleading.
- Geographic service: coverage and conditional risk by region, distance, novelty
  and plot-area bin; report zero-retention groups explicitly rather than dropping
  them from macro averages. Record loss/cost for abstention or expert survey referral.
- Resources: measured preparation/training/inference time, model parameters,
  memory, hardware, and total selection/training budget. Per-run T4 time does not
  represent the cost of the historical search.

Use at least three neural seeds on the final chosen contrast, paired comparisons
and spatially clustered intervals with sensitivity to geographic units. Report
small-group counts and uncertainty without overinterpreting Bulgaria's 31 or
Switzerland's 49 development rows. Separate randomness from training seeds,
sampling variation and uncertainty from distribution shift. Correct the family of
prespecified primary comparisons and disclose the exploratory ones. Do not rely
on an unadjusted lower bootstrap endpoint after a policy search.

Ordinary split conformal guarantees do not automatically survive dependent spatial
data and arbitrary support shift. If a conformal method is included, state its
assumptions and show empirical risk/coverage failures where they fail. Avoid
guarantee language for ordinary calibration, weighted resampling or country quotas.

## Bounded next experiments

| Stage | Work | Compute/admission rule | Outcome |
|---|---|---|---|
| Completed exploratory audit | Metadata coverage; frozen-ranking count pilot; coverage/risk pilot | Local CPU; four count fits and six risk fits | Recorded, including negative results |
| Completed CPU reproduction | New private Kaggle notebook, official input, cached comparator integrity | CPU only; six risk fits; no submission | Remote job completed; all 126 summary rows exactly reproduced |
| Completed R1 fixed-backbone diagnostics/comparison | Full mass, member disagreements, labels, roles, checkpoints; 17 selectors and matched quotas | Six fixed neural fits, 69.45 min full notebook; no submission | +0.014741 over metadata/distance; regional safeguard failed; reused-data evidence |
| Completed D1–D3 diagnosis | Confidence/decoder controls, original versus regenerated averaging, regional and rare-service checks | Two inference-only sessions, zero neural fits; numerical qualification retained | Decision alignment shrinks selector advantage; averaging helps; tested regional extensions fail |
| Deferred architecture/loss mechanism experiment | Pooling and ranking-loss ablation only if needed for a specific claimed mechanism | Smoke/admission after a replication question is locked | No open-ended architecture sweep |
| Planned locked reliability replication | Decision-aligned confidence versus lean/established baselines, explicit regional and rare service | Prespecified domains/coverage; distinguish members from independent ensemble trials; reuse external baselines where feasible | Test the empirical finding; new regional-method development is conditional |
| Independent confirmation | New nonoverlapping PA data, supervisor-held labels, optional frozen Kaggle check | After model/analysis lock | Determine the defensible paper scope |

Keep v35 as a bounded accuracy experiment after its measured smoke/admission
checks. There is no basis for promising a 7% gain or allocating an open-ended GPU
sweep. If it improves F1, it still needs attribution and reliability evaluation.
If it fails, the reliability/survey-design study can still produce useful results.

## Claims currently prohibited

No SOTA claim; no claim that the benchmark is impossible to beat; no general
spatial risk guarantee; no independent significance from consumed labels; no
causal ecological mechanism inferred from model features or co-occurrence alone;
no assertion that withholding predictions automatically helps conservation.
Anonymized labels and heterogeneous observed surveys also limit biological
interpretation. Strong ecological claims require identified taxa, relevant
environmental evidence and appropriate study design, beyond a score improvement.

The next paper draft should be written around validated findings once the main
contrast is resolved. This protocol deliberately does not invent a results section
or abstract claiming discoveries that the experiments have not established.

## Primary literature anchors

- [GeoLifeCLEF 2025 overview](https://ceur-ws.org/Vol-4038/paper_234.pdf): task,
  sample-F1 metric, geographic shift and winning systems.
- [Tighnari v2](https://ceur-ws.org/Vol-4038/paper_246.pdf): weak labels,
  multimodal cross-attention and geographic expert routing.
- [PredComX](https://ceur-ws.org/Vol-4038/paper_261.pdf): joint/deep SDMs and HMSC.
- [Dembczynski et al.](https://proceedings.mlr.press/v28/dembczynski13.html):
  structured F-measure decisions, beyond marginal thresholding.
- [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i): selective prediction
  can amplify group disparities; this general phenomenon is not a new claim.
- [Bu et al.](https://proceedings.mlr.press/v139/lee21b/lee21b.pdf): existing fair
  selective classification work to compare against.
- [Poisot](https://www.sciencedirect.com/science/article/pii/S0065250426000073):
  conformal uncertainty in SDMs is already studied.
- [SAGE, Arens et al., September 2026 preprint](https://arxiv.org/abs/2609.31082):
  global sampling-aware species-level evaluation, with plot-size sensitivity;
  public data/model releases are a candidate replication source, subject to
  overlap/protocol audits. Site-level selective reliability needs separate study.

This is a targeted primary-source review, not an exhaustive novelty clearance.
