# Working paper outline — 17 November 2026

**Latest T1 evidence:** [60 controlled taxonomic-mask settings](taxonomic_resolution_t1_results.md)
are complete, with zero new neural/GPU fits. They quantify how class-known
admission can miss positive-record loss, but the declared spatial-bias criterion
fails and no tested policy ordering reverses. Post-hoc empty-set and positive-
genus diagnostics sharpen the observation-contract question; they do not supply
a new successful method or fresh confirmation. The [primary-source review](taxonomic_evaluation_prior_work.md)
rules out broad novelty claims for active testing or partial identification.
Next is a field taxonomy/observation-contract audit, not a generic model sweep.

**Earlier E4 evidence:** [the NEON study](confirmation_e4_results.md) is complete
but fails admission and joint success. Its first conservative pass omits a
documented morphospecies join; the separately labeled correction improves label
coverage but remains below the required cells/sites. Do not sell attrition as
intrinsic dataset failure. A controlled taxonomic-resolution/paired-policy audit
is the next proposed study, with zero new backbone fits and novelty unestablished.

**Latest implementation/provenance update:** [NEON bridge checks](neon_bridge_status.md)
verify the MLP on CPU/T4 and leave the ResNet excluded after failed reproduction.
Candidate plots overlap original PO training support, so a fresh source test
would concern reliability-calibration transport, not unseen backbone-training
locations. No NEON ecological result or new scientific contribution is established.

**Latest evidence, E3:** [the locked larger-plot confirmation](confirmation_e3_results.md)
fails joint superiority and reverses E2's magnitude-MSE point ordering. A post-hoc
training-mean reference is better than scalar SDC on E2. Revise the manuscript
around distinct evaluation objectives and domain-dependent transport; do not
promote scalar confidence calibration as a generally successful method. The
methods/results draft and matrix include this falsification. NEON metadata and
input contracts are audited; prediction compatibility remains pending.

**E2 follow-up completed:** [ranking/source results](ranking_study_e2_results.md)
reject the tested ranking family in nested calibration utility validation.
Monotone-affine SDC provides a stronger magnitude-calibration baseline: source-
study MSE .017511 versus HGB .024551, while preserving SDC's better decisions.
This narrows the earlier raw-SDC MSE interpretation. These are development
findings; no successful new algorithm or fresh E2 assessment is claimed.
[Claim/evidence matrix](paper_claim_evidence_matrix.md) and
[methods/results draft](paper_methods_results_draft.md) are prepared for review;
the complete manuscript and new primary confirmation remain pending.
E2's scientific promotion criteria fail, so its proposed GPU replay is not run.

Planning artifact, not a finished manuscript or a publication claim.
Current measured outcomes are in [R1 results](reliability_study_r1_results.md) and
the D1–D3 follow-up, plus [completed locked E1](external_reliability_e1_results.md).
Historical benchmark gains remain supporting context. E1 tests new reliability
roles on a public benchmark; pretrained PO models can include evaluation cells.

**New evidence:** the proposed rare-count constraint failed its practical
external improvement target. Do not present it as a successful new algorithm.
The tested learned selector loses F1 to SDC across four frozen model settings;
predicted-set entropy loses to random across settings/decoders. An explanatory
spatial-CV diagnostic finds weak ranking despite strong in-sample error fit, and
assessment error MSE improvements do not yield better quota-constrained F1.
The defensible candidate is a specific empirical evaluation contribution, with
negative results included. Novelty must be assessed against SDC and SAGE.

**Candidate title:** *Decision-Aligned Reliability for Plant Community Prediction
under Geographic and Survey Shift*.

**Central question:** does a model identify reliable species-set predictions
beyond favorable survey conditions, while preserving service to geographic groups
and rare taxa?

## Intended claims and present status

| Candidate claim | Current evidence | What must be earned |
|---|---|---|
| Model confidence contains information beyond metadata/distance at equal geographic/area coverage | R1: +0.014741 F1, both development folds positive | Comparison to stronger confidence controls; multiple models and independent data |
| Unconstrained selective gains can substantially reflect cohort composition | R1 exact descriptive decomposition; earlier excluded-country example | Replication under other geographic units/survey strata; no causal interpretation of a reweighting identity |
| Generic uncertainty scores can be misaligned with the species-set decision | D1 completed: entropy loses to random under three decoders; lean-versus-simple confidence gain shrinks from 0.004585 to 0.000885 with decision alignment, interval spans zero | Cross-model/domain replication and controls for probability calibration; numerical sensitivity is disclosed; no universal entropy-failure claim |
| A proposed regional/ecological method improves reliability without losing service | Not established; R1 safeguard fails, D3 negative, E1 rare constraint fails primary improvement | Abandon success claims for these prototypes; a new objective requires calibration-only practical gain and new confirmation |
| Reliability selection benefits conservation or rare taxa | E1 service metric is explicit; mean gain only +0.536 points, 97.5% interval [-0.357,+1.388], versus a +3-point target | Report failure; occurrence service and rare predictive recall are different; neither establishes conservation causality |

The paper may become a replicated empirical study rather than a new-method paper.
Do not invent an algorithm contribution to fit a preferred narrative. SAGE and
existing selective prediction/F-measure literature must be treated as substantive
prior work. The contribution needs a specific result that survives strong controls.

## Structure to develop

1. **Introduction:** why site-level community decisions and abstention matter;
   explain geographic/survey imbalance and the distinction between predictive
   confidence, species-set utility and service coverage. State the narrow question
   and bounded claims. No promise of beating the historical competition winner.
2. **Related work:** multimodal SDMs/GeoPlant; geographic applicability and spatial
   validation; species-level sampling-aware SAGE evaluation; F-measure decisions;
   selective/fair prediction and risk-control assumptions. Explain the distinction
   between site-level selection and species-level performance stratification.
3. **Data and provenance:** GeoLifeCLEF inputs, observed surveys, anonymity,
   training/calibration/assessment roles, prior label consumption, and external
   source/cohort overlap audit. Label development evidence and independent
   confirmation separately throughout the paper.
4. **Evaluation design:** freeze predictors and decision rules; equal geographic
   and plot-area budgets; metadata/distance and strong confidence baselines;
   decomposition of composition/within-stratum gain; regional and rare-occurrence
   service; explicit hypothetical referral-cost assumptions.
5. **Results:** controlled model-signal contrast, stronger-baseline comparisons,
   decoder diagnosis, individual versus ensemble utility, negative regional
   extensions, then replication. Keep exploratory findings separate from a later
   locked confirmation. Include all prespecified primary outcomes, including failures.
6. **Discussion:** practical implications and limits; conditional intervals versus
   adaptive reuse; sparse-country uncertainty; observed prevalence versus threatened
   status; lack of taxon identities in GeoPlant; spatial/temporal support; no
   guarantee that abstention improves conservation.
7. **Reproducibility:** source/notebook digests, fixed roles, manifests, compute
   history and actual versus planned execution. Keep credentials and protected
   identities out of artifacts. Preserve failed candidates and selection counts.

## Proposed main figures and tables

- Data/domain/survey coverage and a clear role/provenance diagram.
- Full-coverage F1 versus count MAE for the same predictor under different decoders.
- Risk/coverage under unconstrained and matched geographic/survey retention, with
  composition decomposition and strong confidence baselines.
- Regional utility and rare-occurrence coverage alongside retained-survey F1.
- Main table: all fixed methods across models/domains, with independent results
  visibly separated from reused development results.
- Appendix: competitive version history, calibration diagnostics, small-group
  counts, computational resources and negative ablations.

Avoid an abstract with unsupported numerical promises. Draft its final result
sentences only after the locked replication determines the defensible scope.
