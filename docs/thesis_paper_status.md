# Thesis progress and paper path — 21 November 2026

This note separates completed evidence from a proposed paper contribution and
future validation. More runs, software tests or new split names cannot by
themselves establish novelty or independent confirmation.

## What has been built

The project contains a working multimodal GeoLifeCLEF pipeline, archived scored
models, presence-only training components, species-set decoding/calibration and
spatially structured evaluation. The best recorded competition model remains
v32: **.24225 public / .21514 private**, against the recorded winning private
score **.23021**. It is a substantial engineering foundation, but no benchmark
victory or proof that improvement is impossible has been established.

The research extends this into fixed-policy reliability and ecological-service
evaluation. R1 runs six neural fits; decision-alignment studies and E1/E2/E3 add
strong controls, source/spatial comparisons and ecological-service tests.
Several intended claims fail: regional safeguards, the rare-service target and
universal learned-versus-scalar confidence superiority are unsupported. These
failures identify limits of the initial proposals and remain useful evidence.

The released SAGE MLP has a verified CPU/T4 inference bridge to NEON inputs.
Published-output reproduction excludes a failing ResNet path. Field evaluation
does not pass its admission rule: the original 65-cell/10-site result, separately
corrected 77/11 morphospecies replay, and final A3u 33/6 contract all remain below
100 cells/12 sites. Previously omitted join handling and later uncertainty
corrections are disclosed; E4's original failed primary is preserved.

## What we have learned about measurement

T1 runs 60 controlled taxonomic-censoring settings with fixed predictions.
Approximately 98.3–98.4% known class labels coexist with about 26% hidden positive
occurrences at the moderate setting. Its declared spatial criterion fails and
no policy point ordering reverses. Separate diagnostics identify 268 nonempty
recorded inventories receiving false perfect scores after censoring under the
empty/empty-one convention. Simulator-supported genus constraints tighten ranges;
their field applicability is not established.

A1/A2/A3/A3u audit taxonomy, current records, history and explicit aliases. On
the same 77 cells, stricter uncertainty masks raise observed F1 by **.026839**
without changing predictions or confirmed positives. Explicit synonym links
recover **123 supported modeled cell–species presences**. These are label/score
contract effects, not predictive accuracy gains or biological species discovery.

T2 adds **100 CPU query trajectories / 700 checkpoints**. Its 25% practical
criterion fails because local support controls already saturate. Predeclared
secondary costs to identify a practical finite-cohort ordering use **27.7% fewer
median shared queries** and **39.5% fewer local queries** than the best tested
controls. These are conditional synthetic query costs, not measured taxonomist
time. Focused comparison queries can cover fewer countries and recover fewer
rare labels than broader inventory-oriented queries.

## Latest linkage/cost study

T3 completes **360 CPU trajectories / 2,520 checkpoints** with 417 passing tests.
All wrong-link scenarios preserve each cell's true hidden inventory. At 1%/5%
nominal membership error, every unchecked trajectory has a reported range that
excludes the true contrast at some checkpoint; no wrong point ordering is
observed. Ideal audited procedures preserve validity, at an explicit cost.

The declared reference criterion fails: paired acquisition improves mean cost
only 9.34% / 7.71% against the strongest matched shared-query control, below the
10% target. At the assumed audit price .1, verified comparison-focused sharing
is 66.3% / 56.2% cheaper than the best local reference; this advantage disappears
at higher prices. Checking only comparison members and choosing a representative
among them changes the cost frontier, while leaving other inventories unresolved.
These are conditional simulator findings, not measured field error rates or time.
See [T3 results and figure](linkage_robustness_t3_results.md). T2/T3 failed criteria
are preserved; no new general-method success is claimed.

## What a paper could contribute

The strongest current direction is **observation contracts and verification costs
in ecological model evaluation**: quantify how apparent quality, policy ordering
and verification effort depend on what labels are known and what a query truly
reveals. The paper should connect the measurement diagnosis to a valid, fairly
costed procedure and its limitations.

Active testing, paired model selection, linkage-error propagation and performance
partial identification already have prior art. A general new algorithm or a new
theory is not established by the current static heuristic and elementary bounds.
The empirical combination still requires careful novelty review and replication.
See the [claim matrix](paper_claim_evidence_matrix.md) and
[T2 prior-work review](taxonomic_verification_t2_prior_work.md).

I consider a paper a plausible outcome of this thesis, but a strong submission
is not yet earned by the experiment count. Its credibility must come from a
precise contribution, fair comparisons, realistic assumptions, robustness and
adequate validation. No acceptance or venue-level guarantee follows here.

## Remaining evidence and stopping points

The first package in the proposed four-package roadmap, linkage/cost robustness,
is now completed. Three proposed packages remain:

1. **Stronger acquisition comparison:** matched adaptive/active-evaluation
   baselines and targeted ablations. Keep query information and charged costs
   equal. Reassess whether a method contribution survives before more expansion.
2. **Model/source replication:** at least two additional verified prediction
   settings and another survey source, reusing valid saved outputs. Historically
   inspected outcomes remain development, even with another model.
3. **Fresh external confirmation:** freeze procedure/endpoints before opening
   separately uninspected outcomes; audit training overlap, record completeness,
   linkage and adequate clustered sample size. A real record/cost audit would
   help ground the synthetic assumptions. Its availability is unresolved.

These are evidence packages, not four individual training runs. Exact future
replay counts and compute ceilings require semantic/feasibility checks before
protocol freezes. No new backbone-training sweep is currently justified; any
needed inference follows reference checks and existing bounded GPU authorization.

If realistic costs or stronger controls remove the advantage, do not add seeds
or change the primary budget to recover success. Reassess a narrower empirical
measurement-audit paper with the supervisor. Replication and suitable validation
remain important for that scope too.

Write the manuscript alongside the remaining work. Use one coherent research
question; retain competition engineering and less central experiments as
supporting thesis material or supplementary results. All consumed E1/E3/E4/T1/T2
outcomes stay consumed, and new local freezes do not make them fresh.
