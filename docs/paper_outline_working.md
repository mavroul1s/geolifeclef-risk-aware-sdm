# Working paper outline — 22 November 2026

Working title: **Observation Contracts and Verification Costs in Ecological
Model Evaluation**. Planning artifact for supervisor review, not a completed
manuscript or a claim of publication readiness.

## Research question and contribution candidate

How do known-label definitions, taxonomic uncertainty, shared-record linkage
and the cost of verification change apparent predictive quality and the
ability to compare ecological models and their fixed survey-selection policies?

The contribution candidate is a reproducible empirical evaluation with explicit
observation contracts, fair verification costs and documented failure modes.
[T4](adaptive_baselines_t4_results.md) closes the standalone static-score novelty
branch: equal-quota paired width is ordinary width restricted to the same scope
up to a constant. Generic adaptive controls reduce costs but miss the frozen
10% gate. We should not invent an algorithm contribution around a redundant
score or describe simple adaptive/cost heuristics as new methods.

The thesis contains substantial competition/reliability engineering. A focused
paper should use that foundation without trying to make every archived study a
main contribution. [Overall status](thesis_paper_status.md), [claim matrix](paper_claim_evidence_matrix.md)
and [methods/results draft](paper_methods_results_draft.md) retain the measured
scope and failed hypotheses.

## Main evidence and claim limits

| Evidence | Potential paper role | Boundary |
|---|---|---|
| T1: 60 controlled mask settings | High class-known rates can coexist with substantial positive-record loss; scoring/constraint diagnostics | Spatial primary fails; no policy point reversals; empty-set/genus diagnostics are post-hoc and synthetic |
| A1/A2/A3/A3u | Native/current-record/taxonomic contract audit; +.026839 apparent known-F1 from masks alone; 123 supported presences recovered | No prediction gain, biological discovery or rescued E4; final field admission 33 cells/six sites fails |
| T2: 100 trajectories | Shared versus local query units and comparison-versus-inventory tradeoffs | Primary fails through saturation; secondary 27.7%/39.5% median-query reductions are conditional on the tested controls |
| T3: 360 trajectories | Unchecked linkage can invalidate ranges; charged audits change cost frontier | No observed wrong ordering; perfect audits and assumed costs; 10% reference criterion fails |
| T4: 300 executed / 360 priced paths | Scope-equivalence diagnosis; six generic controls and dynamic-width/cost ablations | Static-score novelty closed; learned-cost 4.25–9.03% gains miss 10%; no consistent forecast-learning benefit or new-method claim |
| T5 model and contributor replay | Bias direction, ranking reversals and charged cost responses across fixed settings | SAGE development only, with simulated missingness/charges |
| T6 shared model-comparison bounds | Identify rankable and indeterminate cohorts under the same latent inventory | 80/240 practical signs and 59.3% narrower mean bounds. Existing shared-completion literature |
| Uninspected external record/cost audit | Ground the query contract and test a frozen narrow claim | Proposed; admission, data access and sample/cost adequacy unresolved |

Preserve the labels on every result. A local protocol freeze on consumed data
improves chronology and reproducibility; it does not create fresh confirmation.
Identification ranges are not confidence intervals. Recorded inventories are
not perfect ecological occupancy truth. Tests and GPU reproduction establish
implementation checks, not scientific novelty.

## Manuscript structure

1. **Introduction:** describe the practical model-comparison problem when surveys
   differ in label resolution and a verification can resolve overlapping records.
   State the narrow empirical question. Do not promise a new acquisition method,
   real annotation savings or a GeoLifeCLEF benchmark victory.
2. **Related work:** taxonomic/record linkage and uncertainty; missing-label and
   partial-identification evaluation; active vetting/testing and paired model
   selection; sampling-aware ecological evaluation. Explain what this audit
   measures beyond established concepts. Review novelty before choosing a venue.
3. **Data and provenance:** fixed predictors/decoders/acceptance sets; native and
   target species universes; current-record/history/alias semantics; spatial
   support and original training overlap; prior outcome consumption and study
   roles. Explain NEON's failed admission rather than hiding it.
4. **Observation and verification contracts:** distinguish known classes from
   recorded positives, valid completions from observed-label scores, and group
   membership assumptions from validated links. Define atomic query scope,
   representative, perfect-audit assumption, mismatch fallback and actual cost.
5. **Evaluation design:** frozen primary/secondary criteria; static, adaptive,
   cost and incidence controls with equal information/scope; practical contrast
   margins; independent replay checks; conditional mask variation. Include the
   equal-weight identity as a baseline audit, not a new theorem.
6. **Results:** observation-mask artifacts and supported-label recovery, then
   linkage validity/cost frontier and matched adaptive ablations. Report every
   failed primary. Add model/source replication and eligible fresh validation
   only when actually completed. Avoid combining replay counts as independent
   sample size or selecting a post-hoc strongest arm as the original proposal.
7. **Discussion:** validity versus efficiency, comparison versus broad inventory
   service, unsupported genus/completeness assumptions, real audit errors/costs,
   spatial/model uncertainty and transport. Explain what narrower conclusions
   survive if field admission or replication fails.
8. **Reproducibility:** public code/protocols/compact outputs, immutable history,
   source/input locks, documentation snapshots and actual compute. Protected
   arrays, credentials and local agent notes remain excluded.

## Main displays and supplementary material

- Observation-contract table: what is known, hidden, assumed and actually queried.
- Fixed-prediction score/eligibility changes and positive-loss versus class-known
  curves; distinguish primary tests from post-hoc diagnostics.
- Validity and audit-price/scope frontier, retaining unchecked-range failures.
- Scope-matched static/adaptive all-control cost plot and ablation table, with
  the 10% gate and equal-weight redundancy stated explicitly.
- Replication table across admitted prediction/cohort blocks, including failed
  effects and consumed-versus-uninspected roles.
- Supplement: competition history (v32 private .21514), R1/D1–D3/E1/E2/E3
  decision/service findings, negative prototypes, full taxonomy audit, numerical
  ties, source/reference checks and computational resources.

## Remaining work and stopping point

T5 completes within-SAGE replication across saved model settings and survey
contributors. T6 tests shared-inventory model comparison. Make the central
results figures show positive-occurrence loss, ranking uncertainty and
verification cost at a fixed precision target. Use the competition and
reliability history as supporting thesis evidence.

External confirmation remains open. The [AusPlots follow-up](ausplots_followup_results.md)
leaves 87 candidate sites in 45 blocks after the combined record and spatial
screen. Both model settings pass the published CPU reference, and 44 of 52
predictors reproduce the reference panel. Obtain the eight missing layers and
finish the species-level label contract before freezing an outcome test.
A paired, timed voucher audit with adjudication can estimate the cost ratio and
false-compatible rate that matter for the simulation. Freeze eligibility and
endpoints before a supervisor-held release of uninspected inventories. An
independent source that fails the record contract supports a feasibility report
and a narrower empirical case study.
