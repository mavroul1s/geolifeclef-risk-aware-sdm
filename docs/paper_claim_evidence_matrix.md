# Paper claim and prior-work matrix — 22 November 2026

Working title: **Observation Contracts and Verification Costs in Ecological
Model Evaluation**. This matrix distinguishes empirical evidence from novelty. It is not an
exhaustive literature review or a claim that a venue will accept the paper.

**External-data update:** public, outcome-blinded AusPlots metadata identify
119 cells after the historical AEKOS buffer and published PO training-grid
exclusions. Same-region native masks support 113 cells. None has matching
feature-location inputs. This supports source feasibility and gives a
quantified limit on a stricter >10 km training buffer, with seven blocks
remaining. [AusPlots audit](ausplots_metadata_audit_results.md).

**T5/T6 update:** fixed-prediction replay exposes model- and score-convention-dependent
bias, including a VegBank MLP/DeepSDM point-order reversal under the moderate
rarity-weighted mask. Paired completion of the same per-cell inventory reduces
mean contrast width by 59.3% versus subtracting separate score ranges, identifies
80/240 practical orderings versus 0/240, and correctly leaves all ten reversed
VegBank orderings open. These are empirical results on consumed SAGE records.
Existing work gives sharp shared-label F1 bounds for binary prediction after
disagreement labeling. Here the estimand is the average of per-cell species-set
F1 differences under genus-level censoring. Position it as an ecological
measurement and verification study.
[T5](observation_replication_t5_results.md) · [T6](paired_completion_t6_results.md) ·
[Fernández-Barrios et al.](https://arxiv.org/abs/2611.01712).

**T4 update:** [matched static/adaptive controls](adaptive_baselines_t4_results.md)
complete 300 executed CPU trajectories / 360 priced paths. Equal-quota static
paired width is ordinary width on the same scope up to a constant, closing that
novel-score branch. Learned-cost adaptation saves 4.25–9.03% against the best
static reference, below the 10% target in all four nonzero-error/price strata.
Dynamic widths help; the learned forecast has little consistent incremental
value. Simpler adaptive controls are competitive. These are generic heuristics,
not new methods or faithful published VMA/LURE implementations. T5 completes the within-benchmark replication. External record/cost
validation remains pending. All 426 tests and independent reconstruction pass; this establishes
execution correctness, not scientific novelty or field time savings.

**T3 update:** [charged linkage verification](linkage_robustness_t3_results.md)
runs 360 trajectories with invariant inventories. Unchecked propagation creates
invalid contrast ranges under both nonzero membership-error doses; no tested
ordering reverses. Ideal audits preserve validity, with cost frontiers that
depend on scope and price. The reference criterion fails: 9.34% / 7.71% versus
the strongest matched shared control, below 10%. Conditional savings against
local verification remain at low prices. Perfect audit/category assumptions,
fixed cohort/model and synthetic costs prevent a field-time or general-method
claim. See [paper readiness](thesis_paper_status.md).

**T2 update:** [shared-concept verification costs](taxonomic_verification_t2_results.md)
are evaluated in 100 fixed CPU trajectories. The primary criterion fails through
local-control saturation. Predeclared secondary practical-ordering costs use
27.7% fewer median shared queries and 39.5% fewer local queries than the best
tested controls, with lower geographic query coverage in the shared setting.
This is conditional evidence under reliable linkage and complete recorded
inventories, not a new general active-comparison algorithm, measured annotation
time or fresh confirmation. [Paired active model selection already exists](https://link.springer.com/article/10.1007/s10994-024-06603-1).

**A2/A3/A3u update:** [the selected-record replay](neon_taxonomy_a2_a3_results.md)
reproduces prior labels exactly, audits six history keys and leaves all archived
labels out of current targets. A2 changes only uncertainty masks on confirmed
labels, raising fixed-77-cell observed F1 by .026839 while eligibility drops
77→31. Eight component controls attribute this to the mask implementation.
A3's exact accepted-species aliases recover 123 cell–species presences. The
separately preserved A3u alias-uncertainty guard keeps all positives and marks
199 additional pairs unknown; only 33 cells/six sites pass the final cutoff
(original A3: 34). No new-method success or field
confirmation. T2/T3/T4 now evaluate shared-concept verification cost; T4 closes
the static-score novelty branch. Replication and field grounding remain.

**A1 update:** [the dictionary/metadata audit](neon_taxonomy_a1.md) is complete,
while the field label contract remains partial. Accepted-name links are not
species-parent keys; lexical candidates are not automatic resolved labels.
Only 4,995/44,721 cell–genus pairs pass the necessary species-catalog condition,
which still does not establish a positive eligible species for a field record.
No observation-weighted coverage, new F1 score or genus guarantee is claimed.

**T1 update:** [the controlled resolution study](taxonomic_resolution_t1_results.md)
is complete (60 settings; 5,251 consumed cells). High class-known fraction
(98.3–98.4%) coexists with ~26% positive-occurrence hiding at nominal 20% genus
censoring, but the declared spatial-bias criterion fails and no policy point
ordering reverses. Post-hoc scoring/constraint diagnostics isolate 268 false
perfect empty scores and 64–66% tighter ranges when positive-genus information
is retained. These constraints are supplied by the simulator; NEON applicability
is not established. This supports a narrow measurement-audit candidate, not a
new-method claim. [Active testing and partial-identification precedents](taxonomic_evaluation_prior_work.md)
rule out claiming either general idea as our invention.

**E4 update:** [NEON transfer](confirmation_e4_results.md) runs but fails sample/
label admission (65 cells/10 sites; required 100/12). A post-hoc correction of the
omitted documented morphospecies join yields 77/11, still inadequate. HGB MSE is
worse than the training mean; F1 contrasts are uncertain. Missing-label bounds
and attrition are post-primary, contract-dependent diagnostics, not a new theorem
or a finding that NEON intrinsically lacks usable data. Subsequent T1/A2/A3/A3u
and T2/T3/T4 audits are development evidence, not rescued E4 confirmation or a
successful new method.

**NEON bridge update:** [model/provenance checks](neon_bridge_status.md) establish
MLP CPU/T4 reproduction, not a scientific effect. The ResNet reference fails.
The 368-plot metadata candidate set has 301 plots in original PO training grid
cells and none >10 km from training. A future study may test reliability-head
transport across survey sources; unseen-backbone-training-location claims are
unsupported. The bridge was metadata/reference work; subsequent E4/A2/A3/A3u
outcomes are now consumed. No new novelty is asserted.

**E3 update:** the locally locked larger-plot confirmation fails the proposed
joint superiority claim. All four MSE point estimates favor HGB over scalar SDC,
while F1 intervals are inconclusive. A post-primary training-mean control has
lower point MSE than scalar SDC on E2. The earlier scalar-versus-HGB contrast must
not be sold as useful confidence skill or general transfer dominance. See
[E3 results](confirmation_e3_results.md) and the revised methods/results draft.

| Topic | Established primary precedent | Our completed evidence | Defensible scope / remaining requirement |
|---|---|---|---|
| F1-aligned positive-set confidence | [Soft Dice Confidence, 2026](https://link.springer.com/article/10.1007/s10994-026-07096-w) | The existing F1 surrogate has the same ratio; E1/E2 use it as a strong baseline | No new-score claim. Evaluate its role in site-level ecological decisions |
| Sampling-aware SDM evaluation | [SAGE, 2026](https://arxiv.org/abs/2609.31082) | E1/E2 reuse released predictions and native-range semantics; evaluate site acceptance with geographic/area budgets | No generic sampling-aware benchmark novelty. Narrow to species-set decisions, calibration roles and selective service |
| Pairwise ranking | [RankNet, 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) | E2 linear, SDC-anchored within-group pairwise learner rejected in all eight outer folds by inner utility validation | Standard loss; no new optimizer. Reject success claims for this tested family, not all possible rankers |
| Confidence beyond survey composition | R1 protocol and its matched metadata/distance control; selective-prediction literature remains to be reviewed more broadly | R1 +.014741 F1 over metadata/distance; much smaller gap against stronger confidence after decoding | Useful controlled development result. All GLC labels consumed; regional safeguard failed |
| Stronger baselines change learned-error conclusions | SDC prior; order-preserving scalar calibration and mean-only regression references are elementary | E2 affine SDC beats HGB MSE but not a training-mean-only reference by point MSE; E3 reverses HGB/affine MSE ordering and fails joint superiority | Diagnosis of comparator choice and transport; no global winner or useful incremental scalar-confidence skill established |
| Error magnitude versus ranking utility | Calibration and selective ranking are established distinct concepts | E2 affine control preserves exact SDC decisions while improving magnitude fit; HGB beats affine in one of four spatial MSE cases, loses in all four source MSE cases | Quantified application-specific comparison. No universal dominance or theorem; require fresh confirmation for a principal prospective claim |
| Geographic representation versus rare-occurrence service | Fair/selective prediction and ecological-service literature need fuller coverage | GLC rare constraint promising in development; locked E1 +.536 points, interval [-.357,+1.388], fails +3-point target. E2 source HGB sacrifices F1 while retaining ~1.984 points more rare occurrences | Preserve failed hypothesis and utility tradeoff. No conservation causality or real-coverage guarantee |
| Entropy alignment | Selective prediction on multi-label decisions is prior work | GLC and E1 predicted-set entropy underperform matched random; operational result replicated across fixed decoders/settings | Only this entropy definition and these settings. Not all entropy or uncertainty methods |
| Remaining selection headroom | Label-oracle ceilings are standard diagnostics | Post-hoc E2 country/area oracle gains .0255–.0326 F1 | Nondeployable, no rare constraint; cannot call it achieved accuracy or proof that a real learner will close the gap |
| Metadata/provenance quality | SAGE upstream plot/area pipeline | Area inconsistencies excluded before E1 labels; native masks and source exclusions checked | Audit and reproducibility contribution; differences do not establish upstream error or full independence |
| Scope-matched static score and adaptive controls | Elementary equal-weight algebra; active testing and paired comparison precedents below | T4: 30 score-identity cases; 300 executed/360 priced paths; generic adaptation saves 4.25–9.03% vs best static reference; 10% gate fails; learned forecast adds little consistent value | Static paired-score novelty closed for this setting. No new algorithm/theorem or VMA/LURE superiority. T5 replicates within SAGE. Field grounding remains |
| Link validity versus verification price/scope | Active vetting, paired comparison and [linkage-quality analysis](https://discovery.ucl.ac.uk/id/eprint/10036697/) | T3: 360 trajectories; unchecked ranges exclude truth; ideal paid audits restore validity; cost/scope frontiers measured; 10% reference acquisition target fails | Synthetic target-label compatibility oracle, not a field procedure. No observed wrong ordering, universal sharing benefit or new theorem. T4 and T5 add matched controls and replication within SAGE. Field grounding remains |
| Taxon-query cost and paired utility identification | [Active vetting](https://proceedings.mlr.press/v80/nguyen18d.html), [active testing](https://proceedings.mlr.press/v139/kossen21a.html), [paired variance minimization](https://link.springer.com/article/10.1007/s10994-024-06603-1) | T2: 100 trajectories; primary fails; secondary median query costs fall 27.7% / 39.5% against strongest controls under two cost units; representation tradeoffs | Conditional synthetic query/observation model. No VMA/LURE superiority or new general theory. Audit linkage/error/cost assumptions before promotion |
| Apparent quality versus observation contract | Standard F1 algebra, taxonomic curation and partial-identification evaluation | A2: fixed cells/predictions/confirmed positives, +.026839 known-F1 change through uncertainty masks; A3 adds 123 supported presence labels via exact aliases | Quantified implementation/measurement audit, not new theory or predictive improvement. Distinct contracts change the observed-label estimand |
| Taxonomic observability and fixed policy comparisons | Taxonstand/OCCUR; missing-label learning; [active testing](https://proceedings.mlr.press/v139/kossen21a.html); [performance partial identification](https://papers.neurips.cc/paper_files/paper/2024/hash/f4c6bec746b0aeca8c2cd15096f1ad1f-Abstract-Conference.html) | T1: 60 settings; class-known admission misses substantial positive loss; primary criterion fails; no ordering reversals; post-hoc genus facts tighten ranges and recover some signs | Application-specific controlled development. Standard bounds and active evaluation are established; no empirical NEON mechanism or field-positive-genus contract established |

## Rejected generalization and current empirical scope

The E2-driven proposal that scalar-calibrated SDC would outperform the fixed
learned regressor on both magnitude estimation and selection in a new domain is
**not confirmed by E3**. E3 F1 difference is +.002471 with 97.5% interval
[-.010984,+.011744]; HGB-minus-scalar MSE is -.010583, interval
[-.016076,+.004052]. Post-hoc constant controls further narrow interpretation.

The current empirical contribution candidate is a controlled audit of how
observability and verification contracts change ecological model comparison.
Error-magnitude skill, acceptance ordering and ecological service are distinct
supporting objectives requiring strong comparators. Source/plot-area changes are jointly observed;
their causal effects are not isolated. Standard scoring and calibration theory
already makes the conceptual distinction; our potential contribution is specific
SDM evidence and operational evaluation, subject to novelty review.

The numerical comparisons are measured. A broad claim that learning reliability
is unnecessary is not supported. Neither a successful new algorithm nor a new
theory is established. The next confirmation must fix this narrow statement,
comparators, practical margins and outcome roles on an audited source/domain
whose outcomes have not driven these choices. E1/E2 cannot be renamed fresh.

## Drafting decisions

- Make the empirical evaluation the central contribution candidate; treat the
  failed regional, ecological and ranking prototypes as transparent tests.
- Separate GLC reused development, E1's locked primary failure, E1 descriptive
  secondaries and E2 development throughout tables and prose.
- Report cell/source/block counts, calibration composition, fixed versus refit
  probability transforms, decision policy and rare-definition differences.
- Review neighboring selective SDM, calibration-transfer and multi-label
  decision literature before settling a final novelty paragraph or target venue.
- A subsequent GPU experiment is authorized if its recorded implementation and
  scientific checks pass. More GPU compute alone is not a missing novelty claim.
