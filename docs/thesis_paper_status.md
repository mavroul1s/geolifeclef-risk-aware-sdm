# Thesis progress and paper path — 22 November 2026

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

The [T4 stronger-baseline package](adaptive_baselines_t4_results.md) is now
completed: **300 executed CPU trajectories / 360 priced paths**, six generic
controls, zero fits/inference/GPU jobs. It establishes that equal-quota static
paired width is ordinary width restricted to the same verification scope up to
a constant. That closes the standalone score-novelty branch. Canonical static
first-ordering costs exactly match all 60 repriced saved T3 paired costs.

Generic learned-cost adaptation reduces mean charged cost **4.25–9.03%** against
the per-mask best static reference and is cheaper on every mask. The frozen 10%
gate nevertheless fails in all four nonzero-error/price strata. Width updates
help, but learning the mismatch forecast adds little consistent value; simpler
adaptive controls are competitive. No new algorithm or published-method
superiority follows. **426 tests**, 288 independent dense prefixes, 108 batch
choices and 720 cost endpoints validate the implementation, not external truth.

The strongest current direction is **observation contracts and verification costs
in ecological model evaluation**: quantify how apparent quality, policy ordering
and verification effort depend on what labels are known and what a query truly
reveals. The paper should connect the measurement diagnosis to a valid, fairly
costed procedure and its limitations.

Active testing, paired model selection, linkage-error propagation and performance
partial identification already have prior art. The current static heuristic is
mathematically redundant with a matched generic baseline; it cannot anchor a
new-method claim in this setting. Neither the adaptive controls nor elementary
bounds establish new theory.
The empirical combination still requires careful novelty review and replication.
See the [claim matrix](paper_claim_evidence_matrix.md) and
[T2 prior-work review](taxonomic_verification_t2_prior_work.md).

I consider a paper a plausible outcome of this thesis, but a strong submission
is not yet earned by the experiment count. Its credibility must come from a
precise contribution, fair comparisons, realistic assumptions, robustness and
adequate validation. No acceptance or venue-level guarantee follows here.

## New model and survey-contributor evidence

T5 uses a shared taxonomic universe and the same known labels for the MLP and
two additional verified SAGE prediction settings. It covers all 5,251 cells
and three distinct contributing survey inventories within that cohort. At
moderate synthetic censoring about 98% of class pairs stay known while 26% of
positive occurrences disappear. Known-only F1 increases for the original MLP
under the empty=1 convention, decreases for both additional models and
decreases for all three under empty=0. The VegBank rarity-weighted masks
reverse the small MLP–DeepSDM point ordering in all ten repetitions.
[T5 results](observation_replication_t5_results.md).

T6 carries the same unknown species labels into the scores of both models.
Compared with subtracting separate score ranges, this reduces mean width by
59.3% and identifies a practical order in 80/240 evaluated cases versus 0/240.
All ten VegBank reversals remain indeterminate under the shared inventory.
This is a finite-cohort result under binary completion and correct known
labels. Existing work gives sharp F1 comparison bounds for binary classifiers
after disagreement labeling. In this thesis the ecological estimand is mean
per-cell species-set F1 and arbitrary species can be unresolved.
[T6 results](paired_completion_t6_results.md) ·
[Related F1 work](https://arxiv.org/abs/2611.01712).

T5 also replays six shared verification controls and two local controls. At
5% nominal link error, an audit price of .1 saves 38.6–67.2% across the six
model/cohort settings at the same .002 target width. Raising the price to .5
makes AEKOS 10.7% costlier than local identification. Unchecked propagation
excludes the benchmark contrast in all 48 paths at some checkpoint. Ideal
paid audits preserve the tested endpoint validity. These are normalized
synthetic costs.

## What remains for the paper

Lead with the effect of unresolved species on **which ecological model
comparisons the observations support**. Use charged verification and the
NEON taxonomy audit to explain how that uncertainty arises and what resolving
it could cost. Put the competition and earlier confidence experiments in the
thesis background or paper supplement. Separately report old failed primary
criteria and the exploratory T6 status.

Within-benchmark replication is complete. The outcome-blinded AusPlots screen
found 151 visits across 119 cells outside the historical AEKOS buffer and
published training grid. Region-validated native masks cover 113 candidate
cells. No candidate has an exact released feature-location match, and a
separate voucher-inventory contract and reference-validated comparison model
are needed. Field effort and external outcome confirmation remain open.
[AusPlots results](ausplots_metadata_audit_results.md) ·
[Next evidence packages](next_experiment_budget.md).
