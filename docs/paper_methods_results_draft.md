# Methods and results draft — 19 November 2026

Revised after E3/E4, T1 and A2/A3/A3u development. Working title: **Decision Confidence,
Error Calibration and Ecological Service Across Survey Domains**. The empirical
argument concerns distinct objectives and transport, not a globally superior
scalar or learned method. Post-primary mean controls are labeled explicitly.

Working text for supervisor review. This is not a completed manuscript or final
abstract. Prospective and development evidence must retain the labels below.
Primary-source positioning and claim limits are in
[the evidence matrix](paper_claim_evidence_matrix.md).

## A2/A3/A3u: current records, uncertainty masks and exact synonyms

We audit only the consumed 301 plots/163 NEON cells. The old morphospecies labels
reproduce bit for bit before a stricter development contract is applied. Six
linked history records are used for provenance, never as replacement labels;
unselected/protected higher-rank identities are not reverse-resolved. Full
accepted-chain compatibility with conservative genus/family fallback adds 6,782
unknown label pairs, leaving predictions and 4,161 confirmed positive pairs
unchanged. On the same previous 77 cells, known-label F1 rises .409282→.436122,
while the >=80%-known subset shrinks 77→31 cells. A post-hoc eight-arm component
ablation identifies the mask implementation as the cause of this attrition.
This is an observed-target change, not a predictive gain or bias against a
complete-field oracle.

A separate frozen development extension matches current named taxa to a unique
model alias through explicit accepted-species equivalence, before native-mask
filtering. It adds 553 dictionary links and 123 modeled native cell–species
presence labels, preserving existing positives. Eligibility recovers only to
34 cells/six sites before the alias-uncertainty guard. The separately frozen
A3u overlay expands masks by 199 cell–class pairs while preserving all 4,284
positive pairs. Final eligibility is 33 cells/six sites, HGB MSE .088485 versus
training mean .050904, and HGB-minus-SDC +.002319 with descriptive 97.5% site
interval [−.015017,+.013260]. All paired HGB-minus-SDC site intervals remain inconclusive;
full-set identification ranges still span both signs. These standard taxonomy
corrections strengthen the measurement audit but do not establish a new method,
biological discovery or fresh confirmation. See [A2/A3](neon_taxonomy_a2_a3_results.md).

## A1 metadata extension: applicability of genus information

Before applying the synthetic genus constraints to field records, we audit all
93,698 NEON plant dictionary entries from 94 hash-verified pages. Accepted-name
chains resolve, but accepted identifiers do not generally supply species-parent
links. Among 24,940 infraspecific entries, 9,047 end at an accepted species,
15,853 yield lexical candidates requiring validation and 40 lack usable evidence
under the conservative parser. These are dictionary counts, not observation
coverage. In 44,721 eligible cell–genus metadata pairs, 4,995 pass a necessary
species-only catalog condition. Passing is not a field guarantee: catalog,
infraspecific and outside-universe uncertainty remains. We do not read new
composition values, refit predictors or revise E4 labels. A2 subsequently audits selected-key history, and A3/A3u add exact synonyms and
conservative alias masks. Authority-backed parent reconciliation remains bounded
and unresolved where not supported. See
[A1](neon_taxonomy_a1.md).

## T1 extension: controlled observability and scoring semantics

We then perform an explicitly adaptive development stress test on all 5,251
consumed E1 assessment cells. The released MLP, E4 probability transform,
expected-F1 sets and five country/area-constrained selectors remain fixed.
One CPU reconstruction of the E4 HGB on its original calibration role reproduces
saved predictions exactly; no neural training is added. Canonical genus and
training-only relative prevalence define uniform or rarity-weighted censoring
of positive genera. Doses 0/20/50%, rounded upward per cell, and ten fixed seeds
give 60 mask settings, not independent ecological replicates. All eligible
positive and negative labels in selected genera become unknown. Complete
recorded benchmark labels remain an evaluation oracle within the modeled subset.

At nominal 20%, 98.42% / 98.29% of class labels remain known while 26.02% / 26.14%
of positive occurrences are hidden. All cells still pass >=80% class-known
admission. Known-only score bias averages +.019383 / +.022982, but the paired
97.5% spatial intervals cross zero, failing the declared .01 practical-evidence
criterion. No tested paired policy ordering reverses, and contrast distortions
remain below .01. Independent-label completion ranges leave all three paired
contrast signs unidentified at nonzero doses. This does not establish observed
ranking reversal or a major geographic/rare attrition effect.

An explicitly post-hoc diagnosis isolates scoring semantics. At every nonzero
setting, 268 positive-truth cells with empty predictions lose all observed
positives and receive empty/empty F1 = one, contributing +.051038 to mean bias.
Changing empty/empty to zero changes the estimand and yields negative known-only
bias at 20% (−.031655 / −.028056); it does not rescue the frozen criterion.
Retaining the simulator's remaining positive-genus facts reduces mean completion
width .122958→.041951 / .122618→.044705 and identifies some policy signs.
These are elementary finite-completion constraints, not new theory. A real
higher-rank record may lie outside the modeled universe; field applicability
requires the pending NEON contract audit. See [the T1 report](taxonomic_resolution_t1_results.md)
and [neighboring active-evaluation/partial-identification work](taxonomic_evaluation_prior_work.md).

## E4 extension: modeled-native-set transport with taxonomic uncertainty

A subsequent locally locked study fixes one released MLP and 163 NEON cells
(301 plots, 16 sites) with identical-coordinate public predictor vectors.
All lie in original backbone training support; the holdout concerns probability
and reliability calibration, fitted only on 1,165/1,061 original SAGE calibration
rows. Both domains use the 2,453 unambiguously dictionary-matched classes within
published native masks. Unresolved taxa create unknown labels, not absences;
evaluation removes them without changing frozen sets or confidence features.
The protocol requires >=80% known modeled-native labels per cell, >=100 cells
and >=12 sites. At matched 80% site/plot-count-group quotas, two primary contrasts
test HGB MSE skill over a training mean and HGB selection loss relative to SDC.

E4 fails admission: only 65 cells/10 sites remain. Descriptively, HGB quality MSE
is .104674 versus .049188 for the training mean. HGB-minus-SDC selective F1 is
+.007478, 97.5% site interval [-.010459,+.030027]; the joint claim fails.
The first pass omitted NEON's recommended morphospecies-resolution join.
A post-hoc replay using that join gives 77 cells/11 sites and MSE .093484 versus
.044036 for the mean, still below admission. No model or head is refit, and the
replay is not fresh confirmation. Attrition is specific to this harmonization/
unknown-label contract and does not establish intrinsic NEON inadequacy.

Post-primary elementary binary-completion bounds show that even the original
65-cell subset has mean possible full-modeled-set F1 width .2351. The paired
selection contrast can take either sign. These identification ranges differ
from statistical confidence intervals and are not new theory; they motivate a
controlled taxonomic-censoring development study before another field claim.
See [the complete E4 report](confirmation_e4_results.md) for access order, GPU
verification, failed admission, corrections and limitations. All 163 outcomes
are now consumed. Neither a successful new algorithm nor a novel general
missing-label framework is established.

## Methods: decisions and selective service

We evaluate species-set predictions at the survey or aggregated-cell level.
Predictors and class rankings are fixed when comparing decision rules. Full-
coverage sample F1 measures species-set utility; selective F1 measures utility
among accepted cells. Geographic and plot-area representation are controlled by
giving every selector identical country-by-area-bin budgets, allocated by largest
remainder. Acceptance counts are fixed separately in each evaluation fold.

We distinguish occurrence service from predictive recall. Rare-occurrence service
is the proportion of all recorded rare presences located in accepted cells;
rare recall additionally requires that the predicted set contain those presences.
Neither quantity establishes conservation causality or threatened-species benefit.
We report rare true positives, precision, recall and retention alongside F1.

Strong confidence baselines include mean selected probability and established
Soft Dice Confidence (SDC). We compare the tested learned error regressor with
these baselines, rather than only metadata or distance. A monotone affine fit
calibrates SDC's error magnitude while preserving its acceptance ranking. We do
not claim the SDC ratio, F-measure optimization or monotone calibration as new.

## Methods: evidence roles and source transport

GeoLifeCLEF experiments reuse historically consumed labels and are development
evidence. E1 subsequently fixes a local analysis of a public SAGE subset before
opening its assessment targets. The cohort has 1,165 probability-calibration,
1,061 risk-calibration and 5,251 assessment cells. Roles occupy distinct 5-degree
blocks with at least 50 km between roles. A conservative metadata/proximity audit
and pre-outcome plot-area consistency check define admission. Released native-
range missing targets are excluded, not converted into absences.

Four published model settings represent two model families and two seeds each.
Their original presence-only training can include evaluation cells. Our spatial
holdout therefore concerns the reliability pipeline, not a claim of disjoint
backbone training. The four settings share evaluation cells; uncertainty uses
the same spatial block draws across models rather than treating model-cell pairs
as independent observations.

E2 is explicitly post-E1 development and uses calibration roles only. Five
buffered outer spatial folds and three source-exclusion pipelines each contain
three spatial inner folds. Source-exclusion removes an entire survey source from
both probability calibration and selector training. Probability transforms and
area bins are refit without that source. A regularized linear residual ranker,
anchored to SDC and trained on within-group F1 preferences, is promoted only if
inner utility exceeds unchanged SDC by .001. Fixed penalties and a baseline
fallback limit tuning. E2 does not request E1 assessment labels.

E3 then locks a confirmation on 654 uninspected larger-plot SAGE cells in 19 new
spatial blocks, all >=100 km from consumed reliability cells, with disjoint plot,
cell and block identities. All contributing plot areas exceed 400 m². Original
E1 calibration roles fit the fixed heads; four policies are frozen before E3
target access. The co-primary test requires practical improvements in both
all-cell quality MSE and matched-quota retained F1, with 97.5% paired spatial
intervals and per-model guards. E3 is a new domain within the same published
release; some sources and the pretrained models' original PO support overlap.

## Results: apparent reliability gains depend on the comparator

In GLC development, the controlled R1 selector improves matched-coverage F1 over
metadata/distance by .014741, but fails its regional safeguard. With a stronger
mean-confidence baseline and decision-aligned decoding, the later learned
selector's advantage shrinks to .000885, with an interval crossing zero.
Historical competition gains supply engineering context and are not comparable
with these internal selective F1 values.

In E1, the learned selector loses .004175 F1 to SDC on average at the primary
80% operating point; this is a descriptive secondary contrast. In E2's
source-exclusion study, the corresponding HGB loss is .007246, descriptive
95% spatial interval [-.013156,-.000562]. A scalar calibrated-SDC control obtains
lower error MSE than HGB in all four source-held-out settings: mean MSE .017511
versus .024551. The HGB-minus-control difference is .007039, descriptive interval
[.003595,.010306]. Scalar calibration preserves every SDC acceptance decision.
The spatial MSE contrast is inconclusive and one model favors HGB, limiting any
broader dominance claim.

## Results: prototype gains do not establish ecological benefit

The rare-count constraint improves GLC development occurrence service by 5.41
percentage points at an absolute F1 cost .003461. E1's prespecified primary test
does not reproduce the intended practical gain. Mean occurrence-service change
is +.536 points, 97.5% interval [-.357,+1.388], below the declared +3-point target.
Mean F1 change is -.000050. Both forest policies are unchanged; raw rare-mass
constraints are inactive in all four primary settings. The primary hypothesis
fails and secondary contrasts do not replace it.

E2's pairwise corrections are rejected in all eight outer procedures by inner
validation, so the resulting policy is exactly SDC. Its zero outer contrasts are
policy identity, not a proof that learning cannot help. Under source exclusion,
HGB retains 1.984 percentage points more rare occurrences despite lower F1.
Accuracy, representation and ecological service are therefore reported as
distinct operational outcomes.

## Results: locked domain confirmation rejects joint superiority

The E2-driven joint-superiority hypothesis fails in E3. SDC's mean retained-F1
change relative to HGB is +.002471, 97.5% spatial interval [-.010984,+.011744].
HGB-minus-scalar MSE is -.010583, interval [-.016076,+.004052]; all four model
point estimates favor HGB for magnitude fit. Neither endpoint establishes the
joint claim. The 19-block cohort is geographically heterogeneous and Brazil
supplies 363 cells in only two blocks, limiting inference about source populations.

Under the predeclared finer area-quota sensitivity, mean F1 difference falls to
.001359 and both forest point comparisons favor HGB. All E3 cells occupy the
original top area bin, so the primary bins do not finely control area within
the new range. Empty-set scoring conventions and magnitude clipping do not
change the reported primary numbers. No favorable secondary replaces the primary.

## Post-primary diagnosis: magnitude skill needs a mean reference

We subsequently add a training-mean-only quality predictor, fitted without each
evaluation fold's outcomes. It has point MSE .016988 on the E2 source study,
versus .017511 for scalar SDC and .024551 for HGB. Thus scalar SDC's earlier
advantage over HGB does not itself show useful incremental confidence information.
The distinction between a point forecast's target and its scoring rule is
established ([Gneiting, 2011](https://arxiv.org/abs/0912.0902)); these controls are
diagnostics, not new scoring theory or a replacement primary test.

On E3 the mean-only reference has MSE .029319, scalar SDC .031001 and HGB .020419.
Both DeepSDM scalar fits reach their near-zero positive slope floor and largely
forecast the old training mean. Mean observed F1 rises from roughly .182 in old
riskcal to .267 in E3. HGB's quality estimates track the new mean more closely.
An exact descriptive bias/variance/covariance decomposition reconstructs the MSE
reversal. This does not isolate the causal effect of plot area versus source.
HGB's approximately 30.4% MSE skill over the mean-only reference on E3 remains a
descriptive point result; it does not establish a corresponding selection benefit.

## Limitations and next field-source evaluation

E2 outcomes were already consumed; its intervals condition on development fits.
Source coverage is limited to three supported groups, and SAGE's aggregated
cells differ from GLC surveys. Rare species are defined by recorded prevalence,
with different operational definitions across datasets. Metadata equality cannot
prove complete provenance; published backbone training follows its original
overlap protocol. These results motivate a narrow empirical claim about strong
baselines and source transport, not universal failure of learned uncertainty,
a successful new algorithm, biological causality or guaranteed risk control.
A new primary confirmation requires an audited source/domain and fixed claim.

E3 supplies a failed locally locked domain confirmation, and those outcomes are
now consumed. It does not provide a separate independent data release. The next
candidate field source is NEON: official metadata for 47 sites and released
neural input contracts are audited, but plot completeness, taxonomy, predictor
extraction and reference inference remain pending. No NEON outcomes or new model
performance are claimed. A neutral inference study may use the authorized GPU
after these input/reference checks; scientific outcomes need not be favorable
to admit that study. Novelty and a successful new algorithm remain unestablished.
