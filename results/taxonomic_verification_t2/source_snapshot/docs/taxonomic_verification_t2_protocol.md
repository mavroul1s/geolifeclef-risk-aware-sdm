# T2: shared taxon verification cost — development protocol

Prepared 19 November 2026. Freeze sources, configuration and input hashes before
the main replay. This is adaptive development on consumed E1 assessment outcomes;
neither the local lock nor anonymous keys creates independent confirmation.

## Fixed source and estimand

Reuse all 5,251 T1 cells, its released MLP predictions/decoder and HGB versus SDC
acceptance sets: 4,201 cells each, identical country/area quotas. No model,
reliability head or probability-calibration fit. The target remains mean per-cell
F1 over the fixed 2,453-class universe intersected with published native masks;
empty/empty equals one. Contrast weights are accept_HGB/4201 − accept_SDC/4201.
Shared accepted cells cancel exactly. No selection or prediction uses labels.

The simulator reads all 5,771 recorded native class labels, including positives
outside the 2,453-class target. These are complete benchmark records, not a
claim of complete biological occupancy or of species outside the released
vocabulary. Target truth/F1 must reproduce T1 exactly.

## Observation and query semantics

One hidden concept is a species within an original five-degree spatial block.
Each of ten fixed seeds hides these concepts independently with probability .5;
all positive native occurrences of a hidden concept become unresolved records
with public genus and cell incidence. Existing named positives remain known.
Randomly permuted key IDs expose neither species indices nor oracle losses.
The query policy receives no private species answers before resolution.

Price precisely the SAME hidden records two ways: one block-shared concept
query updates all linked cells, whereas one cell-local query resolves one
cell's record. Both begin with identical known labels. Assume reliable linkage
and one-unit cost per query; neither cost is measured taxonomist time. Report
record/cell multiplicity and total queries as well as budget fractions. An
outside-target resolution produces no modeled positive. Native eligibility is
respected separately in each cell.

An unresolved record masks compatible target classes in its public genus,
except independently confirmed positives. Querying one positive reveals that
positive but does not reveal other absences while another unresolved compatible
record remains. When all unresolved records in a cell/genus resolve, remaining
target absences become known under the complete-recorded-inventory assumption.
This semantic contract is tested on exhaustive tiny cases; it is not a NEON
field guarantee or an accepted-species harmonization procedure.

## Acquisition controls, cost and budget

Five fixed batch orders: uniform all keys; uniform keys affecting contrast
support; support-incidence priority (number of affected disagreement cells);
global width-credit priority; paired width-credit priority. Relevant support
requires an unknown eligible class and a nonzero contrast weight, not hidden
identity. All tied scores use the same seed-specific random tiebreaks.

For each cell, distribute its initial F1 range width across genera in proportion
to unknown eligible classes, then divide genus credit among its unresolved
keys. Global credit sums across all linked cells with weight 1/N. Paired credit
uses absolute contrast weights. This static heuristic does not forecast answers
or retrain. No parameter tuning from replay utility. Support-only orders stop
when their initial relevant keys are exhausted; report actual charged queries.

Ten masks × two cost units × five orders = **100 trajectories**, with checkpoints
at 0/1/5/10/25/50/100% of each unit's initial hidden keys. Query counts use ceiling.
The same number of queries is available to every policy within a unit/seed.
Also record the exact first query count at which contrast width is <=.002 or
the range lies wholly beyond +.001 or −.001. Failure is right-censored at the
available key count. Query fractions across cost units are not equal absolute
cost; report counts and affected-record costs explicitly. Maximum main replay:
3,600 seconds, zero model fits/GPU jobs. No budget expansion after results.

## Outcomes and frozen practical target

Primary: at 25% budget, paired-width priority must reduce mean contrast width
by at least **10% relative to every strong control** (uniform-support, support-
incidence and global-width), in BOTH cost units, with improvement in at least
8/10 paired mask seeds for every comparison. Both-zero widths count as no
incremental gain. This is a descriptive development criterion, not a proof or
population hypothesis test. Report all arms even when it fails.

Secondary: exact cost to width tolerance/practical sign, represented countries
and rare-positive hiding/recovery, key/record/cell counts, outside-target
resolutions, and residual cell-level uncertainty. Descriptive paired mask-
bootstrap intervals (4,000 draws, 97.5%) measure mask Monte Carlo variation only.
No spatial-population inference or learned-model variation claim.

Ranges use independent binary class completions and the fixed linear contrast.
They are conservative outer ranges: they omit shared-species, genus-record and
cross-cell consistency constraints, so need not be sharp. They are not CIs.
At every checkpoint require oracle containment and monotone interval contraction;
practical-sign declarations must agree with the finite recorded oracle. All-key
orders must restore complete target truth. Independent dense reconstruction
checks 0/25/100% for both units/all five policies in the first two mask seeds
(60 checks), separately from incremental query bookkeeping.

## Interpretation

The [prior-work review](taxonomic_verification_t2_prior_work.md) rules out broad
claims for active evaluation or paired comparison. A successful practical target
would establish only a conditional benefit over these matched heuristics. Shared
linkage, complete benchmark records, nominal taxonomy and the cost model need
field validation. No general algorithm, ecological causal discovery, rescued E4
result or fresh confirmation follows from this study.
