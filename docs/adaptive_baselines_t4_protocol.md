# T4: scope-matched adaptive baseline audit

Prepared 21 November 2026. Freeze scientific sources, configuration, validation
and input hashes before the main replay. This is development on consumed T2/T3
records and one fixed model/cohort, not fresh confirmation. Preserve every
earlier failed criterion and its source/results.

## Analytical audit before further method claims

Both fixed policies accept n=4,201 cells. Their contrast weights are
`w_i = A_i/n - B_i/n`; absolute nonzero weights are the constant 1/n.
For any public per-cell/genus credit c, a group's paired score is
`sum_{i in group and disagreement} c_i/n`.
An ordinary width-credit baseline using exactly the same scope sums the same
credits, differing by a positive constant. Rankings therefore coincide in exact
arithmetic, including division by a common predicted query cost. This remains
true after observing labels if both controls use the same public state.

T3's global-width control included credit from members that its scoped procedure
did not verify. A fully matched baseline removes that mismatch. This elementary
equivalence rules out a novel standalone paired score for this equal-quota
setting. It is not a new theorem or a universal statement for unequal weights.

Construction/numeric preparation checks all 30 saved cases: maximum unscaled
score discrepancy is 3.56e-15. IEEE rescaling changes some tied orders in 11 cases.
Use a canonical unscaled sum and common random tiebreaks for matched names;
report earlier floating-point order differences, never as method headroom.
Do not run identical aliases as independent experimental arms.

## Fixed information, scope and verification

Reuse all 5,251 T2 cells, target/native universe, predictions, HGB/SDC sets,
initial observed positives and all 30 T3 catalogs at nominal 0/1/5% membership
error. Retain T3's fixed initial comparison-member scope, representative rule,
perfect binary target-label compatibility audit and separately charged mismatch
identifications. Do not give adaptive arms cheaper or dynamically narrower
verification scopes. Genus, outside-target/category and complete-recorded-
inventory assumptions remain explicit; this is not measured field cost/time.

One group remains an atomic action. Observe no private identities before it is
queried. Selection uses only predictions, public incidence/initial observations,
subsequently revealed labels and observed audit/resolution counters. Full oracle
F1 is used only for validation, not selection or stopping.

## Six generic controls and targeted ablations

1. Static scope width: the canonical initial width credit on actual members.
2. Static scope width per expected cost: same initial credit divided by a fixed
   initial cost forecast.
3. Adaptive scope width: recompute width credit from the current public state.
4. Adaptive scope width per cost, fixed mismatch forecast: isolates changing
   width information while retaining the initial cost forecast.
5. Adaptive scope width per cost, learned mismatch forecast.
6. Adaptive active-member incidence per cost: counts members with remaining
   unknown eligible classes and positive current F1 width; no width magnitude.

Width credit distributes current cell range width by unknown eligible classes
in its genus, divided by total unknown classes and unresolved record count, as
in T2/T3. Every adaptive control refreshes after batches of 32 group queries;
the next batch is fixed before any of its answers are seen. Stop checks occur
after each atomic group. Use the same seed+101 random tiebreaks in every arm.

Expected group cost is `1 + (members-1)*(audit_price + mismatch_forecast)`.
Static and fixed-forecast arms use Beta(1,19)'s mean .05. Learned arms use
`(1 + observed_mismatches)/(20 + observed_audits)`, where mismatches equal
full_identifications − queried_groups. This is a heuristic forecast under
selectively queried audits, not a calibrated population linkage-error estimate.
No prior/hyperparameter tuning on replay costs. Audit prices .1/.5 are sensitivity
ratios. Charge the actual identities/audits, regardless of forecast error.

## Budget, endpoints and decision

10 masks × 3 error scenarios × 2 audit prices × 6 controls = **360 priced paths**.
The two unpriced acquisition rules have identical query paths at both prices;
reuse those traces and reprice them, giving **300 executed query trajectories**.
Do not count repricing or identical mathematical aliases as new replication.
No model/head/calibration fits, inference, field downloads or GPU jobs. Maximum
main replay 3,600 seconds. Record batch curves plus exact first costs to identify
a practical contrast beyond .001 and reach width <=.002. Execute until BOTH
endpoints occur or all public relevant groups are exhausted; report censoring.

Primary empirical estimand: paired cost differences for learned adaptive width
per cost versus the best of the two static controls per mask, at each nonzero
error/price stratum. A reference operational criterion requires >=10% smaller
mean cost and improvement in >=8/10 masks in all four strata, with validity intact.
The best-static reference is an oracle minimum used as a conservative evaluation
control, not a deployable per-mask selector. Report every other adaptive arm;
passing this criterion cannot imply superiority over incidence or established
active-testing methods. Intervals are 4,000-draw descriptive mask bootstraps,
not spatial-population inference or fitted-model uncertainty.

The analytical audit already closes the old static paired-score novelty branch.
This empirical gate evaluates generic adaptive controls, not a newly invented
algorithm. If an established/simple control wins, report it. Do not enlarge
budgets or alter prior/thresholds to make the named width strategy win.

## Validation and scope limits

At every recorded batch require oracle containment; at both endpoints also
require no incorrect known labels. Ranges contract monotonically. All endpoints must have the correct
finite-cohort sign under the ideal audit contract. Cache actual executed orders.
Independently reconstruct both stopping boundaries and preceding prefixes for
all 72 trajectories in the first two seeds (288 dense prefixes). Verify charged
operations at both endpoints for all 360 paths. For all 36 paths in the first
seed, reconstruct current public information at the initial, middle and last
batch starts and check the selected batch (108 decisions). Check algebraic
equivalence on every saved catalog and all order hashes/chronology.

The controls are transparent interval/incidence and cost heuristics.
[Active testing](https://proceedings.mlr.press/v139/kossen21a.html),
[active vetting](https://proceedings.mlr.press/v80/nguyen18d.html) and
[paired variance minimization](https://link.springer.com/article/10.1007/s10994-024-06603-1)
are established. These paths do not implement VMA/LURE estimators, whose ordinary
queries reveal a complete instance outcome rather than overlapping partial
records. No faithful reproduction or published-method superiority is claimed.
The next evidence needs remain model/source replication and suitable external
record/cost validation, rather than a neural architecture sweep.
