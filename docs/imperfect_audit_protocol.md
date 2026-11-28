# Sensitivity to imperfect voucher compatibility audits

This development study implements the sensitivity proposed in the
[experiment budget](next_experiment_budget.md) while timed, independently
adjudicated human reviews remain unavailable. It replays the ten saved T3 masks
at both the 1% and 5% nominal wrong-link doses. It retains the same public
comparison scope, static `paired_width` order, representative choice, fixed
SAGE predictions and contrast between two selection policies. All inventories
are historically consumed. The existing T6 two-model comparison is unchanged.

At each selected group, fully identify the earliest cell's representative.
For other records, a compatibility decision can erroneously accept an
incompatible member or reject a compatible one. Abstentions and rejections
trigger full identification, assumed correct here. The one-review rule accepts
the representative answer after one compatible decision. The two-review rule
requires a second compatible decision and otherwise resolves the follower
individually. Compare both to a perfect compatibility audit and to direct
identification of every record in the same ordered group. Costs are measured
at whole-group boundaries for all arms.

| Assumed reviewer profile | False compatible | False incompatible | Abstention |
|---|---:|---:|---:|
| Lower error | .02 | .02 | .05 |
| Higher error | .10 | .05 | .10 |

Draw abstention, incorrect decision and correct decision as disjoint outcomes
conditional on actual compatibility. For the independent scenario use separate
draws for each follower and reviewer. For the correlated scenario select half
the groups in expectation to use one common uniform draw for all followers and
both reviewers. Remaining groups use independent draws. This preserves the
per-review marginal assumptions while introducing a shared decision error.
Use deterministic, matched draws across arms and profiles at each mask and dose.
These scenarios bracket assumed noise and dependence. They do not estimate
actual error or an upper bound on it.

Record containment of the original finite-cohort contrast, known-label errors,
wrong practical signs and charged work at 0%, 5%, 25%, 50%, 75% and 100% of the
fixed query order. Record the first apparent sign beyond .001 and the first
apparent width of at most .002, along with their validity against the recorded
benchmark. Report the valid subset alongside all apparent endpoints. Preserve
the conditions of failed or censored endpoints. Evaluate costs at .1 and .5
times one full identification per audit, including second audits and
escalations. Compare with matched local identification under the same order;
this does not benchmark the best separately optimized local ordering.

The perfect-audit arm must reproduce the T3 stopping prefixes and charges.
Check complete local and ideal trajectories for benchmark containment. Rebuild
the known labels, intervals and operation counts independently on the first
mask at both linkage doses. Freeze source, inputs, parameters and role before
reading the replay inputs. This is a CPU-only sensitivity analysis with a
20-minute execution cap. Human time and field validity are separate evidence.
