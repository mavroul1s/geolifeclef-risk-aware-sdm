# T3: linkage validity and charged verification — development protocol

Prepared 21 November 2026. Freeze configuration, scientific/validation code and
input hashes after semantic tests and construction-only feasibility, before the
main replay. Preserve T2's failed primary and separately reported secondary
costs. All outcomes are consumed; this is not independent confirmation.

## Fixed data and target

Reuse T2's ten masks, 5,251 cells, 2,453-class/native target, frozen predictions,
HGB/SDC acceptance sets and complete recorded oracle. No new fit, inference,
composition download or GPU job. The finite-cohort contrast remains HGB − SDC,
with tolerance .002 and practical sign margin .001. Bounds remain independent-
binary outer ranges, not sharp shared-key bounds or population CIs.

T2 cached identities distinguish target species indices and one outside-target
category. T3 therefore stresses **target-label compatibility**, not every
botanical synonym/identity error. Outside/outside botanical mismatches are not
represented. The complete benchmark inventory is not complete occupancy truth.

## Corruption with invariant records

Requested error doses are 0/1/5% of hidden record memberships. Swap records
between different-label groups within the same original spatial block and
public genus. Each group has at least two members and retains an original
member; no group gains duplicate cells. A record participates at most once.
Records carry their actual cell and private target label together. Thus each
cell's hidden genus/label multiset, known positives and initial uncertainty are
UNCHANGED. Only proposed record-to-key linkage changes. This is intentionally
informative sensitivity stress, not an estimated NEON linkage mechanism/rate.

Use one bounded, seeded maximum-dose swap sequence and prefixes for nested
doses. Choose pairs uniformly among eligible block/genus key pairs and records
uniformly among unused positions; at most 100 attempts per desired maximum-dose
swap (minimum 1,000). Report requested/realized rate and feasibility without
redrawing to improve outcomes. Construction feasibility must attain the planned
doses in every mask before freezing; inspect no new policy/F1 outcomes there.

## Three information procedures

**Unchecked all:** identify one representative record and propagate its label
to all members without checking links. This is a deliberately assumption-breaking
reference under corruption. Charge one full identification. Report incorrect
known labels, oracle-containment failures and wrong practical declarations;
exclude unsafe output from valid-cost superiority claims.

**Audit all:** identify the representative, then pay for a perfect binary
target-label compatibility audit of every follower. A mismatch returns only
false, so a separate full identification is charged before resolving it. This
procedure can recover full labels for queried groups even when links are wrong.

**Audit support:** use the same audit/fallback procedure only for public members
with nonzero fixed comparison weight and at least one initially unknown eligible
class in the genus. Skip groups with no such members, without a charge. All
other records stay unresolved; never propagate unverified labels to them. This
scope is fixed before querying and uses no private identity or oracle F1.

Choose the representative with smallest public row index among the procedure's
members. One group is an atomic action; test stopping after that action completes.
Every verified member updates ordinary record-level uncertainty counts. Other
compatible absences remain unknown until remaining records in that cell/genus
resolve. Known positives override overlapping uncertainty as before.

The perfect compatibility audit is an explicitly ideal information oracle, not
a tested taxonomist or an error-tolerant field guarantee. Its answers and full
identification answers are separate private operations. All guard-validity claims
are conditional on this oracle and the complete recorded-inventory contract.

## Cost, comparators and compute budget

Full record identification costs one unit. Total charged cost is
`identity_resolutions + audit_price * follower_audits`.
Audit price ratios are 0/.1/.5/1, sensitivity scenarios rather than observed time
or money. Mismatching followers incur both an audit and a full identification.
Prices rescore the same query paths; no query policy receives hidden answers or
unpriced audits. No label-driven tuning of acquisition orders.

Retain four T2 fixed batch policies: support-uniform, support-incidence, global
width and paired width. Recompute public priorities for the corrupted incidence,
using the same T2 seed+101 tiebreaks. Audit-support filters its order to public
relevant groups; the global score itself remains unchanged. Unchecked/all-audit
procedures keep their ordinary policy order. Stronger adaptive baselines remain
the next separate study; these heuristics are not VMA/LURE reproductions.

10 masks × 3 error doses × 3 procedures × 4 policies = **360 new trajectories**,
with 7 T2-style group-count checkpoints (2,520). Four prices generate 1,440 cost
evaluations without rerunning queries. Reuse the already validated T2 local
stopping costs as the unchanged record-level reference: swapping proposed shared
memberships does not change those original local records or keys. Include the
best available local acquisition per mask, INCLUDING paired width; do not use
only the weaker local controls from the earlier secondary comparison.
Main replay maximum is 3,600 seconds; zero new neural or CPU model fits.

## Declared primary and secondary outcomes

Validity first: both audit procedures must have zero incorrect known labels,
oracle-containment failures and wrong practical declarations at every checkpoint
and finite practical stopping point. At completion the comparison range must
collapse to the true contrast. Unchecked procedures may fail and are diagnostic.

The **reference robustness criterion** is conditional on a .1 audit price:
at BOTH nonzero error doses, audit-support paired width must lower mean cost to
a practical ordering by >=10% versus EACH audit-support acquisition control and
the best local method per mask, improving in >=8/10 masks for each comparison.
Guarded validity must also pass. This is a descriptive development gate, not a
population test or a claim that real auditing costs .1 identifications.

Report all prices, not only the reference. For the same stopping prefix, cost
is linear in price. Compute break-even prices against the best local cost as
`(local_cost - identity_resolutions) / audits`, reporting nonpositive/no-audit
cases explicitly. Compare whole cost frontiers, mismatch/fallback counts,
width-tolerance costs, query representation, residual observability, label and
contrast failures. Descriptive paired mask-bootstrap intervals use 4,000 draws
and 97.5% confidence; these quantify mask variation only.

Independent dense reconstruction verifies 0/25/100% for every error/procedure/
policy in the first seed (108 checkpoints). Recompute corruption invariants for
all masks/doses and guard costs from private member labels. Independently check
the practical stopping boundary and preceding prefix for all 120 audit-support
trajectories (240 prefixes), and operation counts at both stopping endpoints for
all 360 trajectories. Verify clean all-
audit/unchecked event prefixes against the original T2 trajectories. Stop before
promoting a method if realistic costs or stronger controls remove the advantage.

## Prior work and limits

[Nguyen et al. (2018)](https://proceedings.mlr.press/v80/nguyen18d.html) already
study active vetting of imperfect annotations. [Hara et al. (2024)](https://link.springer.com/article/10.1007/s10994-024-06603-1)
already target paired model-comparison costs. [Harron et al. (2017)](https://discovery.ucl.ac.uk/id/eprint/10036697/)
discuss evaluating linkage quality for downstream analysis. Neither linkage-error
propagation nor active comparison is our broad invention. T3 tests the empirical
consequences of explicit ecological record, verification and cost assumptions.
No real error-rate estimate, validated botanical workflow, new general theorem,
benchmark gain, rescued E4/T2 primary or fresh confirmation is implied.
