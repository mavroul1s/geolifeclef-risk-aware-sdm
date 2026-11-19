# Proposed T2: cost of resolving paired ecological utility

**Proposed, not executed.** A2/A3/A3u complete a bounded current-record, exact-
synonym and alias-uncertainty audit; the guarded contract leaves only 33
cells/six sites above the field completeness
cutoff. A further generic model sweep does not address the remaining question:
how much taxonomic verification is required to compare fixed selection policies?

## The experiment to define next

Use fully recorded historical SAGE labels as a development oracle, with the
fixed verified MLP, decoder and country/area-constrained selectors. Keep source
probability/risk roles separate from the replay cohort. Existing consumed data
cannot provide a new independent confirmation.

Simulate unresolved **shared taxon-concept keys**, analogous to site-specific
morphospecies identifiers. A key may occur in multiple cells, and one resolution
updates all of its linked records. Give the policy anonymous keys, their observed
genus/cell incidence and predictions; do not expose the hidden species ID in a
key or use oracle losses to prioritize queries. Key consistency is a declared
generating assumption, not an inferred field fact.

Generate records from the complete native label space, including positives
outside the fixed modeled target subset. A query can resolve to a target class
**or an outside-target species**. An unresolved genus must not imply at least
one modeled positive. Resolving one positive record does not automatically
reveal every absence in its genus: class masks clear only when remaining
unresolved records no longer support that class. State the inventory and
shared-identity assumptions explicitly.

The query unit/cost is one shared-concept resolution, distinct from one complete
cell inventory or one individual field specimen. Report how many records/cells
each query affects. This is a synthetic cost model; do not call it measured
taxonomist time or monetary cost.

Compare fixed candidate policies under the same cost: uniform keys,
uniform keys affecting the paired-policy disagreement support, generic
uncertainty/interval-width prioritization, and paired-contrast-weighted
prioritization. Shared accepted cells cancel in a fixed paired contrast; that
elementary algebra alone is not a new algorithm. Keep geographic and rare-
occurrence representation explicit, and do not hide a trivial advantage over
only a weak global-random control.

Primary outcomes should concern remaining full-set contrast identification
width at matched cost and cost to identify a practically nonzero finite-cohort
policy ordering. Report oracle containment and wrong-sign declarations,
including failures to identify the contrast. Distinguish finite-cohort
identification from spatial-population inference. Keep mask-seed variation
separate from ecological sampling and fitted-model variation.

## Before execution

Implement and test the observation/query semantics on small exhaustive cases,
including overlapping unknown concepts and outside-target resolutions. Freeze
mechanisms, source roles, budgets, strong controls, practical margins and CPU
limits before the main replay. Use calibration/development roles for any choice
of query-policy parameters. Do not enlarge budgets after looking for a favorable
sign. This is one bounded CPU study with **zero new backbone fits**; exact replay
counts and query budgets await the cost/feasibility check.

## Novelty boundary

[Active Testing](https://proceedings.mlr.press/v139/kossen21a.html) already studies
label-efficient evaluation; [partial-identification evaluation](https://papers.neurips.cc/paper_files/paper/2024/hash/f4c6bec746b0aeca8c2cd15096f1ad1f-Abstract-Conference.html)
already bounds performance without full truth. Shared-label/group-query and
paired active-comparison literature must be checked before a method claim.
The specific candidate is an ecological observation/query model with shared
taxon identities, outside-target records and quota-constrained utility. Novelty,
operational benefit and a paper-ready new contribution remain unestablished.
