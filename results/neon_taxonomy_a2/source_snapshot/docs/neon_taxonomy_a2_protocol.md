# A2: selected-record taxonomy and identification-history development audit

17 November 2026. Rules fixed before this continuation retrieves selected
records. All 301 plots / 163 cells are the consumed E4 cohort. This is a
development audit and corrected-contract replay, never a fresh confirmation or
a replacement of E4's failed primary. Original E4, post-hoc morphospecies, T1
and A1 artifacts remain unchanged. Zero fitting or new GPU inference.

## Source and linkage contract

Use the exact original E4 plot/bout selection and previously frozen observation
file identities. Read plant composition fields only after selecting plot/bout
identities at byte level. Add event and identification-history IDs to audit
linkage; do not change the selected events or include another plot/bout. Fetch
only existing RELEASE-2026 files, with published size/MD5 checks and a 100 MB
aggregate bound. Keep selected records in ignored local caches; discard
unselected values without decoding and do not retain raw source tables.

Use the documented current morphospecies join by siteID/morphospeciesID, selecting
the latest valid dated current resolution. Equal-date contradictions remain
unknown. Reproduce the previous 77-cell replay bit for bit as an implementation
control before applying the A2 rules. Original source plant rows and current
morphospecies data supply current determinations. The history table archives
past determinations; it is not an independent occurrence table or a source of
replacement labels.

Read history records only for keys linked to selected current records. Use site
identity as a consistency guard around identificationHistoryID; audit the
identificationTable field as well. For keys with any current higher-rank or
unknown identification, decode only linkage/table/date metadata. Do not decode
their archived taxon or accepted IDs. No staff IDs, free text or unselected
taxon values are needed. Never narrow protected/higher-rank records through
history, catalog closure or a unique modeled candidate.

The official join/history distinction is described in the
[quick-start guide](https://data.neonscience.org/api/v0/documents/quick-start-guides/NEON.QSG.DP1.10058.001v2?fallback=html&inline=true)
and the [RELEASE-2026 product record](https://data.neonscience.org/data-products/DP1.10058.001/RELEASE-2026).
Current/history handling is provenance, not active taxonomic verification.

## Conservative accepted-concept mapping

Keep existing exact species-name matches. Follow explicit accepted-name chains
with cycle/missing-link checks; a terminal species can add a match only when
all exact species-level candidates agree on one SAGE class. Infraspecific
records can map through an **explicit accepted species concept**, not an
assumed species-parent key or lexical prefix. A1's lexical parent candidates
remain candidates and are not used as positive labels. This resolves only the
supported portion of rank/name harmonization.

If the original or resolved record's declared rank contradicts its dictionary
rank, leave it unknown.
When current named observations and current morphospecies resolutions imply
different accepted concepts, retain uncertainty over compatible classes even
if only one concept happens to match the model. A qualifier can be cleared by
an explicit published later morphospecies resolution, but a nonzero original
data-quality flag cannot disappear through the join. Resolution qualifiers and
quality flags still govern the resolved row.

An unambiguously mapped species outside the frozen native/universe mask is an
identified outside-target record. An unmatched name is **not** certified outside
the target and continues to mask compatible classes conservatively. Higher-rank
and missing IDs retain the existing genus/family/all-class unknown policy.
Compatibility follows every genus/family along the accepted-name chain. If no
genus matches modeled classes, fall back to family and then all classes instead
of treating an unrecognized genus as a certified outside-target identification.
Known confirmed positives remain known despite other uncertain records.

No new class, fuzzy match, guessed synonym, genus-positive constraint or model
parameter is admitted. Genus-level records may represent outside-target taxa;
the T1 simulator's positive-genus assumption is not transported to NEON.

## Replay and interpretation

Freeze source/input/config hashes before retrieval, then the revised label
cache before scoring. Use unchanged prediction ranks/counts and reliability
scores. Report record/key counts, history linkage, event-ID multiplicity,
new accepted-species mappings, conflicts, unresolved ranks and label observability.
History taxon differences are diagnostic only and cannot choose current labels.

Score four explicit cohorts: original E4's fixed 65 cells, the previous fixed
77-cell morphospecies subset, all 163 cells, and the revised >=80%-known subset.
Within a cohort, selectors receive identical 80% site × plot-area-group quotas.
For fixed cohorts, also compare previous and revised known-label scores on the
same sets. This is an estimand/contract change, not measured error against an
unavailable full-label oracle. Quotas can change across cohorts; do not attribute
cohort differences to a model improvement.

Report known-label F1, mean/HGB/affine-SDC magnitude MSE, paired HGB-minus-SDC
utility, 97.5% paired site-bootstrap intervals conditional on these fixed
policies/cohorts, and independent-label completion ranges. Shared accepted
cells cancel in paired identification ranges. The >=100 cells/12 sites E4 rule
is only a historical diagnostic reference; satisfying it after correction
would not restore prospective admission or unconsumed outcomes.

Validate selected-key privacy behavior, concept/rank/conflict rules, old replay
identity, independent confusion scoring, quotas and all historical manifests.
Document supported corrections and unresolved parent/name/observation semantics
without optimizing rules for more cells or favorable F1.
