# A1: NEON taxonomy and observation-contract audit

17 November 2026. **Metadata audit completed; field label contract remains
partial.** No new composition values or F1 scores are read, no model is fitted,
and no E4 labels, predictions, thresholds or results are changed.

## Question and scope

T1's post-hoc bounds assume that each censored genus contains a positive
**eligible modeled species**. A real unresolved genus record does not establish
that fact automatically. It may refer to an unmodeled or outside-native-subset
species, an unmatched name, a protected identification or an unresolved
infraspecific concept. A1 screens the dictionary and documents the information
needed before any field application.

Inputs are the previously downloaded NEON plant taxonomy dictionary, original
species-name mapping, fixed E4 native/class masks and product metadata. All
**94 dictionary-page hashes** are verified before reading their **93,698 entries**.
The screen uses the 163 existing cells' eligible masks, not their outcomes.

## Completed checks

- All 93,698 accepted-name chains terminate without missing links or cycles.
  `acceptedTaxonID` identifies the accepted concept; it is **not a species-parent
  field**. A link from an infraspecific name can terminate at another
  infraspecific concept or at species rank.
- Among 24,940 infraspecific dictionary entries, **9,047** explicitly terminate
  at an accepted species concept. Another **15,853** have lexical binomial-parent
  candidates requiring validation; **40** lack usable evidence under the
  conservative parser. These are dictionary counts, not observed-record counts
  or new class mappings. Names and accepted concepts can disagree in rank or
  nomenclature; no automatic prefix/subspecies collapse is applied.
- **44,721 cell–genus pairs** contain eligible modeled classes. Only **4,995
  (11.17%)** pass the necessary species-only catalog screen: a nonempty list of
  accepted catalog species, all uniquely mapped to that cell's eligible set.
- **36,839** pairs contain at least one unmapped accepted catalog species;
  **24,653** contain mapped species outside the eligible set; **30,263** have
  accepted infraspecific concepts requiring a separate contract. Reasons
  overlap. An unmatched name remains unresolved, rather than becoming an
  asserted outside-universe absence.

Passing the species-only screen still does not establish catalog completeness,
infraspecific coverage, actual field absence, or that a genus-level record
belongs to a modeled eligible species. **Zero field-positive-genus guarantees
are established in A1.** Failing this screen is not proof that a particular
NEON observation is unusable; this is a conservative metadata condition, not an
observation-weighted or botanical range validation.

## Identification-history and morphospecies semantics

The cached RELEASE-2026 readme and variable dictionary describe current published
determinations and an `identificationHistoryID` linking archived former
determinations. The archive is not an additional independent occurrence list or
a pool from which to choose favorable labels. The published product documentation
describes the same current/history distinction from RELEASE-2024 onward.
[NEON product documentation](https://data.neonscience.org/data-products/DP1.10058.001/RELEASE-2024)

The documented morphospecies join uses site and morphospecies identities.
The already preserved E4 post-hoc replay applies a latest dated resolution with
conflicts left unresolved. A1 does not retune or repeat that ecological scoring.
The next audit must inspect only the consumed plots' selected keys, reconcile
current records with applicable history, and keep quality/identification
qualifiers and protected taxa explicit. A later label-contract revision must
record every rule and remain development on consumed outcomes.

## Validation and next work

Twenty-nine focused taxonomy/morphospecies/censoring tests pass. The new cases
cover accepted-name cycles/missing links, accepted species versus lexical parent
candidates, hybrid/genus mismatch rejection and eligible-catalog screens.
The complete taxonomy dictionary input is hash checked. Larger release validation
is recorded separately; this report does not claim that a fresh field
confirmation or all taxonomic harmonization is complete.

Next audit selected identification-history keys, validate explicit accepted
species mappings before considering lexical parent candidates, and resolve
named concepts against documented authorities. Protect unresolved/obfuscated
identities. Only then define a corrected observation contract and determine
whether a realistic query-cost/verification study is justified. No new neural
or GPU sweep is needed for this work.

Reproduce with `python scripts/audit_neon_taxonomy_contract.py` after restoring
the documented local inputs. Detailed dictionary-only parent candidates remain
in ignored `data/neon_taxonomy_a1/`; compact metadata summaries and the
cell–genus screen are in `results/neon_taxonomy_a1/`. Completed execution refuses
to overwrite its report.
