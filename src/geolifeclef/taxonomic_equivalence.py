"""Exact accepted-species equivalence; no fuzzy names or rank-parent guesses."""
from collections import defaultdict

from geolifeclef.taxonomy_bridge import canonical_name
from geolifeclef.taxonomy_contract import accepted_concept
from geolifeclef.neon_record_contract import NAMED_RANKS


def accepted_alias_classes(taxa, sage_names):
    groups=defaultdict(set)
    for code,row in taxa.items():
        if row.get('dwc:taxonRank')!='species':
            continue
        terminal,_=accepted_concept(code,taxa)
        if terminal is not None and terminal.get('dwc:taxonRank')=='species':
            groups[terminal['taxonID']].update(sage_names.get(canonical_name(row),[]))
    return dict(groups)


def extend_exact_alias_mapping(taxa, existing, groups):
    """Extend only previously unmatched named records; uniqueness is global.

    Existing exact matches retain their original class-name target semantics.
    New aliases require one class over the full model vocabulary, before native
    or universe filtering. A same-concept group with multiple model labels does
    not establish a new unique mapping. No higher-rank identity is inferred.
    """
    mapping=dict(existing);audit={}
    for code,row in taxa.items():
        if code in existing:
            audit[code]='existing_exact_mapping';continue
        if row.get('dwc:taxonRank') not in NAMED_RANKS:
            audit[code]='higher_rank_not_resolved';continue
        terminal,status=accepted_concept(code,taxa)
        if terminal is None:
            audit[code]=status;continue
        if terminal.get('dwc:taxonRank')!='species':
            audit[code]='non_species_terminal_requires_parent_evidence';continue
        candidates=groups.get(terminal['taxonID'],set())
        if len(candidates)==1:
            mapping[code]=next(iter(candidates));audit[code]='unique_explicit_species_alias'
        else:
            audit[code]='multiple_model_class_aliases' if candidates else 'no_exact_model_alias'
    return mapping,audit
