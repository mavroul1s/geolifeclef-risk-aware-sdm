"""Metadata-only taxonomy checks; candidates are not resolved field labels."""
import re

from geolifeclef.taxonomy_bridge import canonical_name

INFRASPECIFIC = {'subspecies', 'variety', 'form', 'subvariety'}


def accepted_concept(code, taxa):
    """Follow accepted-name links, which are not species-parent relations."""
    seen = set()
    while code not in seen:
        seen.add(code)
        row = taxa.get(code)
        if row is None:
            return None, 'missing_link'
        target = row.get('acceptedTaxonID')
        if not target:
            return None, 'missing_accepted_id'
        if target == code:
            return row, 'resolved_accepted_concept'
        code = target
    return None, 'accepted_name_cycle'


def parent_candidates(taxon, taxa, species_by_name):
    """Audit explicit accepted-species links versus lexical parent candidates.

    A name-prefix match is recorded as a candidate only. No infraspecific
    identification is automatically collapsed to a modeled species.
    """
    if taxon.get('dwc:taxonRank') not in INFRASPECIFIC:
        return {'status': 'not_infraspecific', 'parent_ids': []}
    accepted, status = accepted_concept(taxon['taxonID'], taxa)
    if accepted is None:
        return {'status': status, 'parent_ids': []}
    if accepted.get('dwc:taxonRank') == 'species':
        return {'status': 'explicit_accepted_species_concept', 'parent_ids': [accepted['taxonID']]}
    names = set()
    for row in [taxon, accepted]:
        match = re.match(r'^([A-Z][a-z-]+) ([a-z][a-z-]+)(?:\s|$)', row.get('dwc:scientificName') or '')
        if match and match[1].casefold() == (row.get('dwc:genus') or '').casefold():
            names.add((match[1]+' '+match[2]).casefold())
    candidates = set()
    for name in names:
        candidates.update(species_by_name.get(name, []))
    return {'status': 'nomenclatural_candidates_require_validation' if candidates else 'no_unambiguous_parent_evidence',
            'parent_ids': sorted(candidates)}


def species_catalog_condition(members, mapping, eligible):
    """Necessary catalog-relative screen, never a field-presence certificate.

    An unmatched species is unresolved, not a proven outside-universe species.
    Even passing excludes neither unlisted taxa nor unresolved infra-species.
    """
    members = list(members)
    unresolved = sum(code not in mapping for code in members)
    outside = sum(code in mapping and not eligible[mapping[code]] for code in members)
    return {'accepted_species_concepts': len(members), 'unmapped_concepts': unresolved,
            'mapped_outside_eligible': outside,
            'necessary_species_catalog_condition': bool(members and not unresolved and not outside)}


def species_name_index(taxa):
    index = {}
    for row in taxa.values():
        if row.get('dwc:taxonRank') == 'species':
            index.setdefault(canonical_name(row), []).append(row['taxonID'])
    return index
