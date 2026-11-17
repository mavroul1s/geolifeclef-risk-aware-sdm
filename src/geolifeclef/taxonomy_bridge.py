"""Conservative dictionary-only mapping to public, non-anonymized SAGE classes."""
import unicodedata


def normalized_name(name):
    return ' '.join(unicodedata.normalize('NFKC', str(name)).split()).casefold()


def canonical_name(taxon):
    name = str(taxon.get('dwc:scientificName') or '').strip()
    author = str(taxon.get('dwc:scientificNameAuthorship') or '').strip()
    # Remove only the separate, explicitly supplied authorship suffix.
    if author and name.endswith(' ' + author):
        name = name[:-(len(author) + 1)].rstrip()
    return normalized_name(name)


def map_taxon(taxon, by_id, sage_names):
    """No fuzzy names, guessed synonyms, or automatic genus/subspecies collapse."""
    if taxon.get('dwc:taxonRank') != 'species':
        return {'sage_index': None, 'status': 'unmapped_rank'}
    candidates = {}
    direct = sage_names.get(canonical_name(taxon), [])
    for index in direct:
        candidates[index] = 'exact_name_after_declared_authorship_removal'
    accepted_id = taxon.get('acceptedTaxonID')
    accepted = by_id.get(accepted_id)
    if accepted and accepted_id != taxon.get('taxonID') and accepted.get('dwc:taxonRank') == 'species':
        for index in sage_names.get(canonical_name(accepted), []):
            candidates.setdefault(index, 'exact_NEON_accepted_name')
    if len(candidates) == 1:
        index, status = next(iter(candidates.items()))
        return {'sage_index': index, 'status': status}
    return {'sage_index': None, 'status': 'ambiguous' if candidates else 'unmapped_name'}
