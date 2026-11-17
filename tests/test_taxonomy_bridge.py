from geolifeclef.taxonomy_bridge import map_taxon, normalized_name


def taxon(code, name, author='', accepted=None, rank='species'):
    return {'taxonID': code, 'acceptedTaxonID': accepted or code, 'dwc:scientificName': name,
            'dwc:scientificNameAuthorship': author, 'dwc:taxonRank': rank}


def test_declared_authorship_and_unicode_whitespace_only():
    row = taxon('A', 'Abies\u00a0alba Mill.', 'Mill.')
    result = map_taxon(row, {'A': row}, {normalized_name('Abies alba'): [12]})
    assert result['sage_index'] == 12
    row['dwc:scientificNameAuthorship'] = ''
    assert map_taxon(row, {'A': row}, {'abies alba': [12]})['sage_index'] is None


def test_explicit_accepted_identifier_and_ambiguous_names():
    old, accepted = taxon('OLD', 'Old name', accepted='NEW'), taxon('NEW', 'Accepted name')
    assert map_taxon(old, {'OLD': old, 'NEW': accepted}, {'accepted name': [4]})['sage_index'] == 4
    result = map_taxon(old, {'NEW': accepted}, {'old name': [2], 'accepted name': [4]})
    assert result == {'sage_index': None, 'status': 'ambiguous'}


def test_unknown_and_infraspecific_taxa_never_become_species_absences():
    for rank in ['genus', 'family', 'subspecies', 'variety', 'speciesGroup']:
        row = taxon('U', 'Abies alba', rank=rank)
        assert map_taxon(row, {}, {'abies alba': [1]}) == {'sage_index': None, 'status': 'unmapped_rank'}
