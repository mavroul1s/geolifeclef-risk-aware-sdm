import numpy as np

from geolifeclef.taxonomy_contract import accepted_concept, parent_candidates, species_catalog_condition, species_name_index


def taxon(code, name, rank, accepted=None, genus='Abies'):
    return {'taxonID': code, 'acceptedTaxonID': accepted or code, 'dwc:scientificName': name,
            'dwc:taxonRank': rank, 'dwc:genus': genus}


def test_accepted_chain_follows_synonyms_without_claiming_species_parent():
    taxa = {'a': taxon('a', 'Abies alba var. a', 'variety', 'b'),
            'b': taxon('b', 'Abies alba var. b', 'variety')}
    row, status = accepted_concept('a', taxa)
    assert row['taxonID'] == 'b' and row['dwc:taxonRank'] == 'variety'
    assert status == 'resolved_accepted_concept'


def test_cycles_and_missing_links_remain_unresolved():
    taxa = {'a': taxon('a', 'A a', 'species', 'b'), 'b': taxon('b', 'B b', 'species', 'a')}
    assert accepted_concept('a', taxa)[1] == 'accepted_name_cycle'
    assert accepted_concept('absent', taxa)[1] == 'missing_link'


def test_explicit_accepted_species_is_distinguished_from_candidate_name_prefix():
    taxa = {'s': taxon('s', 'Abies alba', 'species'),
            'i': taxon('i', 'Abies alba var. a', 'variety', 's'),
            'v': taxon('v', 'Abies alba var. b', 'variety')}
    names = species_name_index(taxa)
    assert parent_candidates(taxa['i'], taxa, names)['status'] == 'explicit_accepted_species_concept'
    assert parent_candidates(taxa['v'], taxa, names) == {
        'status': 'nomenclatural_candidates_require_validation', 'parent_ids': ['s']}


def test_hybrids_or_inconsistent_genus_are_not_lexically_collapsed():
    for name, genus in [('Abies × alba var. a', 'Abies'), ('Abies alba var. a', 'Picea')]:
        row = taxon('v', name, 'variety', genus=genus)
        assert not parent_candidates(row, {'v': row}, {'abies alba': ['s']})['parent_ids']


def test_unknown_catalog_member_prevents_species_screen_without_becoming_absence():
    screen = species_catalog_condition(['a', 'b'], {'a': 0}, np.array([True]))
    assert not screen['necessary_species_catalog_condition']
    assert screen['unmapped_concepts'] == 1 and screen['mapped_outside_eligible'] == 0


def test_native_subset_and_nonempty_catalog_required():
    assert not species_catalog_condition([], {}, [True])['necessary_species_catalog_condition']
    assert not species_catalog_condition(['a'], {'a': 0}, [False])['necessary_species_catalog_condition']
    assert species_catalog_condition(['a'], {'a': 0}, [True])['necessary_species_catalog_condition']
