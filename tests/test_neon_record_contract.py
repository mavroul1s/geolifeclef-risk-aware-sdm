import numpy as np

from geolifeclef.neon_record_contract import (AcceptedConceptAssessment, selected_key_rows, accepted_species_mapping,
    combine_current_record, current_resolution, history_table_matches, rank_matches)


def taxon(code, name, rank='species', accepted=None):
    return {'taxonID': code, 'acceptedTaxonID': accepted or code,
            'dwc:scientificName': name, 'dwc:taxonRank': rank, 'dwc:genus': name.split()[0]}


def test_unselected_history_and_guarded_taxa_never_decode():
    body = b'siteID,identificationHistoryID,identifiedDate,taxonID\nA,unselected,2020,\xff\nA,guarded,2021,\xff\nA,safe,2022,SPECIES\n'
    rows = list(selected_key_rows(body, {('A','guarded'), ('A','safe')}, ['siteID','identificationHistoryID'],
        ['identifiedDate'], {('A','safe')}, ['taxonID']))
    assert len(rows) == 2 and 'taxonID' not in rows[0] and rows[1]['taxonID'] == 'SPECIES'


def test_history_site_identity_is_a_selection_guard():
    body = b'siteID,identificationHistoryID,taxonID\nOTHER,same,\xff\nSELECTED,same,X\n'
    rows = list(selected_key_rows(body, {('SELECTED','same')}, ['siteID','identificationHistoryID'], ['taxonID']))
    assert len(rows) == 1 and rows[0]['taxonID'] == 'X'


def test_explicit_accepted_species_maps_but_lexical_parent_does_not():
    taxa = {'s': taxon('s','Abies alba'), 'i': taxon('i','Abies alba var. x','variety','s'),
            'v': taxon('v','Abies alba var. y','variety')}
    assert accepted_species_mapping(taxa['i'], taxa, {'abies alba':[0]})['sage_index'] == 0
    assert accepted_species_mapping(taxa['v'], taxa, {'abies alba':[0]})['sage_index'] is None


def test_direct_and_accepted_species_conflicts_cannot_choose_model_coverage():
    taxa = {'a':taxon('a','Abies alba',accepted='b'), 'b':taxon('b','Abies balsamea')}
    result = accepted_species_mapping(taxa['a'], taxa, {'abies alba':[0], 'abies balsamea':[1]})
    assert result['sage_index'] is None and result['status'] == 'conflicting_species_candidates'


def test_higher_rank_does_not_resolve_from_unique_species_candidate():
    t = taxon('g','Abies','genus')
    assert accepted_species_mapping(t, {'g':t}, {'abies alba':[0]})['sage_index'] is None


def test_current_named_concept_conflict_remains_unknown_even_if_only_one_maps():
    taxa = {'a':taxon('a','Abies alba'), 'b':taxon('b','Abies balsamea')}
    old, new = {'taxonID':'a','taxonRank':'species'}, {'taxonID':'b','taxonRank':'species'}
    _, reason, evidence = combine_current_record(old, new, 'dated_resolution', [], taxa)
    assert reason == 'conflicting_current_named_concepts' and evidence == [old,new]


def test_published_current_resolution_can_refine_genus_but_cannot_erase_bad_quality():
    taxa = {'g':taxon('g','Abies','genus'), 's':taxon('s','Abies alba')}
    old = {'taxonID':'g','taxonRank':'genus','dataQF':'1','identificationQualifier':'cf.'}
    new = {'taxonID':'s','taxonRank':'species','dataQF':'','identificationQualifier':''}
    chosen, reason, evidence = combine_current_record(old,new,'dated_resolution',[],taxa)
    assert chosen['taxonID'] == 's' and chosen['dataQF'] == '1' and not evidence
    assert reason == 'published_current_resolution'


def test_equal_date_conflicts_do_not_restore_an_older_favorable_resolution():
    rows = [{'morphospeciesResolvedDate':'2020-01-01','taxonID':'old'},
            {'morphospeciesResolvedDate':'2025-01-01','taxonID':'a'},
            {'morphospeciesResolvedDate':'2025-01-01','taxonID':'b'}]
    chosen,status,conflicts = current_resolution(rows)
    assert chosen is None and status == 'conflicting_latest_resolution'
    assert {r['taxonID'] for r in conflicts} == {'a','b'}


def test_rank_and_history_table_guards():
    taxa={'a':taxon('a','Abies alba')}
    assert not rank_matches({'taxonID':'a','taxonRank':'genus'},taxa)
    assert rank_matches({'taxonID':'a','taxonRank':'species'},taxa)
    assert history_table_matches({'identificationTable':'div_1m2Data_pub; div_10m2Data100m2Data'}, 'div_1m2Data')
    assert not history_table_matches({'identificationTable':'div_morphospecies'}, 'div_1m2Data')


def test_uncertainty_includes_full_accepted_chain_genus():
    taxa={'a':taxon('a','Oldgenus alba','variety','b'),
          'b':taxon('b','Othergenus alba','variety','c'),
          'c':taxon('c','Newgenus alba','variety')}
    reducer=AcceptedConceptAssessment(np.ones((1,2),bool),taxa,{},['Newgenus alba','Abies alba'])
    assert reducer.compatible(taxa['a']).tolist()==[0]


def test_unrecognized_genus_falls_back_to_family_or_all_instead_of_empty_mask():
    taxa={'a':{**taxon('a','Abies alba'),'dwc:family':'Pinaceae'},
          'unknown':{**taxon('unknown','Novelgenus alba'),'dwc:family':'Pinaceae'}}
    reducer=AcceptedConceptAssessment(np.ones((1,2),bool),taxa,{'a':0},['Abies alba','Othergenus b'])
    assert reducer.compatible(taxa['unknown']).tolist()==[0,1]
    assert reducer.compatible(taxon('x','Novelgenus alba')).tolist()==[0,1]
