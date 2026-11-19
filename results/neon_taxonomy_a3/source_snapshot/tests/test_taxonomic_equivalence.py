from geolifeclef.taxonomic_equivalence import accepted_alias_classes,extend_exact_alias_mapping


def row(code,name,rank='species',accepted=None):
    return {'taxonID':code,'acceptedTaxonID':accepted or code,'dwc:scientificName':name,'dwc:taxonRank':rank}


def test_reverse_synonym_link_resolves_current_name_to_one_exact_model_alias():
    taxa={'old':row('old','Oldgenus alba',accepted='new'),'new':row('new','Newgenus alba')}
    groups=accepted_alias_classes(taxa,{'oldgenus alba':[3]})
    mapping,audit=extend_exact_alias_mapping(taxa,{'old':3},groups)
    assert mapping=={'old':3,'new':3} and audit['new']=='unique_explicit_species_alias'


def test_multiple_model_aliases_remain_ambiguous_without_native_mask_selection():
    taxa={'old':row('old','Oldgenus alba',accepted='new'),'new':row('new','Newgenus alba')}
    groups=accepted_alias_classes(taxa,{'oldgenus alba':[1],'newgenus alba':[2]})
    mapping,audit=extend_exact_alias_mapping(taxa,{},groups)
    assert not mapping and audit['new']=='multiple_model_class_aliases'


def test_infraspecific_terminal_is_not_collapsed_to_lexical_parent():
    taxa={'v':row('v','Abies alba var. minor','variety'),'s':row('s','Abies alba')}
    groups=accepted_alias_classes(taxa,{'abies alba':[0]})
    mapping,audit=extend_exact_alias_mapping(taxa,{},groups)
    assert 'v' not in mapping and audit['v']=='non_species_terminal_requires_parent_evidence'


def test_higher_rank_never_resolves_even_if_accepted_link_has_one_species():
    taxa={'g':row('g','Abies','genus','s'),'s':row('s','Abies alba')}
    mapping,audit=extend_exact_alias_mapping(taxa,{},accepted_alias_classes(taxa,{'abies alba':[0]}))
    assert 'g' not in mapping and audit['g']=='higher_rank_not_resolved'


def test_existing_class_name_matches_are_preserved_as_an_extension():
    taxa={'s':row('s','Abies alba')}
    mapping,_=extend_exact_alias_mapping(taxa,{'s':7},{'s':{7,8}})
    assert mapping=={'s':7}


def test_missing_and_cyclic_concepts_do_not_produce_new_aliases():
    taxa={'a':row('a','Abies alba',accepted='b'),'b':row('b','Abies minor',accepted='a')}
    mapping,audit=extend_exact_alias_mapping(taxa,{},accepted_alias_classes(taxa,{'abies alba':[0]}))
    assert not mapping and audit['a']=='accepted_name_cycle'
