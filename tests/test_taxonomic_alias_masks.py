import numpy as np

from geolifeclef.taxonomic_alias_masks import AliasSafeAssessment
from geolifeclef.neon_record_contract import AcceptedConceptAssessment


def taxa():
    return {'old':{'taxonID':'old','acceptedTaxonID':'new','dwc:taxonRank':'species','dwc:scientificName':'Oldgenus alba','dwc:genus':'Oldgenus'},
            'new':{'taxonID':'new','acceptedTaxonID':'new','dwc:taxonRank':'species','dwc:scientificName':'Newgenus alba','dwc:genus':'Newgenus'},
            'g':{'taxonID':'g','acceptedTaxonID':'g','dwc:taxonRank':'genus','dwc:scientificName':'Newgenus','dwc:genus':'Newgenus'}}


def test_qualified_cross_genus_alias_cannot_become_confident_absence():
    t=taxa();e=np.ones((1,2),bool);names=['Oldgenus alba','Newgenus minor'];mapping={'old':0,'new':0}
    base=AcceptedConceptAssessment(e,t,mapping,names);new=AliasSafeAssessment(e,t,mapping,names)
    assert base.compatible(t['new']).tolist()==[1] and new.compatible(t['new']).tolist()==[0,1]
    new.add(0,{'taxonID':'new','targetTaxaPresent':'Y','identificationQualifier':'cf.'},'div_10m2Data100m2Data')
    _,known,_,_=new.finish();assert not known.any()


def test_higher_rank_genus_adds_possible_aliases_without_identifying_a_species():
    t=taxa();new=AliasSafeAssessment(np.ones((1,2),bool),t,{'old':0,'new':0},['Oldgenus alba','Newgenus minor'])
    new.add(0,{'taxonID':'g','targetTaxaPresent':'Y'},'div_10m2Data100m2Data')
    truth,known,_,_=new.finish();assert not known.any() and not (truth>0).any()


def test_alias_overlay_never_removes_existing_uncertainty():
    t=taxa();e=np.ones((1,3),bool);names=['Oldgenus alba','Newgenus minor','Othergenus b'];m={'old':0,'new':0}
    old=AcceptedConceptAssessment(e,t,m,names);new=AliasSafeAssessment(e,t,m,names)
    for row in [*t.values(),None]:assert set(old.compatible(row))<=set(new.compatible(row))


def test_confirmed_positive_stays_known_under_overlapping_uncertain_aliases():
    t=taxa();new=AliasSafeAssessment(np.ones((1,2),bool),t,{'old':0,'new':0},['Oldgenus alba','Newgenus minor'])
    new.add(0,{'taxonID':'new','targetTaxaPresent':'Y'},'div_10m2Data100m2Data')
    new.add(0,{'taxonID':'g','targetTaxaPresent':'Y'},'div_10m2Data100m2Data')
    truth,known,_,_=new.finish();assert truth[0,0]==1 and known[0,0] and not known[0,1]
