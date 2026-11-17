import numpy as np
import pytest

from geolifeclef.neon_assessment import selected_csv_rows,TaxonomicAssessment,score_fixed_sets,site_primary_comparison


def test_unselected_species_fields_are_not_decoded():
    raw=b'plotID,boutNumber,taxonID,scientificName\np1,1,\xff,secret\np2,1,A,"quoted,\nname"\np2,2,\xfe,other\n'
    rows=list(selected_csv_rows(raw,{'p2':'1'},['taxonID','scientificName']))
    assert rows==[{'plotID':'p2','boutNumber':'1','taxonID':'A','scientificName':'quoted,\nname'}]


def test_unknown_genus_masks_compatible_absences_but_preserves_known_presence():
    taxa={'A':{'dwc:genus':'Abies','dwc:family':'Pinaceae'},'U':{'dwc:genus':'Abies','dwc:family':'Pinaceae'}}
    reducer=TaxonomicAssessment(np.ones((1,3),bool),taxa,{'A':0},['Abies alba','Abies balsamea','Quercus robur'])
    reducer.add(0,{'taxonID':'A','targetTaxaPresent':'Y'},'div_10m2Data100m2Data')
    reducer.add(0,{'taxonID':'U','targetTaxaPresent':'Yes'},'div_10m2Data100m2Data')
    y,known,fraction,_=reducer.finish()
    np.testing.assert_array_equal(y,[[1,-1,0]]);assert known.tolist()==[[True,False,True]] and fraction[0]==2/3


def test_unidentified_plant_is_missing_not_all_absences():
    reducer=TaxonomicAssessment(np.ones((1,2),bool),{}, {},['Abies alba','Quercus robur'])
    reducer.add(0,{'taxonID':'UNKNOWN','targetTaxaPresent':'Yes'},'div_10m2Data100m2Data')
    y,known,_,_=reducer.finish();assert (y==-1).all() and not known.any()


def test_partial_scoring_keeps_frozen_set_and_removes_only_unknown_denominator():
    rank=np.array([[0,1,2]]);count=np.array([2]);original=rank.copy()
    result=score_fixed_sets(np.array([[1,-1,0]]),np.array([[True,False,True]]),rank,count)
    assert result['F1'][0]==1 and result['unknown_prediction_count'][0]==1
    np.testing.assert_array_equal(rank,original);assert count[0]==2


def test_site_bootstrap_preserves_fixed_cluster_pairs():
    q=np.array([.2,.4,.6,.8]);p=np.array([.3,.3,.7,.7]);a=np.array([1,0,1,0],bool);b=~a
    result=site_primary_comparison(q,p,.5,a,b,['A','A','B','B'],repetitions=50)
    assert result['sites']==2 and result['valid_draws']==50
    assert np.isclose(result['HGB_minus_SDC_selective_F1'],-.2)
    assert np.isclose(result['MSE_gain_mean_minus_HGB'],.04)
