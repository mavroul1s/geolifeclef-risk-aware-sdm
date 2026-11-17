from geolifeclef.neon_morphospecies import selected_morpho_rows,latest_resolution


def test_unselected_morphospecies_taxa_are_not_decoded():
    body=b'siteID,morphospeciesID,taxonID\nA,x,\xff\nA,y,KNOWN\n'
    assert list(selected_morpho_rows(body,{('A','y')}))==[{'siteID':'A','morphospeciesID':'y','taxonID':'KNOWN'}]


def test_latest_resolution_cannot_choose_an_older_more_favorable_taxon():
    rows=[{'morphospeciesResolvedDate':'2020-01-01','taxonID':'OLD'},
          {'morphospeciesResolvedDate':'2025-01-01','taxonID':'NEW','identificationQualifier':'cf. genus'}]
    chosen,status=latest_resolution(rows)
    assert status=='dated_resolution' and chosen['taxonID']=='NEW' and chosen['identificationQualifier']=='cf. genus'


def test_equal_date_conflicting_identifications_stay_unresolved():
    rows=[{'morphospeciesResolvedDate':'2025-01-01','taxonID':v} for v in ['A','B']]
    assert latest_resolution(rows)==(None,'conflicting_latest_resolution')
