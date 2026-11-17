import csv
import io

import pytest

from geolifeclef.neon_metadata import project_csv_bytes, metadata_csv_bytes
from geolifeclef.neon_survey import EXPECTED_SUBPLOTS, full_plot_completeness


def test_projection_discards_undecodable_outcomes_and_richness_multiplicity():
    payload = b'plotID,taxonID,remarks,endDate\r\np1,\xff,"secret, quoted",2023-07-01\r\np1,\xfe,"other\nsecret",2023-07-01\r\n'
    columns, rows = project_csv_bytes(payload, ['plotID', 'endDate'])
    assert columns == ['plotID', 'endDate']
    assert rows == [('p1', '2023-07-01')]
    assert metadata_csv_bytes(columns, rows) == b'plotID,endDate\r\np1,2023-07-01\r\n'


def test_projection_matches_standard_csv_for_selected_fields():
    payload = b'plotID,discard,flag\n"p,1","a ""quote""\nand newline","OK ""yes"""\np2,a,\n'
    columns, rows = project_csv_bytes(payload, ['plotID', 'flag'])
    expected = sorted({(r['plotID'], r['flag']) for r in csv.DictReader(io.StringIO(payload.decode()))})
    assert columns == ['plotID', 'flag'] and rows == expected


@pytest.mark.parametrize('payload', [b'plotID,taxonID\np1\n', b'plotID,plotID\na,b\n', b'plotID,taxonID\n"p1,x'])
def test_projection_rejects_malformed_input(payload):
    with pytest.raises(ValueError):
        project_csv_bytes(payload, ['plotID'])


def test_complete_nested_inventory_requires_every_unit_and_quality_flag():
    rows = [{'subplotID': name, 'table': 'div_1m2Data' if scale == '1m2' else 'div_10m2Data100m2Data',
             'samplingImpractical': 'OK', 'biophysicalCriteria': 'OK - no known exceptions'}
            for scale, names in EXPECTED_SUBPLOTS.items() for name in sorted(names)]
    assert full_plot_completeness(rows)['full_plot_eligible']
    assert not full_plot_completeness(rows[:-1])['full_plot_eligible']
    rows[-1]['samplingImpractical'] = 'logistical'
    result = full_plot_completeness(rows)
    assert result['nested_units_present'] and not result['full_plot_eligible']
