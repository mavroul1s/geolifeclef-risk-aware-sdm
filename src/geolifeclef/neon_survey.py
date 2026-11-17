"""Metadata-only completeness rules for the 2023 NEON nested survey design."""

EXPECTED_SUBPLOTS = {
    '1m2': {'31_1_1', '32_1_2', '32_1_4', '40_1_1', '40_1_3', '41_1_4'},
    '10m2': {'31_10_1', '32_10_2', '32_10_4', '40_10_1', '40_10_3', '41_10_4'},
    '100m2': {'31_100', '32_100', '40_100', '41_100'},
}


def full_plot_completeness(records):
    """Require all nested units and acceptable flags; the union area is 400 m²."""
    seen = {scale: set() for scale in EXPECTED_SUBPLOTS}
    quality = True
    for row in records:
        subplot = row['subplotID']
        for scale, expected in EXPECTED_SUBPLOTS.items():
            table = 'div_1m2Data' if scale == '1m2' else 'div_10m2Data100m2Data'
            if subplot in expected and row['table'] == table:
                seen[scale].add(subplot)
                quality &= (row['samplingImpractical'] == 'OK'
                            and row['biophysicalCriteria'] == 'OK - no known exceptions')
    complete = all(seen[scale] == expected for scale, expected in EXPECTED_SUBPLOTS.items())
    return {'nested_units_present': complete, 'required_unit_quality_OK': bool(quality),
            'full_plot_eligible': bool(complete and quality),
            **{f'n_{scale}_subplots': len(units) for scale, units in seen.items()}}
