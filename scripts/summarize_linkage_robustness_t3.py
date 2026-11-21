"""Report all planned T3 prices/modes and linear cost break-even points."""
import numpy as np
import pandas as pd
from run_linkage_robustness_t3 import OUT
from run_taxonomic_resolution_t1 import save


def main():
    curves = pd.read_csv(OUT/'curves.csv'); events = pd.read_csv(OUT/'query_events.csv')
    frontiers = pd.read_csv(OUT/'cost_frontiers.csv')
    curves.groupby(['error_rate', 'mode', 'policy', 'budget_fraction'])[
        ['groups', 'resolutions', 'audits', 'resolved_records', 'width', 'cell_containment_failures',
         'incorrect_known_labels', 'known_fraction', 'affected_countries']].mean().reset_index().to_csv(OUT/'curve_means.csv', index=False)
    safe = frontiers[~frontiers['mode'].eq('unchecked_all')]
    safe.groupby(['error_rate', 'mode', 'policy', 'audit_price'])[['sign_cost', 'tolerance_cost']].agg(
        ['mean', 'median', 'min', 'max', 'count']).to_csv(OUT/'valid_cost_summary.csv')
    rows = []
    for row in events[~events['mode'].eq('unchecked_all')].itertuples():
        r, a, local = row.sign_resolutions, row.sign_audits, row.local_best_practical_cost
        if a > 0: threshold = (local-r)/a; status = 'finite'
        else: threshold = np.nan; status = 'always_cheaper' if r < local else 'never_strictly_cheaper'
        rows.append({'seed': row.seed, 'error_rate': row.error_rate, 'mode': row.mode, 'policy': row.policy,
            'sign_resolutions': r, 'sign_audits': a, 'best_local_cost': local,
            'break_even_audit_price': threshold, 'no_audit_status': status,
            'reference_price_cost': r+.1*a})
    pd.DataFrame(rows).to_csv(OUT/'break_even_local.csv', index=False)
    validity = curves.groupby(['error_rate', 'mode', 'seed', 'policy']).agg(
        any_contrast_failure=('oracle_contained', lambda x: (~x).any()),
        any_wrong_sign=('wrong_practical_sign', 'any'),
        max_incorrect_known_labels=('incorrect_known_labels', 'max'),
        max_cell_containment_failures=('cell_containment_failures', 'max'))
    validity.to_csv(OUT/'validity_by_trajectory.csv')
    save(OUT/'reporting_scope.json', {'all_predeclared_prices_reported': True,
        'unchecked_costs_excluded_from_valid_superiority': True,
        'reference_price_is_not_measured_field_cost': True,
        'prior_T2_primary_and_secondary_roles_unchanged': True,
        'cost_frontier_assumes_fixed_batch_orders': True, 'new_queries_or_fits': 0})


if __name__ == '__main__': main()
