"""Descriptive T4 control/ablation tables; no adaptive outcome selection."""
import pandas as pd
from run_adaptive_baselines_t4 import ROOT, OUT
from run_taxonomic_resolution_t1 import save


def main():
    events = pd.read_csv(OUT/'query_events.csv')
    summary = events.groupby(['error_rate', 'audit_price', 'policy'])[
        ['sign_cost', 'tolerance_cost', 'sign_groups', 'sign_resolutions', 'sign_audits', 'executed_groups']].agg(
        ['mean', 'median', 'min', 'max'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.reset_index().to_csv(OUT/'cost_summary.csv', index=False)
    rows = []
    for (error, price), frame in events.groupby(['error_rate', 'audit_price']):
        table = frame.pivot(index='seed', columns='policy', values='sign_cost')
        for candidate in table.columns:
            for control in table.columns:
                if candidate == control: continue
                delta = table[control]-table[candidate]
                rows.append({'error_rate': error, 'audit_price': price, 'candidate': candidate, 'control': control,
                    'mean_cost_saving': delta.mean(), 'relative_mean_reduction': delta.mean()/table[control].mean(),
                    'better_masks': int((delta > 1e-10).sum()), 'descriptive_only': True})
    pd.DataFrame(rows).to_csv(OUT/'all_control_comparisons.csv', index=False)
    old = pd.read_csv(ROOT/'results/linkage_robustness_t3/query_events.csv')
    old = old[old['mode'].eq('audit_support') & old.policy.eq('paired_width')]
    numeric = []
    for row in events[events.policy.eq('static_scope_width')].itertuples():
        prior = old[(old.seed == row.seed) & (old.error_rate == row.error_rate)].iloc[0]
        old_cost = prior.sign_resolutions+row.audit_price*prior.sign_audits
        numeric.append({'seed': row.seed, 'error_rate': row.error_rate, 'audit_price': row.audit_price,
            'canonical_static_cost': row.sign_cost, 'frozen_T3_paired_cost': old_cost,
            'cost_difference': row.sign_cost-old_cost, 'role': 'numeric/tie audit, not a new-method comparison'})
    pd.DataFrame(numeric).to_csv(OUT/'frozen_T3_numeric_comparison.csv', index=False)
    save(OUT/'reporting_scope.json', {'static_paired_novelty_branch_closed_by_equal_weight_identity': True,
        'generic_adaptive_controls_not_claimed_as_new_methods': True, 'all_control_arms_reported': True,
        'no_VMA_or_LURE_reproduction_claim': True, 'new_queries_or_fits': 0,
        'prices_and_errors_are_sensitivity_scenarios': True})


if __name__ == '__main__': main()
