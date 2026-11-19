"""Descriptive tables for the predeclared T2 outcomes; no outcome selection."""
import json
import numpy as np
import pandas as pd
from run_taxonomic_verification_t2 import ROOT, OUT
from run_taxonomic_resolution_t1 import save


def main():
    config = json.loads((ROOT/'configs/taxonomic_verification_t2.json').read_text())
    curves = pd.read_csv(OUT/'curves.csv'); events = pd.read_csv(OUT/'query_events.csv')
    cols = ['queries', 'queried_records', 'outside_target_queries', 'affected_cells',
            'queried_countries', 'width', 'mean_cell_width', 'known_fraction', 'known_rare_positives']
    curves.groupby(['unit', 'policy', 'budget_fraction'])[cols].mean().reset_index().to_csv(OUT/'curve_means.csv', index=False)
    events.groupby(['unit', 'policy'])[['first_width_queries', 'first_practical_sign_queries', 'available_keys']].agg(
        ['median', 'min', 'max', 'count']).to_csv(OUT/'stopping_costs.csv')
    rows = []; rng = np.random.default_rng(config['bootstrap']['seed'])
    draws = rng.integers(10, size=(config['bootstrap']['repetitions'], 10)); tail = (1-config['bootstrap']['confidence'])/2
    for unit in config['query_units']:
        for endpoint in ['first_width_queries', 'first_practical_sign_queries']:
            table = events[events.unit.eq(unit)].pivot(index='seed', columns='policy', values=endpoint)
            for control in config['primary_controls']:
                diff = table[control].to_numpy()-table.paired_width.to_numpy()
                median_control, median_candidate = table[control].median(), table.paired_width.median()
                ci = np.quantile(diff[draws].mean(1), [tail, 1-tail])
                rows.append({'unit': unit, 'endpoint': endpoint, 'control': control,
                    'median_candidate_queries': median_candidate, 'median_control_queries': median_control,
                    'ratio_of_median_cost_reduction': 1-median_candidate/median_control,
                    'mean_paired_query_saving': diff.mean(), 'better_seeds': int((diff > 0).sum()),
                    'descriptive_mask_interval_low': ci[0], 'descriptive_mask_interval_high': ci[1],
                    'secondary_endpoint': True})
    pd.DataFrame(rows).to_csv(OUT/'secondary_cost_comparisons.csv', index=False)
    save(OUT/'reporting_scope.json', {'scope': 'all predeclared endpoints and strong controls; descriptive reporting',
        'primary_criterion_unchanged': True, 'secondary_cost_endpoint_promoted_to_primary': False,
        'spatial_or_population_inference': False, 'new_queries_or_fits': 0})


if __name__ == '__main__': main()
