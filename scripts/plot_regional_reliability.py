"""Render saved D3 results without refitting or changing any decision."""
from pathlib import Path
import pandas as pd


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = Path(__file__).resolve().parents[1]/'results/reliability_followup_d1_d3/regional'
    results = pd.read_csv(p/'risk_coverage.csv').query('coverage == .8').set_index('method')
    countries = pd.read_csv(p/'country_results.csv')
    pivot = countries.pivot(index='country', columns='method', values='F1')
    counts = countries[countries.method == 'global_lean'].set_index('country').surveys
    order = ['mean_set_confidence', 'global_lean', 'country_balanced_lean', 'country_blend', 'bioregion_blend']
    labels = ['Mean confidence', 'Pooled lean', 'Country weighted', 'Country blend', 'Bioregion blend']
    selected = results.loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    axes[0].plot(selected.F1, labels, 'o')
    axes[0].set(xlim=(.36, .38), xlabel='Retained-survey F1', title='Matched 80% retention')
    axes[1].plot(selected.rare_occurrence_coverage, labels, 'o')
    axes[1].axvline(.8, color='black', linestyle='--', linewidth=.8)
    axes[1].set(xlim=(.7, .81), xlabel='Fraction of rare occurrences retained', title='Rare taxa: training prevalence 1–25')
    delta = (pivot.country_balanced_lean-pivot.global_lean)[counts >= 30].sort_values()
    axes[2].barh(delta.index, delta)
    axes[2].axvline(-.005, color='red', linestyle='--', linewidth=.8)
    axes[2].axvline(0, color='black', linewidth=.8)
    axes[2].set(xlabel='F1 change versus pooled lean', title='Calibration-selected country weighting')
    fig.suptitle('D3 exploratory diagnostic — reused development data, no neural training')
    for suffix in ['png', 'svg']:
        fig.savefig(p/f'regional_diagnostic.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
