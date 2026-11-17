"""Plot saved exploratory pilot results without refitting any model."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/research_audit_2026_11_13'


def main():
    results = pd.read_csv(OUT/'selective_pilot.csv')
    coverage = pd.read_csv(OUT/'selective_country_coverage.csv')
    target = pd.read_csv(OUT/'country_support.csv').sort_values('test_surveys', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    methods = {'random': 'Random', 'distance_only': 'Distance only',
        'area_only': 'Plot area only', 'metadata_distance': 'Metadata + distance',
        'metadata_distance_country_balanced': 'Metadata + distance, country quotas'}
    for method, label in methods.items():
        part = results[(results.method == method) & (results.group == 'all')].sort_values('overall_retained_fraction')
        axes[0].plot(part.overall_retained_fraction, part.mean_sample_F1, 'o-', label=label)
    axes[0].set(xlabel='Fraction of surveys retained', ylabel='F1 on retained surveys',
                title='Reliability and coverage tradeoff')
    axes[0].legend(fontsize=8)
    names = target.country.head(8).tolist()
    # Show only countries represented in the existing assessment; missing support
    # is reported explicitly in the audit, not drawn as zero model performance.
    names = [n for n in names if n in set(coverage.country)]
    x = np.arange(len(names))
    for offset, method, label in [(-.19, 'metadata_distance', 'Unconstrained'),
        (.19, 'metadata_distance_country_balanced', 'Country quotas')]:
        c = coverage[(coverage.method == method) & (coverage.overall_retained_fraction == .8)].set_index('country').reindex(names)
        axes[1].bar(x+offset, c.country_retained_fraction, .36, label=label)
    axes[1].axhline(.8, color='black', linestyle='--', linewidth=.8)
    axes[1].set(xticks=x, xticklabels=names, ylim=(0, 1.05), ylabel='Country fraction retained',
                title='Country coverage at 80% overall retention')
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].legend(fontsize=8)
    fig.suptitle('Exploratory results on reused spatial development data', fontsize=11)
    for extension in ['png', 'svg']:
        fig.savefig(OUT/f'selective_pilot.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
