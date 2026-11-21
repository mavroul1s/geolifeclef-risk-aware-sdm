"""Presentation-only T3 figure layout; original frozen renderer preserved."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'results/linkage_robustness_t3'


def main():
    curves = pd.read_csv(OUT/'curves.csv'); costs = pd.read_csv(OUT/'cost_frontiers.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rates = sorted(curves.error_rate.unique()); colors = ['#1b8495', '#c87d24', '#8757b3']
    for i, (mode, label, color) in enumerate([('unchecked_all', 'Unchecked propagation', '#bd514c'),
        ('audit_all', 'Audit all members', '#1b8495'), ('audit_support', 'Audit comparison members', '#8757b3')]):
        values = curves[curves['mode'].eq(mode)].groupby(['error_rate', 'seed', 'policy']).oracle_contained.all()
        mean = (~values).groupby('error_rate').mean()*100
        axes[0].bar(np.arange(len(rates))+(i-1)*.24, mean.reindex(rates), width=.23, color=color, label=label)
    axes[0].set_xticks(np.arange(len(rates)), [f'{100*r:g}%' for r in rates]); axes[0].set_ylim(0, 105)
    axes[0].set_xlabel('Misassigned hidden record memberships')
    axes[0].set_ylabel('Trajectories with any invalid contrast range (%)')
    axes[0].set_title('Link assumptions and range validity')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, fontsize=8, frameon=False)
    for rate, color in zip(rates, colors):
        for mode, style, label in [('audit_all', '--', 'all'), ('audit_support', '-', 'comparison')]:
            take = costs[(costs.error_rate == rate) & costs['mode'].eq(mode) & costs.policy.eq('paired_width')].copy()
            take['relative_cost'] = take.sign_cost/take.local_best_practical_cost
            mean = take.groupby('audit_price').relative_cost.median()
            axes[1].plot(mean.index, mean.values, color=color, linestyle=style, marker='o', markersize=3,
                label=f'{100*rate:g}% error; audit {label}')
    axes[1].axhline(1, color='#666666', linewidth=1, label='Best saved local method')
    axes[1].set_xlabel('Audit cost / full identity-resolution cost')
    axes[1].set_ylabel('Median cost ratio to best local acquisition')
    axes[1].set_ylim(bottom=0); axes[1].set_title('Cost to identify a practical ordering')
    axes[1].legend(fontsize=7, frameon=False); axes[1].grid(alpha=.2)
    fig.suptitle('Charged verification under synthetic linkage errors', fontsize=14)
    fig.text(.5, .015, 'Ten saved masks; fixed predictions. Perfect binary audit oracle assumed. '
        'Costs and error rates are sensitivity scenarios, not field measurements.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .21, 1, .93))
    for suffix in ['png', 'pdf']: fig.savefig(OUT/f'linkage_cost_t3_final.{suffix}', dpi=180, bbox_inches='tight')


if __name__ == '__main__': main()
