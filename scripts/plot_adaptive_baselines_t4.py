"""Scientific T4 plot of equally scoped generic baseline costs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'results/adaptive_baselines_t4'
LABELS = {'static_scope_width': 'Static width', 'static_scope_width_per_cost': 'Static width / cost',
    'adaptive_scope_width': 'Adaptive width', 'adaptive_scope_width_per_cost_fixed': 'Adaptive / fixed cost forecast',
    'adaptive_scope_width_per_cost': 'Adaptive / learned cost forecast', 'adaptive_incidence_per_cost': 'Adaptive incidence / cost'}


def main():
    events = pd.read_csv(OUT/'query_events.csv'); fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    colors = ['#999999', '#567a99', '#25956e', '#d99a40', '#8057aa', '#bd5a50']
    for ax, price in zip(axes, [.1, .5]):
        for j, ((policy, label), color) in enumerate(zip(LABELS.items(), colors)):
            data = events[(events.audit_price == price) & events.policy.eq(policy)].groupby('error_rate').sign_cost.mean()
            ax.plot(data.index*100, data.values, color=color, marker='o', linewidth=2 if j == 4 else 1.5, label=label)
        ax.set_title(f'Audit price {price:g} per full identification'); ax.set_xlabel('Membership-error scenario (%)')
        ax.set_xticks([0, 1, 5]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean charged cost to identify a practical ordering')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .06), ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Scope-matched static and adaptive verification controls', fontsize=14)
    fig.text(.5, .015, 'Ten saved masks; fixed verification scope and ideal audit oracle. '
        'Costs are sensitivity units, not measured annotation time. Paired credit is a redundant alias.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .2, 1, .92))
    for suffix in ['png', 'pdf']: fig.savefig(OUT/f'adaptive_costs_t4.{suffix}', dpi=180, bbox_inches='tight')


if __name__ == '__main__': main()
