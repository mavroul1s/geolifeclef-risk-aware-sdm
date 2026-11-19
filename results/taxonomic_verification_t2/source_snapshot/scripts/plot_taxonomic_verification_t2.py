"""Standalone scientific figure for the fixed T2 replay."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/taxonomic_verification_t2'
LABELS = {'uniform': 'Uniform all keys', 'uniform_support': 'Uniform disagreement support',
    'incidence_support': 'Support incidence', 'global_width': 'Global width credit',
    'paired_width': 'Paired width credit'}
COLORS = ['#8a8a8a', '#167d9a', '#d68022', '#34915f', '#7a4eb0']


def main():
    frame = pd.read_csv(OUT/'curves.csv'); summary = json.loads((OUT/'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, unit, title in zip(axes, ['block_shared', 'cell_local'], ['Shared block-concept cost', 'Local record cost']):
        for (name, label), color in zip(LABELS.items(), COLORS):
            subset = frame[(frame.unit == unit) & (frame.policy == name)]
            mean = subset.groupby('budget_fraction').width.mean()
            ax.plot(mean.index*100, mean.values, color=color, label=label,
                    linewidth=2.4 if name == 'paired_width' else 1.7,
                    linestyle='--' if name == 'uniform' else '-', marker='o', markersize=3)
        ax.set_title(title); ax.set_xlabel('Budget (% of initial unresolved keys)')
        ax.grid(alpha=.2); ax.set_xlim(-2, 102); ax.set_ylim(bottom=0)
    axes[0].set_ylabel('HGB − SDC F1 outer-range width')
    fig.suptitle('Taxonomic verification of a fixed policy contrast', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .06), fontsize=9)
    status = 'passes' if summary['practical_target_passed'] else 'fails'
    fig.text(.5, .015, f'Means over ten censoring masks; practical target {status}. '
        'Identical initial information; query units have different absolute costs. Ranges are not CIs.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .2, 1, .92))
    for suffix in ['png', 'pdf']: fig.savefig(OUT/f'verification_cost_t2.{suffix}', dpi=180, bbox_inches='tight')


if __name__ == '__main__': main()
