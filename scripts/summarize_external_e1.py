"""Descriptive secondary contrasts and figures for completed, fixed E1 policies."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_external_e1 import DATA, OUT, ROOT, check_mask, save, summarize, target_rows
from geolifeclef.external_reliability import decision_outcomes, joint_ecological_comparison, RISK_FEATURES


def main():
    config = json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    cohort = pd.read_csv(DATA/'cohort_roles.csv')
    take = np.flatnonzero(cohort.role.eq('assessment'));meta = cohort.iloc[take].reset_index(drop=True)
    eligible = np.load(DATA/'eligible_species.npy', allow_pickle=False)[take]
    rare = np.load(DATA/'rare_species_mask.npy', allow_pickle=False)
    y = target_rows(cohort, 'assessment', 'assess', True);check_mask(y, eligible)
    truth = (y[:, rare] > 0).sum(1)
    comparisons = [('lean_risk','SDC'), ('entropy','random'), ('SDC','mean_confidence'),
                   ('SDC','random'), ('total_count_constraint','SDC'), ('raw_rare_constraint','SDC')]
    results, diagnostics, source_rows = [], [], []
    sources = pd.read_csv(ROOT/'data/independent_data_audit/sage_eval_plot_metadata.csv',
                          usecols=['cell_row','GIVD_ID']).drop_duplicates()
    grouped = sources.groupby('cell_row').GIVD_ID
    source_name = grouped.first().where(grouped.nunique().eq(1), 'multiple_sources')
    meta['source_group'] = source_name.reindex(meta.cell_id).to_numpy()
    for decoder in config['decoders']:
        scores, masks = [], {name: [] for name in set(sum(([a,b] for a,b in comparisons), []))}
        for model in config['models']:
            with np.load(OUT/'policies'/f'{model}.npz', allow_pickle=False) as p:
                count = p['counts'][config['decoders'].index(decoder)]
                out = decision_outcomes(y, eligible, p['rank'], count, rare)
                scores.append(out.sample_F1.to_numpy());keys = p['keys'].tolist()
                for method in masks:
                    masks[method].append(p['accept'][keys.index(f'{decoder}|0.8|{method}')])
                f = p[decoder+'_features']
                diagnostics.append({'model': model, 'decoder': decoder,
                    'median_mean_selected_probability': float(np.median(f[:, RISK_FEATURES.index('mean_set_confidence')])),
                    'fraction_mean_selected_probability_below_half': float((f[:, RISK_FEATURES.index('mean_set_confidence')] < .5).mean()),
                    'mean_predicted_richness': float(count.mean()),
                    'observed_empty_fraction': float(out.true_richness.eq(0).mean())})
                if decoder == config['primary_decoder']:
                    for method in ['SDC','rare_count_constraint']:
                        accepted = p['accept'][keys.index(f'{decoder}|0.8|{method}')]
                        for source, positions in meta.groupby('source_group').groups.items():
                            ix = np.asarray(positions);sub = out.iloc[ix].reset_index(drop=True)
                            metrics = (summarize(sub, count[ix], accepted[ix], int(sub.rare_truth.sum()))
                                if accepted[ix].any() else {'retained': 0, 'sample_F1': None,
                                    'rare_occurrence_coverage': 0. if sub.rare_truth.sum() else None})
                            source_rows.append({'model': model, 'method': method, 'source': source,
                                'cells': len(ix), **metrics})
        for a,b in comparisons:
            joint = joint_ecological_comparison(scores, truth, masks[a], masks[b], meta.spatial_block, confidence=.95)
            results.append({'decoder': decoder, 'candidate': a, 'baseline': b, **joint})
    save(OUT/'secondary_comparisons.json', {'status': 'descriptive_secondary_analyses',
        'multiplicity': '95% descriptive intervals, not primary success tests; no selection of a replacement primary claim',
        'contrasts': results, 'mechanism_diagnostics': diagnostics})
    pd.DataFrame(source_rows).to_csv(OUT/'source_summary.csv', index=False)
    table = pd.read_csv(OUT/'summary.csv')
    main_table = table[table.decoder.eq(config['primary_decoder']) & table.coverage.eq(.8)]
    main_table.to_csv(OUT/'primary_table.csv', index=False)
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    methods = ['random','entropy','mean_confidence','SDC','lean_risk','rare_count_constraint']
    colors = ['#777777','#b74a48','#73a5bc','#205d8f','#bc8b3b','#3b8467']
    average = main_table.groupby('method').mean(numeric_only=True)
    labels = ['Random','Set entropy','Mean confidence','Soft Dice','Learned risk','Rare constraint']
    for ax,metric,title in zip(axes,['sample_F1','rare_occurrence_coverage'],
                              ['Retained-cell F1','Rare occurrences in retained cells']):
        values = average.loc[methods,metric].to_numpy()
        ax.barh(labels,values,color=colors)
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Fixed-model average')
        for i,v in enumerate(values): ax.text(v+.002,i,f'{v:.3f}',va='center',fontsize=9)
        ax.set_xlim(0, max(values)*1.15);ax.spines[['top','right']].set_visible(False)
        if metric == 'rare_occurrence_coverage':ax.axvline(.8,color='#333',ls='--',lw=1)
    fig.suptitle('Locked SAGE analysis: 80% retention, identical country / area budgets\nFour frozen model settings, 5,251 assessment cells',fontsize=12)
    fig.savefig(OUT/'external_primary_comparison.png',dpi=180)
    fig.savefig(OUT/'external_primary_comparison.pdf');plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for method,color,label in zip(['SDC','rare_count_constraint','lean_risk','entropy'],
                                ['#205d8f','#3b8467','#bc8b3b','#b74a48'],
                                ['Soft Dice','Rare constraint','Learned risk','Set entropy']):
        sub = table[table.decoder.eq(config['primary_decoder']) & table.method.eq(method)]
        g = sub.groupby('coverage').mean(numeric_only=True)
        axes[0].plot(g.index,g.sample_F1,'o-',label=label,color=color)
        axes[1].plot(g.index,g.rare_occurrence_coverage,'o-',label=label,color=color)
    axes[0].set_ylabel('Retained-cell F1');axes[1].set_ylabel('Rare occurrence coverage')
    for ax in axes:ax.set_xlabel('Survey retention');ax.spines[['top','right']].set_visible(False)
    axes[1].plot([.6,1],[.6,1],'--',color='gray',lw=1,label='Proportional reference')
    axes[1].legend(fontsize=8)
    fig.suptitle('Secondary coverage curves: fixed-model averages, descriptive')
    fig.savefig(OUT/'external_coverage_curves.png',dpi=180)
    fig.savefig(OUT/'external_coverage_curves.pdf');plt.close(fig)
    report = json.loads((OUT/'report.json').read_text())
    entries = list(report['model_primary_comparisons'].values())+[report['primary_joint_comparison']]
    labels = ['DeepSDM 0','DeepSDM 1','Random forest 0','Random forest 1','Fixed-model mean']
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,point,interval,scale,title,margin in [
        (axes[0],'F1_gain','F1_interval',1,'F1 difference',-.005),
        (axes[1],'rare_coverage_gain','rare_coverage_interval',100,'Rare-occurrence gain (percentage points)',3)]:
        for i, item in enumerate(entries):
            x = item[point]*scale;lo,hi = np.array(item[interval])*scale
            ax.errorbar(x,i,xerr=[[max(x-lo,0)],[max(hi-x,0)]],fmt='o',color='#205d8f' if i<4 else '#3b8467',capsize=3)
        ax.axvline(0,color='gray',lw=1);ax.axvline(margin,color='#b74a48',ls='--',lw=1)
        ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlabel(title)
        if point == 'F1_gain':
            ax.set_xticks([-.005,-.003,-.001,.001])
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Primary constraint vs Soft Dice: 97.5% paired spatial intervals\nDashed lines: declared practical margins; primary success failed',fontsize=11)
    fig.savefig(OUT/'external_primary_intervals.png',dpi=180)
    fig.savefig(OUT/'external_primary_intervals.pdf');plt.close(fig)
    for item in results:
        if item['decoder'] == config['primary_decoder']:
            print(json.dumps(item))


if __name__ == '__main__':
    main()
