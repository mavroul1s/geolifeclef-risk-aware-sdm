"""Secondary magnitude-calibration and oracle diagnostics, with no policy tuning."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_ranking_e2 import OUT, SOURCE, ROOT, save
from geolifeclef.ecological_selection import QuotaSelection


def mse_comparison(frame, repetitions=2000):
    arrays=[];blocks=None;ids=None
    for model in sorted(frame.model.unique()):
        g=frame[(frame.model==model)&frame.method.eq('HGB')].sort_values('cell_id')
        b=frame[(frame.model==model)&frame.method.eq('affine_SDC')].sort_values('cell_id')
        assert np.array_equal(g.cell_id,b.cell_id)
        if ids is None:ids=g.cell_id.to_numpy();blocks=g.spatial_block.to_numpy()
        assert np.array_equal(g.cell_id,ids)
        arrays.append((g.score.to_numpy()-g.quality.to_numpy())**2-
            (b.score.to_numpy()-b.quality.to_numpy())**2)
    differences=np.asarray(arrays)
    _,inverse=np.unique(blocks,return_inverse=True);n=int(inverse.max()+1)
    totals=np.array([np.bincount(inverse,weights=d,minlength=n) for d in differences])
    counts=np.bincount(inverse,minlength=n)
    draws=np.random.default_rng(20261116).integers(n,size=(repetitions,n))
    boot=totals[:,draws].sum(2)/counts[draws].sum(1)
    return {'contrast':'HGB loss MSE minus monotone-affine SDC loss MSE',
        'mean_MSE_difference':float(differences.mean()),
        'model_MSE_differences':differences.mean(1).tolist(),
        'descriptive_95_interval':np.quantile(boot.mean(0),[.025,.975]).tolist(),
        'clusters':n,'scope':'secondary development contrast, not an independent confirmation'}


def main():
    table=pd.read_csv(OUT/'summary.csv')
    rows=pd.read_csv(OUT/'heldout_development_rows.csv')
    report=json.loads((OUT/'report.json').read_text())
    config=json.loads((ROOT/'configs/ranking_study_e2.json').read_text())
    # The zero intervals are algebraic policy identity, not evidence of a small
    # unknown population effect for every possible trained ranker.
    baseline=rows[rows.method.eq('SDC')].sort_values(['study','model','cell_id'])
    pairwise=rows[rows.method.eq('pairwise')].sort_values(['study','model','cell_id'])
    assert np.array_equal(baseline.accepted,pairwise.accepted)
    comparisons={study:mse_comparison(g) for study,g in rows.groupby('study')}
    save(OUT/'magnitude_calibration_comparison.json',comparisons)
    inner=[]
    for split in report['splits']:
        for candidate,value in split['inner']['inner_values'].items():
            if candidate=='baseline':continue
            inner.append({'study':split['study'],'fold':str(split['fold']),'candidate':candidate,
                'inner_F1_gain':value-split['inner']['inner_values']['baseline']})
    pd.DataFrame(inner).to_csv(OUT/'inner_candidate_results.csv',index=False)
    # A nondeployable, label-oracle ceiling separates remaining selection
    # headroom from what these tested summary-feature learners can achieve.
    meta=pd.read_csv(SOURCE/'cohort_roles.csv',dtype={'area_bin':str}).set_index('cell_id')
    oracle=[]
    for (study,fold,model),g in baseline.groupby(['study','fold','model']):
        if study!='spatial':continue  # source study refits its own area-bin edges
        cov=meta.loc[g.cell_id]
        selector=QuotaSelection(g.cell_id.to_numpy(),config['coverage'],cov.country+'|'+cov.area_bin)
        chosen=selector.accept(-g.quality.to_numpy())
        oracle.append({'study':study,'fold':fold,'model':model,'retained':int(chosen.sum()),
            'oracle_F1_sum':float(g.quality.to_numpy()[chosen].sum()),
            'SDC_F1_sum':float(g.loc[g.accepted,'quality'].sum())})
    oracle=pd.DataFrame(oracle)
    oracle.to_csv(OUT/'label_oracle_headroom.csv',index=False)
    points=oracle.groupby('model').sum(numeric_only=True)
    save(OUT/'label_oracle_headroom.json',{'label_oracle_is_deployable':False,
        'model_conditional_F1_headroom':((points.oracle_F1_sum-points.SDC_F1_sum)/points.retained).to_dict(),
        'interpretation':'optimistic same-quota ceiling using validation outcomes; not an achieved method or benchmark improvement'})
    checks=report['gpu_checks']
    save(OUT/'gpu_admission.json',{'user_authorized_automatic_GPU_after_passing_tests':True,
        'correctness_tests_passed':checks['correctness_tests'],'scientific_checks':checks,
        'GPU_job_launched':False,'reason':'scientific promotion failed; all pairwise corrections rejected in inner CV',
        'authorization_for_future_passing_candidates_remains':True})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    methods=['SDC','affine_SDC','HGB','mse_residual']
    labels=['Raw SDC','Affine SDC','HGB risk','MSE residual']
    for ax,study in zip(axes,['spatial','source']):
        g=table[table.study.eq(study)].groupby('method').mean(numeric_only=True)
        ax.bar(labels,g.loc[methods,'raw_quality_score_MSE'],color=['#205d8f','#3b8467','#bc8b3b','#777'])
        ax.set_title('Spatial nested CV' if study=='spatial' else 'Source-held-out pipeline')
        ax.set_ylabel('Error-estimation MSE (lower is better)')
        ax.tick_params(axis='x',labelrotation=20);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Secondary magnitude-calibration control: fixed-model averages\nAffine SDC preserves every SDC acceptance decision',fontsize=11)
    fig.savefig(OUT/'magnitude_calibration_control.png',dpi=180)
    fig.savefig(OUT/'magnitude_calibration_control.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    candidates=pd.DataFrame(inner)
    for name,color in [('pairwise:0.1','#b74a48'),('pairwise:1.0','#bc8b3b'),('pairwise:10.0','#205d8f')]:
        g=candidates[candidates.candidate.eq(name)]
        ax.plot(np.arange(len(g)),g.inner_F1_gain,'o-',label=name,color=color)
    ax.axhline(.001,color='#3b8467',ls='--',label='Inner activation threshold')
    ax.axhline(0,color='gray',lw=1)
    ax.set_xticks(range(8),['Geo 0','Geo 1','Geo 2','Geo 3','Geo 4','AU source','US source','RU source'],rotation=25)
    ax.set_ylabel('Inner F1 gain over unchanged SDC');ax.legend(fontsize=8)
    ax.set_title('All ranking corrections rejected before outer evaluation')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'inner_ranking_admission.png',dpi=180)
    fig.savefig(OUT/'inner_ranking_admission.pdf');plt.close(fig)
    print(json.dumps({'MSE_comparisons':comparisons,'oracle_headroom':
        ((points.oracle_F1_sum-points.SDC_F1_sum)/points.retained).to_dict()}))


if __name__=='__main__':main()
