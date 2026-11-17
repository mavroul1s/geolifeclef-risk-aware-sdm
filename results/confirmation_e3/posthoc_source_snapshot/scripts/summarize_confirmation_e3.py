"""Confirmation figures and explicitly post-primary skill diagnostics."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_confirmation_e3 import OUT,ROOT,SOURCE,paired,save


def main():
    config=json.loads((ROOT/'configs/confirmation_e3.json').read_text())
    report=json.loads((OUT/'report.json').read_text())
    cohort=pd.read_csv(ROOT/'data/confirmation_e3/locked_cohort.csv')
    data=np.load(OUT/'GPU_replay_inputs.npz',allow_pickle=False)
    q=[];p=[];r=[];a=[];b=[];individual=[]
    for model in config['models']:
        with np.load(OUT/'policies'/f'{model}.npz',allow_pickle=False) as policy:
            q.append(data[model+'__test_quality']);p.append(policy['affine_quality']);r.append(policy['HGB_quality'])
            a.append(data[model+'__candidate_accept']);b.append(data[model+'__baseline_accept'])
            individual.append(paired(config,[q[-1]],[p[-1]],[r[-1]],[a[-1]],[b[-1]],cohort.spatial_block))
    recomputed=paired(config,q,p,r,a,b,cohort.spatial_block)
    for key in ['F1_gain','MSE_gain','F1_interval','MSE_interval']:
        np.testing.assert_allclose(recomputed[key],report['primary_comparison'][key],atol=1e-14)
    save(OUT/'model_primary_comparisons.json',dict(zip(config['models'],individual)))
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    entries=individual+[report['primary_comparison']]
    labels=['DeepSDM 0','DeepSDM 1','Random forest 0','Random forest 1','Fixed-model mean']
    for ax,metric,title in [(axes[0],'F1','Retained F1: SDC minus HGB'),
                            (axes[1],'MSE','MSE: HGB minus calibrated SDC')]:
        for i,item in enumerate(entries):
            point=item[metric+'_gain'];lo,hi=item[metric+'_interval']
            ax.errorbar(point,i,xerr=[[max(point-lo,0)],[max(hi-point,0)]],fmt='o',capsize=3,
                color='#205d8f' if i<4 else '#3b8467')
        ax.axvline(0,color='gray',lw=1);ax.axvline(.002,color='#b74a48',ls='--',lw=1)
        ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlabel(title)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Locked E3 confirmation: joint superiority failed\n97.5% shared spatial intervals; dashed line is the practical margin',fontsize=11)
    fig.savefig(OUT/'primary_confirmation_intervals.png',dpi=180)
    fig.savefig(OUT/'primary_confirmation_intervals.pdf');plt.close(fig)
    diag=json.loads((OUT/'diagnostic_summary.json').read_text())
    skills={}
    for study,values in diag['MSE_means'].items():
        skills[study]={method:1-mse/values['training_mean_only'] for method,mse in values.items()}
    save(OUT/'posthoc_MSE_skill.json',{'definition':'1 minus ratio of fixed-model mean MSE to training-mean-only mean MSE',
        'posthoc':True,'skills':skills,'primary_result_changed':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained',sharey=True)
    for ax,study,title in zip(axes,['E2_spatial','E2_source','E3_domain'],
                              ['E2 spatial development','E2 source development','E3 locked larger-plot domain']):
        values=diag['MSE_means'][study];methods=['training_mean_only','affine_SDC','HGB']
        heights=[values[k] for k in methods]
        ax.bar(['Training mean','Affine SDC','HGB'],heights,color=['#777777','#205d8f','#bc8b3b'])
        for i,h in enumerate(heights):ax.text(i,h+.0005,f'{h:.4f}',ha='center',fontsize=8)
        ax.tick_params(axis='x',labelrotation=22);ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('All-cell quality-estimation MSE (lower is better)')
    axes[0].set_ylim(0,.035)
    fig.suptitle('Post-primary magnitude diagnostic: the preferred estimator changes across domains\nTraining-only constant controls were added after E3 outcomes; no new primary claim',fontsize=11)
    fig.savefig(OUT/'calibration_transport_reversal.png',dpi=180)
    fig.savefig(OUT/'calibration_transport_reversal.pdf');plt.close(fig)
    table=pd.read_csv(OUT/'summary.csv')
    empty_difference=float(np.max(np.abs(table.sample_F1-table.sample_F1_zero_empty)))
    save(OUT/'sensitivity_notes.json',{'maximum_empty_convention_F1_difference':empty_difference,
        'original_area_bins_all_saturated':bool(cohort.area_bin.nunique()==1),
        'Brazil_cells':int(cohort.country.eq('Brazil').sum()),
        'Brazil_blocks':int(cohort.loc[cohort.country.eq('Brazil'),'spatial_block'].nunique()),
        'source_population_inference_from_Brazil_supported':False})
    save(OUT/'gpu_admission.json',{'user_authorized':True,
        'primary_scientific_success':report['primary_success'],
        'correctness_tests_passed':report['correctness_tests_passed'],
        'GPU_job_launched':False,'reason':'co-primary confirmation and model guards failed',
        'future_qualifying_GPU_authorization_remains':True})
    print(json.dumps({'posthoc_skill':skills,'empty_F1_convention_difference':empty_difference}))


if __name__=='__main__':main()
