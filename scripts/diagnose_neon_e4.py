"""Explicit post-primary measurement/transport diagnostics; no policy changes."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geolifeclef.partial_set_bounds import fixed_set_f1_bounds,linear_contrast_bounds,mse_gain_bounds
from geolifeclef.calibration_diagnostics import mse_decomposition


def main():
    data,out=ROOT/'data/confirmation_e4',ROOT/'results/confirmation_e4'
    primary=json.loads((out/'primary_result.json').read_text());rows=pd.read_csv(out/'cell_results.csv')
    y=np.load(data/'assessment_labels.npz');contract=np.load(data/'label_contract.npz');policy=np.load(data/'frozen_policy.npz')
    lower,upper=fixed_set_f1_bounds(y['truth'],y['known'],contract['neon_eligible'],policy['rank'],policy['count'])
    rows['full_modeled_set_F1_lower']=lower;rows['full_modeled_set_F1_upper']=upper
    rows['F1_identification_width']=upper-lower
    valid=rows.scorable.to_numpy(bool);a=rows.loc[valid,'accept_HGB'].to_numpy(bool);b=rows.loc[valid,'accept_SDC'].to_numpy(bool)
    weights=a/a.sum()-b/b.sum()
    selection=linear_contrast_bounds(lower[valid],upper[valid],weights)
    magnitude=mse_gain_bounds(lower[valid],upper[valid],float(policy['mean_quality'][0]),policy['HGB_quality'][valid])
    q=rows.loc[valid,'F1'].to_numpy();decompositions={}
    for name,key in [('HGB','HGB_quality'),('training_mean','mean_quality'),('affine_SDC','affine_SDC_quality'),('raw_SDC','SDC')]:
        decompositions[name]=mse_decomposition(policy[key][valid],q)
    per_site=rows.groupby('site').agg(input_cells=('cell_id','size'),scorable_cells=('scorable','sum'),
                median_known_fraction=('known_fraction','median'),mean_F1_bound_width=('F1_identification_width','mean'))
    per_site.to_csv(out/'taxonomy_observability_by_site.csv')
    rows.to_csv(out/'posthoc_identification_bounds.csv',index=False)
    result={'status':'post_primary_diagnostic_complete','primary_result_sha256':hashlib.sha256((out/'primary_result.json').read_bytes()).hexdigest(),
            'primary_scientific_admission_unchanged':primary['scientific_admission'],'primary_success_unchanged':primary['joint_primary_success'],
            'all_input_cells':len(rows),'scorable_cells':int(valid.sum()),'excluded_cells':int((~valid).sum()),
            'all_cells_median_known_fraction':float(rows.known_fraction.median()),
            'all_cells_mean_F1_identification_width':float((upper-lower).mean()),
            'scorable_mean_F1_identification_width':float((upper[valid]-lower[valid]).mean()),
            'selection_difference_identification_bounds_scorable':selection,
            'MSE_gain_identification_bounds_scorable':magnitude,
            'magnitude_bias_variance_identity':decompositions,
            'no_policy_refit_or_threshold_relaxation':True,
            'interpretation':'Elementary sharp bounds under independent unknown binary labels, conditional on native/class/missingness contracts; not statistical confidence intervals or new theory. Additional taxonomic constraints may tighten them. This post-primary diagnostic cannot rescue failed admission.'}
    (out/'posthoc_measurement_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    per_site[['input_cells','scorable_cells']].plot.bar(ax=axes[0],color=['#b9c7d8','#21618c'])
    axes[0].set_ylabel('1 km cells');axes[0].set_title('Taxonomic observability limits admission');axes[0].set_xlabel('NEON site');axes[0].legend(['Input cohort','Pass 80% known-label rule'])
    order=np.argsort(rows.known_fraction.to_numpy());axes[1].plot(np.arange(len(rows)),rows.known_fraction.to_numpy()[order],color='#21618c')
    axes[1].axhline(.8,color='#a93226',linestyle='--',label='Frozen admission threshold')
    axes[1].set_ylim(0,1.03);axes[1].set_xlabel('Cells ordered by known-label fraction');axes[1].set_ylabel('Fraction of native modeled labels known');axes[1].set_title('163 inputs; 65 scorable cells / 10 sites');axes[1].legend()
    fig.suptitle('E4: failed admission; post-primary measurement audit')
    fig.savefig(out/'taxonomy_observability.png',dpi=180);fig.savefig(out/'taxonomy_observability.pdf');plt.close(fig)
    print(json.dumps(result))


if __name__=='__main__':main()
