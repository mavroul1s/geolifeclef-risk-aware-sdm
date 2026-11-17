"""Post-confirmation explanation of MSE reversal; no primary policy modification."""
import json

import numpy as np
import pandas as pd

from run_confirmation_e3 import OUT,ROOT,SOURCE,save,sha
from study_ranking_e2 import load_inputs,get_features
from geolifeclef.ranking_selection import buffered_training,monotone_affine,spatial_splits
from geolifeclef.calibration_diagnostics import mse_decomposition


def main():
    save(OUT/'diagnostic_design.json',{'status':'post_primary_explanatory_analysis',
        'primary_failure_already_known':True,'source_sha256':sha(ROOT/'scripts/diagnose_confirmation_e3.py'),
        'questions':['Do fitted scalar heads collapse toward training means?',
            'Which exact MSE identity terms change between E2 and E3?'],
        'new_primary_or_parameter_choices':False,'training_mean_control_is_posthoc':True})
    e2=ROOT/'results/ranking_study_e2'
    rows=pd.read_csv(e2/'heldout_development_rows.csv')
    config=json.loads((ROOT/'configs/ranking_study_e2.json').read_text())
    meta,eligible,rare,targets,raw,cal,risk=load_inputs(config)
    rm=meta.iloc[risk].reset_index(drop=True)
    constants=[];head_params=[]
    # Spatial quality uses the same fixed original Platt transform in all folds.
    for model in config['models']:
        observations=rows[rows.study.eq('spatial')&rows.model.eq(model)&rows.method.eq('SDC')].set_index('cell_id')
        quality=observations.reindex(rm.cell_id).quality.to_numpy()
        for fold,(train,val) in enumerate(spatial_splits(rm,5,50)):
            mean=float(quality[train].mean())
            for cell in rm.iloc[val].cell_id:
                constants.append({'study':'spatial','model':model,'cell_id':int(cell),'constant':mean})
    report=json.loads((e2/'report.json').read_text())
    for split in report['splits']:
        if split['study']!='source':continue
        source=split['fold'];val=np.flatnonzero(rm.source.eq(source))
        train=buffered_training(rm,np.flatnonzero(~rm.source.eq(source)),val,50)
        allowed=cal[~meta.iloc[cal].source.eq(source).to_numpy()]
        allowed=buffered_training(meta,allowed,risk[val],50)
        sm,features,_=get_features(config,meta,eligible,rare,targets,raw,allowed,risk)
        for model,data in features.items():
            prediction,params=monotone_affine(data['base'][train],data['quality'][train],data['base'][val])
            reference=rows[rows.study.eq('source')&rows.fold.eq(source)&rows.model.eq(model)&rows.method.eq('affine_SDC')].set_index('cell_id')
            np.testing.assert_allclose(prediction,reference.reindex(sm.iloc[val].cell_id).score,atol=1e-12,rtol=1e-10)
            head_params.append({'study':'E2_source','fold':source,'model':model,**params})
            for cell in sm.iloc[val].cell_id:
                constants.append({'study':'source','model':model,'cell_id':int(cell),'constant':float(data['quality'][train].mean())})
    constants=pd.DataFrame(constants)
    diagnostics=[]
    for (study,model),g in rows.groupby(['study','model']):
        for method in ['affine_SDC','HGB','SDC']:
            take=g[g.method.eq(method)].sort_values('cell_id')
            diagnostics.append({'study':'E2_'+study,'model':model,'method':method,
                **mse_decomposition(take.score,take.quality)})
        take=g[g.method.eq('SDC')].sort_values('cell_id')
        matched=constants[constants.study.eq(study)&constants.model.eq(model)].set_index('cell_id').reindex(take.cell_id)
        diagnostics.append({'study':'E2_'+study,'model':model,'method':'training_mean_only',
            **mse_decomposition(matched.constant,take.quality)})
    with np.load(OUT/'GPU_replay_inputs.npz',allow_pickle=False) as inputs:
        for model in config['models']:
            with np.load(OUT/'policies'/f'{model}.npz',allow_pickle=False) as policy:
                q=inputs[model+'__test_quality']
                for method,p in [('affine_SDC',policy['affine_quality']),('HGB',policy['HGB_quality']),
                    ('SDC',policy['SDC']),('training_mean_only',np.full(len(q),policy['train_quality'].mean()))]:
                    diagnostics.append({'study':'E3_domain','model':model,'method':method,**mse_decomposition(p,q)})
            params=json.loads((OUT/'policies'/f'{model}.json').read_text())['affine_quality_calibration']
            head_params.append({'study':'E3_domain','fold':'full_calibration','model':model,**params})
    d=pd.DataFrame(diagnostics);d.to_csv(OUT/'magnitude_MSE_decomposition.csv',index=False)
    pd.DataFrame(head_params).to_csv(OUT/'affine_head_parameters.csv',index=False)
    differences=[]
    for (study,model),g in d.groupby(['study','model']):
        a=g.set_index('method').loc['affine_SDC'];b=g.set_index('method').loc['HGB']
        values={name:float(b[name]-a[name]) for name in ['MSE','squared_bias','prediction_variance','minus_twice_covariance']}
        assert np.isclose(values['MSE'],sum(values[n] for n in ['squared_bias','prediction_variance','minus_twice_covariance']),atol=1e-12)
        differences.append({'study':study,'model':model,**values})
    pd.DataFrame(differences).to_csv(OUT/'MSE_gain_decomposition.csv',index=False)
    save(OUT/'diagnostic_summary.json',{'status':'completed_post_primary_diagnostics',
        'E2_source_affine_heads_exactly_reproduced':True,'original_primary_modified':False,
        'MSE_means':d.groupby(['study','method']).MSE.mean().unstack('method').to_dict('index'),
        'HGB_minus_affine_components':pd.DataFrame(differences).groupby('study')[[
            'MSE','squared_bias','prediction_variance','minus_twice_covariance']].mean().to_dict('index'),
        'limits':'algebraic descriptive explanation, not causal identification of plot-area or source effects; constant control posthoc'})
    print(d.groupby(['study','method'])[['MSE','prediction_mean','quality_mean','squared_bias']].mean().to_string())


if __name__=='__main__':main()
