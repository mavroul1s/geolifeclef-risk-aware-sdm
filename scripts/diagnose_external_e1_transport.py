"""Post-assessment explanatory diagnostic; never selects an E1 policy."""
import json
import time

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from run_external_e1 import DATA, OUT, ROOT, save, sha, target_rows
from geolifeclef.ecological_selection import QuotaSelection
from geolifeclef.external_reliability import (META_FEATURES, RISK_FEATURES,
    apply_calibration, decision_features, decision_outcomes, decode_counts,
    prediction_context, read_rows)


def metrics(prediction, truth, name, role):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    return {'predictor': name, 'role': role, 'rows': len(truth),
        'loss_MSE': float(np.mean((prediction-truth)**2)),
        'loss_bias': float(np.mean(prediction-truth)),
        'loss_Spearman': float(spearmanr(prediction,truth).statistic)}


def main():
    start = time.monotonic()
    # Explicitly post-hoc; this declaration is saved before fitting diagnostics.
    save(OUT/'transport_diagnostic_design.json', {
        'status': 'post_assessment_explanatory_analysis', 'primary_result_already_known': True,
        'source_sha256': sha(ROOT/'scripts/diagnose_external_e1_transport.py'),
        'question': 'Does fixed learned risk fail during spatial riskcal CV, assessment, or both?',
        'choice': 'five GroupKFold splits of riskcal 5-degree blocks, unchanged HGB hyperparameters/features',
        'new_neural_fits': 0, 'diagnostic_HGB_fits': 24,
        'original_frozen_policies_will_change': False})
    config = json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    cohort = pd.read_csv(DATA/'cohort_roles.csv',dtype={'area_bin':str})
    ixr = np.flatnonzero(cohort.role.eq('risk_calibration'))
    ixa = np.flatnonzero(cohort.role.eq('assessment'))
    yr = target_rows(cohort,'risk_calibration','fit')
    ya = target_rows(cohort,'assessment','assess',True)
    eligible = np.load(DATA/'eligible_species.npy',allow_pickle=False)
    rare = np.load(DATA/'rare_species_mask.npy',allow_pickle=False)
    cells = cohort.iloc[ixa]
    selector = QuotaSelection(cells.cell_id,.8,cells.country.astype(str)+'|'+cells.area_bin)
    outputs = []
    for model in config['models']:
        detail = json.loads((OUT/'policies'/f'{model}.json').read_text())
        with h5py.File(DATA/model/'test_predictions.h5') as f:
            raw = read_rows(f['probabilities'],cohort.prediction_row.to_numpy())
        p = apply_calibration(raw,detail['calibration'],eligible);ctx = prediction_context(p,eligible)
        count = decode_counts(ctx,config['primary_decoder'],detail['threshold'])
        features = decision_features(ctx,count,cohort[META_FEATURES[:-1]],rare)
        risk_out = decision_outcomes(yr,eligible[ixr],ctx['rank'][ixr],count[ixr],rare)
        assess_out = decision_outcomes(ya,eligible[ixa],ctx['rank'][ixa],count[ixa],rare)
        X = features.iloc[ixr][RISK_FEATURES];Xa = features.iloc[ixa][RISK_FEATURES]
        loss = risk_out.loss.to_numpy();folds = GroupKFold(5)
        oof = np.zeros(len(ixr))
        for train,val in folds.split(X,loss,groups=cohort.iloc[ixr].spatial_block):
            fit = HistGradientBoostingRegressor(**config['risk_regressor']).fit(X.iloc[train],loss[train])
            oof[val] = fit.predict(X.iloc[val])
        fit = HistGradientBoostingRegressor(**config['risk_regressor']).fit(X,loss)
        pa = fit.predict(Xa)
        with np.load(OUT/'policies'/f'{model}.npz',allow_pickle=False) as policy:
            key = f"{config['primary_decoder']}|0.8|lean_risk"
            if not np.array_equal(selector.accept(pa),policy['accept'][policy['keys'].tolist().index(key)]):
                raise ValueError('Diagnostic refit failed to reproduce the original frozen selection')
        for pred,target,name,role in [
            (fit.predict(X),loss,'HGB_in_sample','riskcal_training'),
            (oof,loss,'HGB_spatial_CV','riskcal_5_fold'),
            (1-features.iloc[ixr].SDC,loss,'SDC','riskcal_5_fold'),
            (pa,assess_out.loss,'HGB','assessment'),
            (1-features.iloc[ixa].SDC,assess_out.loss,'SDC','assessment')]:
            outputs.append({'model':model,**metrics(pred,target,name,role)})
        del raw,p,ctx
    pd.DataFrame(outputs).to_csv(OUT/'transport_diagnostic.csv',index=False)
    save(OUT/'transport_diagnostic_validation.json',{'all_four_frozen_selections_exactly_reproduced':True,
        'seconds':time.monotonic()-start,'primary_result_changed':False})
    print(pd.DataFrame(outputs).to_string(index=False))


if __name__ == '__main__':
    main()
