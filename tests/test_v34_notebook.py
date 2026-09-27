import hashlib
import json

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import v34_notebook_core as core
from scripts.build_v34_notebook import ROOT, OUTPUT
from test_v31_notebook import sensors


def test_graph_exact_shrinkage_self_edges_and_training_only():
    y = np.array([[1,1,0,0], [1,0,1,0], [0,1,1,0], [1,1,1,1]], np.uint8)
    take = np.arange(3)
    graph = core.CommunityGraph().fit(y, take)
    count = y[take].sum(0)
    prior = (count+.5)/4
    pairs = y[take].astype(float).T@y[take]
    expected = np.clip(np.log((pairs+100*prior)/(count[:,None]+100)/prior),-2,2)
    np.fill_diagonal(expected,0)
    assert np.allclose(graph.lift,expected,atol=1e-6)
    changed = y.copy(); changed[3] = 0
    assert np.array_equal(graph.lift,core.CommunityGraph().fit(changed,take).lift)
    assert np.isfinite(graph.lift).all()
    assert graph.record['training_rows'] == 3 and graph.record['minimum_species_count_filter'] is None
    with pytest.raises(ValueError,match='repeated'):
        core.CommunityGraph().fit(y,[0,0])


def test_context_formula_zero_graph_and_anchor_probabilities():
    rng = np.random.default_rng(100)
    y = (rng.random((50,140)) < .15).astype(np.uint8)
    graph = core.CommunityGraph().fit(y,np.arange(40))
    base = [list(range(12)),list(range(8,24))]
    bag = {'rank':np.tile(np.arange(128),(2,1)), 'value':np.tile(np.linspace(.9,.01,128),(2,1)).astype(np.float16)}
    specialist = {'rank':bag['rank'][:,::-1], 'value':bag['value']}
    result = graph.predict_rank(base,bag,specialist)
    for i in range(2):
        p = np.full(140,1e-7,np.float32)
        for expert in (bag,specialist):
            p[expert['rank'][i]] += expert['value'][i].astype(np.float32)/2
        anchors = base[i][:min(12,int(np.ceil(.6*len(base[i]))))]
        weights = p[anchors]/p[anchors].sum()
        context = weights@graph.lift[anchors]
        for alpha in core.GRAPH_STRENGTHS:
            s = np.log(p)+alpha*context
            expected = np.argsort(-s,kind='stable')[:128]
            assert np.array_equal(result[str(alpha)]['rank'][i],expected)
            assert np.allclose(result[str(alpha)]['value'][i],np.exp(s[expected]),atol=.001)


def test_all_17_policies_preserve_each_count_prefix_and_no_duplicates():
    base = [list(range(n)) for n in (8,18,40)]
    expert = {'rank':np.tile(np.arange(200,72,-1),(3,1)), 'value':np.ones((3,128))}
    data = {'base':base, 'community':{str(a):expert for a in core.GRAPH_STRENGTHS}}
    for p in core.POLICIES:
        core.v31.validate_residual(base,core.decode(data,p),p)
    assert core.self_tests()['passed']


def test_selection_never_reads_assessment_and_rejects_outside_core_loss():
    rows = pd.DataFrame({'surveyId':np.arange(60),'country':['Denmark']*30+['France']*30})
    old = np.full(60,.2)
    bundles = [{'split':{'calibration':np.arange(60)},'data':{'assessment':'DO NOT READ'},
                'trials':{p['id']:old.copy() for p in core.POLICIES}} for _ in range(2)]
    policy = core.POLICIES[3]
    for b in bundles:
        b['trials'][policy['id']] += np.r_[np.full(30,.1),np.full(30,-.01)]
    assert core.select_policy(bundles,rows)[0]['id'] == 'control'
    for b in bundles:
        b['trials'][policy['id']] = old+.01
    assert core.select_policy(bundles,rows)[0] == policy
    bundles[1]['trials'][policy['id']] = old-.001
    assert core.select_policy(bundles,rows)[0]['id'] == 'control'


def test_runtime_admission_includes_all_reference_fits_and_production():
    records = [{'group':g,'member':i,'training_surveys':100,'history':[{'seconds':10}]}
        for g,cs in (('bag',core.previous.BAG_CONFIGS),('specialist',core.previous.SPECIALISTS)) for i in range(len(cs))]
    b = {'records':records,'wall_seconds':5000,'split':{'training':np.arange(100)}}
    expected = (sum(core.previous.BAG_EPOCHS)+sum(c['epochs'] for c in core.previous.SPECIALISTS))*20*1.5+3600
    assert core.production_estimate([b],200) == expected
    assert core.remaining_plan_estimate([b],110,200) == 5000*1.1*1.25+expected+600


def test_standalone_frozen_sources_exact_csv_and_scored_calibrations(tmp_path):
    notebook = nbformat.read(OUTPUT,as_version=4)
    nbformat.validate(notebook)
    assert OUTPUT.stat().st_size < 1_000_000
    space = {}
    for cell in notebook.cells[1:3]:
        assert not cell.outputs and cell.execution_count is None
        exec(compile(cell.source,'standalone-v34','exec'),space)
    assert space['self_tests']()['passed']
    assert space['previous'] is not core.previous
    assert space['V34_SOURCE_HASH'] == hashlib.sha256((ROOT/'scripts/v34_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest()
    for name,digest in space['FROZEN_SOURCE_HASHES'].items():
        assert hashlib.sha256((ROOT/f'scripts/{name}_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest() == digest
    control = pd.read_csv(ROOT/'results/v32_kaggle_output/GLC25_PA_submission_v32.csv')
    species = np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    prediction = space['decode_control'](space['CONTROL_B64'],control,species)
    proof = space['legacy'].write_submission(tmp_path/'roundtrip.csv',control,control.surveyId.to_numpy(),prediction,species)
    assert proof['sha256'] == core.CONTROL_HASH
    with pytest.raises(ValueError,match='payload mismatch'):
        space['decode_control'](space['CONTROL_B64'][:-4]+'AAAA',control,species)
    report = json.loads((ROOT/'results/v32_kaggle_output/v32_report.json').read_text())
    assert core.FROZEN_V32_POLICY == report['policy']
    for r in report['training']['production']:
        if r['group'] == 'specialist':
            assert core.SPECIALIST_PRODUCTION_CALIBRATIONS[r['config']['id']] == r['calibration']


def test_v33_archival_hashes_and_duplicate_diagnosis():
    summary = json.loads((ROOT/'results/v33_summary.json').read_text())
    for name,digest in summary['artifacts'].items():
        if name.endswith(('.csv','.npz','.json')):
            assert hashlib.sha256((ROOT/'results/v33_kaggle_output'/name).read_bytes()).hexdigest() == digest
    assert not summary['eligible_for_submission'] and summary['exact_duplicate_of_v32']


@pytest.mark.parametrize('fault',['none','gate','budget','export','late'])
def test_real_22_development_5_production_and_failure_contracts(sensors,tmp_path,monkeypatch,fault):
    store,rows = sensors
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core.os,'cpu_count',lambda:2)
    store.labels = np.pad(store.labels,((0,0),(0,5016-12)))
    store.species_ids = np.arange(5016)
    store.cache = tmp_path/'sensors'; store.cache.mkdir()
    np.save(store.cache/'train_sentinel64.npy',store.high_train)
    np.save(store.cache/'test_sentinel64.npy',store.high_test)
    data = tmp_path/'official'; data.mkdir()
    rows.to_csv(data/'GLC25_PA_metadata_train.csv',index=False)
    template = pd.DataFrame({'surveyId':rows.surveyId,'predictions':['']*32})
    template.to_csv(data/'GLC25_SAMPLE_SUBMISSION.csv',index=False)
    monkeypatch.setattr(core.legacy,'EXPECTED_TEST_ROWS',32)
    base = [list(range(100,108)) for _ in range(32)]
    proof = core.legacy.write_submission(tmp_path/'control.csv',template,store.test_ids,base,store.species_ids)
    monkeypatch.setattr(core,'CONTROL_HASH',proof['sha256'])
    splits = [{'training':np.arange(16),'selection':np.arange(16,20),'calibration':np.arange(20,24),
               'assessment':np.arange(24+4*f,28+4*f)} for f in range(2)]
    monkeypatch.setattr(core.v31.previous,'make_splits',lambda *a,**k:(splits,[{'minimum_distance_km':21.}]*2))
    monkeypatch.setattr(core.legacy,'require_gpu',lambda:torch.device('cpu'))
    monkeypatch.setattr(core.legacy,'discover_data_root',lambda:data)
    monkeypatch.setattr(core,'decode_control',lambda *a:base)
    monkeypatch.setattr(core.legacy,'prepare_feature_store',lambda *a,**k:{})
    monkeypatch.setattr(core.legacy,'FeatureStore',lambda p:store)
    monkeypatch.setattr(core.legacy,'load_rows_and_pairs',lambda *a:(rows,rows,None))
    for module,field in ((core.previous,'BAG_CONFIGS'),(core.previous,'SPECIALISTS'),(core.v31,'CONFIGS')):
        monkeypatch.setattr(module,field,tuple({**c,'width':16,'epochs':1} for c in getattr(module,field)))
    monkeypatch.setattr(core.previous,'BAG_EPOCHS',(1,1,1))
    monkeypatch.setattr(core.v31,'EPOCHS',(1,1,1))
    # Actual neural models, calibration, graph and decoding run. The controlled
    # comparator ensures deterministic acceptance; this is NOT accuracy evidence.
    original = core.previous.fit_fold
    def reference(*a,**k):
        b = original(*a,**k)
        for d in b['data'].values():
            d['base'] = [list(range(100,108)) for _ in d['base']]
        return b
    monkeypatch.setattr(core.previous,'fit_fold',reference)
    monkeypatch.setattr(core,'FROZEN_V32_POLICY',{'id':'fixture','bag':0.,'specialist':0.,'swaps':0})
    original_geo = core.v31.geographic_gain
    def small_gain(*a,**k):
        r = original_geo(*a,**k); r['geographic_gain_positive'] = r['gain'] > 0
        return r
    monkeypatch.setattr(core.v31,'geographic_gain',small_gain)
    policy = {'id':'smoke','alpha':.25,'weight':2.,'swaps':2}
    monkeypatch.setattr(core,'select_policy',lambda *a:(core.POLICIES[0] if fault == 'gate' else policy,[]))
    monkeypatch.setattr(core,'V34_SOURCE_HASH','synthetic',raising=False)
    monkeypatch.setattr(core,'FROZEN_SOURCE_HASHES',{},raising=False)
    fits = []
    train = core.CommunityGraph.fit
    def checked_graph(self,labels,indices,guard=None):
        fits.append(np.asarray(indices).copy())
        assert np.array_equal(indices,np.arange(16)) or np.array_equal(indices,np.arange(32)) or labels.shape == (40,40)
        return train(self,labels,indices,guard)
    monkeypatch.setattr(core.CommunityGraph,'fit',checked_graph)
    if fault == 'budget':
        monkeypatch.setattr(core,'remaining_plan_estimate',lambda *a:99*3600)
    if fault == 'export':
        def fail(*a,**k):
            raise ValueError('export interrupted')
        monkeypatch.setattr(core,'save_compact',fail)
    if fault == 'late':
        save = core.legacy.save_json
        def extra(path,content):
            save(path,content)
            if path.name == 'v34_manifest.json':
                save(path.parent/'unexpected_extra.json',{'force':'late contract failure'})
        monkeypatch.setattr(core.legacy,'save_json',extra)
    export = tmp_path/'artifacts/v34_export'
    if fault in ('budget','export','late'):
        with pytest.raises((ValueError,RuntimeError,TimeoutError)):
            core.run_v34('')
        assert not (export/'GLC25_PA_submission_v34.csv').exists()
        failure = json.loads((export/'failure_report.json').read_text())
        assert not failure['eligible_for_submission'] and failure['prediction_file'] is None
        if fault == 'budget':
            assert len(failure['completed_folds']) == 1
            assert not any(len(x) == 32 for x in fits)
        else:
            assert (export/'failed_predictions_DO_NOT_SUBMIT.txt').exists()
        if fault == 'late':
            report = json.loads((export/'v34_report.json').read_text())
            assert report['status'] == 'failed' and not report['eligible_for_submission']
            assert not (export/'v34_manifest.json').exists()
    else:
        result = core.run_v34('')
        assert result['status'] == 'complete' and result['eligible_for_submission'] == (fault == 'none')
        assert len(list(export.iterdir())) == 5 and result['output_bytes'] < 16_000_000
        report = json.loads((export/'v34_report.json').read_text())
        for fold in report['training']['development']:
            assert len(fold['reference_members']) == 6 and len(fold['members']) == 5
            assert fold['graph']['training_rows'] == 16
        assert len(report['training']['production']) == (5 if fault == 'none' else 0)
        if fault == 'none':
            assert report['production_diagnostics']['all_PA_rows_used']
            assert report['production_diagnostics']['graph']['training_rows'] == 32
            output = pd.read_csv(export/result['prediction_file'])
            assert all(len(x.split()) == 8 for x in output.predictions)
        else:
            assert result['prediction_file'] is None
            assert (export/'NO_SUBMISSION.json').exists()
            assert not list(export.glob('GLC25*')) and not list(export.glob('*DO_NOT_SUBMIT.csv'))
        with np.load(export/'calibration_top128_v34.npz',allow_pickle=False) as archive:
            assert archive['ranked_species_columns'].shape == (8,6,128)
            assert archive['true_offsets'][-1] == len(archive['true_species_columns'])
        manifest = json.loads((export/'v34_manifest.json').read_text())
        for name,digest in manifest['outputs'].items():
            assert hashlib.sha256((export/name).read_bytes()).hexdigest() == digest
    assert not (tmp_path/'artifacts/v34_runtime').exists()


def test_gpu_preflight_and_no_ineligible_csv_on_start_failure(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    def missing():
        raise RuntimeError('Enable a Kaggle GPU')
    monkeypatch.setattr(core.legacy,'require_gpu',missing)
    monkeypatch.setattr(core.legacy,'prepare_feature_store',lambda *a:pytest.fail('GPU check too late'))
    with pytest.raises(RuntimeError,match='Enable a Kaggle GPU'):
        core.run_v34('')
    assert not (tmp_path/'artifacts/v34_runtime').exists()
    assert not list((tmp_path/'artifacts/v34_export').glob('*.csv'))
