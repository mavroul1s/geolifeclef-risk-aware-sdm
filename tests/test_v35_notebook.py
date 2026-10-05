import base64
import hashlib
import json
import lzma

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import v35_notebook_core as core
from scripts.build_v35_notebook import ROOT, OUTPUT
from test_v31_notebook import sensors


def fixture_reference(rows, store, splits):
    manifests, folds = [], []
    for split in splits:
        hashes = {k: hashlib.sha256(rows.surveyId.to_numpy(np.int64)[v].astype('<i8').tobytes()).hexdigest() for k,v in split.items()}
        manifests.append({'id_hashes':hashes,'minimum_distance_km':21.})
        fold = {'id_hashes':hashes}
        for role in ('calibration','assessment'):
            take = split[role]
            pred = [list(range(100,108)) for _ in take]
            fold[role] = {'ids':rows.iloc[take].surveyId.tolist(), 'f1':core.legacy.score_prediction_lists(store.labels[take],pred).tolist()}
            if role == 'calibration':
                fold[role]['predictions'] = pred
        folds.append(fold)
    ref = {'folds':folds,'provenance':{'scope':'synthetic fixture'},'labels_sha256':core.label_hash(store.labels),
           'species_sha256':hashlib.sha256(store.species_ids.astype('<i8').tobytes()).hexdigest()}
    return ref,manifests


def test_actual_width_balanced_multisensor_and_rare_head_backward(sensors):
    store,_ = sensors
    store.labels = np.pad(store.labels,((0,0),(0,5016-12))); store.species_ids = np.arange(5016)
    config = core.CONFIGS[1]
    model,_ = core.create_model(store,config,np.arange(16))
    assert isinstance(model,core.BalancedAttention)
    model.head = core.previous.RareResidualHead(model.head,np.array([2,3]))
    stats = core.v29.fit_normalization(store,np.arange(16))
    x = core.v29.batch_inputs(store,np.arange(2),stats,torch.device('cpu'))
    logits,count = model(x,False)
    target = torch.as_tensor(store.labels[:2],dtype=torch.float32)
    loss = core.previous.asymmetric_loss(logits,target,torch.ones(5016))+.08*core.ranking_loss(logits,target)
    loss.backward()
    assert logits.shape == (2,5016) and count.shape == (2,)
    assert torch.isfinite(loss)
    assert model.fusion[0].weight.grad.abs().sum() > 0
    assert model.head.rare[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ranking_loss_row_normalization_soft_targets_and_zero_rows():
    logits = torch.zeros(3,12,requires_grad=True)
    targets = torch.zeros(3,12); targets[0,0] = 1; targets[1,:4] = .5
    loss = core.ranking_loss(logits,targets)
    assert float(loss.detach()) == pytest.approx(2/3*np.log(12))
    loss.backward()
    assert logits.grad[0].abs().sum() > 0 and torch.all(logits.grad[2] == 0)
    assert torch.isfinite(logits.grad).all()


def test_decoding_full_sets_counts_not_inherited_and_ensemble():
    p = np.tile(np.linspace(.9,.01,80),(3,1)).astype(np.float32)
    for policy in core.POLICIES:
        pred = core.decode(p,policy['count_mode'])
        assert all(8 <= len(x) <= 40 and len(x) == len(set(x)) for x in pred)
        assert all(x[0] == 0 for x in pred)
    assert all(len(x) == 18 for x in core.decode(p,'fixed18'))
    assert all(len(x) == 24 for x in core.decode(p,'fixed24'))
    assert np.allclose(core.combine([p,p*.5,p*.25],'ecology_lean'),p*(1+1+.25)/4)
    assert core.self_tests()['passed']


def test_cached_reference_roles_species_labels_and_calibration_truth(sensors):
    store,rows = sensors
    store.labels = np.pad(store.labels,((0,0),(0,5016-12))); store.species_ids = np.arange(5016)
    splits = [{'training':np.arange(16),'selection':np.arange(16,20),'calibration':np.arange(20,24),
               'assessment':np.arange(24+4*f,28+4*f)} for f in range(2)]
    ref,manifests = fixture_reference(rows,store,splits)
    bound = core.bind_reference(ref,rows,store.species_ids,splits,manifests,store.labels)
    assert np.array_equal(bound[1]['assessment']['indices'],np.arange(28,32))
    changed = store.labels.copy(); changed[31,0] ^= 1
    with pytest.raises(ValueError,match='PA labels changed'):
        core.bind_reference(ref,rows,store.species_ids,splits,manifests,changed)
    with pytest.raises(ValueError,match='vocabulary'):
        core.bind_reference(ref,rows,store.species_ids[::-1],splits,manifests,store.labels)
    corrupted = json.loads(json.dumps(ref)); corrupted['folds'][0]['id_hashes']['training'] = 'wrong'
    with pytest.raises(ValueError,match='split IDs'):
        core.bind_reference(corrupted,rows,store.species_ids,splits,manifests,store.labels)
    corrupted = json.loads(json.dumps(ref)); corrupted['folds'][0]['calibration']['f1'][0] = .1
    with pytest.raises(ValueError,match='Calibration labels'):
        core.bind_reference(corrupted,rows,store.species_ids,splits,manifests,store.labels)


def test_practical_selection_rejects_v34_sized_gain_and_keeps_attempt_diagnostics():
    rows = pd.DataFrame({'surveyId':np.arange(60),'country':['Denmark']*30+['France']*30})
    ref = np.full(60,.2)
    bundles = [{'data':{'calibration':{'indices':np.arange(60),'reference_f1':ref}, 'assessment':'DO NOT READ'},
                'trials':{p['id']:ref+.00043 for p in core.POLICIES}} for _ in range(2)]
    p,trials,allowed = core.select_policy(bundles,rows)
    assert not allowed and all(not t['allowed'] for t in trials)
    good = core.POLICIES[3]
    for b in bundles:
        b['trials'][good['id']] = ref+.003
    assert core.select_policy(bundles,rows)[0] == good and core.select_policy(bundles,rows)[2]
    bundles[1]['trials'][good['id']] = ref-.001
    assert not core.select_policy(bundles,rows)[2]


def test_epoch_transfer_and_conservative_runtime_estimate():
    a = {'records':[{'best_epoch':12,'training_surveys':100,'history':[{'seconds':10}]} for _ in core.CONFIGS],
         'wall_seconds':5000,'split':{'training':np.arange(100)}}
    b = {'records':[{'best_epoch':36,'training_surveys':100,'history':[{'seconds':11}]} for _ in core.CONFIGS]}
    assert core.production_epochs([a,b]) == [24]*3
    assert core.production_estimate([a,b],200) == pytest.approx(3*11/100*200*24*1.5+2700)
    assert core.remaining_plan_estimate([a],110,200) == pytest.approx(5000*1.1*1.25+3*10/100*200*48*1.5+2700+900)


def test_shorter_production_keeps_lr_prefix_and_never_queries_selection(sensors,tmp_path,monkeypatch):
    store,rows = sensors
    config = {**core.CONFIGS[0],'width':16,'epochs':4}
    take = np.arange(16); stats = core.v29.fit_normalization(store,take)
    monkeypatch.setattr(core.v29,'predict',lambda *a,**k:pytest.fail('Production must not query selection labels'))
    _,record = core.train_model(store,rows,take,np.array([],np.int64),stats,config,tmp_path,
        core.legacy.RuntimeGuard(10.75),torch.device('cpu'),production_epochs=2)
    expected = [1e-5+(.0003-1e-5)*(1+np.cos(np.pi*e/4))/2 for e in (1,2)]
    assert [h['learning_rate_after_epoch'] for h in record['history']] == pytest.approx(expected)
    assert record['best_epoch'] == 2 and record['scheduler_total_epochs'] == 4


def test_standalone_authentic_reference_and_no_reference_neural_refits():
    notebook = nbformat.read(OUTPUT,as_version=4); nbformat.validate(notebook)
    assert OUTPUT.stat().st_size < 1_000_000
    space = {}
    for cell in notebook.cells[1:3]:
        assert not cell.outputs and cell.execution_count is None
        exec(compile(cell.source,'standalone-v35','exec'),space)
    assert space['self_tests']()['passed']
    assert space['previous'] is not core.previous
    assert space['V35_SOURCE_HASH'] == hashlib.sha256((ROOT/'scripts/v35_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest()
    for name,digest in space['FROZEN_SOURCE_HASHES'].items():
        assert hashlib.sha256((ROOT/f'scripts/{name}_notebook_core.py').read_text(encoding='utf-8').encode()).hexdigest() == digest
    reference = space['load_reference'](space['REFERENCE_B64'])
    manifest = json.loads((ROOT/'results/v34_kaggle_output/v34_manifest.json').read_text())
    assert [f['id_hashes'] for f in reference['folds']] == [f['id_hashes'] for f in manifest['splits']]
    assert sum(len(f['assessment']['ids']) for f in reference['folds']) == 31570
    assert sum(len(f['calibration']['ids']) for f in reference['folds']) == 3000
    assert reference['train_metadata_sha256'] == hashlib.sha256((ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv').read_bytes()).hexdigest()
    with pytest.raises((ValueError,lzma.LZMAError),match='mismatch|Corrupt'):
        space['load_reference'](space['REFERENCE_B64'][:-4]+'AAAA')
    source = (ROOT/'scripts/v35_notebook_core.py').read_text(encoding='utf-8')
    assert 'previous.fit_fold(' not in source and 'previous.train_specialist(' not in source


@pytest.mark.parametrize('fault',['none','gate','budget','export','late','metadata'])
def test_real_pipeline_six_new_fits_production_and_fail_closed(sensors,tmp_path,monkeypatch,fault):
    store,rows = sensors
    monkeypatch.chdir(tmp_path); monkeypatch.setattr(core.os,'cpu_count',lambda:2)
    store.labels = np.pad(store.labels,((0,0),(0,5016-12))); store.species_ids = np.arange(5016)
    store.cache = tmp_path/'sensors'; store.cache.mkdir()
    np.save(store.cache/'train_sentinel64.npy',store.high_train); np.save(store.cache/'test_sentinel64.npy',store.high_test)
    root = tmp_path/'official'; root.mkdir()
    rows.to_csv(root/'GLC25_PA_metadata_train.csv',index=False)
    template = pd.DataFrame({'surveyId':rows.surveyId,'predictions':['']*32})
    template.to_csv(root/'GLC25_SAMPLE_SUBMISSION.csv',index=False)
    splits = [{'training':np.arange(16),'selection':np.arange(16,20),'calibration':np.arange(20,24),
               'assessment':np.arange(24+4*f,28+4*f)} for f in range(2)]
    reference,manifests = fixture_reference(rows,store,splits)
    reference['train_metadata_sha256'] = core.legacy.sha256_file(root/'GLC25_PA_metadata_train.csv') if fault != 'metadata' else 'wrong'
    raw = json.dumps(reference).encode(); packed = lzma.compress(raw)
    monkeypatch.setattr(core,'REFERENCE_PAYLOAD_HASH',hashlib.sha256(packed).hexdigest(),raising=False)
    monkeypatch.setattr(core,'REFERENCE_RAW_HASH',hashlib.sha256(raw).hexdigest(),raising=False)
    monkeypatch.setattr(core,'V35_SOURCE_HASH','fixture',raising=False); monkeypatch.setattr(core,'FROZEN_SOURCE_HASHES',{},raising=False)
    monkeypatch.setattr(core.v31.previous,'make_splits',lambda *a,**k:(splits,manifests))
    monkeypatch.setattr(core.legacy,'EXPECTED_TEST_ROWS',32)
    monkeypatch.setattr(core.legacy,'require_gpu',lambda:torch.device('cpu'))
    monkeypatch.setattr(core.legacy,'discover_data_root',lambda:root)
    feature_calls = []
    def prepare(*a,**k):
        feature_calls.append(True); return {}
    monkeypatch.setattr(core.legacy,'prepare_feature_store',prepare)
    monkeypatch.setattr(core.legacy,'FeatureStore',lambda p:store)
    monkeypatch.setattr(core.legacy,'load_rows_and_pairs',lambda *a:(rows,rows,None))
    monkeypatch.setattr(core,'CONFIGS',tuple({**c,'width':16,'epochs':1} for c in core.CONFIGS))
    monkeypatch.setattr(core,'CHECKPOINT_EPOCHS',(1,))
    geo = core.v31.geographic_gain
    def small(*a,**k):
        result = geo(*a,**k); result['geographic_gain_positive'] = result['gain'] > 0
        return result
    monkeypatch.setattr(core.v31,'geographic_gain',small)
    train, calls = core.train_model, []
    def checked(*a,**k):
        is_prod = k.get('production_epochs') is not None
        calls.append('production' if is_prod else 'development')
        assert np.array_equal(a[2],np.arange(32 if is_prod else 16))
        return train(*a,**k)
    monkeypatch.setattr(core,'train_model',checked)
    # Force equal ensemble to exercise all three production models, not accuracy.
    select = core.select_policy
    def chosen(*a):
        _,trials,allowed = select(*a)
        return core.POLICIES[3],trials,False if fault == 'gate' else allowed
    monkeypatch.setattr(core,'select_policy',chosen)
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
            if path.name == 'v35_manifest.json':
                save(path.parent/'unexpected.json',{'force':'late failure'})
        monkeypatch.setattr(core.legacy,'save_json',extra)
    export = tmp_path/'artifacts/v35_export'
    payload = base64.b64encode(packed).decode()
    if fault in ('budget','export','late','metadata'):
        with pytest.raises((ValueError,RuntimeError,TimeoutError)):
            core.run_v35(payload)
        assert not (export/'GLC25_PA_submission_v35.csv').exists()
        failure = json.loads((export/'failure_report.json').read_text())
        assert not failure['eligible_for_submission'] and failure['prediction_file'] is None
        if fault == 'metadata':
            assert not calls and not feature_calls
        if fault == 'budget':
            assert calls == ['development']*3 and len(failure['completed_folds']) == 1
        if fault in ('export','late'):
            assert (export/'failed_predictions_DO_NOT_SUBMIT.txt').exists()
        if fault == 'late':
            report = json.loads((export/'v35_report.json').read_text())
            assert report['status'] == 'failed' and not report['eligible_for_submission']
            assert not (export/'v35_manifest.json').exists()
    else:
        result = core.run_v35(payload)
        assert result['eligible_for_submission'] == (fault == 'none')
        assert calls.count('development') == 6 and calls.count('production') == (3 if fault == 'none' else 0)
        assert len(list(export.iterdir())) == 5 and result['output_bytes'] < 16_000_000
        report = json.loads((export/'v35_report.json').read_text())
        assert report['regression']['gain'] > 0  # attempted candidate, not fake zero-control
        if fault == 'none':
            assert report['production_diagnostics']['all_PA_rows_used']
            assert report['production_epochs_frozen_before_assessment'] == [1]*3
            for record in report['training']['production']:
                assert record['scheduler_total_epochs'] == 1
            output = pd.read_csv(export/result['prediction_file'])
            assert all(len(x.split()) == 18 for x in output.predictions)
        else:
            assert result['prediction_file'] is None and (export/'NO_SUBMISSION.json').exists()
        with np.load(export/'predictions_top128_v35.npz',allow_pickle=False) as archive:
            assert archive['ranked_species_columns'].shape == (16,3,128)
            assert set(archive['role']) == {'calibration','assessment'}
            assert archive['probability_mass'].shape == (16,3)
            assert archive['true_offsets'][-1] == len(archive['true_species_columns'])
        manifest = json.loads((export/'v35_manifest.json').read_text())
        for name,digest in manifest['outputs'].items():
            assert hashlib.sha256((export/name).read_bytes()).hexdigest() == digest
    assert not (tmp_path/'artifacts/v35_runtime').exists()


def test_gpu_fail_fast(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    def fail():
        raise RuntimeError('Enable a Kaggle GPU')
    monkeypatch.setattr(core.legacy,'require_gpu',fail)
    with pytest.raises(RuntimeError,match='Enable a Kaggle GPU'):
        core.run_v35('')
    assert not list((tmp_path/'artifacts/v35_export').glob('*.csv'))
