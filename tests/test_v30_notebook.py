import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import nbformat

from scripts import v30_notebook_core as core
from scripts.build_v30_notebook import ROOT, OUTPUT
from scripts.build_v29_notebook import consumed_paths


@pytest.fixture
def sensors():
    torch.set_num_threads(2)
    rng = np.random.default_rng(101)
    n = 32
    rows = pd.DataFrame({'surveyId':np.arange(n), 'lat':np.linspace(30, 65, n),
        'lon':np.linspace(-10, 30, n), 'country':['France','Denmark']*16})
    vectors = {k:rng.normal(size=(n,5)).astype(np.float32) for k in core.legacy.MODALITIES}
    rasters = {k:rng.normal(size=(n,*shape)).astype(np.float16) for k,shape in core.legacy.RASTER_SHAPES.items()}
    labels = (rng.random((n,12)) < .2).astype(np.uint8)
    labels[:,0] = 1
    labels[:,-1] = 0; labels[0,-1] = 1
    store = SimpleNamespace(train=vectors,test=vectors,raster_train=rasters,raster_test=rasters,
        high_train=rng.uniform(size=(n,4,64,64)).astype(np.float16),labels=labels,species_ids=np.arange(12),
        dims={k:5 for k in vectors},train_ids=rows.surveyId.to_numpy(),test_ids=rows.surveyId.to_numpy(),
        static_train=core.previous.candidate_static(rows),eco_train=core.previous.candidate_static(rows,False))
    store.high_test=store.high_train; store.static_test=store.static_train; store.eco_test=store.eco_train
    return store,rows


def test_multiscale_actual_backpropagation_and_rare_head(sensors,tmp_path):
    store,rows=sensors
    config={**core.CONFIGS[-1],'width':16,'epochs':3}
    training,selection=np.arange(24),np.arange(24,32)
    stats=core.previous.fit_normalization(store,training)
    initial=core.create_model(store,config,training).head.weight.detach().clone()
    old_factory=core.previous.create_model
    model,record=core.train_model(store,rows,training,selection,stats,config,tmp_path,
        core.legacy.RuntimeGuard(),torch.device('cpu'))
    assert core.previous.create_model is old_factory
    assert not torch.equal(initial[-1],model.head.weight[-1])
    assert model.head.out_features==12 and record['best_epoch'] in (1,3)
    p=core.previous.predict(model,store,selection,stats,torch.device('cpu'),config,views=2)
    assert p.shape==(8,12) and np.isfinite(p).all()


def test_factory_restored_after_failure(sensors,tmp_path,monkeypatch):
    before=core.previous.create_model
    def fail(*a,**k):
        raise RuntimeError('expected failure')
    monkeypatch.setattr(core.previous,'train_candidate',fail)
    with pytest.raises(RuntimeError,match='expected failure'):
        core.train_model()
    assert core.previous.create_model is before


def test_official_split_and_consumed_union():
    path=ROOT/'artifacts/v20_frozen/raw/GLC25_PA_metadata_train.csv'
    if not path.exists():
        pytest.skip('Official metadata not installed')
    rows=pd.read_csv(path,usecols=['surveyId','lat','lon','country']).drop_duplicates('surveyId').reset_index(drop=True)
    splits,manifests=core.make_splits(rows)
    assert [m['counts']['training'] for m in manifests]==[35811,37487]
    assert [m['counts']['assessment'] for m in manifests]==[15792,15778]
    assert all(not m['fresh_assessment'] and m['minimum_distance_km']>=20 for m in manifests)
    assert not np.intersect1d(splits[0]['assessment'],splits[1]['assessment']).size
    prod,manifest=core.production_split(rows)
    assert manifest['counts']=={'training':72063,'anchor':8302}
    assert manifest['minimum_distance_km']>=20
    old=consumed_paths()+[ROOT/'results/v29_kaggle_output/assessment_per_survey_v29.csv']
    used=np.unique(np.concatenate([pd.read_csv(p,usecols=['surveyId']).surveyId.to_numpy() for p in old]))
    assert len(used)==88987 and np.isin(rows.surveyId,used).all()


def test_role_assignment_is_row_order_invariant():
    rows=pd.DataFrame({'surveyId':np.arange(120),'lat':np.arange(120)+.1,
                      'lon':np.zeros(120),'country':['France','Italy','Germany']*40})
    first,_=core.registered_roles(rows)
    shuffled=rows.sample(frac=1,random_state=3)
    second,_=core.registered_roles(shuffled)
    assert np.array_equal(first, pd.Series(second,index=shuffled.index).sort_index().to_numpy())


def test_count_and_decoder_contract(sensors):
    store,rows=sensors
    p=np.full(store.labels.shape,.05,np.float32)+store.labels*.2
    distance=np.arange(len(rows),dtype=float)+20
    model,record=core.fit_count(p,store.labels,rows,distance)
    args=core.count_values(p,rows,distance,model)
    base=[list(range(8)) for _ in rows.index]
    assert not record['fitted_on_assessment']
    assert np.array_equal(core.count_features(p,rows,distance),core.count_features(p,rows.assign(country='Ukraine'),distance))
    for policy in core.POLICIES:
        predictions=core.decode(base,*args,policy)
        assert all(8<=len(x)<=12 and len(set(x))==len(x) for x in predictions)
    assert core.decode(base,*args,core.POLICIES[0])==base


def test_policy_rejects_a_one_fold_regression():
    rows=pd.DataFrame({'surveyId':np.arange(80),'country':['France']*80,
                      'lat':np.repeat([40,50],40),'lon':np.zeros(80)})
    bundles=[]
    for fold in range(2):
        scores={p['id']:np.full(80,.3) for p in core.POLICIES}
        scores[core.POLICIES[1]['id']]=np.full(80,.4 if fold==0 else .299)
        scores[core.POLICIES[2]['id']]=np.full(80,.305)
        bundles.append({'split':{'calibration':np.arange(80)},'trials':scores})
    selected,trials=core.select_policy(bundles,rows)
    assert selected['id']==core.POLICIES[2]['id']
    assert not next(t for t in trials if t['id']==core.POLICIES[1]['id'])['allowed']


def test_embedded_notebook_and_exact_v29_control(tmp_path):
    notebook=nbformat.read(OUTPUT,as_version=4)
    nbformat.validate(notebook)
    assert OUTPUT.stat().st_size<1_000_000
    source=(ROOT/'scripts/v30_notebook_core.py').read_text(encoding='utf-8')
    assert notebook.metadata.glc_v30.source_sha256==hashlib.sha256(source.encode()).hexdigest()
    namespace={}
    for cell in notebook.cells[1:3]:
        assert cell.outputs==[] and cell.execution_count is None
        exec(compile(cell.source,'<embedded-v30>','exec'),namespace)
    assert namespace['self_tests']()['passed']
    template=pd.read_csv(ROOT/'results/v29_kaggle_output/GLC25_PA_submission_v29.csv')
    species=np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    predictions=namespace['decode_control'](namespace['CONTROL_B64'],template,species)
    proof,decision=core.publish(tmp_path,template,template.surveyId.to_numpy(),predictions,species,True)
    assert proof['sha256']==core.CONTROL_HASH and not decision['eligible_for_submission']
    assert decision['prediction_file']=='unchanged_v29_DO_NOT_SUBMIT.csv'
    predictions[0][0]=next(i for i in range(5016) if i not in predictions[0])
    for gate in (False,True):
        output=tmp_path/str(gate); output.mkdir()
        _,decision=core.publish(output,template,template.surveyId.to_numpy(),predictions,species,gate)
        assert decision['eligible_for_submission']==gate
        assert decision['prediction_file']==('GLC25_PA_submission_v30.csv' if gate else 'candidate_DO_NOT_SUBMIT.csv')
    with pytest.raises(ValueError,match='payload mismatch'):
        namespace['decode_control']('',template,species)


def test_cpu_gpu_preflight_stops_immediately(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core.torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(core.legacy,'prepare_feature_store',lambda *a,**k:pytest.fail('GPU preflight too late'))
    with pytest.raises(RuntimeError,match='GPU'):
        core.run_v30('')
    assert not (tmp_path/'artifacts/v30_runtime').exists()
    assert json.loads((tmp_path/'artifacts/v30_export/failure_report.json').read_text())['status']=='failed'


def test_production_admission_precedes_training(sensors,tmp_path,monkeypatch):
    store,rows=sensors
    record={'best_epoch':10,'history':[{'seconds':100}]}
    bundles=[{'records':[record]*4,'split':{'training':np.arange(16)}}]*2
    class Expired:
        def require(self, seconds, stage):
            assert seconds > 2700 and 'whole' in stage
            raise TimeoutError('refit rejected before fitting')
    monkeypatch.setattr(core,'train_model',lambda *a,**k:pytest.fail('Admission was too late'))
    with pytest.raises(TimeoutError,match='before fitting'):
        core.fit_production(bundles,{'weights':'balanced'},rows,rows,store,[],
                            {'training':np.arange(24),'anchor':np.arange(24,32)},tmp_path,Expired(),torch.device('cpu'))


def test_archive_hashes_and_v29_gate_are_preserved():
    directory=ROOT/'results/v29_kaggle_output'
    manifest=json.loads((directory/'v29_manifest.json').read_text())
    for name,digest in manifest['outputs'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    report=json.loads((directory/'v29_report.json').read_text())
    assert report['submission_gate']['eligible_for_submission']
    assert report['submission']['sha256']==core.CONTROL_HASH
    assert report['hardware']['gpu']=='Tesla T4'


def test_end_to_end_miniature_with_actual_fits_and_production_anchors(sensors,tmp_path,monkeypatch):
    store,rows=sensors
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core.os,'cpu_count',lambda:2)
    store.labels=np.pad(store.labels,((0,0),(0,5016-12)))
    store.species_ids=np.arange(5016)
    store.cache=tmp_path/'sensors'; store.cache.mkdir()
    np.save(store.cache/'train_sentinel64.npy',store.high_train)
    np.save(store.cache/'test_sentinel64.npy',store.high_test)
    data=tmp_path/'official'; data.mkdir()
    rows.to_csv(data/'GLC25_PA_metadata_train.csv',index=False)
    template=pd.DataFrame({'surveyId':rows.surveyId,'predictions':['']*32})
    template.to_csv(data/'GLC25_SAMPLE_SUBMISSION.csv',index=False)
    monkeypatch.setattr(core.legacy,'EXPECTED_TEST_ROWS',32)
    base=[list(range(100,108)) for _ in range(32)]
    proof=core.legacy.write_submission(tmp_path/'control.csv',template,store.test_ids,base,store.species_ids)
    monkeypatch.setattr(core,'CONTROL_HASH',proof['sha256'])
    splits=[{'training':np.arange(16),'selection':np.arange(16,20),'calibration':np.arange(20,24),
             'assessment':np.arange(24+4*f,28+4*f)} for f in range(2)]
    prod={'training':np.arange(24),'anchor':np.arange(24,32)}
    monkeypatch.setattr(core,'make_splits',lambda *a,**k:(splits,[{'minimum_distance_km':21.}]*2))
    monkeypatch.setattr(core,'production_split',lambda *a,**k:(prod,{'minimum_distance_km':21.}))
    monkeypatch.setattr(core.legacy,'require_gpu',lambda:torch.device('cpu'))
    monkeypatch.setattr(core.legacy,'discover_data_root',lambda:data)
    monkeypatch.setattr(core,'decode_control',lambda *a:base)
    monkeypatch.setattr(core.legacy,'prepare_feature_store',lambda *a,**k:{})
    monkeypatch.setattr(core.legacy,'FeatureStore',lambda p:store)
    monkeypatch.setattr(core.legacy,'load_rows_and_pairs',lambda *a:(rows,rows,None))
    monkeypatch.setattr(core,'CONFIGS',tuple({**c,'width':16,'epochs':1} for c in core.CONFIGS))
    monkeypatch.setattr(core,'fold_reference',lambda arrays:[list(range(100,108)) for _ in arrays[0]])
    for name in ('V30_SOURCE_HASH','V29_SOURCE_HASH','LEGACY_SOURCE_HASH'):
        monkeypatch.setattr(core,name,'synthetic',raising=False)
    real_train=core.train_model
    production_indices=[]
    def checked_train(*args,**kwargs):
        if kwargs.get('phase')=='production':
            production_indices.append(args[2].copy())
            assert not np.intersect1d(args[2],prod['anchor']).size
            assert len(args[3])==0  # No checkpoint selection against anchor labels.
        return real_train(*args,**kwargs)
    monkeypatch.setattr(core,'train_model',checked_train)
    result=core.run_v30('')
    assert result['status']=='complete' and not result['fresh_assessment']
    assert result['eligible_for_submission'] and len(production_indices)>=2
    export=tmp_path/'artifacts/v30_export'
    assert len(list(export.iterdir()))==5 and result['output_bytes']<16_000_000
    report=json.loads((export/'v30_report.json').read_text())
    assert report['production_diagnostics']['anchor_not_in_weight_training']
    assert all(not r['calibration_transferred_from_development'] for r in report['training']['production'])
    assert not (tmp_path/'artifacts/v30_runtime').exists()
    with np.load(export/'calibration_top64_v30.npz',allow_pickle=False) as evidence:
        assert evidence['ranked_species_columns'].shape==(8,4,64)
        assert evidence['true_offsets'][-1]==len(evidence['true_species_columns'])
        assert set(evidence['survey_id'])==set(rows.iloc[splits[0]['calibration']].surveyId)
    manifest=json.loads((export/'v30_manifest.json').read_text())
    for name,digest in manifest['outputs'].items():
        assert hashlib.sha256((export/name).read_bytes()).hexdigest()==digest
