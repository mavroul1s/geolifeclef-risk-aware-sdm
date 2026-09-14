import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import scripts.run_retained_po as pipeline
from scripts.train_retained_po import pretrain,fit_arm
from scripts.prepare_po_expert import coordinate_features


def test_real_cpu_orchestration_freezes_three_arm_policies_and_predictions(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    rng=np.random.default_rng(7); n=48; species=24
    rows=pd.DataFrame({'lat':rng.uniform(40,42,n),'lon':rng.uniform(3,5,n),'surveyId':np.arange(n)})
    test=rows.iloc[:4].copy()
    labels=(rng.random((n,species))<.2).astype(np.uint8); labels[:,0]=1
    pool={'pa':rng.normal(size=(n,2)).astype(np.float32),'test':rng.normal(size=(4,2)).astype(np.float32),
          'po':rng.normal(size=(n,2)).astype(np.float32),'geo':coordinate_features(rows[['lat','lon']].to_numpy()),
          'labels':sparse.csr_matrix(labels),'weights':np.ones(n)}
    split=np.array([0]*30+[1]*6+[2]*6+[3]*6)
    monkeypatch.setattr(pipeline,'pretrain',lambda *a,**k:pretrain(*a,**k,epochs=1,draws=32,width=8))
    monkeypatch.setattr(pipeline,'fit_arm',lambda *a,**k:fit_arm(*a,**k,epochs=2,minimum_epochs=1,patience=1))
    base=np.full((6,species),.1,dtype=np.float16)
    policies,history=pipeline.fit_new_arms(rows,test,labels,split,np.full(n,30.),pool,base,base,
        tmp_path,torch.device('cpu'),time.monotonic()+7200)
    assert set(policies)=={'retained_po','single_head','zero_po'}
    for arm in policies:
        assert len(policies[arm]['trials'])==14
        assert len(policies[arm]['checkpoint_sha256'])==64
        assert np.load(tmp_path/f'{arm}_assessment.npy').shape==(6,species)
        assert np.load(tmp_path/f'{arm}_calibration.npy').shape==(6,species)
    assert history['pretraining']['probe_used_for_selection'] is False
