import time
import numpy as np
import torch
from scipy import sparse
from scripts.train_retained_po import pretrain,fit_arm,predict,po_indices


def test_real_tiny_training_retains_shared_initialization_and_selects_complement(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(1)
    features=rng.normal(size=(48,6)).astype(np.float32)
    labels=(rng.random((48,24))<.2).astype(np.uint8)
    labels[:,0]=1
    po_labels=sparse.csr_matrix(labels)
    train,probe=po_indices(48)
    assert not set(train)&set(probe)
    device=torch.device('cpu'); deadline=time.monotonic()+90
    initial,diagnostic=pretrain(features,po_labels,np.ones(48),device,tmp_path,deadline,epochs=1,draws=32,width=8)
    state={k:v.clone() for k,v in initial.state_dict().items()}
    selection=np.arange(32,40); base=np.full((8,24),.1,dtype=np.float16)
    for arm in ('retained_po','single_head','zero_po'):
        model,report=fit_arm(initial,arm,features,labels,np.arange(32),selection,base,np.full(48,30.),
            features,po_labels,np.ones(48),device,tmp_path,deadline,epochs=2,minimum_epochs=1,patience=1)
        assert report['selection_policy']['alpha']==.1
        saved=torch.load(tmp_path/f'{arm}_best.pt',weights_only=True)
        assert saved['selection_mixture_f1']==report['selection_mixture_f1']
        assert predict(model,features[:2],device,deadline).shape==(2,24)
        for name,value in initial.state_dict().items():
            torch.testing.assert_close(value,state[name],rtol=0,atol=0)
    assert diagnostic['probe_used_for_selection'] is False
