import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from geolifeclef.range_io import BoundedHTTPFile
from geolifeclef.sage_inference import SimpleMLP,ResNet,normalize,load_network


def fake_range(monkeypatch,payload,prefix=700):
    data=b'x'*prefix+payload+b'tail'
    class Response(io.BytesIO):
        status=206
        def __init__(self,body,left,right):
            super().__init__(body);self.headers={'Content-Range':f'bytes {left}-{right}/{len(data)}'}
    def open_request(request,timeout):
        left,right=map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
        return Response(data[left:right+1],left,right)
    monkeypatch.setattr('geolifeclef.range_io.urllib.request.urlopen',open_request)


def test_range_member_reads_cross_blocks_without_leaking_adjacent_archive_bytes(monkeypatch):
    payload=bytes(range(200));fake_range(monkeypatch,payload)
    with BoundedHTTPFile('https://example.test/archive',700,len(payload),budget=300,block_size=31) as f:
        f.seek(23);assert f.read(85)==payload[23:108]
        f.seek(-8,io.SEEK_END);assert f.read(100)==payload[-8:]
        assert f.read()==b'' and f.fetched_bytes<=300


def test_hdf5_reads_from_bounded_tar_member(monkeypatch,tmp_path):
    p=tmp_path/'values.h5';expected=np.arange(500).reshape(25,20)
    with h5py.File(p,'w') as f:f.create_dataset('probabilities',data=expected,chunks=(5,5),compression='gzip')
    payload=p.read_bytes();fake_range(monkeypatch,payload)
    with BoundedHTTPFile('https://example.test/archive',700,len(payload),budget=len(payload)+100,block_size=256) as stream:
        with h5py.File(stream,'r') as f:np.testing.assert_array_equal(f['probabilities'][:3],expected[:3])


def test_range_budget_stops_before_over_fetch(monkeypatch):
    fake_range(monkeypatch,b'abcde'*100)
    with BoundedHTTPFile('https://example.test/archive',700,500,budget=100,block_size=64) as f:
        assert len(f.read(64))==64
        with pytest.raises(OSError,match='budget'):f.read(64)
        assert f.fetched_bytes==64


def test_normalization_uses_published_mean_for_nan_and_rejects_infinity():
    result=normalize([[np.nan,6]],np.array([3,2]),np.array([1,2]))
    torch.testing.assert_close(result,torch.tensor([[0.,2.]]))
    with pytest.raises(ValueError):normalize([[np.inf,1]],[0,0],[1,1])


@pytest.mark.parametrize('kind',[SimpleMLP,ResNet])
def test_weight_only_checkpoint_reload_preserves_inference(kind,tmp_path):
    torch.manual_seed(3);model=kind(num_classes=7,hidden_dim=8,num_blocks=2).eval()
    path=tmp_path/'model.ckpt'
    torch.save({'state_dict':{'net.'+k:v for k,v in model.state_dict().items()}},path)
    config={'model':{'init_args':{'net':{'class_path':'models.'+kind.__name__,
        'init_args':{'num_classes':7,'hidden_dim':8,'num_blocks':2,'use_location':False}}}}}
    loaded=load_network(path,config);x=torch.randn(5,52)
    torch.testing.assert_close(loaded(x),model(x),rtol=0,atol=0)
