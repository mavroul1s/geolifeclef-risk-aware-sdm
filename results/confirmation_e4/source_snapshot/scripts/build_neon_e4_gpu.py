"""Embed outcome-free source/NEON covariates for one private inference job."""
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data/confirmation_e4';path=ROOT/'notebooks/neon_e4_inference.ipynb'
    body=(data/'input_bundle.npz').read_bytes();encoded=base64.b64encode(body).decode()
    index=json.loads((ROOT/'data/neon_bridge/checkpoint_index.json').read_text())
    member=next(m for m in index['members'] if m['name']=='aggregation_s0/checkpoints/best.ckpt')
    record=json.loads((ROOT/'data/external_replication_e1/zenodo_21335338.json').read_text())
    url=next(f for f in record['files'] if f['key']=='checkpoints.tar')['links']['self']
    config=json.loads((ROOT/'data/neon_bridge/model_configs.json').read_text())['aggregation_s0']
    net={'model':{'init_args':{'net':config['model']['init_args']['net']}}}
    code='''import base64, hashlib, io, json, signal, time, urllib.request
from pathlib import Path
import numpy as np
import torch
started=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('E4 20-minute budget')))
signal.alarm(1200)
assert torch.cuda.is_available(), 'Actual CUDA required'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
'''
    code+=(ROOT/'src/geolifeclef/sage_inference.py').read_text()+'\n'
    code+=f"payload=base64.b64decode({encoded!r})\nassert hashlib.sha256(payload).hexdigest()=={hashlib.sha256(body).hexdigest()!r}\n"
    code+=f"inputs=np.load(io.BytesIO(payload))\nmember={member!r}\nurl={url!r}\nconfig=json.loads({json.dumps(net)!r})\n"
    expected=json.loads((ROOT/'results/neon_bridge/cpu_reference.json').read_text())['runs'][0]['checkpoint_sha256']
    code+=f"expected_checkpoint={expected!r}\n"
    code+='''path=Path('/kaggle/working/reference_model.ckpt');digest=hashlib.sha256()
with path.open('wb') as stream:
 for offset in range(0,member['size'],8*1024**2):
  count=min(8*1024**2,member['size']-offset);left=member['data_offset']+offset
  request=urllib.request.Request(url,headers={'Range':f'bytes={left}-{left+count-1}'})
  with urllib.request.urlopen(request,timeout=40) as response:
   assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {left}-{left+count-1}/')
   chunk=response.read(count+1)
  assert len(chunk)==count
  stream.write(chunk);digest.update(chunk)
assert digest.hexdigest()==expected_checkpoint
model=load_network(path,config).to('cuda').eval()
outputs={}
for role in ['reference','source','neon']:
 x=normalize(inputs[role],inputs['means'],inputs['stds']).to('cuda')
 with torch.inference_mode():
  prediction=torch.cat([torch.sigmoid(model(part)).cpu() for part in x.split(256)]).numpy()
 assert np.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()
 outputs[role]=prediction
np.savez_compressed('/kaggle/working/e4_probabilities.npz',**outputs)
report={'status':'inference_complete_pending_local_reference_verification','GPU':torch.cuda.get_device_name(),
        'torch':torch.__version__,'CUDA':torch.version.cuda,'elapsed_seconds':time.monotonic()-started,
        'input_bundle_sha256':hashlib.sha256(payload).hexdigest(),'checkpoint_sha256':digest.hexdigest(),
        'shapes':{k:list(v.shape) for k,v in outputs.items()},'training_performed':False,
        'NEON_credentials_or_labels_included':False,'competition_submission_made':False}
Path('/kaggle/working/e4_gpu_report.json').write_text(json.dumps(report,indent=2)+'\\n')
path.unlink();signal.alarm(0);print(json.dumps(report))
'''
    compile(code,'<E4 GPU inference>','exec')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'cell_type':'markdown','metadata':{},'source':['# E4 outcome-free inference\nOne fixed released MLP; no fitting, labels, API token or competition submission.']},
                       {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    text=json.dumps(notebook,indent=1)+'\n'
    key=json.loads((ROOT.parent/'kaggle_api_key/kaggle.json').read_text())['key'];token=(ROOT.parent/'neon_api_key/neon_api_key').read_text().strip()
    if key in text or token in text:raise ValueError('Credential in upload')
    path.write_text(text)
    print(json.dumps({'notebook':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'input_sha256':hashlib.sha256(body).hexdigest(),'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
