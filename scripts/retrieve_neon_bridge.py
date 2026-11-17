"""Bounded public SAGE artifacts for outcome-closed NEON/reference engineering."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

from retrieve_sage_e1 import fetch_member

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/neon_bridge'
OUT=ROOT/'results/neon_bridge'


def range_header(url,offset):
    request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+511}'})
    with urllib.request.urlopen(request,timeout=45) as response:
        if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-{offset+511}/'):
            raise ValueError('Exact range response required')
        body=response.read(513)
    if len(body)!=512:raise ValueError('Invalid TAR header length')
    return body


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['index','artifacts'])
    args=parser.parse_args();DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/neon_bridge.json').read_text())
    model_record=json.loads((ROOT/'data/external_replication_e1/zenodo_21335338.json').read_text())
    data_record=json.loads((ROOT/'data/external_replication_e1/zenodo_21297133.json').read_text())
    if args.phase=='index':
        archive=next(f for f in model_record['files'] if f['key']=='checkpoints.tar')
        wanted={m+'/checkpoints/best.ckpt' for m in config['model_runs']}
        offset=0;members=[];found=set();headers=0
        while offset+512<=archive['size'] and headers<config['maximum_checkpoint_headers']:
            body=range_header(archive['links']['self'],offset);headers+=1
            if body==b'\0'*512:break
            item=tarfile.TarInfo.frombuf(body,'utf-8','surrogateescape')
            if item.isfile():
                member={'name':item.name,'size':item.size,'data_offset':offset+512,
                    'header_sha256':hashlib.sha256(body).hexdigest()}
                members.append(member)
                if item.name in wanted:found.add(item.name)
            offset+=512+((item.size+511)//512)*512
            if headers%20==0:print(json.dumps({'headers':headers,'wanted_found':len(found)}),flush=True)
            if found==wanted:break
        if found!=wanted:raise ValueError('Required checkpoint names not found within header bound')
        result={'status':'targeted_header_index_completed','at_utc':datetime.now(timezone.utc).isoformat(),
            'archive':'checkpoints.tar','archive_declared_checksum':archive['checksum'],
            'whole_archive_checksum_recomputed':False,'header_bytes_downloaded':headers*512,
            'members':members,'wanted':sorted(wanted),'full_archive_index_completed':False}
        (DATA/'checkpoint_index.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'headers':headers,'checkpoints':[(m['name'],m['size']) for m in members if m['name'] in wanted]}))
        return
    index=json.loads((DATA/'checkpoint_index.json').read_text())
    old_index=json.loads((ROOT/'data/independent_data_audit/sage_sage_benchmark_data_tar_index.json').read_text())
    tasks=[]
    for run in config['model_runs']:
        name=run+'/checkpoints/best.ckpt';member=next(m for m in index['members'] if m['name']==name)
        tasks.append((model_record,'checkpoints.tar',member,DATA/'checkpoints'/run/'best.ckpt'))
    for group in ['chelsa','soilgrids','topography','human_footprint']:
        name=f'predictors/eval_{group}.h5';member=next(m for m in old_index['members'] if m['name']==name)
        tasks.append((data_record,'sage_benchmark_data.tar',member,DATA/'reference_predictors'/f'eval_{group}.h5'))
    source_item=next(f for f in model_record['files'] if f['key']=='sage-sdm-benchmark_code.zip')
    total=sum(t[2]['size'] for t in tasks)+source_item['size']
    if total>config['maximum_new_artifact_bytes']:raise ValueError('Artifact byte budget exceeded')
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda task:fetch_member(*task),tasks))
    code=DATA/'sage_published_source.zip'
    if not code.exists():
        with urllib.request.urlopen(source_item['links']['self'],timeout=60) as response:body=response.read(source_item['size']+1)
        if len(body)!=source_item['size'] or 'md5:'+hashlib.md5(body).hexdigest()!=source_item['checksum']:
            raise ValueError('Published code archive checksum mismatch')
        code.write_bytes(body)
    with code.open('rb') as stream:
        if 'md5:'+hashlib.file_digest(stream,'md5').hexdigest()!=source_item['checksum']:raise ValueError('Code checksum changed')
    wanted=['src/data_modules/datamodule.py','src/models/simple_mlp.py','src/models/resnet.py',
        'scripts/data_prep_scripts/geo_utils.py','scripts/data_prep_scripts/predictors.py']
    with zipfile.ZipFile(code) as archive:
        for name in wanted:
            matches=[n for n in archive.namelist() if n==name or n.endswith('/'+name)]
            if len(matches)!=1:raise ValueError('Ambiguous published source path: '+name)
            destination=DATA/'published_source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(matches[0]))
    results.append({'file':str(code.relative_to(ROOT)),'bytes':code.stat().st_size,
        'published_MD5_verified':True,'sha256':hashlib.sha256(code.read_bytes()).hexdigest()})
    (OUT/'artifact_retrieval.json').write_text(json.dumps({'status':'completed','files':results,
        'logical_bytes':total,'no_NEON_outcomes_requested':True},indent=2)+'\n')
    print(json.dumps({'files':len(results),'bytes':total}))


if __name__=='__main__':main()
