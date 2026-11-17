"""Bounded official NEON metadata audit; plant observation outcomes stay closed."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT=Path(__file__).resolve().parents[1]


def projected_site(code):
    if not re.fullmatch('[A-Z0-9]{4}',code):raise ValueError('Unexpected site identifier')
    url='https://data.neonscience.org/api/v0/sites/'+code
    required={'siteCode','siteName','siteLatitude','siteLongitude'}
    with urllib.request.urlopen(url,timeout=40) as response:
        body=b''
        for _ in range(16):
            body+=response.read(4096)
            text=body.decode('utf-8',errors='ignore')
            start=re.match(r'^\s*\{\s*"data"\s*:\s*\{',text)
            if not start:continue
            position=start.end();result={};decoder=json.JSONDecoder()
            try:
                while position<len(text):
                    while text[position] in ' \r\n\t,':position+=1
                    key,position=decoder.raw_decode(text,position)
                    while text[position].isspace():position+=1
                    if text[position]!=':':raise ValueError('Invalid site JSON')
                    position+=1
                    while text[position].isspace():position+=1
                    value,position=decoder.raw_decode(text,position)
                    result[key]=value
                    if required.issubset(result):
                        answer={key:result[key] for key in sorted(required)}
                        if answer['siteCode']!=code:raise ValueError('Site identity changed')
                        return {**answer,'metadata_bytes_read':len(body),'source_url':url}
            except (json.JSONDecodeError,IndexError):pass
    raise ValueError('Site coordinate projection unavailable within 64 KiB')


def main():
    data=ROOT/'data/neon_confirmation_feasibility';out=ROOT/'results/neon_confirmation_feasibility'
    out.mkdir(parents=True,exist_ok=True)
    product_file=data/'plant_product.json'
    product=json.loads(product_file.read_text())['data']
    with ThreadPoolExecutor(max_workers=4) as pool:
        sites=list(pool.map(projected_site,[s['siteCode'] for s in product['siteCodes']]))
    frame=pd.DataFrame(sites)
    with h5py.File(ROOT/'data/external_replication_e1/predictors/eval_location.h5') as f:
        coordinates=f['predictor_data'][:]
    nearest=BallTree(np.deg2rad(coordinates[:,[1,0]]),metric='haversine').query(
        np.deg2rad(frame[['siteLatitude','siteLongitude']]))[0][:,0]*6371.0088
    consumed=pd.read_csv(ROOT/'data/external_replication_e1/cohort_roles.csv')[['lat','lon']]
    e3=pd.read_csv(ROOT/'data/confirmation_e3/locked_cohort.csv')[['lat','lon']]
    prior=pd.concat([consumed,e3])
    frame['nearest_published_SAGE_eval_centroid_km']=nearest
    frame['nearest_consumed_SAGE_cell_km']=BallTree(np.deg2rad(prior),metric='haversine').query(
        np.deg2rad(frame[['siteLatitude','siteLongitude']]))[0][:,0]*6371.0088
    frame.to_csv(data/'site_metadata_projection.csv',index=False)
    released=next(r for r in product['releases'] if r['release']=='RELEASE-2026')
    result={'status':'official_metadata_feasibility_completed','at_utc':datetime.now(timezone.utc).isoformat(),
        'product':product['productCode'],'product_name':product['productName'],'sites':len(frame),
        'release':released,'product_metadata_sha256':hashlib.sha256(product_file.read_bytes()).hexdigest(),
        'site_projection_sha256':hashlib.sha256((data/'site_metadata_projection.csv').read_bytes()).hexdigest(),
        'projected_site_metadata_bytes_read':int(frame.metadata_bytes_read.sum()),
        'site_centroids_within_1km_any_published_SAGE_cell':int((nearest<=1).sum()),
        'site_centroids_within_10km_any_published_SAGE_cell':int((nearest<=10).sum()),
        'site_centroids_at_least_100km_from_consumed_SAGE':int(frame.nearest_consumed_SAGE_cell_km.ge(100).sum()),
        'plant_observation_outcomes_downloaded_or_inspected':False,
        'no_plot_level_overlap_or_completeness_claim':True,
        'prediction_compatibility_established':False,
        'next_requirements':['Plot-level coordinates/ID/date and completeness audit using non-composition metadata',
            'Respect full-plot versus 1 m2-only surveys and repeated visits',
            'Versioned taxonomy mapping to the public SAGE class list; do not identify anonymous GLC taxa',
            'Extract matching 52 environmental predictors and reuse versioned model/preprocessing artifacts',
            'Lock source/site roles, quality and service hypotheses before NEON composition values'],
        'download_note':'full all-sites API exceeds metadata cap; used bounded <=64 KiB prefix projections per site instead'}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
