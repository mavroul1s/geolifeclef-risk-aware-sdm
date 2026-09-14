"""Single master v22 launcher with an end-to-end 10.5-hour parent timeout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    os.environ.setdefault('GLC_PIPELINE_STARTED_AT',str(time.time()))
    os.environ['PYTHONUNBUFFERED']='1'
    started=float(os.environ['GLC_PIPELINE_STARTED_AT'])
    def remaining():
        seconds=10.5*3600-(time.time()-started)
        if seconds<=0: raise TimeoutError('Total v22 budget exceeded')
        return seconds
    root=Path('/kaggle/input')
    data=next((p for p in (root/'competitions/geolifeclef-2025',root/'geolifeclef-2025') if p.is_dir()),None)
    if data is None: raise FileNotFoundError('Competition input missing')
    old=[p.parent for p in root.rglob('provenance.json') if 'geolifeclef-v20-frozen-control' in str(p)]
    new=[p.parent for p in root.rglob('v21_provenance.json') if 'geolifeclef-v21-frozen-control' in str(p)]
    if len(old)!=1 or len(new)!=1: raise ValueError('Exactly one pinned v20 and v21 frozen input required')
    subprocess.run([sys.executable,'-m','pytest','-q'],check=True,timeout=remaining())
    subprocess.run([sys.executable,'-m','scripts.run_retained_po','--data-root',str(data),
        '--frozen-v20-dir',str(old[0]),'--frozen-v21-dir',str(new[0])],check=True,timeout=remaining())
    subprocess.run([sys.executable,'-m','pytest','-q'],check=True,timeout=remaining())
    path=Path('artifacts/retained_po_v22/v22_report.json')
    report=json.loads(path.read_text())
    remaining()
    report['total_pipeline_hours']=(time.time()-started)/3600
    report['notebook_tests_before_and_after_passed']=True
    report['source_commit']=os.environ.get('GLC_SOURCE_COMMIT','local')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'v22_complete':True,'hours':report['total_pipeline_hours'],
        'official_submission_allowed':report['submission_gate']['eligible_for_official_submission']}))


if __name__=='__main__': main()
