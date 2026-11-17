"""Bounded public SAGE retrieval. TAR members are range-fetched, never executed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/external_replication_e1'


def fetch_member(record, archive_name, member, destination):
    if destination.exists() and destination.stat().st_size == member['size']:
        with destination.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'file': str(destination.relative_to(ROOT)), 'bytes': destination.stat().st_size, 'sha256': digest, 'cached': True}
    item = next(f for f in record['files'] if f['key'] == archive_name)
    url, offset, size = item['links']['self'], member['data_offset'], member['size']
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset-512}-{offset-1}'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise ValueError('Range support required; bulk archive download refused')
        header = response.read(513)
    info = tarfile.TarInfo.frombuf(header, 'utf-8', 'surrogateescape')
    if info.name != member['name'] or info.size != size or not info.isfile():
        raise ValueError('Tar header differs from audited member inventory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.partial')
    current = temporary.stat().st_size if temporary.exists() else 0
    while current < size:
        last_error = None
        for attempt in range(3):
            current = temporary.stat().st_size if temporary.exists() else 0
            end = min(size, current+32*1024*1024)
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset+current}-{offset+end-1}'})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f'bytes {offset+current}-{offset+end-1}/'
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Incorrect range response')
                    with temporary.open('ab') as stream:
                        while chunk := response.read(1024*1024):
                            current += len(chunk)
                            if current > end:
                                raise ValueError('Member range size exceeded')
                            stream.write(chunk)
                if current != end:
                    raise OSError('Incomplete range response')
                last_error = None
                break
            except (OSError, TimeoutError) as error:
                last_error = error
                time.sleep(min(2**attempt, 5))
        if last_error is not None:
            raise last_error
    if current != size:
        raise ValueError('Incomplete member download')
    temporary.replace(destination)
    with destination.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'file': str(destination.relative_to(ROOT)), 'archive': archive_name, 'member': info.name,
            'bytes': size, 'sha256': digest, 'whole_archive_checksum_recomputed': False,
            'archive_declared_checksum': item['checksum'], 'range_and_tar_header_checked': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['metadata', 'predictions', 'targets'])
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    old = ROOT/'data/independent_data_audit'
    records = {number: json.loads((DATA/f'zenodo_{number}.json').read_text()) for number in [21297133, 21335338]}
    tasks = []
    if args.mode in ['metadata', 'predictions']:
        archive_name = 'sage_benchmark_data.tar' if args.mode == 'metadata' else 'predictions.tar'
        record = records[21297133 if args.mode == 'metadata' else 21335338]
        inventory = json.loads((old/('sage_sage_benchmark_data_tar_index.json' if args.mode == 'metadata' else 'sage_predictions_tar_index.json')).read_text())
        wanted = (['val_indices.npy', 'test_indices.npy', 'targets/species_properties_1km.csv',
                   'range_masks_1km/eval_range_mask.h5', 'predictors/eval_location.h5'] if args.mode == 'metadata'
                  else [m+'/test_predictions.h5' for m in config['models']])
        for name in wanted:
            member = next(m for m in inventory['members'] if m['name'] == name)
            tasks.append((record, archive_name, member, DATA/name))
        if sum(t[2]['size'] for t in tasks) > config['maximum_download_bytes']:
            raise ValueError('Declared transfer cap exceeded')
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda t: fetch_member(*t), tasks))
    else:
        item = next(f for f in records[21335338]['files'] if f['key'] == 'eval_test_targets.h5')
        destination = DATA/'sealed_eval_test_targets.h5'
        if not destination.exists():
            with urllib.request.urlopen(item['links']['self'], timeout=90) as response:
                data = response.read(item['size']+1)
            if len(data) != item['size'] or 'md5:'+hashlib.md5(data).hexdigest() != item['checksum']:
                raise ValueError('Target archive size/checksum mismatch')
            destination.write_bytes(data)
        with destination.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        results = [{'file': str(destination.relative_to(ROOT)), 'bytes': destination.stat().st_size,
                    'sha256': digest, 'target_values_inspected': False}]
    out = ROOT/'results/external_replication_e1';out.mkdir(parents=True, exist_ok=True)
    (out/f'retrieval_{args.mode}.json').write_text(json.dumps({'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': args.mode, 'files': results}, indent=2)+'\n')
    print(json.dumps({'mode': args.mode, 'files': len(results), 'bytes': sum(r['bytes'] for r in results)}))


if __name__ == '__main__':
    main()
