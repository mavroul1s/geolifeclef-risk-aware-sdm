"""Project NEON survey metadata while discarding composition field bytes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geolifeclef.neon_metadata import project_csv_bytes, metadata_csv_bytes


def main():
    data, out = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'
    config_path = ROOT / 'configs/neon_metadata_projection.json'
    config = json.loads(config_path.read_text())
    manifest = json.loads((data / 'neon_2023_manifest.json').read_text())['data']
    releases = manifest['releases']
    if len(releases) != 1 or releases[0]['release'] != config['release']:
        raise ValueError('Pinned release mismatch')
    files = []
    for package in releases[0]['packages']:
        if not package['month'].startswith(str(config['year']) + '-') or package['packageType'] != 'basic':
            raise ValueError('Year/package mismatch')
        for file in package['files']:
            tables = [t for t in config['allowed_tables'] if '.' + t + '.' in file['name']]
            if tables:
                files.append({**file, 'table': tables[0], 'site': package['siteCode'], 'month': package['month']})
    if sum(f['size'] for f in files) > config['raw_transport_byte_limit']:
        raise ValueError('Metadata-projection transport budget exceeded')
    freeze_path = out / 'metadata_projection_freeze.json'
    identity = {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'projection_source_sha256': hashlib.sha256((ROOT / 'src/geolifeclef/neon_metadata.py').read_bytes()).hexdigest(),
                'manifest_sha256': hashlib.sha256((data / 'neon_2023_manifest.json').read_bytes()).hexdigest()}
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Metadata projection freeze changed')
    else:
        freeze_path.write_text(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                                         'identity': identity, 'composition_fields_closed': True}, indent=2) + '\n')
    destination = data / 'survey_metadata'
    destination.mkdir(exist_ok=True)

    def project(file):
        path = destination / (file['name'] + '.metadata.csv')
        receipt = path.with_suffix('.receipt.json')
        if path.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            if record['metadata_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Projected metadata digest changed')
            return record
        parsed = urlsplit(file['url'])
        if parsed.scheme != 'https' or parsed.hostname != 'storage.googleapis.com':
            raise ValueError('Unexpected NEON-manifest storage host')
        if file['size'] > config['raw_file_byte_limit']:
            raise ValueError('Raw transport file exceeds cap')
        for attempt in range(4):
            try:
                # Storage URLs from the manifest never receive account credentials.
                with urllib.request.urlopen(file['url'], timeout=40) as response:
                    body = response.read(file['size'] + 1)
                break
            except urllib.error.HTTPError as error:
                if error.code not in [429, 500, 502, 503, 504] or attempt == 3:
                    raise RuntimeError(f'Storage HTTP status {error.code}') from None
                time.sleep(min(8, 2 ** (attempt + 1)))
        if len(body) != file['size'] or hashlib.md5(body).hexdigest() != file['md5']:
            raise ValueError('Published NEON size/MD5 mismatch')
        columns, rows = project_csv_bytes(body, config['allowed_fields'])
        del body
        required = {'siteID', 'plotID', 'subplotID', 'boutNumber', 'decimalLatitude', 'decimalLongitude',
                    'samplingImpractical', 'biophysicalCriteria'}
        if not required.issubset(columns):
            raise ValueError('Missing survey metadata contract')
        if any(row[columns.index('siteID')] != file['site'] for row in rows):
            raise ValueError('Manifest/site mismatch')
        metadata = metadata_csv_bytes(columns, rows)
        path.write_bytes(metadata)
        record = {'metadata_file': str(path.relative_to(ROOT)), 'source_name': file['name'],
                  'source_MD5_verified': file['md5'], 'raw_bytes_transported': file['size'],
                  'metadata_sha256': hashlib.sha256(metadata).hexdigest(), 'metadata_bytes': len(metadata),
                  'table': file['table'], 'site': file['site'], 'month': file['month'],
                  'raw_file_retained': False, 'composition_values_decoded_or_retained': False}
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = []
        for record in pool.map(project, files):
            reports.append(record)
            if len(reports) % 25 == 0:
                print(json.dumps({'projected_files': len(reports), 'total_files': len(files)}), flush=True)
    result = {'status': 'complete', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'file_count': len(reports), 'raw_transport_bytes': sum(r['raw_bytes_transported'] for r in reports),
              'retained_metadata_bytes': sum(r['metadata_bytes'] for r in reports),
              'raw_observation_files_retained': False, 'composition_values_inspected': False,
              'blinding_scope': config['transport_note'], 'files': reports}
    (out / 'survey_metadata_retrieval.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    main()
