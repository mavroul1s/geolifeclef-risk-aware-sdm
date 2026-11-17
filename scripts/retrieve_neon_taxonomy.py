"""Retrieve the public taxonomic dictionary, not site-level species observations."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/neon_bridge/neon_taxonomy'


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    initial = json.loads((DATA.parent / 'neon_taxonomy_compact_0.json').read_text())
    total, step = initial['total'], initial['count']
    if not 0 < step <= 1000 or total > 150000:
        raise ValueError('Unexpected taxonomy pagination')
    token = (ROOT.parent / 'neon_api_key/neon_api_key').read_text().strip()
    lock, byte_count = threading.Lock(), [0]

    def fetch(offset):
        path = DATA / f'offset_{offset:06d}.json'
        if path.exists():
            body = path.read_bytes()
        else:
            url = f'https://data.neonscience.org/api/v0/taxonomy?taxonTypeCode=PLANT&offset={offset}&limit={step}&verbose=false'
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url, headers={'X-API-Token': token})
                    with urllib.request.urlopen(request, timeout=40) as response:
                        body = response.read(1_500_001)
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in [429, 500, 502, 503, 504] or attempt == 3:
                        raise RuntimeError(f'Taxonomy HTTP status {error.code}') from None
                    time.sleep(min(10, 2 ** (attempt + 1)))
            if len(body) > 1_500_000:
                raise ValueError('Taxonomy page exceeds cap')
            obj = json.loads(body)
            if obj['total'] != total or obj['count'] != min(step, total - offset):
                raise ValueError('Taxonomy changed or pagination incomplete')
            path.write_bytes(body)
            time.sleep(.5)
        with lock:
            byte_count[0] += len(body)
            if byte_count[0] > 80_000_000:
                raise ValueError('Taxonomy total byte cap exceeded')
        return {'file': str(path.relative_to(ROOT)), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}

    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(fetch, range(0, total, step)))
    result = {'status': 'complete', 'expected_dictionary_entries': total, 'pages': len(files),
              'bytes': byte_count[0], 'files': files, 'site_observations_requested': False,
              'warning': 'Dictionary coverage is not observed-species or survey coverage.'}
    (ROOT / 'results/neon_bridge/taxonomy_retrieval.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    main()
