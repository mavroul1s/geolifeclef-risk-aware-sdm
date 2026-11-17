"""Restore the v24 test's documented v23 control from committed notebook bytes."""
import ast
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '9da01ce45a3478e8073cd93e22dbf69dde65def0f86b7ef2700e84630f2c30f8'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition-root', type=Path, help='Optional directory of extracted official competition CSVs')
    args = parser.parse_args()
    book = json.loads((ROOT/'notebooks/geolifeclef_v24_multimodal_rare_species_sdm.ipynb').read_text())
    encoded = None
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FROZEN_V23_PAYLOAD_B64' for t in node.targets):
                encoded = ast.literal_eval(node.value)
    if encoded is None:
        raise ValueError('Frozen control payload missing')
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(encoded))), mode='r:') as archive:
        # Read only this expected member; never extract arbitrary archive paths.
        body = archive.extractfile('GLC25_PA_submission_v23.csv').read()
    if hashlib.sha256(body).hexdigest() != EXPECTED or not body.endswith(b'\r\n'):
        raise ValueError('Frozen v23 control digest mismatch')
    destination = ROOT/'artifacts/manual_upload_v23_retry2/user_provided_v25_exports/GLC25_PA_submission_v23.csv'
    fixture = body[:-2]  # Original regression test covers a missing final CRLF.
    if destination.exists() and destination.read_bytes() != fixture:
        raise ValueError('Different local evidence exists; refusing overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(fixture)
    print('Restored verified v23 missing-final-CRLF test fixture; no predictions changed.')
    if args.competition_root is not None:
        import numpy as np
        import pandas as pd
        raw = ROOT/'artifacts/v20_frozen/raw'; raw.mkdir(parents=True, exist_ok=True)
        for name in ['GLC25_PA_metadata_train.csv', 'GLC25_SAMPLE_SUBMISSION.csv']:
            sources = list(args.competition_root.rglob(name))
            if len(sources) != 1:
                raise ValueError('Expected exactly one extracted '+name)
            source, target = sources[0], raw/name
            if target.exists() and hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise ValueError('Different local metadata exists; refusing overwrite: '+name)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        species = np.sort(pd.read_csv(raw/'GLC25_PA_metadata_train.csv', usecols=['speciesId']).speciesId.dropna().astype(np.int64).unique())
        if len(species) != 5016:
            raise ValueError('Unexpected official PA species vocabulary')
        vocabulary = ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy'
        if vocabulary.exists() and not np.array_equal(np.load(vocabulary, allow_pickle=False), species):
            raise ValueError('Different local species vocabulary exists; refusing overwrite')
        vocabulary.parent.mkdir(parents=True, exist_ok=True); np.save(vocabulary, species)
        print('Restored metadata paths and PA species vocabulary from supplied official files.')


if __name__ == '__main__':
    main()
