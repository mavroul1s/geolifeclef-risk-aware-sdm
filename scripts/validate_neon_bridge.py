"""Validate engineering/provenance evidence without opening NEON composition."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'data/neon_bridge', ROOT / 'results/neon_bridge'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    previous = {}
    for folder in ['external_replication_e1', 'ranking_study_e2', 'confirmation_e3', 'neon_confirmation_feasibility']:
        base = ROOT / 'results' / folder
        manifest = json.loads((base / 'artifact_manifest.json').read_text())
        for name, item in manifest['files'].items():
            path = base / name
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
        previous[folder] = len(manifest['files'])
    projection = json.loads((OUT / 'metadata_projection_freeze.json').read_text())['identity']
    for key, name in [('config_sha256', 'configs/neon_metadata_projection.json'),
                      ('script_sha256', 'scripts/retrieve_neon_survey_metadata.py'),
                      ('projection_source_sha256', 'src/geolifeclef/neon_metadata.py'),
                      ('manifest_sha256', 'data/neon_bridge/neon_2023_manifest.json')]:
        assert projection[key] == sha(ROOT / name), key
    config = json.loads((ROOT / 'configs/neon_metadata_projection.json').read_text())
    retrieval = json.loads((OUT / 'survey_metadata_retrieval.json').read_text())
    assert retrieval['file_count'] == 198 and not retrieval['composition_values_inspected']
    for item in retrieval['files']:
        path = ROOT / item['metadata_file']
        assert sha(path) == item['metadata_sha256']
        assert set(pd.read_csv(path, nrows=0).columns) <= set(config['allowed_fields'])
        assert not (path.parent / item['source_name']).exists()
    # Independent pandas set/count reconstruction of complete quality-controlled bouts.
    frames = []
    for item in retrieval['files']:
        frame = pd.read_csv(ROOT / item['metadata_file'], dtype=str, keep_default_na=False)
        frame['table'] = item['table']
        frames.append(frame)
    metadata = pd.concat(frames).drop_duplicates()
    complete = eligible = 0
    for _, frame in metadata.groupby(['siteID', 'plotID', 'boutNumber']):
        small = frame[frame.table.eq('div_1m2Data')].subplotID.nunique()
        large = frame[frame.table.eq('div_10m2Data100m2Data')]
        sizes = large.subplotID.str.split('_').str[1]
        full = small == 6 and large.loc[sizes.eq('10'), 'subplotID'].nunique() == 6 and large.loc[sizes.eq('100'), 'subplotID'].nunique() == 4
        good = frame.samplingImpractical.eq('OK').all() and frame.biophysicalCriteria.eq('OK - no known exceptions').all()
        complete += int(full)
        eligible += int(full and good)
    audit = json.loads((OUT / 'survey_metadata_audit.json').read_text())
    assert complete == audit['full_nested_unit_plot_bouts'] == 768
    assert eligible == audit['full_units_and_quality_plot_bouts'] == 555
    candidates = pd.read_csv(DATA / 'full_plot_metadata_candidates.csv')
    assert len(candidates) == 368 and candidates.plot_id.nunique() == 368 and candidates.site.nunique() == 16
    assert not candidates.same_grid_cell_any_SAGE_eval.any()
    assert candidates.nearest_consumed_SAGE_km.ge(100).all() and candidates.nearest_GLC_PA_km.ge(500).all()
    training = json.loads((OUT / 'training_overlap_audit.json').read_text())
    assert training['candidate_metadata_sha256'] == sha(DATA / 'full_plot_metadata_candidates.csv')
    assert training['training_coordinates_sha256'] == sha(DATA / 'train_location_1km.h5')
    overlap = pd.read_csv(DATA / 'candidate_training_overlap.csv')
    assert int(overlap.same_grid_any_published_PO_training_location.sum()) == training['plots_same_training_grid'] == 301
    # Brute-force haversine on five fixed rows independently checks tree/coordinate ordering.
    with h5py.File(DATA / 'train_location_1km.h5') as f:
        locations = f['predictor_data'][:]
    lon, lat = np.deg2rad(locations[:, 0].astype(float)), np.deg2rad(locations[:, 1].astype(float))
    for row in overlap.iloc[np.linspace(0, len(overlap) - 1, 5, dtype=int)].itertuples():
        rlat, rlon = np.deg2rad(row.lat), np.deg2rad(row.lon)
        a = np.sin((lat - rlat) / 2)**2 + np.cos(lat) * np.cos(rlat) * np.sin((lon - rlon) / 2)**2
        distance = float(2 * np.arcsin(np.sqrt(np.clip(a.min(), 0, 1))) * 6371.0088)
        assert abs(distance - row.nearest_any_training_location_km) < 1e-7
    cpu = json.loads((OUT / 'cpu_reference.json').read_text())
    assert cpu['status'] == 'failed'  # Preserve the failed two-model admission, not a universal pass.
    assert {r['run']: r['reference_passed'] for r in cpu['runs']} == {'aggregation_s0': True, 'optimized_deepsdm_s0': False}
    gpu = json.loads((OUT / 'gpu_reference_validation.json').read_text())
    assert gpu['status'] == 'passed' and len(gpu['runs']) == 1 and gpu['runs'][0]['run'] == 'aggregation_s0'
    mapping = json.loads((OUT / 'taxonomy_mapping.json').read_text())
    assert sha(DATA / 'neon_to_sage_taxonomy.csv') == mapping['mapping_sha256']
    assert mapping['distinct_SAGE_classes_matched'] == 2453
    raster = json.loads((OUT / 'raster_reference_probe.json').read_text())
    assert raster['all_three_sampled_columns_match'] and len(raster['results']) == 3
    suites = ET.parse(OUT / 'focused_tests.xml').getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites) == 21
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) == 0 for s in suites)
    key = json.loads((ROOT.parent / 'kaggle_api_key/kaggle.json').read_text())['key']
    token = (ROOT.parent / 'neon_api_key/neon_api_key').read_text().strip()
    scan = [ROOT / 'notebooks/sage_gpu_reference.ipynb']
    scan += [p for base in [ROOT / 'scripts', ROOT / 'src', ROOT / 'tests', ROOT / 'configs', ROOT / 'docs', OUT]
             for p in base.rglob('*') if p.is_file() and p.suffix in ['.py', '.json', '.md', '.xml']]
    for path in scan:
        body = path.read_bytes()
        assert key.encode() not in body and token.encode() not in body, 'Credential detected'
    result = {'status': 'engineering_and_provenance_audit_passed', 'focused_tests': 21,
              'prior_manifest_files_unchanged': previous, 'metadata_projection_freeze_verified': True,
              'independent_completeness_and_distance_checks_passed': True,
              'MLP_CPU_and_GPU_reference_passed': True, 'ResNet_CPU_reference_failure_preserved': True,
              'taxonomy_mapping_hash_verified': True, 'three_raster_column_checks_passed': True,
              'credentials_absent_from_code_reports_docs_and_upload': True,
              'NEON_composition_values_inspected': False, 'NEON_ecological_performance_experiment_run': False,
              'fresh_NEON_inference_admitted': False,
              'remaining': ['ResNet reference or prospective single-model scope', 'full 52-feature/native-mask bridge',
                            'outcome-free role/label/claim freeze with original-training overlap disclosed']}
    manifest_path = OUT / 'artifact_manifest.json'
    if args.freeze:
        if manifest_path.exists():
            raise ValueError('Refusing to replace an existing stage manifest')
        (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
        paths = [p for base in [DATA, OUT, ROOT / 'artifacts/neon_bridge_gpu'] for p in base.rglob('*') if p.is_file()]
        names = ['configs/neon_bridge.json', 'configs/neon_metadata_projection.json', 'docs/neon_bridge_status.md',
                 'notebooks/sage_gpu_reference.ipynb', 'scripts/kaggle_research_job.py',
                 'scripts/retrieve_neon_bridge.py', 'scripts/check_sage_reference.py', 'scripts/build_sage_gpu_reference.py',
                 'scripts/verify_sage_gpu_reference.py', 'scripts/retrieve_neon_taxonomy.py', 'scripts/map_neon_taxonomy.py',
                 'scripts/retrieve_neon_survey_metadata.py', 'scripts/audit_neon_survey_metadata.py',
                 'scripts/audit_neon_training_overlap.py', 'scripts/probe_sage_raster_reference.py',
                 'scripts/validate_neon_bridge.py']
        names += ['src/geolifeclef/' + n + '.py' for n in ['range_io', 'sage_inference', 'neon_metadata', 'neon_survey', 'taxonomy_bridge', 'raster_bridge']]
        names += ['tests/' + n + '.py' for n in ['test_sage_bridge', 'test_neon_metadata', 'test_taxonomy_bridge', 'test_raster_bridge']]
        paths += [ROOT / n for n in names]
        files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))}
        manifest_path.write_text(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                                             'scope': 'Local engineering-stage files; hashes do not establish publisher-level whole-TAR checksums.',
                                             'files': files}, indent=2) + '\n')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, item in manifest['files'].items():
            path = ROOT / name
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
        result['new_stage_manifest_files_verified'] = len(manifest['files'])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
