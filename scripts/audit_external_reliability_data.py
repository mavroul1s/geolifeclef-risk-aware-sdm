"""Covariate-only SAGE/sPlotOpen feasibility and overlap audit; no species labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT/'data/independent_data_audit'
    output = ROOT/'results/reliability_followup_d1_d3/independent_data'
    output.mkdir(parents=True, exist_ok=True)
    record = json.loads((data/'sage_zenodo_record.json').read_text())
    source = data/'sage_eval_plot_metadata.csv'
    expected = next(f for f in record['files'] if f['key'] == 'eval_plot_metadata.csv')
    if 'md5:'+hashlib.md5(source.read_bytes()).hexdigest() != expected['checksum']:
        raise ValueError('Public survey metadata checksum differs from Zenodo')
    # Explicit projection excludes species lists, richness, covers and functional traits.
    cols = ['cell_row', 'PlotObservationID', 'GIVD_ID', 'Dataset', 'Continent', 'Country',
            'Date_of_recording', 'Latitude', 'Longitude', 'Location_uncertainty', 'Releve_area']
    frame = pd.read_csv(source, usecols=cols)
    train = pd.read_csv(ROOT/'data/raw/GLC25_PA_metadata_train.csv', usecols=['surveyId', 'lat', 'lon']).drop_duplicates('surveyId')
    test = pd.read_csv(ROOT/'data/raw/GLC25_PA_metadata_test.csv', usecols=['surveyId', 'lat', 'lon']).drop_duplicates('surveyId')
    reference = pd.concat([train, test])
    finite = np.isfinite(frame[['Latitude', 'Longitude']]).all(axis=1)
    distance = np.full(len(frame), np.nan)
    tree = BallTree(np.deg2rad(reference[['lat', 'lon']].to_numpy()), metric='haversine')
    distance[finite] = tree.query(np.deg2rad(frame.loc[finite, ['Latitude', 'Longitude']].to_numpy()), k=1)[0][:, 0]*6371.0088
    frame['nearest_existing_PA_km'] = distance
    known_non_europe = frame.Continent.notna() & frame.Continent.ne('Europe')
    candidate = (known_non_europe & finite & (distance >= 500) &
        frame.Location_uncertainty.between(0, 1000) & frame.Releve_area.between(10, 400) &
        frame.GIVD_ID.notna() & ~frame.GIVD_ID.astype(str).str.startswith('EU-'))
    # Existing target rows aggregate multiple plots. Keep a cell only if every contributor qualifies.
    all_contributors = candidate.groupby(frame.cell_row).all()
    cells = all_contributors[all_contributors].index.to_numpy(np.int64)
    kept = frame[frame.cell_row.isin(cells)]
    (data/'candidate_SAGE_cells_metadata_only.txt').write_text('\n'.join(map(str, np.sort(cells)))+'\n')
    metadata_html = (data/'splotopen_metadata.html').read_text()
    license_match = re.search(r'Name="Data access policy"[^>]*value="([^"]+)"', metadata_html)
    page = (data/'sage_zenodo_page.html').read_text() if (data/'sage_zenodo_page.html').exists() else ''
    license_evidence = []
    for pattern in ['creativecommons.org/licenses', 'license', 'License']:
        location = page.find(pattern)
        if location >= 0:
            license_evidence.append(re.sub('<[^>]+>', ' ', page[max(0, location-80):location+260]))
    cohort = {'definition': 'all contributing plots: known non-Europe continent, finite coordinates, >=500 km from every existing GLC25 PA train/test location, recorded positional uncertainty <=1 km, area 10..400 m2, known non-EU-prefixed source',
        'unique_candidate_cells': len(cells), 'source_plot_rows': len(kept),
        'unique_plot_observation_IDs': int(kept.PlotObservationID.nunique()),
        'unique_source_groups': int(kept.GIVD_ID.nunique()),
        'countries': kept.groupby('Country').cell_row.nunique().to_dict(),
        'continents': kept.groupby('Continent').cell_row.nunique().to_dict(),
        'cell_IDs_sha256': hashlib.sha256(np.sort(cells).astype('<i8').tobytes()).hexdigest(),
        'minimum_distance_to_existing_PA_km': float(kept.nearest_existing_PA_km.min()) if len(kept) else None,
        'final_confirmation_cohort_locked': False}
    report = {'status': 'completed_covariate_feasibility_audit', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'species_outcomes_downloaded_or_inspected': False, 'model_predictions_downloaded_or_inspected': False,
        'metadata_columns_read': cols, 'metadata_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'metadata_source': 'https://zenodo.org/records/21297133',
        'metadata_rows': len(frame), 'evaluation_cells': int(frame.cell_row.nunique()),
        'unique_plot_observation_IDs': int(frame.PlotObservationID.nunique()),
        'missing_plot_observation_ID_rows': int(frame.PlotObservationID.isna().sum()),
        'duplicate_plot_ID_rows': int(frame.PlotObservationID.duplicated().sum()),
        'known_non_Europe_cells': int(frame.loc[known_non_europe, 'cell_row'].nunique()),
        'metadata_rows_with_unknown_continent': int(frame.Continent.isna().sum()),
        'all_metadata_rows_within_1km_of_existing_GLC_PA': int(np.sum(distance <= 1)),
        'all_metadata_rows_within_20km_of_existing_GLC_PA': int(np.sum(distance <= 20)),
        'candidate_cohort': cohort,
        'splotopen_access_policy': license_match.group(1) if license_match else None,
        'SAGE_API_access_right': record['metadata'].get('access_right'),
        'SAGE_API_license_field': record['metadata'].get('license'),
        'SAGE_HTML_license_evidence': license_evidence,
        'raw_splotopen_download_status': 'individual metadata links return HTTP 200 with empty bodies; archive HEAD timed out; empty files are not counted as retrieved data',
        'released_model_inventory': [(f['key'], f['size']) for f in json.loads((data/'sage_models_zenodo_record.json').read_text())['files']],
        'novelty_relevant_primary_source': 'https://arxiv.org/abs/2609.31082',
        'independent_confirmation_completed': False,
        'remaining_requirements': ['Verify original source/completeness fields and predictor/target join semantics',
            'Respect SAGE -1 out-of-native-range targets; never treat them as observed absences',
            'SAGE aggregates plots within 1 km and collapses time, unlike GLC25 plot-level predictions',
            'No Sentinel/Landsat inputs in the released 52-feature tabular external design; use compatible model families',
            'Specify calibration, selector fitting, geographic evaluation, species support and final source-cohort protocol before opening labels',
            'Audit potential PO/source overlap for any future PO-trained system; this distance check covers existing PA train/test only',
            'Code MIT and upstream sPlotOpen CC BY 4.0 are distinct from any missing downstream archive license field']}
    report['archive_header_only_inventory'] = {}
    for name in ['sage_sage_benchmark_data_tar_index.json', 'sage_predictions_tar_index.json']:
        if not (data/name).is_file():
            continue
        index = json.loads((data/name).read_text())
        report['archive_header_only_inventory'][index['file']] = {
            'status': index['status'], 'members': len(index['members']),
            'bytes_downloaded': index['header_bytes_downloaded'], 'species_or_prediction_values_read': False}
        if index['file'] == 'sage_benchmark_data.tar':
            report['ready_made_external_covariate_files_bytes'] = sum(
                m['size'] for m in index['members'] if m['name'].startswith('predictors/eval_'))
    report['released_predictions_can_be_retrieved_by_HTTP_range'] = bool(
        report['archive_header_only_inventory'].get('predictions.tar', {}).get('status') == 'complete_header_index')
    report['completed_vs_pending'] = ('Covariate-only feasibility and spatial-proximity audit completed. '
        'Species outcomes, exact confirmation protocol, target-mask semantics and confirmation results pending.')
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['status', 'evaluation_cells', 'known_non_Europe_cells', 'all_metadata_rows_within_1km_of_existing_GLC_PA']}))
    print(json.dumps(cohort))


if __name__ == '__main__':
    main()
