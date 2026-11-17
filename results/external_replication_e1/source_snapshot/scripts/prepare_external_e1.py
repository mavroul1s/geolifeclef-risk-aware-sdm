"""Prepare E1 roles/masks using covariates and PO-only species properties."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import hdf5plugin  # register the released Blosc filter
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from geolifeclef.external_reliability import read_rows

ROOT = Path(__file__).resolve().parents[1]


def digest_ids(values):
    return hashlib.sha256(np.asarray(values, '<i8').tobytes()).hexdigest()


def main():
    config = json.loads((ROOT/'configs/external_reliability_e1.json').read_text())
    data = ROOT/'data/external_replication_e1';output = ROOT/'results/external_replication_e1'
    output.mkdir(exist_ok=True, parents=True)
    candidates = np.loadtxt(ROOT/'data/independent_data_audit/candidate_SAGE_cells_metadata_only.txt', dtype=np.int64)
    old = json.loads((ROOT/'results/reliability_followup_d1_d3/independent_data/audit.json').read_text())
    if digest_ids(np.sort(candidates)) != old['candidate_cohort']['cell_IDs_sha256']:
        raise ValueError('Previously audited candidate set changed')
    # Do not load test_prevalence or other PA-derived species-property fields.
    props = pd.read_csv(data/'targets/species_properties_1km.csv', usecols=[
        'species_id', 'relative_prevalence', 'n_species_obs', 'sampling_effort']).sort_values('species_id')
    if not np.array_equal(props.species_id, np.arange(len(props))):
        raise ValueError('Species-column identity is not the documented canonical index')
    positive = props.relative_prevalence.gt(0) & np.isfinite(props.relative_prevalence)
    cutoff = float(props.loc[positive, 'relative_prevalence'].quantile(.2))
    rare = (positive & props.relative_prevalence.le(cutoff)).to_numpy()
    np.save(data/'rare_species_mask.npy', rare, allow_pickle=False)
    columns = ['cell_row', 'PlotObservationID', 'Country', 'Continent', 'GIVD_ID',
               'Date_of_recording', 'Releve_area', 'Location_uncertainty']
    metadata = pd.read_csv(ROOT/'data/independent_data_audit/sage_eval_plot_metadata.csv', usecols=columns)
    metadata = metadata[metadata.cell_row.isin(candidates)].copy()
    metadata['year'] = pd.to_numeric(metadata.Date_of_recording.astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
    grouped = metadata.groupby('cell_row')
    coherent = grouped.Country.nunique().eq(1) & grouped.Continent.nunique().eq(1)
    table = grouped.agg(country=('Country', 'first'), continent=('Continent', 'first'),
        area=('Releve_area', 'sum'), geo_uncertainty=('Location_uncertainty', 'max'),
        year=('year', 'median'), source_count=('GIVD_ID', 'nunique'), plot_count=('PlotObservationID', 'nunique'))
    table = table.loc[coherent]
    test_indices = np.load(data/'test_indices.npy', allow_pickle=False)
    lookup = pd.Series(np.arange(len(test_indices)), index=test_indices)
    table = table.loc[table.index.isin(test_indices)].sort_index()
    table['prediction_row'] = lookup.loc[table.index].to_numpy()
    # Sampled area is a covariate, not a species outcome. The public metadata
    # can omit plots removed during species preprocessing; reject ambiguities.
    with h5py.File(data/'sealed_eval_test_targets.h5') as f:
        published_area = read_rows(f['sampled_area_m2'], table.prediction_row.to_numpy())
    consistent_area = np.isclose(published_area, table.area, atol=.1, rtol=1e-4)
    excluded_area = table.loc[~consistent_area].copy()
    excluded_area['published_sampled_area_m2'] = published_area[~consistent_area]
    excluded_area.to_csv(data/'area_provenance_exclusions.csv')
    table = table.loc[consistent_area].copy()
    with h5py.File(data/'predictors/eval_location.h5') as f:
        locations = f['predictor_data'][:]
    table['lon'], table['lat'] = locations[table.index, 0], locations[table.index, 1]
    size = config['spatial_block_degrees']
    table['spatial_block'] = [f'{int(np.floor(lat/size))}:{int(np.floor(lon/size))}' for lat, lon in table[['lat', 'lon']].to_numpy()]
    names = np.sort(table.country.unique());country = pd.Index(names).get_indexer(table.country)
    blocks = table.spatial_block.to_numpy()
    unique = np.unique(blocks)
    vectors = {b: np.bincount(country[blocks == b], minlength=len(names)) for b in unique}
    fractions = np.asarray(config['role_fractions'])
    total = np.bincount(country, minlength=len(names))
    assigned = np.zeros((3, len(names)), np.int64)
    allocation = {}
    order = sorted(unique, key=lambda b: (-int(vectors[b].sum()), hashlib.sha256((config['role_seed']+b).encode()).hexdigest()))
    for b in order:
        v = vectors[b]
        cost = .5*(assigned.sum(1)+v.sum())/(len(table)*fractions)
        cost += .5*(((assigned+v)/np.maximum(fractions[:, None]*total, 1))*(v/v.sum())[None]).sum(1)
        role = int(np.argmin(cost));allocation[b] = role;assigned[role] += v
    table['role_number'] = [allocation[b] for b in blocks]
    # Assessment stays fixed; calibration rows near another role are removed.
    coordinates = np.deg2rad(table[['lat', 'lon']].to_numpy())
    keep = np.ones(len(table), bool)
    assessment = table.role_number.to_numpy() == 2
    distance = BallTree(coordinates[assessment], metric='haversine').query(coordinates)[0][:, 0]*6371.0088
    keep[(~assessment) & (distance < config['role_buffer_km'])] = False
    probcal = (table.role_number.to_numpy() == 0) & keep
    riskcal = (table.role_number.to_numpy() == 1) & keep
    distance = BallTree(coordinates[probcal], metric='haversine').query(coordinates[riskcal])[0][:, 0]*6371.0088
    keep[np.flatnonzero(riskcal)[distance < config['role_buffer_km']]] = False
    removed = int((~keep).sum())
    table = table.iloc[np.flatnonzero(keep)].copy()
    table['role'] = [config['roles'][i] for i in table.role_number]
    probability = table[table.role == 'probability_calibration']
    edges = np.unique(np.quantile(np.log1p(probability.area), [.2, .4, .6, .8]))
    table['area_bin'] = np.searchsorted(edges, np.log1p(table.area), side='right').astype(str)
    table['log_area'] = np.log1p(table.area)
    table['log_geo_uncertainty'] = np.log1p(table.geo_uncertainty)
    reference = BallTree(np.deg2rad(probability[['lat', 'lon']].to_numpy()), metric='haversine')
    distance = reference.query(np.deg2rad(table[['lat', 'lon']].to_numpy()))[0][:, 0]*6371.0088
    table['log_calibration_distance'] = np.log1p(distance)
    counts = table.role.value_counts().to_dict()
    for role, minimum in zip(config['roles'], config['minimum_role_counts']):
        if counts.get(role, 0) < minimum:
            raise ValueError('Covariate-only admission failed: '+str(counts))
    table.reset_index().rename(columns={'cell_row': 'cell_id'}).to_csv(data/'cohort_roles.csv', index=False)
    # Native-range mask is a covariate, distinct from sealed PA outcomes.
    with h5py.File(data/'range_masks_1km/eval_range_mask.h5') as f:
        offsets = f['cumulative_lengths'][:]
        species = f['species_index'][:]
    mask = np.lib.format.open_memmap(data/'eligible_species.npy', mode='w+', dtype=bool, shape=(len(table), len(props)))
    mask[:] = False
    for row, cell in enumerate(table.index):
        mask[row, species[offsets[cell]:offsets[cell+1]]] = True
    mask.flush();del mask
    role_hashes = {role: digest_ids(table.index[table.role == role]) for role in config['roles']}
    result = {'status': 'covariate_roles_prepared_outcomes_closed', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_candidate_cells': len(candidates), 'external_published_test_cohort': len(table),
        'buffer_excluded_rows': removed, 'role_counts': counts, 'role_ID_hashes': role_hashes,
        'area_provenance_excluded_cells': len(excluded_area),
        'area_admission_before_species_outcomes': True,
        'role_blocks': table.groupby('role').spatial_block.nunique().to_dict(),
        'area_edges_log1p': edges.tolist(), 'rare_relative_prevalence_cutoff': cutoff,
        'rare_species_count': int(rare.sum()), 'species_properties_fields_read': ['species_id','relative_prevalence','n_species_obs','sampling_effort'],
        'test_derived_species_properties_used': False, 'species_outcomes_inspected': False,
        'calibration_holdout_scope': 'reliability calibration and assessment; pretrained SAGE models follow their original PO overlap protocol'}
    (output/'cohort_design.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
