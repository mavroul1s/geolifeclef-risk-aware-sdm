"""Executed after the frozen v35 definitions/evidence cells, on CPU only.

This reproduces the retrospective risk pilot; it does not run run_v35, train a
neural model, or emit a competition submission. Cache provenance remains v34.
"""


def run_reliability_pilot():
    import importlib.metadata
    import platform
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neighbors import BallTree

    started = time.monotonic()
    local = Path('data/raw')
    root = local if (local/'GLC25_PA_metadata_train.csv').is_file() else legacy.discover_data_root()
    reference = load_reference(REFERENCE_B64)
    if legacy.sha256_file(root/'GLC25_PA_metadata_train.csv') != reference['train_metadata_sha256']:
        raise ValueError('Official metadata changed; refusing cached comparisons')
    raw = pd.read_csv(root/'GLC25_PA_metadata_train.csv')
    rows = raw.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    splits, manifests = v31.previous.make_splits(rows)
    if [m['id_hashes'] for m in manifests] != [m['id_hashes'] for m in reference['folds']]:
        raise ValueError('Cached spatial fold identities differ')
    species = np.sort(raw.speciesId.unique())
    if hashlib.sha256(species.astype('<i8').tobytes()).hexdigest() != reference['species_sha256']:
        raise ValueError('Species vocabulary changed')
    pairs = raw[['surveyId', 'speciesId']].drop_duplicates()
    pair_columns = pd.Index(species).get_indexer(pairs.speciesId)
    truth = pd.Series(pair_columns, index=pairs.surveyId).groupby(level=0).agg(list).to_dict()
    coords = np.deg2rad(rows[['lat', 'lon']].to_numpy())
    seed = 20261113

    def features(frame, distance):
        area = np.log1p(pd.to_numeric(frame.areaInM2, errors='coerce').clip(lower=0))
        uncertainty = np.log1p(pd.to_numeric(frame.geoUncertaintyInM, errors='coerce').clip(lower=0))
        return np.column_stack([area, frame.year, uncertainty, np.log1p(distance)])

    bundles = []
    for fold, split in enumerate(splits):
        cached = reference['folds'][fold]
        cal = cached['calibration']
        cal_index = pd.Index(rows.surveyId).get_indexer(cal['ids'])
        if not np.isin(cal_index, split['calibration']).all():
            raise ValueError('Calibration role mismatch')
        cal_f1 = []
        for survey, prediction in zip(cal['ids'], cal['predictions']):
            labels, prediction = set(truth[survey]), set(prediction)
            cal_f1.append(2*len(labels & prediction)/(len(labels)+len(prediction)))
        if not np.allclose(cal_f1, cal['f1'], atol=1e-12):
            raise ValueError('Cached calibration scores disagree with official labels')
        assessed = cached['assessment']
        assessment_index = pd.Index(rows.surveyId).get_indexer(assessed['ids'])
        if set(assessment_index) != set(split['assessment']):
            raise ValueError('Assessment coverage mismatch')
        tree = BallTree(coords[split['training']], metric='haversine')
        cd = tree.query(coords[cal_index])[0][:, 0]*6371.0088
        ad = tree.query(coords[assessment_index])[0][:, 0]*6371.0088
        xc, xa = features(rows.iloc[cal_index], cd), features(rows.iloc[assessment_index], ad)
        frame = rows.iloc[assessment_index][['surveyId', 'country']].copy()
        frame['loss'] = 1-np.asarray(assessed['f1'])
        frame['risk_distance_only'] = ad
        frame['risk_random'] = [int(hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()[:12], 16) for i in frame.surveyId]
        for name, cols in [('area_only', [0]), ('metadata_only', [0, 1, 2]), ('metadata_distance', [0, 1, 2, 3])]:
            model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7,
                min_samples_leaf=50, l2_regularization=5, learning_rate=.05,
                early_stopping=False, random_state=seed)
            model.fit(xc[:, cols], 1-np.asarray(cal_f1))
            frame[f'risk_{name}'] = model.predict(xa[:, cols])
        bundles.append(frame)
    frame = pd.concat(bundles).sort_values('surveyId').reset_index(drop=True)
    if frame.surveyId.duplicated().any():
        raise ValueError('Duplicate assessment IDs')
    results, country_results, areas = [], [], []
    methods = ['random', 'distance_only', 'area_only', 'metadata_only', 'metadata_distance',
               'distance_only_country_balanced', 'metadata_distance_country_balanced']
    for method in methods:
        source = method.removesuffix('_country_balanced')
        order = np.lexsort((frame.risk_random.to_numpy(), frame[f'risk_{source}'].to_numpy()))
        if method.endswith('_country_balanced'):
            ordered = frame.iloc[order]
            percentile = (ordered.groupby('country').cumcount()+.5)/ordered.groupby('country').country.transform('size')
            order = order[np.lexsort((ordered.risk_random.to_numpy(), percentile.to_numpy()))]
        losses = frame.loss.to_numpy()[order]
        areas.append({'method': method, 'AURC': float(np.mean(losses.cumsum()/np.arange(1, len(losses)+1)))})
        for coverage in [1., .9, .8, .6, .4, .2]:
            selected = frame.iloc[order[:int(np.ceil(coverage*len(frame)))]]
            for group, mask in [('all', np.ones(len(selected), bool)),
                ('core', selected.country.isin(['Denmark', 'Netherlands'])),
                ('outside_core', ~selected.country.isin(['Denmark', 'Netherlands']))]:
                s = selected.loc[mask]
                results.append({'method': method, 'overall_retained_fraction': coverage,
                    'group': group, 'surveys': len(s), 'mean_sample_F1': float(1-s.loss.mean())})
            for country, full in frame.groupby('country'):
                s = selected[selected.country == country]
                country_results.append({'method': method, 'overall_retained_fraction': coverage,
                    'country': country, 'available_surveys': len(full), 'retained_surveys': len(s),
                    'country_retained_fraction': len(s)/len(full),
                    'mean_sample_F1': float(1-s.loss.mean()) if len(s) else None})
    export = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('artifacts/kaggle_reliability_local')
    export.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(export/'selective_pilot.csv', index=False)
    pd.DataFrame(country_results).to_csv(export/'selective_country_coverage.csv', index=False)
    report = {'status': 'completed_CPU_reproduction', 'fresh_assessment': False,
        'scope': 'reused matched-v32 recipe development scores from v34; no new hidden-test result',
        'risk_model_fits': 6, 'neural_fits': 0, 'competition_submission_made': False,
        'calibration_scores_recomputed_from_official_metadata': True,
        'training_metadata_sha256': reference['train_metadata_sha256'],
        'split_hashes_verified': True, 'assessment_surveys': len(frame),
        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'torch']},
        'python': platform.python_version(), 'seed': seed, 'area_under_risk_coverage': areas,
        'results': results, 'seconds': time.monotonic()-started,
        'limitations': ['Retrospective adaptively reused data.', 'No conformal or distribution-free risk guarantee.',
            'Selective F1 does not measure full-coverage Kaggle performance.',
            'Country coverage and plot-area confounding must accompany pooled curves.']}
    (export/'reliability_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


RELIABILITY_REPORT = run_reliability_pilot()
print(json.dumps({'status': RELIABILITY_REPORT['status'], 'seconds': RELIABILITY_REPORT['seconds'],
                  'neural_fits': 0, 'fresh_assessment': False}))
