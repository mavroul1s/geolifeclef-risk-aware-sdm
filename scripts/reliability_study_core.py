"""Controlled R1 GPU study, executed after frozen definitions and R1 helpers.

This is development research on consumed PA labels. Fixed model/epochs/seeds;
no v32 chain refits, production training, hidden-test inference or submission.
"""

import shutil


class R1Guard(legacy.RuntimeGuard):
    def require(self, reserve_seconds, stage):
        # Old training reserves 45 minutes for production, which this smoke never runs.
        reserve = min(reserve_seconds, 60) if STUDY_MODE == 'smoke' else reserve_seconds
        return super().require(reserve, stage)


def r1_read_predictors(task):
    import rasterio
    from rasterio.enums import Resampling
    root, survey = task
    lp, cp, ip = legacy.feature_paths(Path(root), 'PA-train', int(survey))
    land = legacy._clean_raw_tensor(torch.load(lp, map_location='cpu', weights_only=True).numpy())
    climate = legacy._clean_raw_tensor(torch.load(cp, map_location='cpu', weights_only=True).numpy())
    if land.shape != (6, 4, 21) or climate.shape != (4, 19, 12):
        raise ValueError('Official temporal tensor shape changed')
    with rasterio.open(ip) as image:
        bands = image.read(out_shape=(4, 64, 64), out_dtype='float32', resampling=Resampling.bilinear)
        order = v29.sentinel_band_indices(image.descriptions, [c.name for c in image.colorinterp])
    bands = np.clip(np.nan_to_num(bands[order]/10000, nan=0, posinf=0, neginf=0), 0, 2)
    return land.astype(np.float16), climate.astype(np.float16), bands.astype(np.float16)


def r1_subsample(rows, indices, maximum, tag):
    if maximum is None or len(indices) <= maximum:
        return np.asarray(indices)
    scores = np.array([r1_hash(f'{tag}:{i}') for i in rows.iloc[indices].surveyId])
    return np.sort(np.asarray(indices)[np.argsort(scores, kind='stable')[:maximum]])


def r1_prepare(root, rows, pairs, chosen, directory, guard):
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    kept = rows.iloc[chosen].reset_index(drop=True)
    ids = kept.surveyId.to_numpy(np.int64)
    species = np.sort(pairs.speciesId.unique().astype(np.int64))
    labels = np.lib.format.open_memmap(directory/'labels.npy', mode='w+', dtype=np.uint8,
                                      shape=(len(kept), len(species)))
    labels[:] = 0
    use = pairs[pairs.surveyId.isin(ids)].drop_duplicates()
    ri = pd.Index(ids).get_indexer(use.surveyId)
    ci = pd.Index(species).get_indexer(use.speciesId)
    if min(ri.min(), ci.min()) < 0:
        raise ValueError('Label indexing failed')
    labels[ri, ci] = 1
    labels.flush()
    values, environment_names = [], []
    for path, _ in legacy.discover_environment_pairs(root):
        frame = legacy.aligned_environment(path, ids)
        if frame.shape[1]:
            # Keep the supplied numeric schema; do not select columns on held-out variance.
            array = frame.to_numpy(np.float32)
            values.extend([array, np.isnan(array).astype(np.float32)])
            environment_names.extend([str(path.relative_to(root))+':'+str(c) for c in frame])
            environment_names.extend([str(path.relative_to(root))+':missing:'+str(c) for c in frame])
    if not values:
        raise ValueError('No environmental covariates')
    environment = np.concatenate(values, axis=1)
    np.save(directory/'environment.npy', environment, allow_pickle=False)
    shapes = {'landsat': (6, 4, 21), 'bioclim': (4, 19, 12), 'sentinel': (4, 64, 64)}
    arrays = {n: np.lib.format.open_memmap(directory/f'{n}.npy', mode='w+', dtype=np.float16,
        shape=(len(kept), *shape)) for n, shape in shapes.items()}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(kept), 128):
            guard.require(60*60 if STUDY_MODE == 'full' else 120, 'R1 predictor preparation')
            tasks = [(str(root), int(i)) for i in ids[start:start+128]]
            for offset, sample in enumerate(executor.map(r1_read_predictors, tasks)):
                for name, value in zip(shapes, sample):
                    arrays[name][start+offset] = value
            if start % 2048 == 0:
                guard.stamp('r1_prepare', completed=min(start+128, len(kept)), total=len(kept))
    for array in arrays.values():
        array.flush()
    store = SimpleNamespace(train={'environment': environment},
        raster_train={n: arrays[n] for n in ['landsat', 'bioclim']}, high_train=arrays['sentinel'],
        static_train=v29.candidate_static(kept), eco_train=v29.candidate_static(kept, False),
        labels=labels, species_ids=species, dims={'environment': environment.shape[1]},
        train_ids=ids, rows=kept)
    return store, {'surveys': len(kept), 'environment_columns': environment_names,
        'seconds': time.monotonic()-started, 'test_predictors_loaded': False,
        'missing_indicator_policy': 'every environmental column; normalization fitted within training fold'}


def r1_covariates(store, split, take):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import BallTree
    train = split['training']
    e = np.asarray(store.train['environment'], float)
    with np.errstate(invalid='ignore'):
        median = np.nanmedian(e[train], axis=0)
    median = np.nan_to_num(median)
    clean = np.where(np.isfinite(e), e, median)
    mean, std = clean[train].mean(0), np.maximum(clean[train].std(0), 1e-4)
    z = np.clip((clean-mean)/std, -8, 8)
    pca = PCA(n_components=min(32, z.shape[1], len(train)-1), random_state=20261113)
    pca.fit(z[train])
    metric = BallTree(pca.transform(z[train]))
    novelty = metric.query(pca.transform(z[take]))[0][:, 0]
    geo = BallTree(np.deg2rad(store.rows.iloc[train][['lat', 'lon']].to_numpy()), metric='haversine')
    distance = geo.query(np.deg2rad(store.rows.iloc[take][['lat', 'lon']].to_numpy()))[0][:, 0]*6371.0088
    frame = store.rows.iloc[take][['surveyId', 'country', 'region', 'areaInM2', 'year', 'geoUncertaintyInM']].reset_index(drop=True)
    frame['log_area'] = np.log1p(pd.to_numeric(frame.areaInM2, errors='coerce').clip(lower=0))
    frame['log_geo_uncertainty'] = np.log1p(pd.to_numeric(frame.geoUncertaintyInM, errors='coerce').clip(lower=0))
    frame['log_pa_distance'] = np.log1p(distance)
    frame['environment_novelty'] = novelty
    frame['spatial_block'] = legacy.spatial_blocks(store.rows.iloc[take])
    frame['area_bin'], edges = r1_area_bins(frame.areaInM2, store.rows.iloc[train].areaInM2)
    return frame, edges


def r1_evaluate_probabilities(mean, second, entropy, sets, labels, frequency, frame):
    signals, ranks = r1_signals(mean, second, entropy, sets)
    for name, value in signals.items():
        frame[name] = value
    y = np.asarray(labels, np.uint8)
    hit = np.take_along_axis(y, ranks, 1)
    truth_count = y.sum(1)
    frame['sample_F1'] = 2*hit.sum(1)/(truth_count+18)
    frame['loss'] = 1-frame.sample_F1
    frame['true_richness'] = truth_count
    frame['predicted_richness'] = 18
    rare = (frequency >= 1) & (frequency <= 25)
    frame['rare_truth'] = y[:, rare].sum(1)
    frame['rare_hits'] = (hit*rare[ranks]).sum(1)
    frame['rare_predictions'] = rare[ranks].sum(1)
    frame['hits'] = hit.sum(1)
    p = np.asarray(mean, np.float32)
    frame['brier'] = np.square(p-y).mean(1)
    # Diagnostics for the frozen secondary count decoder, no policy selection.
    rank40, value40 = legacy.top_rank(p, 40)
    objective = 2*np.cumsum(value40, axis=1)/(np.arange(1, 41)+p.sum(1)[:, None]+1e-8)
    counts = objective[:, 7:40].argmax(1)+8
    allhits = np.take_along_axis(y, rank40, 1)
    hits = np.cumsum(allhits, axis=1)[np.arange(len(y)), counts-1]
    frame['secondary_count'] = counts
    frame['secondary_F1'] = 2*hits/(counts+truth_count)
    # Compute classwise metrics without a survey x label predicted dense array.
    tp = np.bincount(ranks.ravel(), weights=hit.ravel(), minlength=y.shape[1])
    pred = np.bincount(ranks.ravel(), minlength=y.shape[1])
    actual = y.sum(0)
    confusion = {'tp': tp, 'predicted': pred, 'actual': actual}
    return frame, ranks, confusion


def r1_fit_fold(number, split, store, temporary, export, guard, model_config, seeds, device=None):
    started = time.monotonic()
    device = device or torch.device('cuda')
    stats = v29.fit_normalization(store, split['training'])
    roles = ['risk_calibration', 'assessment']
    take = np.concatenate([split[r] for r in roles])
    output_rows, area_edges = r1_covariates(store, split, take)
    output_rows['fold'] = number
    output_rows['role'] = np.concatenate([np.full(len(split[r]), r) for r in roles])
    n, s = len(take), len(store.species_ids)
    fold_dir = temporary/f'fold{number}'
    fold_dir.mkdir(exist_ok=True)
    first = np.lib.format.open_memmap(fold_dir/'first.npy', mode='w+', dtype=np.float32, shape=(n, s))
    second = np.lib.format.open_memmap(fold_dir/'second.npy', mode='w+', dtype=np.float32, shape=(n, s))
    first[:] = 0
    second[:] = 0
    entropies = np.zeros(n, np.float64)
    member_sets = np.empty((n, len(seeds), 18), np.uint16)
    records = []
    for member, seed in enumerate(seeds):
        guard.require(50*60 if STUDY_MODE == 'full' else 150, 'R1 member admission')
        config = {**model_config, 'seed': seed, 'id': f'r1_fold{number}_seed{seed}'}
        # Fixed-epoch path never queries selection/assessment labels or changes epoch count.
        model, record = v29.train_candidate(store, store.rows, split['training'], np.array([], np.int64),
            stats, config, fold_dir, guard, device, fixed_epochs=config['epochs'], phase='fixed_r1')
        cal = v29.predict(model, store, split['probability_calibration'], stats, device, config, guard=guard)
        calibration = v29.fit_platt(cal, store.labels[split['probability_calibration']])
        if not calibration['optimizer_success']:
            raise RuntimeError('Probability calibration did not converge')
        del cal
        probability = v29.predict(model, store, take, stats, device, config, guard=guard)
        for begin in range(0, n, 256):
            stop = min(begin+256, n)
            p = v29.calibrated(probability[begin:stop], calibration)
            first[begin:stop] += p
            second[begin:stop] += p*p
            entropies[begin:stop] += r1_binary_entropy(p).mean(1)
            member_sets[begin:stop, member] = legacy.top_rank(p, 18)[0]
        checkpoint = fold_dir/f"fixed_r1_{config['id']}.pt"
        if STUDY_MODE == 'full':
            shutil.copy2(checkpoint, export/checkpoint.name)
        records.append({**record, 'probability_calibration': calibration,
            'probability_calibration_surveys': len(split['probability_calibration'])})
        del model, probability
        previous.release(device)
        guard.stamp('r1_member_complete', fold=number, member=member+1, total=len(seeds))
    first /= len(seeds)
    second /= len(seeds)
    entropies /= len(seeds)
    frequency = legacy._frequency(store.labels, split['training'])
    frames, ranks = [], []
    for begin in range(0, n, 256):
        stop = min(begin+256, n)
        f, rank, cm = r1_evaluate_probabilities(first[begin:stop], second[begin:stop], entropies[begin:stop],
            member_sets[begin:stop], store.labels[take[begin:stop]], frequency, output_rows.iloc[begin:stop].copy())
        frames.append(f)
        ranks.append(rank)
    frame = pd.concat(frames).reset_index(drop=True)
    calibration = frame[frame.role == 'risk_calibration'].copy()
    assessment = frame[frame.role == 'assessment'].copy()
    assessed_ranks = np.concatenate(ranks)[frame.role.to_numpy() == 'assessment']
    assessed_labels = store.labels[take[frame.role.to_numpy() == 'assessment']]
    multilabel = v29.multilabel_summary(assessed_labels, assessed_ranks)
    rare_metrics = legacy.species_group_metrics(assessed_labels, assessed_ranks, frequency)
    risks, fitting = r1_fit_selectors(calibration, assessment.drop(columns=['loss', 'sample_F1']), STUDY['risk_regressor'])
    if set(risks) != set(STUDY['selectors']):
        raise ValueError('Implemented selector family differs from the frozen configuration')
    curves, countries, selected = r1_selection_tables(assessment, risks, STUDY['coverage_grid'], STUDY['coverage_policies'])
    for method, score in risks.items():
        assessment['risk_'+method] = score
        assessment['primary_accept_'+method] = selected[method]
    frame.to_csv(export/f'fold{number}_confidence_roles.csv', index=False)
    assessment.to_csv(export/f'fold{number}_assessment.csv', index=False)
    np.savez_compressed(export/f'fold{number}_predictions.npz', survey_id=store.train_ids[take],
        role=frame.role.to_numpy(dtype='U24'), species_ids=store.species_ids,
        mean_probability=np.asarray(first, np.float16), member_top18_columns=member_sets,
        true_offsets=np.r_[0, np.cumsum(np.asarray(store.labels[take]).sum(1))].astype(np.uint32),
        true_species_columns=np.concatenate([np.flatnonzero(store.labels[j]).astype(np.uint16) for j in take]),
        training_frequency=frequency, mean_entropy_members=entropies.astype(np.float32))
    result = {'fold': number, 'training': records, 'risk_fitting': fitting,
        'normalization_training_surveys': len(split['training']), 'area_edges_log1p': area_edges.tolist(),
        'role_counts': {k: len(v) for k, v in split.items()}, 'curves': curves, 'countries': countries,
        'assessment_sample_F1': float(assessment.sample_F1.mean()),
        'assessment_secondary_F1': float(assessment.secondary_F1.mean()),
        'assessment_brier': float(assessment.brier.mean()),
        'multilabel_metrics': multilabel, 'species_groups': rare_metrics,
        'seconds': time.monotonic()-started}
    del first, second
    shutil.rmtree(fold_dir)
    return result, assessment


def run_reliability_study():
    import importlib.metadata
    from sklearn.neighbors import BallTree
    mode = STUDY_MODE
    limit = STUDY['maximum_runtime_hours'] if mode == 'full' else STUDY['smoke']['maximum_runtime_hours']
    guard = R1Guard(limit)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable before data preparation')
    torch.set_num_threads(2)
    work = Path('/kaggle/working')
    export, temporary = work/'r1_export', work/'r1_runtime'
    export.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    root = legacy.discover_data_root()
    raw_path = root/'GLC25_PA_metadata_train.csv'
    if legacy.sha256_file(raw_path) != STUDY['metadata_sha256']:
        raise ValueError('Training metadata changed')
    raw = pd.read_csv(raw_path)
    rows = raw.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    original_splits, manifests = v31.previous.make_splits(rows)
    folds = STUDY['folds'] if mode == 'full' else STUDY['smoke']['folds']
    splits = []
    for fold in folds:
        original = original_splits[fold]
        split = {'training': original['training'], 'probability_calibration': original['selection'],
                 'risk_calibration': original['calibration'], 'assessment': original['assessment']}
        if mode != 'full':
            split = {role: r1_subsample(rows, take, STUDY['smoke'][role+'_surveys'], f'r1-smoke:{role}')
                     for role, take in split.items()}
        splits.append(split)
    chosen = np.unique(np.concatenate([v for split in splits for v in split.values()]))
    store, features = r1_prepare(root, rows, raw[['surveyId', 'speciesId']], chosen, temporary/'features', guard)
    del raw
    mapped = []
    lookup = pd.Index(store.train_ids)
    for split in splits:
        mapped.append({role: lookup.get_indexer(rows.iloc[take].surveyId) for role, take in split.items()})
    for split in mapped:
        roles = list(split)
        for i, role in enumerate(roles):
            if (split[role] < 0).any():
                raise ValueError('Missing role IDs')
            for other in roles[i+1:]:
                if np.intersect1d(split[role], split[other]).size:
                    raise ValueError('Overlapping model/calibration/assessment roles')
        training_xy = np.deg2rad(store.rows.iloc[split['training']][['lat', 'lon']].to_numpy())
        held = np.concatenate([v for k, v in split.items() if k != 'training'])
        distance = BallTree(training_xy, metric='haversine').query(np.deg2rad(store.rows.iloc[held][['lat', 'lon']].to_numpy()))[0].min()*6371.0088
        if distance < 20:
            raise ValueError('20 km training buffer failed')
    model = dict(STUDY['model'])
    seeds = STUDY['neural_seeds'] if mode == 'full' else STUDY['smoke']['seeds']
    if mode != 'full':
        model['epochs'] = STUDY['smoke']['epochs']
    results, frames = [], []
    for fold, split in zip(folds, mapped):
        if mode == 'full' and results:
            estimate = results[0]['seconds']*len(split['training'])/len(mapped[0]['training'])*1.5+600
            if guard.remaining_seconds() < estimate:
                raise RuntimeError('Measured remaining-plan admission failed; partial results retained')
        result, frame = r1_fit_fold(fold, split, store, temporary, export, guard, model, seeds)
        results.append(result)
        frames.append(frame)
        legacy.save_json(export/f'fold{fold}_report.json', result)
    pooled = pd.concat(frames).sort_values('surveyId').reset_index(drop=True)
    if pooled.surveyId.duplicated().any():
        raise ValueError('Outer assessment IDs overlap')
    contrast = STUDY['primary_contrast']
    candidate = pooled['primary_accept_'+contrast['candidate']].to_numpy(bool)
    baseline = pooled['primary_accept_'+contrast['baseline']].to_numpy(bool)
    comparison = r1_paired_selected_bootstrap(pooled.sample_F1, candidate, baseline, pooled.spatial_block,
        repetitions=2000 if mode == 'full' else 50)
    curves = pd.DataFrame([r for f in results for r in f['curves']])
    countries = pd.DataFrame([r for f in results for r in f['countries']])
    # Preserve fold identities in summary files.
    curves = pd.concat([pd.DataFrame(r['curves']).assign(fold=r['fold']) for r in results])
    countries = pd.concat([pd.DataFrame(r['countries']).assign(fold=r['fold']) for r in results])
    curves.to_csv(export/'risk_coverage.csv', index=False)
    countries.to_csv(export/'country_coverage.csv', index=False)
    model_records = [r for fold in results for r in fold['training']]
    report = {'experiment': STUDY['experiment'], 'mode': mode, 'status': 'complete',
        'config_sha256': STUDY_CONFIG_SHA256, 'fresh_assessment': False,
        'development_analysis_enabled': mode == 'full', 'independent_claims_permitted': False,
        'neural_fits': len(model_records), 'epochs': model['epochs'], 'seeds': seeds,
        'folds': folds, 'primary_comparison': comparison,
        'minimum_gain_margin_met': comparison['gain'] >= contrast['minimum_absolute_F1_gain'] if mode == 'full' else None,
        'features': features, 'runtime_hours': guard.elapsed_hours(),
        'training_epoch_seconds': [[x['seconds'] for x in r['history']] for r in model_records],
        'hardware': {'gpu': torch.cuda.get_device_name(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'torch': torch.__version__, 'sklearn': importlib.metadata.version('scikit-learn')},
        'probability_calibration_and_risk_fitting_disjoint': True,
        'neural_training_queried_assessment_labels': False,
        'competition_submission_made': False, 'validation_scope': 'historically consumed spatial development data',
        'limitations': ['No independent statistical confirmation.',
            'Seed disagreement combines model and per-member calibration variability.',
            'Model probability entropy is not an assumption-free measure of epistemic uncertainty.',
            'Country/area quotas control finite-cohort retention, not prediction error guarantees.',
            'Fixed model is a research baseline, not a recreation of the full scored v32 ensemble.']}
    legacy.save_json(export/'r1_report.json', report)
    np.save(export/'species_ids.npy', store.species_ids, allow_pickle=False)
    del store
    shutil.rmtree(temporary)
    # Keep only documented artifacts as Kaggle outputs; cap total output size.
    final = list(export.iterdir())
    total = sum(p.stat().st_size for p in final)
    if total > STUDY['maximum_output_bytes']:
        raise ValueError('Final research output envelope exceeded')
    manifest = {'experiment': STUDY['experiment'], 'mode': mode, 'config_sha256': STUDY_CONFIG_SHA256,
        'original_split_manifests': [manifests[f] for f in folds],
        'effective_role_hashes': [{role: legacy.sha256_bytes(rows.iloc[chosen[indices]].surveyId.to_numpy(np.int64).astype('<i8').tobytes())
            for role, indices in split.items()} for split in mapped],
        'outputs': {p.name: legacy.sha256_file(p) for p in final}, 'output_bytes_before_manifest': total,
        'test_predictors_loaded': False, 'official_submission_made': False}
    legacy.save_json(export/'r1_manifest.json', manifest)
    return report


try:
    R1_RESULT = run_reliability_study()
    print(json.dumps({'status': R1_RESULT['status'], 'mode': R1_RESULT['mode'],
                      'runtime_hours': R1_RESULT['runtime_hours'], 'neural_fits': R1_RESULT['neural_fits']}))
except Exception as error:
    failure = Path('/kaggle/working/r1_failure.json')
    failure.write_text(json.dumps({'status': 'failed', 'mode': STUDY_MODE,
        'error_type': type(error).__name__, 'message': str(error)[:1500],
        'config_sha256': STUDY_CONFIG_SHA256, 'competition_submission_made': False}, indent=2)+'\n')
    shutil.rmtree('/kaggle/working/r1_runtime', ignore_errors=True)
    raise
