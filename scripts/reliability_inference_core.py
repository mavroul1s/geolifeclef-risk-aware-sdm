"""Reuse immutable R1 checkpoints. No optimizer, backward or training call."""


def infer_saved_fold(fold, split, store, source, temporary, export, guard, device):
    report = json.loads((source/f'fold{fold}_report.json').read_text())
    stats = v29.fit_normalization(store, split['training'])
    legacy.save_json(export/f'fold{fold}_normalization.json', stats)
    roles = ['probability_calibration', 'risk_calibration', 'assessment']
    take = np.concatenate([split[r] for r in roles])
    metadata, edges = r1_covariates(store, split, take)
    metadata['fold'] = fold
    metadata['role'] = np.concatenate([np.full(len(split[r]), r) for r in roles])
    start = len(split['probability_calibration'])
    n = len(take)
    mean = np.lib.format.open_memmap(temporary/f'mean{fold}.npy', mode='w+', dtype=np.float32,
                                    shape=(n, len(store.species_ids)))
    mean[:] = 0
    frequency = legacy._frequency(store.labels, split['training'])
    with np.load(source/f'fold{fold}_predictions.npz', allow_pickle=False) as previous_archive:
        previous_ids = previous_archive['survey_id']
        previous_sets = previous_archive['member_top18_columns']
    if not np.array_equal(previous_ids, store.train_ids[take[start:]]):
        raise ValueError('Saved confidence role order differs from reconstructed source')
    all_frames, threshold_records, member_reproduction = [], [], []
    for member, record in enumerate(report['training']):
        guard.require(15*60, 'Saved checkpoint inference')
        config = record['config']
        path = source/f"fixed_r1_{config['id']}.pt"
        if legacy.sha256_file(path) != record['checkpoint_sha256']:
            raise ValueError('Saved checkpoint digest mismatch')
        model = v29.create_model(store, config, split['training']).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        raw = v29.predict(model, store, take, stats, device, config, guard=guard)
        calibration = record['probability_calibration']
        pcal = v29.calibrated(raw[:start], calibration)
        rank = decision_ranks(pcal)
        values = np.take_along_axis(pcal, rank, axis=1)
        hits = np.take_along_axis(np.asarray(store.labels[take[:start]]), rank, axis=1)
        threshold, grid = choose_threshold(values, hits, store.labels[take[:start]].sum(1), FOLLOWUP['threshold_grid'])
        threshold_records.append({'predictor': f'seed_{config["seed"]}', 'threshold': threshold,
                                  'fitting_role': 'probability_calibration', 'grid': grid})
        del pcal
        identical = []
        for begin in range(0, n, 256):
            stop = min(begin+256, n)
            p = v29.calibrated(raw[begin:stop], calibration)
            mean[begin:stop] += p
            if stop <= start:
                continue
            lo = max(begin, start)
            q = p[lo-begin:]
            old_set = legacy.top_rank(q, 18)[0]
            identical.extend(np.all(np.sort(old_set, axis=1) ==
                np.sort(previous_sets[lo-start:stop-start, member], axis=1), axis=1))
            for decoder in FOLLOWUP['decoders']:
                frame = decision_frame(q, store.labels[take[lo:stop]], metadata.iloc[lo:stop],
                                       frequency, decoder, threshold)
                frame['predictor'] = f'seed_{config["seed"]}'
                all_frames.append(frame)
        member_reproduction.append({'seed': config['seed'], 'original_top18_set_match_fraction': float(np.mean(identical))})
        del model, raw
        previous.release(device)
        guard.stamp('checkpoint_inferred', fold=fold, member=member+1)
    mean /= len(report['training'])
    pcal = np.asarray(mean[:start])
    rank = decision_ranks(pcal)
    values = np.take_along_axis(pcal, rank, axis=1)
    hits = np.take_along_axis(np.asarray(store.labels[take[:start]]), rank, axis=1)
    threshold, grid = choose_threshold(values, hits, store.labels[take[:start]].sum(1), FOLLOWUP['threshold_grid'])
    threshold_records.append({'predictor': 'ensemble', 'threshold': threshold,
                              'fitting_role': 'probability_calibration', 'grid': grid})
    del pcal
    for begin in range(start, n, 256):
        stop = min(begin+256, n)
        for decoder in FOLLOWUP['decoders']:
            frame = decision_frame(mean[begin:stop], store.labels[take[begin:stop]], metadata.iloc[begin:stop],
                                   frequency, decoder, threshold)
            frame['predictor'] = 'ensemble'
            all_frames.append(frame)
    frames = pd.concat(all_frames, ignore_index=True)
    frames.to_csv(export/f'fold{fold}_decisions.csv', index=False)
    old = pd.read_csv(source/f'fold{fold}_assessment.csv', float_precision='round_trip').set_index('surveyId')
    new = frames[(frames.predictor == 'ensemble') & (frames.decoder == 'top18') &
                 (frames.role == 'assessment')].set_index('surveyId').loc[old.index]
    delta = new.sample_F1.to_numpy()-old.sample_F1.to_numpy()
    match_fraction = float(np.mean(np.isclose(delta, 0, atol=1e-12)))
    reproduction = {'fold': fold, 'survey_F1_match_fraction': match_fraction,
        'mean_F1_difference': float(delta.mean()), 'maximum_survey_F1_difference': float(np.abs(delta).max()),
        'member_sets': member_reproduction, 'area_edges_log1p': edges.tolist(),
        'normalization_training_surveys': len(split['training']), 'thresholds': threshold_records}
    reproduction['passed'] = (match_fraction >= FOLLOWUP['reproduction_minimum_exact_survey_F1_fraction'] and
        abs(delta.mean()) <= FOLLOWUP['reproduction_maximum_absolute_mean_F1_difference'] and
        min(r['original_top18_set_match_fraction'] for r in member_reproduction) >= .999)
    legacy.save_json(export/f'fold{fold}_reproduction.json', reproduction)
    # Exact float32 ensemble scores enable future decision studies without another inference run.
    np.savez_compressed(export/f'fold{fold}_mean_probability.npz',
        mean_probability=np.asarray(mean[start:]), survey_id=store.train_ids[take[start:]],
        role=metadata.role.to_numpy(dtype='U24')[start:], species_ids=store.species_ids)
    del mean
    (temporary/f'mean{fold}.npy').unlink()
    return reproduction


def run_saved_inference():
    from sklearn.neighbors import BallTree
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError('Inference job requires its requested GPU')
    def forbid_training(*args, **kwargs):
        raise RuntimeError('New neural training is forbidden in this study')
    v29.train_candidate = forbid_training
    guard = R1Guard(FOLLOWUP['inference_maximum_hours'])
    export, temporary = Path('/kaggle/working/inference_export'), Path('/kaggle/working/inference_runtime')
    export.mkdir(exist_ok=True); temporary.mkdir(exist_ok=True)
    candidates = []
    for pattern in ['*/r1_export/r1_manifest.json', 'notebooks/*/*/r1_export/r1_manifest.json',
                    'kernels/*/*/r1_export/r1_manifest.json', '*/r1_manifest.json']:
        candidates.extend(Path('/kaggle/input').glob(pattern))
    candidates = list({p.resolve() for p in candidates})
    if len(candidates) != 1:
        raise ValueError('Expected one attached immutable R1 output source: '+str(candidates))
    source = candidates[0].parent
    if legacy.sha256_file(candidates[0]) != SOURCE_MANIFEST_SHA256:
        raise ValueError('R1 input manifest changed')
    manifest = json.loads(candidates[0].read_text())
    for name, digest in manifest['outputs'].items():
        if name.endswith('.pt') or 'fold' in name:
            if legacy.sha256_file(source/name) != digest:
                raise ValueError('R1 source output digest mismatch: '+name)
    root = legacy.discover_data_root()
    if legacy.sha256_file(root/'GLC25_PA_metadata_train.csv') != STUDY['metadata_sha256']:
        raise ValueError('Official metadata changed')
    raw = pd.read_csv(root/'GLC25_PA_metadata_train.csv')
    rows = raw.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    original, _ = v31.previous.make_splits(rows)
    splits = [{'training': original[f]['training'], 'probability_calibration': original[f]['selection'],
               'risk_calibration': original[f]['calibration'], 'assessment': original[f]['assessment']} for f in [0, 1]]
    chosen = np.unique(np.concatenate([v for split in splits for v in split.values()]))
    store, features = r1_prepare(root, rows, raw[['surveyId', 'speciesId']], chosen, temporary/'features', guard)
    del raw
    lookup = pd.Index(store.train_ids)
    mapped = [{role: lookup.get_indexer(rows.iloc[take].surveyId) for role, take in split.items()} for split in splits]
    results = []
    for fold, split in enumerate(mapped):
        if any(np.any(indices < 0) for indices in split.values()):
            raise ValueError('Missing source role IDs')
        for role, indices in split.items():
            digest = legacy.sha256_bytes(store.train_ids[indices].astype('<i8').tobytes())
            if digest != manifest['effective_role_hashes'][fold][role]:
                raise ValueError('R1 split role changed')
        results.append(infer_saved_fold(fold, split, store, source, temporary, export, guard, torch.device('cuda')))
    result = {'status': 'complete' if all(r['passed'] for r in results) else 'reproduction_failed',
        'new_neural_fits': 0, 'checkpoint_inferences': 6, 'fresh_assessment': False,
        'source_manifest_sha256': SOURCE_MANIFEST_SHA256, 'config_sha256': FOLLOWUP_CONFIG_SHA256,
        'runtime_hours': guard.elapsed_hours(), 'reproduction': results, 'features': features,
        'hardware': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__},
        'competition_submission_made': False}
    legacy.save_json(export/'inference_report.json', result)
    del store
    shutil.rmtree(temporary)
    total = sum(p.stat().st_size for p in export.iterdir())
    if total > FOLLOWUP['maximum_output_bytes']:
        raise ValueError('Inference output cap exceeded')
    legacy.save_json(export/'inference_manifest.json', {'config_sha256': FOLLOWUP_CONFIG_SHA256,
        'source_manifest_sha256': SOURCE_MANIFEST_SHA256,
        'outputs': {p.name: legacy.sha256_file(p) for p in export.iterdir()}, 'bytes_before_manifest': total})
    return result


try:
    RESULT = run_saved_inference()
    print(json.dumps({'status': RESULT['status'], 'runtime_hours': RESULT['runtime_hours'], 'new_neural_fits': 0}))
except Exception as error:
    legacy.save_json(Path('/kaggle/working/inference_failure.json'),
        {'status': 'failed', 'error_type': type(error).__name__, 'message': str(error)[:1500],
         'new_neural_fits': 0, 'competition_submission_made': False})
    shutil.rmtree('/kaggle/working/inference_runtime', ignore_errors=True)
    raise
