"""Technical alignment overrides; keep original scientific choices and audit failures."""


def find_input_manifest(name):
    found = []
    for pattern in [f'*/{name}', f'*/*/{name}', f'notebooks/*/*/{name}']:
        found.extend(Path('/kaggle/input').glob(pattern))
    found = list({p.resolve() for p in found})
    if len(found) != 1:
        raise ValueError('Expected one attached input manifest: '+name)
    return found[0]


def install_aligned_inference():
    original_source = find_input_manifest('r1_export/r1_manifest.json')
    first_source = find_input_manifest('inference_export/inference_manifest.json')
    if legacy.sha256_file(first_source) != FIRST_INFERENCE_MANIFEST_SHA256:
        raise ValueError('Attempt-1 cache manifest changed')
    first_manifest = json.loads(first_source.read_text())
    original_manifest = json.loads(original_source.read_text())
    source = original_source.parent
    normalization = {}
    def restore(value):
        return {k: restore(v) for k, v in value.items()} if isinstance(value, dict) else np.asarray(value, np.float32)
    for fold in [0, 1]:
        name = f'fold{fold}_normalization.json'
        path = first_source.parent/name
        if legacy.sha256_file(path) != first_manifest['outputs'][name]:
            raise ValueError('Exported normalization digest mismatch')
        normalization[original_manifest['effective_role_hashes'][fold]['training']] = restore(json.loads(path.read_text()))
    def use_saved_normalization(store, indices, **kwargs):
        digest = legacy.sha256_bytes(store.train_ids[indices].astype('<i8').tobytes())
        if digest not in normalization:
            raise ValueError('Saved normalization does not match these training IDs')
        return normalization[digest]
    v29.fit_normalization = use_saved_normalization
    root = legacy.discover_data_root()
    rows = pd.read_csv(root/'GLC25_PA_metadata_train.csv', usecols=[
        'surveyId', 'lat', 'lon', 'country', 'region', 'year', 'areaInM2', 'geoUncertaintyInM'])
    rows = rows.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    original_splits, _ = v31.previous.make_splits(rows)
    required = set()
    for split in original_splits:
        for role in ['selection', 'calibration', 'assessment']:
            required.update(rows.iloc[split[role]].surveyId.tolist())
    original_reader = r1_read_predictors
    empty = (np.zeros((6, 4, 21), np.float16), np.zeros((4, 19, 12), np.float16), np.zeros((4, 64, 64), np.float16))
    def inference_reader(task):
        return original_reader(task) if int(task[1]) in required else empty
    globals()['r1_read_predictors'] = inference_reader
    original_constructor = v29.create_model
    def saved_ema_model(*args, **kwargs):
        return original_constructor(*args, **kwargs).requires_grad_(False)
    v29.create_model = saved_ema_model
    previous_sets, records, role_masks = {}, {}, {}
    for fold in [0, 1]:
        records[fold] = json.loads((source/f'fold{fold}_report.json').read_text())
        with np.load(source/f'fold{fold}_predictions.npz', allow_pickle=False) as archive:
            previous_sets[fold] = archive['member_top18_columns']
            role_masks[fold] = archive['role'] == 'assessment'
    original_predict = v29.predict
    audit = []
    def aligned_predict(model, store, indices, stats, device, config, **kwargs):
        fold = 0 if 'fold0' in config['id'] else 1
        count = records[fold]['role_counts']['probability_calibration']
        first = original_predict(model, store, indices[:count], stats, device, config, **kwargs)
        second = original_predict(model, store, indices[count:], stats, device, config, **kwargs)
        member = STUDY['neural_seeds'].index(config['seed'])
        gaps = []
        for begin in range(0, len(second), 256):
            stop = min(begin+256, len(second))
            gaps.extend(reference_topk_gap_ulps(second[begin:stop], previous_sets[fold][begin:stop, member]))
        gaps = np.asarray(gaps)
        audit.append({'fold': fold, 'seed': config['seed'],
            'reference_sets_within_two_raw_ULPs': float(np.mean(gaps <= 2)),
            'gap_ULPs_quantiles': np.quantile(gaps, [0, .99, .999, 1]).tolist()})
        return np.concatenate([first, second])
    v29.predict = aligned_predict
    return audit, len(required)


def finalize_aligned_report(result, audit, required):
    export = Path('/kaggle/working/inference_export')
    exact_pass = all(r['passed'] for r in result['reproduction'])
    numeric_pass = (all(r['survey_F1_match_fraction'] >= .999 and abs(r['mean_F1_difference']) <= .0001
                        for r in result['reproduction']) and
                    all(r['reference_sets_within_two_raw_ULPs'] >= .999 for r in audit))
    result['original_exact_member_set_guard_passed'] = exact_pass
    result['numerical_sensitivity_audit_passed'] = numeric_pass
    result['alignment_config_sha256'] = ALIGNMENT_CONFIG_SHA256
    result['normalization_reused_from_attempt_1'] = True
    result['predictor_surveys_read'] = required
    result['reference_topk_audit'] = audit
    result['status'] = 'complete' if exact_pass else ('complete_with_numerical_tie_sensitivity' if numeric_pass else 'reproduction_failed')
    legacy.save_json(export/'inference_report.json', result)
    manifest = json.loads((export/'inference_manifest.json').read_text())
    manifest['outputs']['inference_report.json'] = legacy.sha256_file(export/'inference_report.json')
    manifest['alignment_config_sha256'] = ALIGNMENT_CONFIG_SHA256
    manifest['bytes_before_manifest'] = sum((export/n).stat().st_size for n in manifest['outputs'])
    legacy.save_json(export/'inference_manifest.json', manifest)
    return result


try:
    ALIGNMENT_AUDIT, REQUIRED = install_aligned_inference()
    RESULT = run_saved_inference()
    RESULT = finalize_aligned_report(RESULT, ALIGNMENT_AUDIT, REQUIRED)
    print(json.dumps({'status': RESULT['status'], 'runtime_hours': RESULT['runtime_hours'], 'new_neural_fits': 0}))
except Exception as error:
    legacy.save_json(Path('/kaggle/working/inference_failure.json'), {'status': 'failed',
        'error_type': type(error).__name__, 'message': str(error)[:1500], 'new_neural_fits': 0})
    shutil.rmtree('/kaggle/working/inference_runtime', ignore_errors=True)
    raise
