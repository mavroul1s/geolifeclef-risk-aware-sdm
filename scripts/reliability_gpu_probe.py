"""Technical probe executed after frozen notebook definitions; no accuracy study."""


def run_gpu_probe():
    import importlib.metadata
    import rasterio
    from rasterio.enums import Resampling
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; stop before data preparation')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    root = legacy.discover_data_root()
    metadata = root/'GLC25_PA_metadata_train.csv'
    if legacy.sha256_file(metadata) != STUDY['metadata_sha256']:
        raise ValueError('Training metadata digest changed')
    frame = pd.read_csv(metadata)
    rows = frame.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    species = np.sort(frame.speciesId.unique())
    selected = rows.sample(n=16, random_state=20261113).sort_values('surveyId')
    samples = []
    io_started = time.monotonic()
    for survey in selected.surveyId:
        land_path, climate_path, image_path = legacy.feature_paths(root, 'PA-train', int(survey))
        land = legacy._clean_raw_tensor(torch.load(land_path, map_location='cpu', weights_only=True).numpy())
        climate = legacy._clean_raw_tensor(torch.load(climate_path, map_location='cpu', weights_only=True).numpy())
        if land.shape != (6, 4, 21) or climate.shape != (4, 19, 12):
            raise ValueError('Unexpected real tensor shape')
        with rasterio.open(image_path) as src:
            image = src.read(out_shape=(4, 64, 64), out_dtype='float32', resampling=Resampling.bilinear)
            order = v29.sentinel_band_indices(src.descriptions, [c.name for c in src.colorinterp])
        image = np.clip(np.nan_to_num(image[order]/10000, nan=0, posinf=0, neginf=0), 0, 2)
        red, green, _, nir = image
        image = np.concatenate([image, ((nir-red)/np.maximum(nir+red, 1e-4))[None],
                                ((green-nir)/np.maximum(green+nir, 1e-4))[None]])
        samples.append((land, climate, image))
    io_seconds = time.monotonic()-io_started
    x = {'landsat': torch.tensor(np.stack([s[0] for s in samples]), device=device),
         'bioclim': torch.tensor(np.stack([s[1] for s in samples]), device=device),
         'sentinel': torch.tensor(np.stack([s[2] for s in samples]), device=device),
         'environment': torch.zeros((16, 8), device=device)}
    static = torch.tensor(v29.candidate_static(selected), device=device)
    x['static_geo'] = static
    x['static_eco'] = torch.tensor(v29.candidate_static(selected, False), device=device)
    for name in ['landsat', 'bioclim']:
        x[name] = (x[name]-x[name].mean())/x[name].std().clamp_min(1e-4)
    model = v29.SensorAttention(8, static.shape[1], len(species), width=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    target = torch.zeros((16, len(species)), device=device)
    for i, survey in enumerate(selected.surveyId):
        cols = pd.Index(species).get_indexer(frame.loc[frame.surveyId == survey, 'speciesId'].unique())
        target[i, cols] = 1
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits, count = model(x, True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite real-data GPU backward')
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    output = Path('/kaggle/working/gpu_probe.json')
    report = {'status': 'technical_probe_passed', 'gpu': torch.cuda.get_device_name(device),
        'gpu_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'species': len(species), 'real_surveys_checked': 16, 'io_seconds': io_seconds,
        'toy_backward_losses_not_accuracy': losses, 'seconds': time.monotonic()-started,
        'torch': torch.__version__, 'sklearn': importlib.metadata.version('scikit-learn'),
        'scientific_performance_claim': False, 'submission_made': False}
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


print(json.dumps(run_gpu_probe(), indent=2))
