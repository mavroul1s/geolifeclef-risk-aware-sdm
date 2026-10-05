"""Real-data identity audit and v34 postmortem; no model training or submission."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import sparse
from scripts import v35_notebook_core as core

ROOT = Path(__file__).resolve().parents[1]


class DenseSlices:
    def __init__(self,matrix):
        self.matrix = matrix
    def __len__(self):
        return self.matrix.shape[0]
    def __getitem__(self,take):
        result = self.matrix[take].toarray()
        return result.ravel() if np.isscalar(take) else result


def main():
    folder = ROOT/'results/v34_kaggle_output'
    manifest = json.loads((folder/'v34_manifest.json').read_text())
    for name,digest in manifest['outputs'].items():
        assert Path(name).name == name
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    report = json.loads((folder/'v34_report.json').read_text())
    old = pd.read_csv(ROOT/'results/v32_kaggle_output/GLC25_PA_submission_v32.csv')
    new = pd.read_csv(folder/'GLC25_PA_submission_v34.csv')
    assert np.array_equal(old.surveyId,new.surveyId)
    old_sets,new_sets = [[set(x.split()) for x in f.predictions] for f in (old,new)]
    swaps = np.array([len(b-a) for a,b in zip(old_sets,new_sets)])
    assert all(len(a) == len(b) for a,b in zip(old_sets,new_sets))
    raw = ROOT/'artifacts/v20_frozen/raw'
    pairs = pd.read_csv(raw/'GLC25_PA_metadata_train.csv',usecols=['surveyId','speciesId','lat','lon','country'])
    rows = pairs.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    test_rows = pd.read_csv(raw/'GLC25_PA_metadata_test.csv')
    notebook = json.loads((ROOT/'notebooks/geolifeclef_v35_full_set_balanced_attention.ipynb').read_text(encoding='utf-8'))
    space = {}; exec(''.join(notebook['cells'][2]['source']),space)
    core.REFERENCE_PAYLOAD_HASH,core.REFERENCE_RAW_HASH = space['REFERENCE_PAYLOAD_HASH'],space['REFERENCE_RAW_HASH']
    reference = core.load_reference(space['REFERENCE_B64'])
    assert core.legacy.sha256_file(raw/'GLC25_PA_metadata_train.csv') == reference['train_metadata_sha256']
    species = np.load(ROOT/'artifacts/v20_frozen_bundle_v21/species_ids.npy')
    ri = pd.Index(rows.surveyId).get_indexer(pairs.surveyId)
    ci = pd.Index(species).get_indexer(pairs.speciesId)
    assert min(ri.min(),ci.min()) >= 0
    y = sparse.csr_matrix((np.ones(len(ri),np.uint8),(ri,ci)),shape=(len(rows),len(species)))
    y.data[:] = 1
    labels = DenseSlices(y)
    splits,manifests = core.v31.previous.make_splits(rows)
    bound = core.bind_reference(reference,rows,species,splits,manifests,labels)
    summary = {'experiment':report['experiment'],'status':'complete_official_regression_keep_v32','recorded_at':'2026-10-05',
        'score_source':'user-provided Kaggle screenshot; not independently API verified',
        'public_score':.24203,'private_score':.21483,'public_delta_from_v32':-.00022,'private_delta_from_v32':-.00031,
        'runtime_hours':report['runtime_hours'],'hardware':report['hardware'],'selected_policy':report['policy'],
        'development_gates_passed':report['eligible_for_submission'],'repeated_regression':report['regression'],
        'changed_test_sets':int((swaps>0).sum()),'total_test_replacements':int(swaps.sum()),
        'every_v32_row_count_preserved':True,'output_bytes':sum(p.stat().st_size for p in folder.iterdir()),
        'train_country_counts':rows.country.value_counts().to_dict(),'test_country_counts':test_rows.country.value_counts().to_dict(),
        'train_denmark_netherlands_fraction':float(rows.country.isin(core.v31.CORE_COUNTRIES).mean()),
        'test_denmark_netherlands_fraction':float(test_rows.country.isin(core.v31.CORE_COUNTRIES).mean()),
        'diagnosis':'The graph policy passed weak repeated-development gates (CI lower ~4.27e-7) but regressed on both hidden splits. No execution/schema defect is established. Country/domain shift and adaptive development are limitations, not proven causal explanations for this small regression.',
        'next_version':'v35 full-set checkpoint-selected neural ensemble; cached benchmark saves reference retraining; larger promotion margins',
        'fresh_assessment':False,'artifacts':{'directory':'results/v34_kaggle_output',
            'screenshot':'results/v34_kaggle_scores.png','screenshot_sha256':core.legacy.sha256_file(ROOT/'results/v34_kaggle_scores.png'),
            'source_zip_at_ingestion':'C:/Users/nickb/Downloads/results.zip',
            'source_zip_sha256':'2cf04786c792158e501a51a23c2d76dd2b4818bdc2d544dad15024fd86b14433',
            'hashes':{p.name:core.legacy.sha256_file(p) for p in sorted(folder.iterdir())}}}
    core.legacy.save_json(ROOT/'results/v34_summary.json',summary)
    # Exercise the actual compact exporter at its maximum planned row count.
    # Random ranks/probabilities give a conservative compressibility check.
    rng = np.random.default_rng(3500)
    probe_rows = rows.iloc[:4000].copy().reset_index(drop=True)
    probe_labels = (rng.random((4000,64)) < .1).astype(np.uint8)
    probe_labels = np.pad(probe_labels,((0,0),(0,len(species)-64)))
    bundles = []
    for f in range(2):
        data = {}
        for role,start,n in (('calibration',f*2000,1500),('assessment',f*2000+1500,500)):
            members = []
            for _ in core.CONFIGS:
                p = np.zeros((n,len(species)),np.float16)
                for row in range(n):
                    columns = rng.choice(len(species),128,replace=False)
                    p[row,columns] = rng.uniform(1e-5,.95,128).astype(np.float16)
                members.append(p)
            data[role] = {'indices':np.arange(start,start+n),'members':members,'reference_f1':np.zeros(n)}
        bundles.append({'number':f,'data':data})
    with tempfile.TemporaryDirectory(prefix='v35-capacity-',dir=ROOT/'artifacts') as directory:
        path = Path(directory)/'predictions.npz'
        core.save_compact(path,bundles,probe_rows,SimpleNamespace(labels=probe_labels,species_ids=species))
        npz_bytes = path.stat().st_size
    # <=40 species, each up to five digits plus a separator; generous CSV reserve.
    projection = npz_bytes+4_000_000+3_000_000+1_000_000
    result = {'scope':'real official-metadata/label identity check; capacity uses synthetic probabilities, not accuracy evidence',
        'reference_payload_sha256':space['REFERENCE_PAYLOAD_HASH'],'source_sha256':space['V35_SOURCE_HASH'],
        'metadata_sha256_verified':True,'all_pa_label_bytes_verified':True,'pa_surveys':len(rows),'species':len(species),
        'all_training_selection_calibration_assessment_id_hashes_verified':True,
        'calibration_rows':[len(b['calibration']['indices']) for b in bound],
        'assessment_rows':[len(b['assessment']['indices']) for b in bound],
        'calibration_f1_recomputed_from_official_labels':True,'old_reference_neural_fits_this_run':0,
        'capacity_npz_bytes_4000_rows_3_models':npz_bytes,
        'projected_total_bytes_with_4MB_submission_3MB_diagnostics_1MB_JSON':projection,
        'within_16MB':projection <= 16_000_000,'full_kaggle_runtime_measured':False}
    if not result['within_16MB']:
        raise ValueError('Output capacity estimate exceeds contract')
    core.legacy.save_json(ROOT/'results/v35_reference_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
