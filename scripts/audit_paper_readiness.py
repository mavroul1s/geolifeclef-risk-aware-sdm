"""Reproduce an exploratory evidence audit without training or using test labels.

PYTHONPATH=src:. python scripts/audit_paper_readiness.py
Optional official metadata belong in ignored data/raw/, never in results/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from geolifeclef.research_audit import cluster_bootstrap, prefix_f1, set_size_ceiling

ROOT = Path(__file__).resolve().parents[1]
CORE = ['Denmark', 'Netherlands']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_archived_hash(path, expected):
    """Distinguish exact files from Git's reversible CRLF normalization.

    Never modify an archive or accept a changed value merely because it parses.
    """
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual == expected:
        return {'path': str(path.relative_to(ROOT)), 'verification': 'exact_sha256',
                'checkout_sha256': actual, 'expected_sha256': expected}
    if path.suffix == '.csv':
        crlf = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if hashlib.sha256(crlf).hexdigest() == expected:
            return {'path': str(path.relative_to(ROOT)),
                    'verification': 'original_digest_recovered_by_CRLF_only',
                    'checkout_sha256': actual, 'expected_sha256': expected}
    raise ValueError(f'Archived output does not match manifest: {path}')


def archived_audit(output, repetitions):
    registry = json.loads((ROOT/'results/experiment_registry.json').read_text())
    history = pd.DataFrame(registry['official_submissions'])
    best = history.loc[history.private_score.idxmax()]
    target = registry['target']['private_score']
    history[['version', 'description', 'public_score', 'private_score', 'status']].to_csv(output/'official_history.csv', index=False)
    manifest_checks, comparisons, country_rows, inputs = [], [], [], {}
    for folder in sorted((ROOT/'results').glob('*_kaggle_output')):
        version = folder.name.split('_')[0]
        manifest_file = folder/f'{version}_manifest.json'
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            for name, expected in manifest.get('outputs', {}).items():
                path = folder/name
                manifest_checks.append(check_archived_hash(path, expected))
        matches = list(folder.glob('*per_survey*.csv'))
        if not matches:
            continue
        path = matches[0]
        inputs[str(path.relative_to(ROOT))] = digest(path)
        frame = pd.read_csv(path)
        if frame.surveyId.duplicated().any() or not np.isfinite(frame.delta_f1).all():
            raise ValueError(f'Invalid paired evidence: {path}')
        score_col = f'{version}_f1'
        baseline_cols = [c for c in frame if c.endswith('_f1') and c.startswith(('matched_', 'frozen_'))]
        if len(baseline_cols) != 1:
            raise ValueError(f'Ambiguous comparator: {path}')
        baseline = baseline_cols[0]
        if not np.allclose(frame[score_col]-frame[baseline], frame.delta_f1, atol=1e-10):
            raise ValueError(f'Paired deltas disagree: {path}')
        core = frame.country.isin(CORE)
        countries = frame.groupby('country').delta_f1.agg(['mean', 'size'])
        supported = countries[countries['size'] >= 30]
        boot = cluster_bootstrap(frame.delta_f1, frame.spatial_block, repetitions=repetitions)
        comparisons.append({'version': version, 'comparator': baseline,
            'surveys': len(frame), 'core_share': float(core.mean()),
            'pooled_gain': float(frame.delta_f1.mean()),
            'core_gain': float(frame.loc[core, 'delta_f1'].mean()),
            'outside_core_gain': float(frame.loc[~core, 'delta_f1'].mean()),
            'country_macro_gain_min30': float(supported['mean'].mean()),
            'equal_block_gain': float(frame.groupby('spatial_block').delta_f1.mean().mean()),
            'largest_block_share': float(frame.spatial_block.value_counts().max()/len(frame)),
            'spatial_bootstrap': boot,
            'fresh_assessment': False})
        country_rows.extend({'version': version, 'country': c, 'surveys': int(r['size']),
                             'gain': r['mean']} for c, r in countries.iterrows())
    pd.DataFrame(country_rows).to_csv(output/'country_gains.csv', index=False)
    duplicate_pairs = []
    for a, b in [(27, 28), (32, 33)]:
        pa = ROOT/f'results/v{a}_kaggle_output/GLC25_PA_submission_v{a}.csv'
        paths = list((ROOT/f'results/v{b}_kaggle_output').glob('*.csv'))
        same = [p.name for p in paths if digest(p) == digest(pa)]
        duplicate_pairs.append({'versions': [a, b], 'identical_csvs': same})
    return {'best_version': int(best.version), 'best_public': float(best.public_score),
            'best_private': float(best.private_score), 'winner_private_recorded': target,
            'absolute_gap': target-best.private_score,
            'relative_gain_needed': target/best.private_score-1,
            'best_gain_over_v18_relative': best.private_score/history.iloc[0].private_score-1,
            'registered_score_entries': len(history),
            'score_provenance': 'repository registry/screenshots, not a live score API attestation',
            'identical_submission_checks': duplicate_pairs,
            'verified_manifest_outputs': len(manifest_checks),
            'manifest_verification_details': manifest_checks, 'comparisons': comparisons,
            'input_sha256': inputs}


def prefix_audit(output):
    records, archive_info = [], []
    for path in sorted((ROOT/'results').glob('*_kaggle_output/*.npz')):
        archive = np.load(path, allow_pickle=False)
        if 'ranked_species_columns' not in archive:
            continue
        offsets = archive['true_offsets']
        truths = [archive['true_species_columns'][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        scores = prefix_f1(archive['ranked_species_columns'], truths)
        sizes = np.diff(offsets)
        ids = archive['survey_id']
        # Avoid double counting surveys appearing in both archived calibration views.
        # Compute each view's diagnostic first, then average by survey ID.
        names = archive['model_ids'] if 'model_ids' in archive else archive['expert_ids']
        archive_info.append({'archive': path.name, 'sha256': digest(path),
            'views': len(ids), 'unique_surveys': len(np.unique(ids)),
            'scope': str(archive['evidence_scope']), 'fresh_assessment': False})
        for expert, name in enumerate(names):
            for group, mask in [('all', np.ones(len(ids), bool)),
                                ('core', np.isin(archive['country'], CORE)),
                                ('outside_core', ~np.isin(archive['country'], CORE))]:
                s = scores[mask, expert]
                data = pd.DataFrame({'id': ids[mask], 'truth_size': sizes[mask],
                    'fixed18': s[:, 17], 'fixed24': s[:, 23],
                    'oracle_8_40': s[:, 7:40].max(1), 'oracle_1_K': s.max(1),
                    'size_only_ceiling_8_40': set_size_ceiling(sizes[mask], sizes[mask], 8, 40)})
                unique = data.groupby('id').mean()
                records.append({'archive': path.name, 'expert': str(name), 'group': group,
                    'surveys': len(unique), **unique.drop(columns='truth_size').mean().to_dict(),
                    'oracle_range_gap': float((unique.oracle_1_K-unique.oracle_8_40).mean()),
                    'oracle_rank_limit': scores.shape[2]})
    pd.DataFrame(records).to_csv(output/'prefix_oracle_diagnostics.csv', index=False)
    return {'archives': archive_info, 'diagnostics': records,
            'warning': 'Label-using, retrospectively selected oracle diagnostics; not deployable policies or expected gains. Top-K truncation prevents a full-ranking oracle. No probability calibration is inferred from reranking scores.'}


def metadata_audit(output, data_root):
    paths = [data_root/f'GLC25_PA_metadata_{s}.csv' for s in ['train', 'test']]
    if not all(p.exists() for p in paths):
        return {'status': 'not_run', 'reason': 'Official PA metadata unavailable'}
    raw = pd.read_csv(paths[0])
    pairs = raw[['surveyId', 'speciesId']].drop_duplicates()
    train = raw.drop_duplicates('surveyId').sort_values('surveyId').reset_index(drop=True)
    test = pd.read_csv(paths[1]).drop_duplicates('surveyId')
    if 'speciesId' in test and test.speciesId.notna().any():
        raise ValueError('Expected unlabeled official test metadata')
    train_country, test_country = train.country.value_counts(), test.country.value_counts()
    counts = pairs.groupby('surveyId').size()
    countries = sorted(set(train_country.index) | set(test_country.index))
    table = pd.DataFrame([{'country': c, 'train_surveys': int(train_country.get(c, 0)),
        'test_surveys': int(test_country.get(c, 0)),
        'train_fraction': train_country.get(c, 0)/len(train),
        'test_fraction': test_country.get(c, 0)/len(test)} for c in countries])
    table.to_csv(output/'country_support.csv', index=False)
    regression = pd.read_csv(ROOT/'results/v32_kaggle_output/regression_per_survey_v32.csv')
    supported = set(regression.country.unique())
    result = {'status': 'complete', 'input_sha256': {p.name: digest(p) for p in paths},
        'train_surveys': len(train), 'test_surveys': len(test), 'train_species': pairs.speciesId.nunique(),
        'train_core_fraction': float(train.country.isin(CORE).mean()),
        'test_core_fraction': float(test.country.isin(CORE).mean()),
        'test_fraction_in_countries_absent_from_PA': float((~test.country.isin(train_country.index)).mean()),
        'test_fraction_in_countries_absent_from_repeated_assessment': float((~test.country.isin(supported)).mean()),
        'train_richness': {str(k): float(v) for k, v in counts.quantile([0, .25, .5, .75, 1]).items()},
        'train_fraction_richness_below8': float((counts < 8).mean()),
        'train_fraction_richness_above40': float((counts > 40).mean()),
        'country_support': table.to_dict(orient='records')}
    # Reconstruct existing spatial folds; do not choose any new fold using scores.
    from scripts.v30_notebook_core import make_splits
    from sklearn.neighbors import BallTree
    splits, manifests = make_splits(train)
    expected = json.loads((ROOT/'results/v34_kaggle_output/v34_manifest.json').read_text())['splits']
    if any(a['id_hashes'] != b['id_hashes'] for a, b in zip(manifests, expected)):
        raise ValueError('Reconstructed development folds differ from archived comparator')
    result['reconstructed_fold_hashes_match'] = True
    proxy = []
    for number, split in enumerate(splits):
        train_ids = train.iloc[split['training']].surveyId
        assessment = train.iloc[split['assessment']]
        positive_species = set(pairs.loc[pairs.surveyId.isin(train_ids), 'speciesId'])
        available = pairs.assign(available=pairs.speciesId.isin(positive_species)).groupby('surveyId').available.sum()
        sizes = counts.reindex(assessment.surveyId).to_numpy()
        a = available.reindex(assessment.surveyId).to_numpy()
        proxy.append({'fold': number, 'training_positive_species': len(positive_species),
            'assessment_surveys': len(assessment), 'positive_label_mass_without_training_positive': float(1-a.sum()/sizes.sum()),
            'support_limited_oracle_f1': float(set_size_ceiling(sizes, a).mean()),
            'support_and_8_40_limited_oracle_f1': float(set_size_ceiling(sizes, a, 8, 40).mean())})
    result['proxy_support_diagnostics'] = proxy
    result['support_warning'] = 'Bounds apply only to predictors restricted to training-positive labels. The actual 5016-output model can output labels lacking fold-training positives; these are not its strict ceilings. Test label support is unknown.'
    tree = BallTree(np.deg2rad(train[['lat', 'lon']].to_numpy()), metric='haversine')
    distance = tree.query(np.deg2rad(test[['lat', 'lon']].to_numpy()), k=1)[0][:, 0]*6371.0088
    result['test_distance_km_to_full_PA_quantiles'] = {str(q): float(np.quantile(distance, q)) for q in [0, .5, .75, .9, .95, 1]}
    result['test_fraction_over100km_from_PA'] = float((distance > 100).mean())
    return result


def plot_findings(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = pd.read_csv(output/'official_history.csv')
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    ax[0].plot(history.version, history.private_score, 'o-', label='Private score (recorded)')
    ax[0].axhline(report['archived']['winner_private_recorded'], color='black', linestyle='--', label='Recorded winner')
    ax[0].set(xlabel='Experiment version', ylabel='Sample F1', title='External score history')
    ax[0].legend(fontsize=8)
    values = [r for r in report['archived']['comparisons'] if r['version'] in ['v30', 'v31', 'v32', 'v34']]
    x = np.arange(len(values))
    for offset, key, label in [(-.25, 'pooled_gain', 'Pooled'), (0, 'core_gain', 'DK/NL'), (.25, 'outside_core_gain', 'Outside DK/NL')]:
        ax[1].bar(x+offset, [r[key] for r in values], width=.24, label=label)
    ax[1].axhline(0, color='black', linewidth=.7)
    ax[1].set(xticks=x, xticklabels=[r['version'] for r in values], ylabel='Paired F1 change', title='Reused development evidence')
    ax[1].legend(fontsize=8)
    if report['metadata']['status'] == 'complete':
        c = pd.read_csv(output/'country_support.csv').sort_values('test_surveys', ascending=False).head(8)
        x = np.arange(len(c))
        ax[2].bar(x-.18, c.train_fraction, .35, label='PA train')
        ax[2].bar(x+.18, c.test_fraction, .35, label='Unlabeled test')
        ax[2].set(xticks=x, xticklabels=c.country, ylabel='Survey fraction', title='Geographic support mismatch')
        ax[2].tick_params(axis='x', labelrotation=55)
        ax[2].legend(fontsize=8)
    fig.savefig(output/'evidence_audit.png', dpi=180)
    fig.savefig(output/'evidence_audit.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/research_audit_2026_11_13')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--bootstrap-repetitions', type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'exploratory retrospective audit; no new trained model or hidden-test score',
        'source_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'archived': archived_audit(args.output, args.bootstrap_repetitions),
        'prefix_oracles': prefix_audit(args.output),
        'metadata': metadata_audit(args.output, args.data_root)}
    (args.output/'findings.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot_findings(args.output, report)
    print(json.dumps({'output': str(args.output), 'best_private': report['archived']['best_private'],
                      'metadata_status': report['metadata']['status'],
                      'manifest_outputs_verified': report['archived']['verified_manifest_outputs']}))


if __name__ == '__main__':
    main()
