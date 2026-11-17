"""Explicitly post-hoc diagnosis of empty-set scoring and genus constraints."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from geolifeclef.genus_completion_bounds import positive_genus_f1_bounds
from geolifeclef.taxonomic_censoring import censor_genera, paired_spatial_interval
from geolifeclef.partial_set_bounds import linear_contrast_bounds
from run_taxonomic_resolution_t1 import DATA, OUT, sha, save, now


def main():
    target = OUT/'posthoc'; target.mkdir(exist_ok=True)
    if (target/'summary.json').exists():
        raise ValueError('Post-hoc diagnosis already complete')
    sources = ['scripts/analyze_taxonomic_resolution_t1.py', 'src/geolifeclef/genus_completion_bounds.py',
               'tests/test_genus_completion_bounds.py', 'docs/taxonomic_resolution_t1_posthoc.md']
    originals = [OUT/'summary.json', OUT/'replay_summary.csv', OUT/'mask_receipts.json', OUT/'validation.json',
                 DATA/'oracle.npz', DATA/'replay_cells.npz', DATA/'fixed_policy.npz', DATA/'contract.npz']
    identities = {str(p.relative_to(ROOT)): sha(p) for p in originals}
    save(target/'analysis_lock.json', {'at_utc': now(), 'posthoc_after_primary': True,
        'primary_and_cache_identities': identities, 'sources': {name: sha(ROOT/name) for name in sources}})
    for name in sources:
        dest = target/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    config = json.loads((ROOT/'configs/taxonomic_resolution_t1.json').read_text())
    config_boot = {k: v for k, v in config['bootstrap'].items() if k != 'cluster'}
    c = pd.read_csv(DATA/'assessment_roles.csv'); contract = np.load(DATA/'contract.npz')
    policy = np.load(DATA/'fixed_policy.npz'); oracle = np.load(DATA/'oracle.npz')
    replay = np.load(DATA/'replay_cells.npz'); table = pd.read_csv(OUT/'replay_summary.csv')
    receipts = json.loads((OUT/'mask_receipts.json').read_text())
    y, q = oracle['truth'], oracle['F1']; empty = policy['count'] == 0
    positive_count = (y > 0).sum(1); qzero = np.where(empty, 0, q)
    chosen = {name: policy['accept_'+name] for name in config['selectors']}
    oracle_zero = {'full_F1': float(qzero.mean()), 'contrasts': {}}
    for a, b in config['contrasts']:
        oracle_zero['contrasts'][a+'_'+b] = paired_spatial_interval(qzero, c.spatial_block, chosen[a], chosen[b], **config_boot)
    rows, lows, highs = [], [], []
    for index, record in enumerate(receipts):
        mechanism, dose, seed = record['mechanism'], record['dose'], record['seed']
        weights = np.ones_like(contract['genus_weights']) if mechanism.startswith('uniform') else contract['genus_weights']
        known, _, _ = censor_genera(y, contract['eligible'], contract['genus_codes'], weights, c.cell_id, dose, seed)
        assert hashlib.sha256(np.packbits(known, bitorder='little').tobytes()).hexdigest() == record['known_mask_sha256']
        truth = np.where(known, y, -1)
        lower, upper = positive_genus_f1_bounds(truth, known, contract['eligible'], contract['genus_codes'],
                                                policy['rank'], policy['count'])
        assert np.all(lower <= q+1e-12) and np.all(q <= upper+1e-12)
        assert np.all(lower >= replay['lower'][index]-1e-12) and np.all(upper <= replay['upper'][index]+1e-12)
        lows.append(lower); highs.append(upper)
        false_empty = empty&(positive_count > 0)&((truth > 0).sum(1) == 0)
        observed_naive, observed_known = replay['unknown_as_absence'][index], replay['known_labels_only'][index]
        assert np.all(observed_naive[false_empty] == 1) and np.all(observed_known[false_empty] == 1)
        row = {'mechanism': mechanism, 'dose': dose, 'seed': seed,
            'false_perfect_empty_cells': int(false_empty.sum()),
            'false_perfect_bias_contribution': float(false_empty.mean()),
            'independent_bound_width': float((replay['upper'][index]-replay['lower'][index]).mean()),
            'genus_positive_bound_width': float((upper-lower).mean())}
        for treatment in ['unknown_as_absence', 'known_labels_only']:
            scores = np.where(empty, 0, replay[treatment][index])
            row[treatment+'_zero_empty_bias'] = float((scores-qzero).mean())
            # Exact decomposition, not a fitted explanation.
            assert np.isclose((replay[treatment][index]-q).mean(),
                              (scores-qzero).mean()+false_empty.mean(), atol=1e-12)
            for a, b in config['contrasts']:
                key = a+'_'+b
                row[treatment+'_zero_empty_'+key] = float(scores[chosen[a]].mean()-scores[chosen[b]].mean())
        for a, b in config['contrasts']:
            w = chosen[a]/chosen[a].sum()-chosen[b]/chosen[b].sum()
            bounds = linear_contrast_bounds(lower, upper, w)
            row['genus_bound_'+a+'_'+b+'_lower'], row['genus_bound_'+a+'_'+b+'_upper'] = bounds
            row['genus_bound_'+a+'_'+b+'_sign_identified'] = bool(bounds[0] > 0 or bounds[1] < 0)
        rows.append(row)
        if (index+1)%10 == 0:
            print('Post-hoc genus-completion checks', index+1, flush=True)
    lower, upper = np.asarray(lows), np.asarray(highs)
    np.savez_compressed(DATA/'posthoc_genus_bounds.npz', lower=lower, upper=upper)
    diagnostic = pd.DataFrame(rows); diagnostic.to_csv(target/'replay_summary.csv', index=False)
    sections = []
    for mechanism in config['mechanisms']:
        for dose in config['doses']:
            ix = np.flatnonzero((table.mechanism == mechanism)&(table.dose == dose))
            section = {'mechanism': mechanism, 'dose': dose,
                'false_perfect_empty_cells_range': [int(diagnostic.iloc[ix].false_perfect_empty_cells.min()), int(diagnostic.iloc[ix].false_perfect_empty_cells.max())],
                'independent_bound_width': float(diagnostic.iloc[ix].independent_bound_width.mean()),
                'genus_positive_bound_width': float(diagnostic.iloc[ix].genus_positive_bound_width.mean()),
                'treatments_zero_empty': {}, 'genus_contrasts': {}}
            for treatment in ['unknown_as_absence', 'known_labels_only']:
                scores = np.where(empty[None, :], 0, replay[treatment][ix]).mean(0)
                value = {'bias': paired_spatial_interval(scores-qzero, c.spatial_block, **config_boot), 'contrasts': {}}
                for a, b in config['contrasts']:
                    value['contrasts'][a+'_'+b] = {'observed': paired_spatial_interval(scores, c.spatial_block, chosen[a], chosen[b], **config_boot),
                        'distortion': paired_spatial_interval(scores-qzero, c.spatial_block, chosen[a], chosen[b], **config_boot)}
                section['treatments_zero_empty'][treatment] = value
            for a, b in config['contrasts']:
                section['genus_contrasts'][a+'_'+b] = {
                    'lower_seed_mean': float(diagnostic.iloc[ix]['genus_bound_'+a+'_'+b+'_lower'].mean()),
                    'upper_seed_mean': float(diagnostic.iloc[ix]['genus_bound_'+a+'_'+b+'_upper'].mean()),
                    'sign_identified_seeds': int(diagnostic.iloc[ix]['genus_bound_'+a+'_'+b+'_sign_identified'].sum())}
            sections.append(section)
    for name, digest in identities.items():
        assert sha(ROOT/name) == digest
    result = {'at_utc': now(), 'posthoc': True, 'primary_unchanged': True,
        'empty_prediction_cells': int(empty.sum()), 'complete_empty_empty_cells': int((empty&(positive_count == 0)).sum()),
        'complete_positive_truth_empty_prediction_cells': int((empty&(positive_count > 0)).sum()),
        'oracle_zero_empty': oracle_zero, 'sections': sections,
        'validated_oracle_containment_and_nested_bounds': len(receipts)*len(c),
        'assumption': 'simulator supplies at least one eligible positive in every fully censored genus; not inferred for NEON',
        'new_method_or_primary_success': False}
    save(target/'summary.json', result)
    plot(json.loads((OUT/'summary.json').read_text()), result)


def plot(primary, diagnostic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    colors = {'uniform_positive_genera': '#245da8', 'rarity_weighted_positive_genera': '#a94245'}
    labels = {'uniform_positive_genera': 'Uniform genera', 'rarity_weighted_positive_genera': 'Rarity weighted'}
    for mechanism, color in colors.items():
        sections = [s for s in primary['sections'] if s['mechanism'] == mechanism]
        x = np.array([s['dose']*100 for s in sections])
        ax = axes[0, 0]
        ax.plot(x, [100*s['diagnostics']['known_fraction_mean']['mean'] for s in sections], 'o-', color=color, label=labels[mechanism]+': classes known')
        ax.plot(x, [100*s['diagnostics']['hidden_positive_occurrence_fraction']['mean'] for s in sections], 's--', color=color, label=labels[mechanism]+': presences hidden')
        ax = axes[0, 1]
        for treatment, style, suffix in [('unknown_as_absence', '--', 'absence'), ('known_labels_only', '-', 'known only')]:
            values = [s['treatments'][treatment]['bias'] for s in sections]
            points = np.array([v['point'] for v in values])
            ax.errorbar(x+(1 if mechanism.startswith('rarity') else -1), points,
                yerr=np.array([[p-v['interval'][0] for p, v in zip(points, values)], [v['interval'][1]-p for p, v in zip(points, values)]]),
                color=color, linestyle=style, marker='o', capsize=3, label=labels[mechanism]+': '+suffix)
        ax = axes[1, 0]
        post = [s for s in diagnostic['sections'] if s['mechanism'] == mechanism]
        ax.plot(x, [s['independent_bound_width'] for s in post], 'o--', color=color, label=labels[mechanism]+': independent')
        ax.plot(x, [s['genus_positive_bound_width'] for s in post], 's-', color=color, label=labels[mechanism]+': positive genus*')
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel('Nominal positive-genus censoring dose (%)'); ax.set_xticks([0, 20, 50]); ax.grid(alpha=.2)
        ax.legend(fontsize=7.8)
    axes[0, 0].set_ylabel('Fraction (%)'); axes[0, 0].set_title('Label counts conceal lost positive records')
    axes[0, 1].axhline(0, color='black', linewidth=.8); axes[0, 1].axhspan(-.01, .01, color='grey', alpha=.12)
    axes[0, 1].set_ylabel('Masked F1 − complete-label F1'); axes[0, 1].set_title('Primary score bias: 97.5% spatial intervals')
    axes[1, 0].set_ylabel('Mean full-set F1 identification width'); axes[1, 0].set_title('Retaining genus information tightens bounds*')
    ax = axes[1, 1]
    contrasts = [('HGB_SDC', 'HGB − SDC'), ('SDC_random', 'SDC − random')]
    ypos, ticknames = [], []
    for j, (key, label) in enumerate(contrasts):
        for i, mechanism in enumerate(colors):
            y = j*3+i; ypos.append(y); ticknames.append(label+' / '+labels[mechanism])
            s = next(s for s in primary['sections'] if s['dose'] == .2 and s['mechanism'] == mechanism)
            post = next(s for s in diagnostic['sections'] if s['dose'] == .2 and s['mechanism'] == mechanism)
            b = s['treatments']['known_labels_only']['contrasts'][key]
            ax.plot([b['completion_lower']['mean'], b['completion_upper']['mean']], [y+.12, y+.12], '--', color=colors[mechanism], linewidth=2)
            g = post['genus_contrasts'][key]
            ax.plot([g['lower_seed_mean'], g['upper_seed_mean']], [y-.12, y-.12], '-', color=colors[mechanism], linewidth=3)
            ax.plot(primary['oracle']['contrasts'][key]['point'], y, 'k|', markersize=11)
    ax.set_yticks(ypos, ticknames, fontsize=8); ax.axvline(0, color='black', linewidth=.8)
    ax.set_xlabel('Paired selective-F1 identification range'); ax.set_title('20% dose; seed-mean endpoints, not CIs')
    ax.text(.01, -.29, 'Dashed: independent; solid: genus-positive*; black: oracle', transform=ax.transAxes, fontsize=8)
    fig.suptitle('T1 development stress test: fixed predictions on 5,251 consumed SAGE cells\n*Post-hoc genus constraints supplied by the simulator; field applicability unestablished', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT/('taxonomic_resolution_t1.'+suffix), dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    main()
