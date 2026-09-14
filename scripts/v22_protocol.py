"""Fixed v22 cross-fitting and calibration; no assessment-dependent choices."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from scripts.ood_po_protocol import (spatial_block_ids, nearest_training_support,
    make_partitions, sample_f1, EARTH_RADIUS_KM)
from scripts.prepare_environmental_challenger import spatial_partitions

SEED = 20250922
ALPHAS = (0., .025, .05, .1, .2, .35, .5)


def cap_publisher_weights(weights, codes, maximum_share=.30):
    weights = np.asarray(weights, dtype=np.float64)
    codes = np.asarray(codes)
    if weights.ndim != 1 or weights.shape != codes.shape or not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid sampling weights")
    _, inverse = np.unique(codes, return_inverse=True)
    masses = np.bincount(inverse, weights=weights)
    cap = max(float(maximum_share), 1 / len(masses))
    if not 0 < cap <= 1:
        raise ValueError("Invalid publisher cap")
    target, free, remaining = np.zeros(len(masses)), np.ones(len(masses), bool), 1.
    while free.any():
        proposal = remaining * masses[free] / masses[free].sum()
        capped = proposal > cap + 1e-14
        if not capped.any():
            target[free] = proposal
            break
        hit = np.flatnonzero(free)[capped]
        target[hit] = cap
        free[hit] = False
        remaining = 1 - target.sum()
    return weights * (target / masses)[inverse]


def hashed_blocks(rows, salt):
    # SHA avoids the strong correlations between consecutive XOR salts.
    return np.array([int.from_bytes(hashlib.sha256(f'{salt}:{b}'.encode()).digest()[:8], 'little') % 100
                     for b in spatial_block_ids(rows)], dtype=np.int64)


def crossfit_partitions(rows, minimum=100):
    """Two disjoint outer folds, drawn only from unassessed v21 training rows.

    Inner development may use consumed v21 data, never this fold's assessment.
    Models are separately fitted; no outer score is read until both paths freeze.
    """
    original = spatial_partitions(rows)
    previous, _, _ = make_partitions(rows, original, minimum_partition_size=minimum)
    fresh = previous == 0
    bucket = hashed_blocks(rows, SEED)
    outer_fold = np.full(len(rows), -1, dtype=np.int8)
    outer_fold[fresh & (bucket < 20)] = 0
    outer_fold[fresh & (bucket >= 20) & (bucket < 40)] = 1
    # Exclude an entire assessment block even where old membership differs.
    block = spatial_block_ids(rows)
    result = []
    coordinates = rows[['lat', 'lon']].to_numpy(dtype=np.float64)
    for fold in (0, 1):
        inner = hashed_blocks(rows, SEED + 100 + fold)
        split = np.select([inner < 10, inner < 20], [1, 2], default=0).astype(np.int8)
        split[original != 0] = -1
        heldout_blocks = np.unique(block[outer_fold == fold])
        split[np.isin(block, heldout_blocks)] = -1
        split[outer_fold == fold] = 3
        # Globally reserve both outer folds from selection/calibration, so
        # neither fold's label can select any model or policy in the other.
        split[(outer_fold >= 0) & (split > 0) & (split != 3)] = -1
        eval_ix = split > 0
        train = np.flatnonzero(split == 0)
        if not eval_ix.any() or not len(train):
            raise ValueError('Empty geographic split; no adaptive seed retry')
        tree = BallTree(np.deg2rad(coordinates[eval_ix]), metric='haversine')
        distance = tree.query(np.deg2rad(coordinates[train]), k=1)[0][:, 0] * EARTH_RADIUS_KM
        split[train[distance < 20]] = -1
        counts = {name:int((split == i).sum()) for i,name in enumerate(('training','checkpoint_selection','calibration','assessment'))}
        if min(counts.values()) < minimum:
            raise ValueError(f'Insufficient preregistered fold {fold}: {counts}')
        support = nearest_training_support(rows.loc[split == 0], rows)['distance_km']
        assert np.all(support[split > 0] >= 20 - 1e-7)
        assert np.all(previous[split == 3] == 0)
        manifest = {'fold':fold, 'seed':SEED, 'partition_counts':counts,
            'assessment_v21_training_only':True, 'minimum_evaluation_distance_km':float(support[split > 0].min()),
            'assessment_blocks':len(heldout_blocks), 'partition_countries':{
                name:rows.loc[split == i, 'country'].fillna('unknown').value_counts().to_dict()
                for i,name in enumerate(counts)},
            'assessment_ids_sha256':hashlib.sha256(np.sort(rows.surveyId.to_numpy()[split == 3]).astype('<i8').tobytes()).hexdigest()}
        result.append((split, support, manifest))
    assert not np.any((result[0][0] == 3) & (result[1][0] == 3))
    return result


def mix(base, expert, distances, policy):
    if policy.get('alpha') not in ALPHAS or policy.get('gate') not in ('uniform','pa_distance') or policy.get('k') != 20:
        raise ValueError('Unregistered v22 policy')
    if base.shape != expert.shape or np.asarray(distances).shape != (len(base),):
        raise ValueError('Mixture shape mismatch')
    if not all(np.isfinite(v).all() for v in (base, expert, distances)) or (np.asarray(distances) < 0).any():
        raise ValueError('Invalid mixture values')
    if (base < 0).any() or (base > 1).any() or (expert < 0).any() or (expert > 1).any():
        raise ValueError('Invalid probabilities')
    if policy['alpha'] == 0:
        return base
    gate = np.ones(len(base)) if policy['gate'] == 'uniform' else np.clip(np.log1p(distances)/np.log(51),0,1)
    weight = (policy['alpha'] * gate).astype(np.float32)[:,None]
    return (1-weight)*base.astype(np.float32) + weight*expert.astype(np.float32)


def select_policy(labels, base, expert, distances):
    trials=[]
    for alpha in ALPHAS:
        for gate in ('uniform','pa_distance'):
            policy={'alpha':alpha,'gate':gate,'k':20}
            policy['calibration_f1']=float(sample_f1(labels,mix(base,expert,distances,policy)).mean())
            trials.append(policy)
    # Stable ordering prefers the smallest intervention on exact ties.
    return max(trials,key=lambda p:p['calibration_f1']),trials


CHECKPOINT_POLICY = {'alpha':.1, 'gate':'pa_distance', 'k':20}
