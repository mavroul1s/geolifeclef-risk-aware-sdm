"""Rejected exploratory v34 hypothesis, NOT the released notebook core.

Repeated development only: every PA survey has already been assessed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import scripts.v32_notebook_core as previous

v31, v29, legacy = previous.previous, previous.v29, previous.legacy
EXPERIMENT = "v34_learned_f1_set_editor"
CONTROL_HASH = "71c863d9cb05f3efc63ac54dd7faa05efd2fc137a11da9adcd2064c82c88dfee"
MAX_HOURS = 10.75
FROZEN_V32_POLICY = {"id": "bag1_specialist1_swap4", "bag": 1., "specialist": 1., "swaps": 4}
META_ITERATIONS = 120
META_MAX_ROWS = 6000
FEATURE_NAMES = ("in_base", "base_position", "base_count", "bag_p", "specialist_p",
    "bag_rank", "specialist_rank", "bag_missing", "specialist_missing", "probability_product",
    "probability_difference", "bag_base_mass", "specialist_base_mass", "bag_top128_mass",
    "specialist_top128_mass", "bag_top10_mass", "specialist_top10_mass", "bag_tail_p", "specialist_tail_p")
POLICIES = ({"id": "control", "swaps": 0, "count_delta": 0, "margin": 0.},) + tuple(
    {"id": f"{name}_margin{margin:g}", "swaps": swaps, "count_delta": delta, "margin": margin}
    for name, swaps, delta in (("swap2", 2, 0), ("swap4", 4, 0), ("count1", 0, 1),
                               ("count2", 0, 2), ("joint2", 2, 2))
    for margin in (0., .0005))


def edit_features(base, bag, specialist, row):
    """No labels, species identity, country, coordinates or survey ID features.

    The candidate pool is the base plus the union of top64 from both experts.
    Missing probabilities are explicit truncated observations, NOT calibrated zeros.
    """
    base = np.asarray(base, dtype=np.int64)
    candidate = np.unique(np.r_[base, bag["rank"][row, :64], specialist["rank"][row, :64]]).astype(np.int64)
    positions = {int(s): i for i, s in enumerate(base)}
    inside = np.array([int(s) in positions for s in candidate])
    pos = np.array([positions.get(int(s), 40) for s in candidate], np.float32)
    probabilities, ranks, missing, summaries = [], [], [], []
    for expert in (bag, specialist):
        ranking = expert["rank"][row].astype(np.int64)
        value = expert["value"][row].astype(np.float32)
        lookup = {int(s): i for i, s in enumerate(ranking)}
        rank = np.array([lookup.get(int(s), len(ranking)) for s in candidate])
        probability = np.where(rank < len(value), value[np.minimum(rank, len(value)-1)], value[-1]*.5)
        probabilities.append(probability); ranks.append(rank/128.); missing.append(rank == len(value))
        summaries.append((float(probability[inside].sum()), float(value.sum()), float(value[:10].sum()), float(value[-1])))
    p, q = probabilities
    features = np.column_stack((inside, pos/40., np.full(len(candidate), len(base)/40.), p, q,
        *ranks, *missing, p*q, p-q,
        *[np.full(len(candidate), summaries[g][s]/(40. if s < 3 else 1.)) for s in range(4) for g in range(2)]))
    return candidate, np.asarray(features, np.float32)


def edit_targets(base, candidates, truth):
    """Exact single-addition/single-removal F1 deltas, scaled by 100.

    Multi-edit decoding uses these as a local approximation; actual set F1,
    not the sum of predicted utilities, is evaluated for every policy.
    """
    truth = np.asarray(truth)
    base_set = set(base)
    inside = np.array([int(c) in base_set for c in candidates])
    tp = float(truth[np.asarray(base, np.int64)].sum())
    denominator = len(base)+float(truth.sum())
    y = truth[candidates].astype(np.float32)
    change = np.where(inside, -1., 1.)
    return (100.*(2.*(tp+change*y)/(denominator+change)-2.*tp/denominator)).astype(np.float32)


def fit_editor(data, labels, row_weights=None, guard=None):
    """Fit only supplied meta-training rows. Caller enforces role isolation."""
    if len(data["base"]) != len(labels) or not len(labels):
        raise ValueError("Meta-training rows/labels mismatch")
    row_weights = np.ones(len(labels)) if row_weights is None else np.asarray(row_weights)
    features, targets, weights = [], [], []
    for i, base in enumerate(data["base"]):
        if guard and i % 128 == 0:
            guard.require(45*60, "meta-training features")
        candidate, x = edit_features(base, data["bag"], data["specialist"], i)
        features.append(x); targets.append(edit_targets(base, candidate, labels[i]))
        weights.append(np.full(len(candidate), row_weights[i]/len(candidate), np.float32))
    x, y, weight = np.concatenate(features), np.concatenate(targets), np.concatenate(weights)
    weight /= weight.mean()
    model = HistGradientBoostingRegressor(learning_rate=.05, max_iter=20, max_leaf_nodes=15,
        max_depth=4, min_samples_leaf=100, l2_regularization=25., max_bins=63,
        early_stopping=False, warm_start=True, random_state=20263401)
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for iterations in range(20, META_ITERATIONS+1, 20):
            if guard:
                guard.require(45*60, "bounded meta-regression training")
            model.set_params(max_iter=iterations)
            model.fit(x, y, sample_weight=weight)
    return model, {"rows": len(labels), "candidate_pairs": len(y), "iterations": model.n_iter_,
        "seconds": time.monotonic()-started, "features": list(FEATURE_NAMES),
        "objective": "squared error of exact single-edit sample-F1 deltas; targets scaled by 100",
        "early_stopping": False, "country_or_id_features": False, "equal_total_weight_per_survey_before_country_weights": True}


def score_edits(data, model, guard=None):
    result = []
    with threadpool_limits(limits=4):
        for start in range(0, len(data["base"]), 128):
            if guard:
                guard.require(15*60, "meta-editor inference")
            batch = [edit_features(data["base"][i], data["bag"], data["specialist"], i)
                     for i in range(start, min(start+128, len(data["base"])))]
            scores = model.predict(np.concatenate([x for _, x in batch]))/100.
            offset = 0
            for candidates, _ in batch:
                result.append((candidates, scores[offset:offset+len(candidates)]))
                offset += len(candidates)
    return result


def decode(base, utilities, policy, guard=None):
    if policy["id"] == "control":
        return [list(row) for row in base]
    if len(base) != len(utilities):
        raise ValueError("Editor row mismatch")
    output = []
    for i, (old, (candidate, gain)) in enumerate(zip(base, utilities)):
        if guard and i % 512 == 0:
            guard.require(10*60, "bounded F1 set decoding")
        old_set, protected = set(old), set(old[:math.ceil(.6*len(old))])
        remove = sorted(((float(g), int(c)) for c, g in zip(candidate, gain) if c in old_set-protected), reverse=True)
        add = sorted(((float(g), int(c)) for c, g in zip(candidate, gain) if c not in old_set), reverse=True)
        removed, added = set(), []
        # A swap has zero count drift. Each species can be edited only once.
        for _ in range(policy["swaps"]):
            if not remove or not add or remove[0][0]+add[0][0] <= policy["margin"]:
                break
            removed.add(remove.pop(0)[1]); added.append(add.pop(0)[1])
        # At most delta unilateral edits, not a hidden second sequence of swaps.
        direction = 0
        for _ in range(policy["count_delta"]):
            k = len(old)-len(removed)+len(added)
            plus = add[0][0] if add and k < 40 and direction >= 0 else -np.inf
            minus = remove[0][0] if remove and k > 8 and direction <= 0 else -np.inf
            if max(plus, minus) <= policy["margin"]:
                break
            if plus >= minus:
                added.append(add.pop(0)[1]); direction = 1
            else:
                removed.add(remove.pop(0)[1]); direction = -1
        output.append([c for c in old if c not in removed]+added)
    validate_edits(base, output, policy)
    return output


def validate_edits(base, output, policy):
    if len(base) != len(output):
        raise ValueError("Changed survey count")
    for old, new in zip(base, output):
        additions, removals = len(set(new)-set(old)), len(set(old)-set(new))
        if (not 8 <= len(new) <= 40 or len(new) != len(set(new)) or
            abs(len(new)-len(old)) > policy["count_delta"] or
            additions+removals > 2*policy["swaps"]+policy["count_delta"] or
            min(additions, removals) > policy["swaps"] or
            not set(old[:math.ceil(.6*len(old))]).issubset(new)):
            raise ValueError("Bounded set-edit contract broken")
