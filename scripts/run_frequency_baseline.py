#!/usr/bin/env python
"""Evaluate a PA-survey frequency prior directly on GeoLifeCLEF's long labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("surveyId", "speciesId")


def evaluate_frequency_baseline(
    metadata_path: Path, *, seed: int = 2025, validation_fraction: float = 0.2
) -> dict[str, int | float]:
    """Use a survey-level split and a train-only, fixed top-k species prior."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    metadata = pd.read_csv(metadata_path, usecols=list(REQUIRED_COLUMNS))
    if any(column not in metadata for column in REQUIRED_COLUMNS):
        raise ValueError(f"{metadata_path} must contain {REQUIRED_COLUMNS}")
    pairs = metadata.dropna(subset=REQUIRED_COLUMNS).copy()
    pairs["surveyId"] = pairs["surveyId"].astype("int64")
    pairs["speciesId"] = pairs["speciesId"].astype("int64")
    pairs = pairs.drop_duplicates(list(REQUIRED_COLUMNS))
    if pairs.empty:
        raise ValueError("No labelled PA survey/species pairs were found")

    survey_ids = pairs["surveyId"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(survey_ids)
    validation_count = max(1, int(round(len(survey_ids) * validation_fraction)))
    validation_ids = survey_ids[:validation_count]
    is_validation = pairs["surveyId"].isin(validation_ids)
    train_pairs, validation_pairs = pairs.loc[~is_validation], pairs.loc[is_validation]
    if train_pairs.empty or validation_pairs.empty:
        raise ValueError("Survey split produced an empty partition")

    species_frequency = train_pairs.groupby("speciesId")["surveyId"].nunique().sort_values(
        ascending=False, kind="stable"
    )
    mean_cardinality = len(train_pairs) / train_pairs["surveyId"].nunique()
    top_k = min(max(1, int(round(mean_cardinality))), len(species_frequency))
    predicted_species = set(species_frequency.index[:top_k].tolist())
    validation_species_counts = validation_pairs.groupby("speciesId")["surveyId"].nunique()
    true_positives = int(validation_pairs["speciesId"].isin(predicted_species).sum())
    predicted_positives = int(len(validation_ids) * top_k)
    actual_positives = int(len(validation_pairs))
    micro_f1 = 2 * true_positives / max(predicted_positives + actual_positives, 1)
    macro_terms = [
        (2 * int(count) / (len(validation_ids) + int(count))) if species_id in predicted_species else 0.0
        for species_id, count in validation_species_counts.items()
    ]
    return {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_surveys": int(train_pairs["surveyId"].nunique()),
        "validation_surveys": int(len(validation_ids)),
        "train_species": int(len(species_frequency)),
        "validation_species": int(len(validation_species_counts)),
        "validation_species_unseen_in_train": int(
            (~validation_species_counts.index.isin(species_frequency.index)).sum()
        ),
        "train_mean_labels_per_survey": float(mean_cardinality),
        "predicted_species_per_survey": int(top_k),
        "micro_f1": float(micro_f1),
        "macro_f1_observed_validation_species": float(np.mean(macro_terms)),
        "precision": float(true_positives / max(predicted_positives, 1)),
        "recall": float(true_positives / max(actual_positives, 1)),
        "true_positive_labels": true_positives,
        "predicted_labels": predicted_positives,
        "actual_labels": actual_positives,
        "policy": "Top-k training prevalence; k is rounded train mean labels per survey.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=Path("artifacts/frequency_pa/metrics.json"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = evaluate_frequency_baseline(
        args.metadata_path, seed=args.seed, validation_fraction=args.validation_fraction
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
