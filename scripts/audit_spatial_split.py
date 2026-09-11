#!/usr/bin/env python
"""Audit geographic holdout candidates before defining a spatial validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("surveyId", "speciesId", "lon", "lat", "region", "country")


def _clean_group(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.isin(["", "nan", "None", "<NA>"]), "__missing__").fillna(
        "__missing__"
    )


def _group_summary(
    survey_table: pd.DataFrame, pairs: pd.DataFrame, column: str
) -> list[dict[str, int | float | str]]:
    total_surveys = int(len(survey_table))
    total_labels = int(len(pairs))
    global_species = set(pairs["speciesId"].unique().tolist())
    results: list[dict[str, int | float | str]] = []
    for value, group_surveys in survey_table.groupby(column, sort=False, dropna=False):
        validation_ids = set(group_surveys["surveyId"].tolist())
        validation_pairs = pairs.loc[pairs["surveyId"].isin(validation_ids)]
        train_pairs = pairs.loc[~pairs["surveyId"].isin(validation_ids)]
        validation_species = set(validation_pairs["speciesId"].unique().tolist())
        train_species = set(train_pairs["speciesId"].unique().tolist())
        unseen_species = validation_species - train_species
        unseen_labels = int(validation_pairs["speciesId"].isin(unseen_species).sum())
        results.append(
            {
                "value": str(value),
                "validation_surveys": int(len(validation_ids)),
                "validation_fraction": float(len(validation_ids) / max(total_surveys, 1)),
                "validation_labels": int(len(validation_pairs)),
                "validation_label_fraction": float(len(validation_pairs) / max(total_labels, 1)),
                "validation_species": int(len(validation_species)),
                "train_species": int(len(train_species)),
                "validation_species_unseen_in_train": int(len(unseen_species)),
                "unseen_validation_species_fraction": float(
                    len(unseen_species) / max(len(validation_species), 1)
                ),
                "unseen_validation_label_fraction": float(
                    unseen_labels / max(len(validation_pairs), 1)
                ),
                "global_species_coverage": float(
                    len(validation_species) / max(len(global_species), 1)
                ),
            }
        )
    return sorted(results, key=lambda row: int(row["validation_surveys"]), reverse=True)


def audit_spatial_splits(metadata_path: Path) -> dict[str, object]:
    """Summarize country/region holdouts without choosing a split post hoc."""
    metadata = pd.read_csv(metadata_path, usecols=list(REQUIRED_COLUMNS))
    pairs = metadata[["surveyId", "speciesId"]].dropna().drop_duplicates().copy()
    pairs["surveyId"] = pairs["surveyId"].astype("int64")
    pairs["speciesId"] = pairs["speciesId"].astype("int64")
    if pairs.empty:
        raise ValueError("No labelled PA survey/species pairs were found")

    survey_rows = metadata.dropna(subset=["surveyId"]).copy()
    survey_rows["surveyId"] = survey_rows["surveyId"].astype("int64")
    survey_rows["country"] = _clean_group(survey_rows["country"])
    survey_rows["region"] = _clean_group(survey_rows["region"])
    conflicts = {
        column: int((survey_rows.groupby("surveyId")[column].nunique(dropna=False) > 1).sum())
        for column in ("lon", "lat", "country", "region")
    }
    survey_table = survey_rows.drop_duplicates("surveyId", keep="first")[
        ["surveyId", "lon", "lat", "country", "region"]
    ]
    country_candidates = _group_summary(survey_table, pairs, "country")
    region_candidates = _group_summary(survey_table, pairs, "region")

    eligible = [
        {"level": level, **candidate}
        for level, candidates in (("country", country_candidates), ("region", region_candidates))
        for candidate in candidates
        if 0.08 <= float(candidate["validation_fraction"]) <= 0.30
        and float(candidate["unseen_validation_label_fraction"]) <= 0.20
    ]
    eligible.sort(
        key=lambda row: (
            abs(float(row["validation_fraction"]) - 0.20),
            float(row["unseen_validation_label_fraction"]),
        )
    )
    coordinates = survey_table[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
    valid_coordinates = coordinates.dropna()
    bounds = None
    if not valid_coordinates.empty:
        bounds = {
            "longitude_min": float(valid_coordinates["lon"].min()),
            "longitude_max": float(valid_coordinates["lon"].max()),
            "latitude_min": float(valid_coordinates["lat"].min()),
            "latitude_max": float(valid_coordinates["lat"].max()),
        }
    return {
        "metadata_path": str(metadata_path),
        "surveys": int(len(survey_table)),
        "label_pairs": int(len(pairs)),
        "species": int(pairs["speciesId"].nunique()),
        "surveys_missing_coordinates": int(coordinates.isna().any(axis=1).sum()),
        "coordinate_bounds": bounds,
        "survey_metadata_conflicts": conflicts,
        "country_count": int(survey_table["country"].nunique(dropna=False)),
        "region_count": int(survey_table["region"].nunique(dropna=False)),
        "country_holdouts": country_candidates,
        "region_holdouts": region_candidates,
        "eligible_holdouts": eligible,
        "recommended_holdout": eligible[0] if eligible else None,
        "selection_rule": (
            "Pre-registered: country or region containing 8%-30% of surveys, at most 20% "
            "unseen validation labels, then closest to a 20% validation share."
        ),
        "warning": (
            "This audit selects a defensible candidate only. Model comparison must use the "
            "same frozen holdout and must not tune the holdout choice to maximize F1."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument(
        "--report-path", type=Path, default=Path("data/reports/spatial_split_audit.json")
    )
    args = parser.parse_args()
    report = audit_spatial_splits(args.metadata_path)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
