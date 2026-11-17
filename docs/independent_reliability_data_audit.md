# External-data feasibility — 14 November 2026

**16 November update:** this dated feasibility audit was followed by completed
[locked E1 evaluation](external_reliability_e1_results.md). The original 11,130
metadata candidates were intersected with released predictions, checked for
country/continent consistency and area provenance, and split with buffers.
Final roles: 1,165 probability calibration, 1,061 risk calibration, 5,251 assessment.
Native-mask and row alignment checks pass. The proposed method fails its primary
gain target; released-model PO overlap and metadata-completeness limits remain.
These outcomes are now consumed. The paragraphs below preserve the earlier
metadata-only audit state rather than asserting that targets remain closed now.

**Outcome:** metadata-only audit completed; a plausible external replication
cohort exists, but independent model evaluation has not been performed. External
species targets, model prediction values and species-name mappings were not
downloaded or inspected. No protected GeoPlant taxon identities were inferred.

## sPlotOpen availability

The [official portal](https://idata.idiv.de/ddm/Data/ShowData/3474?version=76)
and its metadata/file-list fragments were retrieved. The access-policy field says
**Open (CC BY 4.0)**. The listed plot header is 24.98 MB, but its published file
download endpoint returned HTTP 200 with zero bytes; ordinary decoded-path
variants also returned empty bodies. The archive HEAD request timed out. These
empty files are not treated as downloaded survey data or as an authorization
barrier. Species-composition, community-trait and RData files were excluded.

The raw release contains overlapping resampling indicators, heterogeneous survey
completeness and coordinates/dates/plot area. Composition-derived community traits
must not be predictors. These facts and upstream field descriptions are available
in the [authors' data description](https://fmsabatini.github.io/sPlotOpen_Manuscript/).

## Newly identified relevant prior work: SAGE

[SAGE](https://arxiv.org/abs/2609.31082), submitted 25 September 2026, is a preprint
on sampling-aware SDM evaluation. It supplies a global sPlotOpen/GBIF benchmark,
species-level sampling-effort/prevalence strata and model comparisons; it also
examines plot-area effects. Its broad framing overlaps a generic paper about
sampling-aware evaluation. Our candidate distinction is site-level species-set
decisions and selective reliability under matched geographic/ecological coverage.
This is a narrower research question, not a novelty clearance.

Its data and model releases are available on Zenodo, with code under MIT. The
data API reports open access but exposes no explicit `metadata.license` field;
upstream sPlotOpen attribution and the downstream release's terms must remain in
the provenance record. A code license is not a substitute for a data-license record.

## Completed metadata checks

Downloaded the 18,653,082-byte `eval_plot_metadata.csv` from the
[public SAGE data release](https://zenodo.org/records/21297133), verified its
published checksum and recorded SHA256. Only cell/plot IDs, source, continent,
country, date, coordinates, positional uncertainty and area were read. Cover,
community structure, functional information and outcome columns were excluded
from analysis. The table has **77,444 source-plot rows, 77,443 unique plot IDs and
53,336 aggregated evaluation cells**. Multiple source plots can contribute to one
cell. There are no duplicate plot IDs; one row lacks its plot ID, source and
continent and is excluded from the strict candidate cohort. Known non-European
records span **30,904** cells.

Compared all source-plot coordinates with the union of **88,987 PA training and
14,784 test locations** from the existing GeoLifeCLEF study, using haversine
nearest-neighbor distances. **2,398 source-plot rows are within 1 km** of an
existing PA location. This is proximity evidence, not proof of identical surveys.
It makes treating the entire release as nonoverlapping unjustified.

A conservative candidate cohort requires every contributing plot in a cell to
have: a known non-European continent; finite coordinates; at least 500 km from
all existing PA train/test locations; recorded positional uncertainty <=1 km;
recorded area 10–400 m²; and a known source ID without a European `EU-` prefix.
Requiring **all contributors** avoids filtering plot metadata while retaining a
cell's target vector assembled from excluded plots.

This yields **11,130 candidate cells**, **15,654 unique source plots**, **36 source
groups** and **34 countries**. Minimum distance to existing PA locations is
**567.94 km**. The largest country counts are Australia 3,396 cells, United States
2,790 and Russia 1,465. Thus the candidate still has geographic imbalance. Its
cell-ID set is hashed and stored locally under ignored `data/`; no final
confirmation cohort or species-support rule has been locked from these counts.

## Feasible artifact reuse

Verified public archive inventories by downloading TAR **headers only** via HTTP
range requests: 22,016 bytes for the data archive and 26,112 for predictions.
No target or prediction payload was read. The data archive is 3.85 GB, and the
prediction archive 42.65 GB, but selective range retrieval is supported. Its five
evaluation predictor files total only **7,348,036 bytes**. This can avoid bulk
downloads when assembling a later locked replication.

The [model release](https://zenodo.org/records/21335338) lists checkpoints,
configurations and predictions. Existing trained baselines may reduce the
proposed six-fit external training allocation. This is feasibility evidence,
not a completed model download, inference run or validated external result.

## What remains before confirmation

- Lock a specific comparison, calibration/selector/assessment roles, source/cohort
  inclusion rule and species-support policy before opening external labels.
- Confirm target/predictor/cell-row alignment and source-completeness provenance.
  Existing metadata omit some original survey-completeness fields.
- Use dataset-specific taxon identities. Do not join anonymous GeoPlant classes
  to named SAGE species.
- Account for 1 km aggregation, collapsed survey dates and native-range masks.
  A `-1` out-of-range target is not an observed absence. This external task differs
  from GeoLifeCLEF plot-level prediction.
- Use model families compatible with the released tabular predictors. Sentinel
  imagery and Landsat time series are not provided by that external design.
- Audit training/evaluation overlap for the models actually used. The published
  SAGE protocol does not remove all PO occurrences in evaluation cells; reusing
  its models inherits that design. The completed geographic check covers existing
  GeoLifeCLEF **PA** train/test, not all historical PO records.
- Keep final outcomes with a supervisor where feasible, or accurately describe
  a newly locked analysis of a public benchmark. Do not call it a secret test.

Evidence: `results/reliability_followup_d1_d3/independent_data/audit.json`;
raw metadata/docs and archive-header inventories are ignored under
`data/independent_data_audit/`. No new training has been admitted by this audit.
