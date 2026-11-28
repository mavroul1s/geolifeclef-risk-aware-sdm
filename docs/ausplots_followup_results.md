# AusPlots predictor and record audit — 28 November 2026

We recovered **44 of 52** published environmental inputs and sampled them at all
119 candidate cells. The second SAGE setting reproduced published probabilities
on CPU. The combined geography and record screen leaves **87 candidate sites in
45 spatial blocks** after selecting the earliest qualifying visit per site.
The field comparison remains pending while eight original Human Footprint
components are unavailable. Species composition remains unopened.

## Independent predictor reproduction

We used eight original bridge locations, 12 additional Australian SAGE rows and
12 other release rows chosen from site metadata and released row order. All 44
recovered layers match all 32 reference values exactly. This covers 19 CHELSA
v2.1, eight SoilGrids 2017, 16 EarthEnv and the cumulative Human Footprint 2009
layer. The code reproduced raw raster values and the released 50-pixel
nearest-valid fill, including 16 fills in the reference panel. Sampling the 119
cells produced **5,236 finite values** and 66 fills across the 44 layers.
The 44-layer replay used cached byte ranges for the first 43 sources. The
provider's exact TIFF filenames and reference order are checked against the
archived SAGE extraction code. [Predictor receipt](../results/ausplots_followup/predictors_with_wcs/predictors.json).

The [Dryad deposit](https://doi.org/10.5061/dryad.052q5) lists two archives.
Both public file links returned HTTP 403 in bounded tests. The 2016 file API
returned HTTP 401 and requested a bearer token. A separate documented download
from the [authors' WCS site](https://www.wcshumanfootprint.org/v2/) supplied
`HFP2009.tif`. That archive contains the cumulative layer. The eight individual
pressures are still needed. The receipt records the file identifiers, expected
MD5 digests and actual access results. [Access tests](../results/ausplots_followup/human_footprint_access.json) ·
[WCS check](../results/ausplots_followup/human_footprint_wcs_probe.json).

## Second model and training exposure

`baseline_deepsdm_s0` uses the same MLP architecture as `aggregation_s0` and a
different training representation. The original-resolution model reproduced all
738,688 reference probabilities, with maximum absolute CPU error **1.22e-6**
and mean error **5.52e-10** within the fixed 1e-5/1e-6 tolerances. The earlier
`optimized_deepsdm_s0` ResNet mismatch remains documented separately. No model
was refitted. [CPU reference](../results/ausplots_followup/second_model_reference.json) ·
[Earlier model checks](neon_bridge_status.md).

The first private Kaggle submission remained queued and was deleted after an
identical private submission started on a second account. The second completed
on a Tesla T4 in 42.46 seconds inside the notebook. All 738,688 probabilities
passed comparison with the CPU and published references. Maximum absolute
error against publication was 1.34e-6. Both submissions used the same notebook
bytes and published SAGE inputs.
[GPU verification](../results/ausplots_followup/second_model_gpu_reference.json) ·
[Original queue receipt](../results/ausplots_followup/gpu_reference_job_state.json).

We checked all **30,138,089** released original-resolution presence-only
training coordinates. Each of the 151 candidates stays outside the grid of both
training releases. Requiring a distance of more than 1 km from both training
sets leaves 110 visits before applying the record and native-mask screens.
Sixteen independent great-circle distance checks agreed. The geographic screen
concerns the released presence-only locations. The pretrained models also used
background inputs, and these checks cannot identify individual training
records. [Training screen](../results/ausplots_followup/nonaggregated_training_overlap.json).

## Voucher linkage and sampling effort

All 151 visits have the one-hectare flag, 146 have the full 10-by-101
point-intercept layout and 149 have voucher records. We requested 8,371 voucher
rows and 180,864 point rows limited to visit keys, effort, dates, barcode and
uncertainty metadata. All voucher keys are unique within visit and have matching
dates. Of 104,966 point records with barcodes, **104,965** link uniquely to a
voucher on the same visit. Voucher records include 921 with uncertain
identification and 326 without a recorded uncertainty flag. Full effort and a
unique barcode establish a traceable record, and botanical review is still
needed to establish species-level inventory coverage.
[TERN vouchering protocol](https://www.tern.org.au/wp-content/uploads/TERN-Rangelands-Survey-Protocols-Manual_web.pdf) ·
[Record receipt](../results/ausplots_followup/voucher_metadata.json).

| Cumulative screen | Visits | Sites | 1 km cells | 100 km blocks |
|---|---:|---:|---:|---:|
| Outside the legacy screen and both PO training grids | 151 | 123 | 119 | 58 |
| Also complete structural record check | 145 | 122 | 118 | 57 |
| Also strict same-region native masks | 139 | 116 | 112 | 53 |
| Also >1 km from both PO training sets | 103 | 87 | 84 | 45 |
| Earliest qualifying visit per site | 87 | 87 | 84 | 45 |
| Also >10 km from both PO training sets, strict records and native masks | 15 | 15 | 14 | 5 |

These are candidate counts. Several sites share an input cell, and the 100 km
blocks describe spatial support. The >10 km stratum provides only five blocks
after all metadata screens. The 103 visits in the >1 km stratum contain 5,812
voucher records, including 626 uncertain determinations and 194 missing flags.
The combined receipt verifies the geography and record joins independently of
the subsequent WCS download. [Independent combined check](../results/ausplots_followup/combined_admission.json).

## Consequences of filling unavailable variables

A controlled omission test used 7,530 released SAGE AEKOS cells, their native
range masks and both checked models. Using the published training means for all
nine Human Footprint features changed the top-10 species set at **99.7%** of
cells for `aggregation_s0` and **98.7%** for `baseline_deepsdm_s0`. Median set
Jaccard overlap was .333 and .429. Keeping the recovered cumulative layer and
filling only the eight unavailable components still changed the top-10 set in
**99.5%** and **98.5%** of cells, respectively, with the same median overlaps.
This measures prediction sensitivity on released inputs without scoring any
ecological outcomes.
[Stress-test protocol](ausplots_predictor_stress_protocol.md) ·
[All-nine omission](../results/ausplots_followup/human_footprint_stress.json) ·
[Eight-component omission](../results/ausplots_followup/human_footprint_components_stress.json).

## Validation and remaining work

All 428 repository tests passed after repairing local links moved with the
workspace. The independent reconciliations check all candidate coordinate and
record keys, repeat-visit counts, native masks and feature artifact hashes.
All 56 T5/T6 and 450 T4 manifest entries and the earlier checkable artifacts
remain verified. Public receipts contain aggregate counts and hashes. The raw
provider arrays, field barcodes and coordinates stay in the ignored data tree.
[Validation receipt](../results/ausplots_followup/validation.json).

Obtain the Dryad archive, verify its declared digest and test the eight
remaining layers against the same 32 expected rows. Before starting the field
comparison, finalize the accepted taxonomy and ambiguity rules, the source-only
prediction decoder and the visit analysis described in the
[protocol](ausplots_followup_protocol.md). A timed audit still needs independent
reviewers and adjudicated voucher material. TERN documents the
[herbarium collection](https://www.tern.org.au/soil-sample-and-herbarium-collection/).

## Reproduction entry points

Use the pinned SAGE inputs referenced by the receipts and the ignored archives
described above. `--cached` rechecks the exact stored projections. The predictor
script uses a new run name for each read and accepts an optional local raster
directory with `provenance.json`, as in the WCS replay. The independent
validator defaults to the initial 43-layer run so its original combined receipt
can be reproduced on that run's inputs.

```bash
export PYTHONPATH=src:scripts
.venv/bin/python scripts/check_ausplots_second_model.py
.venv/bin/python scripts/check_ausplots_training_overlap.py
.venv/bin/python scripts/audit_ausplots_voucher_metadata.py
.venv/bin/python scripts/audit_ausplots_voucher_metadata.py --cached
.venv/bin/python scripts/check_ausplots_predictors.py --run predictors_reference_v1
.venv/bin/python scripts/check_ausplots_predictors.py --run predictors_with_wcs \
  --human-dir "$PWD/data/ausplots_followup/human_footprint_wcs"
.venv/bin/python scripts/validate_ausplots_followup.py
.venv/bin/python scripts/stress_ausplots_human_footprint.py
.venv/bin/python scripts/stress_ausplots_hfp_components.py
```
