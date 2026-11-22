# AusPlots metadata audit

The [public ausplotsR site endpoint](https://github.com/ternaustralia/ausplotsR/blob/master/R/ausplots_queries.r)
returned all **1,152 visits across 824 sites**, with complete visit dates and
coordinates in the five-field projection. This screen used neither voucher
or species records nor model scores. The [metadata protocol](ausplots_metadata_audit_protocol.md)
defines the grid, date and distance screens.

## Spatial screen

SAGE's Australian evaluation metadata contain 8,431 AEKOS plot rows in 7,530
cells. Its latest parseable Australian plot date is **3 September 2014**.
Four plot rows lack a parseable date and remain in the spatial reference.
The training reference contains all 4,100,969 published occurrence-training
coordinates.

| Screen | Visits | Sites | 1 km cells | 100 km blocks |
|---|---:|---:|---:|---:|
| After the legacy record cutoff | 753 | 610 | 577 | 150 |
| Outside the legacy 1 km buffer and cells | 657 | 521 | 495 | 146 |
| Also outside published training cells | **151** | **123** | **119** | **58** |
| Also >1 km from training | 121 | 97 | 93 | 48 |
| Also >10 km from training | **17** | **17** | **16** | **7** |

Among the 151 off-training-grid visits, 132 visits (107 sites, 103 cells,
51 blocks) are at sites with only post-cutoff visits in this release.
This addresses the visible visit history. The site export provides no exact
crosswalk to AEKOS source plot IDs, and absence from the training grid applies
to released geographic coordinates. Nearby training records remain possible.
A 100 km block here describes geographic spread and does not establish
independence or statistical power. Repeat visits remain grouped by site.

## Model-input feasibility

None of the **119** off-training-grid candidate cells matches a released
evaluation or training location at its exact float32 cell center. Reusing
predictions or covariates from nearby cells would change the input contract.
All 119 need independently reproduced 52-feature vectors. By contrast,
**113/119 cells** have strict same-region native masks supported by the
published evaluation masks. Three cells lack a represented Tasmanian region
and three require the publisher's nearest-polygon fallback. Keep the fallback
stratum explicit. No feature vectors, voucher joins, or model predictions were
generated. The previous SAGE reference validation admitted the aggregation
MLP. A second model needs a passing reproduction check before it is used
for fresh inference. The [Human Footprint archive](https://doi.org/10.5061/dryad.052q5)
contains the nine human-pressure inputs. A request for its first 32 bytes
returned HTTP 403, so this audit did not recover those rasters.
[SAGE model bridge](neon_bridge_status.md).

These results establish an outcome-blinded candidate pool. Final confirmation
requires plant-inventory completeness and provenance, a verified environmental
feature bridge, and a prespecified analysis with an adequate number of
sites and spatial blocks. The strict >10 km stratum has only seven blocks.
An adjudicated time study remains necessary to replace synthetic audit costs.

## Reproduction and validation

The legacy client endpoint uses HTTP. The HTTPS probe failed, and the official
HTTP path returned the requested public projection with an exact server count
of 1,152. The result receipts record input hashes and the source response date.
The raw response and visit coordinates are retained in the ignored data
directory.

The native-mask stage uses exact longitude/latitude region assignment and
checks mask agreement across all published evaluation rows within each
represented region. The independent validator reconciles all post-cutoff
visit keys and training memberships and recalculates 16 nearest distances by
direct great-circle distances rather than the tree used for the production
audit. The independent calculation detected float32 rounding in the
provisional conversion to radians. Using float64 before conversion corrected
the distances, with identical admission counts.

[Spatial receipt](../results/ausplots_metadata_audit/spatial_feasibility.json) ·
[Native-mask receipt](../results/ausplots_metadata_audit/native_mask_feasibility.json) ·
[Validation receipt](../results/ausplots_metadata_audit/validation.json).

With the recorded local inputs restored, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_ausplots_metadata_audit.py
```
