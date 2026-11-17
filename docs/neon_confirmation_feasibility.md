# NEON independent-source feasibility — 16 November 2026

**Subsequent status:** see [the completed bridge/GPU checks](neon_bridge_status.md).
The MLP reproduces on CPU/T4; the ResNet does not. Plot metadata/taxonomy are now
audited, while complete predictors/native masks remain pending. Observation-file
bytes were traversed by a frozen metadata-only projection; composition fields
were discarded without decoding or inspection. Original PO training support
overlaps 301 of 368 candidate plots, narrowing a future study to reliability/source
transport unless a training-disjoint design is established.

**Original audit below:** metadata audit completed; model-compatible evaluation was not yet ready.
At this initial audit no NEON plant observation/composition outcomes had been downloaded or inspected.
This is the next candidate for a separate field-data source rather than another
partition of SAGE outcomes.

## Verified source and retained metadata

Official product [DP1.10058.001: Plant presence and percent cover](https://data.neonscience.org/data-products/DP1.10058.001)
has 47 sites in the retrieved metadata. Pin the stable
[RELEASE-2026 DOI](https://doi.org/10.48443/9g4s-hq52), not provisional updates.
The product metadata JSON (147,099 bytes) and a bounded projection of coordinate
metadata for all 47 sites were retained with hashes. The full all-sites response
exceeded the 2 MB and then 12 MB caps; it was not saved. Reading only the initial
site-coordinate fields consumed 192,512 bytes across the 47 site requests.

This coarse site-centroid screen finds three sites within 1 km of a published
SAGE evaluation centroid and 14 within 10 km; 32 site centroids are >=100 km from
the now-consumed E1/E3 cells. Site centroids are not plot locations. These counts
do not establish plot-level overlap, nonoverlap or source independence.

The official [quick-start guide](https://data.neonscience.org/api/v0/documents/quick-start-guides/NEON.QSG.DP1.10058.001v2?fallback=html&inline=true)
describes nested 1, 10, 100 and 400 m² observations with different schedules.
The [RELEASE-2026 notes](https://data.neonscience.org/data-products/DP1.10058.001/RELEASE-2026)
document reduced full-plot coverage in recent sampling. Do not label every
observation a complete 400 m² inventory or turn unobserved taxa into absences.
Repeated visits and unresolved taxa also require explicit rules before outcomes.

## Input/model bridge: concrete verified contract

The immutable SAGE `configs.tar` archive (184,320 bytes) was downloaded and its
published MD5 checked. It contains 25 released deep-model configurations. Relevant
s0/s1 input contracts are saved in
`results/neon_confirmation_feasibility/input_contracts.json`:

| Input group | Ordered feature count |
|---|---:|
| CHELSA | 19 |
| SoilGrids | 8 |
| EarthEnv topography | 16 |
| Human Footprint | 9 |
| Total environmental | **52** |

Coordinates are present in the data loader's group dictionary, but the released
model configuration explicitly sets `use_location: false`. Do not infer model
input dimension by blindly concatenating every data-loader field.

`baseline_deepsdm`/`aggregation`/`subsampling` use `models.SimpleMLP`; `architecture`
and `optimized_deepsdm` use `models.ResNet`. Both s0/s1 variants use the 52-feature
environmental contract. Released deep checkpoints/configs permit a possible
two-architecture inference-only comparison, subject to normalization and exact
reference-output reproduction. No checkpoints were downloaded in this audit.

The [released-model documentation](https://github.com/earens/sage-sdm-benchmark/blob/main/docs/MODELS.md)
states that the single-species traditional models are available through their
predictions rather than reusable fitted weights. Therefore do not promise a
zero-training random-forest transfer to arbitrary new NEON locations. A new
traditional baseline would need its own justified training budget; alternatively
use the two released neural architectures with that changed scope declared.

## Next executable work

1. Obtain plot IDs, coordinates, dates and survey/completeness metadata, projecting
   away composition fields. Audit plot/site/temporal overlap against published
   SAGE metadata and all consumed reliability locations. Select roles with site
   or source separation before composition values.
2. Establish a versioned mapping from NEON taxonomic identifiers to the public
   SAGE class list, retaining unresolved/missing taxon semantics. No mapping of
   protected anonymous GeoLifeCLEF taxa is needed or permitted by this design.
3. Reproduce the 52 feature definitions, units, raster versions, grid aggregation,
   missing-value rules and checkpoint normalization at the new plot locations.
   Nearby cached SAGE points are not automatically substitute predictors.
4. Retrieve only admitted deep-model artifacts, verify their provenance and
   reproduce reference predictions on existing SAGE covariates before fresh
   inference. Begin with one architecture/seed CPU smoke test.
5. Once these input/reference tests pass, run the already-authorized bounded
   private GPU inference study under a fixed outcome-independent protocol.
   Scientific evaluation may legitimately be negative; it does not need to look
   favorable before a neutral confirmation run is launched.

The predictor bridge, full taxonomy/completeness audit and reference reproduction
are **pending**, not implemented. No new model performance, NEON sample count,
nonoverlap guarantee or GPU runtime is asserted. Metadata audit/code:
`scripts/audit_neon_confirmation.py`; retained audit and contracts:
`results/neon_confirmation_feasibility/`.
