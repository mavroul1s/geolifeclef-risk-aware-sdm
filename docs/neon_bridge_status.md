# NEON bridge and GPU reference — 16 November 2026

**One released model now reproduces published probabilities on CPU and GPU.
NEON survey metadata and dictionary mappings are audited. No NEON ecological
performance experiment has run. Original model-training overlap rules out an
unseen-training-location interpretation of the current candidate cohort.**

## Completed model checks

The two pipelines were chosen from released configuration metadata before NEON
composition outcomes. Preprocessing uses 52 ordered environmental features and
the publisher's fixed normalization. Coordinates are not network inputs
(`use_location=false`). Minimal local adapters load ZIP-serialized checkpoints
with `torch.load(weights_only=True)` and strict state matching. Downloaded source
is inspected, not executed.

The fixed reference is the first 128 released test-prediction rows and all 5,771
classes, without target labels. Original tolerances remain maximum absolute
probability error <=1e-5 and mean absolute error <=1e-6.

| Pipeline | CPU maximum error | CPU mean error | Decision |
|---|---:|---:|---|
| `aggregation_s0` / SimpleMLP | 9.239e-7 | 1.083e-9 | Pass |
| `optimized_deepsdm_s0` / ResNet | 0.438365 | 0.006827 | Fail; excluded from fresh inference |

The ResNet loads strictly and its checkpoint ZIP passes CRC verification. Its
probabilities correlate with the reference, but that is not reproduction. The
cause remains unresolved: do not attribute it to an upstream defect or choose a
looser tolerance. E1–E3 retain their unchanged published predictions and results.

The original plan named two pipelines. After the CPU comparison, the GPU contract
explicitly admits only the passing pipeline and records the ResNet exclusion.
This is a neutral implementation check, not a completed two-pipeline ecological
comparison or isolated architecture ablation.

The authorized private [Kaggle reference notebook](https://www.kaggle.com/code/nmavros/sage-reference-neon-20261116)
completed on a **Tesla T4**, Torch 2.11.0+cu130, in **42.81 seconds inside the
notebook**, excluding startup/queue time. Full local verification compares all
738,688 probabilities against CPU and published outputs: maximum error
1.103e-6 and mean error about 1e-9. Both pass. Input, notebook, checkpoint and
collected-output hashes are checked. One job, zero training fits, zero competition
submissions; no NEON token enters the notebook.

Checkpoint TAR headers/exact byte ranges, local SHA256s and serialization checks
are verified. The entire multi-GB checkpoint TAR checksum is **not** recomputed.
The published code ZIP and configuration TAR have their published MD5s checked.
Aggregation reference probabilities were read through bounded HDF5 byte ranges
instead of downloading the whole 934 MB prediction member.

## NEON access and survey audit

The provided token authenticates successfully, stays outside Git/vault, and has
owner-only file permissions. Storage downloads do not receive the API token.
See [NEON authentication](https://data.neonscience.org/data-api/authentication/).

The pinned source is [Plant presence and percent cover, DP1.10058.001](https://data.neonscience.org/data-products/DP1.10058.001),
year 2023, [RELEASE-2026](https://doi.org/10.48443/9g4s-hq52). The file query yields
127 site-month packages over 47 sites and 198 relevant basic observation files.
Survey metadata and composition share CSVs. A frozen byte-level projection
transports **60,165,227 bytes**, checks every declared size and MD5, discards
unselected fields without text decoding, deduplicates metadata, and retains
**2,787,868 bytes**. Raw files are not retained. Species identity, presence,
cover, richness and performance values have not been inspected. This is
computational field-level separation, **not physical non-access to observation
file bytes**.

Allowlisted fields cover plot/subplot IDs, coordinates/uncertainty, dates, bouts,
protocol and quality flags. Free text, taxa, cover, presence flags and composition
multiplicities are excluded. Parser tests cover quoted commas/newlines and
undecodable bytes in discarded outcome fields.

Completeness requires four 100 m², six 10 m² and six 1 m² units within a plot/bout,
all with acceptable sampling and biophysical flags. Their nested union is
**400 m²**, not 466 m². Repeated visits are not independent plots.

| Metadata screen | Count |
|---|---:|
| Distinct plots / plot-bouts | 1,478 / 1,577 |
| All required nested units present | 768 plot-bouts |
| Full units, quality and coordinate checks pass | 555 plot-bouts |
| Additional spatial screen passes | 410 plot-bouts |
| Earliest eligible bout per plot | **368 plots / 16 sites / 209 1 km cells** |

Spatial candidates are >=100 km from consumed SAGE cells, >=500 km from GLC PA
locations, and outside all published SAGE evaluation cells. Observed minimum
distances are 102.94 km and 4,716.02 km. Seven plot-bouts in the full inventory
occupy existing SAGE evaluation grid cells and are excluded. The candidate set
is feasibility evidence, **not a locked cohort or 368 independent geographic
replicates**. Source-role allocation remains pending.

## Original-training overlap changes the interpretation

The aggregated PO training-coordinate file contains **4,100,969 locations**.
Of the 368 candidate plots, **301 occupy a training grid cell** (163 of the 209
candidate cells); only 41 plots are more than 1 km from any training location,
and **none is more than 10 km away**. Median nearest training-location distance
is 0.449 km. All 16 sites have a candidate plot in a training cell.

This demonstrates spatial support, not proof that a particular NEON species
record/label was in training. It does not erase separation from E1–E3 reliability
locations. However, do not call this a backbone evaluation at unseen training
locations. A possible study can test reliability-calibration transport across
survey sources/sites while reporting pretrained support. A stronger training-
independent geographic claim requires a justified buffered retraining design or
another source. Changing distance thresholds after outcomes cannot create it.

## Taxonomy and environmental inputs

All **93,698 PLANT dictionary entries**, in 94 pages, are retained with hashes.
Mapping uses exact normalized names after removing only separately supplied
authorship, or an explicit accepted-taxon ID followed by an exact name. It maps
4,346 identifiers to **2,453 of 5,771 SAGE classes**; 24 direct/accepted-name
conflicts remain ambiguous. Higher ranks, infraspecific taxa and unresolved
names remain unmapped. No fuzzy synonym guesses or anonymous GLC mapping.

Dictionary counts do not measure actual occurrence coverage. Missing/unresolved
taxa must not silently become absent species; label-resolution and native-mask
semantics remain to be fixed before outcomes.

Exact source URLs support bounded range access for tested CHELSA v2.1, SoilGrids
2017 and EarthEnv rasters. At eight fixed released input rows, each group's first
column matches exactly after the published missing-value rule. One SoilGrids
nodata requires a one-pixel nearest-valid fill and reproduces value 1,576.
This checks **three of 52 columns**, not the full fresh-location vector. Remaining
columns, nine Human Footprint layers, broader missingness and native masks are
pending.

## Validation and integrity

**21 focused tests pass**. Independent pandas reconstruction reproduces the
768 complete and 555 quality-eligible plot-bouts. Five fixed brute-force
haversine queries agree with the training-location tree. That independent check
caught float32 degree-to-radian conversion; converting coordinates to float64
before the angle calculation corrected the precision, with candidate/overlap
counts unchanged. This was fixed before any outcome or cohort-role freeze.

All **44 E1, 24 E2, 48 E3 and three initial NEON audit manifest files** remain
unchanged. The frozen metadata projection, dictionary mapping, input/output
hashes and absence of credentials from code, reports, documentation and the
uploaded notebook are checked. The engineering audit preserves the failed
two-model reference and explicitly leaves fresh NEON inference unadmitted. It
does not claim a historical full-suite pass or scientific success.

## Next bounded work

1. Resolve the ResNet discrepancy before admission, or prospectively declare a
   single verified-model study. Do not choose based on NEON scores.
2. Complete all 52 fresh-location inputs and native-range masks with exact
   reference-vector checks and retained provider versions/units/missingness.
3. Fix the scope to survey-source/reliability-head transport unless a training-
   disjoint design is established. Allocate sites/cells to roles; define label
   coverage, repeated-visit rules, primary contrasts, practical margins and
   site-level uncertainty before composition.
4. Run one neutral bounded inference/confirmation study when these checks pass,
   using standing authorization. Positive outcomes are not an admission rule.
   No architecture/seed sweep or new backbone fit is admitted by this step.

This is implementation and provenance progress, not a new ecological effect,
successful algorithm, benchmark gain or publication novelty. E1/E3 failures and
the post-hoc status of mean controls remain unchanged. Changes are local and
uncommitted.

Reports: `results/neon_bridge/`; large inputs/metadata and GPU arrays remain in
ignored `data/neon_bridge/` and `artifacts/neon_bridge_gpu/`. Reproduction scripts
cover CPU comparison, GPU notebook/validation, survey projection/audit, taxonomy
mapping, training overlap and raster reference probes.
