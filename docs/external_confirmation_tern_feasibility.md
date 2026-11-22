# External confirmation feasibility — TERN AusPlots

Metadata and documentation screen, 22 November 2026. The candidate is a
site-visit-level voucher-backed survey. No new plant composition, specimen
identity or model-evaluation outcome was accessed.

## What the source supports

TERN's [Ecosystem Surveillance release](https://portal.tern.org.au/metadata/9210c331-ba96-4e0c-9554-f029d61534be)
documents repeated monitoring at standardized one-hectare plots.
The [Plant Specimen release](https://portal.tern.org.au/metadata/6a2b44c1-97d3-4c3f-b096-567672c06fc0)
describes specimen barcodes, collection dates and herbarium identification
dates and authorities. The [Plant Occurrence release](https://portal.tern.org.au/metadata/d6307ee4-68de-49ff-8a89-eab946fa03b2)
describes the separate point-intercept observations. An occurrence sample,
a voucher and a complete site-visit inventory serve different roles in the
evaluation. The [official client](https://github.com/ternaustralia/ausplotsR)
documents a site-and-visit-only request and separates standardized taxonomy
from original herbarium determinations. This supports a staged admission audit.

## Overlap and admission

The local SAGE evaluation metadata contain **8,431 Australian source-plot rows
in 7,530 cells**. Every one is from **AU-AU-002**, which the
[sPlotOpen source inventory](https://fmsabatini.github.io/sPlotOpen_Manuscript/)
identifies as TERN's AEKOS archive. Four dates are unparseable. The latest
parseable Australian date in this metadata is **3 September 2014**. Shared
archive lineage rules out treating AusPlots as a new, independent source solely
by name. A post-2014 visit filter is useful for screening and still needs exact
site/visit and distance joins against SAGE and the published occurrence-training
support. Repeat visits can have new records at previously observed sites.

Before accessing outcomes, get only site/visit IDs, dates and coordinates,
verify archive/version and plot footprints, exclude already used site-visits,
and quantify geographic blocks and published-model support. Test the 52-feature
and native-mask bridge against known inputs. Document the inventory coverage,
current name and unresolved-label contract before admitting any comparison. In
particular, a missing point-intercept hit should never imply absence in the
voucher inventory. Freeze the effect measure, spatial power criterion,
comparison margin and eligible cohort before releasing the study's species sets
to analysis. Site counts, environmental input readiness and complete voucher
joins remain unverified for the proposed new visits.

## Field cost experiment

With a botanist or curator, sample candidate linked records by genus, linkage
size and specimen evidence. Present a fixed record pair in a blinded
same-target-category check, allow an explicit unresolved response and compare
with independent adjudicated species determinations. Record hands-on effort
separately for compatibility checks and full IDs, counting follow-up
identifications and abstentions. Collection-to-determination dates can support
a workflow latency description. Estimating staff time and false-compatible
rates requires timed tasks with an adjudicated reference. This is a candidate
field study. No annotation time, error rate or fresh confirmation was measured.
