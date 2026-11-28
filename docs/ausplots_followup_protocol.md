# AusPlots source and input verification

This follow-up uses the visit keys and geographic exclusions frozen in the
[metadata audit](ausplots_metadata_audit_protocol.md). The stages below can be
run independently. The field comparison depends on their results.

## Predictor and model references

Use the archived SAGE predictor code, released 1 km evaluation locations and
four complete evaluation feature tables from [SAGE](https://doi.org/10.5281/zenodo.21297133).
Require exact float32 value and NaN-pattern agreement per feature. The 32
reference rows are the eight original bridge rows, 12 additional Australian
evaluation rows evenly spaced by released row index, and 12 further rows
evenly spaced over the remaining release. Row selection uses no target labels.

Use the published source filenames, stored values, coordinate transformation,
integer pixel rounding/clamping and 50-pixel nearest-valid fill. Maintain the
published feature order: 19 CHELSA v2.1, eight SoilGrids 2017, 16 EarthEnv
topography, and nine Human Footprint 2009 inputs from the
[Dryad deposit](https://doi.org/10.5061/dryad.052q5). Store float32 geographic
coordinates for comparison with released 1 km input precision. Compare samples
against archived values before sampling any new cell with that layer. A source
failure and a numerical mismatch are separate reasons for leaving a feature
unverified. Track raw source nodata separately from an unavailable column.
Use bounded reads from official providers. Any local Human Footprint source
needs its archive provenance, exact member identity and the same value check.

Test `baseline_deepsdm_s0` as the second [released prediction setting](https://doi.org/10.5281/zenodo.21335338).
It uses a different pretrained training representation (nonaggregated
occurrences) from `aggregation_s0` and the same feature contract. Test it against
the first 128 released test-prediction rows, all 5,771 classes, with the original
maximum absolute error of 1e-5 and mean absolute error of 1e-6. Retain the previous
`optimized_deepsdm_s0` failure. Read only checkpoint tensors through weights-only
loading. These engineering comparisons use no AusPlots species records. If the
nonaggregated setting passes, independently screen its released training
coordinates before drawing any training-disjoint conclusion for that pair.

## Field-record metadata

Use the 151 off-training-grid visits in the metadata audit. Request only the
public `site`, `veg_voucher` and `veg_pi` projections of the
[ausplotsR API](https://github.com/ternaustralia/ausplotsR/blob/master/R/ausplots_queries.r).
Use visit and site identifiers, 100 m by 100 m flag, voucher barcode and
determination-uncertainty flag, point transect/number, and voucher visit date.
Leave determinations, species, taxon IDs, observer identifiers, free text and
genetic barcodes out of the projection. Check response totals, exact visit
identity, voucher-date agreement and distinct point keys. Join barcode-bearing
point records to a unique voucher at the same site and visit. Record missing,
duplicate, unlinked and uncertain records separately. Do not infer plant
absence or identification accuracy from a barcode or from point counts.

The enhanced [TERN cover protocol](https://www.tern.org.au/wp-content/uploads/EMSA-Manual_Cover-Module_v1_20230721.pdf)
uses 10 transects of 101 points each. Require each transect to cover the complete
0–100 or 1–101 integer sequence. Report the 404-point standard and other
effort strata as metadata diagnostics. Use the site geometry flag, enhanced
effort, at least one voucher, unique nonblank within-visit voucher barcodes,
matching dates, and all nonblank point barcodes linked for the strict
record-provenance screen. Missing/uncertain determinations remain potential
unknowns. This is a structural screen. Site visits can pass it while missing
species that were present but undetected.

## Subsequent field comparison

Before opening species fields, freeze the exact released visit snapshot,
training screens for both models, native-range assignments and prediction
decoder. Use strict same-region masks, the greater-than-1-km training stratum,
and the earliest admissible visit per site. Keep repeats in the same site role
for sensitivity analyses. The primary field estimand is the equal-site mean
of the difference in sample F1 over the same feasible recorded inventories.
Use the T6 practical margin of .001, primary empty/empty F1 of one and secondary
empty/empty F1 of zero. Compare shared and separate bounds at the same records
and masks. Use site-clustered inference within 100 km blocks and report the
strict >10 km stratum descriptively when its block support is limited. The
sampling unit and geographical support constrain inference. The precision of
a paired bound describes recorded-inventory uncertainty separately from
between-site sampling uncertainty.

Set the accepted-name and ambiguity rules using the published site metadata and
taxonomic dictionaries before opening species fields. Voucher adjudication and
detection effort determine which occurrences have support and which labels
remain unresolved. No observed voucher list establishes all true absences.
Record and predict on both settings before looking at target rankings. Keep
this source distinct from the previously inspected SAGE replication cohorts.

## Real verification time

Use a paired, blinded review of target compatibility and full identification
on independently adjudicated voucher records. Assign records to the two tasks
in balanced order across reviewers and record active review seconds,
escalations, disagreements and abstentions separately. Preserve separate
costs for common and rare taxa, and for duplicate and individual cases. Report
reviewer- and site-level variation. An automated lookup or a simulation cannot
measure a botanist's time. This stage needs access to actual reviewers, images
or specimens, and separate ground-truth adjudication.
