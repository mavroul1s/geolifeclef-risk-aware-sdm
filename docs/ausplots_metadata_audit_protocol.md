# AusPlots metadata-only spatial feasibility audit

Freeze the screen before requesting the full site-visit table. Retrieve the
public ausplotsR site endpoint with a projection to site ID, visit ID, visit
start, latitude and longitude. Require a complete server-counted response, and
check identifiers, duplicates, date parsing and coordinates. Use no voucher,
species occurrence, herbarium or environmental feature table in this audit.

Compare dates with the last parseable SAGE Australian AEKOS occurrence in
the evaluation metadata. For visits after that date, report exact coordinates,
1 km grid overlap and nearest distances to observed SAGE evaluation source
plots and published SAGE presence-only training locations. Use EPSG:6933 for
the published 1 km grid and great-circle distance to the plot coordinates.
Use all valid Australian evaluation plot coordinates, including undated
rows. Compute exact grid membership across the entire reference rather than
assuming the nearest point is in the candidate's grid cell. For the
SAGE model-input location archive, check whether candidates have a
float32-identical inverse-transformed 1 km center. A match flags a possible
existing input row, with readiness requiring a separate feature audit.

Treat each site as a dependence cluster for visits. Report counts of visits,
sites, distinct grid cells and 100 km grid blocks. Report counts at increasing
restrictions: post-date, greater than 1 km from the legacy evaluation source
plots and outside their grid cells, outside the PO training grid, and more
than 1 km and 10 km from training. Show the effect of avoiding known AusPlots
sites already visited by the legacy cutoff. These are screening strata.
An exact crosswalk from legacy source plots to modern visit keys may still
be absent, and spatial support does not identify particular training records
or guarantee independence. Do not assign primary outcomes, admit a final
cohort, estimate field cost or choose a statistical power threshold here.
