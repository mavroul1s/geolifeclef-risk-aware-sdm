# Follow-up D1–D3

Exploratory development follow-up, 14 November 2026. No new neural training,
competition submission or independent accuracy result.

- `protocol_freeze.json`: configuration, initial timestamp, inference notebook and
  source digests; designed after the observed R1 study.
- `inference_execution.json`, `inference_summary/`: first completed inference job;
  literal member-set reproduction failed and is preserved.
- `aligned_inference_execution.json`, `aligned_inference_summary/`: completed
  technical alignment run. Exact member-set guard still fails; a separate recorded
  raw-float16 rank-equivalence audit passes. Regenerated results carry numerical
  tie sensitivity. No scientific F1/regional success margin was changed.
- `alignment_amendment.json`: post-reproduction technical handling, recorded before
  new D1/D2 reliability contrasts, with the scientific configuration unchanged.
- `regional/`: completed D3 CPU analysis. Spatial calibration selected country
  weighting, but its assessment F1 regressed by 0.000545 versus pooled lean and
  the regional safeguard failed. Preserve this negative result.
- `independent_data/audit.json`: completed covariate-only SAGE/sPlotOpen feasibility
  audit. External species labels and predictions were not inspected. Candidate
  non-European, spatially buffered cells are not a locked confirmation cohort.
- `inference_output/`: ignored large role tables and float32 mean probabilities.
- `decisions_ensembles/`: completed D1/D2 exploratory sensitivity results. The
  expected-F1 decoder shrinks lean selection's advantage over simple confidence
  to +0.000885 (interval spans zero). Original immutable R1 averaging comparison
  gives +0.006932 F1 and is reported separately from regenerated predictions.
- `focused_tests.xml`, `validation.json`: final checks and integrity/status scope.

Methods: `../../docs/reliability_followup_d1_d3.md`.
Findings: `../../docs/reliability_followup_d1_d3_results.md`.
Budget and later admission: `../../docs/next_experiment_budget.md`.
The failed R1 safeguard and historical data-reuse limitations remain unchanged.
