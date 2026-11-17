# Exploratory evidence audit, 13 November 2026

All prediction analyses use historically consumed development/calibration labels.
These files do not contain a new official score or independent confirmation.

| File | Content |
|---|---|
| `findings.json` | Paired development changes, hash checks, metadata coverage, prefix oracles |
| `official_history.csv` | Repository-recorded public/private scores, without new score-API attestation |
| `country_support.csv` | Official PA train versus unlabeled test country counts |
| `country_gains.csv` | Within-run paired F1 changes by country |
| `prefix_oracle_diagnostics.csv` | True-label-selected prefix diagnostics; never deployable gains |
| `cardinality_pilot.{csv,json}` | Four training-only metadata count fits with frozen neural rankings |
| `selective_pilot.{csv,json}` | Six calibration-only risk fits; reused spatial assessment |
| `selective_country_coverage.csv` | Country retention, including zero-coverage groups |
| `evidence_audit.{png,svg}` | Competitive trajectory, geography and development transfer |
| `selective_pilot.{png,svg}` | Reliability/coverage curves and geographic exclusion |
| `validation.json` | Actual checks, library versions and analysis-source hashes |
| `kaggle_execution.json` | Private CPU reproduction request/status; no competition submission |
| `kaggle_cpu_output/` | Three downloaded outputs from the completed CPU run; 126 summary rows exactly match local results |

The exact GPU competition recipes, original score registry and archived outputs
were preserved. Six CSV hashes require original CRLF bytes, which the audit
checks in memory without editing the archive. The historical full suite initially
had 22 fixture/hash failures; its status is disclosed rather than reported green.

Read `../../docs/research_audit_2026_11_13.md` for interpretation and
`../../docs/paper_research_protocol.md` for proposed new research.
