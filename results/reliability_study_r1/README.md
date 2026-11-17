# Controlled reliability study R1

Development research on previously consumed PA labels; no new competition score.

- `protocol_freeze.json`, `protocol_initial_config.json`, `protocol_amendment_1.json`:
  initial protocol and a documented amendment before inspecting new GPU accuracy.
  The primary contrast isolates model signals from environmental novelty.
- `gpu_probe_execution.json`, `gpu_probe_output/`: completed Tesla T4 probe,
  actual modalities and finite 5,016-label backward; 13.09 s inside the notebook.
- `smoke_execution.json`, `smoke_output/r1_export/`: completed real-data smoke,
  3,712 surveys, two seeds, two epochs; all manifest-listed outputs verified.
  Promotion used integrity/numerics/runtime, never accuracy changes.
- `smoke_amendment_validation.json`: amended CPU selector family checks on real
  model confidence features without assessment outcomes.
- `full_admission.json`: conservative estimate 2.465 hours inside a 5.5-hour guard,
  excluding Kaggle queue/provisioning latency.
- `full_execution.json`: completed private two-fold/three-seed GPU run; all six
  neural fits completed in 69.45 minutes inside the notebook, with no submission.
- `full_summary/`: compact downloaded reports, curves and the output manifest.
- `analysis/`: verified primary analysis, country/area coverage, spatial intervals,
  individual-member metrics, rare-species coverage and composition decomposition;
  `posthoc_signal_ablation.*` are explicitly exploratory CPU feature comparisons.
- `evaluation_validation.json`: final analysis/source integrity and focused tests.
- `local_validation.json`: unit/integration checks and source hashes.
- `external_overlap_audit.json`: official new GeoPlant IID and OOD metadata have
  disjoint ID sets whose union equals the existing 14,784 GLC25 test IDs. They have
  no species labels and do not supply new independent confirmation data.

Large full prediction matrices/checkpoints belong in ignored `full_output/`.
Keep compact reports, manifests, hashes and derived analysis separately.
Protocol: `../../docs/reliability_study_r1.md`.
Interpretation and next work: `../../docs/reliability_study_r1_results.md`.

**Outcome:** primary candidate 0.375382 versus metadata/distance 0.360641;
gain +0.014741 (descriptive spatial interval +0.011134 to +0.018025), positive in
both folds. Gain margin passes, regional safeguard fails (Switzerland -0.012164,
49 surveys). Mean confidence reaches 0.370927. Post-hoc lean decision-confidence
features match the full bundle, with no clear extra value from disagreement.
Independent confirmation and a conservation benefit remain unestablished.

Primary comparison: metadata/distance/model signals versus identical
metadata/distance inputs at 80% retention and identical country-by-area quotas.
Full mean probabilities use float16 precision. Exact primary scores, risk features
and acceptance masks are in CSV; tied species rankings recomputed from quantized
probabilities may differ.

Private jobs: [probe](https://www.kaggle.com/code/nmavros/glc25-reliability-r1-gpu-probe),
[smoke](https://www.kaggle.com/code/nmavros/glc25-reliability-r1-smoke),
[full](https://www.kaggle.com/code/nmavros/glc25-reliability-r1-full).
