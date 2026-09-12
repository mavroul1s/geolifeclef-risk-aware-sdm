# Results index

- `experiment_registry.json` is the canonical official-score history and competition target.
- `v20_summary.json` contains the complete compact numerical handoff for the current best run, including splits, model-selection results, audit metrics, official scores and artifact inventory.
- The immutable raw v20 artifacts are stored with Kaggle kernel `con1los/geolifeclef-risk-aware-sdm-phase-1`, version 20. Their exact remote directory is `geolifeclef-risk-aware-sdm/artifacts/environmental_challenger/`.

Large model checkpoints and probability matrices are deliberately not committed to Git. The remote version contains three checkpoints, nine probability arrays, three histories, raw data/policy/report manifests, per-survey audit results and the validated submission CSV. Retrieve only the needed exact filenames through the authenticated Kaggle kernel-output API. The ignored credential is authorized for use but must never be printed or committed.

When a new official submission is scored, append it to `experiment_registry.json`, update the current-best status, and add a version-specific summary before changing the handoff document.
