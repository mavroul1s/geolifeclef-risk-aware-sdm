# Results index

- `experiment_registry.json` is the canonical official-score history and competition target.
- `v27_summary.json` records the current official best: public 0.23339 and private 0.20831.
- `v27_kaggle_output/` contains exactly the four compact files returned by the completed v27 run; `v27_kaggle_scores.png` preserves the leaderboard evidence.
- `v26_summary.json` preserves the previous official best: public 0.23052 and private 0.20693.
- `v26_kaggle_output/` contains exactly the four compact files returned by the completed v26 run; `v26_kaggle_scores.png` preserves the leaderboard evidence.
- `v25_summary.json` preserves the previous official best: public 0.22812 and private 0.20503.
- `v25_kaggle_output/` contains exactly the four compact files returned by the completed v25 run; `v25_kaggle_scores.png` preserves the leaderboard evidence.
- `v24_summary.json` preserves the previous official best: public 0.22397 and private 0.20094.
- `v24_kaggle_output/` contains exactly the four compact files returned by the completed Kaggle run; `v24_kaggle_scores.png` preserves the user-provided leaderboard evidence.
- The v25 preregistration is `docs/fresh_holdout_adaptive_ensemble_v25.md`; its notebook is `notebooks/geolifeclef_v25_fresh_holdout_adaptive_ensemble.ipynb`.
- The next registered candidate is v28 (`docs/presence_only_shift_moe_v28.md`), delivered as the compact self-contained notebook `notebooks/geolifeclef_v28_presence_only_shift_moe.ipynb`. It embeds the exact scored v27 list and has not yet received an official score.
- `v21_summary.json` and `v21_schema_preflight.json` preserve the earlier PO-expert experiment.
- `v20_summary.json` contains the complete compact numerical handoff for the earlier environmental-challenger run, including splits, model-selection results, audit metrics, official scores and artifact inventory.
- The immutable raw v20 artifacts are stored with Kaggle kernel `con1los/geolifeclef-risk-aware-sdm-phase-1`, version 20. Their exact remote directory is `geolifeclef-risk-aware-sdm/artifacts/environmental_challenger/`.

Large model checkpoints and probability matrices are deliberately not committed to Git. The remote version contains three checkpoints, nine probability arrays, three histories, raw data/policy/report manifests, per-survey audit results and the validated submission CSV. Retrieve only the needed exact filenames through the authenticated Kaggle kernel-output API. The ignored credential is authorized for use but must never be printed or committed.

When a new official submission is scored, append it to `experiment_registry.json`, update the current-best status, and add a version-specific summary before changing the handoff document.
