# Prompt for a fresh conversation

Copy the text below into a new Codex conversation opened on this repository.

---

Continue the GeoLifeCLEF project from the completed v20; do not restart from zero. Work in `C:\Users\nickb\Documents\projects\geolifeclef-risk-aware-sdm`.

First read completely:

- `docs/HANDOFF_V20_TO_V21.md`
- `results/experiment_registry.json`
- `results/v20_summary.json`
- `docs/environmental_challenger_v20.md`

Then inspect the relevant existing implementation and Git status. Treat those files as the source of truth, but verify claims against code/artifacts before acting.

Current official best is v20: **0.21601 public / 0.19360 private**. The GeoLifeCLEF 2025 winning private target is **0.2302**. v20 beat our previous baseline but is not SOTA. Its standalone challenger lost internally; only the frozen ensemble won. The old v20 audit has already been observed and cannot be reused as an untouched v21 assessment.

Implement the recommended next step: one compute-bounded **v21 competition-only OOD/PO expert** that targets transfer to PA-sparse test geographies. Use the five-million-row competition PO data intelligently through deduplicated/sampling-aware pseudo-surveys or a masked PO representation, and combine it conservatively with the frozen v20 predictor. Inspect the actual Kaggle schemas before fixing the design. Include unchanged-v20 and zero-PO controls, new geographic validation/cross-fitting, separate selection/calibration/final assessment, equal-budget comparisons where needed, all 5,016 PA species, and explicit integrity/runtime checks. Do not reuse v19's raw nearest-96-record heuristic unchanged.

Keep everything in the existing master notebook and kernel; supporting scripts are fine, but do not create many notebooks. Use only GeoLifeCLEF 2025 competition PA/PO and predictors—no external data or pretrained weights. It must fit a Kaggle T4 run within an enforceable 10.5-hour end-to-end guard under the 12-hour limit. Reuse/download the existing v20 checkpoints and probability artifacts from Kaggle version 20 when useful; do not retrain v20 just to recreate them. Never expose the ignored `api_key/kaggle_2.json` or signed URLs. The user has explicitly authorized using that existing key for Kaggle API execution.

Work autonomously end-to-end: research only what is necessary, implement, test locally, update documentation/results, commit the tested source because the push script archives Git HEAD, launch through the Kaggle API, monitor until completion, and analyze results. If and only if v21 beats a compatible frozen-v20 control on the new untouched assessment and validation checks pass, submit it once to the official Kaggle competition, retrieve public/private scores, and save them in `results/experiment_registry.json` and the handoff. Never guarantee SOTA or confuse internal with hidden-test scores. Explain in plain Greek what changed, how long it ran, whether it beat 0.19360, and how far it remains from 0.2302.

---
