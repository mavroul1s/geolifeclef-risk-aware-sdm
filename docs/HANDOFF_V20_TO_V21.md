# Handoff: completed v20 and recommended v21

Read this file before taking any action. It is the compact source of truth for a fresh Codex conversation.

## v21 implementation progress — 2026-09-13

The v21 source is implemented and 85 local tests pass. See `docs/ood_po_expert_v21.md`, `results/v21_schema_preflight.json` and `results/v21_summary.json`. Kaggle training has not yet been launched at this checkpoint; there is no v21 assessment or official score yet. The master notebook now runs only v21 through `scripts/launch_ood_po_expert.py` and `scripts/run_ood_po_expert.py`.

Actual Kaggle reads confirmed the P0 metadata spelling, publisher field and all five PO environmental families (64 raw predictors). Six original v20 calibration/test probability arrays were downloaded selectively and combined without retraining. Their compact verified input is **private and ready**, pinned at `con1los/geolifeclef-v20-frozen-control/1` on the same authenticated account. Original top20 ties are preserved by retaining the exact original submission CSV; reconstructed float32 rank scores match every row exactly.

The old v20 checkpoints cannot provide an untouched assessment on a rehashed subset. A newly trained, subsequently frozen control uses the unchanged v20 recipe on new buffered training rows; the original probabilities remain the production baseline. New assessment: 17,163 surveys in 28 blocks, at least 20 km from the new PA training. It is Denmark-dominated and contains no Bulgaria/Ukraine/Switzerland surveys. This limits evidence about the intended priority countries. The separate production fit must never influence the outer candidate or policy.

Next: commit the tested source and launch `scripts/push_kaggle_kernel.ps1 -RunMode ood_po_expert_v21`, which archives Git HEAD. Monitor to completion. The existing legacy submission helper is not sufficient for v21: independently enforce the saved v21 assessment/integrity gate, runtime, both notebook test passes, exact version/source commit, vocabulary/template/hash checks and an at-most-once receipt before any official submission. No submission if any gate fails. Record actual public/private scores only if a permitted submission is made.

## Current state

- Repository: `C:\Users\nickb\Documents\projects\geolifeclef-risk-aware-sdm`
- Branch: `master`; v20 source commit `1a2528b`; submission validation commit `f6f98f0`.
- Private Kaggle kernel: `con1los/geolifeclef-risk-aware-sdm-phase-1`; version 20 is complete.
- Direct API execution is implemented in `scripts/push_kaggle_kernel.ps1`. It archives **Git HEAD**, so commit source changes before pushing.
- The user explicitly authorized the ignored repository credential `api_key/kaggle_2.json`. Read it only in memory. Never print, commit, copy, or expose its values or signed output URLs.
- The user wants one master notebook, not many notebooks: `notebooks/geolifeclef_research_pipeline.ipynb`.
- Kaggle limit is 12 hours. Use an enforceable 10.5-hour total guard including setup, preprocessing, training and inference. Request one T4. Internet remains disabled inside the notebook.
- Data scope: only the GeoLifeCLEF 2025 competition data, including competition PA and PO. No external datasets or pretrained weights unless the user explicitly changes this rule. Retain all 5,016 PA species.
- Run tests locally and on Kaggle. Current suite passed 29 tests before v20.

## What has actually been established

The exact numeric registry is `results/experiment_registry.json`; the full compact v20 record is `results/v20_summary.json`.

v20 completed in 0.8682 hours. Its internal untouched audit selected a frozen 75% two-seed challenger / 25% matched-reference blend with top-20 output:

- Calibration: 0.36838 versus best reference-only 0.35329.
- Untouched audit: 0.32342 versus calibrated reference 0.31413 (+0.00929).
- Italy/Switzerland aggregate: 0.18713 versus 0.18150.
- Single challenger seed: 0.30724, below the reference. The architectural claim alone is therefore not supported.

Official result: **0.21601 public / 0.19360 private**. This beats v18 (0.21594 / 0.18900) and is the current repository best, but remains 0.03660 below the 0.2302 winning private target. v19 regressed to 0.19891 / 0.17516.

Do not compare internal F1 numerically to hidden-test F1 as though they were the same split. The direction of v20's audit gain transferred, but the official gain was only +0.00460. The v20 audit has now been examined and is no longer an untouched set for v21 decisions.

## Available v20 implementation and artifacts

Core files:

- `scripts/prepare_environmental_challenger.py`: survey-ID joins, training-only normalization, memory-mapped PA features, fixed reflectance scaling and spatial partitions.
- `src/geolifeclef/environmental_model.py`: environmental/temporal/Sentinel challenger.
- `scripts/run_environmental_challenger.py`: reference plus two seeds, frozen calibration and audit, submission creation.
- `scripts/submit_kaggle_output.ps1`: validation and official submission.
- `docs/environmental_challenger_v20.md`: preregistered design and interpretation.

The Git repository intentionally does not contain large checkpoints and probability matrices. They remain retrievable from Kaggle version 20 under `geolifeclef-risk-aware-sdm/artifacts/environmental_challenger/`. Available items include all three best checkpoints; calibration/audit/test probabilities for each; per-model histories; the raw report; data manifest; frozen policy; per-survey audit CSV; and final submission CSV. Do not retrain v20 merely to recreate these. Use the authenticated `/api/v1/kernels/output` endpoint, select entries by exact `fileName`, and download only what v21 needs. Never print the returned signed URLs.

## Known weaknesses

1. The v20 standalone challenger did not beat the reference. Its win depends on ensembling and two challenger seeds versus one reference seed, so capacity and seed-count effects are not isolated.
2. OOD performance is still weak. Italy improved, while Switzerland fell from 0.24772 reference to 0.23124 selected. Official test is concentrated in countries poorly represented by PA training, especially Bulgaria, Ukraine and Switzerland.
3. v20 does not train a representation from the five-million-row competition PO table. The v19 nearest-record PO heuristic was weak: it used the nearest 96 raw records and was sensitive to sampling density. Do not revive it unchanged.
4. Half-degree spatial blocks are not a distance buffer. The v20 country audit had very small samples for several countries. A stronger new evaluation protocol is needed.
5. v20 uses 32x32 Sentinel patches and random initialization. Larger imagery or heavier models must be justified under the T4/10.5-hour budget.
6. The 0.2302 target belongs to the finished 2025 competition. Do not call a method universal SOTA, and do not promise that it will beat the target.

## Recommended next experiment: v21 competition-only OOD/PO expert

The next step should target the hidden geographic shift rather than enlarge the same PA model again. Build one bounded v21 notebook around a competition-only PO expert and a conservative mixture with the frozen v20 predictor.

Before coding, inspect the exact PO/environmental schemas mounted on Kaggle and the v20 artifacts. Then choose the smallest viable design, preferably:

1. Aggregate/deduplicate PO occurrences into spatial cells and ecological/land-cover strata, restricting outputs to the 5,016 PA species. Correct or cap publisher/sampling-density dominance; save support counts.
2. Train a compact environmental/location representation or multi-label PO expert on these pseudo-surveys. Do not treat unobserved species as ordinary reliable absences without masking/down-weighting.
3. Combine it with frozen v20 probabilities through a small OOD-aware gate using only features available for both validation and test (for example PA/PO distance or density, environmental embedding and region/country proxies). Always include zero-PO and unchanged-v20 controls.
4. Use new preregistered geographic folds or cross-fitting. The old v20 audit cannot be the new untouched assessment. Keep checkpoint selection, policy calibration and final assessment separate. If a final refit changes the predictor, recalibrate predictions produced by the matching training recipe.
5. Compare equal seed counts/budgets where architecture claims are made. Report sample F1 first, then per-country/distance strata, rare/common species, cardinality, runtime and uncertainty.
6. Only create an official submission if the v21 candidate beats a compatible frozen-v20 control on the new assessment and all integrity checks pass. Submit once, save its official public/private result back into the registry and handoff.

Relevant primary/public references already identified: PredComX `https://ceur-ws.org/Vol-4038/paper_261.pdf`, Tighnari `https://arxiv.org/abs/2602.08282`, and public comparator `https://www.kaggle.com/code/lonansyayf/2025-model-geolifeclef`. These informed direction only; their external weights/data and larger compute are not authorized here.

## Completion discipline

Inspect `git status` and preserve unrelated user changes. Reuse existing scripts rather than starting a second repository. Use `apply_patch` for edits. Commit the tested v21 source before API launch. Monitor the kernel until completion, analyze the raw report, and report failures honestly. A notebook that merely runs is not evidence; an official score above 0.19360 is progress, and above 0.2302 beats the recorded competition target.
