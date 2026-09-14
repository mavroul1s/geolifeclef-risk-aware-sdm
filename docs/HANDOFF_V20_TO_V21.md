# Handoff: completed v21, continuing from v20

Read this file before taking any action. It is the compact source of truth for a fresh Codex conversation.

## Authorized v22 implementation — 2026-09-14

The user authorized the next experiment after reviewing the v21 diagnosis. See `docs/retained_po_v22.md` and `results/v22_summary.json`. Implementation includes two new outer folds from former v21 training, complementary checkpoint selection, smaller PO cells with unique-survey environmental means, a global publisher sampling cap, and retained-PO/single-head/zero-PO arms. **101 local tests pass**, including real CPU training/orchestration for all three arms and independent submission validation. Master notebook code and deployment-script syntax pass. Kaggle version **22** was successfully launched from commit `519ae6c`; no v22 official submission has been made.

Original v21 artifacts have been selectively preserved under ignored `artifacts/v21_frozen_source/`. The compact input `artifacts/v21_frozen_bundle/` passed exact original-rank probability parity; the original CSV is preserved. Local tests/docs were completed and committed (`1ba5cae`). **The user explicitly approved the private 223.7 MB transfer to `con1los/geolifeclef-v21-frozen-control`; do not ask again.** Dataset version `/1` is private and ready. The payload contains competition-derived probabilities/policies/report/submission only, no credentials or external data. Attach it alongside the existing frozen-v20 bundle before pushing. The full suite passed again (101 tests) immediately before launch preparation.

New assessment preflight: 7,435 + 7,044 surveys, 28 + 36 disjoint blocks, 20km training buffers, 38 Bulgarian surveys and no Ukraine/Switzerland. All models/policies and production predictions must freeze before either fold is scored. `scripts/submit_v22.py` independently validates the source commit, per-survey metrics/intervals, IDs, policies and CSV before its exclusive at-most-once submission. Current official best remains v21 until a permitted v22 official result establishes otherwise.

## Completed v21 — 2026-09-14

**Current best: v21, 0.21633 public / 0.19426 private.** Exactly one official submission was made after the registered assessment and integrity gate passed: ref `56221053`, status `complete`, request started `2026-09-14T04:02:39.372790+00:00`. Private gain over v20 is **0.00066**; gap to the 0.2302 target is **0.03594**. This is a modest improvement, not SOTA. Do not resubmit or push another run without a new experiment request.

Kaggle version 21 completed from tested source commit `c04358425500da006acc0572b786f23f0fd9f4a2` in **1.132513 hours (67 min 57 sec)** including setup, preparation, training, inference and tests, below the 10.5-hour guard. Both notebook test passes succeeded (85 tests); the final local suite including submission validation passed 88 tests. Kaggle exposed two CUDA devices, but the implementation used the default device only. The user temporarily stopped API monitoring, then announced completion and authorized retrieval and continuation. No duplicate notebook run was launched.

PO processing read 5,079,797 competition records, retained 4,212,323 after vocabulary/coordinate exclusions, deduplicated to 2,853,929 presences and constructed 335,592 sampling-weighted pseudo-surveys. PO covered 4,308 species; all 5,016 PA outputs were retained. The deployment mixture is frozen original v20 plus at most 2.5% PO expert with the registered PA-distance gate, top20. The separately calibrated evaluation mixture used 20% PO uniformly; this difference was frozen before assessment and must remain explicit.

New assessment results: PO mixture **0.3662031171**, matched frozen-v20 recipe **0.3646102178**, zero-PO mixture **0.3653365284**. Gain versus matched v20 is +0.0015928993 with spatial-bootstrap 95% CI [0.0004665888, 0.0031759479]; gain versus zero-PO is +0.0008665887, CI [0.0003444368, 0.0031763624]. All report integrity checks passed. Local review independently recomputed means and intervals from 17,163 saved per-survey scores, checked frozen policies/digests, template order/vocabulary and exact original v20 CSV parity. The final CSV changes species sets on 4,171/14,784 rows. These internal scores are not hidden-test scores.

**Both the old v20 audit and new v21 assessment have now been examined and are consumed.** Do not relabel either untouched or tune against them while claiming a fresh evaluation. No changes were made to the candidate after assessment or leaderboard feedback.

Actual Kaggle reads confirmed the P0 metadata spelling, publisher field and all five PO environmental families (64 raw predictors). Six original v20 calibration/test probability arrays were downloaded selectively and combined without retraining. Their compact verified input is **private and ready**, pinned at `con1los/geolifeclef-v20-frozen-control/1` on the same authenticated account. Original top20 ties are preserved by retaining the exact original submission CSV; reconstructed float32 rank scores match every row exactly.

The old v20 checkpoints cannot provide an untouched assessment on a rehashed subset. A newly trained, subsequently frozen control uses the unchanged v20 recipe on new buffered training rows; the original probabilities remain the production baseline. New assessment: 17,163 surveys in 28 blocks, at least 20 km from the new PA training. It is Denmark-dominated and contains no Bulgaria/Ukraine/Switzerland surveys. This limits evidence about the intended priority countries. The separate production fit must never influence the outer candidate or policy.

Evidence: `results/v21_summary.json`, `results/experiment_registry.json`, and ignored `artifacts/v21_review/` containing the original report, per-survey assessment, manifests, frozen policies, independent validation, CSV, exclusive submission receipt and official scores. Submission SHA-256: `93c1d03258f87131eaaf3000dc9df01fe66b2dd61771e0fb4642a1a4a98847a1`. Report SHA-256: `60858bca1e00cbc7fcd52641785975cb6c43a950e6dbd410b2f7e02a55effce2`. Large v21 checkpoints/probabilities remain in the existing kernel version 21 under `geolifeclef-risk-aware-sdm/artifacts/ood_po_expert_v21/`. The guarded reader only accepts the current requested version, so preserve needed artifacts before any future version advance; do not silently retrieve a newer output as v20/v21.

## Historical v20 handoff (retained context; current result above)

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

Official v20 result: **0.21601 public / 0.19360 private**. This beat v18 (0.21594 / 0.18900) and was the repository best before v21. Its gap to the 0.2302 winning private target was 0.03660. v19 regressed to 0.19891 / 0.17516.

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

## Original v21 recommendation (now implemented and completed)

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
