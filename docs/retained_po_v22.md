# v22: retained PO knowledge and complementary checkpoint selection

Preregistered 2026-09-14, before any v22 training or outer assessment. One master notebook, existing private Kaggle kernel, competition PA/PO/provided predictors only, random initialization, all 5,016 PA species. Current official best is v21: **0.21633 public / 0.19426 private**. The target is 0.2302 private (gap 0.03594), not a SOTA guarantee.

## Diagnosis and bounded hypotheses

v21's production calibration improved only 0.3683761122 to 0.3684556763 and selected up to 2.5% PO. Raising uniform PO weight to 20% reduced calibration to 0.3668758123. Its outer PO mixture recovered 124 rare-species presences versus 127 for the matched reference, and zero of 389 presences from species absent in that fold's PA training. These are consumed development diagnostics, not a v22 assessment.

The previous expert selected checkpoints by standalone F1 although deployment combined it with the baseline. The v22 selection criterion instead scores a fixed **10% PA-distance-gated mixture** with the matching frozen baseline, top20. Only the checkpoint-selection partition supplies those labels; final mixture calibration stays separate.

The hypothesis that PA adaptation loses useful PO representation is unproven. Test it with a fixed held-out PO-group diagnostic and before/after PA checkpoint diagnostics, without selecting models on the PO diagnostic. The primary candidate is preregistered; the ablation must not become a replacement after observing assessment results.

## New cross-fitting and actual preflight counts

Reconstruct the exact old v20 and v21 partitions from coordinates. Only the **25,511 former v21 training surveys** are eligible for new outer assessment; both consumed audits and old checkpoint/calibration rows are excluded from that role. Hash whole one-degree blocks with SHA-256 and salt 20250922; buckets 0–19 form fold 0, 20–39 fold 1. Do not retry salts or change folds after performance inspection. This preflight used coordinates/country counts, not scores.

Within each fold, train a new matching baseline and expert. Inner checkpoint/calibration assignments use separate fixed hashes (20251022/20251023), 10% each, within original v20 training. All outer-assessment IDs from both folds are excluded from inner selection/calibration. An outer fold may enter the other model's training, as in cross-fitting, but never its model-selection labels. Exclude entire own-assessment blocks from fitting and remove PA training rows within 20 km of any selection, calibration or own-assessment row. Fit all PA-derived normalizers/support statistics on that fold's actual buffered training rows.

| Role | Fold 0 | Fold 1 |
| --- | ---: | ---: |
| Training | 43,207 | 36,847 |
| Checkpoint selection | 1,579 | 3,040 |
| Calibration | 5,141 | 6,793 |
| Final assessment | 7,435 | 7,044 |
| Assessment blocks | 28 | 36 |

The pooled assessment has **14,479 surveys in 64 disjoint one-degree blocks**, including 6,074 Danish surveys and 38 Bulgarian surveys. Ukraine and Switzerland still have no new assessment samples. Country counts and distance strata are descriptive; small-country uncertainty must be disclosed. Cross-fit training overlap means the spatial bootstrap does not capture every source of dependence; this is not multiple fully independent experiments or an unbiased hidden-test estimate.

Both folds' checkpoints, calibration policies and the separate production outputs must be persisted and frozen before either outer fold is scored. Neither old audit can be relabeled untouched in a future experiment.

## Frozen baseline and separate deployment

Preserve v21 version 21 before advancing the same kernel. Reuse its exact two deployment expert probability matrices, policy, report and submission, plus the existing private v20 probability bundle. No original v20/v21 weights are retrained merely to reconstruct outputs. Source v21 report hash: `60858bca1e00cbc7fcd52641785975cb6c43a950e6dbd410b2f7e02a55effce2`; official CSV hash: `93c1d03258f87131eaaf3000dc9df01fe66b2dd61771e0fb4642a1a4a98847a1`. Locally reconstructed float32 rank probabilities match exactly; five tie-order rows differ across platforms without top20 membership changes. Preserve the original CSV at zero new weight.

Private inputs are `con1los/geolifeclef-v20-frozen-control/1` and `con1los/geolifeclef-v21-frozen-control/1`. The latter contains only five allowlisted, hashed v21 artifacts plus provenance/metadata; source checkpoint and additional manifests remain in ignored local preservation storage. Upload requires the same authenticated owner and private visibility. Credentials/signed URLs must never be logged.

For each outer fold, refit the v20 reference plus two challenger architectures on its fresh buffered training set, with the original seeds, schedule and fixed 25%/37.5%/37.5% blend. Fit the original v21 width-256 PO expert on the original 0.05-degree pseudo-survey recipe. Freeze the matched v21 control as that blend plus the **official production v21 policy**, 2.5% PA-distance-gated PO. This uses the actual deployed recipe, not v21's different 20% experimental evaluation mixture. Baseline selection predictions are saved for complementary v22 checkpoint selection. Report both v20 and v21 controls.

Production reuses the exact original v21 predictor, with no new multimodal fit. The new experts train on the 71,332 original v20 training rows. Split the old v20 calibration partition by a fixed whole-block hash (20251122) into **2,536 checkpoint-selection** and **4,807 calibration** rows. This provides available, matching original v21 probabilities for both roles without reconstructing missing original selection outputs. Those rows are development data already used by the frozen baseline's historic calibration; production calibration is not a new unbiased assessment.

The separate production training includes some outer-assessment labels but cannot influence either outer model or policy. The final gate therefore evaluates recipe transfer, not unseen-label performance of the exact deployed predictor.

## PO construction

Keep the original v21 construction unchanged for its control. For all new PO arms, use **0.01-degree cells**, publisher and the existing coarse numerical land-cover stratum. Retain competition schema checks, all-species vocabulary, binary cell/publisher/stratum/species deduplication, missingness handling and 100m exclusion around every PA train/test coordinate without reading PA labels.

Replace the minimum-ID environmental representative with the finite mean of **unique source survey IDs within each group**. Repetition and the number of species recorded at a survey cannot multiply its environmental weight. This reduces the mismatch between aggregated labels and a single representative; it still does not establish ecological homogeneity or make PO unobserved species reliable absences. Preserve group support and species coverage.

Start from v21's within-cell publisher/stratum and block-density weighting, then cap the **global publisher sampling share at 30%** by redistributing remaining mass proportionally among uncapped publishers. If fewer than four publishers exist in a synthetic test, use a feasible cap max(0.30, 1/publisher_count). This is a sampling control, not proof of unbiased ecology. Save actual publisher draw shares. Do not use external land-cover maps or reinterpret supplied numerical columns as class probabilities.

## Three fixed arms and training

Common new model: random width-384 MLP with two residual blocks, the same 163 environmental/geographic/missingness inputs, and separate 5,016-class PA/PO linear heads. All three arms use the same PA seed (20250922), training row order, width, optimizer schedule and complementary selection criterion.

PO pretraining is shared byte-identically by the two PO arms: 16 epochs, at most 300,000 weighted draws with replacement per epoch, batch256, AdamW lr0.0008, weight decay0.001, original positive-mean plus 0.05 weak-background loss. Hold out min(10,000, 5% of groups) using a fixed random group permutation. Those groups never supply pretraining or rehearsal updates. Their diagnostic is not a spatial-independent PO assessment and never selects a checkpoint, epoch count or hyperparameter.

1. **retained_po (primary):** copy the pretrained encoder and classifier into PA/PO heads. PA adaptation trains the PA task plus a PO-head auxiliary loss on every batch. A frozen copy of the pretrained encoder anchors normalized PO representations. Total loss = PA asymmetric loss + **0.001 × PO weak-background loss + 0.01 × normalized-encoder MSE**. Save each component and PO diagnostic before/after. Final predictions use only the adapted PA head, never uncalibrated raw PO scores.
2. **single_head (retention ablation):** identical pretrained initialization, inputs, PA schedule and selection rule; no PO auxiliary/anchor during adaptation. The preserved PO head is diagnostic only and receives no updates. This tests the joint retention intervention, not each loss term independently.
3. **zero_po:** same new architecture initialized from scratch, identical PA schedule/selection, no PO updates. PO and zero-PO arms have unequal total training exposure; an architecture-only causal claim is not supported.

PA adaptation: AdamW lr0.0004, weight decay0.001, cosine decay to0.00002 over48 epochs; minimum20, patience10, batch256, mixed precision and gradient clipping2. Checkpoints maximize sample F1 of the fixed 10% PA-distance mixture on checkpoint-selection rows. PO diagnostics and standalone checkpoint F1 are recorded without controlling selection.

The full intervention does not guarantee recovery of species absent from PA fitting, since final inference still uses the PA head. Report training-frequency recall and before/after diagnostics to evaluate that limitation honestly.

## Calibration, assessment and submission gate

For each path and arm, independently calibrate alpha in **[0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50]**, uniform or the existing PA-distance gate clip(log1p(distance)/log51,0,1), always top20. Zero weight is included and exact ties prefer smaller intervention. No candidate architecture is selected from the final assessment.

Report pooled sample F1 first, each fold, countries/distances and rare/common/zero-PA-support recall. Compare the primary mixture to frozen matched v21, frozen matched v20, zero-PO and the retention ablation. Recompute paired whole-one-degree-block bootstrap with the existing fixed 500 resamples/seed20250921; keep training-dependence caveats explicit.

Make at most one official submission only if: primary gain over matched v21 has positive lower95% spatial interval; primary beats v21 in **each fold**; pooled primary gain over v20 and zero-PO is positive; both outer and production PO weights are nonzero; and all data/source/freeze/runtime/output checks pass. The single-head ablation is diagnostic, never a post-assessment replacement candidate. A failure does not authorize a revised model or second submission in this run.

The external validator must independently reproduce per-survey means, fold counts/IDs, spatial intervals, policy/CSV hashes and exact unchanged-v21 parity; enforce 14,784 template-ordered unique IDs, 5,016 vocabulary, top20; verify the launched source commit and both notebook test passes. It writes an exclusive receipt before the only submit request. Unknown outcomes are inspected, never automatically retried. Retrieve and record public/private scores only if actually provided by Kaggle.

## Compute and implementation discipline

One master notebook and one default T4 device, Internet disabled, hard **10.5-hour end-to-end** guard including source setup, preparation, all controls/arms, inference, assessment and tests. Preprocessing reserves subsequent training time; each model has a bounded stage deadline. Prepare each fold's multimodal arrays with its own training-only normalization. Remove only resolved generated cache paths after needed outputs are saved. No new run merely to reproduce v20/v21 outputs.

Use apply_patch, meaningful unit/real CPU smoke tests, then commit before API push (which archives Git HEAD). The push now refuses uncommitted source. Save all policies/checkpoints/diagnostics, exact source commit and manifests. Keep final assessment closed until every model and production policy is frozen. Monitor only the authorized run; honor any subsequent user instruction to take over monitoring.
