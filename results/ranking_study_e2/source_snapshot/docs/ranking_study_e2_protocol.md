# E2: ranking utility and source transport

Specified 16 November 2026 following E1 and the user's instruction to continue
and automatically run a GPU experiment when tests pass. This authorizes a bounded
GPU stage without another confirmation if both implementation and scientific
admission checks pass. It does not turn old outcomes into a new holdout.

E2 is development: E1 labels have already been consumed. Use only its probability-
and risk-calibration roles; do not read its assessment targets again for this study.
Keep E1 artifacts immutable. Hash this configuration, implementation and split
identities before fitting E2 candidates. No GPU architecture sweep or competition
submission is part of this protocol.

## Question and candidate

Does optimizing ranking within the groups that receive fixed acceptance budgets
improve selection over Soft Dice Confidence (SDC), where squared-error fitting
failed? Fit a linear residual score `SDC + z(x) @ w`, with training-only robust
scaling and L2 shrinkage toward zero (the unchanged SDC rule).

The pairwise loss is logistic on the score difference between two surveys in the
same country/area group. Orient pairs by observed sample F1 and weight by its
absolute difference. Weight groups by training cell count and cap each group's
pairs deterministically. This is an established ranking approach, not a new
optimization theorem; see [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).
The domain-specific usefulness remains an empirical question.

Fixed penalties .1, 1 and 10; pair temperature 10. Include unchanged SDC in inner
selection. Choose one penalty shared across four released model settings using
inner matched-quota F1; require an inner gain >=.001 to activate a correction.
Tie preference: unchanged SDC, then larger penalty. Controls are a linear
squared-error residual with the same penalties, fixed E1 HGB, mean confidence,
and monotone-affine-calibrated SDC. The last control preserves SDC ordering and
separates magnitude calibration from ranking. No new species-set decoder tuning.

## Two studies

1. Five outer spatial GroupKFold partitions of risk-calibration cells. Three
   geographic inner folds choose the penalty without outer outcomes. Remove
   training cells within 50 km of each validation partition, in outer and inner
   folds. Fixed probability calibration uses the disjoint original probcal role.
   Compute exact country × area acceptance budgets separately within each outer
   evaluation fold, identical for every policy; combine held-out decisions.
2. Hold out the three largest single-source groups with >=80 riskcal cells and
   >=3 riskcal spatial blocks, chosen by metadata before candidate fitting.
   Exclude that source from **both** probability calibration and risk training;
   buffer remaining probcal/risk training 50 km from held-out riskcal cells.
   Refit the probability transform and area bins from remaining probcal only.
   Recompute distance to remaining probcal. Use nested spatial inner validation
   to select penalties from the remaining risk training cells. Require >=200
   remaining probcal and risk-training cells. The published species backbone is
   unchanged; its original PO overlap is not removed by these reliability roles.

All four model settings share spatial/source splits. Report all controls, chosen
penalties, unsupported groups, sample sizes, rare occurrence coverage and error
MSE. A model-wise calibration fit may not see its own validation labels. Use
shared spatial block resamples for fixed-model averages; 2,000 draws and 95%
descriptive intervals do not confer prospective confirmation on this study.

## Automatic GPU admission

Pairwise ranking is the primary candidate; a favorable control cannot replace it
after inspection. Require both studies to show >=.002 mean F1 gain over SDC,
positive interval lower bounds, no model's point loss >.001, no supported source's
mean point loss >.005, and rare-service non-regression (interval lower >=-.02;
every model point >=-.01). Correctness and integrity tests must also pass.

If all checks pass, launch one private Kaggle GPU replay of the selected head,
using fixed saved-prediction inputs/splits, maximum 20 minutes. Verify actual CUDA
execution and numerical/selection stability. This GPU stage tests the accepted
head and does not claim new backbone training or fresh ecological confirmation.
Otherwise record the failed gate and the user's continuing authorization; do not
spend GPU time on a candidate that failed the scientific test. Further backbone
training remains conditional on a concrete hypothesis that requires it.
