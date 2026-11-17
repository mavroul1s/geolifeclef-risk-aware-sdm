# E4 NEON transfer: completed, confirmation not admitted

**The GPU/data pipeline ran, but the proposed scientific claim is not confirmed.**
The original analysis retains only 65 cells / 10 sites under the frozen
known-label rule, below the required 100 cells / 12 sites. HGB's MSE is worse
than the training-mean control, and its selective-F1 contrast with SDC is
uncertain. A later correction using NEON's documented morphospecies join raises
eligibility to 77 cells / 11 sites; it still falls below admission and does not
reverse the descriptive MSE result. The corrected replay is explicitly post-hoc.

## What actually ran

The [locally frozen protocol](confirmation_e4_protocol.md) and configuration
were locked at **17:41:20 UTC on 16 November 2026**, before NEON composition.
The fixed released MLP `aggregation_s0` uses 52 environmental features. All
vectors for the 163-cell cohort come from identical published float32 grid-center
coordinates; this is a **training-support/input-availability stratum**, not
unseen-backbone-training-location generalization. It contains 301 plots at
16 sites. The other 46 candidate cells remain outside E4 because their full
predictor bridge is incomplete, not because of scores.

Probability calibration uses 1,165 original E1 probability-calibration rows;
quality heads use 1,061 original E1 risk-calibration rows. These are consumed
development labels. No E1 assessment labels enter E4 fitting. Both domains use
the 2,453 publicly matched SAGE classes intersected with native masks.
The target is a modeled native subset, not complete biodiversity.

One private [Kaggle GPU inference job](https://www.kaggle.com/code/nmavros/neon-e4-inference-20261116)
completed on a **Tesla T4 in 52.14 seconds inside the notebook**, excluding
queue/startup. It produces reference, calibration and NEON-location probabilities,
with no labels, token, training or submission. Full reference verification keeps
the original maximum 1e-5 / mean 1e-6 tolerances: observed maximum **1.103e-6**,
mean about 1e-9. Two collected files total **55,579,067 bytes**, with hashes checked.

Transforms, heads, scores and sets were frozen at **17:47:06 UTC**. First NEON
assessment access is recorded at **17:48:06 UTC**. Unselected plot/bout taxon
values are discarded as bytes without decoding. Unknown taxon labels affect
evaluation only; they never alter frozen sets or confidence features.

The co-primary hypothesis required HGB to improve transported MSE over the
training-mean reference by >=.001 while losing selective F1 to SDC by >=.002,
with the corresponding 97.5% site intervals excluding zero. Sample-size
admission is also required. It fails; no secondary replaces it.

## Original frozen result

All methods receive identical 80% site × single/multiple-plot quotas. The
numbers below are **descriptive on an admission-failing subset**, not a successful
confirmation or an independent model replication.

| Quantity | Original E4 |
|---|---:|
| Scorable cells / sites | 65 / 10 |
| Full-coverage known-label F1 | .424802 |
| Training-mean quality MSE | .049188 |
| HGB quality MSE | .104674 |
| Positive-affine SDC quality MSE | .051257 |
| Raw SDC MSE | .046757 |
| HGB-minus-SDC selective F1 | +.007478 |
| 97.5% site interval | [-.010459, +.030027] |
| Mean-minus-HGB MSE gain | -.055486 |
| 97.5% site interval | [-.075903, -.027711] |

At matched quotas, F1 is HGB .427830, SDC .420352, mean selected probability
.422184, selected-set entropy .418298, deterministic random .435666. These
points do not establish that random selection is universally superior. Rare
occurrence retention is secondary: HGB .7805 versus SDC .8171, with only the
modeled known native occurrences in scope. No conservation benefit is established.

The primary known-label F1 has no empty/empty scoring sensitivity here. The
learned head predicts mean quality .1387, versus observed known-label mean .4248;
the source training-mean reference is .2506. A post-primary MSE identity assigns
much of HGB's error to mean bias. That is standard algebra and does not isolate
ecological shift from measurement, taxonomic or survey effects.

## Taxonomic audit and correction

The frozen conservative pipeline treats unresolved, qualified and unmapped
taxa as possible unknown model classes rather than absences. Under that
contract, median known-label fraction is .6532, and 98/163 cells fail the >=80%
rule. Higher-rank records and missing taxon IDs can obscure broad sets of native
classes. Named unmatched species and infraspecific ranks also contribute under
this intentionally strict mapping. Reason masks overlap; their counts are not
an additive or causal attribution of the exclusions.

**The first pass omitted a recommended data join.** NEON's
[official quick-start guide](https://data.neonscience.org/api/v0/documents/quick-start-guides/NEON.QSG.DP1.10058.001v2?fallback=html&inline=true)
specifies linking observations to `div_morphospecies` by `siteID` and
`morphospeciesID`. A separately labeled post-hoc replay audits that join on the
same plots, with latest dated resolution and no favorable choice among conflicts.
There are 389 selected morphospecies keys: 349 have dated resolutions, 40 do not.
The join changes 411 observation records; eight conflicting resolved-species
records remain unknown. No classifier, calibrator or reliability head is refit.

| Quantity | Post-hoc documented join |
|---|---:|
| Scorable cells / sites | 77 / 11 |
| Median known-label fraction | .7680 |
| Training-mean MSE | .044036 |
| HGB MSE | .093484 |
| HGB-minus-SDC selective F1 | +.001800 |
| 97.5% descriptive site interval | [-.019636, +.022638] |

The original primary remains unchanged, and even the corrected replay misses
both sample-size requirements. These results describe the bridge and observation
contract, **not intrinsic unusability of NEON**. Further parent-rank/name
harmonization and identification-history checks must precede a new field study.

## Post-primary identification bounds

An additional diagnostic varies unknown binary labels while keeping the class
universe, known observations and predicted set fixed. Its elementary sharp
bounds assume independent unknown labels; extra botanical constraints could
tighten them. They are **identification ranges, not confidence intervals or new
theory**, and do not repair failed admission.

On the original 65-cell subset, mean possible full-modeled-set F1 width is
.2351 despite >=80% class-wise observability. The HGB-minus-SDC selective-F1
contrast can range from **-.06250 to +.06290** under those completions. Its
sign is therefore not identified for full modeled sets. Mean-minus-HGB MSE gain
ranges **[-.06888, -.01246]**: HGB remains worse than the mean control on that
fixed subset under these completions. This is a conditional post-hoc robustness
finding, not prospective confirmation of a broad transport law.

## Input engineering and remaining gaps

All 209 candidate native masks are now available through verified TDWG region
identity. All 53,336 released evaluation masks are constant within their 143
represented regions; 174 same-region training-mask probes extend the missing
candidate regions and cross-check the overlapping masks. This reuses published
range artifacts and does not validate new botanical ranges independently.

Exact released covariates cover 163 cells / 301 plots / all 16 sites, including
all nine Human Footprint columns. This avoids substituting nearby input values.
The public Human Footprint archive download did not return a verifiable usable
file; direct extraction for the other 46 cells remains pending. The ResNet stays
excluded after its failed reference; independent functional computation exactly
matches our adapter, leaving the checkpoint/prediction discrepancy unresolved.

The first GPU upload was automatically rejected as possible nonpublic NEON
egress. Before retrying, every payload array was verified against existing public
SAGE inputs/constants; no NEON observation, coordinate, ID, API response or
credential was included. The unchanged notebook/command was approved with that
new evidence. The review record is retained; no alternate upload route was used.

## Validation and paper implications

Twenty-nine focused tests passed before NEON target access; the final focused
suite, including missing-label bounds and morphospecies joins, has **34 tests**.
Independent confusion-matrix reconstruction, quota equality, source/input/policy
hashes, before-target ordering, exact target replay and GPU export hashes are
checked. E1/E2/E3, the initial NEON audit and all 665 prior bridge-manifest files
remain unchanged. No historical full-suite pass is claimed. All 163 E4 cells'
outcomes are now consumed; the replay does not make them fresh again.

This is useful engineering and falsification progress, **not a new successful
method, benchmark improvement or paper-ready novelty claim**. The next bounded
study should test how taxonomic censoring changes score estimates, confidence
comparisons and geographic inclusion using known labels and saved predictions.
See [the planned resolution study](taxonomic_resolution_study_plan.md). Taxonomic
harmonization, missing labels and quality/coverage tradeoffs already have prior
work; the contribution must be more specific than rediscovering those issues.

Artifacts: `results/confirmation_e4/`, `results/neon_input_followup/`; large
inputs/labels in ignored `data/confirmation_e4/` and `data/neon_input_followup/`,
GPU exports in ignored `artifacts/neon_e4_gpu/`. The original primary, post-hoc
diagnostics and morphospecies replay remain separate. Changes local/uncommitted.
