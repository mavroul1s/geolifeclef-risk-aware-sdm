# T1: controlled taxonomic-resolution development study

Written before this study materializes assessment labels on 17 November 2026.
This is adaptive development using already consumed E1 outcomes, not a fresh
confirmation or external registration. The earlier E4 failure and its original
artifacts remain unchanged. The complete recorded benchmark labels are an
**evaluation oracle within a modeled native subset**, not error-free occupancy.

## Fixed roles, model and taxonomy

Use all 5,251 original E1 assessment cells, original country × area-bin quotas
at 80% retention, and the same 73 five-degree spatial blocks. Reuse E4's 2,453
unambiguous dictionary-matched classes intersected with original native masks.
The released aggregation_s0 MLP and expected-F1 decoder (0–80 species) are fixed.
Reuse E4's probability transform and mean/affine parameters. Reconstruct its HGB
once on the original 1,061 risk-calibration rows and require numerical reproduction
of the already frozen NEON predictions before opening T1 assessment labels.
This is one small CPU head reconstruction, zero backbone fits or new GPU jobs.
CPU inference must reproduce the published 128-row MLP reference at max 1e-5 /
mean 1e-6 error. Keep prediction sets and all selectors fixed across masks.

The genus column in SAGE's species metadata must agree with every canonical
species-name first token and the model's class index. Read only species identity,
genus and **training-only** relative prevalence; exclude test-prevalence columns.
Audit missing/ambiguous genus identities before execution. This controlled study
uses canonical benchmark genera; it neither requires nor solves the pending
NEON parent-rank, name and identification-history semantics. Those are still
required before another field-study contract. E4 cannot become fresh again.

## Sixty synthetic label-mask replays

Two mechanisms × doses 0/20/50% × ten fixed seeds, specified in the JSON config.
Per cell, select `ceil(dose * number of distinct positive eligible genera)`;
zero dose and zero-positive cells select none. Report the realized fraction,
including rounding. Sample without replacement by ordering exponential-race
keys `-log(U)/weight`; uniforms come from a cell-ID/seed-specific generator.
Uniform weights equal one. The second arm weights each genus by the inverse
square root of its median training relative prevalence among universe classes
(floor 1e-6). Same random uniforms couple the two arms and doses are nested.

Hide **all** eligible labels in each selected genus, including positive and
negative labels. Do not restore hidden positives using the oracle. This models
full species-identity censoring within a genus, not a faithful empirical model
of NEON identification. The mechanism is deliberately informative: positive
genera determine missingness, while neither predictions nor selectors receive
the mask or oracle. It is a stress test, not a missing-at-random assumption.

The 20 zero-dose arm/seed settings are intentional identity controls. Thus 60
settings contain 40 nonzero stochastic settings and one unique baseline; they
are not 60 independent datasets, fits or replicates of ecological sampling.

## Outcomes and fixed analysis

On identical prediction sets, compare unknown-as-absence F1, known-label-only F1
(remove unknown predictions only from its scoring denominator), and elementary
independent-binary-completion bounds on full-set F1. Empty/empty F1 is one.
All cells remain in the main evaluation. Keep HGB, SDC, deterministic random,
mean-set-confidence and predicted-set entropy selectors fixed; random uses the
hash `T1:<cell_id>`. Affine SDC has the same ordering as SDC.

Report full-coverage and selected-score bias against the complete recorded-label
oracle; HGB-minus-SDC, HGB-minus-random and SDC-minus-random contrast distortion;
completion-range width, oracle containment and sign identifiability. Use paired
contrast weights so shared accepted cells cancel. Count a sign reversal only
when the fixed oracle contrast has absolute magnitude >=.002. Ranges are
identification ranges, not confidence intervals or new theorems. Independent
binary completions deliberately relax any surviving genus-level positive-count
constraint; describe their conservatism.

Separately audit the >=80% class-known rule used by E4: retained cells/countries/
blocks, country retention rates, complete-oracle rare-occurrence retention and
the oracle-score shift in admitted cells. Do not reallocate quotas after this
outcome-dependent admission diagnostic. Also report positive-occurrence hiding,
which need not track the fraction of all class labels hidden.

For each arm/dose, average per-cell scores across the ten mask seeds, then use
4,000 paired draws of the original spatial blocks for 97.5% intervals conditional
on fixed model/policies and this benchmark cohort. Resample the same blocks for
the observed and oracle contrasts. Report the separate ten-seed min/max; do not
treat cell × mask entries or mask seeds as independent ecological samples.
No outcome-based model, genus-weight, dose or selector tuning is allowed.

The declared development threshold is **.01 absolute F1 bias at 20% censoring**:
at least one scoring treatment must have a seed-mean bias >=.01 in magnitude
with its spatial interval wholly outside [-.01,.01] under both mechanisms.
Separately report whether paired-selector distortion reaches .01 and whether
practically nonzero oracle contrasts reverse; a score-bias result alone does
not establish a substantial ranking effect. This threshold controls whether
the stress test is operationally worth further work, not scientific novelty.

## Interpretation and integrity

Missing-label learning and taxonomic curation already have substantial prior
work; see the preceding frozen study plan. A generic missing-label effect is
not a new contribution. The candidate is an application-specific, reproducible
audit connecting observability, paired constrained utility and representation.
Any verification-allocation method needs a separate query model, prior-work
review and calibration-only selection. Any later confirmation needs genuinely
uninspected outcomes and an audited field label contract.

Freeze this config, sources, input identities and historical manifests before
assessment access. Save the fixed policy before access, record chronology, use
synthetic correctness tests, independently recompute confusion scores/quotas/
oracle containment, and verify historical manifests. Preserve all negative and
positive findings. No competition submission, public upload or new credentials.
