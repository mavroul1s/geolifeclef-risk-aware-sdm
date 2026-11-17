# T1 post-hoc scoring and genus-information diagnosis

Added after reading the completed T1 result on 17 November 2026. The frozen
primary and its failed development criterion remain unchanged. This diagnostic
introduces no fitted models, new censoring settings or independent confirmation.

The primary output has 1,384 empty prediction sets, including 861 cells with
positive recorded truth. At every nonzero censoring setting, 268 of those cells
lose all observed positives and acquire an apparent empty/empty F1 of one.
This adds exactly 268/5,251 to the cohort-average score bias, while nonempty-set
cells lose F1. The nominal 20% genus dose rounds up; all single-positive-genus
cells are fully censored. This is a consequence of the declared mechanism and
score convention, not a measured NEON identification rate.

Report two explanatory sensitivities without changing any primary endpoint:

1. Empty/empty F1 = zero for oracle and masked scores. This changes the estimand;
   it is not a statistically justified correction to the original primary. For
   empty predictions the score and both completion endpoints are zero.
2. Preserve the simulator's remaining genus-level information: each fully
   censored genus contains at least one positive **eligible modeled species**.
   Derive exact completion endpoints under those finite constraints and test
   them by exhaustive enumeration of small binary universes. Use the original
   empty/empty convention. These standard finite-completion calculations are
   not claimed as novel theory. The original independent-label bounds relax
   this information, so they are conservative for this synthetic mechanism.

For a fixed prediction size P, known positives K, known true positives A,
unknown predicted classes U_p and unknown unpredicted classes U_o: let m be the
number of censored positive genera with no unpredicted eligible class, and n
the number with no predicted eligible class. Then the constrained lower endpoint
is 2(A+m)/(P+K+U_o+m), and upper endpoint is
2(A+U_p)/(P+K+U_p+n), with empty/empty = one. The lower completion assigns every
unknown unpredicted class positive and only the required predicted positives;
the upper assigns every unknown predicted class positive and only the required
unpredicted positives. Within each category these choices are feasible.

Genus-positive metadata cannot be inferred from real missingness masks alone.
The simulator provides it by construction. A real higher-rank NEON record can
belong to an outside-universe species, so this constraint requires a separately
validated field contract and may be inapplicable. Do not retrofit it into E4.

Use the same ten seeds and spatial bootstrap, report score and paired-contrast
sensitivities, width reduction and sign identification, and preserve primary
hashes. This is a diagnosis of measurement semantics, not a second confirmation
or a promoted method. A new method still needs a concrete operational query/
cost model and prior-work comparison.
