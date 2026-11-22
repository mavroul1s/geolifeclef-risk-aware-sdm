# T5 protocol — observation and cost replication

T5 is a development replay on previously consumed SAGE E1 assessment cells.
It has two experiments and uses no new training, calibration fits, inference or
GPU jobs. It asks whether the original measurement and cost findings persist
across fixed predictions and across survey contributors. The independent NEON
confirmation gate and all prior T1–T4 outcomes remain historical.

Use the original assessment rows, target/native universe, taxonomy, prevalence
weights and frozen MLP HGB/SDC confidence scores. Verify released DeepSDM and
RF seed-zero probability files against E1 receipts and reproduce E1 ranks and
counts under their original native masks, using the original E1 Platt parameters.
Then apply the same decoder on the fixed T1 narrower universe. The MLP uses T1
saved predictions. Hold the MLP acceptance policies fixed across all three
models, so the model comparison changes the predicted sets only. For source
analyses, use single-contributor cells with full metadata provenance and
reapply the original quota rule to the frozen MLP policy scores within each
source. Source differences include geography and survey design. Contributors
with at least 500 cells and eight blocks are AU-AU-002, NA-US-002 and AS-RU-002.
Cost replay also needs at least 40 policy-disagreement cells. No source is a
separately uninspected benchmark.

## A. Score sensitivity

Reuse T1 uniform and rarity-weighted positive-genus missingness, doses 0/.2/.5
and all ten mask seeds. Keep predictions fixed, and score projected known pairs
or treat hidden labels as absence. Apply empty/empty values of both 1 and 0 to
the full-inventory reference and to both treatments. For each model/contributor,
report labels known, positives hidden, mean score error, false perfect counts,
and HGB-minus-SDC contrast error. Also report changes of model and policy
ordering using the same .002 oracle-gap threshold as T1 for practical reversals.
Use descriptive paired spatial bootstrap intervals over 5-degree blocks for
the mask-averaged errors. Report both directions and all doses. Primary
measurement comparison is known-only bias at dose .2 with empty/empty=1.
The convention sensitivity and error-of-contrast provide checks on a single
headline F1 shift. Small contributor-block counts limit spatial inference.

## B. Cost and validity

Run the three prediction settings on all 5,251 cells and run the saved MLP on
each of the three contributor cohorts. Use the first four T4 mask seeds,
synthetic 1%/5% membership corruption and audit prices .1/.5. No post-outcome
dose replacement. Reuse the T2 same-initial-information mask generator and T3
nested within-block/genus corruption, including the outside-target category.
Construct the exact common scope for the HGB–SDC disagreement. Log actual
realized corruption, even where there are insufficient valid swaps. Results
from a shortfall scenario are labeled by nominal and achieved rate. Verify that
the hidden record multiset is invariant and that initial known pairs agree
across corruption and query units.

Compare all six frozen T4 priority controls with identical information and
charges and two singleton local controls (static/adaptive scope width). At a
price-independent order, reprice the identical path. The primary endpoint is
the first atomic query prefix with contrast range width <=.002. Use first
practical sign beyond .001 as a secondary endpoint and classify no practical
sign when the available scope is exhausted. The width endpoint applies even
when the reference contrast lies within the practical margin. Compare the
named adaptive width vs static width and the learned vs fixed cost forecast,
plus charged audited shared vs the lower of the two local costs per mask.
Compute fixed-prefix break-even price against that local reference for static
scope width. Show all arms and no new pass threshold. Report coverage of
disagreement cells and recovered rare labels at the primary endpoint.
Additionally replay unchecked propagation with the same static scoped order
through the full scope and inspect range containment at 0/25/50/75/100% of
groups. Record inappropriate signs and compare with guarded prefix validity.
Simulated audits assume perfect target-category classification. Identification
ranges concern recorded inventories, with costs in normalized synthetic units.

The maximum budget is 24 block–mask cases, 48 priced-error cases, 480
executed shared trajectories plus 96 price-only reuses, 48 local
trajectories and 48 unchecked paths, within one hour on CPU. Hash inputs and
sources before reading target values for this replay. Independently reconstruct
charged operations at endpoints, validate dense stopping prefixes on the first
seed, and record results under new paths alongside a manifest. Preserve all
sealed earlier studies.
