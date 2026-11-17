# Proposed next study: taxonomic resolution and selective evaluation

**Planned, not executed.** One controlled development study, zero new backbone
fits. E4 is completed and fails admission; it must not be renamed a success or
retuned into another fresh confirmation. This plan is not proof of novelty.

## Why this is the next useful experiment

E4's conservative bridge retains 65/163 cells at ten sites; its post-hoc
documented morphospecies join raises this to 77 cells at 11 sites, still below
100/12. Known-label fraction does not determine how tightly species-set F1 is
identified. Simple bounds are wide even among the originally admitted cells.
The observation process, harmonization and score definition therefore need a
controlled test before further model-complexity experiments can be interpreted.

First finish the label contract: apply the documented morphospecies join,
audit identification-history semantics, separate identifiable outside-universe
species from genuinely unresolved concepts, and validate explicit species-parent
relationships for identified subspecies/varieties. Use metadata/authorities and
logged rules, not favorable F1. Preserve unresolved/protected taxa. Any replay
on E4 is development because these outcomes are consumed.

## One bounded controlled development study

Use historically consumed SAGE assessment rows with known modeled-species
labels and fixed predictions. The probability/risk fitting roles remain
separate. Explicitly label adaptive data reuse; the purpose is controlled
measurement stress testing, not new hidden-test generalization. Cache inference
on CPU if needed; the MLP's CPU/T4 path is already verified. No architecture,
seed-training or decoder sweep is required.

Freeze two censoring mechanisms: uniformly chosen positive genera, and
rarity-weighted positive genera. At three doses (0, 20%, 50% of positive genera),
simulate higher-rank identification by hiding the corresponding eligible labels.
Use ten fixed mask seeds: **60 lightweight label-mask replays, not 60 model
fits**. Preserve the complete labels as the oracle for evaluation only. Freeze
which source rows, censoring rules, group budgets and contrasts are used before
running this development study.

Compare three evaluation treatments on identical frozen sets/selectors:

1. Unknowns incorrectly treated as absences, as a diagnostic control.
2. Known-label-only F1, with explicit observable-label and geographic attrition.
3. Identification bounds on full modeled-set utility, retaining all cells.

Measure score bias against the complete-label oracle, distortion of paired
HGB/SDC/random selection contrasts, sign identifiability, interval width and
site/rare-occurrence representation. Keep predictor/selection scores independent
of artificial target masks. Bootstrap the same spatial units and report fixed
mask variability separately from fitted-model variability. Exhaustive small
binary-completion tests already validate the elementary bound implementation;
oracle coverage in larger replay is a correctness check, not a novel theorem.

Predefine a practical distortion threshold (candidate: .01 absolute F1 or a
material rate of contrast-sign reversals at moderate censoring) before execution.
Only if a reproducible operational effect remains after proper taxonomy and
strong controls should a more ambitious method be developed. One possible later
method is budgeted taxonomic verification to tighten policy-comparison bounds;
it would need active-evaluation prior-work review, a clear cost/query model and
calibration-only promotion. It is not implemented or established here.

A later field confirmation requires a corrected, frozen label/input contract
and separately uninspected outcomes. Existing E4 cells cannot supply that role.
Disclose original backbone training support in any subsequent NEON cohort;
do not promise an unseen-training-location claim from temporal/site partitions.

## Prior work and novelty boundary

- [Taxonstand](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210X.2012.00232.x)
  already addresses plant-name standardization and heterogeneous taxonomic
  concepts. A name-harmonization script is not itself a new contribution.
- [OCCUR](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14271)
  explicitly discusses curation and the certainty/coverage tradeoff. Losing
  records under stricter filters is established, not a discovery here.
- [Scalable Generative Models for Multi-label Learning with Missing Labels](https://proceedings.mlr.press/v70/jain17a.html)
  models label observability; missing-label learning is established.
- [Informative labels / missing-data mechanisms](https://proceedings.mlr.press/v202/sportisse23a.html)
  studies nonrandom label missingness. Simulating nonrandom censoring does not
  create a new general missing-data theory.

The narrower contribution candidate is a reproducible audit of **paired
quota-constrained ecological service under structured taxonomic uncertainty**,
with calibrated magnitude, ranking utility, observed-label scores and latent-set
bounds kept distinct. The elementary bounds and standard scalar calibration are
not claimed as new. Neighboring partial-label evaluation and active verification
literature still need a fuller review before a novelty claim or venue decision.
