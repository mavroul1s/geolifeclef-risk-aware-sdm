# T6 protocol — compare models under the same hidden labels

Exploratory follow-up motivated by T5 model score results. Use its three fixed
prediction sets, four cohorts and moderate 20% dose. Reuse both uniform and
rarity-weighted positive-genus mechanisms with all ten T5 censor seeds. Use
F1(empty,empty)=1 as in T5's primary. Count all model pairs without selecting
for an observed reversal. No new model training, prediction or data access.

For a fixed cell, measure the difference between two model predictions when
both score against the same possible completed inventory. First use the
ordinary outer range obtained by subtracting separate per-model score ranges.
Then compute the sharp difference range under independent binary completion of
unknown species within each cell. Restrict to the original native/eligible
universe. The completion treats no unobserved genus counts or identities as
given and imposes no cross-cell constraints. It is conditional on correct
known labels and the stated class space.

With k known positives and t additional positives, both denominators are fixed.
The incremental contribution of any unknown species to the difference depends
only on which predictions contain it. Group unknown species into four types.
For each t, the extremes allocate t positives to the largest or smallest
available coefficients. Evaluate all t and take the extrema. The mean of
cell-level extrema is attainable for the finite cohort under unrestricted
cross-cell completion. Treat zero denominators independently for each model.

Report relative width change, model-order identification beyond .001 F1,
and containment of the complete recorded inventory difference for both bounds.
Use each simulated mask as a paired replicate; report the full set of endpoints
and descriptive summaries. Include the T5 observed known-only contrast for
comparison. Estimate no population uncertainty from the ten mask repetitions.
This is an empirical comparison of ranges under a specified information model.
The optimization argument is elementary.

Validate the shared-completion optimizer against exhaustive binary completions
on tiny cases including both empty conventions. Verify that each separate
outer range contains the paired range and both contain the recorded contrast
on every replay. Hash inputs and sources before execution. Bounded CPU replay
of 240 cohort–mask–pair cases on previously used outcomes.
