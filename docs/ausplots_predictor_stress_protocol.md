# SAGE Human Footprint input stress test

This is an engineering development test on already used SAGE AEKOS geography.
It isolates the change in prediction sets if the nine unavailable Human
Footprint features are passed as missing and filled by the release's training
means. Use all valid AEKOS evaluation cells listed in released plot metadata,
their released environmental vectors and their published native-range masks.
Compare `aggregation_s0` and `baseline_deepsdm_s0`, both checked against their
released probabilities. Use each model's published normalization and weights.
Pass the first reference with its released inputs and set the nine Human
Footprint columns to missing for the second pass. Natural missingness elsewhere
is the same in both passes.

Summarize the fraction of cells whose native top-10 set changes, median and
10th-percentile set Jaccard overlap, and mean and maximum absolute probability
change over the native classes. Use species column order to break probability
ties, and cap set size at the number of eligible species. Report top-5 and
top-20 overlap as sensitivities. This test uses no target labels or AusPlots
predictions. It measures the effect of imputing the whole group. It does not
measure F1, ecological accuracy, or the effect of removing one layer.
