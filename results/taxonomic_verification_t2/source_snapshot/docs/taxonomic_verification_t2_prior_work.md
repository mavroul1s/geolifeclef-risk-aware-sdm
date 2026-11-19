# T2 positioning: active comparison is established

Reviewed 19 November 2026, before the T2 replay. This targeted review narrows
the contribution; it is not an exhaustive novelty proof.

[Nguyen, Ramanan and Fowlkes (ICML 2018)](https://proceedings.mlr.press/v80/nguyen18d.html)
study active vetting of imperfect test annotations, including multilabel and
instance-segmentation evaluation. Label-efficient evaluation and failures of
naive comparative estimates predate this project.

[Kossen et al. (ICML 2021)](https://proceedings.mlr.press/v139/kossen21a.html)
study acquisition tailored to model evaluation and importance-weighted loss
estimation. Acquiring evaluation labels rather than training labels is already
an established distinction.

[Hara et al. (Machine Learning 2024)](https://link.springer.com/article/10.1007/s10994-024-06603-1)
explicitly select queries to reduce variance in differences of model test losses.
Identifying which model performs better with fewer labels, including paired
loss differences and adaptive variance reduction, cannot be our broad novelty.

T2 instead evaluates a specific synthetic observation/cost contract: fixed
quota-constrained ecological selection, shared taxon-concept identities,
outside-target recorded species and conservative F1 completion ranges. The same
censored records are priced once as shared block concepts and once as local
records. Static width-credit and incidence controls are transparent heuristics;
none is claimed as a faithful VMA/LURE reproduction or superior to published
active-testing methods. This study measures finite-cohort outer-range contraction,
not unbiased risk estimation or a confidence sequence.

Shared-identity propagation is an assumed information advantage, not a new
discovery by itself. An eventual method paper needs fuller group-query/active-
comparison positioning and matched established baselines. T2 can supply bounded
empirical measurement evidence, even if its candidate fails its practical target.
