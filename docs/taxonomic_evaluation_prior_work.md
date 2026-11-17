# Positioning taxonomic evaluation and verification — 17 November 2026

Targeted primary-source review, not an exhaustive novelty certification. This
extends the taxonomic-curation/missing-label references in the frozen T1 plan.
These sources were checked against publisher/proceedings pages on 17 November.

| Existing work | Established scope | Consequence for this thesis |
|---|---|---|
| [Kossen et al., ICML 2021, Active Testing](https://proceedings.mlr.press/v139/kossen21a.html) | Select evaluation points to reduce labeling cost; address acquisition bias and estimator variance | Acquiring labels specifically for evaluation is established. A future query policy must compare against active-testing controls, not just uniform random |
| [Maia Polo et al., NeurIPS 2024, Weak Supervision Performance Evaluation via Partial Identification](https://papers.neurips.cc/paper_files/paper/2024/hash/f4c6bec746b0aeca8c2cd15096f1ad1f-Abstract-Conference.html) | Partial identification and performance bounds including F1 under programmatic weak supervision, using marginal constraints and convex optimization | Bounding evaluation metrics when truth is unavailable is established. Our elementary fixed-set completion bounds are not a new general framework; the observation and constraint models differ |
| [Miller et al., PMLR 2018, Classifier Performance Estimation with Unbalanced, Partially Labeled Data](https://proceedings.mlr.press/v88/miller18a.html) | Binary performance estimation when investigation is restricted by classifier score; compares labeling schemes under class imbalance | Score-dependent observation and label-efficient evaluation are established. Our synthetic positive-genus mechanism cannot be sold as the first informative-label evaluation problem |
| [Jain et al., ICML 2017](https://proceedings.mlr.press/v70/jain17a.html) and [Sportisse et al., ICML 2023](https://proceedings.mlr.press/v202/sportisse23a.html) | Missing-label learning and informative missingness | Generic missing-label bias or an informative censoring simulator is insufficient novelty |
| [Taxonstand, 2012](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210X.2012.00232.x) and [OCCUR, 2024](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14271) | Taxonomic standardization and biodiversity-record curation/coverage tradeoffs | Botanical name resolution and stricter-filter attrition are supporting engineering, not automatically a method contribution |

The narrower candidate is a reproducible empirical evaluation of **fixed,
quota-constrained plant-community decisions with genus-structured uncertainty**:
contrast measurement bias, constrained service and sign identification on the
same fitted predictions. Its value must come from demonstrated operational
effects and a defensible ecological observation contract, not renaming existing
partial-identification or active-evaluation theory.

## Requirements before a verification-method experiment

1. Specify the query unit and what information it reveals: one specimen,
   genus-level record, complete cell inventory, or all class labels in a genus
   are different costs and scientific assumptions. Synthetic free revelation
   of every absence in a genus must not be represented as a field procedure.
2. State the target: estimate a single policy's quality, identify a paired policy
   ordering, or certify a practical difference. Shared accepted cells cancel
   from a fixed paired contrast; exploiting this algebra alone is not a new
   algorithm. Preserve any geographic or rare-occurrence constraints explicitly.
3. Compare uniform, uncertainty/bound-width and established active-testing
   approaches at matched query cost. Select parameters on calibration/development
   roles; evaluate label cost, remaining identification width, sign errors and
   representation. Include empirical genus constraints when justified.
4. Separate a synthetic mechanism from observed identification probabilities.
   T1 knows which genera have positives to simulate missingness; a deployed
   selector cannot receive the hidden species identities or oracle performance.
5. Finish the NEON species-parent/name/history audit before deriving a field
   observation contract. All 163 E4 outcomes and all E1/E3 outcomes are consumed;
   a new random partition cannot create independent confirmation.

No claim is made here that this exact combination is absent from the literature.
The present review rules out broad novelty claims. A fuller comparison with
taxonomic uncertainty in ecological validation, partial multilabel evaluation,
and paired active model comparison remains required before selecting a venue.
