# GeoLifeCLEF Risk-Aware SDM

MSc thesis research on plant-community prediction: when species-set predictions
are reliable, which surveys a model should accept, and whether that selection
preserves geographic and rare-species service. The project combines a
GeoLifeCLEF 2025 competition pipeline with separate reliability studies using
released SAGE models and NEON data.

**Current status — 21 November 2026:** the competition models and controlled
reliability studies are implemented. Several proposed improvements failed their
declared targets. The current paper direction is a reproducible empirical study
of decision confidence, ecological service and taxonomic observability.
A successful new method and publication-level novelty remain unestablished.

## What we have achieved

| Work | Completed evidence | Interpretation |
|---|---|---|
| GeoLifeCLEF competition | Best official v32: **0.24225 public / 0.21514 private**; recorded winning private target **0.23021** | Real gains and failed successors are preserved. Remaining private-score gap .01507; no benchmark victory claimed |
| R1 and decision alignment | Six neural fits; confidence/metadata, fixed-decoder and regional diagnostics | The geographic safeguard fails; stronger decision-aligned controls narrow learned gains |
| Locked E1 | 5,251 assessment cells, 73 blocks; rare-service gain +0.536 percentage points versus +3-point target | The proposed ecological constraint fails its primary target |
| E2 / E3 | Eight ranking procedures reject corrections; 654-cell larger-plot confirmation fails joint superiority | Magnitude fit and selective utility differ; comparator ordering changes across domains |
| NEON E4 | Verified MLP CPU/T4 inference; 163 evaluated cells. Original contract admits 65 cells/10 sites; corrected morphospecies replay 77/11, below 100/12 | Scientific admission fails. Corrections do not turn consumed outcomes into fresh confirmation |
| Taxonomic study T1 | **60 mask settings**, fixed predictions, 5,251 consumed cells, zero new backbone fits | About 98.3–98.4% known class labels coexist with ~26% hidden positive occurrences at nominal 20% censoring. Statistical criterion fails; no policy ordering reverses |
| Taxonomy audit A1 | **93,698 dictionary entries / 94 verified pages**, 44,721 cell–genus metadata screens | Accepted-name links, parent candidates and unresolved concepts are distinguished. Field-level genus guarantees remain unestablished |
| A2 selected-record audit | Six linked history records; eight component controls; unchanged predictions/positive labels | Conservative uncertainty masks reduce eligible cells 77→31 while known-label F1 rises +.026839 on the same 77 cells; this is an observation-contract effect |
| A3 / A3u exact synonyms and uncertainty guard | 553 dictionary links; **123 additional modeled cell–species presences**; alias uncertainty propagated conservatively | Original A3 admits 34 cells; guarded A3u admits **33 cells / six sites**, still below the original field rule; no model is retrained |
| Verification-cost study T2 | **100 CPU query trajectories**, 700 checkpoints, identical censored records under shared/local costs | Frozen 25% criterion fails through local-control saturation. Predeclared secondary ordering costs fall 27.7% / 39.5% versus the best tested controls, conditional on the synthetic contract |
| Linkage/cost robustness T3 | **360 CPU trajectories**, 2,520 checkpoints, 1,440 price evaluations; invariant recorded inventories | Unchecked links invalidate ranges under synthetic errors. Ideal audits preserve validity; sharing savings depend on audit price/scope. Reference criterion fails: 9.34% / 7.71% versus strongest matched shared control, below 10% |

T1 separately diagnoses 268 nonempty recorded inventories receiving apparent
perfect scores after censoring under empty/empty F1 = one. Retaining the
simulator's positive-genus information narrows completion ranges by 64–66% at
the moderate dose. These are **post-hoc, synthetic findings**; the constraints
are not automatically valid for real NEON observations.

## What we are doing now

We have completed a bounded **current-record, history and exact-synonym audit**
on the consumed NEON cohort. A2 reproduces the previous labels exactly, then
checks stricter current-record/uncertainty rules. A3 uses explicit accepted-name
equivalence, without guessed parents or archived replacement labels. The A3u
correctness guard propagates uncertainty across synonym genera and preserves
all recovered presences. The field contract remains incomplete, and the original E4 failure remains unchanged.

The [T2 verification-cost study](docs/taxonomic_verification_t2_results.md) is
now completed. It compares five fixed acquisition orders under two cost units,
including outside-target records and overlapping unknown identities. Its frozen
primary criterion fails because support-focused local controls already exhaust
relevant records at 25% budget. The predeclared secondary endpoint shows fewer
queries to identify a practical finite-cohort ordering. This is conditional
measurement evidence; it does not establish a new general algorithm or field
annotation-time savings.

The [T3 linkage/cost study](docs/linkage_robustness_t3_results.md) is completed.
Unchecked propagation gives invalid contrast ranges in every nonzero-error
trajectory at some checkpoint, while no wrong ordering is observed. Perfect
charged audits preserve validity. Comparison-focused sharing can be cheaper
than local verification under low audit prices, but loses that advantage at
higher prices. The candidate's incremental acquisition gain misses the frozen
10% reference target. These are controlled sensitivity findings, not field
error-rate or annotation-time estimates.

The next steps are:

1. Complete the stronger-baseline package with matched adaptive/cost-aware
   acquisition and targeted ablations. Reassess the method contribution before
   expanding; preserve all failed T2/T3 criteria and secondary labels.
2. Replicate on additional verified prediction settings and another survey
   source, reusing valid outputs. No new backbone-training sweep is allocated.
3. Seek separately uninspected external outcomes and a record/cost audit, with
   source/training overlap, completeness and sample adequacy checked before
   confirmation. Availability remains unresolved; consumed data stay consumed.
4. Write the manuscript around observation contracts, valid comparisons and
   verification costs. See the [overall thesis/paper status](docs/thesis_paper_status.md)
   for completed findings, limitations and the remaining three study packages.

## Research reports and evidence

- [Claim/evidence matrix](docs/paper_claim_evidence_matrix.md),
  [methods/results draft](docs/paper_methods_results_draft.md) and
  [working outline](docs/paper_outline_working.md).
- [T1 results and figure](docs/taxonomic_resolution_t1_results.md),
  [frozen protocol](docs/taxonomic_resolution_t1_protocol.md) and
  [post-hoc diagnosis](docs/taxonomic_resolution_t1_posthoc.md).
- [T3 linkage/cost results and figure](docs/linkage_robustness_t3_results.md),
  [frozen protocol](docs/linkage_robustness_t3_protocol.md) and
  [overall paper-readiness assessment](docs/thesis_paper_status.md).
- [T2 verification-cost results and figure](docs/taxonomic_verification_t2_results.md),
  [frozen protocol](docs/taxonomic_verification_t2_protocol.md) and
  [active-comparison positioning](docs/taxonomic_verification_t2_prior_work.md).
- [A1 taxonomy audit](docs/neon_taxonomy_a1.md) and
  [evaluation prior work](docs/taxonomic_evaluation_prior_work.md).
- [A2/A3 results and figure](docs/neon_taxonomy_a2_a3_results.md),
  [A2 protocol](docs/neon_taxonomy_a2_protocol.md),
  [A3 protocol](docs/neon_taxonomy_a3_protocol.md),
  [A3u uncertainty guard](docs/neon_taxonomy_a3u_protocol.md) and
  [T2 plan and status](docs/taxonomic_verification_t2_plan.md).
- [E1](docs/external_reliability_e1_results.md),
  [E2](docs/ranking_study_e2_results.md),
  [E3](docs/confirmation_e3_results.md),
  [E4](docs/confirmation_e4_results.md), and
  [MLP/reference engineering](docs/neon_bridge_status.md).
- [Next steps and compute budget](docs/next_experiment_budget.md),
  [results index](results/README.md), and
  [competition-score registry](results/experiment_registry.json).

Protocols, source snapshots and hash manifests preserve each study's actual
analysis role. Identification ranges are not statistical confidence intervals.
Recorded benchmark labels are not perfect occupancy truth; original backbone
training support limits independence claims.

## Installation and tests

The current snapshot passes **all 417 local tests**, including artifact-backed
checks after restoring the documented inputs, plus independent A2/A3/A3u/T2/T3
validation. T3 independently checks 108 dense checkpoints, 240 primary stopping
prefixes, 720 charged-operation endpoints and 12 acquisition orders. T2 independently reconstructs 60 dense checkpoints, 20 acquisition
orders and 400 stopping-boundary prefixes; see the [T2 report](docs/taxonomic_verification_t2_results.md).
Earlier checks are preserved in the [A2/A3 release report](docs/taxonomy_audits_validation_2026_11_19.md)
and [reproduction/portability report](docs/reproduction_validation_2026_11_17.md).

Use **Python 3.12** for research. The original competition package supports
Python 3.10–3.12; research plotting dependencies require >=3.11.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,research]"
pytest -q
```

The CPU numerical environment is recorded in
[`configs/external_reliability_e1_environment.txt`](configs/external_reliability_e1_environment.txt).
See individual protocols for input versions, weights, calibration roles, GPU
reference tolerances and targeted validation. Tests include synthetic cases and
artifact-backed competition checks. The full suite additionally needs the
documented competition metadata/species cache; the legacy v23 control fixture
can be restored without a download using
`python scripts/restore_legacy_test_control.py`. Add `--competition-root data/raw`
to mirror downloaded, extracted official metadata and derive the PA species
vocabulary in the legacy fixture paths. The committed consumed-ID union
has explicit provenance and replaces unavailable per-version CSVs only where
their union is needed. Full study reproduction requires its documented inputs.

## Data access and reproduction

Keep credentials outside Git. Competition downloads require Kaggle access and
acceptance of the [GeoLifeCLEF 2025 rules](https://www.kaggle.com/competitions/geolifeclef-2025/rules).
NEON uses its own account/API token where required by the download service.
Neither key belongs in notebooks, reports or committed files.

```bash
python scripts/download_data.py --list-only
python scripts/download_data.py --dry-run
python scripts/download_data.py
```

Research entry points include `scripts/run_external_e1.py`,
`scripts/study_ranking_e2.py`, `scripts/run_confirmation_e3.py`,
`scripts/run_neon_e4.py`, `scripts/run_taxonomic_resolution_t1.py` and
`scripts/audit_neon_taxonomy_contract.py`, `scripts/run_neon_taxonomy_a2.py`,
`scripts/run_neon_taxonomy_a3.py`, `scripts/run_neon_taxonomy_a3u.py` and
`scripts/run_taxonomic_verification_t2.py` and `scripts/run_linkage_robustness_t3.py`. Read each
protocol and CLI help first.
Completed frozen phases refuse overwrites. Restoring inputs requires the original
sources; a hash manifest does not include ignored local data.

Original sources include [SAGE](https://arxiv.org/abs/2609.31082),
[NEON plant presence and percent cover](https://data.neonscience.org/data-products/DP1.10058.001/RELEASE-2026),
and GeoLifeCLEF. Follow source attribution and usage terms. Large data, model
checkpoints and working caches remain ignored; compact scientific results and
reproducibility snapshots are public.

## Competition pipeline

The multimodal pipeline supports Landsat, bioclimatic and Sentinel inputs,
presence-only training, spatial validation and calibration-only set selection.
V32 remains the best scored model. V33 was unchanged at submission; v34 regressed.
The v35 full-set attention notebook has local validation but **no recorded new
official score**. Current work is the separate thesis reliability/audit program.

```bash
python scripts/audit_data.py --data-root data/raw
python scripts/train.py --config configs/frequency.yaml
python scripts/train.py --config configs/landsat_tcn.yaml
python scripts/evaluate.py --checkpoint artifacts/landsat_tcn/best.pt --split data/processed/val.npz
```

Research docs describe scientific progress, limitations and plans. Machine-local
agent instructions, chat handoffs and assistant worklogs stay local and are
excluded from the public repository. The active documentation is this README,
the evidence matrix, next-step budget and linked results.

## Layout

`src/geolifeclef/`: reusable methods; `scripts/`: entry points; `tests/`: checks;
`configs/`: study settings; `docs/`: research protocols/reports; `results/`:
compact evidence; `notebooks/`: competition and private-inference workflows.
