# v23: diverse single-head PO initialization and calibrated OOD cardinality

Preregistered before any v23 training or assessment. The official baseline is v22: **0.21684 public / 0.19491 private**, source commit `519ae6cecadb30e4339d6fcf5354c4f0ed19f479`, kernel version 22, submission ref `56233226`. The target is 0.2302 private; this experiment does not guarantee reaching it and internal F1 is not a hidden-test score.

## Evidence used before freezing this protocol

The v22 PO pretraining loss was still falling at epoch 16, but PA checkpoint-selection F1 commonly peaked well before the maximum schedule while training loss continued falling. More epochs alone are therefore not the intervention. v23 uses a lower PA learning rate, a fixed longer schedule, and evaluates only predetermined checkpoints. The v22 assessment's single-head point estimate exceeded retained-PO by 0.0001348, but its interval crossed zero. That consumed result motivates a new preregistered single-head primary; it is not reused as untouched evidence.

Only the two necessary v22 deployment probability matrices were retrieved from exact Kaggle version 22: retained-PO calibration and test float16 probabilities. The official CSV, policies, report and provenance complete the private v22 input. No v22 checkpoint or fold probability matrix is packaged.

## Fixed split

The v22 assessment consumed SHA buckets 0--39 of the 25,511 former-v21-training surveys. v23 uses the untouched remainder without changing the old salt or trying alternative splits:

- fold 0 assessment: buckets 40--69, 5,183 surveys;
- fold 1 assessment: buckets 70--99, 5,849 surveys;
- checkpoint selection: consumed v22 fold 0, 7,435 surveys, development only;
- calibration/policy selection: consumed v22 fold 1, 7,044 surveys, development only.

Every fold refits the matching v22 recipe. It excludes whole evaluation blocks from fitting and removes fitting rows within 20 km of checkpoint selection, calibration, or its own assessment. The two v23 assessment folds are disjoint and contain 11,032 surveys. As ordinary cross-fitting, the other outer fold may enter training. Checkpoint selection, policy calibration and assessment remain separate. The v20, v21 and v22 assessments are never presented as new v23 assessment evidence.

## Fixed candidate family

The primary is the mean of three single-head, PO-initialized residual MLPs with seeds `20250923`, `3408` and `9173`, width 512 and three residual blocks. Each seed gets 24 PO pretraining epochs with 350,000 capped weighted draws per epoch, followed by 72 PA adaptation epochs at AdamW learning rate 0.0002 with cosine decay to 0.00001. Only epochs 12, 24, 36, 48, 60 and 72 can be selected. Seed checkpoints are selected sequentially by complementarity of the current ensemble to the matching frozen-v22 control under a fixed 10% top-20 blend. There is no adaptive patience or post-assessment training.

Controls and diagnostics are fixed:

- three matched retained-PO seeds with the same architecture, initialization and schedule;
- three matched zero-PO seeds with the same architecture and PA schedule;
- one width-768, four-block single-head model using seed `27183`, diagnostic only;
- the refitted matched-v22 recipe for every outer fold;
- the exact unchanged official v22 CSV and its reconstructed production ranks.

The retained and single-head arms share each seed's PO-pretrained initialization. The large model is not eligible to replace the primary after assessment. Individual seed checkpoint histories and ensemble disagreement are recorded so seed diversity can be compared with the single larger architecture.

## OOD gate and cardinality

Policy calibration considers only a fixed grid. Candidate weights are 0, 0.025, 0.05, 0.10, 0.20 and 0.35. Gates are uniform, PA distance, PA distance plus PO coverage, and PA distance plus PO coverage plus seed disagreement. PO coverage uses the distance-weighted species richness of the eight nearest v22 pseudo-survey groups; no PA labels or external data enter this gate. Gate formulas and normalizations are fixed in `scripts/v23_protocol.py`.

Cardinality choices are `(near, far)` pairs `(20,20)`, `(16,20)`, `(18,22)`, `(20,24)` and `(20,28)`, with a fixed gate transition at 0.5. Uniform policies use fixed cardinality only. Calibration chooses the complete blend/gate/cardinality policy once; exact ties prefer unchanged v22/top-20 and then the smaller intervention.

## Freeze and manual-submission gate

Both outer paths, deployment models, checkpoint choices, calibrated policies and the production CSV are hashed before either new assessment is scored. The primary is eligible for later **manual** submission only when all of the following are true:

- the paired whole-one-degree-block 95% interval over matched frozen v22 has a positive lower bound;
- the primary beats matched frozen v22 in each fold;
- its pooled mean gain over matched zero-PO is positive;
- its pooled mean gain over matched retained-PO is positive;
- every outer and deployment primary policy has nonzero candidate weight;
- all source, data, split, parity, runtime, vocabulary, ordering and freeze checks pass;
- notebook tests pass both before and after the run;
- the candidate CSV differs from the official v22 CSV.

The bootstrap remains 500 whole-block resamples with seed 20250921. The assessment measures recipe transfer under overlapping cross-fit training, not independent hidden-test performance. A failed gate means `DO_NOT_SUBMIT`; it does not authorize a revised candidate or second assessment.

## Compute and data contract

One master notebook, one T4 used as `cuda:0`, Internet off, strict 10.5-hour end-to-end parent timeout. Only GeoLifeCLEF 2025 PA, PO and provided predictors are permitted. All 5,016 PA species remain. No external data, pretrained weights, nearest-96 heuristic, automatic Kaggle upload, kernel push, run, monitoring or competition submission exists in the manual package.

Kernel version 25 is the operational rerun of the unchanged preregistered v23
experiment. Version 23 stopped before assessment because the selected policy's
recorded `calibration_f1` audit field was compared against bare grid records.
Version 24 stopped during bootstrap before tests or training because updating
`PYTHONPATH` did not update the already-running notebook process's `sys.path`.
The rerun changes only those two operational defects and the expected Kaggle
version; the candidate grid, splits, seeds, training, assessment, and submission
gate are unchanged.
