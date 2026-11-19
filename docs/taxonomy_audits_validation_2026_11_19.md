# A2/A3/A3u validation — 19 November 2026

The local full suite passes **397 tests in 261.34 seconds**, with zero failures,
errors or skipped tests. Three warnings concern PyTorch nested-tensor behavior
and an upstream rasterio affine-operator deprecation. The preceding 17 November
A2/A3 run passed 393 tests; the final run includes four alias-mask regression
tests. These checks validate implementation and artifact integrity, not a new
scientific hypothesis or data independence.

Independent study validators reproduce all 163-cell confusion scores and
completion endpoints. They verify positive current-record witnesses (4,161 A2;
4,284 A3/A3u), all 553 exact synonym extensions, chronology/input identities and
20 A2 / 25 A3 / 25 A3u quota-selector rankings. A3u checks 24,040 uncertain records
and 25,286 eligible alias possibilities without a supported uncertain alias
being retained as a confident absence. All A3 positive labels are preserved;
only uncertainty expands. The A2 figure was visually inspected.

The original six E1/E2/E3/NEON/bridge/E4 manifests contain **1,033 files**. These,
the **70-file T1 manifest** and **11-file A1 manifest** remain unchanged. A2/A3/A3u
add separate source/input locks, label-before-score records, original source
snapshots and final artifact manifests. Earlier failed primary results are not
overwritten. The publication snapshot keeps documentation separate from frozen
study inputs so maintained README/manuscript documents can continue evolving.

Recheck the batch after restoring the documented local inputs:

```bash
python scripts/validate_taxonomy_audits.py
```

The script verifies hashes and existing independent-validation receipts; it does
not rerun experiments or replace sealed files. Its `--seal` operation is for the
initial release only. Detailed check output and the full JUnit report are in
`results/publication_validation_2026_11_19/`. Input caches are represented by
hashes in manifests and remain ignored, rather than bundled into Git.

Research code, protocols/results, tests and manuscript drafts are the publication
scope. Credentials, data caches, agent instructions, chat handoffs and assistant
worklogs remain local and excluded. This batch introduces **zero model fits,
zero GPU jobs and zero competition submissions**. The final field contract
admits only **33 cells / six sites**, below 100/12; original E4 is not rescued.
The T2 verification-cost study is proposed and has not run.
