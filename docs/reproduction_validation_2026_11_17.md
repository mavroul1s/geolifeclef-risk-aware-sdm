# Research snapshot reproducibility checks — 17 November 2026

**Final full-suite result: 376 passed, zero failures, errors or skips.** This is
the complete current local suite, including the restored artifact-backed checks.
The final JUnit report and compact validation record are retained in
`results/publication_validation_2026_11_17/`. The initial failed run and its
causes are described below rather than being relabeled as an initial pass.

The release includes the completed reliability studies, T1 synthetic taxonomic
stress test and A1 dictionary/metadata audit. Scientific failures remain
failures; release correctness checks do not establish a successful new method.
The README distinguishes current work, achieved evidence and future plans.

## Archive portability repairs

The initial broad test run reports **353 passed / 21 failed**. Failures expose
missing historical fixtures, sibling-script import paths and archival CSV
line-ending conversion. The original numerical models and notebook cores are
not changed to make the checks pass.

- Eleven archived submission CSVs recover their **exact pre-existing SHA-256
  hashes** by restoring CRLF bytes. No prediction values or official scores
  change. Before/after hashes are in
  `results/publication_validation_2026_11_17/archive_byte_recovery.json`.
- Git attributes preserve scientific-result and notebook bytes across checkouts,
  while normal source text uses LF. Pytest includes source and script paths.
- The exact **55,325-survey v21–v24 consumed-ID union** is decoded from the
  already committed v25 notebook, after checking the embedded compressed hash.
  The compact CSV/provenance record lives in `results/competition_history/`.
  Builders and tests use the equivalent union rather than unavailable individual
  old assessment files. No per-version scores or records are fabricated.
- The legacy v23 control fixture is restored from the verified v24 notebook
  payload. The test intentionally uses the documented missing-final-CRLF variant;
  it is explicitly a reconstructed test fixture, not a new raw-original claim.
- The small official sample-submission template is downloaded from Kaggle for
  legacy round-trip tests. No new training, GPU run or competition submission
  occurs. The retrieval receipt records source, size and digest.

The provenance test independently confirms that the restored exclusion CSV
equals the original embedded union, including order and count, and rejects
modified history. Archived model cores, selection policies and historical
study manifests are preserved.

## Reproducing tests

Install Python 3.12 and `.[dev,research]`. Synthetic tests are self-contained;
historical competition checks also require the official metadata/species cache.
After downloading and extracting the official competition CSVs, run:

```bash
python scripts/restore_legacy_test_control.py --competition-root data/raw
pytest -q
```

This mirrors the supplied official train metadata and sample template into the
legacy ignored cache, derives the 5,016-class PA vocabulary and restores the
embedded control fixture. It refuses conflicting local inputs. Restoring only
the control needs no competition download; omit `--competition-root`.

Full scientific experiments additionally require their documented data/model
inputs and role contracts. Frozen assessment phases refuse overwrites. Exact
source/data hashes are integrity evidence, not a claim of fresh outcomes,
biological truth or publication novelty.
