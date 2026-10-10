# CRPS implementation audit reproducibility package

This folder contains the revised manuscript and the evidence files for the CRPS implementation audit.

Public repository: https://github.com/jopoku16/crps-affine-audit

## Main files

- `manuscript_simple_revised.tex`: main manuscript source.
- `manuscript_simple_revised.pdf`: compiled manuscript.
- `real_data/expanded_validation_results.json`: expanded validation on 39,575 temperature forecasts.
- `real_data/affine_extension_results.json`: validation on 2,749 precipitation forecasts and the runtime study.
- `real_data/run_expanded_validation.py`: script that generated the expanded real-data validation.
- `real_data/run_affine_extension.py`: script for precipitation, affine recoding, runtime, and accuracy checks.
- `real_data/corrected_crps.py`: translation-anchored and affine-standardized wrappers.
- `supporting/`: preregistered robustness scripts, JSON outputs, source mapping records, and float32 diagnosis.
- `requirements-main-lock.txt` and `requirements-real-data-lock.txt`: exact Python package sets for the two analysis environments.
- `DATA_SOURCES.md`: public data and source provenance.

## One-command check

Run this from the folder:

```bash
python run_all.py
```

The script fetches and verifies the pinned public source when needed, runs unit tests, recompiles the manuscript twice, checks all stored validation results, and writes `SHA256SUMS.txt`.

The expanded real-data validation itself is computationally heavier. To rerun it directly:

```bash
cd real_data
python run_expanded_validation.py
python run_affine_extension.py
```

## Current manuscript claim

The paper is a source-specific audit of released CRPS code. It does not claim that a published figure is wrong. It does not introduce a new proper score. It shows that the released compressed routine applies an equal-weight correction to unequal weights, derives the resulting translation error, verifies the formula against unchanged code, and tests correction paths on synthetic and real forecasts. It also proves the transformation law of an affine-standardized wrapper.

## Real-data coverage

The expanded real-data validation includes:

- 2,749 Innsbruck minimum-temperature cases from `ensemblepp`.
- 36,826 Pacific Northwest surface-temperature cases.
- 2,749 Innsbruck precipitation cases.
- Celsius/Kelvin paired scoring.
- millimetre/inch and positive-affine paired scoring.
- released compressed score, normalized pair correction, weighted empirical score, anchored weighted correction, and full fair CRPS.
- equal-budget comparisons and an Innsbruck rolling ranking study.
- runtime and accuracy comparisons against pairwise and sorted exact fair CRPS.

## Reproducibility notes

The audited source is not included in the public release. Run `python real_data/fetch_pinned_source.py` to retrieve and verify commit `dddd3ea46a6199d2ed48019c7825e79fb242787e` from the public repository. The package includes locked environments, public-data provenance, scripts, stored outputs, tests, and checksums. `CITATION.cff` and `.zenodo.json` provide metadata for the permanent archived release.
