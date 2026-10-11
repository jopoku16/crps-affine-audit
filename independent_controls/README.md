# Independent affine controls

This folder contains two checks that are separate from the source-specific compression experiments.

`run_cross_library_controls.py` tests the positive-affine law of empirical CRPS on 4,000 forecast cases. It uses four distributions, four ensemble sizes, and five recodings. Each equal-weight interface completes 20,000 comparisons. The weighted reference and two public weighted interfaces also complete 20,000 comparisons each. The script stores package versions, failures, maximum mapped-back errors, and agreement with direct reference formulas.

The package interfaces are useful negative controls, but they are not all independent algorithms. The tested version of `xskillscore` calls `properscoring`. The `scores` project records adapted CRPS code in its notice file. The separate `scoringrules` path and the local formulas give additional calculations. The manuscript states these relations directly.

`audit_source_history.py` clones the public target repository into a temporary directory. It records commit identifiers and Git blob identifiers for the two target files. It does not copy the upstream source into this release.

Install the locked control environment from the project root, then run:

```bash
python independent_controls/run_cross_library_controls.py
python independent_controls/audit_source_history.py
```

The stored JSON files are checked by `python run_all.py`.
