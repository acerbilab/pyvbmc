# Acquisition oracle conditioning across platforms

The full OS/Python gate first failed on macOS in
[run 34251746916](https://github.com/acerbilab/pyvbmc/actions/runs/34251746916).
Five retries reproduced `corr_D5_warped/acq_AcqFcn`: maximum absolute error
1.38e-14, scaled error 1.49e-3 against rtol 1e-3 and atol zero. Fail-fast
cancelled the other eight jobs; they were not passing results.

## Evidence

[Diagnostic run 34259734758](https://github.com/acerbilab/pyvbmc/actions/runs/34259734758)
at `0370c5b` disabled fail-fast and evaluated all stored acquisition states
before pytest. Windows/Python 3.12 passed; the other eight jobs failed the
same oracle. This is a cross-platform comparison issue, not macOS alone.
The diagnostic's maximum raw scaled `AcqFcn` errors on the corr fixture were:

| OS | Python 3.10 | Python 3.11 | Python 3.12 |
| --- | ---: | ---: | ---: |
| Ubuntu | 0.00167021 | 0.00167021 | 0.00252283 |
| Windows | 0.00164972 | 0.00164972 | 0 |
| macOS | 0.00148388 | 0.00142927 | 0.00136967 |

Resolved NumPy/SciPy versions were respectively 2.2.6/1.15.3 (Python 3.10),
2.4.6/1.17.1 (3.11), and 2.5.3/1.18.1 (3.12). Gpyreg remained pinned to
`a2f8ddce867f502e29717959cf0ff3529f598618`. The diagnostic logs record actual
import paths, platform and BLAS configuration. Local copies are under the
ignored `dev/scripts/runs/latent_fixes/eta_A_20260908/diag_ci_*.log`.

The corr snapshot has 512 candidates, 299 subject to variance regularization
and 104 exact acquisition underflow zeros. Its variance threshold is 1e-4.
At the largest discrepancy (Ubuntu 3.12, index 252), reference variance
6.791153451893594e-7 becomes 6.791268875859837e-7: a relative change of
1.70e-5. The acquisition changes from -1.4157800769380085e-72 to
-1.4193518562966716e-72. Substituting only the current variance into the
reference formula predicts -1.4193518510711368e-72; the residual is
5.23e-81 in magnitude. The exponential penalty explains the amplification.
On macOS 3.10, separate GP predictions also drift slightly within the
diagnostic; its formula residual is not a comparison on identical inputs.

## Correction

Phase 4 activated variance regularization and updated the affected reference
values. The earlier cross-platform tolerance calibration did not account for
the newly active penalty. For simple acquisitions, with variance `v` and
threshold `t`, `A(v) = C v exp(-(t/v - 1))` below the threshold. Its local
relative condition number is `1 + t/v`, reaching 148.25 in this fixture.

The oracle comparison therefore scales its existing per-element denominator
by this reference-derived condition number on finite, nonzero, regularized
entries of `AcqFcn` and `AcqFcnVanilla`. `AcqFcnNoisy` uses the conservative
`2 + t/v` (its exact extra term is `sn2/(v+sn2)`). Applying `1 + t/v` to the
log acquisition's relative denominator is conservative rather than its
exact condition number. VIQR/IMIQR retain their existing comparisons; their
different raw formulas are outside this derivation.

This is a local sensitivity calibration, not a rigorous nonlinear error
bound. Unregularized entries, underflow zeros, nonfinite patterns, and exact
checks retain their previous gates. Canonical flag precedence is preserved.
Pytest and the fixture generator use the same helper; `--against` remains
anchored to the snapshot's stored GP predictions. Reported absolute errors
are unchanged, while scaled errors include the condition factor.

No solver, stored reference, base rtol/atol, or dependency pin changes.
`dev/scripts/diagnose_acquisition_oracles.py` remains available for manual
reproduction and reports both raw and conditioned comparisons. The temporary
CI diagnostic step is removed; fail-fast stays disabled so a failed platform
cannot conceal the status of the others.

## Validation

The focused oracle suite passed (143 tests, 15 platform-gated skips), all
11 fixtures passed `--check --exact`, and the strengthened comparison module
passed its nine tests. Independent Sol review found no issues.

Matrix 34261973325 at `fe7b304` passed all oracle gates and all six Python
3.11/3.12 jobs. It exposed a separate Python 3.10 mock-resolution issue in
the active-sampling rollback test: the package exports a function named
`active_sample`, so dotted mock targets reached that function instead of
the submodule. Commit `24cf369` uses `importlib.import_module` and four
`patch.object` calls, preserving the test's behavior and assertions.
The light mocked regression and independent review passed.

The final [matrix 34263042113](https://github.com/acerbilab/pyvbmc/actions/runs/34263042113)
at `24cf369` passed all nine OS/Python jobs. The live
[latent-fix plan](plans/latent-bug-fixes.md) tracks integration and subsequent
release work; no final population run was started.
