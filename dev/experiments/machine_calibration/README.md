# Machine-local chunk calibration evidence

Results: [2026-09-09 report](../../results/2026-09-09-machine-local-calibration.md).
Scope: [execution plan](../../plans/machine-local-calibration.md).
Numerical source base: `f7f0ce1af1664571dc55f2a7c69f14124b85db88`.
The experiment changes no production source and needs no saved solver states.

## Files

- `primary.json`: balanced five-round selection/four-round held-out sweep,
  nine workloads, five candidate budgets, allocation diagnostics.
- `repeat.json`: reversed candidate list, ten/eight rounds, same numerical
  source and synthetic recipe. All observations, including timing variability,
  are retained.
- `two_second_budget.json`: two-second discovery allowance, five-second
  overall allowance. Incomplete calibration returns the existing defaults.
  Numerical diagnostic batches can overrun the discovery allowance.
- `summary.json`: tables derived by `summarize.py` from primary/repeat only;
  includes candidate medians, held-out observations, same-budget controls,
  cost and traced allocation summaries. No same-setting timing fluctuation
  is counted as an amortizable gain.
- `initial_unbalanced.json`, `initial_unbalanced_runner.py`: excluded
  exploratory run and its exact measured source. Its rotate-and-reverse
  ordering was not position-balanced. The primary/repeat repair rotates
  within complete cycles and reverses only between them.

The balanced sweeps and timeout probe used
`dev/scripts/calibrate_chunks.py` with byte SHA256
`156008d9e59b269876e7f4fb1a3cbaa2edd972a53898ce565cdb00ed1a590938`.
The excluded runner has byte SHA256
`0e78c525f99f41fb35075ab70f21d30afc7a28d526072616d793190f95366860`.
Each JSON records the measured runner hash and production-function source
hashes. Byte hashes depend on checkout line endings; the function-source
hashes identify the inspected numerical implementation.

## Reproduce

These are pre-integration measurements. Use the numerical source at the
commit named above, with the retained developer runner copied into that
checkout. Its source guards intentionally reject the later shared-kernel
implementation. New measurements of the integrated feature use
`pyvbmc.calibrate()`; the historical runner and JSON are kept unchanged.

Use the repository's ordinary development installation and optionally
`pip install threadpoolctl` to inspect effective BLAS threads. The measured
environment was Python 3.12.6 / NumPy 2.5.2 / SciPy 1.18.1 / GPyReg 1.1.0.
This session reused `.venv` and loaded an existing system installation's
pure `threadpoolctl.py` via an ignored directory; no numerical dependency
was replaced. That directory is unnecessary with threadpoolctl installed
normally. No compiler, Torch environment or historical raw run is needed.

From the repository root in PowerShell, using a fresh output directory:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONPATH = '.'
.\.venv\Scripts\python.exe dev/scripts/calibrate_chunks.py --quick --output dev/scripts/runs/calibration_reproduce/pilot.json
.\.venv\Scripts\python.exe dev/scripts/calibrate_chunks.py --trace-allocations --output dev/scripts/runs/calibration_reproduce/primary.json
.\.venv\Scripts\python.exe dev/scripts/calibrate_chunks.py --budgets 262144 131072 65536 32768 16384 --repeats 10 --heldout-repeats 8 --trace-allocations --output dev/scripts/runs/calibration_reproduce/repeat.json
.\.venv\Scripts\python.exe dev/scripts/calibrate_chunks.py --calibration-seconds 2 --deadline 5 --output dev/scripts/runs/calibration_reproduce/two_second_budget.json
```

Run commands sequentially with no other heavy computation. Full defaults
allow 90 seconds for discovery, 180 seconds overall, 20 seconds per case.
Limits are experimental soft allowances, not interruptible NumPy calls or
hard memory caps. Check JSON `status`, numerical gates, RNG state, completed
case counts and thread metadata before interpreting timings. CLI exit zero
includes `partial_defaulted`; it does not imply complete measurement.

Rebuild the retained summary with:

```powershell
.\.venv\Scripts\python.exe dev/experiments/machine_calibration/summarize.py
```

The script reads the retained primary/repeat JSON in this directory and
writes `summary.json`. Candidate medians use only measured representatives;
held-out speedups are medians of paired default/selected ratios. Control
ratios pair each primary arm with its same-budget copy. Amortization, if
applicable, uses discovery wall time excluding numerical diagnostics divided
by the difference of held-out median call times; it excludes imports and
shared setup and is null for unchanged budgets or nonpositive savings.

Focused checks:

```powershell
.\.venv\Scripts\python.exe -m pytest dev/scripts/test_calibrate_chunks.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest pyvbmc/testing/entropy/test_entmc_vbmc.py pyvbmc/testing/variational_posterior/test_variational_posterior_grad_fd.py -q
```

If the sandbox cannot access pytest's existing temporary directory, point
`PYTEST_DEBUG_TEMPROOT` at a fresh writable directory under
`dev/scripts/runs/`. The focused suite passed 11 tests; the existing
numerical suite passed eight. No production default or tuning policy is
applied by this runner, and no population benchmark is part of reproduction.
