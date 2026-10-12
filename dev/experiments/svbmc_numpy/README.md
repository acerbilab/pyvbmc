# S-VBMC optimized NumPy prototype evidence

The human interpretation belongs in
[the detailed report](../../results/2026-09-09-svbmc-numpy-prototype.md),
summarized in the existing ecosystem integration proposal. The
[plan](../../plans/svbmc-numpy-prototype.md) records scope and execution.

- `comparison.json`: primary 18 NumPy/upstream-Torch fit pairs, six separate
  diagnostic pairs, three preparation/kernel cases, Adam parity and source,
  corpus, environment and timing metadata.
- `control.json`: separate 18 NumPy/shared-preparation-Torch fit pairs,
  upstream entropy/gradient/RNG gates, setup and provenance evidence.
- `comparison.sources/` and `control.sources/`: exact measured script bytes,
  saved as `.py.txt` to keep formatting hooks from modifying them. Hashes
  and filenames are recorded in the corresponding JSON. `.gitattributes`
  preserves those bytes across checkout newline settings.

Run the scripts serially from the repository root, after recreating the
Windows/Python 3.12 environment and isolated upstream checkout described in
the [compatibility record](../../results/2026-09-08-svbmc-compatibility.md).
The runners require the recorded source commit, all thirty corpus hashes,
NumPy 2.5.2, SciPy 1.18.1, Torch 2.14.0+cpu and pinned gpyreg. They verify
actual imports and set BLAS/Torch to one thread. These intentionally strict
checks describe this measurement environment, not minimum package versions.

```console
python -m pytest dev/scripts/test_svbmc_numpy_core.py -q
python dev/scripts/svbmc_numpy_prototype.py --phase all --output dev/scripts/runs/svbmc_numpy_reproduction/comparison.json
python dev/scripts/svbmc_numpy_control.py --output dev/scripts/runs/svbmc_numpy_reproduction/control.json
```

The default source and Torch-overlay paths reuse the compatibility checkout;
`--source` and `--deps` can select another copy with the same pinned contents.
To reproduce frozen code after live scripts change, copy the archived
`.py.txt` files back to their recorded `dev/scripts/*.py` paths in an
isolated checkout first. Use a new output location; the scripts refuse to
overwrite a report unless explicitly requested.

Numerical tolerances are recorded in the JSON before measurements.
`status: complete` and `validation.passed: true` indicate a completed passing
run. Failed comparisons remain explicit, and exceptions are checkpointed.
The primary fit timings exclude diagnostic tracing; all complete-fit timings
include fresh preparation and any per-iteration NumPy-to-Torch conversion.
Imports, warmup and construction are separate. Kernel timings reuse a fixed
matrix and must not be substituted for complete-fit timings.

Developmental smoke/profile outputs remain ignored under
`dev/scripts/runs/svbmc_numpy_20260909/`. No production S-VBMC integration,
GPU study, or final PyVBMC 870-case benchmark is represented by these files.
