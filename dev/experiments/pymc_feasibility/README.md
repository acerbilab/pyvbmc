# PyMC adapter feasibility check

Outputs of `dev/scripts/pymc_feasibility.py`, the bounded prototype of a
PyMC model adapter for PyVBMC that the
[PyMC integration proposal](../../2026-09-13-pymc-integration.md#feasibility-check)
asks for; the write-up is
[results/2026-09-14-pymc-feasibility.md](../../results/2026-09-14-pymc-feasibility.md).
Each directory holds one run's `report.json` (the environment with the
PyTensor compiler state, and per model the coordinates and bounds VBMC
received, the density and Jacobian checks with their numbers, the VBMC
fit against the model's evidence, the posterior moments, the structured
export and return path, and the rejections of the two unsupported models)
and `report.md` (the same as a table).

- `fit_laplace/`: starting point and plausible box from the mode and its
  Laplace approximation (`--plausible laplace`, the script's default);
  every model fits.
- `fit_prior/`: starting point from the model's initial point and the
  plausible box from prior quantiles (`--plausible prior`); the regression
  model's GP fit fails on the initial design, recorded under `errors`.
- `nofit/`: the checks that need no VBMC fit (`--no-fit`).

The environment that produced them (PyMC 6.3.2, PyTensor 3.3.1 without a
C++ compiler, ArviZ 1.3.0, Python 3.12, PyVBMC and gpyreg 1.2.1 installed
editable) is a dedicated virtual environment listed, with the command that
recreates it, in the gitignored `dev/scripts/runs/LOCAL.md`.
