## What this is

PyVBMC is the Python port of the MATLAB VBMC toolbox (Variational Bayesian Monte
Carlo): sample-efficient posterior and model-evidence estimation for black-box,
possibly noisy, log-likelihoods with up to ~10–20 parameters. Plain NumPy/SciPy,
no autodiff. The algorithm papers live in `papers/` (verified Markdown
transcriptions); dated developer notes live in `dev/` (see `dev/README.md` for
the convention). `pyvbmc/vbmc/README.md` is the MATLAB-to-Python porting log.
`dev/scripts/` holds the benchmark target suite (`benchmark_targets.py`),
the profiler (`profile_run.py`, `profile_suite.py`) and the golden-trace
regression harness (`golden_trace.py`); `dev/README.md` describes them and
`dev/plans/benchmark-suite-and-golden-traces.md` records how they were built
and what they measured.

Before touching the numerical core, read `dev/2026-09-02-modernization-discussion.md`:
it maps the hot paths, catalogues every hand-derived gradient, lists known
latent bugs (§9), and records the plan and decisions for the NumPy → PyTorch
port. Do not re-derive what is written there.

## Setup and commands

`gpyreg` (the lab's GP library, `acerbilab/gpyreg`) is a sibling repo. CI installs
it from GitHub `main`, unpinned, so PyVBMC CI can break without a PyVBMC change.

```console
git clone https://github.com/acerbilab/gpyreg ../gpyreg
pip install -e "../gpyreg[dev]"
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

Extras: `test` (pytest, pytest-mock, pytest-rerunfailures; what the test
workflows install), `examples` (plotly for notebook 2), `dev` (both plus
docs and formatting tools).

Tests (no pytest config, no conftest, no markers; discovery is plain
`pyvbmc/testing/**/test_*.py`):

```console
python -m pytest --reruns=5 -x -vv                      # exactly what CI runs
python -m pytest pyvbmc/testing/vbmc/test_options.py::test_del -vv
pytest --reruns 5 --cov=. --cov-report html:cov_html    # coverage report
```

Formatting is enforced only by the pre-commit hook (black 79, isort profile
black, pycln, black-jupyter for notebooks); no lint or format job runs in CI.
`pylintrc` is stock and disagrees with black on line length; ignore it.

Docs (Sphinx, numpydoc docstrings; `VariationalPosterior` is the reference
docstring style):

```console
cd docsrc && make github     # builds and copies into ../docs  (Windows: .\make.bat github in cmd)
cd docsrc && make clean
```

`docs/` is the gitignored build output that `docs.yml` pushes to `gh-pages`.
Never put source material there. Every new public class or function needs a
hand-written `.rst` under `docsrc/source/api/` and a link in `index.rst`.

Notebooks in `examples/` ship in the wheel as `pyvbmc.examples` and are never
executed in CI or docs (`nb_execution_mode = "off"`). `examples/scripts/*.py`
are generated from them by `examples/scripts/Makefile`; regenerate, do not edit.

Version comes from git tags via setuptools_scm with no fallback: a shallow
clone or exported tarball fails to build. Commit messages follow conventional
commits. Work on feature branches; PRs to `main` run the full 3-OS × 3-Python
matrix, skipped when only docs changed. Pushes to `dev*` branches that touch
`pyvbmc/`, `pyproject.toml` or `setup.py` run a reduced smoke (Ubuntu,
newest Python) of the same `tests` workflow, which also runs the full
matrix on manual dispatch and twice a month on `main`.

## Architecture

`VBMC.optimize()` in `pyvbmc/vbmc/vbmc.py` is a ~760-line Python loop of
bookkeeping around three numerical stages, repeated until termination:

1. **Active sampling** (`vbmc/active_sample.py`): pick `fun_evals_per_iter`
   (default 5) new points by maximizing an acquisition function
   (`acquisition_functions/`) built from the current GP and variational
   posterior. A batched sieve over 2^13 candidates, then CMA-ES (`cma`
   package) or Nelder-Mead on single points; rank-1 GP update between points.
   Noisy targets use VIQR/IMIQR with `vbmc/active_importance_sampling.py`.
   Evaluations go through `FunctionLogger`.
2. **GP training** (`vbmc/gaussian_process_train.py` → `gpyreg.GP.fit`):
   squared-exponential ARD kernel (hard-wired; several paths fail on anything
   else), negative-quadratic mean, Gaussian noise with optional user-provided
   noise estimates. Hyperparameters are optimized with L-BFGS-B then
   slice-sampled; the number of samples `Ns` shrinks with `N` and drops to 0
   once `N >= 200 + 10 D`.
3. **Variational optimization** (`vbmc/variational_optimization.py`): update
   `K`, sieve candidate posteriors, then Adam (`vbmc/minimize_adam.py`) on the
   negative ELCBO. The expected log joint under the mixture is analytic
   (`_gp_log_joint`, Gaussian × SE-kernel integrals); the entropy comes from
   `entropy/` (Monte Carlo or Jensen lower bound). Low-weight components are
   pruned.

Then `iteration_history.record(...)`, warmup-end and termination checks
(reliability index, long-term stability), and at the end `final_boost`
(`K` → 50), `kl_div`, and a results dict. `whitening/` rotates the space using
the variational covariance.

Things you must hold in your head across files:

- **PyVBMC reaches into GP internals.** Six modules read
  `gp.posteriors[s].{hyp, alpha, L, L_chol, sW}` directly rather than calling
  `predict`; `_gp_log_joint` reimplements gpyreg's `quad` to add gradients.
  Any change to the GP representation touches all of them.
- **Gradients are hand-derived** for the ELBO with respect to `mu, sigma,
  lambd, w` across `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`,
  `entlb_vbmc`, `entmc_vbmc`, and for `vp.pdf`. Changing the variational
  parameterization means updating every one of them, plus the softmax
  Jacobian that is duplicated four times. Only the entropy gradients are
  finite-difference tested (`pyvbmc/testing/_check_grad.py`); the rest are
  pinned to stored reference arrays.
- **Two coordinate spaces.** The algorithm runs in an unbounded transformed
  space; users see the original constrained space. `ParameterTransformer`
  (`__call__` forward, `.inverse()` back; probit by default) mediates.
  `VariationalPosterior` methods take `orig_flag=True` by default and only
  provide gradients with `orig_flag=False`. The same transformer object must
  be shared by `vbmc`, `vp`, and `function_logger` (tests assert identity).
- **Shapes are rigid.** VP: `w (1,K)`, `mu (D,K)`, `sigma (1,K)`, `lambd (D,1)`.
  Bounds `(1,D)`, `x0 (n0,D)`. `decorators/handle_0D_1D_input.py` promotes
  1-D inputs for methods; it assumes a `self` first argument and misbehaves on
  module-level functions.
- **`FunctionLogger`** preallocates 500 rows and grows; `Xn` is the index of
  the last filled row, `X_flag` the boolean mask of live rows. Always index
  through `X_flag`; rows beyond `Xn` are NaN. `finalize()` trims.
- **`optim_state`** is a plain dict read and written everywhere (hundreds of
  sites). `IterationHistory` stores object-dtype arrays, deep-copies
  everything it records (VP, GP, logger, `optim_state`, every iteration), and
  only accepts keys pre-declared in the list in `VBMC.__init__`.
- **Options** are layered: `option_configs/basic_vbmc_options.ini`, then
  `advanced_vbmc_options.ini`, then `options_path=`, with the `options=` dict
  winning. `.ini` values are `eval`'d with `D` bound and only the names
  imported in `options.py` available (`np`, `ceil`, the acquisition function
  classes, lambdas); the `options=` dict is used verbatim. To add an option,
  add a `# description` line followed by `name = <expr>` to the right `.ini`;
  the comment is the user documentation. Unknown keys raise at validation.
  Options are frozen after init; use `options.__setitem__(k, v, force=True)`.
- **Randomness goes through `numpy.random.Generator` objects.** `VBMC(seed=)`
  creates `vbmc.rng` (`pyvbmc/rng.py: get_rng`), shared with `vbmc.vp`;
  `VariationalPosterior.__deepcopy__` shares the generator so every copy of a
  VP stays on one stream, and functions that receive a `vp` draw from
  `vp.rng`. `seed=None` derives the generator from the global `np.random`
  state so that `np.random.seed()` beforehand still fixes a run (the example
  notebooks rely on this). gpyreg (slice sampler, `f_min_fill`,
  `GP.random_function`) and the `cma` noise handler still draw from the
  global state, so a seeded instance reseeds it at construction and
  reinstalls its own snapshot when `optimize()` starts; drop that seam once
  gpyreg accepts a generator. The per-iteration `random_state` holds both
  the generator state and the legacy tuple; `load(set_random_state=True)`
  restores both. `test_vbmc_seed.py` deliberately holds two short (2
  iteration, `D=2`) `optimize()` runs: they are the only end-to-end check of
  the seam. Unseeded tests remain the reason CI uses `--reruns=5 -x`.
- **float64 everywhere, implicitly.** The Cholesky retry ladder in gpyreg
  exists because the matrices are borderline singular.

## Testing conventions and traps

- Seed explicitly, keep statistical tolerances loose, and do not add more
  full `optimize()` runs: `pyvbmc/testing/vbmc/test_vbmc_optimize.py` already
  holds six end-to-end runs and dominates runtime.
- `test_*_save_dynamic` write `.pkl` files into the source tree that the
  matching `load` tests read; running a `load` test alone fails.
- `test_*_save_static.pkl` fixtures are pickled instances of the current
  classes. Renaming or removing an attribute on `VBMC`, `VariationalPosterior`,
  `Options`, or `IterationHistory` breaks them, and they cannot be regenerated
  without rerunning `optimize()`.
- MATLAB `.mat` fixtures are read with `scipy.io.loadmat` (no MATLAB needed)
  but are opaque and unregenerable. New fixture files or directories must be
  added to `MANIFEST.in` by hand or they are missing from the sdist.
- New test directories need an `__init__.py` (`entropy/` and `whitening/`
  currently lack one).
- `pyvbmc/priors/__init__.py` has `# isort:skip` markers preserving a
  circular-import-safe order; do not reorder.
- `import pyvbmc` eagerly imports matplotlib.pyplot, corner, cma, and imageio.
  `pyvbmc.timer.main_timer` is a process-wide singleton shared by all `VBMC`
  instances, so concurrent runs in one interpreter are not safe.
