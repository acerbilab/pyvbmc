## What this is

PyVBMC is the Python port of the MATLAB VBMC toolbox (Variational Bayesian Monte
Carlo): sample-efficient posterior and model-evidence estimation for black-box,
possibly noisy, log-likelihoods with up to ~10–20 parameters. Plain NumPy/SciPy,
no autodiff. The algorithm papers live in `papers/` (verified Markdown
transcriptions); dated developer notes live in `dev/` (see `dev/README.md` for
the convention). `pyvbmc/vbmc/README.md` is the MATLAB-to-Python porting log.
`dev/scripts/` holds the benchmark target suite (`benchmark_targets.py`),
the profiler (`profile_run.py`, `profile_suite.py`), the golden-trace
regression harness (`golden_trace.py`) and its per-change replay gate
(`golden_replay.py`); `dev/README.md` describes them and
`dev/plans/benchmark-suite-and-golden-traces.md` records how they were built
and what they measured.

Before touching the numerical core, read `dev/2026-09-02-modernization-discussion.md`:
it maps the hot paths, catalogues every hand-derived gradient, lists known
latent bugs (§9), and records the plan and decisions for the NumPy → PyTorch
port. Do not re-derive what is written there.

## Setup and commands

`gpyreg` (the lab's GP library, `acerbilab/gpyreg`) is a sibling repo. CI
installs it from GitHub at the commit pinned in
`.github/workflows/test-matrix.yml` (`GPYREG_PIN`); only the twice-monthly
scheduled run tests against gpyreg `main`, so that is where a gpyreg change
breaking PyVBMC shows up first. Bump the pin when gpyreg `main` moves and
that run is green. PyVBMC requires gpyreg 1.1.0 or later (`GP.fit(rng=)`,
acerbilab/gpyreg#43, released 2026-09-05); the sibling checkout below is
for developing against gpyreg `main`.

```console
git clone https://github.com/acerbilab/gpyreg ../gpyreg
pip install -e "../gpyreg[dev]"
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

After gpyreg is tagged, `git fetch --tags` in the sibling checkout and
reinstall it editable: its setuptools_scm version otherwise stays below the
minimum in `pyproject.toml`, and the next `pip install -e .` installs
gpyreg from PyPI over the checkout.

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
newest Python). Both call the one job in `test-matrix.yml`, which `tests.yml`
also runs as the full matrix on manual dispatch and twice a month on `main`.

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
  Jacobian that is duplicated four times. Finite-difference checks
  (`pyvbmc/testing/_check_grad.py`) exist for the entropies and, since
  2026-09-02, for `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`,
  `_soft_bound_loss` and `vp.pdf` (`pyvbmc/testing/**/test_*_grad_fd.py`);
  the stored MATLAB arrays remain the value gate, and the oracles pin the
  numerics stage by stage. `_gp_log_joint` is vectorized over
  hyperparameter samples and mixture components (2026-09-04): one
  `(Ns, K, D, N)` array, `einsum` contractions, the variance from
  multi-RHS solves; the formulas are the loop's.
- **Two coordinate spaces.** The algorithm runs in an unbounded transformed
  space; users see the original constrained space. `ParameterTransformer`
  (`__call__` forward, `.inverse()` back; probit by default) mediates.
  `VariationalPosterior` methods take `orig_flag=True` by default and only
  provide gradients with `orig_flag=False`. The same transformer object must
  be shared by `vbmc`, `vp`, and `function_logger` (tests assert identity);
  the sieve candidates that `_vb_init` builds share it and the generator
  rather than copying them (a transformer is never mutated after
  construction; a warp installs a fresh copy), while `copy.deepcopy` of a
  VP copies the transformer and shares only the generator.
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
  only accepts keys pre-declared in the list in `VBMC.__init__`. The GP is
  recorded without its posterior factors (training data, hyperparameter
  samples and model only, through `_lean_gp` in `gaussian_process_train.py`),
  so the history does not grow with `Ns` times the square of the
  training-set size. `vbmc.get_gp(iteration)` returns a copy with the
  factors recomputed from the record, identical to the ones dropped; it is
  what `final_boost` and `load` use, and it copies a record that still
  carries its factors (files saved by versions that stored complete GPs)
  as it is.
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
  notebooks rely on this). The GP hyperparameter fit receives the generator
  (`train_gp(rng=)` → `gpyreg.GP.fit(rng=)`, which covers the space-filling
  design and the slice sampler; needs the gpyreg commit pinned in
  `test-matrix.yml` or later) and the CMA-ES noise-handler subclass in
  `active_sample.py` draws its re-evaluation count from `vp.rng`, so a run
  never reads or writes NumPy's global state (since 2026-09-05; before, a
  seeded instance reseeded the global state and the per-iteration
  `random_state` also held the legacy tuple, which
  `load(set_random_state=True)` now ignores). `test_vbmc_seed.py`
  deliberately holds three short (2 iteration, `D=2`) `optimize()` runs:
  the end-to-end reproducibility checks. Unseeded tests remain the reason
  CI uses `--reruns=5 -x`.
- **float64 everywhere, implicitly.** The Cholesky retry ladder in gpyreg
  exists because the matrices are borderline singular.

## Testing conventions and traps

- Seed explicitly, keep statistical tolerances loose, and do not add more
  full `optimize()` runs: `pyvbmc/testing/vbmc/test_vbmc_optimize.py` already
  holds six end-to-end runs and dominates runtime.
- An autouse fixture that saves and restores the global random state
  around each test (four modules have one) makes `--reruns` replay a failed
  attempt's draws exactly; where a module holds unseeded statistical tests
  (`test_variational_optimization.py`), the fixture advances the stream by
  the attempt number (`request.node.execution_count`, set by
  pytest-rerunfailures) before yielding, so a rerun sees fresh draws and
  later tests still find the stream where they used to.
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
- `pyvbmc/testing/oracles/` pins the numerics stage by stage: each fixture
  under `fixtures/` is one algorithm state (GP, VP, transformer, logger,
  `optim_state`, options, candidate set) saved as plain arrays, with the
  reference outputs of GP prediction, `vp.pdf`, every acquisition function,
  `_gp_log_joint`, `_neg_elcbo`, both entropies, the transformer and one
  seeded `active_sample` call. `pytest pyvbmc/testing/oracles` rebuilds the
  state through public constructors and recomputes in seconds; a failure
  means the numerics changed. Regenerate with
  `python dev/scripts/make_oracle_fixtures.py` only to set a new baseline
  on purpose, never to make a refactor pass; `--check` runs the comparison
  outside pytest. The one sanctioned exception is the `active_sample_step`
  oracle (a full CMA-ES search, which a few-ulp change in the acquisition
  sends to a different point): once every `acq_*` oracle is green, replace
  its references alone with `--rebaseline active_sample_step --reason
  "..."`, which recomputes from the stored state and keeps every other
  reference bit-identical (`dev/plans/stage2-batched-acquisition.md`);
  the same holds for `gp_fit` (a slice-sampling chain), and a change to
  the random stream moves both, so `--expect-moving` names the second
  while the first is re-baselined. A new oracle is added to the existing
  fixtures with `--add-oracle NAME --reason "..."` (never by rerunning the
  recipes, which would move every snapshot). The committed references pin
  the numerics of the day they were made and several outputs have since
  moved within tolerance, so the gate for a change that must not move any
  output is `--dump-outputs DIR` before it and `--check --exact --against
  DIR` after (`dev/plans/stage2-gpyreg-predict-and-sampler.md`); the
  per-step trajectory check is `python dev/scripts/golden_replay.py`
  (`dev/README.md`). The tests need a repository checkout: the testing
  package ships in the sdist, not the wheel, and the `active_sample_step`
  oracle also imports the benchmark targets from `dev/scripts` (skipped
  when absent); it and `gp_fit` run only on the platform that generated
  the fixtures (a CMA-ES search or a slice-sampling chain turns BLAS
  rounding differences into different decisions; `PYVBMC_ORACLES_ALL=1`
  forces the *tests* elsewhere, while the targeted generator modes refuse
  to run off that platform).
