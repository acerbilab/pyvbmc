## What this is

PyVBMC is the Python port of the MATLAB VBMC toolbox (Variational Bayesian Monte
Carlo): sample-efficient posterior and model-evidence estimation for black-box,
possibly noisy, log-likelihoods with up to ~10–20 parameters. Plain NumPy/SciPy,
no autodiff. The algorithm papers live in `papers/` (verified Markdown
transcriptions); dated developer notes live in `dev/` (see `dev/README.md` for
the convention). `pyvbmc/vbmc/README.md` is the MATLAB-to-Python porting log.
`dev/scripts/` holds the benchmark target suite (`benchmark_targets.py`;
its real-data targets read the plain `.npz` archives under
`dev/scripts/data/`, exported from the lab-private benchflow repository by
`export_benchflow_data.py`, and their regenerated ground truths under
`dev/scripts/data/truths/`, written by `make_benchmark_truths.py`), the
profiler (`profile_run.py`, `profile_suite.py`), the golden-trace
regression harness (`golden_trace.py`) and its per-change replay gate
(`golden_replay.py`); `dev/README.md` describes them,
`dev/plans/benchmark-suite-and-golden-traces.md` records how they were built
and what they measured, and `dev/plans/benchmark-realistic-targets.md`
records the real-data targets. Raw campaign artifacts (traces, run pools,
captured states, frozen worktrees) are gitignored under `dev/scripts/runs/`
and exist only on the machines that produced them; such a machine lists
them in the gitignored `dev/scripts/runs/LOCAL.md`, so if that file is
absent nothing is stored locally, and tracked documents point at it
rather than saying "this machine".

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
that run is green. PyVBMC requires gpyreg 1.2.1 or later: 1.2.0 added
`GP.predict(return_cross_covariance=True)` for VIQR kernel reuse
(acerbilab/gpyreg#45), and 1.2.1 fixed the heteroskedastic quadrature
variance, which moves the VIQR acquisition by one ulp and nothing else
(the `acq_AcqFcnVIQR` oracle was re-baselined to it on 2026-09-14; every
other oracle is bit-identical). The sibling checkout below is for
developing against gpyreg `main`.

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

Extras: `torch` (Torch posterior export and S-VBMC), `arviz` (DataTree
export, Python >=3.12), `pymc` (PyMC >=6.3 adapter plus the ArviZ
dependencies, Python >=3.12), `test` (pytest, pytest-mock,
pytest-rerunfailures; what the test workflows install), `examples` (plotly
for notebook 2), `dev` (the test and examples dependencies plus docs and
formatting tools, and
scikit-learn for the mixture proposal of the benchmark ground-truth
generator).

Tests (no conftest or markers; `pyproject.toml` limits default pytest
discovery to the shipped package suite under `pyvbmc/testing`):

```console
python -m pytest --reruns=5 -x -vv                      # exactly what CI runs
python -m pytest pyvbmc/testing/vbmc/test_options.py::test_del -vv
pytest --reruns 5 --cov=. --cov-report html:cov_html    # coverage report
```

Developer experiment checks under `dev/scripts/` are outside default
discovery but remain runnable by explicit path. The eta-bound comparison is
deliberately tied to its validated historical source, so run
`python -m pytest dev/scripts/test_eta_bound_variants.py -vv` only from the
historical checkout `c4c692c`, which includes the checks and retains the
`03650a2` numerical source pinned in `dev/scripts/eta_bound_variants.py`.

On the machine that generated the oracle fixtures, run the suite with
BLAS single-threaded (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`): the platform-bound oracles were generated that way,
and the CMA-ES search of `active_sample_step` picks different points under
default threading. Elsewhere those oracles are skipped (unless
`PYVBMC_ORACLES_ALL=1` forces them) and the thread count does not matter.

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
commits and carry no `Claude-Session:` trailer (a `Co-Authored-By:` line is
fine). A change that a user can notice is listed in `CHANGELOG.md` under
`Unreleased` with the work that makes it, in a sentence written for users and
relative to the last release (a fix to a feature that no release has shipped
belongs to that feature's entry). A change that can stop a script written for
the last release, or change what it returns, also has one line in the
"Upgrading from" list that opens the section, kept in step with its entry.
Work on feature branches; PRs to `main` run the full 3-OS × 3-Python matrix,
skipped when only docs changed. Pushes to `dev*` branches that touch
`pyvbmc/`, `pyproject.toml` or `setup.py` run a reduced smoke (Ubuntu, newest
Python). Both call the one job in
`test-matrix.yml`, which `tests.yml` also runs as the full matrix on manual
dispatch and twice a month on `main`. `feat-*` and `dev-*` branches are live
work; an implementation that is rejected or parked after evaluation leaves the
working line and is kept for the record on a branch named `retain/<topic>`
(`retain/experimental-acquisitions`, `retain/viqr-rqmc-nodes`), cut at the last
commit that holds it and named for the topic its records describe.

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
  be shared by `vbmc`, `vp`, and `function_logger` (tests assert identity,
  during a run and after `load`, which shares the transformer of the
  iteration it restores); `vbmc.x0` holds the starting points in the
  initial transformed space, which a warp leaves behind, and `vbmc.x0_orig`
  holds them in the caller's coordinates;
  the sieve candidates that `_vb_init` builds share it and the generator
  rather than copying them (a transformer is never mutated after
  construction; a warp installs a fresh copy), while `copy.deepcopy` of a
  VP copies the transformer and shares only the generator.
- **Shapes are rigid.** VP: `w (1,K)`, `mu (D,K)`, `sigma (1,K)`, `lambd (D,1)`.
  Bounds `(1,D)`, `x0 (n0,D)`. `decorators/handle_0D_1D_input.py` promotes
  1-D inputs for methods; it assumes a `self` first argument and misbehaves on
  module-level functions.
- **Vectorized targets are opt-in.** With `vectorized_target=True`,
  `FunctionLogger.batch_call` evaluates missing initial-design rows in one
  original-coordinate `(M,D)` call, then records supplied/evaluated rows in
  design order. Later target calls remain sequential with shape `(1,D)`.
  Separate priors retain their scalar interface. Missing flags in old saves
  mean False; load-time overrides synchronize the logger and prior wrapper.
- **Posterior exports.** `vp.to_torch()` copies CPU float64 state by default
  and returns a distribution in original coordinates; conversion makes no
  draws, its samples use torch's RNG. Bounded density inputs must be strictly
  interior. `vp.to_arviz()` exports one-chain DataTree samples and advances
  `vp.rng`. Extras are `torch` (torch >=2.7), `arviz` (current API,
  Python >=3.12) and `pymc` (PyMC >=6.3 plus ArviZ, Python >=3.12); core
  PyVBMC remains Python >=3.10. No exported objects are retained on the VP,
  so the dtype canary is unchanged.
- **PyMC targets.** `PyMCTarget` snapshots PyMC-managed data, residual
  numeric shared graph inputs, dimensions and coordinates before setup.
  Free variables retain the given model's order; two-sided variables use
  model coordinates, while supported one-sided variables keep PyMC's value
  transform and Jacobian. The adapter returns `-inf` only on or outside its
  hard box. A variable that keeps a log transform is accepted only if its
  own prior density stays finite down to sixteen decades below its initial
  value, so a shifted support (`pm.Wald` with a nonzero `alpha`) is
  rejected at construction. `to_model_variables` returns a backward-map
  output that rounded onto a finite support bound as the adjacent value
  inside the support, and rejects non-finite and out-of-support values.
  Five undocumented reaches are capability-guarded: transform
  classes; transform `args_fn`/`forward`/`backward`; the default-transform
  registry; Model mappings; and graph reconstruction/random-variable
  recognition (including PyMC symbolic random variables). Setup
  observations reach `VBMC(target)` through `precomputed_evaluations`, and
  their actual `setup_cost` is charged once as `initialization_cost`.
  Focused tests live under `pyvbmc/testing/pymc/` and run in the optional
  Python 3.12 PyMC CI cell.
- **S-VBMC** (`pyvbmc/svbmc/`, the standalone `svbmc` package moved in at
  0.1.1) stacks finished posteriors of several runs. It needs the `torch`
  extra, imported lazily: `pyvbmc.SVBMC` resolves on request and
  constructing without Torch raises naming the extra, so `import pyvbmc`
  never imports Torch. `sample()` draws independently from the stacked
  mixture through copies that use the object's generator (`seed=`), never
  the inputs'. Construction rejects runs whose original-space hard bounds
  differ (the bounded transform and any warp are free per run), and every
  `optimize()` starts from the runs' own weights whatever `w` holds. Its
  Torch-dependent tests run only where Torch is installed
  (one CI cell); the fixtures are plain-array snapshots under
  `pyvbmc/testing/svbmc/fixtures/` written by
  `dev/scripts/make_svbmc_fixtures.py`, and `references.npz` there pins the
  numerics of a seeded short optimization, the gate for changes to the
  entropy code. `dev/plans/svbmc-integration.md` records the decisions.
  `elbo` is a scalar headline (component-median capped for noisy stacks,
  raw for noiseless ones); `elbo_details` retains all estimates and noise
  provenance. `elbo_sd` combines entropy sampling and GP quadrature
  uncertainty for the raw evaluation at selected weights, excluding
  selection bias and cap uncertainty. Jacobian expectations are computed
  deterministically at construction; `optimize(n_samples_final=100)` uses
  fresh final draws. VP stats record `uncertainty_handling_level`; old
  posteriors use the `elbo_sd > 0.1` proxy unless `SVBMC(noisy=)` overrides
  it. `dev/plans/svbmc-elbo-reporting.md` records this reporting contract.
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
  what `optimize()` hands to `final_boost` and what `load` uses, and it
  copies a record that still
  carries its factors (files saved by versions that stored complete GPs)
  as it is. The recorded `optim_state` leaves out the importance samples
  of the noisy acquisitions (`active_importance_sampling`, drawn afresh
  every active-sampling step and not rebuildable from the record) unless
  the option `record_full_history_details` is set.
- **Options** are layered: `option_configs/basic_vbmc_options.ini`, then
  `advanced_vbmc_options.ini`, then `options_path=`, with the `options=` dict
  winning. `.ini` values are `eval`'d with `D` bound and only the names
  imported in `options.py` available (`np`, `ceil`, the acquisition function
  classes, lambdas); the `options=` dict is used verbatim. To add an option,
  add a `# description` line followed by `name = <expr>` to the right `.ini`;
  the comment is the user documentation, and an option that no module reads
  through an options mapping must be registered in `INERT_OPTIONS`
  (`options.py`), which a test recomputes from the package. A name that
  neither shipped file declares raises on every route (the dict, the
  `options_path=` file, `load(new_options=)`). An option set in the
  `options_path=` file counts as set by the user, as one in the dict does,
  and the defaults that depend on other options (`update_defaults`: the
  noisy-target defaults, applied whenever uncertainty handling is on, by
  `specify_target_noise` or by `uncertainty_handling=True`) are settled
  after every source is read. `uncertainty_handling` (a boolean),
  `integer_vars` (a boolean mask or 0-based indices), `max_fun_evals` and
  `max_iter` (positive integers, `max_iter` raised to `min_iter`),
  `gp_mean_fun` and `gp_hyp_sampler` (the values the package implements:
  `zero`, `const`, `negquad`; `slicesample`) and `f_vals` (not with
  `specify_target_noise`) are checked at construction. Options are frozen
  after init against assignment and removal; use `options.__setitem__(k, v,
  force=True)` and `options.__delitem__(k, force=True)`.
- **Randomness goes through `numpy.random.Generator` objects.** `VBMC(seed=)`
  creates `vbmc.rng` (`pyvbmc/rng.py: get_rng`), shared with `vbmc.vp`;
  `VariationalPosterior.__deepcopy__` shares the generator so every copy of a
  VP stays on one stream, and functions that receive a `vp` draw from
  `vp.rng`. `seed=None` derives the generator from the global `np.random`
  state so that `np.random.seed()` beforehand still fixes a run (the example
  notebooks rely on this). The GP hyperparameter fit receives the generator
  (`train_gp(rng=)` → `gpyreg.GP.fit(rng=)`, which covers the space-filling
  design and the slice sampler; needs the gpyreg commit pinned in
  `test-matrix.yml` or later), the CMA-ES search in `active_sample.py`
  draws its populations through the generator (the `randn` option of
  `cma`), and
  `active_importance_sampling` passes `vp.rng` to the slice sampler of its
  MCMC step (IMIQR), so a run never reads or writes NumPy's global state
  (since 2026-09-05, and for IMIQR runs since 2026-09-10, when the
  `acq_AcqFcnIMIQR` oracle was re-baselined; before, a seeded instance
  reseeded the global state and the per-iteration `random_state` also held
  the legacy tuple, which `load(set_random_state=True)` now ignores). `test_vbmc_seed.py`
  deliberately holds three short (2 iteration, `D=2`) `optimize()` runs:
  the end-to-end reproducibility checks, one of them shared through a
  module-scoped fixture with the live check of the dtype canary. Unseeded
  tests remain the reason CI uses `--reruns=5 -x`.
- **float64 everywhere, implicitly.** The Cholesky retry ladder in gpyreg
  exists because the matrices are borderline singular. No class declares
  the dtype of its state; the dtype canary (`pyvbmc/testing/_dtype.py`,
  run from the oracle tests on every raw oracle output and rebuilt state
  and from `test_vbmc_seed.py` on a live run) pins it. `VBMC.__init__`
  widens bounds and `x0` to float64 without modifying caller arrays;
  `test_vbmc_init.py` checks narrow floating inputs and the resulting state.
  Mathematical variance placeholders are floating zeros; counts remain integers.

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
- Several subpackages export a function under the name of the module that
  defines it (`pyvbmc.vbmc.active_sample`, `active_importance_sampling` and
  `create_vbmc_animation`; likewise in `pyvbmc.entropy` and `pyvbmc.stats`),
  so inside the package that name is the function. A mock target written as a
  dotted name through it, such as `"pyvbmc.vbmc.active_sample.cma.fmin"`,
  resolves to the module on Python 3.11 and later and to the function on
  Python 3.10, where it raises `ModuleNotFoundError`. Patch the module object
  (`importlib.import_module("pyvbmc.vbmc.active_sample")` with
  `mocker.patch.object`), or the defining module of what is patched
  (`"cma.fmin"`).
- `test_*_save_dynamic` write `.pkl` files into the source tree that the
  matching `load` tests read; running a `load` test alone fails.
- `test_*_save_static.pkl` fixtures are pickled instances of the current
  classes. Renaming or removing an attribute on `VBMC`, `VariationalPosterior`,
  `Options`, or `IterationHistory` breaks them, and they cannot be regenerated
  without rerunning `optimize()`. Load the VBMC one, never save it again: it
  holds the target function and PyVBMC's log-joint wrapper pickled by value,
  as bytecode of the Python version that wrote the file, and pickling such a
  function makes dill disassemble it, which corrupts memory on Python 3.11
  (a segmentation fault in that test, in a later garbage collection, or when
  the interpreter exits). A test that needs a saved run with history makes a
  short run of its own. A saved posterior holds no function: a
  `ParameterTransformer` pickles and copies without its bounded-transform
  functions and rebuilds them from `bounded_types` when restored, so
  posterior files and `SVBMC` objects move between Python versions.
  `test_vp_save_bounded_py311.pkl` and `_py312.pkl` are one bounded
  posterior written under two Python versions by the last commit that
  stored those functions, so that every CI cell uses a file holding
  bytecode of another Python version; calling it ended the interpreter.
- MATLAB-derived reference values live in plain `.npz` files (flat keys,
  `np.load(path, allow_pickle=False)`) under `entropy/`,
  `variational_posterior/` and `vbmc/compare_MATLAB/`, each directory with a
  `FIXTURES.md` listing what every array holds, which test reads it at what
  tolerance, and where it came from. Regenerating any of them needs MATLAB
  and the VBMC toolbox, so the numbers are fixed. New fixture files or
  directories must be added to `MANIFEST.in` by hand or they are missing
  from the sdist.
- New test directories need an `__init__.py` (`entropy/` and `whitening/`
  currently lack one).
- The package is installed editable from one checkout. In any other
  checkout or worktree, `python dev/scripts/<script>.py` imports the
  installed checkout's package, because `sys.path[0]` is the script's
  directory, not the current one; `python -m pytest` from that checkout's
  root is unaffected, since it puts the current directory first. Run every
  script gate (`make_oracle_fixtures.py`, `golden_replay.py`, the benchmark
  runners) with `PYTHONPATH=<that checkout>` and print `pyvbmc.__file__`
  once before trusting the result.
- `pyvbmc/priors/__init__.py` has `# isort:skip` markers preserving a
  circular-import-safe order; do not reorder.
- `import pyvbmc` eagerly imports matplotlib.pyplot, cma, and imageio.
  Corner is imported inside `vp.plot`; torch, ArviZ and PyMC integrations
  import their dependencies lazily, so importing `pyvbmc` does not import
  torch, PyMC or PyTensor.
  `pyvbmc.timer.main_timer` is a process-wide singleton shared by all `VBMC`
  instances, so concurrent runs in one interpreter are not safe.
- `pyvbmc/testing/oracles/` pins the numerics stage by stage: each fixture
  under `fixtures/` is one algorithm state (GP, VP, transformer, logger,
  `optim_state`, options, candidate set) saved as plain arrays, with the
  reference outputs of GP prediction, `vp.pdf`, every acquisition function,
  `_gp_log_joint`, `_neg_elcbo`, both entropies, the transformer and one
  seeded `active_sample` call. `pytest pyvbmc/testing/oracles` rebuilds the
  state through public constructors and recomputes in seconds; a failure
  means the numerics changed, or a dtype did (every raw oracle output must
  be float64, the state an oracle worked on is walked for stray
  floating-point dtypes, and the load-bearing arrays of every rebuilt
  state are asserted float64). Regenerate with
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
  while the first is re-baselined. The `acq_AcqFcnIMIQR` reference moves
  when the importance sampler's random stream changes. After the 2026-09-10
  RNG fix, it was re-baselined from the stored state on the original Windows
  generating machine on 2026-09-11; all 11 fixtures pass `--check --exact`.
  A new oracle is added to the existing
  fixtures with `--add-oracle NAME --reason "..."` (never by rerunning the
  recipes, which would move every snapshot). The committed references
  equal the current numerics on the generating machine (re-baselined at
  the end of Stage 2, 2026-09-06, after the population check; the
  `meta["rebaselined"]` entries record what moved and why), so the gate
  for a change that must not move any output is `--check --exact`
  against them; `--dump-outputs DIR` before a change and `--check --exact
  --against DIR` after remain available for a change made while the
  references are known to lag (`dev/plans/fixture-generator-and-oracles.md`);
  the per-step trajectory check is `python dev/scripts/golden_replay.py`
  (`dev/README.md`). The tests need a repository checkout: the testing
  package ships in the sdist, not the wheel, and the `active_sample_step`
  oracle also imports the benchmark targets from `dev/scripts` (skipped
  when absent); it and `gp_fit` run only on the platform that generated
  the fixtures (a CMA-ES search or a slice-sampling chain turns BLAS
  rounding differences into different decisions; `PYVBMC_ORACLES_ALL=1`
  forces the *tests* elsewhere, while the targeted generator modes refuse
  to run off that platform).
