## About this file

This file states what holds in this repository, for an agent who would not
meet it at the point of need: couplings that span files, procedures that gate
a change, traps that fail silently or at a cost, and conventions that nothing
enforces. It is not a record of changes: what a change did belongs to its
plan, its ledger or its fixture metadata. What an agent meets where it
matters stays there too, in a module's docstring, an option's description or
the user documentation. Add a line only when an agent without it would go
wrong, and write it as a fact about the repository as it stands.

## The project

PyVBMC is the Python port of the MATLAB VBMC toolbox (Variational Bayesian
Monte Carlo): sample-efficient posterior and model-evidence estimation for
black-box, possibly noisy, log-likelihoods with up to ~10–20 parameters.
Plain NumPy/SciPy, no autodiff. The GP layer is the lab's `gpyreg`
(`acerbilab/gpyreg`), a sibling repository.

- `papers/` holds Markdown transcriptions of the algorithm papers.
- `dev/` holds the developer notes, plans, results, experiment records and
  tooling. `dev/README.md` says where each kind of record goes and describes
  the tooling under `dev/scripts/` (benchmark targets, profiler, golden
  traces and their replay gate, fixture generators); `dev/TODO.md` lists the
  open work.
- `dev/2026-09-02-modernization-discussion.md` maps the hot paths of the
  numerical core, catalogues every hand-derived gradient and lists the latent
  defects found. Read it before changing the numerical core; do not re-derive
  what is written there.
- `pyvbmc/vbmc/README.md` is the MATLAB-to-Python porting log and holds the
  catalogue of the deliberate differences from MATLAB. Check it before
  treating a difference from MATLAB as a porting error, and update its entry
  with any change that creates, removes or alters such a difference. The
  records of the port correctness review under
  `dev/experiments/port_review_20260919/` hold the evidence behind the
  catalogue, citing the revisions that their headers name.
  `known_differences.md` is the sheet the catalogue was consolidated from,
  with each entry's history, and `counterpart_map.md` pairs every MATLAB
  file with its Python counterpart. The review's findings and their
  dispositions are in `dev/results/2026-09-23-port-correctness-review.md`.
- `docsrc/` is the Sphinx source. `docs/` is its gitignored build output;
  never put source material there.

Raw campaign artifacts (traces, run pools, captured states, frozen worktrees)
are gitignored under `dev/scripts/runs/` and exist only on the machines that
produced them; such a machine lists them in the gitignored
`dev/scripts/runs/LOCAL.md`. The frozen worktrees are whole copies of the
repository at old revisions, so a file search that does not respect
`.gitignore` returns their stale copies of every file.

## Setup and commands

`pyproject.toml` sets the minimum gpyreg version, and CI installs gpyreg at
the commit pinned as `GPYREG_PIN` in `.github/workflows/test-matrix.yml`, so
a change that needs a newer gpyreg also moves the pin. PyVBMC requires the
latest gpyreg release: each gpyreg release moves the minimum and the pin to
it, whether or not PyVBMC needs its changes. When the minimum names
a gpyreg release, the pin is that release's tagged commit, so that the
pinned checkout reads the minimum from gpyreg's tags. CI installs PyVBMC
first, which takes gpyreg from PyPI, and the pinned checkout over it last:
an untagged commit before the tag reads lower than the minimum, so that pip
warns of the conflict and CI tests a version the minimum excludes, and
before the release is on PyPI the install of PyVBMC fails. For
development, install gpyreg from a sibling checkout:

```console
git clone https://github.com/acerbilab/gpyreg ../gpyreg
pip install -e "../gpyreg[dev]"
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

After gpyreg is tagged, `git fetch --tags` in the sibling checkout and
reinstall it editable: its setuptools_scm version otherwise stays below the
minimum in `pyproject.toml`, and the next `pip install -e .` installs gpyreg
from PyPI over the checkout. PyVBMC's own version comes from git tags in the
same way, with no fallback: a shallow clone, which holds no tags, builds as
version 0.1.dev1 (the version the machine calibration's provenance then
records), and an exported tarball, which holds no git metadata, fails to
build.

```console
python -m pytest --reruns=5 -x -vv                      # what CI runs
```

Default discovery is limited to the shipped suite under `pyvbmc/testing`
(`pyproject.toml`); the checks under `dev/scripts/` run only when named by
path. The tests of the optional integrations (Torch export, S-VBMC, ArviZ,
PyMC) skip without a word where their extras are not installed, so a green
local run may not have exercised them; CI installs the extras in one cell
(Ubuntu, newest Python).

Formatting is enforced by the pre-commit hooks alone (black at line length
79, isort with the black profile, pycln); no CI job checks it. `pylintrc` is
stock and contradicts black's line length; ignore it.

Docstrings are numpydoc, with `VariationalPosterior` as the reference style;
`docsrc/source/development.rst` gives the commands that build the docs.
Nothing generates the API pages: a new public class or function needs a
hand-written `.rst` under `docsrc/source/api/` and an entry in the toctree
that owns it (`documentation.rst` for a headline page,
`api/classes/classes.rst` or `api/functions/functions.rst` otherwise).
`skills/pyvbmc/SKILL.md`, the skill that the agents of PyVBMC's users load,
links to documentation pages, FAQ section titles and example notebooks by
name, so renaming one of those means updating it.

The notebooks in `examples/` are never executed in CI or in the docs build,
so a change that breaks one goes unnoticed until someone runs it.
`examples/scripts/*.py` are generated from them by `examples/scripts/Makefile`;
regenerate, do not edit.

## Architecture

`VBMC.optimize()` in `pyvbmc/vbmc/vbmc.py` is one long loop of bookkeeping
around three numerical stages, repeated until termination:

1. **Active sampling** (`vbmc/active_sample.py`): choose the iteration's new
   points by maximizing an acquisition function (`acquisition_functions/`)
   built from the current GP and variational posterior. Noisy targets use
   VIQR/IMIQR, whose importance samples come from
   `vbmc/active_importance_sampling.py`. Every evaluation goes through
   `FunctionLogger`.
2. **GP training** (`vbmc/gaussian_process_train.py` → `gpyreg.GP.fit`):
   squared-exponential ARD kernel, negative-quadratic mean, Gaussian noise.
3. **Variational optimization** (`vbmc/variational_optimization.py`): fit
   the Gaussian mixture by minimizing the negative ELCBO. The expected log
   joint under the mixture is analytic (`_gp_log_joint`); the entropy comes
   from `entropy/`.

`whitening/` rotates the space using the variational covariance.

## What spans files

- **PyVBMC reaches into GP internals.** Several modules (the noisy
  acquisitions, active sampling, GP training, the variational optimization,
  the whitening) read `gp.posteriors[s].{hyp, alpha, L, L_chol, sW}` directly
  rather than calling `predict`, and `_gp_log_joint` reimplements gpyreg's
  `quad` to add gradients. A change to the GP representation, here or in
  gpyreg, touches all of them.
- **Hyperparameters are indexed by position.** The SE-ARD kernel is
  hard-wired: several paths index its block of the hyperparameter vector
  directly and fail on any other kernel. The number of noise hyperparameters
  follows the uncertainty level, so code that indexes past the kernel's block
  goes through the `hyperparameter_count()` of the covariance, noise and mean
  functions.
- **Gradients are hand-derived** for the ELBO with respect to `mu, sigma,
  lambd, w` across `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`,
  `entlb_vbmc`, `entmc_vbmc`, and for `vp.pdf`. Changing the variational
  parameterization means updating every one of them, plus the softmax
  Jacobian, which is written out four times (both entropies, `_neg_elcbo`,
  `_gp_log_joint`).
- **Two coordinate spaces.** The algorithm runs in an unbounded transformed
  space; users see the original constrained space. `ParameterTransformer`
  (`__call__` forward, `.inverse()` back) mediates. `VariationalPosterior`
  methods take `orig_flag=True` by default and provide gradients only with
  `orig_flag=False`; `vbmc.x0` is in the transformed space the run was
  constructed with (a warp does not re-express it, so after one the run's
  transformer no longer maps it back), `vbmc.x0_orig` in the caller's. One
  transformer object is shared by `vbmc`, `vp` and `function_logger`, during
  a run and after `load` (tests assert identity).
  No code mutates a transformer after construction (a warp installs a fresh
  copy), which is what makes sharing safe.
- **Shapes are rigid.** VP: `w (1,K)`, `mu (D,K)`, `sigma (1,K)`,
  `lambd (D,1)`. Bounds `(1,D)`, `x0 (n0,D)`.
  `decorators/handle_0D_1D_input.py` promotes 1-D inputs for methods; it
  assumes a `self` first argument and misbehaves on module-level functions.
- **`FunctionLogger`** preallocates its arrays and grows them; `Xn` is the
  index of the last filled row, `X_flag` the boolean mask of live rows.
  Always index through `X_flag`: the rows beyond `Xn` are unwritten (NaN in
  the float arrays). `finalize()` trims.
- **`optim_state`** is a plain dict read and written everywhere (hundreds of
  sites). `IterationHistory` deep-copies everything it records and accepts
  only the keys declared in the list in `VBMC.__init__`. The GP is recorded
  without its posterior factors; `vbmc.get_gp(iteration)` returns a copy with
  the factors recomputed, so read a past GP through it and not from the
  record.
- **Options** are layered: `option_configs/basic_vbmc_options.ini`, then
  `advanced_vbmc_options.ini`, then the `options_path=` file, with the
  `options=` dict winning. `.ini` values are `eval`'d in the namespace of
  `options.py` with `D` bound; the dict is used verbatim. To add an option,
  add a `# description` line followed by `name = <expr>` to the right `.ini`:
  the comment is the user documentation (an option that no module reads is
  also registered in `INERT_OPTIONS`). Options are frozen once validated,
  against assignment and removal; package code writes with
  `options.__setitem__(k, v, force=True)` and removes with
  `options.__delitem__(k, force=True)`.
- **Randomness goes through `numpy.random.Generator` objects.** `VBMC(seed=)`
  creates `vbmc.rng` (`pyvbmc/rng.py: get_rng`) and shares it with `vbmc.vp`;
  `VariationalPosterior.__deepcopy__` shares the generator, so every copy of
  a VP stays on one stream, and a function that receives a `vp` draws from
  `vp.rng`. The generator is handed to everything that draws: the GP fit
  (`train_gp(rng=)` → `gpyreg.GP.fit(rng=)`), the CMA-ES search (the `randn`
  option of `cma`) and the slice sampler of the importance sampling. No draw
  of a run goes through NumPy's global stream, and code that draws takes a
  generator. `seed=None` derives the generator from the global state, so that
  `np.random.seed()` before construction fixes a run (the example notebooks
  rely on this).
- **float64 everywhere, implicitly.** No class declares the dtype of its
  state, and the matrices are borderline singular (hence the Cholesky retry
  ladder in gpyreg), so a narrower dtype anywhere changes results. The dtype
  canary (`pyvbmc/testing/_dtype.py`) walks the state in the oracle tests and
  on a live run in `test_vbmc_seed.py`.
- **Saved objects are pickles of the classes as they are.** `VBMC.load`
  back-fills what older files lack (its `hasattr` and `not in options`
  checks), and rewrites what they hold in a form the current code reads
  otherwise (among them a stored `integer_vars`, `uncertainty_handling`,
  `specify_target_noise` or `search_optimizer` in a form that 1.0.4 took,
  and a `last_warmup` of 0); an attribute or option added to a saved class
  needs the same, and renaming or removing one breaks users' files along
  with the static test pickles. `VariationalPosterior.load` migrates the
  posterior it loads (`_ensure_calibration_state`), and `VBMC.load` and
  `SVBMC.load` apply the same migration to the posteriors their files hold,
  which never pass through `VariationalPosterior.load`: a back-fill of a
  posterior goes into all three.
- **Optional integrations import lazily.** `import pyvbmc` imports none of
  Torch, ArviZ, PyMC and PyTensor: `pyvbmc.SVBMC` and `pyvbmc.PyMCTarget`
  resolve on request (`pyvbmc/__init__.py`), and the posterior exports import
  inside their methods. The PyMC adapter's reaches past PyMC's documented
  interface are guarded by capability in `pyvbmc/pymc/_compat.py`; keep it
  that way. The decisions behind each integration are in its plan under
  `dev/plans/`.
- **`pyvbmc.timer.main_timer`** is a process-wide singleton shared by all
  `VBMC` instances, so concurrent runs in one interpreter are not safe.

## Tests and their traps

- Do not add uncapped `optimize()` runs: the ones in
  `pyvbmc/testing/vbmc/test_vbmc_optimize.py` dominate the runtime of the
  suite. Cap a new run at a few iterations, or share an existing one through
  a module-scoped fixture as `test_vbmc_seed.py` does. Seed explicitly and
  keep statistical tolerances loose; unseeded tests are the reason CI uses
  `--reruns=5 -x`.
- Several subpackages export a function under the name of the module that
  defines it (`pyvbmc.vbmc.active_sample`, for one), so inside the package
  that name is the function. A mock target written as a dotted name through
  it, such as `"pyvbmc.vbmc.active_sample.cma.fmin"`, resolves to the module
  on Python 3.11 and later and to the function on Python 3.10, where it
  raises `ModuleNotFoundError`. Patch the module object
  (`importlib.import_module("pyvbmc.vbmc.active_sample")` with
  `mocker.patch.object`), or the defining module of what is patched
  (`"cma.fmin"`).
- `test_*_save_dynamic` write `.pkl` files into the source tree that the
  matching `load` tests read; a `load` test run alone fails.
- `test_*_save_static.pkl` are pickled instances of the classes and cannot be
  regenerated without rerunning `optimize()`. Load the VBMC one, never save
  it again and never continue its run: it holds the target function,
  PyVBMC's log-joint wrapper and the options whose value is a function
  (`ns_ent`, `k_fun_max` and the like; every saved `VBMC` instance holds
  these) pickled by value, as bytecode of the Python version that wrote the
  file. Pickling such a function makes dill disassemble it, which corrupts
  memory on Python 3.11 (a segmentation fault in that test, in a later
  garbage collection, or when the interpreter exits), and a continued run
  calls the option functions in every iteration. A test that needs a saved
  run with history makes a short run of its own.
- A new test directory needs an `__init__.py`: without one, two test files
  with the same basename collide.

## Numerical gates

A gate is evidence for a change only if it reaches the changed code: a test
that skips without a word, or an oracle whose stored state never enters the
changed branch, is green and proves nothing. The same holds for the package a
gate imports. The package is installed editable from one checkout, and in any
other checkout or worktree a script under `dev/scripts/` imports the
installed checkout's package unless it puts its own repository root on
`sys.path`, because `sys.path[0]` is the script's directory; `python -m
pytest` from that checkout's root is unaffected. Run script gates there with
`PYTHONPATH=<that checkout>` and print `pyvbmc.__file__` once before trusting
the result. A change to gpyreg is a change to PyVBMC's numerics: its gate is
PyVBMC's whole suite run against the gpyreg checkout that holds it
(`PYTHONPATH` naming that checkout, `gpyreg.__file__` printed), beside
gpyreg's own suite, which cannot see PyVBMC's uses of its interface.

- **Gradients.** Finite-difference checks (`pyvbmc/testing/_check_grad.py`)
  cover both entropies, `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`,
  `_soft_bound_loss` and `vp.pdf` (`pyvbmc/testing/**/test_*_grad_fd.py` and
  the entropy tests).
- **MATLAB references.** The reference values derived from MATLAB (under
  `pyvbmc/testing/entropy/`, `variational_posterior/`, `vbmc/` and
  `whitening/`) are the value gate. They cannot be regenerated without MATLAB
  and the VBMC toolbox: never change one to make a test pass. The
  `FIXTURES.md` of each directory says what each array holds and which test
  reads it.
- **Oracles.** `pyvbmc/testing/oracles/` pins the numerics stage by stage:
  each fixture is one algorithm state saved as plain arrays, with the
  reference outputs of each stage computed from it, and
  `pytest pyvbmc/testing/oracles` rebuilds the state and recomputes. A
  failure means the numerics moved, or a dtype did. The committed references
  equal the current numerics on the machine that generated them, so the gate
  for a change that must move nothing is
  `python dev/scripts/make_oracle_fixtures.py --check --exact`. Never
  regenerate the fixtures, replace a reference or loosen a tolerance to make
  a change pass. When a change moves an output on purpose, replace that
  reference alone with one of the generator's targeted modes (`--rebaseline
  ORACLE --reason "..."` and its kin in `--help`), never by rerunning the
  recipes, which moves every snapshot; the reason is recorded in the
  fixture's `.json`. The `make_oracle_fixtures.py` entry of `dev/README.md`
  gives the procedure.
- **Platform-bound oracles.** `active_sample_step`, `gp_fit` and
  `gp_fit_history` run a CMA-ES search or a slice-sampling chain, which turns
  BLAS rounding differences into different decisions. They reproduce only on
  the machine that generated the fixtures and only with BLAS single-threaded,
  so on that machine run the suite with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Elsewhere the
  tests and the generator's `--check` skip them unless
  `PYVBMC_ORACLES_ALL=1`, and the generator's targeted modes refuse to run.
- **Trajectories.** `python dev/scripts/golden_replay.py` checks a change
  step by step against the golden traces. A change that moves the default
  trajectories is assessed on the benchmark suite before it is accepted; the
  golden references are then updated and the old ones preserved
  (`dev/README.md`).
- **S-VBMC.** `pyvbmc/testing/svbmc/fixtures/references.npz` pins a seeded
  short optimization: the gate for changes to the numerics of `pyvbmc/svbmc/`,
  its entropy code in particular.

## Conventions

- **Commits** follow conventional commits. A `Co-Authored-By:` line is fine;
  a `Claude-Session:` trailer is not, even where the session's own
  attribution instructions ask for one.
- **Changelog.** A change that a user can notice is listed in `CHANGELOG.md`
  under `Unreleased` with the work that makes it, in a sentence written for
  users and relative to the last release (a fix to a feature that no release
  has shipped belongs to that feature's entry). A change that can stop a
  script written for the last release, or change what it returns, also has
  one line in the "Upgrading from" list that opens the section, kept in step
  with its entry.
- **Branches.** Work on feature branches; `feat-*` and `dev-*` branches are
  live work. An implementation that is rejected or parked after evaluation
  leaves the working line and is kept for the record on a branch named
  `retain/<topic>`, cut at the last commit that holds it.
- **CI.** A PR to `main` runs the test matrix only when it touches
  `pyvbmc/`, `pyproject.toml` or `setup.py`; a PR that changes anything else,
  the workflows included, runs no tests.
- **Records.** An error in a record, a document or code is corrected in the
  file that holds it; where a record cannot change, the flag goes into the
  record, or as close to it as its format allows. A note somewhere else is
  not a correction. A tracked document points at `dev/scripts/runs/LOCAL.md`
  and never says "this machine".
- **Rounding.** Where MATLAB has `round`, call `round_half_away_from_zero`
  (`pyvbmc/stats/_rounding.py`): Python's `round` and `np.round` send a half
  to the even integer.
- **Heavy computation.** Run one heavy process at a time (the full test
  suite, benchmark and campaign runs): concurrent VBMC runs, each
  multi-threaded, can bring a workstation down. Short gates (the oracles, one
  module's tests, the replay) can run at any time as one process. Long
  campaigns start only on the PI's instruction.
