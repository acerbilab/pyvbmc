# Stage 0: the dtype canary

Created: 2026-09-06 10:56. Status: **done 2026-09-06 midday; committed
to `dev-next` as `282e0ed`** (tracker at the end).
This file is the plan and the worklog. Roadmap Stage 0, last bullet, and
pickup point 3f (`plans/modernization-roadmap.md`); rationale in
`dev/2026-09-02-modernization-discussion.md` §2 ("float64 is
non-negotiable"), §7 (the float32 defaults of PyTorch and JAX) and §10
(Stage 0: "a dtype-assertion canary so a float32 regression fails
loudly"). Replaces a draft of the same morning that was discarded because
its verification section recorded gates that had not been run.

## Summary

PyVBMC computes in float64 because NumPy's defaults give it; no class
declares the dtype of its state, and the tests see the property only
indirectly. The oracle tests (`pyvbmc/testing/oracles/`) compare values
after casting every output to float, so a float32 result shows up, if at
all, as a value mismatch in the tolerance classes tight enough to see one
(1e-10 on the densities, the entropies and the transformer) and passes
unnoticed in the loose classes (1e-4 on GP prediction, 1e-3 on the
acquisitions and on every variance, 2e-2 on the marginal-likelihood
gradient). A float32 array whose values are exactly representable, or one
that a later cast widens before it reaches an output, passes everywhere.
The package widens at several boundaries (the transformer's
`astype(float)`, the `dtype=float` copies of the VP blocks at the head of
`_gp_log_joint` and `entmc_vbmc`, the float64 preallocation of the
function logger), so a check on outputs alone is systematically blind to
precision lost in a stored array.

The canary makes the property explicit, in three tests-only pieces plus
one recorded gap:

1. **Raw oracle outputs.** `test_oracle` checks the dtype of every output
   *before* the harness cast: float64, except three variance placeholders
   that are the integer literal `0` when their branch does not run. It
   also walks the rebuilt objects the oracle worked on, scratch arrays
   included. This covers every oracle wherever it runs, the
   platform-bound ones included, at no added cost.
2. **A manifest of the load-bearing arrays** (the VP blocks, the GP data
   and posterior factors, the logger columns, the transformer arrays),
   asserted present and float64 on the eight rebuilt oracle states
   (together with the candidate set) and on one live run (together with
   the optimizer state's bounds and initial points).
3. **A walk over a live run.** Every dtype-carrying leaf reachable from
   the `VBMC` instance (the history, `optim_state`, the GP, the
   transformer) and from its results must be float64. The walk enters
   containers and `pyvbmc`/`gpyreg` objects only, treats any object with
   a `dtype` as a leaf so a tensor of another backend is an offender
   rather than something stepped over, and the instance walk asserts a
   minimum number of leaves so a broken descent cannot pass green. It
   rides on a short run that `test_vbmc_seed.py` already performs; no
   `optimize()` run is added.
4. **The one live leak, recorded.** `VBMC.__init__` widens integer inputs
   to float64 and leaves float32 and float16 inputs as they are: the
   bounds and `x0` keep their dtype in `optim_state` and in the
   transformer, and every downstream array is float64 holding rounded
   values. A strict expected failure pins this until the boundary cast
   lands (§Follow-ups).

The walker and the manifest live in `pyvbmc/testing/_dtype.py`, tested on
synthetic graphs (`pyvbmc/testing/testing/test_dtype.py`): a nested
float32, complex arrays and scalars, a foreign dtype, a boundary object, a
cycle, an empty walk, the manifest's failure modes.

## Scope

In: `pyvbmc/testing/_dtype.py`, `pyvbmc/testing/testing/test_dtype.py`,
`pyvbmc/testing/__init__.py` (the export),
`pyvbmc/testing/oracles/_oracles.py` (`cast_outputs`),
`pyvbmc/testing/oracles/test_oracles.py`,
`pyvbmc/testing/vbmc/test_vbmc_seed.py`,
`pyvbmc/testing/vbmc/test_vbmc_init.py` (the new test; the pinned black
also reformats two pre-existing loop headers of that file). Records: this
file, `dev/README.md` (plans index), `AGENTS.md` (a thread note after the
test commands, the float64 bullet, the seed-module sentence, the
oracle-package sentence), the roadmap (Stage 0 bullet, pickup points 3f
and 6), `dev/TODO.md`, devlog §9 (the findings below).

Out: any production change. The boundary cast (§Follow-ups) is a no-op on
float64 inputs and would leave every golden trace and oracle reference
where it is; it still waits, because the reference extension of pickup
3f is valid only on code identical to the reference's, and a tests-only
change keeps that trivially true. `active_sample_step` is covered where
it runs (piece 1) and not re-run elsewhere.

## Findings the plan rests on

Verified 2026-09-06 on `353118a` (tree clean; the project venv, NumPy
2.5.2, gpyreg from the sibling checkout), by reading the code and by
running the probes named. Leaf counts in this section come from a probe
walker that enters containers and object arrays and visits each object
once; the canary's own walker counts differently (§Results).

- **What the oracles catch.** The cast is in `Oracle.__call__`
  (`_oracles.py:122`) and again in `compare` (`:614`, `:618`). Tolerance
  classes (module docstring and decorators): densities, entropies,
  transformer, `neg_elcbo`'s `H` and `theta` at 1e-10/1e-12; GP-solve
  outputs at 1e-6; predictive mean 1e-4 and variance 1e-3; acquisitions
  1e-5 to 1e-3; variance-type keys 1e-3; `gp_nlZ` gradients 2e-2.
  Float32 rounding is 6e-8 relative, so the first class fails on value
  and the rest do not.
- **Casts in the package** (grep `dtype=|astype\(` outside `testing/`):
  `vbmc.py:450-468` widens integer `x0` and bounds only;
  `parameter_transformer.py:151, 192, 239` `astype(float)` on the inputs
  of the forward, inverse and Jacobian paths;
  `variational_optimization.py:1397-1401` and `entmc_vbmc.py:77-84` copy
  the VP blocks with `dtype=float`; `active_sample.py:374` the CMA-ES
  batch; `function_logger.py:54-57` preallocates the float columns with
  `np.full(..., np.nan)` (the evaluation counts at `:59` and the mask at
  `:66` are integer and bool by design); `kde_1d.py` explicit float64.
  Everything else relies on defaults.
- **Float32 and float16 inputs leak.** Constructing `VBMC` with float32
  (float16) `x0` and bounds: `optim_state["lb_orig"]`, `["ub_orig"]`,
  `["plb_orig"]`, `["pub_orig"]` and `["cache"]["x_orig"]` are float32
  (float16), and so are `parameter_transformer.lb_orig` and `.ub_orig`;
  `pt.mu`, `pt.delta`, the transform outputs, the VP blocks and the
  logger arrays are float64. `test_vbmc_init.py::test_init_integer_input`
  asserts float64 for the integer case only.
- **Existing dtype assertions in the suite**: that test, and
  `test_iteration_history.py:168` (the object dtype of a history slot).
  Nothing else.
- **Rebuilt states.** Over the six rebuilt objects (`pt`, `vp`, `gp`,
  `logger`, `optim_state`, `cand`) of the eight states: 732 float64
  leaves, 74 bool, 16 int64. The non-float ones are
  `gp.posteriors[*].L_chol` (57 NumPy bool scalars, all the same
  singleton object, so a walk that visits each object once reports 8),
  `logger.X_flag`, `logger.n_evals`, `optim_state["N_eff"]` (a `(1,)`
  int64 from `active_sample.py:255`), `optim_state["integer_vars"]`, and
  `vp.stats["stable"]` on the one fixture where it is an array. Load and
  build, 0.11 s. Most of what a state walk sees is the fixture's own
  dtype; the GP posterior factors are recomputed by `build_gp`. Each
  fixture also stores 53 reference arrays, float64 by construction, which
  `build_state` returns under `ref` beside the objects.
- **Live run** (`max_iter=2`, `D=2`, `do_final_boost=False`, seed 0, the
  shape of `test_vbmc_seed.py`'s runs), the instance and its results: 316
  float64 leaves, 34 bool, 12 int64. The non-float ones are the above
  plus `results["best_iter"]`, `optim_state["n_eff"]` and
  `["cache_active"]`, `options["fun_eval_start"]`, the same masks in
  every history entry, and gpyreg's six `_prior_cache["*_idx"]` masks.
  2.1 s. `vbmc.logger` is a process-wide `logging.Logger` whose instance
  `__dict__` holds `manager` (every logger of the interpreter) and
  `parent` (the root logger and its handlers): a walk that entered
  arbitrary objects would reach them.
- **Raw oracle outputs** over the 14 non-platform-bound oracles on their
  98 applicable calls: 376 float64, 10 int64. The integers are
  `neg_elcbo`'s `varH` on every snapshot
  (`variational_optimization.py:1242`, `varH = 0`, never reassigned) and
  `var_ss` / `varG_ss` on the single-sample snapshot (`:1605`,
  `var_ss = 0`, reassigned at `:1612` only when `Ns > 1`, and returned
  by `_neg_elcbo` under the name `varG_ss`); `varF = 0` at `:1246` is
  the same pattern on the no-variance path, which no oracle returns.
  `gp_nlZ` casts its four outputs inside the oracle (`:516-519`) and
  `gp_fit` two of three (`:581-582`). 1.85 s for the 98 calls; `gp_fit`
  1.7 s over the eight snapshots; `active_sample_step` 10.3 s over its
  seven.
- **Docstrings.** `optimize_vp` documents `var_ss : int` (`:125`);
  `_gp_log_joint`, which produces it, documents `var_ss : float`
  (`:1357`).
- **`active_sample_step` needs single-threaded BLAS here.** On this
  machine it fails on 4 of 7 snapshots under default threading (`X_new`
  off by 3e-4 on `halfnormal_D2_bounded`) and passes 7 of 7 with
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, the setting
  its references were generated with; every local full-suite run since
  2026-09-04, the green run of this morning included, set the three
  variables. AGENTS.md's test command did not say so.
- **Placement precedents.** `pyvbmc/testing/vbmc/test_gp_records.py:17`
  imports the oracle state helpers from outside the oracle package;
  `pyvbmc/testing/__init__.py` star-exports `_check_grad` and
  `_compare_matlab`, tested under `pyvbmc/testing/testing/`.
  `test_vbmc_seed.py` holds the suite's three short runs behind one
  helper (`_make_vbmc`). A new test module needs no `MANIFEST.in` line
  (it lists fixture data only; tracked `.py` files reach the sdist
  through setuptools_scm).

## Design

`pyvbmc/testing/_dtype.py`, exported through `pyvbmc.testing`:

- `iter_dtype_leaves(obj, path)`: breadth first over a queue; each object
  visited once (an `id` set plus a list of references, so an id cannot be
  recycled during the walk). A NumPy scalar yields its dtype and is
  tested before the atoms, because `np.float64` and `np.complex128` are
  subclasses of `float` and `complex`. Atoms (`str`, `bytes`, `int`,
  `float`, `bool`, `complex`, `None`, classes, `np.dtype`) end a branch.
  A non-object `ndarray` yields its dtype; an object array is entered
  element by element; `dict`, `list`, `tuple` and `set` are entered; any
  other object with a `dtype` attribute yields that attribute as a leaf
  (a torch or JAX tensor, a pandas column) and is not entered; an
  instance whose class is defined under `pyvbmc` or `gpyreg` is entered
  through its `__dict__`; everything else is a boundary. No depth cap.
- `non_float64_leaves(obj, path)` returns the offenders (a dtype that is
  not a NumPy dtype; kind `c`; kind `f` other than float64) and the leaf
  count. `assert_float64(obj, path, min_leaves)` fails if fewer leaves
  than `min_leaves` were found or any offender exists, naming each path
  and dtype.
- `load_bearing_arrays(vp, gp, logger, pt)` returns `{name: array}` for
  `vp.w, eta, mu, sigma, lambd`; `gp.X, y`, `gp.s2` when set, every
  `posteriors[s].hyp, alpha, sW, L` (a GP without posteriors, or a lean
  history record whose factors are `None`, is reported as missing
  arrays); `logger.X_orig, y_orig, X, y`, `S` when noisy; `pt.lb_orig,
  ub_orig, mu, delta`, `R_mat` and `scale` when set. A renamed attribute
  raises. `assert_manifest_float64(arrays)` requires each value to be an
  `ndarray` of float64.

`test_oracles.py`: `test_oracle` calls `orc.fn` for the raw outputs,
checks them (float64, except `INTEGER_PLACEHOLDERS = {varH, var_ss,
varG_ss}` when they are a Python `int`), walks the rebuilt objects and
the candidate set (`STATE_KEYS`; the stored references and the metadata
are the fixture's own and are left out, so the leaf floor counts what the
oracle worked on), then compares `cast_outputs(raw)` as before.
`test_rebuilt_state_arrays_are_float64` applies the manifest and the
candidate set to every snapshot. `_oracles.py` gains `cast_outputs`,
which `Oracle.__call__` uses, so the generator is unchanged.

`test_vbmc_seed.py`: a module-scoped fixture `seeded_run` performs one
of the module's three runs (seed 42) inside its own snapshot of the
global random state, since a module-scoped fixture runs before the
per-test snapshot; `test_seed_fixes_optimization` compares it with a
second run whose construction and optimization are each preceded by a
reseed of the global state; `test_seeded_run_state_is_float64` walks the
instance with a floor and the results without one (they hold a handful
of scalars), and applies the manifest to the live objects and to the
eight bound arrays and `cache["x_orig"]` of `optim_state`.

`test_vbmc_init.py`: `test_init_narrow_float_input`, parametrized over
float32 and float16, asserting float64 on the five arrays of the integer
test plus the transformer's bounds, marked `xfail(strict=True)` so that
the boundary cast, when it lands, turns it into a required pass.

Leaf floors are half the smallest count the canary's walker measured
(§Results), so a refactor that moves a large piece of state out of the
walked roots fails while ordinary growth or shrinkage does not.

## Steps

- [x] Write `_dtype.py`, its tests, the harness change, the three
      test-module changes
- [x] Measure leaf counts on the rebuilt states and the live run; set the
      floors
- [x] `pre-commit run --files` on the touched files (black 23.3.0, isort,
      pycln, whitespace hooks)
- [x] Negative controls on real state (scratch script, not committed): a
      float32 `vp.mu` and a float32 `gp.posteriors[0].alpha` injected
      into a rebuilt state fail the walk naming the path; a float32
      `vp.sigma` fails the manifest; a deleted attribute fails the
      manifest with its name
- [x] `pytest pyvbmc/testing/testing/test_dtype.py pyvbmc/testing/oracles
      pyvbmc/testing/vbmc/test_vbmc_seed.py
      pyvbmc/testing/vbmc/test_vbmc_init.py`, BLAS single-threaded
- [x] Full suite, BLAS single-threaded, one process, logged under
      `dev/scripts/runs/`
- [x] Read-only review by two Opus agents (the doublecheck): code and
      records separately; findings addressed (§Review)
- [x] Re-measure the floors and repeat the targeted tests and the full
      suite after the review's code changes
- [x] `python dev/scripts/golden_replay.py` (vacuous for a tests-only
      change; run once as the record: 0 flagged of 5, all `identical`)
- [x] Records (§Scope)
- [x] Commit and push to `dev-next` (2026-09-06, on `/handoff`; tests
      only, the two new files added)

## Verification

Filled in from the runs (§Results); nothing here is asserted before its
gate has run.

- [x] `test_dtype.py` green: each synthetic offender named, the boundary
      object not entered, the cycle terminates, the empty walk fails on
      the floor
- [x] Negative controls on real state as listed in §Steps
- [x] Oracle package green single-threaded (`active_sample_step` and
      `gp_fit` included on this machine)
- [x] Seed and init modules green; the two narrow-float cases report
      `xfail`
- [x] Full suite green (`--reruns=5 -x`), no reruns caused by the new
      tests (first run; the second, after the review's changes, in
      §Results)
- [x] `golden_replay.py` reports `identical` (0 flagged of 5 configs)

## Decisions

1. **Check dtypes where the oracles already run, not in a parallel
   sweep.** The raw-output check inside `test_oracle` covers every oracle
   on every applicable snapshot, the platform-bound ones where they run,
   at no added time; a separate sweep would recompute the same 98 calls
   and re-run `gp_fit` for outputs that oracle casts itself.
2. **No new `optimize()` run.** AGENTS.md's rule is flat and about
   runtime; the live check rides on a run the seed module already
   performs, through a module-scoped fixture.
3. **Stored arrays, not only outputs.** The manifest exists because the
   package widens at its boundaries: a float32 stored array produces
   float64 outputs with rounded values, which an output check cannot see.
4. **Bounded descent, no allowlist of paths.** The walk enters
   `pyvbmc`/`gpyreg` objects and containers only, so every attribute of a
   PyVBMC class is covered without registration, and third-party objects
   (loggers, distributions, figures) cannot fail it for reasons of their
   own. Non-float dtypes (bool masks, integer counts) pass by kind; a
   float that turned into an integer is the manifest's job.
5. **A foreign dtype is an offender.** Anything with a `dtype` that is
   not NumPy's fails the walk, in float64 too. A tensor backend has to
   teach the walker its dtypes deliberately, which is where that decision
   should be recorded.
6. **Leaf floors instead of a depth cap.** A cap prunes silently; a floor
   turns a broken descent into a failure. The floor of the oracle test
   counts the rebuilt objects only: the 53 stored reference arrays of a
   fixture would satisfy a floor on their own and hide a walk that never
   entered the objects.
7. **The float32-input leak is pinned as a strict expected failure, not
   fixed here.** The cast is a production change; the test flips to a
   required pass when it lands.
8. **Placement.** Helpers beside `_check_grad.py`; the oracle-package
   checks in `test_oracles.py` because they reuse its fixture and its
   per-oracle loop, with the package's stated contract widened to "the
   numerics or a dtype changed"; the live check in the seed module
   because that is where the run is.
9. **NumPy scalars count per occurrence.** They are yielded before the
   identity guard, so the bool singleton behind every `L_chol` is a leaf
   each time it appears, and a `np.float64` or `np.complex128` scalar is
   seen although it is a Python `float` or `complex`.

## Risks

- The manifest names attributes; a rename fails it with the name, which
  is intended. An accessor that keeps resolving while the numerics move
  to another store would not be noticed by the manifest; the walk still
  covers the new store if it hangs off a PyVBMC object.
- The leaf floors are set from one machine's counts; they count objects,
  not arithmetic, so they do not vary across platforms. They change when
  state is added or removed; the message says which floor and which
  count.
- The walk inside `test_oracle` runs once per oracle per snapshot (113
  walks on this machine: 98 non-platform-bound calls, 7 of
  `active_sample_step`, 8 of `gp_fit`); measured cost in §Results.
- `xfail(strict=True)` under `--reruns=5 -x`: an xfail is not a failure
  and is not rerun.
- A `pyvbmc` class that acquires a `dtype` attribute would be treated as
  a leaf and its subtree would leave the walk; only the floor would
  notice. No such attribute exists.

## Review

Two read-only Opus reviewers, one on the code and one on the plan and
records, after the first full suite. Code findings, all addressed: the
state-walk floor counted the 53 stored reference arrays of each fixture
(now `STATE_KEYS`, Decision 6); `np.float64` and `np.complex128` scalars
were swallowed by the atom test (Decision 9, unit test extended); the
manifest raised a bare `TypeError` on a GP without posteriors (now a
named missing array); the shared run executed outside the per-test
global-state snapshot (the fixture takes its own); the docstring of the
seed-reproducibility test described the run it replaced. Records
findings, all addressed: dangling references to an empty §Results, the
manifest's reach misstated between rebuilt and live state, an AGENTS.md
edit missing from §Scope, two leaf-count sets presented as one, the
thread requirement overstated for `gp_fit`, a "still" describing the
edit, an orphan line and a previous-version reference in the roadmap, an
over-long line in `dev/TODO.md` and its stale header, §Results placed
after §Follow-ups, "130 walks" for 113. The two reformatted loop headers
in `test_vbmc_init.py` were checked against the pinned hook: black
23.3.0 re-applies them, so they stay and §Scope records them.

## Results

- **Negative controls on real state**: a float32 `vp.mu` and a float32
  `gp.posteriors[0].alpha` injected into the rebuilt `cigar_D4_boosted`
  state are the two offenders the walk reports, by path; a float32
  `vp.sigma` fails the manifest as `vp.sigma <float32>`; a deleted
  `vp.lambd` raises `AttributeError` naming it.
- **Leaf counts of the canary's walker** (after the review): over
  `STATE_KEYS`, 62 leaves on `normal_D2_singlesample` and 97–110 on the
  other seven rebuilt states, 0.3–0.6 ms per walk; 394 on the seed-42
  live instance (0.8 ms) and 5 on its results. Floors 31 and 197. (Before
  the review the walk included `ref` and `meta` and skipped NumPy float
  scalars: 111–150 and 319.)
- **Targeted modules**, BLAS single-threaded, after the review's changes:
  180 passed, 15 skipped, 2 xfailed. The oracle package alone is about
  20 s including `active_sample_step` (7) and `gp_fit` (8) on this
  machine; the seed module 6 s for its three runs.
- **Full suite, first run** (11:10–11:16, `runs/pytest_full_canary_
  1788682211.log`, BLAS single-threaded, code before the review's
  changes): 818 passed, 33 skipped, 2 xfailed, 0 reruns, 6 min 26 s,
  exit 0; the morning baseline on `353118a` had 810 passed, 33 skipped.
- **Full suite, second run** (11:38–11:44, `runs/pytest_full_canary_
  final_1788683882.log`, after the review's changes, BLAS
  single-threaded): 818 passed, 33 skipped, 2 xfailed, 0 reruns, 6 min
  9 s, exit 0 (the same totals as the first run).
- **Replay** (11:44–11:47, `golden_replay.py`, defaults,
  `runs/replay_canary_1788683882.log`): 0 flagged of 5 configs, every
  config `identical`, exit 0. A tests-only change moves no production
  code, so the reference precondition of pickup 3f is intact.

## Follow-ups

- **Widen float32 and float16 `x0` and bounds in `VBMC.__init__`**
  (`vbmc.py:450-468`, the integer branch's pattern: warn and
  `astype(np.float64)`), after the reference extension of pickup 3f. A
  no-op on float64 inputs; then drop the `xfail` marker.
- The single-thread requirement of `active_sample_step` on the
  generating machine: documented in AGENTS.md by this plan; alternatively
  pin the thread count inside `test_oracles.py` for `PLATFORM_BOUND` with
  `threadpoolctl` (not a dependency today).
- The three integer placeholders and the `var_ss` docstring conflict:
  devlog §9 records them; a one-line fix (`0.0`, and one docstring) when
  the numerics are next open.
- A tensor backend (Stage 4) must extend `iter_dtype_leaves` with its
  dtype mapping and decide what "float64" means for it (Decision 5).
- The three tests of `test_oracles.py` that rebuild all eight states
  could share a module-scoped `states` fixture (0.1 s each today).

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-06, from the run logs, the
modification times of the files and the commits; a step without a time
left no record of its own, and the steps listed before the first full
suite ran before it.

- [x] Verification of the discarded draft's claims and of the four Opus
      reviews of it — 10:27–10:56 (§Findings)
- [x] Plan written — 10:56–11:05
- [x] Code in place — by 11:10 (`_oracles.py` last edited 11:07; the
      first full suite started on it at 11:10)
- [x] Floors measured and set
- [x] Formatting (pre-commit: all hooks pass)
- [x] Negative controls on real state
- [x] Targeted tests — 180 passed, 15 skipped, 2 xfailed, 19.6 s
- [x] Full suite, first run — 11:10–11:16: 818 passed, 33 skipped, 2
      xfailed, 0 reruns, exit 0
- [x] Records — 11:10 (devlog §9) to 11:33 (AGENTS.md, dev/README.md,
      roadmap, after the review); TODO with the commits
- [x] Doublecheck — after the first full suite, done by 11:33; two
      read-only Opus reviewers (§Review)
- [x] Review changes — 11:33–11:35, after the PI's pause; floors
      re-measured (62–110; 394), targeted tests green
- [x] Full suite, second run — 11:38–11:44: 818 passed, 33 skipped, 2
      xfailed, 0 reruns, exit 0
- [x] Replay — 11:44–11:47: 0 flagged of 5, every config identical
- [x] Commit and push to `dev-next` — 11:53, `282e0ed` (CI smoke green
      on the push, run 34023111174); `dev/TODO.md` updated in `78431fd`,
      11:55
