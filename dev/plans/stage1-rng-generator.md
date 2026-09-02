# Stage 1 — `seed=` and `numpy.random.Generator` threading

Started 2026-09-02. Roadmap: `plans/modernization-roadmap.md`, Stage 1 of
`dev/2026-09-02-modernization-discussion.md` §10, bundled with the §9
one-liners there. Branch `dev-next`.

Replace the ~55 global `np.random.*` call sites in PyVBMC with a `Generator`
owned by the `VBMC` instance, exposed through a `seed=` constructor argument.
The plan called this "the single highest-leverage API change in the
package": it fixes test flakiness at the source, allows parallel runs, and is
the prerequisite for "randomness injected, not drawn" in the Stage 0 oracles
and the Stage 4 port.

## 1. Checklist

Survey
- [x] Inventory `np.random` sites: PyVBMC ≈55, gpyreg 10, cma (`randn`
  option, `NoiseHandler`)
- [x] Decide the gpyreg/cma seam (global state reseeded from `vbmc.rng`)

One-liners (§9), one commit each
- [x] `compute_vargrad` assembly in `_gp_log_joint` — `b51d4bd`
- [x] `rand_int` in `testing/_compare_matlab.py` — `2bf0d1e`
- [x] `pyproject.toml`: `pytest*` → `dev`, `plotly` → `examples` — `b606b2b`
  (then a `test` extra and the CI workflows fixed, see §7)

Generator threading
- [x] `pyvbmc/rng.py`: `get_rng`, `seed_global_from`
- [x] `VBMC(seed=)` → `vbmc.rng`; global reseed at construction and
  snapshot reinstall at `optimize()` start when seeded; snapshot refreshed
  when `optimize()` returns
- [x] `VariationalPosterior(rng=)`, lazy `rng` property for old pickles,
  `__deepcopy__` shares the generator
- [x] Sites replaced: `variational_posterior.py`, `variational_optimization.py`
  (pruning and `_vb_init` via `vp.rng`), `active_sample.py` (incl. cma
  `randn`), `active_importance_sampling.py`, `gaussian_process_train.py`
  (`train_gp(rng=)`), `whitening.py`, `entropy/entmc_vbmc.py` (`rng=`),
  `priors/*` (`sample(n, rng=None)`)
- [x] `random_state` per iteration = generator state + legacy tuple;
  `load(set_random_state=True)` restores both; legacy files warn; history
  VPs of legacy files get the shared generator
- [x] Docstrings; `AGENTS.md` randomness note and extras note
- [x] Tests: `test_vbmc_seed.py` (8); `test_entmc_vbmc` and the MC-entropy
  FD test pass `rng=` for common random numbers; three
  `test_vbmc_active_sample` tests mock `vp._rng` instead of `numpy.random`;
  `test_product_prior` covers a `UserFunction` marginal
- [x] Suite minus `test_vbmc_optimize.py`: 404 passed
- [x] `test_vbmc_optimize.py` + `test_variational_optimization.py` after the
  pruning fix: 17 passed, 0 reruns, 590 s
- [x] Read-only review (two Opus agents); findings fixed or listed in §7
- [x] Full suite `python -m pytest --reruns=5 -x -vv` after the review
  fixes: 414 passed, 1 rerun (`test_vp_optimize_1D_g_mixture`, unseeded),
  11:29 wall
- [x] Committed on `dev-next` (three commits: CI/test extra, the feature,
  the `dev/` restructure)
- [ ] PR `dev-next` → `main`

## 2. API

- `VBMC(..., seed=None)`. Anything `numpy.random.default_rng` accepts: an
  int, a `SeedSequence`, or an existing `Generator` (used as is). The
  generator is `vbmc.rng`.
- `VariationalPosterior(..., rng=None)`, property `vp.rng`. The `VBMC`
  constructor passes its own generator, so `vbmc.vp.rng is vbmc.rng`, and so
  is the `rng` of every VP returned by `optimize()`.
- `Prior.sample(n, rng=None)` on every prior class; `Product` passes the
  generator to its marginals (except a `UserFunction`, whose `sample` is the
  user's own callable taking only `n`); `SciPy` uses
  `rvs(random_state=rng)`, which overrides a `random_state` frozen into the
  distribution.
- `entmc_vbmc(..., rng=None)`, `train_gp(..., rng=None)`: keyword arguments
  with defaults; behaviour unchanged for callers that do not pass one.
- `VBMC.load(..., set_random_state=True)` keeps its name and meaning.

Helper: `pyvbmc/rng.py` with `get_rng(seed)` and `seed_global_from(rng)`,
not exported from `pyvbmc/__init__.py`.

## 3. Design notes

**`seed=None` derives the generator from the legacy global stream** (four
`uint32` draws from `np.random`). Every example notebook does
`np.random.seed(42)` before constructing `VBMC`, and the end-to-end and
resume tests do the same; that contract is kept. The stream itself changes
(PCG64 instead of MT19937 through the legacy API), so any result pinned to
specific draws moves; nothing in the test suite is pinned that way, but the
two `*_overlapping_mixture` entropy tests build their fixture from the
global stream right after constructing a VP (which now consumes four
`uint32` instead of `D*K` normals), so their `theta0` moved.

**Functions that receive a `vp` draw from `vp.rng`; functions that do not
take an explicit `rng`.** The alternative, an `rng` parameter on every
function in the call chain, would have changed a dozen signatures and every
test that calls them for no behavioural difference, because the `vp` handed
to them always carries the instance's generator. `train_gp` has no `vp` and
got the keyword. Each affected docstring says which generator it uses. (The
component-pruning draw lives in `optimize_vp`, not in `update_K` as first
assumed from diff line numbers; the full suite's Rosenbrock run, which
prunes, caught the unbound name that the short smoke runs never reached.
Nothing in the suite asserts on `pruned`, and `test_vbmc_finalboost` mocks
`optimize_vp` out.)

**`VariationalPosterior.__deepcopy__` shares the generator.** VPs are
deep-copied constantly (`_vb_init` builds every sieve candidate with
`copy.deepcopy(vp)`, `IterationHistory` copies the VP each iteration). A
copied `Generator` would fork the stream: every candidate would then draw
the *same* entropy samples, and the live VP and the history VPs would
diverge from `vbmc.rng`. Sharing keeps one stream per instance. Everything
else is still deep-copied. Pickling the whole object graph in one `dill`
call keeps the sharing across `save`/`load`. Old pickles have no `_rng`;
the `rng` property creates one lazily, and `load` attaches the instance
generator to the live VP and to every VP in the history (a history VP is
what `determine_best_vp` and a continued `optimize()` hand back).

**gpyreg and cma still use the global state; this is the one seam.**
gpyreg draws from `np.random` in `f_min_fill`, `SliceSampler` and
`GP.random_function`; cma's `NoiseHandler` draws `np.random.rand()`. Two
measures make a seeded run reproducible anyway:

1. CMA-ES gets `randn=lambda *shape: rng.standard_normal(shape)` (called as
   `randn(lam, N)` and `randn(1, n)`), so the population comes from our
   generator and cma never seeds the global state (it only does so when
   `randn` is the default). Restart/BIPOP draws are off by default.
2. If `seed` is given, `__init__` reseeds the global state from the
   generator, and `optimize()` reinstalls the instance's own latest snapshot
   (`self.random_state["legacy"]`) before the loop and refreshes the
   snapshot when it returns. The reinstall is what makes the run immune to
   global draws made between construction and `optimize()`, or between two
   `optimize()` calls of different instances; the first version without it
   failed exactly that check. Only the global half is protected: draws from
   `vbmc.rng` before `optimize()` (`vbmc.vp.sample()`, a shared generator)
   change the run, as documented on `seed`.

The cost is a documented side effect on the user's global stream when
`seed` is given (also when an existing `Generator` is passed, which then
also loses one draw). With `seed=None` the global state is never written,
only read once. The seam disappears once gpyreg accepts a generator (§8).

**Save format.** The per-iteration `random_state` becomes
`{"generator": rng.bit_generator.state, "legacy": np.random.get_state()}`;
the legacy tuple is needed as long as the seam exists.
`load(set_random_state=True)` restores both and updates the instance
snapshot. `load(set_random_state=False)` marks the instance as not seeded,
so a later `optimize()` does not reinstall the save-time snapshot either
(it would be the end-of-run state, wrong for `iteration=k`). A file saved
before this change holds a bare tuple: `load` restores the global state,
derives a fresh generator from it, puts the global state back where it was
saved, and logs a warning that the resumed run will not replay the
original exactly. Loading such a file *without* the flag gives it a
generator from OS entropy so that `load` stays free of side effects on the
global state (the static load test asserts exactly that). This resolves
open question 1 of the modernization devlog (§13).

## 4. Reproducibility checks (smoke script, then `test_vbmc_seed.py`)

- Same `seed` → identical `vp.mu` jitter and generator state at construction;
  `seed=None` after `np.random.seed(5)` twice → identical; `seed=None`
  consumes exactly the four derivation draws and does not reseed.
- Two 2-iteration runs at `D=2` with `seed=42`, with `np.random.seed(...)`
  calls between construction and `optimize()` and before the second
  construction: identical `elbo`, `elbo_sd`, `vp.mu`, `vp.w` and evaluated
  points, bit for bit within one process/BLAS configuration. This covers
  gpyreg's slice sampler, cma and the noise handler.
- Run 4 iterations; separately run 2, `save`, `load(set_random_state=True,
  new_options={"max_iter": 4})`, run 2 more: identical `elbo`.
- Loading the committed pre-change fixture `test_vbmc_save_static.pkl` with
  and without `set_random_state` works, warns in the former case, and the
  loaded VP shares the instance generator.

## 5. One-liners fixed on the way (modernization devlog §9)

- `_gp_log_joint` `compute_vargrad` block: `sigma_vargrad` scaled by `sigma`
  (was by itself), `mu_vargrad` reshaped in Fortran order like the value
  path, `dvarG` assembled from `vargrad_list` (was `grad_list`). Still
  unreachable from `optimize()`, and the `*_vargrad` accumulators are
  allocated but never filled: the block is a stub, not working code.
- `testing/_compare_matlab.py: rand_int` referenced an undefined `lo`
  (`res = 1`; MATLAB `randi(hi)` is uniform on `1..hi`). Unused.
- `pyproject.toml`: `pytest`, `pytest-mock`, `pytest-rerunfailures` moved
  out of the runtime dependencies into a `test` extra (also in `dev`);
  `plotly` (only example notebook 2 imports it) into a new `examples`
  extra, also in `dev`. The test workflows now install `.[test]` (they
  installed bare `-e .`, so the first version of this change would have
  broken CI); `installation.rst` tells users to install `[test]` before
  running the suite and `[examples]` for notebook 2.

Not fixed, deliberately: `Product._generic` still picks a marginal class
with `np.random.choice` (test fixture code, outside `sample`; the existing
seed-driven tests depend on which class it picks).

## 6. What the tests had to change, and why

- `test_entmc_vbmc.py` reseeded the global state inside its wrapper to give
  the value and gradient evaluations common random numbers; the entropy now
  draws from `vp.rng`, so the wrapper passes `rng=42` instead. The same for
  the Monte Carlo entropy check in `test_variational_optimization_grad_fd.py`
  (`VariationalPosterior(D, K, rng=7)` per call).
- `test_entmc_vbmc_single_gaussian` failed at `rtol = 0.01` on the new
  stream: for `K = 1` the numerical gradient of the sample-based entropy is
  exact while the reparameterization estimator drops the zero-mean score
  term and so carries Monte Carlo error of relative order `sqrt(4 / Ns)`
  per `lambd` coordinate (antithetic pairs halve the effective sample
  count): 0.6% at `Ns = 1e5`, observed 0.8–1.8%, 1.3–2.9 standard
  deviations; the legacy seed 42 happened to land closer. `rtol = 0.03`
  (about five standard deviations) at the same `Ns`; `Ns = 1e6` was tried
  first and costs ten times more per call. Not a bug in the estimator.
- Three `test_vbmc_active_sample` tests mocked `numpy.random.rand` /
  `standard_normal`; they now mock `vbmc.vp._rng` with a `Mock` whose
  `random` / `standard_normal` return the fixed arrays. The Mock answers any
  other method with a Mock, so an added draw on those paths would fail as
  a numpy `TypeError` rather than a clear message.
- New `test_vbmc_seed.py` (8 tests) covers §4; its two short `optimize()`
  runs take ~7 s and are noted in `AGENTS.md` as deliberate.
- Pre-existing and left alone: `test_entmc_vbmc_nonoverlapping_mixture`
  calls `check_grad(...)` without `assert`.

## 7. Review findings (two read-only Opus agents) and what was done

Fixed: CI workflows installing without extras (§5); history VPs of old
pickles without the shared generator; `set_random_state=False` on a seeded
pickle still reinstalling the save-time snapshot; no snapshot refresh after
the post-loop work; legacy `load` leaving the global state four draws past
the saved one; `Product.sample` passing `rng` to a `UserFunction` sampler;
`test_seed_none_does_not_reseed_global_state` not testing its name;
`full_repr` printing the generator; duplicated `rng` documentation on the
VP; wrong claims in docstrings (`NoiseHandler` uses `rand`, not `randint`;
"keyword-only"); `Ns = 1e6` cost.

Noted, not changed: `active_importance_sampling.py` draws a single MCMC
start with `choice(..., replace=False)` where the comment and a commented
line suggest `Walkers` starts were intended (faithful to the port);
`README.md`/`installation.rst` say `pip install pyvbmc`, which no longer
brings `plotly` for notebook 2 (documented in `installation.rst` instead).

## 8. Follow-ups (not in this PR)

1. gpyreg: `GP.fit(rng=)`, `SliceSampler(rng=)`, `f_min_fill(rng=)`,
   `GP.random_function(rng=)` with `None` falling back to the global draws
   (PyBADS unaffected). Then delete `seed_global_from`, the reseed in
   `VBMC.__init__`, the reinstall in `optimize()`, and the `"legacy"` half
   of `random_state`; bump the minimum gpyreg version.
2. Example notebooks: `seed=` instead of `np.random.seed(42)` when they are
   next regenerated; mention the `examples` extra in notebook 2.
3. Tests that still steer VBMC through `np.random.seed` (six end-to-end
   runs, resume test) keep working via the legacy derivation; switching
   them to `seed=` is optional. The unseeded tests behind `--reruns=5` are a
   separate cleanup, as is the missing `assert` in
   `test_entmc_vbmc_nonoverlapping_mixture`.
4. `Product._generic` still uses `np.random.choice` (test fixture code).

## 9. Commits and test results

- `b51d4bd` fix(vbmc): `compute_vargrad` assembly.
- `2bf0d1e` fix(testing): `rand_int` lower bound.
- `b606b2b` chore: dependencies (extras completed in the Stage 1 commit).
- Stage 1 commit: `feat(vbmc): seed= and Generator threading` (see git log).

Suite minus `test_vbmc_optimize.py`: 404 passed. `test_vbmc_optimize.py`
and `test_variational_optimization.py` after the pruning fix: 17 passed,
0 reruns, 590 s. Full suite after the review fixes, as CI runs it: **414
passed, 1 rerun** (`test_vp_optimize_1D_g_mixture`, an unseeded test that
also reran in the pre-change baseline), 689 s wall (the 2026-09-02
baseline was 389 passed in 1087 s; the difference is machine load, not the
change).
