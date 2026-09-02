# 2026-09-02 — Dev environment, measured profile, first Stage 0 gradient checks

**Status:** steps 1–3 of the immediate next steps in
`2026-09-02-modernization-discussion.md` §14 are done. One latent bug found
and fixed. Nothing committed yet at the time of writing (see §7).

Companion to the modernization devlog: this note records *measurements*;
the plan and its rationale stay in the other file, which is updated in place
where the measurements contradict it.

---

## 1. Dev environment

- Plain `venv` (`.venv/`, gitignored) on the system Python 3.12.6, not conda:
  mirrors CI, and CPU-only torch can be pointed at the PyTorch CPU index
  later without conda. No torch installed yet; nothing before Stage 4 needs it.
- `gpyreg` cloned at `../gpyreg` (commit `236ddd7`, v1.0.4 + 1) and installed
  editable with `[dev]`; PyVBMC installed editable with `[dev]`; pre-commit
  hook installed.
- Resolved versions: numpy 2.5.2, scipy 1.18.1 (scipy-openblas), cma 4.4.4,
  matplotlib 3.11.1, pytest 9.1.1, black 26.5.1 in the venv (the pre-commit
  hook pins black 23.3.0, which is what actually formats commits).
- Machine for all timings below: Windows 11 laptop, Intel Core Ultra
  (family 6 model 170), 22 logical CPUs, single-process runs.

## 2. Baseline test suite (before any code change)

`python -m pytest --reruns=5 -x -vv` (what CI runs):

| | |
|---|---|
| Result | 389 passed, 0 reruns, 0 failures |
| Wall time | 1086.8 s (18:06) |
| Six `optimize()` runs in `test_vbmc_optimize.py` | 216 + 212 + 133 + 128 + 89 + 57 + 47 + 29 s ≈ 15 min |
| Next slowest | `test_entmc_vbmc_nonoverlapping_mixture` 47.6 s, `test_vp_optimize_1D_g_mixture` 45.7 s |

Zero reruns in one local run does not contradict "known flaky"; it is one sample.

## 3. Profiling method

`dev/scripts/profile_run.py` (new; see its docstring). Cheap synthetic targets
with known `ln Z = 0`, seed 0, default options, `display="iter"`:

- `normal`: independent Gaussian with standard deviations `1..D`, the same
  target as `test_vbmc_multivariate_normal`, plausible box `±2D`.
- `corr`: Gaussian with a fixed random rotation and standard deviations
  `linspace(0.2, 1, D)`, mean `linspace(-0.5, 0.5, D)`, plausible box `±2.5`.

Each configuration was run once *plain* (stage timers only, true wall time)
and once under `cProfile` (attribution). The stage timers are the ones VBMC
already keeps (`main_timer`, reset every iteration and snapshotted into
`iteration_history["timer"]`). Caveats that apply to everything below:

- **cProfile inflates Python-heavy paths.** Profiled wall time was 1.5× (D=5)
  and 1.8× (D=10) the plain run. Attribution percentages are therefore biased
  *towards* Python-overhead-bound code and *against* BLAS-bound code. Use the
  plain-run stage totals for absolute numbers.
- One seed per configuration; easy targets; no noisy target profiled yet.
- `active_sampling` wraps intermediate GP refits and variational fits inside
  active sampling, so stage totals overlap slightly and do not sum to wall.
- Work after the main loop (`determine_best_vp`, `final_boost`) is not covered
  by the stage timers at all; it shows up only as "untimed" and in cProfile.

## 4. Results

### 4.1 Plain runs (true wall time)

| Run | Iterations | Evals | Final N | K path | Wall | Active sampling | GP training | Variational fit | Finalize | Untimed (final boost etc.) |
|---|---|---|---|---|---|---|---|---|---|---|
| `normal`, D=5 | 13 | 70 | 69 | 2→5→8→11 | 116 s | 56.2 s (48%) | 38.1 s (33%) | 10.6 s (9%) | 1.0 s | ~10 s (9%) |
| `normal`, D=10 | 16 | 85 | 82 | 2→5→8→11 | 273 s | 161.1 s (59%) | 80.9 s (30%) | 17.9 s (7%) | 2.4 s | ~11 s (4%) |

Both runs terminated on "variational solution stable", with
`|elbo − ln Z|` of 0.002 and 0.006 and posterior-mean RMSE of 0.002 and 0.006.
`Ns` (GP hyperparameter samples) stayed between 8 and 14 throughout; the
`N ≥ 200 + 10D` optimize-only regime was never reached.

Per-iteration cost is roughly flat in `N` at these sizes: active sampling
cost 12.7 s/iteration during the D=10 warmup with `N ≤ 30`, and 7–11 s later
with `N ≈ 60–80`. GP training grew from ~1 s to ~10 s per iteration over the
same range.

### 4.2 cProfile attribution (% of profiled `VBMC.optimize` time)

| Bucket | `normal` D=5 (173 s profiled) | `normal` D=10 (499 s) | `corr` D=5 (219 s; 20 it., N=96, K→18) |
|---|---|---|---|
| **active_sample** | 49.9% | 59.9% | 55.1% |
| ├ `cma.fmin` | 43.5% | 56.2% | 46.7% |
| ├ acquisition `__call__` (calls) | 44.1% (42k) | 52.6% (145k) | 48.2% (57k) |
| ├ `GP.predict` (calls) | 36.9% (42k) | 43.7% (145k) | 39.6% (57k) |
| └ `vp.pdf` | 3.7% | 3.8% | 4.4% |
| **train_gp / GP.fit** | 33.0% | 31.6% | 16.5% |
| ├ `SliceSampler.sample` | 31.7% | 30.7% | 14.2% |
| ├ `GP.__core_computation` (calls) | 23.3% (112k) | 23.1% (217k) | 11.7% (64k) |
| ├ `__compute_log_priors` | — | 5.8% | — |
| ├ `scipy.linalg.cholesky` | 3.6% | 3.9% | 1.4% |
| └ `solve_triangular` (calls) | 8.8% (707k) | 9.3% (1.73M) | 8.2% (698k) |
| **optimize_vp** | 16.5% | 7.8% | 27.2% |
| ├ `_neg_elcbo` (calls, ms/call) | 15.9% (3.1k, 9 ms) | 7.7% (7.7k, 5 ms) | 26.5% (3.5k, 17 ms) |
| ├ `_gp_log_joint` | 12.6% | 5.8% | 18.3% |
| ├ `_sieve` | 5.4% | 1.5% | 5.8% |
| ├ `minimize_adam` | 8.3% | 5.4% | 16.7% |
| ├ `_eval_full_elcbo` | 2.7% | 1.0% | 4.6% |
| └ `entmc_vbmc` | 3.0% | 1.6% | 7.8% |
| **final_boost** (one call) | 8.9% | 2.9% | 14.6% |
| `copy.deepcopy` (calls) | 0.3% (138k) | 0.1% (153k) | 0.4% (241k) |
| target evaluations | 0.0% | 0.0% | 0.0% |

Per-call costs are stable across runs: `GP.predict` ≈ 1.5 ms per single-point
call; one slice-sampler log-posterior evaluation (`__core_computation`)
≈ 0.35–0.55 ms at `N ≤ 100`; `_neg_elcbo` 5–17 ms growing with `K`.

### 4.3 Where the time inside those buckets goes (D=10, internal time)

| Function | Calls | Internal time | Note |
|---|---|---|---|
| `covariance_functions.py:135 compute` (SE kernel) | 2.70M | 58 s (103 s cum) | called per hyperparameter sample per predict/log-posterior |
| `GP.predict` own time | 145k | 39 s | Python loop over `Ns` |
| `numpy.ufunc.reduce` | 10.2M | 26 s | tiny sums |
| `mean_functions.py:340 compute` | 1.46M | 21 s | |
| `GP.__compute_log_priors` | 217k | 21 s (29 s cum) | ~20% of the sampler |
| scipy `solve_triangular` incl. validation wrappers | 1.73M | 46 s cum, of which ~24 s in LAPACK-adjacent code | `_util.wrapper` + `_asarray_validated` ≈ 20 s |
| `scipy.linalg._cholesky` | 217k | 13.5 s | the actual O(N³) work of GP training |
| `_gp_log_joint` own time | 7.7k | 20 s | |
| `vp.pdf` own time | 145k | 13 s | Python loop over `K` |

At `N ≤ 100`, the Cholesky factorization is about 10% of GP training; the
other 90% is Python and scipy call overhead around matrices of a few thousand
doubles.

## 5. What this changes in the modernization plan

Relative to §2 of `2026-09-02-modernization-discussion.md` (estimates):

1. **Active sampling dominates, not the variational stage**, at 50–60% of
   wall time. The cause is exactly the one §2 named but under-weighted:
   CMA-ES calls the acquisition one candidate at a time, ~650 (D=5) to ~1,800
   (D=10) times per new point, and each call runs `gp.predict` with a Python
   loop over `Ns` samples plus `vp.pdf` with a loop over `K`. The cost is
   independent of `N` at these sizes. **Stage 2 item 3 (batch the CMA-ES
   objective; vectorize `predict` over `Ns` and `pdf` over `K`) moves to the
   top of the list.** `cma` supports vectorized/parallel objectives.
2. **GP training is overhead-bound too.** The §2 count (thousands of
   sequential log-posterior evaluations per fit; measured 8k–13k) was right,
   but only ~10% of each evaluation is the Cholesky. Prior evaluation, kernel
   and mean recomputation, and scipy validation wrappers around
   `solve_triangular` are the rest. Vectorizing `__compute_log_priors`,
   calling LAPACK triangular solves directly (or `check_finite=False`), and
   trimming per-sample recomputation in gpyreg are a second Stage 2 item.
   This work lands in gpyreg, which PyBADS also uses, so it goes through
   gpyreg PRs.
3. **The variational stage is 8–27%**, growing with `K` (27% on the
   correlated target with `K → 18`). `_gp_log_joint` is ~70% of it. Stage 2
   item 1 (einsum over `(Ns, K, D, N)`) stands, as third priority.
4. **`final_boost` is 3–15%** in a single call (`K = 50`, `ns_ent_fine`),
   dominated by `_eval_full_elcbo` and `entmc_vbmc`. Stage 2 item 2
   (multi-RHS solve in `_eval_full_elcbo`) pays off here.
5. **The operating point is smaller than assumed.** On these targets VBMC
   converges in 13–20 iterations with `N ≈ 70–100`, so `Ns` never drops below
   8 and the optimize-only GP regime is not reached. The §2 assumption of
   ~70 iterations with `N → 350` describes hard or noisy targets, or runs that
   exhaust `max_fun_evals`. Both regimes matter; profile a noisy target and a
   hard target next.
6. Memory (`iteration_history` deep copies): 138k–241k `deepcopy` calls but
   under 1 s. Not a speed issue at this `N`; the memory-growth concern in §4
   of the plan is unchanged.
7. **GPU conclusion unchanged**: nothing here is a large dense linear-algebra
   op. Every hot spot is call overhead.

## 6. Stage 0: finite-difference gradient checks

New tests, all using `pyvbmc.testing.check_grad`
(`scipy.differentiate.jacobian`):

- `pyvbmc/testing/vbmc/test_variational_optimization_grad_fd.py` (7 tests):
  `_gp_log_joint` (8 hyperparameter samples, and the `Ns = 1` path),
  `_neg_elcbo` with the entropy lower bound (with and without an active
  soft-bound penalty) and with the Monte Carlo entropy, `_vp_bound_loss`
  with mu, ln-scale and weight bounds all violated (at `D = K = 2` on the
  shared fixture and at `D = 3, K = 2` so a reshape-order error is a
  scramble, not a transpose), `_soft_bound_loss`. Uses the raw
  parameterization `(mu, ln sigma, ln lambd, eta)` with `jacobian_flag=True`
  throughout (see the module docstring for why).
- `pyvbmc/testing/variational_posterior/test_variational_posterior_grad_fd.py`
  (2 tests): `∇x` of `vp.pdf` with and without `log_flag`.
- Both files carry an autouse fixture that saves and restores the global
  `np.random` state: `VariationalPosterior.__init__` draws from it, and the
  first full-suite run without the fixture shifted the stream for every later
  unseeded test (three reruns appeared in tests unrelated to the fix).

Results:

- `_gp_log_joint`, `_neg_elcbo` (no penalty), `_soft_bound_loss`, `vp.pdf`,
  `log vp.pdf`: agree with finite differences to ~1e-8 relative.
- **Bug found and fixed: `_vp_bound_loss` unpacked the ln-scale gradient in
  C order after packing it in Fortran order** (`variational_optimization.py`,
  `np.reshape(dL[...], (D, K))` → `order="F"`). For `D = K` this transposes
  the `(D, K)` block, so the sigma block received what should have been the
  lambd gradient and vice versa; for `D ≠ K` it scrambles both. The path is
  active whenever a component's `ln(sigma_k · lambd_d)` leaves its soft
  bounds during `optimize_vp`, so Adam was being pushed in the wrong
  direction on the penalty term in exactly the cases where the penalty
  matters. Pre-existing `test_vp_bound_loss` only violated a weight bound and
  could not see it. MATLAB's `reshape` is column-major, so this is a porting
  slip. Also fixed the `vp.optimize_lambda` → `optimize_lambd` attribute typo
  on the same lines (§9 of the plan).
  An independent read-only review (Opus agent) confirmed the packing order
  against `get_bounds` (`np.tile` of a `(D,)` vector `K` times, so `d`
  fastest) and added the impact assessment: `theta_bnd` is never `None` in
  production, the ln-scale band is `[ln range_d − 13.8, ln range_d]`, and
  when exceeded the penalty gradient is ~52·(excess) and dominates `dF`. On
  the deterministic-entropy path this handed L-BFGS-B an inconsistent
  `(f, grad)` pair, a plausible cause of the "Cannot optimize variational
  parameters" `RuntimeError` seen in the wild; on the Monte Carlo path Adam
  shrank `lambd` when it should have shrunk `sigma_k`.
- **Documented, not fixed (all added to §9 of the plan):**
  `_gp_log_joint(..., jacobian_flag=False)` returns only the `mu` block of
  `dG` (the other blocks are appended inside the `if jacobian_flag`
  branches; unreachable in production but it would break `dF = -dG - dH`).
  `vp.pdf(orig_flag=True, log_flag=False, grad_flag=True)` returns a
  transformed-space `dy` without the Jacobian correction while the
  `log_flag=True` sibling raises. `_neg_elcbo` shifts `eta` to `max = 0` in
  place on a view of the caller's `theta` before the bound loss, so the eta
  upper soft bound can never fire (and the test for `_neg_elcbo` with an
  active penalty therefore violates mu and ln-scale bounds, not eta).
- The Monte Carlo entropy check needs a loose tolerance (`rtol = 1e-2`): the
  reparameterization gradient in `entmc_vbmc` omits the zero-mean score term
  (`∂θ log q(x; θ)` at fixed `x`), so it is an unbiased estimator of the true
  gradient but not the exact derivative of the sample-based value estimate,
  even with common random numbers. Observed relative gap ~2e-4 at
  `Ns = 1e4`; the existing `test_entmc_vbmc` tests use the same tolerance.
- Finite-difference artifacts to remember: `scipy.differentiate.jacobian`
  returns ~1e-7 (status −2, "did not converge") for coordinates whose true
  derivative is exactly zero, so `atol` must be ≥ 1e-6 for hinge-type losses.

Still untested by finite differences (next): `compute_vargrad` in
`_gp_log_joint` (known broken, §9), the gpyreg kernel/mean/noise derivatives
beyond `log_posterior`, the parameter transformer Jacobian.

## 7. State at end of session

Uncommitted on `dev-next`:

- `dev/scripts/profile_run.py`, `dev/scripts/runs/` gitignored, `dev/README.md`
  scripts section, this file.
- Fix in `pyvbmc/vbmc/variational_optimization.py` (`_vp_bound_loss`).
- Two new test files (9 tests, all passing).
- Full suite after the fix, first run (before the RNG fixtures, 397 tests):
  397 passed, 3 reruns (`test_minimize_adam_matyas_with_noise`,
  `test_vp_optimize_1D_g_mixture`, `test_active_uncertainty_sampling`; all
  unseeded, all after the new VP-constructing tests in collection order),
  22:15 wall with the profiling batch overlapping its start.
- Full suite after the fix, second run (with the RNG fixtures, 398 tests):
  **398 passed, 0 reruns, 15:12 wall.** Same random stream as the baseline
  for every pre-existing test, so this is a like-for-like comparison: the fix
  changes no test outcome, and the three earlier reruns were the stream shift.

Suggested commit split: (1) dev tooling + devlog, (2) `fix: unpack ln-scale
gradient in Fortran order in _vp_bound_loss` with the new tests.

**Pre-commit on Python 3.12 is broken by the pinned `pycln` hook**
(`.pre-commit-config.yaml`, `v2.1.3`): its `libcst` dependency has no wheel
for 3.12 at that version and tries to build from source with a Rust
toolchain. `SKIP=pycln pre-commit run` works; isort 5.12 and black 23.3.0
pass on all files touched today. **Fixed the same day:** bumped the `pycln`
rev to `v2.6.0`, which installs on 3.12 and passes over all files unchanged.
The other three hooks were left at their pins on purpose; bumping black past
23 would reformat the codebase.

## 8. Next steps

1. Re-run the full suite after the fix (in progress) and commit as above.
2. Profile a noisy target (`specify_target_noise`, VIQR path) and a harder
   target that reaches `N ≥ 200 + 10D`, to cover the regime §2 assumed.
3. Stage 0 continued: fixture generator script and golden-trace harness
   (plan §10), finite-difference checks for the transformer Jacobian and
   gpyreg derivatives.
4. Stage 1 (RNG `Generator` threading) as the first PR, bundled with the §9
   one-liners.
5. Stage 2 in the measured priority order: batched acquisition evaluation,
   then gpyreg sampler overhead, then `_gp_log_joint`, then `_eval_full_elcbo`.
