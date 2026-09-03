# 2026-09-02 — Modernizing the numerical core: assessment and plan

**Status:** discussion and decisions only. No code changed except reverting an
unfinished local edit (see §12).

All timings below are op-count and BLAS-call estimates from reading the code.
Nothing was profiled. A measured profile is the first action item.

---

## 1. Goal

PyVBMC is plain NumPy/SciPy. Two goals motivated the discussion:

1. **Speed.** Reduce per-iteration algorithm overhead, which matters for
   cheap-to-moderate targets and for large `K`, `N`, `D`.
2. **Compatibility with modern ML pipelines.** Users increasingly have
   PyTorch/JAX models, want batched or parallel evaluation, and want the
   returned posterior to compose with other tools.

### Who this is for

The primary user is a researcher whose main expertise is in their scientific
domain rather than in ML tooling: the PyVBMC/PyBADS audience in computational
and cognitive neuroscience, psychology, and other model-fitting fields, some
of them coming from the MATLAB version of VBMC. A significant share works on
Windows with Anaconda. Typical hardware is a laptop or lab workstation, and
sometimes an HPC cluster with home-directory quotas; CPU-only environments are
common, so GPU acceleration is not something PyVBMC should strictly require.
Many users expect `pip install pyvbmc` or `conda install pyvbmc` to work without
further setup, and what matters to them is install friction, robustness, and
wall-clock time, not which array library sits underneath.

A secondary audience is ML- or SBI-literate users with PyTorch models who want
to plug in a target callable and get back a posterior object that composes
with their stack.

The package is maintained mostly by the PI plus graduate students and postdocs
who rotate through the lab (with the help of AI coding agents), so debuggability
and the stability of the dependency stack are constraints in their own right.

Consequences for this plan: install footprint and setup steps are first-class
constraints, float64 robustness beats raw speed, and GPU execution is neither
a requirement nor a design driver at the current operating point. It must
nevertheless remain *available*: the port is to be device-agnostic by
construction, because GPU becomes valuable if we later scale to many more
evaluations (see §8).

The question asked was whether to move to a multi-backend layer (Keras 3), to
PyTorch, or to JAX.

---

## 2. Where the time goes (one VBMC iteration)

Symbols: `D` problem dimension, `K` mixture components, `N` GP training points,
`Ns` GP hyperparameter samples, `Nc` acquisition candidates. Defaults at `D=5`:
`max_fun_evals = 50(2+D) = 350`, 5 evaluations per iteration, so ~70
iterations with `N` growing 10 → 350. `Ns = round(80/sqrt(N))`, capped at 8 in
warmup, forced to 0 (optimize-only) once `N >= 200 + 10D`. `K` starts at 2 and
grows toward `ceil(N^(2/3))`, and `final_boost` jumps it to 50. `Nc = 2^13`.
Entropy MC samples: `ns_ent = 100 K^(2/3)`, `ns_ent_fine = 4096 K`.

| Stage | What dominates | Per-iteration scale | Backend fit |
|---|---|---|---|
| Variational optimization (`optimize_vp` → `_neg_elcbo` → `_gp_log_joint`) | Python double loop over `Ns × K` inside `min(10000, 100(2+D))` Adam steps × `elbo_starts = 2`; ~50 tiny numpy calls per loop body on `(D,N)` arrays (`variational_optimization.py:1374`, `:1400`). Also 5K–50K candidate VPs in the sieve, each a `copy.deepcopy(vp)` (`:969`). | 10⁴–10⁵ loop bodies, ~1–3 s, grows with `K` | Pure-NumPy `einsum` over an `(Ns,K,D,N)` tensor (a few MB) removes most of it. JIT of the whole Adam loop is the JAX-only extra. |
| Full ELCBO evaluation (`_eval_full_elcbo`) | Triple loop over `(s, k, j<=k)` with two single-RHS triangular solves per pair (`:1473-1512`), recomputing `z_j` each time. Called 2–4× per iteration, once per pruning candidate, and at `K=50` in `final_boost` with `ns_ent_fine`. | `Ns·K²/2` solves, ~0.1–0.3 s per call | One multi-RHS solve plus a `K×K` GEMM. Backend-neutral. |
| GP hyperparameter fit (`gpyreg.GP.fit`) | Space-filling init (1024 → 64 gradient-free evals), one L-BFGS-B run (`scipy.optimize.minimize`, `jac=True`), then coordinate-wise slice sampling, gradient-free, one `N×N` Cholesky per density evaluation. ~90% of `fit` is the sampler. Vanishes once `N >= 200 + 10D`. | ~5,000 sequential Choleskys, ~2–7 s in the `N ≈ 100–250` window | Inherently sequential. Only batched multi-chain sampling turns it into a tensor op. GPU is far below break-even at these `N`. |
| Active sampling (`active_sample.py:244-500`) | CMA-ES (`cma.fmin` + `NoiseHandler`, `maxfevals = 500(D+2)`) calling the acquisition one point at a time; each call is ~100–250 tiny ops through `gp.predict` (Python loop over `Ns`), `vp.pdf` (Python loop over `K`) and the inverse transform. | 5k–17k single-point evaluations, ~1.5–6 s | Batch the CMA-ES population (`cma` supports vectorized objectives). The one `Nc = 8192` sieve call is the only GEMM-sized op in the package. |
| Finalize (`vp.kl_div(N=1e5)`, `gp.predict` on `gp.X`) | RNG-bound, already vectorized. | ~20–50 ms | Low priority. |

Two conclusions:

- **Dispatch overhead, not arithmetic, is ~90% of the variational stage.** The
  arithmetic per Adam step is ~10⁷ flops on arrays of a few thousand doubles.
  A GPU does not help *at this operating point* (`N` in the hundreds): the
  variational stage is 700 sequential steps and the GP fit is 5,000 sequential
  Choleskys of size ~200. Most of the achievable speedup is available in NumPy.
  The picture changes at `N ≈ 2k–5k`; see the GPU note in §8.
- **float64 is non-negotiable.** gpyreg's Cholesky retry ladder
  (`gaussian_process.py:2413-2438`, `sn2_mult *= 10`) and the eigendecomposition
  fallback (`:2332-2355`) exist because the matrices are borderline singular.
  PyTorch defaults to float32 and JAX needs a global `jax_enable_x64` flag;
  either must be pinned explicitly. This also rules out consumer-GPU throughput.

**Measured 2026-09-02** (see `plans/profile-and-gradient-checks.md`):
the table above misjudges the balance. On D=5 and D=10 Gaussian targets,
active sampling is 50–60% of wall time (single-point `gp.predict` calls from
CMA-ES, ~1.5 ms each, ~650–1,800 per new point), GP training 30% (the slice
sampler as predicted, but only ~10% of each step is the Cholesky; the rest is
prior evaluation, kernel recomputation and scipy call overhead), the
variational stage 7–27% depending on `K`, and `final_boost` 3–15%. Those
targets converge in 13–20 iterations with `N ≈ 70–100`, so the `N → 350`
assumption above describes hard or noisy targets only. The two conclusions
above (overhead-bound, GPU irrelevant here) stand; the Stage 2 priority order
in §10 is revised.

**Measured again the same evening on the benchmark target suite**
(`plans/benchmark-suite-and-golden-traces.md` §Results: banana, cigar,
lumpy, Student-t at D=4, a logistic regression at D=5, a noisy banana):
same balance, sharper. Active sampling 55–67 %, of which single-point
`GP.predict` calls are 40–48 % (36k–106k per run); GP training 15–22 %, with
the Cholesky under 2 % and the scipy `solve_triangular` wrappers alone at
9–10 %; the variational stage 16–24 % on ridged posteriors (`_gp_log_joint`
11–18 %); `final_boost` 6–12 % in one call. On the noisy VIQR path the
active-sampling bucket is the per-sample full GP refits (25 %) and VP
optimizations (22 %), not the acquisition search (3.5 %). All these targets
converged in 80–135 evaluations and 1–2 minutes, so the `N → 350` regime is
reached only by the budget-exhausting configuration, which also exposed and
fixed a crash in that regime (§9).

---

## 3. Hand-derived gradients (what autodiff would delete)

| Location | Differentiates | Gradient lines |
|---|---|---|
| `entropy/entlb_vbmc.py:43-180` | entropy lower bound wrt `mu, sigma, lambd, w` | ~96 |
| `entropy/entmc_vbmc.py:39-134` | MC entropy, reparameterization trick, antithetic sampling by hand | ~55 |
| `vbmc/variational_optimization.py:1238-1606` (`_gp_log_joint`) | analytic `E_q[GP mean]` under the mixture wrt all four blocks, plus `compute_vargrad` | ~136 |
| `variational_optimization.py:991-1235` (`_neg_elcbo`) | orchestration, weight penalty, duplicated grad/no-grad branches | ~43 (+24 duplication) |
| `variational_optimization.py:503-606` (`_vp_bound_loss`) | chain rule folding `ln_scale` gradient onto `sigma`/`lambd` | ~31 |
| `variational_optimization.py:609-657` (`_soft_bound_loss`) | hinge-quadratic subgradient | ~9 |
| `variational_posterior/variational_posterior.py:367-565` (`pdf`) | `∇x` of mixture pdf, used by `vp.mode` | ~25 |
| `parameter_transformer.py:222-269` etc. | log-abs-det Jacobian (density correction, not an objective gradient) | ~56 |
| gpyreg `gaussian_process.py:2357-2512`, `:1275-1466` | `d(-log Z)/d hyp` via explicit `Q = K⁻¹ − ααᵀ`; hyperprior gradients | ~99 |
| gpyreg `covariance_functions.py`, `isotropic_covariance_functions.py`, `mean_functions.py`, `noise_functions.py` | `dK`, `dm`, `dsn2` | ~116 |

**Total ≈ 610 lines** eliminable (≈395 in pyvbmc, ≈215 in gpyreg), plus four
independent copies of the softmax Jacobian (`entlb:170-177`, `entmc:122-130`,
`_gp_log_joint:1539-1545`, `_neg_elcbo:1222-1226`). The `log_abs_det_jacobian`
code is better replaced by a bijector abstraction than by raw AD.

Acquisition functions contain **no** gradients: optimization is CMA-ES or
Nelder–Mead. Autodiff would *add* capability there.

**Test coverage of gradients today.** Finite-difference checks exist only for
the two entropy functions (`testing/entropy/test_ent*_vbmc.py` via
`testing/_check_grad.py`) and gpyreg's `log_posterior`. The `_gp_log_joint`,
`_neg_elcbo` and `vp.pdf` gradients are pinned to stored MATLAB arrays or
hard-coded scalars: correct by agreement with MATLAB, not independently.
`compute_vargrad` is untested (see §9).

**Autodiff pitfalls identified.**

1. `_neg_elcbo:1080-1084` mutates `vp` in place via `set_parameters` and does
   an in-place subtract on a view of the optimizer's `theta`; `minimize_adam.py:81,97`
   mutates the caller's `x0`. The objective must become a pure `theta → (F, dF)`.
2. `set_parameters` (`variational_posterior.py:643-650`, `:750-757`) silently
   renormalizes `lambd`/`sigma` by `‖λ‖/√D`. The hand gradients ignore this
   gauge Jacobian (the ELBO is invariant along the ray). Naive AD through it
   gives a different, projected gradient; needs `stop_gradient` or a proof that
   every term, including `_vp_bound_loss` on `ln_scale`, is invariant.
3. `np.errstate(all="ignore")` around Matern `dK` (`covariance_functions.py:279,356`)
   hides `inf·0` NaNs; `__compute_log_priors:1325` writes `NaN` on purpose.
   Torch/JAX propagate NaNs; JAX needs the double-`where` trick.
4. `scipy.special.erfcinv` (probit transform, `parameter_transformer.py:524-528`)
   exists in neither torch nor jax; `erfinv(1−x)` loses precision exactly at
   the boundary. `gammaln`, `erfc` are fine. `pdist/cdist/squareform` must become
   broadcast distances. `brentq`, `interp1d`, `np.polyfit` are off the gradient path.
5. Hard clamps and masks on the gradient path: `_gp_log_joint:1507,1518`
   (`np.maximum(varG, eps)`, inconsistent with `dvarG`), `noise_functions.py:272-276`,
   `parameter_transformer.py:479-492, 504-511, 538` (`np.sign`, `arccos` at
   the domain center, self-referential `u[u==0]` mask).
6. Data-dependent control flow: `gaussian_process.py:2404` selects between two
   algorithms on `min(sn2) >= 1e-6`; `:2413-2438` is a `try/except LinAlgError`
   jitter loop; `_gp_log_joint:1473` is a triangular loop. Fine eagerly,
   hostile to tracing.
7. Slice-assignment gradient buffers everywhere (`mu_grad[:, k, s:s+1] = …`).

---

## 4. The GP layer (gpyreg) — the real scope of any port

**What PyVBMC actually uses.** Covariance is hard-wired to
`SquaredExponential` ARD (`vbmc.py:772`, `gaussian_process_train.py:253`);
several downstream paths hard-fail on anything else (`GP.quad`, VIQR, IMIQR,
`active_importance_sampling`). Mean is `NegativeQuadratic` (`1+2D` hyps).
Noise is `GaussianNoise` with constant + optional user-provided `s2`
(scaled or not). `hyp_N = 3D+3` or `3D+4`. gpyreg does no input
standardization; all warping is PyVBMC's `ParameterTransformer`.

**PyVBMC does not consume the GP through a predict API.** Six modules reach
into `gp.posteriors[s].{hyp, alpha, L, L_chol, sW}` directly:
`variational_optimization.py`, `acq_fcn_imiqr.py`, `acq_fcn_viqr.py`,
`active_importance_sampling.py`, `whitening.py`, `active_sample.py`.
`_gp_log_joint` reimplements gpyreg's `GP.quad` (Gaussian × SE-ARD integrals
with negative-quadratic mean) per VP component, adding the gradients that
`quad` cannot provide; PyVBMC never calls `quad`. VIQR inlines the SE kernel by
hand "for speed".

Consequence: if gpyreg stays NumPy, every ELBO evaluation crosses a
NumPy↔tensor boundary and the gain evaporates. **A backend port must take the
GP layer with it.**

**Off-the-shelf GP libraries do not fit.** GPyTorch/BoTorch abstract the solve
behind `LinearOperator`, expose Cholesky/alpha only through private caches,
and offer hyperparameter MCMC only as SAAS-NUTS with its own prior. GPJax has
thin heteroskedastic support and an unstable API. tinygp is the closest
(per-point noise, exposed solver, autodiff) but supplies only the ~150 easy
lines. None provide a slice-sampled hyperposterior over *our* priors and
bounds, or the analytic integrals.

**Reimplementation estimate (essential code, not docstrings).**

| Piece | Lines |
|---|---|
| SE-ARD kernel, NegativeQuadratic mean, GaussianNoise (forward only) | ~90 |
| `__core_computation` + `Posterior` (no `dnlZ` block) | ~90 |
| `predict` / `predict_full`, batched over `Ns` | ~90 |
| `quad` + `_gp_log_joint` analytic integrals | ~280 |
| priors, bounds, normalization constants | ~320 |
| `f_min_fill` + smoothbox cdf/ppf | ~210 |
| slice sampler | ~250 |
| `update` incl. rank-1 | ~110 |
| **Total** | **~1,400–1,600** |

Of these, ~400 are hand derivatives that vanish under autodiff and ~500
(priors, bounds, space-filling init) run once per `fit` and can stay NumPy.
**Net new tensor code ≈ 600–800 lines** plus the six call sites.

Memory note: `gp.posteriors` holds `Ns` factors of `N×N`, `GP.clean()` is never
called, and every iteration's GP is deep-copied into `iteration_history`
(`vbmc.py:1272`), so memory grows as `Σ_i Ns_i·N_i²`. Fix independently of any port.

---

## 5. Architecture constraints

- **Orchestration stays Python.** `VBMC.optimize` (`vbmc.py:825-1587`) is ~760
  lines of scalar/dict bookkeeping; only 146 of 2,562 lines in the file touch
  `np.`. `IterationHistory` deep-copies VP, GP, FunctionLogger, `optim_state`
  and timer every iteration (pure-Python cost, would be *worse* with tensors).
  `Options` resolves `.ini` values with `eval(value, globals(), {"D": D})`
  against `options.py` module globals, so `np`-shaped names must stay importable there.
- **Every shape is dynamic.** `N` grows 5/iteration and shrinks at warmup end;
  `K` changes inside a data-dependent `while` that deletes a *randomly chosen*
  component (`variational_optimization.py:322-391`); `Ns` walks down and
  switches to 0; 78 `append/concatenate/delete` sites outside tests.
  Near worst-case for top-level JIT. Eager execution is a natural fit.
- **RNG is the global legacy `np.random`** at ~40 call sites, no `Generator`,
  no `seed`/`rng` argument on `VBMC.__init__`. Reproducibility is via
  `np.random.get_state()` snapshots per iteration and `load(set_random_state=True)`.
- **scipy inventory (non-test):** `optimize.minimize` (VP optimizer, `vp.mode`,
  Nelder–Mead acq), `optimize.brentq`, `linalg.solve_triangular`,
  `special.erfc/erfcinv/gammaln`, `stats.norm.ppf`, `spatial.distance.cdist`,
  `integrate.trapezoid`, `interpolate.interp1d`, `fftpack.dct/idct`,
  `stats.multivariate_normal`, plus **private** `scipy.stats._distn_infrastructure`
  / `_multivariate` classes in `priors/scipy.py:5-10`, `priors/product.py:5`.
- **Other deps:** `cma` (default acquisition optimizer), `dill` (save/load of
  the whole object graph including the user's closure), `matplotlib`/`corner`
  imported at module top level (`vbmc.py:13`, `variational_posterior.py:11`).
  `plotly`, `pytest`, `pytest-mock`, `pytest-rerunfailures` are listed as
  runtime dependencies (`pyproject.toml:16-20`) but `plotly` is imported nowhere.

---

## 6. Test infrastructure today

~390 test functions in 43 files under `pyvbmc/testing/`, mirroring the package.
CI: `pytest --reruns=5 -x` on {ubuntu, windows, macos} × Python {3.10–3.12},
checking out gpyreg from `acerbilab/gpyreg`. **`--reruns=5` means the suite is
known-flaky** because of global RNG draws.

MATLAB-derived fixtures are a minority: `entropy/entropy-test.mat`,
`variational_posterior/vp-test.mat`, `test_moments_no_orig_flag_2_MATLAB.mat`,
`whitening/vp_initialized_MATLAB.mat`, `vbmc/compare_MATLAB/*.mat` (with
generating `.m` scripts committed), plus `.txt` arrays and a `.pkl` for
backward-compatible load. They are read with `scipy.io.loadmat`, so no MATLAB
is needed at test time; the problem is that they are opaque, cannot be
regenerated, and cover a handful of internals.

End-to-end: 6 full `optimize()` runs asserting posterior-mean RMSE `< 0.5` and
`|elbo − lnZ| < 0.5`. This cannot distinguish a subtly broken port from an
unlucky seed. Tolerances elsewhere span `1e-14` to `5e-2`.

`testing/_compare_matlab.py:43` `rand_int` is broken (`res = lo`, undefined).

---

## 7. Backend decision

**Keras 3 — rejected.** Its ops layer has no backend-agnostic gradient
function, linear-algebra and special-function coverage is thin, multi-backend
control flow is its weak point, and it is built for layers, not solvers. More
fundamentally, any multi-backend design that keeps NumPy as a supported
backend also keeps the hand-written gradients, forfeiting the main benefit.
The same objection applies to writing kernels against the Array API standard
(a cheap way to let tensors flow through, but no autodiff, no JIT).

**JAX — rejected for this codebase and audience.** Technically it has the
highest speed ceiling (`jit` + `scan` over the Adam loop and the slice sampler
attack exactly where the time goes; `vmap` over `Ns`/`K` batches the rest), but
the algorithm is near worst-case for tracing (§5) and would need shape
bucketing with masks, explicit PRNG keys, and a functional rewrite of objective
and sampler: a "VBMC 2" project, not a port. Audience reasons, checked
2026-09-02:

- `pip install jax` now ships official **CPU wheels for Windows**; GPU on
  Windows remains unsupported outside WSL2 (irrelevant here).
- **No conda-forge `jaxlib` for win-64** (linux-64, linux-aarch64, osx-64,
  osx-arm64 only; latest 0.10.2, 2026-07-03). PyVBMC ships on conda-forge and
  its Windows/Anaconda audience is significant; the recipe could not declare
  the dependency on that platform.
- **`jax_enable_x64` is still a global, whole-program flag** (float32 default;
  issue jax-ml/jax#22688 to decouple default dtype is open). A library that
  fails in float32 either flips a global at import or relies on the user.
- Maintainers rotate through the lab (§1). Eager code debugs with `print`; tracer
  errors, recompilation surprises and compile latency on a one-minute fit do
  not. `jaxlib` is still 0.x with a history of deprecations.

**PyTorch — chosen as the single backend for the numerical core, if a port
happens.** Eager mode absorbs dynamic shapes, in-place idioms and Python
control flow unchanged; float64 is per-tensor with no global side effects;
`torch.autograd` + `torch.func.vmap` cover gradient deletion and `Ns`/`K`
batching; `torch.linalg.cholesky_ex` handles the retry ladder; it is the stack
the SBI ecosystem lives in; strong backward-compatibility discipline. Caveats:

- Eager per-op overhead is several times NumPy's. A line-by-line port of the
  current loops would be **slower**. Speed comes from batching the loops first
  (which we do in NumPy regardless) and `torch.compile` opportunistically.
- Install footprint (torch 2.13.0, PyPI, 2026-07-08): Windows 122 MB, macOS
  arm64 111 MB, **Linux x86_64 527 MB plus declared dependencies on
  `cuda-toolkit==13.0.3`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`,
  `nvidia-nvshmem-cu13`, `nvidia-cusparselt-cu13`, `triton`** — several GB
  whether or not a GPU exists. The CPU-only build needs
  `--index-url https://download.pytorch.org/whl/cpu`, which a library cannot
  set in its own metadata. conda-forge provides CPU variants cleanly.
  Install docs must address this before torch becomes a hard dependency.

---

## 8. "Compatible with modern ML pipelines", concretely

| Need | Status today | Backend-neutral? |
|---|---|---|
| torch/JAX callable as target | Works via a thin adapter (`float(fn(torch.as_tensor(x)))`); VBMC never needs the target's gradient, so solver backend and model backend are independent | yes, needs docs |
| Batched target evaluation | None. Initial design (`active_sample.py:164-168`, `fun_eval_start = max(D,10)` points) is embarrassingly parallel and serial today. Acquisition loop is genuinely sequential (batch-BO is research) | yes |
| Many problems in parallel | `joblib` over independent `VBMC` objects works now except for the global RNG | yes; needs `seed=` |
| Differentiable posterior object | `vp.to_torch()` → `MixtureSameFamily(Categorical(w), Independent(Normal(mu, sigma*lambd)))` inside a `TransformedDistribution`; a few hundred lines. Also ArviZ `InferenceData` export | mostly |
| GPU | Not worth it at the current operating point (`N ≲ 1k`). Becomes valuable at `N ≈ 2k–5k`; see the note below. The port must be device-agnostic so the option exists | design constraint on Stage 4 |
| Serialization | Already good (`save/load`, rewind to any iteration). Tensors make it slightly worse (device tags) | — |

**GPU at larger `N`.** The "GPU does not help" conclusion of §2 holds for the
current operating point, `N ≲ 1k` (default `max_fun_evals = 50(2+D)`, so
~1,100 at `D=20`). If we later scale to `N ≈ 2k–5k` evaluations, the dense-GP
terms that are negligible today scale as `N²` and `N³` and become dominant,
and they are exactly what GPUs are good at. Rough orders of magnitude for one
float64 Cholesky (estimates, not measurements):

| `N` | CPU (laptop → workstation) | Consumer GPU (fp64 at 1/64 rate) | Datacenter GPU (A100/H100 class) |
|---|---|---|---|
| 1,000 | 3–10 ms | ~1 ms, launch-bound | ~1 ms, launch-bound |
| 2,000 | 10–50 ms | ~5 ms | 1–2 ms |
| 5,000 | 0.1–1 s | ~50 ms | ~10 ms |

At `N = 5,000` slice sampling is already off, so one GP fit is ~100
marginal-likelihood + gradient evaluations at ~3 Cholesky-equivalents each:
half a minute to several minutes per iteration on CPU, seconds on a GPU. The
acquisition sieve (`O(N²·Nc)`) and the CMA-ES evaluations (`O(N²)` per point,
~17k points) add a few more CPU seconds *if batched*, far more if not. Reaching
`N = 5,000` at 5 evaluations per iteration means ~1,000 iterations, so
per-iteration overhead multiplies into a day or more on CPU versus hours on a
GPU. Consumer cards close the gap to ~5–10×, datacenter cards (what HPC users
have) to ~50–100×.

Two consequences:

- **Device-agnostic by construction (Stage 4 design constraint).** Every
  tensor carries a `device`; no `.numpy()`/`.item()` round-trips inside inner
  loops; host-side control flow reads only a handful of scalars per iteration.
  The places where host syncs would creep in are the single-point CMA-ES
  objective and the slice sampler's per-step accept/reject. Cheap if designed
  in from the start, expensive to retrofit.
- **Scaling to `N ≫ 1k` is an algorithmic project, not a backend one.**
  Refitting the GP every iteration, 5 evaluations per iteration, and exact
  dense GPs are choices tuned for `N` in the hundreds. Fewer refits, larger
  batches per iteration, and possibly approximate GPs are what would make 5k
  evaluations practical; that is 2.0 territory (§12). The GPU option is what
  makes that project worth attempting, not what delivers it.

GPU remains a non-requirement: everything must run, and run well, on CPU.

**A `seed=`/`rng=` constructor argument is the single highest-leverage API
change in the package.** It fixes flakiness, enables parallel runs, and is a
prerequisite for either backend. It changes the format of the saved random
state used by resume.

---

## 9. Latent bugs and cleanups found along the way (fix regardless)

- `variational_optimization.py:1561` — `sigma_vargrad *= np.reshape(sigma_vargrad, …)`,
  `sigma` intended. `:1574` assembles `dvarG` from `grad_list` instead of
  `vargrad_list`. `:1556` drops the `order="F"` used at `:1525`. All in the
  `compute_vargrad` path, marked `# TODO: compute vargrad is untested` (`:1346`, `:1553`).
- `variational_optimization.py:561,583` — reference nonexistent
  `vp.optimize_lambda` (attribute is `optimize_lambd`); masked by short-circuit
  because `optimize_sigma` is always `True`.
- `testing/_compare_matlab.py:43` — `rand_int` broken.
- `gaussian_process_train.py:768` — `noise_shaping` is an unported stub; the
  option only flips a noise-function flag.
- GPs retained for every iteration with all `Ns` Cholesky factors; `GP.clean()`
  never called (§4).
- `pyproject.toml:16-20` — `plotly`, `pytest*` as runtime deps. `matplotlib`
  and `corner` imported at module top level.
- `priors/scipy.py:5-10`, `priors/product.py:5` — imports of scipy private classes.
- gpyreg `predict` tiles `sW` to `(N, N_star)` instead of broadcasting (13 MB
  per sample at `Nc = 8192`).
- **Fixed 2026-09-02:** `_vp_bound_loss` unpacked the ln-scale gradient block
  with a C-order `reshape` after packing it with `order="F"`, scrambling the
  sigma/lambd penalty gradients (a transpose when `D = K`) whenever a
  component's scale left its soft bounds. `theta_bnd` is never `None` in
  production, and the bounds are exceeded routinely in warmup, after `K`
  grows and in `final_boost`, so L-BFGS-B was fed an inconsistent `(f, grad)`
  pair and Adam pushed `lambd` where it should have pushed `sigma`. Found by
  the finite-difference tests added the same day; MATLAB's column-major
  `reshape` was correct, the port dropped the order.
- `_gp_log_joint(..., jacobian_flag=False)` returns only the `mu` block of
  `dG`: the sigma/lambd/w blocks are appended inside the `if jacobian_flag`
  branches. Unreachable in production (`_neg_elcbo` hard-codes it to 1), but
  it would make `dF = -dG - dH` a shape mismatch.
- `vp.pdf(orig_flag=True, log_flag=False, grad_flag=True)` divides `y` by the
  transform Jacobian but returns `dy` uncorrected (a transformed-space
  gradient); the `log_flag=True` sibling raises `NotImplementedError` instead.
- `_neg_elcbo` shifts `eta` to `max(eta) = 0` in place (on a view of the
  caller's `theta`) before the bound loss, so the eta upper soft bound
  (`ub = 0`) can never fire.
- **Fixed 2026-09-02 (evening), `6f3f0ba`:** with a single GP
  hyperparameter sample (Ns = 1, the regime once GP sampling stops at
  `N ≥ 200 + 10D`), `_gp_log_joint` squeezed `G` and `dG` but not `varG`,
  so `_eval_full_elcbo` received a length-1 `varF` and NumPy 2 raised
  "setting an array element with a sequence": every run that reached the
  optimize-only regime crashed (NumPy 1 squeezed silently). Found by the
  budget-exhausting run of the benchmark suite; three regression tests on
  the single-sample GP fixture.
- **Found 2026-09-02 (evening) while building the benchmark suite**, not
  fixed (`plans/benchmark-suite-and-golden-traces.md`):
  - `active_sample.py:336-338` calls `_gp_log_joint(..., compute_var=1)`
    only when the acquisition function sets `compute_var_log_joint`, which
    none does, and stores `optim_state["var_log_joint_samples"]`, which
    nothing reads: unreachable code.
  - `vbmc.py:1341` tests `optim_state.get("stop_gp_sampling") == 0`, but
    only `"stop_sampling"` is ever set, so `_is_gp_sampling_finished` and
    the `tol_gp_var_mcmc` option are dead; the method itself reads history
    keys `N` and `gp_sample_var` that are never declared, so it is broken
    code behind a dead guard.
  - `true_mean`/`true_cov` options (`vbmc.py:1274-1278`): the guard
    evaluates the raw value's truthiness, so numpy arrays raise
    `ValueError`; when they do run (as lists) they draw 10⁶ samples from
    the run's own `vp.rng` every iteration inside the `finalize` timer,
    changing the trajectory. No test exercises them.
  - The `display` and `log_file_level` option comments advertise
    `"notify"`/`"final"`; the code handles `"off"`, `"iter"`, `"full"`.
  - `FunctionLogger.finalize()` is never called by `VBMC` and does not trim
    `n_evals`.
  - `results["rng_state"]` is the literal string `"rng"`; `search_cmaes_best`
    is read nowhere; the MCMC branch of `active_importance_sampling` is
    unreachable (`mcmc_importance_sampling` is never set by any acquisition
    function).
  - `gaussian_process_train.py:610`: the cubic `f = lambda x_: a*x_**3 +
    b*x**2 + c*x + d` uses the closure `x` for the lower-order terms
    (harmless as called with `f(x)`).
  - `kl_div_mvn` is wrapped by `handle_0D_1D_input`, which swallows `mu1` as
    `self`: `mu1` is never promoted to 2-D and keyword calls fail. Call it
    positionally with a `(1, D)` array.
  - Notebook 1 states `lml_true = -2.272` for Rosenbrock + N(0, 3²) at D=2;
    two independent quadratures give **−2.2598** (the x2 integral is
    analytic, leaving a 1-D integral). Notebook 6's heteroskedastic noise
    uses one norm for the whole batch and broadcasts `(n,) + (n,1)` to
    `(n,n)` for `n > 1`. `noisy_cigar` in `test_vbmc_optimize.py` is dead
    code.

---

## 10. Plan (staged; each stage useful on its own; diffs attributable)

**Stage 0 — Test oracle, backend-neutral.**
The reference is the *current NumPy implementation*, frozen before any port.
- A script in the repo generates regenerable `.npz` fixtures (JSON for small
  scalars) from the current code with fixed seeds. Replace the MATLAB `.mat`
  files with its output after confirming the current code matches them once.
- Stage-level oracles with tight tolerances (~1e-10 in float64): GP log
  marginal likelihood + gradient, ELBO + gradient at a fixed VP (`G`, `dG`,
  `varG`), both entropy estimators, acquisition values on fixed candidate sets,
  parameter transformer + log-Jacobian, analytic GP integrals, `kl_div_mvn`, `kde_1d`.
- **Randomness injected, not drawn:** stage tests take pre-drawn samples as
  inputs so the same fixture is valid across backends.
- Finite-difference checks (`testing/_check_grad.py`) on *every* hand gradient,
  before touching any.
- Golden-trace harness: ~50 seeds × the benchmark target suite defined under
  Stage 2 below (the e2e problems plus banana, lumpy, Student-t, a noisy and a
  real-likelihood target, at several `D`), dumping
  per-iteration `(elbo, elbo_sd, sKL, r_index, K, Ns, func_count, N, X, y,
  gp hyps, vp.get_parameters())` to `.npz`. `IterationHistory` already records
  nearly this. Compare **statistically** (KS / paired bootstrap on
  `elbo − lnZ` and posterior RMSE; median `func_count` within a few percent),
  never per-iteration.
- A dtype-assertion canary so a float32 regression fails loudly.

**Stage 1 — Thread a `numpy.random.Generator` through.** `seed=`/`rng=` on
the constructor; every `np.random.*` call site takes the generator. Re-baseline
the golden traces afterwards. Decide what happens to `load(set_random_state=True)`.

**Stage 2 — Pure-NumPy vectorization of the hot loops + memory fix.**
1. `_gp_log_joint` `(s,k)` loop → `einsum` over `(Ns,K,D,N)` (`variational_optimization.py:1374-1467`).
2. `(j,k)` variance loop → one multi-RHS `solve_triangular` + `K×K` GEMM (`:1473-1512`).
3. Batch the CMA-ES acquisition objective (`active_sample.py:398`).
4. Broadcast `vp.pdf` over `K` (`variational_posterior.py:452`).
5. Vectorize `entmc_vbmc` component loop (`entmc_vbmc.py:64`).
6. Drop per-candidate `deepcopy` in `_vb_init` (`:969`).
7. Call `GP.clean()` / stop retaining full GPs in `iteration_history`.
Expected: the bulk of the wall-clock win, 5–30× on the variational stage.

Priority order after the 2026-09-02 profile: item 3 first (50–60% of wall
time), then a new item 8, gpyreg slice-sampler overhead (vectorize
`__compute_log_priors`, direct LAPACK triangular solves instead of the
validated scipy wrappers, less per-sample kernel/mean recomputation; lands in
gpyreg, which PyBADS shares), then items 1–2 (7–27%, growing with `K`), then
the rest. **This order is provisional**: it was measured on two easy Gaussian
targets and must be re-checked on the benchmark target suite below before
the first vectorization PR.

*Confirmed 2026-09-02 (evening) on the benchmark suite*
(`plans/benchmark-suite-and-golden-traces.md` §Results, decision rule
below applied): the variational stage grew to 16–24 % on ridged posteriors
but overtook nothing, so the order stands, **3 → 8 → 1 → 2**, with two
refinements. Items 8 and 1 are close (GP training 15–22 % vs variational
16–24 %), and item 1 is PyVBMC-local while item 8 is a gpyreg PR, so item 1
may be done first for logistics. On the noisy (VIQR) path the
active-sampling bucket is dominated by the per-sample full GP refits and VP
optimizations rather than by the acquisition search, so items 8 and 1 are
what speed up noisy targets. Concrete targets inside item 8: the scipy
`solve_triangular` validation wrappers (9–10 % of total time, 0.6–1.2 M
calls per run) and `__core_computation` Python overhead; the Cholesky itself
is under 2 %. The budget-exhausting run (`normal` D=5, 350 evaluations, Ns
= 0 from N = 250) answers the rule's second clause in the negative **at
that size**: in the optimize-only regime GP training is 1 % of an iteration
(most iterations reuse the previous hyperparameters through
`gp_retrain_threshold`; `scipy.optimize.minimize` is 0.1 % of profiled
time in every run tonight), active sampling halves (one hyperparameter
sample) and the variational stage is 35 %; the L-BFGS-B path does not join
item 8 on this evidence. *Measured 2026-09-03 on cigar at D = 15 with 750
evaluations* (PI's request; `plans/benchmark-suite-and-golden-traces.md`
§Results): the optimize-only regime (N 350–750) is still 7 % GP training on
average, but two full refits at N ≈ 500 and 700 cost 15–19 s each (a 20×
spike over a reuse iteration), so the L-BFGS-B path is not free at that N;
and in the late sampling regime (N 305–345, Ns 5–4) the slice sampler
**overtakes active sampling** (11.6 vs 8.2 s per iteration). Item 8's
weight rises steeply with D and N; the order 3 → 8 → 1 → 2 stands with the
gap between 3 and 8 closing with dimension.

**Benchmark target suite (decided 2026-09-02, after the profile).** The
profile was taken on an independent and a correlated Gaussian, which
converge in 13–20 iterations with `N ≈ 70–100` and are not representative
of the posteriors users bring. Gaussians stay as the fast smoke set. Both
`dev/scripts/profile_run.py` and the Stage 0 golden-trace harness get one
shared, fixed suite of harder targets (define them once, in one module under
`dev/scripts/`, so profiling and validation never drift apart):

1. The VBMC-paper benchmark densities, partly already in
   `testing/vbmc/test_vbmc_optimize.py`: Rosenbrock/banana, cigar
   (correlated, ill-conditioned), lumpy/multimodal, Student-t heavy tails,
   at `D = 2` to `10`. These exercise large `K`, slow warmup and the
   component-pruning path that the Gaussian runs barely touch.
2. A noisy target on the VIQR/IMIQR path (`specify_target_noise`), where `N`
   grows large and `active_importance_sampling` enters the profile.
3. One real model-fitting likelihood with data (a small regression or
   psychophysics model of the kind in `examples/`), so per-evaluation cost is
   non-trivial and the posterior is skewed or ridged the way users' are.
4. One run that exhausts `max_fun_evals` and reaches `N ≥ 200 + 10D` with
   slice sampling switched off, the regime §2 originally assumed and no
   measurement has covered yet.

Decision rule: profile this suite before committing to the Stage 2 order.
If the banana, lumpy or noisy runs shift the balance toward the variational
stage at large `K`, item 1 (the `_gp_log_joint` einsum) moves back up; if
the budget-exhausting run is dominated by GP refits at `Ns = 0`, the
L-BFGS-B path in `gpyreg.GP.fit` joins item 8. The same suite, with ~50
seeds per problem, is the population for the golden-trace statistical
comparison in Stage 0.

**Stage 3 — Pipeline features, backend-neutral.** Batched evaluation of the
initial design and cache path (`FunctionLogger.batch_call`, `vectorized_target`
option); documented torch/jax target adapter; `vp.to_torch()`; ArviZ export.

**Stage 4 — Port the numerical core to PyTorch (decision point, not default).**
Scope: vendored GP core, `variational_optimization`, `entropy`, `acquisition_functions`,
`parameter_transformer` (as bijectors). Orchestration stays Python. Payoff:
delete ~610 gradient lines; gradient-based multi-start acquisition
optimization replacing CMA-ES (possibly the largest algorithmic speedup);
`vmap` over `Ns`/`K`; multi-chain batched slice sampling. Port bottom-up,
kernel-first, against the Stage 0 oracles. Design constraint: device-agnostic
from the first commit (§8), verified by running the kernel tests on a GPU
device when one is available in CI or on a lab machine.

**Before any of this: a measured profile** on a `D=5` and a `D=10` problem in
a proper dev environment (editable pyvbmc + gpyreg + CPU torch). The numbers
in §2 are estimates.

---

## 11. Process decisions

- **Work on a public branch of `acerbilab/pyvbmc`, not a private fork.**
  Nothing here is secret; a private mirror adds drift, flow-back ceremony and
  paid CI minutes for the 3-OS matrix. (GitHub cannot make a fork of a public
  repo private anyway; it would have to be a mirror push.)
- **Stages 0–3 go to `main` via ordinary PRs** as each stabilizes. Users get
  the improvements now and the eventual port diff shrinks.
  *Revised 2026-09-02 (PI):* no PR from `dev-next` until the work is done;
  one PR at the end. Current pickup point in `plans/modernization-roadmap.md`.
- **A long-lived `torch-backend` branch is created only when the first torch
  commit exists.** Merge `main` into it periodically; do not rebase shared
  history with students on the branch. Optionally a draft PR against `main`
  for diff visibility and CI.
- **Vendor the GP core into `pyvbmc/gp/`** rather than porting gpyreg in
  place: gpyreg is a separate public package that **PyBADS also depends on**.
  The torch core needs only PyVBMC's subset (SE-ARD, negquad, GaussianNoise,
  exposed internals, analytic integrals). Spinning it back out as gpyreg 2 is deferred.
- **Version framing: PyVBMC 1.5, not 2.0.** This is an engineering release:
  same algorithm, same defaults, same results up to random streams. It still
  breaks some contracts (torch dependency, seed argument changing random
  streams, new save format), so it is a deliberate minor-version jump with
  release notes, not a patch. "2.0" is reserved for algorithmic changes, which
  are not planned now but may follow once autodiff makes them cheap (§12).
  Released from the same repo on the same PyPI/conda-forge names.
- **Dev notes live in `dev/`** (this folder). `docs/` is the Sphinx build output.
- Design decisions get recorded here as they are made; a separate
  decision-record format is not needed yet.

---

## 12. Deferred ideas

- **Per-component `lambd` (D×K instead of D×1).** An unfinished local edit
  generalizing `VariationalPosterior.sample` to a D×K `lambd` was reverted
  today: it missed the `K=1` heavy-tailed branch (`NameError`, caught by
  `test_sample_one_k_df`) and nothing else in the class (`pdf`, `set_parameters`,
  ELBO, entropy, GP integrals) knows about a matrix `lambd`. With autodiff in
  place this becomes a one-line parameterization change plus a reshape in the
  sampler; without it, it means threading a new shape through every hand
  gradient. Revisit in Stage 4.
- Gradient-based acquisition optimization (multi-start L-BFGS on the
  acquisition surface) once acquisition functions are differentiable.
- Batched acquisition (local penalization / Kriging believer) to allow parallel
  target evaluations within an iteration — research, not engineering.
- Multi-chain batched slice sampling for GP hyperparameters.
- **Scaling to `N ≈ 2k–5k` evaluations** (currently capped in practice at
  ~1k). Algorithmic, not engineering: fewer GP refits, larger batches per
  iteration, possibly approximate GPs. This is the regime where GPU execution
  pays off (§8), which is why Stage 4 must be device-agnostic even though the
  current operating point does not need it.
- Log-space rewrite of the mixture density sums (`entlb:94-97`, `entmc:77-80`,
  `_gp_log_joint:1402-1406`, `vp.pdf:451-464`), which currently rely on float64
  headroom rather than log-sum-exp.

---

## 13. Open questions

- Keep `load(set_random_state=True)` semantics with a `Generator`, or accept a
  save-format change? *Decided in Stage 1* (`plans/stage1-rng-generator.md`
  §3): same flag and meaning, new per-iteration format holding both the
  generator state and the legacy tuple, old files load with a warning.
- Should Stage 2 vectorization be done in a way that is already
  backend-agnostic (Array API), or plainly in NumPy and re-done in torch?
  Current inclination: plainly in NumPy; the torch version will differ anyway
  because of autodiff.
- torch as a hard dependency vs. `pyvbmc[torch]` extra: only meaningful if a
  NumPy core survives, which conflicts with deleting the hand gradients.
  Decide at Stage 4.

---

## 14. Immediate next steps

1. Set up a dev environment: editable `pyvbmc`, `gpyreg`, CPU-only `torch`.
2. Profile a `D=5` and a `D=10` run (`cProfile` + per-stage timers already in
   `pyvbmc/timer`), confirm or correct §2.
3. Start Stage 0: fixture generator script, `check_grad` coverage for
   `_gp_log_joint`, `_neg_elcbo`, `vp.pdf`, `_vp_bound_loss`, `_soft_bound_loss`.
4. Open the first PR: RNG `Generator` threading (Stage 1), together with
   fixing the §9 one-liners.

**Status at end of 2026-09-02 (updated later the same day).** Steps 1–3 are
done, on branch `dev-next`: venv with editable `gpyreg` and `pyvbmc` on the
main machine (no `torch` yet), suite green (389 passed, 0 reruns, 18 min),
`D=5`/`D=10` profiles measured (`plans/profile-and-gradient-checks.md`;
§2 and §10 annotated above), and finite-difference checks added for
`_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`, `_soft_bound_loss` and
`vp.pdf`, which found and fixed the `_vp_bound_loss` reshape bug (§9).
Pickup point: step 4 (Stage 1 RNG PR bundled with the §9 one-liners), plus
the remaining Stage 0 items (fixture generator, golden-trace harness,
transformer and gpyreg gradient checks). Before any Stage 2 PR: build the
benchmark target suite (§10) into `profile_run.py`, profile it, and confirm
or revise the Stage 2 priority order, which currently rests on two easy
Gaussians.

The paragraph above is the status as of that day. Stage status and the
current pickup point are tracked in `plans/modernization-roadmap.md` from
here on, not in this devlog.

---

## Sources consulted for §7

- JAX installation guide: https://docs.jax.dev/en/latest/installation.html
- jaxlib on conda-forge: https://anaconda.org/conda-forge/jaxlib
- torch 2.13.0 metadata: https://pypi.org/pypi/torch/json
- JAX default dtypes and the X64 flag: https://docs.jax.dev/en/latest/default_dtypes.html
- jax-ml/jax#22688 (decouple `jax_enable_x64` from default dtype): https://github.com/jax-ml/jax/issues/22688
