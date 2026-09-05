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

**Measured again on the benchmark target suite, regenerated 2026-09-03/04
with the papers' procedure** (`plans/benchmark-suite-and-golden-traces.md`
§Results (regenerated): banana, cigar, lumpy, Student-t at D = 4, a bounded
logistic regression at D = 5, lumpy and banana at D = 10, two noisy
targets, a 15-D cigar run to 750 evaluations; a first pass on 2026-09-02
with truth-anchored start points and boxes was withdrawn, and its shares
turned out to lie within a few points of these). Same balance, sharper.
Active sampling 54–69 % of wall on the noiseless targets, of which
single-point `GP.predict` calls are 40–50 % of profiled time (41k–249k per
run) and `vp.pdf` another 4–6 % (15 % at D = 15 with K ≈ 25); GP training
13–20 % at D = 4 and 22–28 % at D = 10, with the Cholesky at 1–3 %, the
scipy `solve_triangular` wrappers alone at 9–10 % (0.6–9.4 M calls) and
`__core_computation` at 10–21 %; the variational stage 9–20 % at D = 4,
largest on ridged posteriors (`_gp_log_joint` 12–18 %), only 4–7 % at
D = 10; `final_boost` 3–12 % in one call. On the noisy VIQR path the
active-sampling bucket is the per-sample full GP refits (26–33 %) and VP
optimizations (20 %), not the acquisition search (5–8 %). The D ≤ 10
targets converged in 85–220 evaluations and 1–5 minutes, so among the
profiled runs only the budget-exhausting configuration reaches the
optimize-only regime (§10); in the golden population 12 of 280 runs reach
it briefly (9 noisy-logreg and 3 lumpy_D10 seeds).

**Measured 2026-09-04 after Stage 2 item 3** (the PyVBMC half:
CMA-ES evaluates each generation's population in one acquisition call and
`vp.pdf` broadcasts over K; `plans/stage2-batched-acquisition.md`
§Results). Same suite, same seed, same machine: the noiseless targets run
1.4–1.8× faster end to end (banana_D4 67 → 43 s, cigar_D4 141 → 79 s,
lumpy_D10 283 → 182 s, banana_D10 175 → 103 s), active sampling is 2.1–2.5×
faster and falls from 54–69 % to 36–47 % of wall, GP training and the
variational stage are unchanged to the second, and the noisy VIQR targets
gain 6–9 % (their active-sampling bucket is the per-sample GP refits and VP
re-optimizations, as recorded above). Six of the nine converging
trajectories are bit-identical to the 2026-09-03 runs. The three stages
are now of comparable size on the noiseless targets (cProfile at D = 4:
active sampling 35–44 %, GP training 24–32 %, variational stage 24–32 %;
nested buckets, not additive), so items 8 and 1 carry relatively more
weight than before; inside active sampling the largest remaining piece is
gpyreg's per-call `predict` overhead over the hyperparameter samples
(23–29 % of the run at 5× fewer calls), which is item 8's. The 15-D
exhaust run (750 evaluations, half of them in the optimize-only regime)
is 1.65× faster end to end (2123 → 1288 s) with active sampling 3.3×
faster (a single hyperparameter sample there, so the call-count reduction
shows in full) and the other stages within ±16 %; a first measurement of
it on a throttling laptop (untouched GP training 2.1–2.5× slower per
iteration after 27 minutes of load) was repeated on a cool machine on the
bit-identical trajectory.

**Measured 2026-09-05 (00:04) after Stage 2 items 1 and 2**
(`_gp_log_joint` vectorized over hyperparameter samples and components,
its variance from two multi-RHS solves per sample;
`plans/stage2-gp-log-joint-einsum.md` §Results). Same suite, same seed,
same machine, speed probe 1.03 (no throttling): the noiseless D ≤ 10
targets run 1.13–1.39× faster end to end (banana_D4 43 → 31 s, cigar_D4
79 → 62, lumpy_D10 182 → 147, banana_D10 103 → 79), the variational fit
is 2.6–5× faster and falls from 7–28 % of wall to 3–12 %, active
sampling and GP training are unchanged within trajectory noise. The 15-D
exhaust run is 1.24× faster (1288 → 1041 s) with its variational fit
2.2× faster (585 → 264 s); its active sampling, untouched, took 20 %
longer on a different trajectory (the search cost of a different run,
unexplained in detail). Combined with item 3 the noiseless targets are
1.5–2.3× faster than the 2026-09-03 baseline (1.9–2.3× except logreg_D5
at 1.5×). Every trajectory changed
(the ELBO arithmetic moved by rounding), every seed-0 final is inside its
population fence. Under cProfile at D = 4, `_gp_log_joint` fell from
18–24 % of the run to about 2 % at the same call counts, `final_boost`
from 10–17 % to 4–6 %, and `entmc_vbmc` (6–9 %) is now the largest piece
of the variational stage; what remains is active sampling (46–54 %,
gpyreg's per-call `predict` overhead 31–34 %) and GP training (31–42 %,
the slice sampler 27–38 %), i.e. item 8.

**Measured 2026-09-05 (10:42) after Stage 2 item 8** (gpyreg PR #43;
`plans/stage2-gpyreg-predict-and-sampler.md` §Results), on the
identity-preserving PyVBMC commit so that every trajectory equals the
00:04 campaign's: the ten configs run 1.14–1.49× faster end to end (the
noiseless ones 1.19–1.49×: banana_D4 31 → 24 s, cigar_D4 62 → 48,
student_D4 36 → 29, lumpy_D10 147 → 98, banana_D10 79 → 54, the 15-D
exhaust run 1041 → 777 s; the noisy VIQR targets 1.14× and 1.24×); GP
training is 2.0–2.6× faster (23–46 % of wall → 13–29 %), active sampling
1.11–1.20× faster (46–85 % of wall). Under cProfile at D = 4, `GP.predict`
is 1.5–1.7× faster per
call and still 30–36 % of the run (its kernel evaluations 15–17 %, the
next `predict` step if wanted), one log-posterior evaluation 2.8–3.3×
faster on average because 47–55 % of the log-posterior evaluations reuse
the Cholesky factor, `SliceSampler.sample` 2.7–3.0× per call; GP training is
18–25 % of the run, the variational stage 15–19 % plus the entropy's
9–13 %. Over Stage 2 so far (items 3, 1, 2, 8) the exhaust run went
2123 → 777 s, 2.7×.

**Measured 2026-09-05 (17:11) after Stage 2 item 5** (`entmc_vbmc`
vectorized; `plans/stage2-entmc.md` §Results), with gpyreg v1.1.0, on
trajectories that all differ from the 10:42 campaign's because the seam
removal moved every seed's stream: the change is visible in the buckets,
not in single-run walls. Under cProfile the Monte Carlo entropy is
1.4–3.0× faster (`entmc_vbmc` 9–13 % → 3–6 % of a D = 4 run, 24 → 18 % of
the exhaust run), and the two call shapes separate cleanly: the
Adam-objective calls (the great majority) are 5× faster per call in situ
(the exhaust run's 147 → 29 s over 20k calls at K = 2–34, D = 15; 1.18 →
0.27 ms, 4.4×, at D = 4), `final_boost` halves on every config, while the
value-only calls of `_eval_full_elcbo` (4096 samples per component) gain
nothing and are now the entropy's dominant cost on large-K runs (132 of
161 s on the exhaust run). In the plain runs the variational fit took
0.37–0.88 of its time on nine of ten configs; the suite wall 1424 →
1384 s, the exhaust run 777 s on both campaigns (its variational fit
244 → 209 s on a new path that spends more time at large K, its GP
training 127 → 189 s from the rare N ≥ 500 refits on that path). The
remaining time at D = 4 is active sampling (61–66 % of a profiled run,
`GP.predict` 34–43 %) and GP training (19–26 %); the variational stage is
8–12 %.

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

*Addendum 2026-09-05.* The line numbers and three of the descriptions in
this section predate Stage 2 items 1 and 2
(`plans/stage2-gp-log-joint-einsum.md`): `_gp_log_joint` no longer has a
`(s, k)` loop, a triangular `(j, k)` loop (pitfall 6) or slice-assigned
gradient buffers (pitfall 7); its `compute_vargrad` path is deleted, so
the "~136 lines" of the table above are now about 100 lines of
broadcasting. The clamps of pitfall 5 remain. The coverage paragraph
below was already stale on 2026-09-02 (see the Stage 0 lines of the
roadmap): finite-difference checks exist for `_gp_log_joint`,
`_neg_elcbo`, `_vp_bound_loss`, `_soft_bound_loss` and `vp.pdf`
(`test_*_grad_fd.py`), not only for the entropies. Since Stage 2 item 5
(2026-09-05, `plans/stage2-entmc.md`) `entmc_vbmc` has no component loop
either: its density and reparameterization gradients are a broadcast over
a `(components, samples, D, K)` tensor with `einsum` contractions, in
blocks, and the density is evaluated once; the softmax Jacobian copies
remain.

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
  **Resolved 2026-09-04 (evening) by deletion** (Stage 2 item 1,
  `plans/stage2-gp-log-joint-einsum.md`): the path was doubly unreachable
  (`compute_var == 2` raises "not implemented" and any other
  `compute_var` with gradients raises before it), so the vectorized
  `_gp_log_joint` drops the accumulators and keeps the two raises with
  their conditions and messages; `dvarG` is always `None`. Porting
  MATLAB's diagonal variance approximation would be a feature, not a fix.
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
  per sample at `Nc = 8192`). **Fixed 2026-09-05** (Stage 2 item 8, gpyreg
  PR acerbilab/gpyreg#43, `plans/stage2-gpyreg-predict-and-sampler.md`):
  `sW * Ks` broadcasts, bit-identically.
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
  it would make `dF = -dG - dH` a shape mismatch. **Fixed 2026-09-04**
  (item 1): all four blocks are returned, the Jacobian corrections alone
  are conditional; finite-difference test in the raw `(mu, sigma, lambd,
  w)` coordinates. The entropies (`entlb_vbmc`, `entmc_vbmc`) already
  returned every requested block at `jacobian_flag=False` (their blocks
  are allocated by `grad_flags` and concatenated unconditionally; the flag
  switches only the three Jacobian corrections), so the three functions
  agree since the fix. *Corrected 2026-09-05 (item 5)*: this bullet said
  until then that the entropies returned the `mu` block only; the
  finite-difference wrapper in `test_entmc_vbmc.py`, which calls with
  `jacobian_flag=False` and compares the full `D K + K + D + K` gradient,
  shows otherwise.
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
- **Found 2026-09-04 in the golden population**, not fixed, decision
  deferred (own devlog: `2026-09-04-final-boost-failure.md`): `final_boost`
  accepts the re-optimized K = 50 posterior unconditionally, as MATLAB's
  `finalboost_vbmc.m` does, and on `student_D4` seed 19 turned a converged
  posterior (ELBO within 0.02 of ln Z, gsKL 0.06) into ELBO −9.03 ± 0.49
  for ln Z = −10.36 and gsKL 54, under a GP whose mean function had gone
  flat; 4 of 6 boost reruns from the same state fail. Algorithmic and
  inherited, not a port bug.
- **Found 2026-09-04 by the review of the oracle plan**
  (`plans/fixture-generator-and-oracles.md`), not fixed:
  - `vbmc.py:755` sets `optim_state["variance_regularized_acqfcn"]` but
    `AbstractAcqFcn.__call__` reads `optim_state.get(
    "variance_regularized_acq_fcn")` (with the underscore), so the
    variance-regularization branch of every acquisition function is dead in
    every run, and `tol_gp_var` (read only inside that branch) has no
    effect at all. The underscore spelling appears only in acquisition unit
    tests; `testing/vbmc/test_vbmc_init.py` asserts the misspelled key, so
    fixing the typo means updating that test. The oracle fixtures pin the
    dead path as it is.
  - `_gp_log_joint(..., compute_var=False, separate_K=True)` raises
    `UnboundLocalError`: `J_sjk` is created only under `if compute_var` but
    returned unconditionally in the `separate_K` branch. Unreachable in
    production only because the one caller that could hit it,
    `_eval_full_elcbo`, drops the variance solely under a
    `skip_elbo_variance` option that no `.ini` defines and that
    `validate_option_names` therefore rejects; `_neg_elcbo`'s own
    `separate_K`-without-variance fallback (`J_sjk = None`) is dead code for
    the same reason. **Fixed 2026-09-04** (item 1): `J_sjk = None` is
    returned in that case; unit test added.
- **Found 2026-09-04 by the review of the `_gp_log_joint` rewrite**
  (`plans/stage2-gp-log-joint-einsum.md`), fixed in the rewrite: the
  softmax Jacobian in the old `_gp_log_joint` was
  `-np.exp(vp.eta).T * np.exp(vp.eta)`, an outer product only for a
  `(1, K)` `eta`; `test_gp_log_joint` passes a 1-D `eta`, for which the
  expression broadcasts to a row and the off-diagonal entries are wrong,
  and the test passed only because the default `eta` is uniform. No run
  is affected (`_neg_elcbo` reshapes `eta` to `(1, K)`). The rewrite uses
  `np.outer`; `_neg_elcbo`'s own copy of the Jacobian (also on a `(1, K)`
  `eta`, so correct) keeps the old form.
- **Found 2026-09-04 by the same review**, not fixed (inert):
  `optimize_vp` prunes `J_sjk` along `axis=2` only
  (`variational_optimization.py:381`), so `vp.stats["J_sjk"]` is no
  longer square after a component is pruned; nothing in the package reads
  that key.
- **Found 2026-09-04 by the reviews of the batched-acquisition plan**
  (`plans/stage2-batched-acquisition.md`), not fixed:
  - `testing/vbmc/test_vbmc_optimize.py:630` asserts `elbo_1 == elbo_1`
    (a self-comparison), so `test_vbmc_resume_optimization` does not pin
    the resumed ELBO at all.
  - The dead variance-regularization block in `AbstractAcqFcn.__call__`
    (`abstract_acq_fcn.py:121-127`, unreachable because of the misspelt
    option key above) would also fail on any batch of more than one point
    when `acq` is still 2-D, which is the case for VIQR/IMIQR with a single
    GP sample: `acq[mask] += ...` with a `(k, 1)` left side and a `(k,)`
    right side. Fixing the key alone would expose it.
  - `AbstractAcqFcn._sq_dist` (`abstract_acq_fcn.py:214-216`) centres both
    point sets on a mean that depends on the size and content of the
    candidate batch, so the squared distances, and the nearest-training-
    point `argmin` in `_estimate_observation_noise` built on them, are not
    batch-invariant (2e-15 relative). Not a bug, but it is why the noisy
    acquisitions evaluated one point at a time and in a batch can differ by
    a finite amount on a near-tie, on top of BLAS rounding.
  - `AbstractAcqFcn._real2int` snaps its input in place, and through the
    `Xs[None, :]` view of a 1-D input the pointwise CMA-ES objective was
    snapping CMA-ES's own solution arrays to the integer grid (an
    undocumented side effect that the batched objective now reproduces
    deliberately).
- **Found 2026-09-05 while building the `gp_fit` oracle and reviewing the
  item 8 plan** (`plans/stage2-gpyreg-predict-and-sampler.md`):
  - `gaussian_process_train.py: _get_hyp_cov`, the weighted branch that
    `weighted_hyp_cov = True` (the default) selects, has two slips: it
    appends `gp_hyp_full[i].T` and then takes `np.shape(hyp_list)[1]` as
    the sample count (:717, :729), so `hyp_cov` comes out `(Ns, Ns)`
    instead of `(hyp_N, hyp_N)`, and its `np.dot` of 1-D rows (:731-735)
    is a scalar, so every entry is the same number. `train_gp:121-124`
    then drops the slice-sampler widths whenever `Ns ≠ hyp_N`, which held
    in every fit measured (all eight oracle snapshots): production
    slice-samples with gpyreg's `widths_default`. When `Ns == hyp_N`
    (reachable at D = 2: `hyp_N = 9`, `Ns = 9`) a degenerate constant
    width vector *is* used. **Not fixed**: fixing changes the sampler's
    proposal widths and therefore every trajectory; it needs its own item
    with a population check. The `gp_fit` oracle asserts the drop so a fix
    is noticed there.
  - gpyreg `GP.log_likelihood` / `GP.log_posterior` with `compute_grad=True`
    applied the unary minus to the returned `(nlZ, dnlZ)` tuple and raised
    `TypeError` (no PyVBMC caller; gpyreg's own gradient test bypasses the
    wrappers). **Fixed in acerbilab/gpyreg#43** (they return `(value,
    gradient)`).
  - gpyreg `SliceSampler` with `step_out=True` (off in PyVBMC and by
    default): `x_l` / `x_r` are copied from `xx` once per sweep
    (`slice_sample.py:387-388`) and never re-synced after `xx[dd] =
    xprime[dd]`, so the step-out evaluations of later axes carry stale
    coordinates from earlier ones (off the coordinate line). Inert for
    PyVBMC; not fixed.
  - A trap, not a defect: after `_gp_hyp` installs PyVBMC's hyperprior on
    a GP and before `GP.fit` runs, `gp.log_posterior` is NaN (`_gp_hyp`
    leaves NaN bounds, so the normalization constants are NaN until
    `fit` fills the prior's `df` and the bounds, in that order). The
    `gp_nlZ` oracle replicates `fit`'s two repairs.

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
the rest. **This order is provisional** (resolved by the measurements
below, 2026-09-03/04): it was measured on two easy Gaussian
targets and must be re-checked on the benchmark target suite below before
the first vectorization PR.

*Measured 2026-09-03/04 on the regenerated benchmark suite* (papers'
procedure: random start in the prior box, the papers' priors and budgets;
`plans/benchmark-suite-and-golden-traces.md` §Results (regenerated). The
withdrawn first pass of 2026-09-02 gave the same order with shares within a
few points, so the correction repaired the procedure, not the conclusion.)
Decision rule applied: the variational stage grew with correlation (16–20 %
on cigar and logreg) but overtook nothing, so the order stands, **3 → 8 →
1 → 2**, with three refinements. (i) Items 8 and 1 are close at D = 4 (GP
training 13–20 % vs variational 9–20 %) and item 1 is PyVBMC-local while
item 8 is a gpyreg PR, so item 1 may be done first for logistics; but item
8's weight rises with dimension: GP training is 22–28 % at D = 10 against
4–7 % for the variational stage, and in the late sampling regime of the
15-D cigar run (N 305–345, Ns 5–4) the slice sampler is 41 % of an
iteration against 35 % for active sampling. Concrete targets inside item 8:
`__core_computation` 10–24 % of profiled time, the scipy `solve_triangular`
validation wrappers 6–10 % (0.4–9.4 M calls per run), the hyperprior
evaluation 1.5–6 %; the Cholesky itself is 1.2–3.2 %. (ii) On the noisy
(VIQR) path the active-sampling bucket is dominated by the per-sample full
GP refits and VP optimizations rather than by the acquisition search, so
items 8 and 1 are what speed up noisy targets. (iii) The budget-exhausting
run (cigar D = 15, 750 evaluations, Ns = 0 from N = 350 for 81 of 150
iterations) answers the rule's second clause in the negative: in the
optimize-only regime GP training is 6 % of an iteration (median 0.29 s,
most iterations reusing the previous hyperparameters through
`gp_retrain_threshold`; one full refit of 35 s at N = 560), active sampling
61 % (one hyperparameter sample, growing 6.5 → 7.7 s per iteration with N)
and the variational stage 32 % (1.3–20 s at K 23–31); `scipy.optimize.
minimize` is 1.7 % of that run and 0.1 % of every converging run, so the
L-BFGS-B path does not join item 8, though the rare refit spike at N ≥ 500
is real. Over the whole 15-D run the variational stage (24 % of wall)
exceeds GP training (17 %) because more than half the iterations are
optimize-only, and inside it `entmc_vbmc` (item 5, 11 %) is the largest
piece ahead of `_gp_log_joint` (9 %); `vp.pdf` reaches 15 % with K ≈ 25
(part of item 3). `copy.deepcopy` is under 1 % everywhere: the
`iteration_history` copies are a memory problem, not a time problem.

*Item 3 done 2026-09-04 for the PyVBMC half* (`plans/stage2-batched-
acquisition.md`): `active_sample` hands `cma.fmin` the acquisition as
`parallel_objective` and batches the noise handler's re-evaluations, so
the acquisition is evaluated twice per generation instead of `popsize + 2`
times; `vp.pdf` broadcasts over K. Same algorithm and random stream (with
per-row evaluation inside the batch call the search reproduces the stored
points bit-for-bit); the batched arithmetic differs by a few ulp and flips
some CMA-ES rankings, which the `active_sample_step` oracle absorbed by a
targeted re-baseline. Measured 1.4–1.8× end to end on noiseless targets
(§2). The gpyreg half (`predict`'s Python loop over the hyperparameter
samples and its `sW` tiling) joins item 8, so that all gpyreg changes land
in one PR (the order of the remaining items is the roadmap's).

*Items 1 and 2 done 2026-09-04 (evening)*
(`plans/stage2-gp-log-joint-einsum.md`): the `(s, k)` loop of
`_gp_log_joint` is broadcasting over an `(Ns, K, D, N)` array of
standardized distances with `einsum` contractions against `alpha`
(measured faster than batched `matmul` once the array leaves the cache,
and it never materializes `delta²`), and the `(j, k)` variance loop is,
per sample, two multi-RHS triangular solves plus one `K × K` product
whose entries are the old summands in a different contraction order
(chosen over the Gram form `VᵀV` because `J = Jbase − C` cancels). Item 2
was done here because its loop lived inside the loop item 1 removed.
Same formulas: the oracles held with no re-baseline on every commit, the
finite-difference tests pass, the old and new code agree to 1e-15 on
well-conditioned states and to 1e-9–2e-7 per element on the
ill-conditioned oracle snapshots (the GP solve's conditioning, the same
order as the cross-BLAS floors the tolerance classes were set from).
Because `G` and `dG` feed Adam and the sieve, every trajectory parts at
iteration 0; on cigar_D4 seed 0 a one-ulp perturbation of the old code's
gradient moves the iteration-0 ELBO by 0.4, so the replay's initial-design
check now reads the design from the trace (`X_init`) instead of demanding
an identical iteration-0 ELBO. Speedup in §2 above. Four latent defects of
the function were fixed on the way (§9).

*Item 8 done 2026-09-05 as one gpyreg PR* (acerbilab/gpyreg#43, merged
into gpyreg `main` the same day as `a2f8ddc` after four review fixes;
`plans/stage2-gpyreg-predict-and-sampler.md`), **identity-preserving
throughout**: `predict` drops scipy's Python layers around a 5 µs
triangular solve (a direct `trtrs` with scipy's own layout rule, which
the measurements showed to be bit-identical where `potrs` and a plain
`potrf` are not), the `sW` tiling and the per-sample mean calls (a
`compute_batched` on the mean functions, whose columns equal the
per-sample values because numpy's last-axis reduction does not depend on
leading dimensions); the sampler's log-posterior evaluation uses
`cdist(Xs, Xs)` for the symmetric kernel (bit-equal to `squareform(pdist)`),
adds the noise in place on the diagonal, caches the hyperprior's type
masks, and reuses the Cholesky factor when the sampler moves a
mean-function hyperparameter (two thirds of the coordinates), the
gradient path excluded; the two public gradient wrappers that raised
`TypeError` are fixed (§9). Every output checked is bit-identical to a
dump of the pre-change code (2219 random-GP arrays, the eight oracle
snapshots through every oracle incl. the new `gp_nlZ` and `gp_fit`), the
golden replay is `identical` after each of the three performance commits,
PyBADS's suite passes against the branch (one metadata-dependent test
deselected). Per call: `predict` 1.4–1.7× at CMA-ES batch sizes, the
log-posterior evaluation 1.4–1.9×, one `train_gp` call 2.1–2.5×; what
remains in an evaluation is the N² exponential of the kernel and the
factorization. End to end (§2, campaign of 10:42 on the
identity-preserving commit, identical trajectories): 1.14–1.49× over
the ten configs (noiseless 1.19–1.49×), GP training 2.0–2.6×, active
sampling 1.11–1.20×. The same PR adds `rng=` to
`GP.fit`, `SliceSampler`, `f_min_fill` and `GP.random_function`
(`None` keeps today's legacy draws call for call), and PyVBMC's seam of
Stage 1 is removed: `train_gp` hands `vbmc.rng` to `gp.fit`, the CMA-ES
noise-handler subclass draws its re-evaluation count from `vp.rng`, and a
run never reads or writes NumPy's global state (a seeded run's stream
shifts by one draw relative to before, since the removed reseed drew one
integer from `vbmc.rng`; the `gp_fit` and `active_sample_step` oracles
were re-baselined for the stream change). The identity gate was found to
need a *dump* of the pre-change oracle outputs rather than the committed
references, which items 3, 1 and 2 had already moved within tolerance;
the generator gained `--dump-outputs` / `--check --exact --against`,
`--add-oracle` and `--expect-moving`.

*Item 5 done 2026-09-05 (afternoon)* (`plans/stage2-entmc.md`):
`entmc_vbmc` draws every component's antithetic samples with one
`standard_normal((K, Ns / 2, D))` call (bit-identical to the
per-component draws, so the random stream is untouched and the `entmc`
oracle holds at 1e-10 without a re-baseline), evaluates the mixture
density once as a broadcast over a `(components, samples, D, K)` tensor
of standardized distances with `einsum` contractions, and takes the
reparameterization gradients from the same tensors, in blocks of 2^16
elements (cache residency, measured as item 3 measured it for `vp.pdf`;
blocks of samples within a component for the 4096-samples-per-component
calls of `_eval_full_elcbo`, whose tensor would otherwise reach 328 MB at
K = 50). The loop had evaluated the density twice per component (once
for the value, once inside the gradient block) with a `K × K` Python
loop for the first; the profile of one call put 56–97 % of the time
there and 1–9 % in the draws. Same formulas, rounding-level differences:
the eight oracle snapshots agree with the loop to 1e-14, and the exact
check against a pre-change dump moves only the entropy-carrying outputs
(`entmc`, and `F`, `dF`, `H`, `F_full`, `H_full` of `neg_elcbo`), by
1e-15. Per call 6.7–12.5× faster at the shape Adam sees (K = 14–50 at
D ≤ 15, 24 → 1.9 ms at K = 50; 3.8× at the D = 20, K = 60 corner) and
0.9–1.35× at the value-only shape of `_eval_full_elcbo` (within ±10 % of
the loop at D = 4, K ≤ 17, 1.2–1.35× at K ≥ 25), where the arithmetic is
the floor (a centered GEMM
expansion of the squared distances would make that shape 2.8–6.9× faster,
but its error grows without bound with the squared width ratio of the
components, the Gram-form objection of item 2, and a broad component next
to a narrow one is an ordinary VBMC state; rejected by the PI, plan Open
question 1). Replay against the item 8 traces: every config parts at
iteration 0 or 1 with the initial design identical and the finals inside
the population envelope; full suite green. End to end: the campaign of
the same evening (§2).

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
