# CMA-ES settings side by side on the stored oracle states

Measurement for findings F1, F5 and F6 of `../../reviews/P2_comparison.md`
(the local acquisition search passes three CMA-ES settings differently from
MATLAB VBMC), by a fresh Opus agent, 2026-09-19, on `dev-port-review`. No
tracked file was changed. `harness.py` captures the `cma.fmin` call of one
real `active_sample` step per state and reruns the search under the
variants; `probe.py` checks the meaning of `CMA_stds` and builds the state
table; `run_grid.py` runs the grid and prints the per-state tables;
`analyze.py` checks that the instrumentation is inert and produces the
pooled and scale-free summaries; `runs.csv` holds the 800 per-run records
(`state, D, seed, variant, f_ret, f_x0, improved, n_evals, n_iters, wall,
dist_insigma, mean_used, lastgen_aligned, stop`). The logs are under the
gitignored `dev/scripts/runs/port_review_20260919/cmaes_side_by_side_logs/`.
Run the scripts from the repository root with the project interpreter and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

The text below is the agent's report. Its per-state tables (acquisition
value, evaluations, wall time and improvement over the sieve for each of
the eight states) are left out here because `run_grid.py` regenerates them
from `runs.csv`; everything else is unedited.

---

## The three settings and the variants

- **S, step sizes.** MATLAB passes the vector `insigma = sqrt(diag(Sigma))`
  to its CMA-ES, which uses `max(insigma)` as the overall step size and
  `insigma / max(insigma)` as per-coordinate scaling, so the initial
  sampling covariance is `diag(insigma**2)`. PyVBMC passes only the scalar
  `np.max(insigma)`, so the search starts isotropic at the largest
  coordinate scale. In `cma` the MATLAB start is `CMA_stds = insigma /
  max(insigma)`.
- **N, noise handling.** PyVBMC passes a noise handler to `cma.fmin`;
  MATLAB runs with `Noise.on = 0`.
- **B, returned point.** PyVBMC takes the best solution ever evaluated;
  MATLAB's default returns the best point of the last generation, or the
  final mean if that is better.

Variants, each with the same 20 seeds per state: `current`; `S`; `N`; `B`;
`SNB` (all three, i.e. MATLAB's settings).

## 1. How the local search was isolated

For each fixture I rebuild the state with `pyvbmc/testing/oracles/_state.py:build_state` (the real benchmark target attached, as `test_oracles.py` does), set `vp.rng = default_rng(20260904)` (the oracles' `DEFAULT_SEED`) and call the real `active_sample(gp, 1, optim_state, fl, history, vp, options)` inside `legacy_seed(seed)` — exactly the `active_sample_step` recipe with `sample_count=1`. Before the call I monkeypatch `sys.modules["pyvbmc.vbmc.active_sample"].cma.fmin` (in-process only, restored in a `finally`) with a stub that records `objective_function` (the `acq_fun` closure), `x0`, `sigma0`, the `options` dict, `parallel_objective` and `noise_handler`, then raises a private exception. `active_sample.py` has no `try/except` around the optimizer, so the exception aborts the step before any target evaluation and before `optim_state` is written further. The captured `acq_fun` keeps the GP, VP, logger and `optim_state` alive, so the acquisition I evaluate afterwards is production's, including the VIQR importance samples that `active_sample` had already drawn into `optim_state["active_importance_sampling"]`.

`insigma` is not passed to `cma.fmin`, so I recompute it from the same objects exactly as `active_sample.py:510-516` and assert `float(np.max(insigma)) == sigma0` of the captured call (holds exactly for all 8 states). `optim_state["integer_vars"]` is all-False in every state, so the acquisition's in-place snapping never fires.

Each variant then calls `cma.fmin` myself with the captured arguments, replacing only:
- `options["randn"]` by `lambda *shape: rng.standard_normal(shape)` over a fresh `np.random.default_rng(seed)` — the same construction `active_sample.py:526` uses, so the CMA-ES population stream is seeded per run and variants are paired by seed;
- `options["CMA_stds"] = insigma / max(insigma)` for the S variants;
- `noise_handler=None` for the N variants (otherwise a fresh `_BatchedNoiseHandler(x0.size, acq_fun, rng)` over the same generator).

The problem (x0, Sigma/insigma, search bounds, tolfun, maxfevals, acquisition, importance samples) is captured **once per state** and held fixed; the seed varies only the CMA-ES stream. Variants `current` and `B` read the same optimizer run (identical by construction), so per state and seed there are 4 optimizer runs: (stds, noise) ∈ {(0,0),(1,0),(0,1),(1,1)}.

**Instrumentation is inert.** To extract MATLAB's point I wrap the objective (recording each generation's `(X, fit)`) and pass a `callback` that reads `es.fit.idx[0]`. On 3 states × 3 seeds, the wrapped+callback run returns a bit-identical `x_best` and identical evaluation and generation counts to a bare `cma.fmin` call with the same seed and no wrapper.

Acquisition values are always re-evaluated pointwise at the returned point (`acq_fun(x)`), never read from cma's bookkeeping. `f(x0)` is likewise the pointwise re-evaluation at the sieve's best point.

## 2. What I verified in the `cma` source (cma 4.4.4)

**`CMA_stds`** (`evolution_strategy.py:1278-1289`, `options_parameters.py:57`): `CMA_stds` initializes `self.sigma_vec`, and `ask_geno` draws `ary = self.sigma_vec * self.sm.sample(Niid)` (`:1993`), so the sampling standard deviations are `sigma * sigma_vec * sqrt(diag(C))`. Empirical confirmation with `insigma = [1.0, 0.01, 0.3]`, `sigma0 = max(insigma) = 1.0`, 200 000 draws from the initial distribution:
- `CMA_stds = insigma/sigma0` → per-coordinate SD `[1.00348, 0.00999, 0.30002]` = `insigma`;
- without it → `[1.00348, 0.99976, 1.00008]` = isotropic at `max(insigma)`, which is what PyVBMC does today.

`CMA_diagonal_decoding` defaults to `0`, so `sigma_vec` is a `DiagonalDecoding` object that never adapts — the anisotropy stays a fixed multiplier while `C` starts at `I` and adapts on top of it. MATLAB instead puts `diagD = insigma/max(insigma)` into `C` itself (`cmaes_modded.m:561-567`), which then adapts. The *initial sampling covariance* is `diag(insigma**2)` in both; the adaptation dynamics are not identical.

**Noise handler** (`evolution_strategy.py:5054-5067`, `:5100-5124`): any non-`None` `noise_handler` sets `noise_handling = True`, which sets `es.opts['tolfacupx'] = inf`, multiplies `es.sigma` by the handler's return every generation, adds `noisehandler.evaluations_just_done` to `es.countevals`, and shrinks `es.sp.cmean` by `exp(-kappa*tanh(noiseS))`. With `NoiseHandler(N)` defaults, `lam_reev = 2 + popsize/20` solutions are re-evaluated per generation. Measured: median evaluations per generation `8.467` with the handler vs `6.044` without (the latter equals popsize), i.e. exactly the predicted `2 + popsize/20 ≈ 2.3-2.4` extra evaluations per generation.

**Returned point** (`evolution_strategy.py:3255` `_result0`, `:5193` `return es._result0 + (es.stop(), es, logger)`): `res[0]/res[1]` are `es.best.get()`, the best solution ever evaluated. `opts['eval_final_mean']` defaults to `True` and `objective_function` is callable here, so `fmin` already evaluates the bounds-repaired final mean and folds it into `es.best` (`:5131-5137`); the final mean is therefore *inside* PyVBMC's best-ever too. MATLAB's default (`cmaes_modded.m:1708-1721`, `SearchCMAESbest = 'no'`, `activesample_vbmc.m:285-288`) is `xmin = arxvalid(:, fitness.idx(1))` — the best point of the last generation — replaced by `xintobounds(xmean,…)` when the final-mean evaluation is better. I reproduce that as: the last generation's `X[es.fit.idx[0]]` (with `fit.idx = argsort(fit_plus_pen)`, `:2648`; BoundTransform adds no penalty, and I assert per run that this index is the argmin of the generation's fitness — 0 misalignments in 800 records), compared against `res[5]` (`xfavorite` = `gp.pheno(mean, into_bounds=repair)`), lower wins.

A consequence worth stating: **B can never beat `current` by construction**, since MATLAB's point is one cma also evaluated and `current` is the minimum over all of them. In 1 of 160 seeds B measures 3.0e-9 lower (2.4e-10 relative) — that is the batched-vs-pointwise evaluation gap of the acquisition flipping a near-tie, not an improvement.

## 3. States used (all 8 stored fixtures; every one has D >= 2)

All have `search_cmaes_vp_init = True` (so `insigma` comes from the VP covariance in transformed space) and `search_cmaes_best = False` — i.e. these states' own options ask for MATLAB's return rule, which PyVBMC ignores. `search_cache_frac = 0` everywhere.

| state | D | K | N train | Ns | noisy | acquisition | insigma | aniso (max/min) | sigma0 | maxfevals | tolfun | f(x0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cigar_D4_boosted | 4 | 50 | 115 | 7 | no | AcqFcnLog | [4.1428, 0.7924, 0.9894, 0.9934] | 5.228 | 4.143 | 3000 | 1e-2 | 19.946 |
| cigar_D4_largeK | 4 | 14 | 115 | 7 | no | AcqFcnLog | [4.1025, 0.7865, 0.9778, 0.9966] | 5.216 | 4.102 | 3000 | 1e-2 | 20.071 |
| corr_D5_warped | 5 | 17 | 100 | 8 | no | AcqFcnLog | [1.182, 1.1606, 1.1378, 1.1205, 0.8051] | 1.468 | 1.182 | 3500 | 1e-2 | 19.507 |
| halfnormal_D2_bounded | 2 | 9 | 70 | 10 | no | AcqFcnLog | [0.1904, 0.2181] | 1.145 | 0.2181 | 2000 | 1e-2 | 12.839 |
| normal_D2_K1 | 2 | 1 | 20 | 8 | no | AcqFcnLog | [0.1291, 0.2364] | 1.832 | 0.2364 | 2000 | 1e-2 | 15.795 |
| normal_D2_singlesample | 2 | 5 | 65 | 1 | no | AcqFcnLog | [0.1247, 0.2493] | 2.000 | 0.2493 | 2000 | 1e-2 | 2203.14 |
| normal_D2_warmup | 2 | 2 | 20 | 8 | no | AcqFcnLog | [0.1293, 0.2410] | 1.863 | 0.2410 | 2000 | 1e-2 | 15.762 |
| rosenbrock_D2_noise1_viqr | 2 | 2 | 25 | 8 | yes | AcqFcnVIQR | [0.0882, 0.1323] | 1.500 | 0.1323 | 2000 | 1e-2 | 4.5557 |

20 seeds (1..20) per state per variant; 800 records, 640 optimizer runs; total grid wall time under 60 s. Every one of the 800 runs stopped on `tolfun`; the largest evaluation count anywhere is 1021 against `maxfevals` of 2000-3500. **The budget never binds on these states.**

## 6. Pooled (160 state×seed pairs, paired against `current`)

`lower` counts the seeds in which the variant reaches a strictly lower acquisition value; `sign p` is a two-sided exact sign test on the non-tied pairs; `dist` is the distance between the returned points in units of `insigma`; `improved` counts the searches that beat the sieve's starting value.

| variant | lower than current | sign p | median paired diff | median &#124;diff&#124; | median evals ratio | median wall ratio | improved/160 | median dist (insigma) |
|---|---|---|---|---|---|---|---|---|
| S | 80/160 | 1.00 | −5.7e-10 | 3.03e-6 | 0.902 | 0.901 | 139 | 0.0030 |
| N | 88/160 | 0.236 | −6.5e-8 | 3.77e-6 | 0.730 | 0.618 | 130 | 0.0049 |
| B | 1/160 | 2.0e-21 | 0 | 0 | 1.000 | 1.000 | 129 | 0 |
| SNB | 70/160 | 0.133 | +1.4e-7 | 1.90e-5 | 0.658 | 0.558 | 134 | 0.0053 |
| current | – | – | 0 | 0 | 1.000 | 1.000 | 129 | 0 |

Pooled medians: generations 32.5 (current) / 31.0 (S) / 33.0 (N) / 32.0 (SNB); evaluations per generation 8.467 (current, S) vs 6.044 (N, SNB); evaluations per search 268.5 / 259 / 199 / 193; wall 0.063 / 0.061 / 0.040 / 0.038 s.

Scale-free view — paired median difference as a fraction of what the local search buys over the sieve (`gain = median(f(x0) − f_current)`), with the seed-to-seed IQR of `f_current` for comparison:

| state | gain | seed IQR of f_current | dS/gain | dN/gain | dB/gain | dSNB/gain |
|---|---|---|---|---|---|---|
| cigar_D4_boosted | 0.0798 | 9.73e-2 | −4.1e-4 | −9.1e-4 | +1.3e-4 | −2.5e-4 |
| cigar_D4_largeK | 0.0210 | 2.32e-1 | −1.3e-2 | −2.0e-3 | +1.1e-4 | −1.3e-2 |
| corr_D5_warped | 0.3932 | 1.14e-4 | +2.4e-4 | +6.8e-6 | +7.7e-5 | +1.1e-4 |
| halfnormal_D2_bounded | 0.0182 | 5.04e-7 | +2.6e-6 | −1.9e-6 | +1.3e-6 | +1.0e-5 |
| normal_D2_K1 | 0.0052 | 1.04e-6 | +8.2e-6 | −4.7e-6 | 0 | −1.5e-5 |
| normal_D2_singlesample | 1.1251 | 3.51e-7 | −4.2e-9 | +7.4e-9 | 0 | +1.8e-7 |
| normal_D2_warmup | 5.76e-4 | 5.65e-7 | −1.7e-6 | −7.1e-5 | 0 | +3.8e-4 |
| rosenbrock_D2_noise1_viqr | −1.55e-4 | 3.92e-4 | +0.204 | +0.054 | 0 | +3.28 |

(The last row's ratios are against a *negative* gain — on that state the search usually fails to beat the sieve at all, so the ratio is not meaningful.)

How often each variant's returned point differs from `current`'s at all (out of 20 seeds): S and N and SNB, 20/20 on every state; B, 11 / 10 / 13 / 11 / 9 / 9 / 5 / 8 (76/160 pooled). MATLAB's rule falls back to the final mean rather than the last generation's best in 10-18 of 20 seeds per state.

## 7. Plain reading

**S (per-coordinate initial step sizes).** It changes the point returned on every seed of every state, sometimes substantially (median 1.09 and 2.13 `insigma` away on `cigar_D4_largeK`), but it does **not** measurably change the acquisition value reached: 80/160 seeds lower, sign p = 1.00; per state the win rate ranges 7/20 to 14/20 and no state is significant at 0.05. The paired medians are between −1.3% and +0.02% of what the search buys over the sieve, and on every state smaller than the seed-to-seed IQR of `current`. What S does change is convergence speed on the anisotropic states: on the two cigar states (aniso 5.2) the median generation count drops 69.5→52.0 and 65.5→52.5, evaluations to 0.78 of `current`, and the rate at which the search beats the sieve rises from 15/20 to 19/20 and from 11/20 to 19/20. On the six near-isotropic states (aniso 1.14-2.00) it changes evaluations by 0-9% and nothing else. Pooled: 10% fewer evaluations and 10% less wall time.

**N (drop the noise handler).** Also changes the point on every seed and, like S, does not measurably change the acquisition value: 88/160 lower, sign p = 0.24. One state out of eight reaches 15/20 (`cigar_D4_boosted`, p = 0.041), which with eight states and no multiplicity correction I read as noise. Its cost effect is unambiguous and exactly the size predicted: the handler adds `2 + popsize/20` ≈ 2.4 evaluations per generation (8.467 vs 6.044 evaluations per generation), while the number of generations is essentially unchanged (pooled median 32.5 vs 33.0) — so the handler's sigma and `cmean` adjustments do not lengthen or shorten the search, they only add evaluations. Dropping it cuts evaluations to 0.73 and wall time to 0.62 of `current`. The disabled `tolfacupx` criterion never mattered here: all 800 runs stop on `tolfun`.

**B (MATLAB's returned point).** By construction it cannot beat `current`, and it does not: 1/160 "wins" is a 3.0e-9 flip from the batched-vs-pointwise acquisition evaluation. It returns a different point than `current` in 76/160 seeds; in the other 84 the best-ever point *is* the last generation's best or the final mean. Where it differs, the cost is small — paired medians 0 to 3.0e-5 absolute, 0.007% to 0.02% of the search's gain, always below the seed IQR of `current` on that state — and the rate at which the search beats the sieve is unchanged (129/160 for both). Evaluations and wall time are identical by construction. So B is a pure, small give-back in acquisition value in exchange for matching MATLAB and for making the shipped `search_cmaes_best` option (False in all eight states) mean something.

**SNB (all three, i.e. MATLAB's settings).** No systematic difference in the value found over the eight states: 70/160 lower, sign p = 0.13; per-state win rates 4/20 to 14/20. The one state where the difference exceeds seed noise is the noisy VIQR state, where SNB is *worse* in 16/20 (p = 0.012), paired median +5.1e-4 against a `current` seed IQR of 3.9e-4 — but on that state the search runs only 4-6 generations and beats the sieve in 1-3 of 20 seeds, so `active_sample` would discard the result most of the time either way. Cost: median evaluations 0.66 and wall time 0.56 of `current` (0.50-0.75 per state).

In one sentence: on these states the three settings are, to within seed-to-seed noise, equivalent in the acquisition value the local search reaches; they differ mainly in cost (MATLAB's settings are about a third cheaper) and in which point comes back (S, N and SNB return a different point on every single seed, occasionally one to two `insigma` away at the same acquisition value, and B on about half the seeds).

Whether a lower acquisition value, or a different acquired point at the same acquisition value, improves inference is outside this measurement.

## 8. Limitations

- **Anisotropy range.** `insigma.max()/insigma.min()` on these states is 1.14 to 5.23 — the transform and the warp have already removed most of the scale disparity by the iterations that were snapshotted. The consequence argument for S is about strongly anisotropic searches; nothing here measures that regime, and the only two states above aniso 5 are where S shows its clearest (cost-side) effect.
- **The budget never binds.** Maximum 1021 evaluations used against `maxfevals` of 2000-3500; every run stopped on `tolfun`. The claim that an isotropic start "will not fully recover" within `maxfevals` cannot be tested on these states.
- **`tolfun = 1e-2` in every state** (all acquisitions here have `log_flag`), against `f` values of 4.6 to 2202. The searches therefore stop at a coarse tolerance and the variants have limited room to differ; on the VIQR state this means 4-6 generations total.
- **One search instance per state.** x0, Sigma, bounds, the acquisition and the VIQR importance samples are fixed per state (captured under the oracles' seed 20260904); the 20 seeds vary only the CMA-ES stream. Per-state conclusions rest on 20 seeds of one instance, not on 20 instances.
- **`CMA_stds` is MATLAB's start, not MATLAB's search.** It reproduces the initial sampling covariance `diag(insigma**2)` exactly, but the anisotropy sits in a non-adapting `sigma_vec` (with `CMA_diagonal_decoding = 0`) while `C` starts at `I`, whereas MATLAB puts it in `C` via `diagD` and lets it adapt.
- **`cma` is not `cmaes_modded`, and only these three settings were varied.** Other differences remain in both arms (notably `TolX`, relative to `max(insigma)` in MATLAB and an absolute 1e-11 in `cma`, plus active-CMA and damping constants). No MATLAB was run; the MATLAB side is read from source only.
- **`f(x0)`** here is the pointwise re-evaluation at the sieve's best point; `active_sample` compares against the batched sieve value, which differs by roughly 1e-15 relative. This can flip the `improved` flag on a near-tie.
- **Coverage of the acquisition set:** seven states use `AcqFcnLog` and one `AcqFcnVIQR`. IMIQR, the other importance-sampling acquisition, is not represented, and no D = 1 state exists (that path is Nelder-Mead anyway).
- `rosenbrock_D2_noise1_viqr` has `active_sample_gp_update`/`vp_update` on; the capture aborts at the search, before that block, so the captured problem for the first of the five acquisitions is production's, but the later intra-step searches on that state (which would see an updated GP/VP) are not measured. Only the first acquisition of a step is measured on every state.
