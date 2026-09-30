# Wave 7 verification, slice O1: the expected log joint, the ELCBO and their gradients

Raw report of a verifier, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent,
read-only on the repositories, verified the findings of the third reader of
slice O1 (`../reviews/O1_third_reader.md`) and its answers to the first
question of its brief, reading PyVBMC's package code at `a65b96f4`, gpyreg
at `v1.3.0` (`0186d89`) and MATLAB VBMC at `396d649`, with no `optimize()`
run. Its scripts are kept as `scripts/wave7_O1_*.py`, under the names the
report gives them with `wave7_` in front; its helper `o1common.py`, which
holds its Python transcription of `misc/negelcbo_vbmc.m` and
`misc/vpbndloss.m`, is kept as `scripts/wave7_O1_common.py`, with the
imports renamed to match. Their logs are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`, "Port correctness review"). The text below
the rule is the agent's final message, unedited; `wave7.md` holds the
consolidated ledger and the dispositions.

---

# Wave 7 verification, slice O1: the expected log joint, the ELCBO and their gradients

## 1. Revisions and interpreter

As printed by every script:
- `pyvbmc.__file__`: `C:\Users\luigi\Documents\GitHub\pyvbmc-wave7\pyvbmc\__init__.py`. The worktree HEAD moved during the session, from `0570d92a` to `df0f502b` ("docs(dev): wave 7, the report of the third reader O3"). `git diff a65b96f4 HEAD -- pyvbmc/` is empty, so the package code is still `a65b96f4`.
- `gpyreg.__file__`: `C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`, at `0186d89` (v1.3.0).
- MATLAB VBMC: `../vbmc` at `396d649`. No MATLAB was run. Statements about MATLAB are read from the source. For F1 they are also measured on my Python transcription of `misc/negelcbo_vbmc.m` plus `misc/vpbndloss.m` (`o1common.matlab_negelcbo`).
- Interpreter: `C:\Users\luigi\Documents\GitHub\pyvbmc\.venv\Scripts\python.exe`, with BLAS single-threaded.
- Scripts and logs are in `...\scratchpad\wave7\verify_O1\`:
  - F1: `O1_F1_frozen_sigma.py`
  - F2 and F3: `O1_F2_F3_defaults_docstrings.py`
  - first-question checks: `O1_Q1_fd_gradients.py`, `O1_Q1_flags.py`, `O1_Q1_variance.py`, `O1_Q1_variance_gh2d.py`, `O1_Q1_mc_path.py`, `O1_Q1_mc_path_paired.py`, `O1_Q1_mc_path_paired_1200.py`, `O1_Q1_mc_path_paired_1200b.py`
  - test claims: `O1_T_test_claims.py`
  - shared helpers: `o1common.py`

## 2. Headline

All three findings reproduce.

- **F1 is a confirmed port discrepancy, and it is worse than the report says once reached.** No VBMC run reaches it, at any option, on a noiseless or a noisy target. Once reached, the optimization is ruined, not just perturbed:
  - A single `optimize_vp` call (K = 1) grows the "frozen" σ from 0.78 to 34648.
  - Scipy BFGS, run the way `optimize_vp` runs it, drives σ from 0.6 to 8e4 in 18 calls.
  - The bound-loss gradient is wrong even at the starting point, where nl = 1.
  - A mirror sub-case exists: σ free and λ frozen at a non-unit RMS, on direct `_neg_elcbo` calls only.
- **F2's Python facts hold, but its MATLAB comparison is wrong.** The MATLAB call the report cites, `[F,dF,G,H,varF] = negelcbo_vbmc(theta,0,vp,gp)`, does not compute the variance. With MATLAB's default `compute_grad = nargout > 1` it stops in `gplogjoint.m:25-29`. I classify F2 as **documentation only**. The MATLAB default combination is itself a dormant MATLAB-side defect (§7).
- **F3 is documentation only, confirmed.**
- **The first-question answers hold in every configuration I checked:**
  - zero, constant and negative-quadratic means, D = 1 and 2, K = 1 to 3, Ns = 1 and 3, with a weight below the penalty threshold and the penalty active;
  - the variance at separated and wide components, including a heteroskedastic GP and a rank-one update;
  - the Monte Carlo (Adam) path;
  - the 12 flag combinations outside F1.
- **The production configuration the report describes is right.** `elcbo_beta` is fixed at 0, so the gradient is always available. Adam with the Monte Carlo entropy runs whenever K ≥ 2, warm-up included.
- **One test-adequacy note has the wrong numbers.** The report says the Monte Carlo FD test lets the η block be 10% wrong. At the test's own point the η block is ±190.75 and is pinned to about 1%. The real weakness is larger than the report says: the entropy's own gradient is almost unconstrained by that test.

## 3. Ledger

| id | report | statement | verdict (class) | fires at defaults? | how verified | dating and recorded reason | already recorded? | note for disposition |
|---|---|---|---|---|---|---|---|---|
| O1-1 | F1 | **What Python does.** With `optimize_sigma=False` and `optimize_lambd=True`, `_neg_elcbo` calls `vp.set_parameters(theta)` (`variational_optimization.py:1228`). That call sets λ = exp(θ_λ)/nl and multiplies the stored σ by nl, where nl = rms(exp θ_λ) (`variational_posterior.py:1237-1240`). The stored σ keeps the factor. `optimize_vp` reuses one `vp0`, so σ is multiplied by nl(θ) on every objective call, in `_eval_full_elcbo`, and again at `optimize_vp:362`. `_vp_bound_loss:615` then reads ln σ from the rescaled σ: its log scale is off by ln nl and its gradient omits d ln nl/dθ_λ. **What MATLAB does.** `misc/negelcbo_vbmc.m:33-48` assigns `exp(theta)` with no rescaling, on a `vp` passed by value, and `misc/vpbndloss.m:16-26` reads `log(vp.sigma)` of that unchanged struct. MATLAB's objective is a function of θ alone, and its gradient is exact (my transcription's analytic gradient matches FD to 1e-10). **Mirror sub-case.** σ free and λ frozen at RMS ≠ 1: the first direct call normalizes the frozen λ in place, so a second call at the same θ gives F = −0.8176 where the first gave −1.6291 (MATLAB transcription: −1.6291). `optimize_vp` is immune to this sub-case because `get_parameters` normalizes λ before θ0 is taken. | **confirmed port discrepancy** (MATLAB right) | **No**, at any option setting, noiseless or noisy. `VariationalPosterior.__init__:218-219` sets both flags True. The only production writers of `optimize_*` set `optimize_mu`/`optimize_weights` from `variable_means`/`variable_weights` (`vbmc.py:588-589`, `:1774-1776`, `:2740-2741`) and the warm-up weight switch (`optimize_vp:165`). No option names σ or λ. Nothing under `pyvbmc/svbmc/`, `calibration/`, the docs, the examples or the skill sets them. **Reached only by assigning the public attribute by hand:** on a posterior handed to `optimize_vp`/`_neg_elcbo`, or, read from the source and not run, on `vbmc.vp` before `optimize()`, since `_candidate_vp` and `deepcopy` carry the flags. MATLAB is equally unreachable: `setupvars_vbmc.m:87-88` sets both true, and `vpoptimizeweights_vbmc.m` freezes both together, its only call (`vbmc.m:717`) commented out. | `O1_F1_frozen_sigma.py`: **(a)** F(θ0) = −0.971197 on a fresh posterior, 3.300823 after five calls at θ_a; σ goes from [0.6, 0.4] to [4.659, 3.106], a ratio of 7.765 = nl(θ_a)^5; the MATLAB transcription gives −0.971197 whatever came before. **(b)** On a fresh copy with the log-scale bound active, −G−H agrees with the transcription to 1.8e-15, but the bound loss reads a log scale 1.889 = ln nl too high: F = 193.44 against 3.67. The ln λ gradient is analytic [203.6, 50.8] against FD [428.6, 67.5]. **(c)** BFGS as in `optimize_vp:247`: 18 calls, σ ends at 7.96e4; the returned scales σ_kλ_d are about 1.5e5, MATLAB's about 1. **(d)** `optimize_vp` itself (K = 1, deterministic branch), with `_neg_elcbo` wrapped: the frozen σ goes from 0.7804 to 20760 over 21 calls, and the returned σ is 34648. At the same final θ, MATLAB's `rescale_params` gives 1.30. `O1_Q1_flags.py`: with the scale bound active, the four F1 flag combinations have gradient errors of 16% to 44% (the value is right at nl = 1); the other 12 combinations agree to ≤ 2.4e-11. | Python matched MATLAB from the first port of `_negelcbo` (`9267816c`, 2021-08-25), which assigned `exp(theta)` directly. `abf5c3ec` (2021-10-31, PR #29) replaced that with `vp.set_parameters(theta)`; the squashed message lists "fix bug with 1D sigma", and the removed lines carried "TODO: wrong size" comments, so this was a shape fix. The nl rescale in `set_parameters` dates from `0089b35a` (2021-03-16), a transcription of `rescale_params.m`, which MATLAB calls only outside the objective. MATLAB's lines are unchanged since `b7ade85` (2019-11-29), before the port. Nothing records the frozen-σ consequence. | **Not recorded.** No row in `wave1_P6.md`. Wave 5 row P7-5 found the in-place rescale of `get_parameters` a no-op in production; that is true, but only with all flags on. The sheet entry "`_vb_init` type 3 with frozen sigma" (PI Q3, 2026-09-07) and `dev/plans/latent-bug-fixes.md:671`, `:963` ("frozen-sigma coverage") cover initialization only. Their one test, `test_vb_init_type3_preserves_fixed_sigma_without_extra_draws`, freezes σ, λ and w together. | First decide whether freezing σ alone is a supported configuration; Q3 already made its initialization work. Options: **(a)** restore MATLAB's assignment inside `_neg_elcbo` (no gauge rescale in the objective, 2-D shapes kept), which also makes the objective independent of the reused `vp0`; **(b)** rescale in `set_parameters` only when both σ and λ are optimized, the only case with a gauge freedom; **(c)** treat the flags as internal: refuse σ frozen with λ free, and record that in the sheet. Nothing in production moves under any option, since every flag is on there. Tests for any fix: an FD check with σ frozen and λ free and a scale bound active, and a check that repeated calls at one θ give the same F. |
| O1-2 | F2 | **What Python does.** `_neg_elcbo(compute_var=None)` computes the variance only for β ≠ 0 (`:1211-1212`) and always returns `varF`, as 0.0 when the variance is skipped. The docstring says only "determined automatically". **What MATLAB does.** The default at `negelcbo_vbmc.m:16` is `beta ~= 0 \|\| nargout > 4`, but it combines with `compute_grad = nargout > 1` (`:10`). A call that takes `varF` therefore also takes `dF`. For β = 0 the guard at `:21` lets it through, and `gplogjoint.m:25-29` raises `gplogjoint:FullVarianceGradient`. MATLAB's default computes the variance only when the caller also passes `compute_grad = 0`. | **documentation only**: a MATLAB default keyed to `nargout`, which Python cannot see; no caller on either side relies on it | **No.** Every caller passes `compute_var`. Python: `optimize_vp:233`, `:266`; `_sieve:855`; `_eval_full_elcbo:530`; `active_sample.py:940`. MATLAB: `vpoptimize_vbmc.m:71`, `:78`, `:289`; `vpsieve_vbmc.m:76`; `activesample_vbmc.m:526`; `vpsample_vbmc.m:58`, `:79`; `vpoptimizeweights_vbmc.m:47`, `:53`, `:173`. | `O1_F2_F3_defaults_docstrings.py`, four calls: (i) all defaults: `varF` = 0.0, returned with `dF`; (ii) `compute_grad=False`: 0.0; (iii) `compute_grad=False, compute_var=True`: 0.1805; (iv) `compute_var=True` with the default gradient: `NotImplementedError`, the Python counterpart of MATLAB's error. MATLAB read from source. | Python line since `9267816c` (2021-08-25), never matched. MATLAB `:10` and `:16` since `662ee27` (2019-04-24). No recorded reason. | Not recorded. | Options: **(a)** docstring only: "computed if and only if `beta` is nonzero"; **(b)** default `compute_var = beta != 0 or not compute_grad`, the nearest analogue of MATLAB's for a caller that reads `varF`, at the cost of the variance solves on any no-gradient call that omits the argument (no such call exists in the package). |
| O1-3 | F3 | Wrong docstrings. `_initialize_full_elcbo` (`:452-457`) calls `D` "The dimension"; it is the length of θ (`optimize_vp:184-186`). It calls `Ns` "Number of samples for entropy approximation"; it is the number of GP hyperparameter samples (`np.size(gp.posteriors)`). `_neg_elcbo`'s `Ns` (`:1162-1163`) is a count per component (`entmc_vbmc`'s docstring: "per component"). `entropy_alpha` is documented at `:1171` but the parameter is `_entropy_alpha` (`:1145`). A further slip in the same docstring: `compute_var : bool`, although the code takes 0, 1 and 2. | **documentation only** | n/a | Same script: prints the signature, the docstring entries and the call site `_initialize_full_elcbo(slow_opts_N * 2, theta_N, K, Ns)`. | All present since `9267816c` (2021-08-25). `9c33d09d` (2022-10-31, docs) touched `:1160-1172` without fixing these. MATLAB passes `Ntheta` and `numel(gp.post)` (`vpoptimize_vbmc.m:32-34`). | Not recorded. | Docstring edits only. |

## 4. The check of the first-question answers

**Production configuration: confirmed.**
- `_sieve` hard-codes `elcbo_beta = 0` and `compute_var = False` (`:821-822`), so `gradient_available` is always true (`optimize_vp:193`).
- `entropy_switch = False` (`advanced_vbmc_options.ini:209`). It is only ever turned from True to False (`vbmc.py:1167`, `:1387`, `:2327`).
- `ns_ent = 100*K^(2/3)` gives 80 samples per component at K = 2, so every K ≥ 2, warm-up included (`k_warmup = 2`), takes the Adam path with the Monte Carlo entropy. K = 1 takes BFGS.
- The no-gradient branch (BFGS with SciPy's finite differences, or a `ValueError` on the stochastic path) and MATLAB's switch to CMA-ES (`vpoptimize_vbmc.m:42`) are as the report says, and unreachable.

**Gradients by FD, in the configurations the tests leave out** (`O1_Q1_fd_gradients.py`). The grid is 36 configurations: zero, constant and negative-quadratic means × D ∈ {1, 2} × K ∈ {1, 2, 3} × Ns ∈ {1, 3}.
- For every K ≥ 2, one weight is placed below the penalty threshold (smallest w 0.015 to 0.029, threshold 0.125 or 0.083), so the penalty is active. The mean and log-scale soft bounds are active as well.
- Worst relative error of `_gp_log_joint`'s dG: 2.5e-10. Worst for `_neg_elcbo`'s dF: 1.3e-11.
- The penalty-only gradient of the η block is analytic [0.0028453, −0.0028453] and the same by FD.
- `I_sk` for the zero and constant means (never value-checked before) agrees with an independent quadrature of gpyreg's posterior mean (`predict`, per sample) to ≤ 4.8e-15. The quadrature is a 20001-point grid for D = 1 and 80×80 Gauss–Hermite for D = 2.
- Flags (`O1_Q1_flags.py`): all 12 combinations outside F1 hold to ≤ 2.4e-11 with the scale and mean bounds active.

**Variance branch** (`O1_Q1_variance.py`, `O1_Q1_variance_gh2d.py`), at K = 3 with separated and wide components, all three means, against quadrature of `predict_full` (latent covariance):
- `J_sjk`: ≤ 1.3e-13 at D = 1 (grid); ≤ 5.5e-15 at D = 2 (Gauss–Hermite, n = 24; the 1e-4 of a 45×45 uniform grid was that grid's own error).
- Heteroskedastic user noise: 8.1e-14.
- After a rank-one update with a point quieter than every training point: 7.9e-14. Here 1/sW[0]² equals the stored `sl` (0.03197) while min(sn2)·mult is 0.0026, and `_gp_log_joint` correctly uses the stored scale.
- `G`, `varG` and `var_ss` are rebuilt exactly (difference 0.0) from the per-sample outputs.

**Adam path** (`O1_Q1_mc_path*.py`), D = 2, K = 3, with bounds and the penalty active.
- dF_mc − dF_lb = −(dH_mc − dH_lb) to 5.6e-15.
- Paired test (analytic gradient minus the pathwise derivative of the same seed's estimate), over 1200 fresh seeds: |z| ≤ 1.31 on every μ, σ and λ entry.
- An earlier 1200-seed run showed z = −3.28 on one entry; the fresh seeds, at z = −0.15 for that entry, show it was a fluctuation.
- The η block equals the pathwise derivative exactly (up to FD round-off, about 1e-11).
- The report's answer holds.

**`separate_K`** raises when gradients are requested (`:1261-1266`), as stated.

## 5. Errors in the report

1. **§1, MATLAB history.** `2044530` did more than delete the integrated-mean-function code. In these files it also removed, from `negelcbo_vbmc.m`, the multi-GP branch, the `altent_flag`/`entmcub_vbmc` path and the `entropy_alpha` lower/upper interpolation. From `vpoptimize_vbmc.m` it removed the `GPStochasticStepsize` block and commented-out code, and it added a header comment to `fminadam.m`. None of this code is ported, so the conclusion that the kept formulas are unchanged stands.
2. **F2.** "MATLAB also computes it whenever the caller asks for varF" and "the MATLAB call `[F,dF,G,H,varF] = negelcbo_vbmc(...)` computes it" are wrong under MATLAB's default `compute_grad`: that call raises `gplogjoint:FullVarianceGradient` (row O1-2).
3. **F2, test adequacy.** `test_neg_elcbo_without_variance_returns_float_placeholders` passes `compute_var=False` explicitly (`test_variational_optimization_single_sample.py:83-94`), so it does not exercise the default. No test asserts `varF` under `compute_var=None`; `test_vbmc_finalboost.py:387` uses the default but reads only F and dF.
4. **§4, Monte Carlo FD test.** "Gradients dominated by a bound term of about 200. The η block (~0.08) could be wrong by more than 10% and still pass" quotes the reviewer's own `check_mc_path.py` state (η ±0.0799, bound −233), not the test's (`O1_T_test_claims.py`).
   - At the test's point, dF = [−1.33e4, −6.19, −22.2, −13.9, 1006.6, 77.9, 1005.3, 79.1, **190.75, −190.75**].
   - `np.allclose` with rtol = atol = 1e-2 allows about 1.0% to 1.2% per entry, so scaling the η block by 1.10 fails the check.
   - What passes is an error in the entropy's own gradient. dH_η = ±0.469 sits at 24% of that entry's tolerance, and scaling the entropy's η gradient by 5 still passes. The μ entries of dH are 1.7e-4 to 0.14 of their tolerances.
5. **Minor.**
   - The table's "softmax Jacobian, line 1619": `:1619` is the comment; the Jacobian is built at `:1621-1627`.
   - "`test_vp_bound_loss_weight_flags` freezes σ only together with λ, where nl = 1 and nothing drifts": that test calls `_vp_bound_loss` directly and never reaches `set_parameters`. The conclusion (no test covers σ frozen with λ free) holds.
   - F1's "low consequence" is right about reach but understates the effect once reached (§2).

## 6. Test notes worth acting on

- **Zero and constant means are user options** (`gp_mean_fun = "zero"`/`"const"`, validated at `vbmc.py:3899-3905`), yet `_gp_log_joint` is neither FD- nor value-checked with them. Add both checks; they hold (§4).
- **Add FD checks at K = 1, at D = 1, and with the weight penalty active** (a weight below 1/(4K)). The FD tests of `_neg_elcbo` start at w = 0.509/0.491 (seed 1) and 0.603/0.397 (seed 2), above the 0.125 threshold, as the report says.
- **In `test_neg_elcbo_retains_capped_small_weight_penalty`,** replace the `not np.allclose` assertion on the gradient with the exact J_w·(p·1[w < thr]).
- **`test_neg_elcbo_grad_fd_mc_entropy` barely constrains the entropy gradient** it exists to check (item 4 of §5). Check it where the G and bound terms do not swamp it: no bound violation and moderate widths, or dH_mc averaged over seeds against FD of the mean, with the entropy slice owning the test.
- **The variance value has no independent pin at separated means.** The only value pin is the MATLAB fixture, whose two components sit at the origin (|μ| ≤ 3e-6) with equal σ = 1e-3. The off-diagonal pair integrals J_jk are therefore never checked independently at separated means. Add a quadrature test at K = 3 with separated components; a 1-D grid suffices (1e-13 in §4).
- **If F1 is fixed:** FD with σ frozen and λ free and a scale bound active, plus repeated-call invariance.
- **Not worth acting on:** the two "implementation mirrors" the report names. The non-Cholesky variance test is a consistency check of a branch no run enters.

## 7. Defects on the MATLAB side

One candidate for `matlab_side_defects.md` (read from source, not run):

| # | Location | What the code does | Consequence | PyVBMC | Record |
|---|---|---|---|---|---|
| 56 | `misc/negelcbo_vbmc.m:10`, `:16`, `:21`, with `misc/gplogjoint.m:25-29` | The defaults `compute_grad = nargout > 1` and `compute_var = beta ~= 0 \|\| nargout > 4` combine so that a call that takes `varF` also requests the gradient of the full variance, which `gplogjoint` refuses (`gplogjoint:FullVarianceGradient`); the guard at `:21` catches the combination only for `beta ~= 0` | `[F,dF,G,H,varF] = negelcbo_vbmc(theta,0,vp,gp)` raises instead of returning the variance; the automatic variance default works only with an explicit `compute_grad = 0`. Dormant: every MATLAB caller passes both arguments | Not shared in that form: PyVBMC's default ignores the requested outputs and returns `varF = 0.0` (`verification`, wave 7, row O1-2) | wave 7, row O1-2 |

## 8. Sheet entries

- **Nothing to withdraw or correct.** The P6 entry "The eta soft bound was removed, and `_neg_elcbo` no longer mutates `theta`" holds as written.
- **If F2 is ruled documentation only, or deliberate,** add to slice P6:

  > ### The variance of `_neg_elcbo` is computed by default only for a nonzero `beta`
  > - Python: `pyvbmc/vbmc/variational_optimization.py: _neg_elcbo` sets `compute_var = beta != 0` when the argument is `None`, and always returns `varF`, 0.0 when the variance is not computed.
  > - MATLAB: `misc/negelcbo_vbmc.m:16`, `compute_var = beta ~= 0 || nargout > 4`, with `compute_grad = nargout > 1` at `:10`; a call that takes `varF` with the default `compute_grad` stops in `misc/gplogjoint.m:25-29`, so MATLAB's default computes the variance only when the caller also passes `compute_grad = 0`.
  > - What differs: a caller that omits `compute_var` and passes `compute_grad=False` receives `varF = 0.0` where MATLAB computes the variance. Every caller on both sides passes `compute_var`.
  > - Why: Python has no counterpart of `nargout`; `verification/` wave 7, row O1-2.
  > - Kind: deliberate change (interface).
- **If F1 is ruled option (c),** add to slice P6:

  > ### A frozen `sigma` requires a frozen `lambd`
  > - Python: `_neg_elcbo` assigns θ through `VariationalPosterior.set_parameters`, which normalizes `lambd` to unit root mean square and multiplies `sigma` by the same factor; with `optimize_sigma=False` and `optimize_lambd=True` the stored `sigma` would change on every evaluation, so that combination is refused.
  > - MATLAB: `misc/negelcbo_vbmc.m:33-48` assigns θ to a by-value copy without rescaling and supports the combination, which no MATLAB path sets.
  > - Why: `verification/` wave 7, row O1-1.
  > - Kind: deliberate change (a dormant configuration).

  Also add a sentence to the existing entry "`_vb_init` type 3 with frozen sigma copies existing widths": "The objective supports a frozen sigma only together with a frozen lambd (wave 7, row O1-1)." Under options (a) or (b), no sheet entry is needed.

## 9. Beyond the report

- **F1's severity and mirror sub-case** (row O1-1). The growth is geometric: nl(θ) ≈ 1.5 to 1.8 per call at ordinary λ.
- **`_neg_elcbo` mutates the caller's posterior,** where MATLAB passes it by value. With all flags on this is harmless: every call re-sets every parameter, and P7-5's "bitwise no-op" holds. F1 is its only consequence I found.
- **Rank-one updates:** `_gp_log_joint`'s variance reads the factor's scale from `sW[0]`, which the update keeps. It stays right after a noise-lowering rank-one update (§4), unlike the pre-fix `GP.quad` of W6-27.
- **Nothing else in the slice** looked like a discrepancy beyond the sheet entries the report lists.

## 10. `git status --porcelain` at the end

- worktree `pyvbmc-wave7`: (empty); HEAD `df0f502b`, package code identical to `a65b96f4`
- gpyreg: (empty); `0186d89`
- `../vbmc`: (empty); `396d649`

My scripts and logs are only in `...\scratchpad\wave7\verify_O1\`.
