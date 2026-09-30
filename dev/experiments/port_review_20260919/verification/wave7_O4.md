# Wave 7 verification, slice O4: the GP's marginal likelihood and its gradient

Raw report of a verifier, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent,
read-only on the repositories, verified the findings of the third reader of
slice O4 (`../reviews/O4_third_reader.md`) and its answers to the first
question of its brief, reading gpyreg at `v1.3.0` (`0186d89`), PyVBMC's
package code at `a65b96f4` and `gplite/` of MATLAB VBMC at `396d649`, with
no `optimize()` run. Its scripts are kept as `scripts/wave7_O4_*.py`, under
the names the report gives them with `wave7_` in front; its helper
`o4v_common.py` is kept as `scripts/wave7_O4_common.py`, with the imports
renamed to match. Their logs are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`, "Port correctness review"). The text below
the rule is the agent's final message, unedited; `wave7.md` holds the
consolidated ledger and the dispositions.

---

# Wave 7 verification, slice O4: the GP's marginal likelihood and its gradient

## 1. Revisions and interpreter

Every script printed the following:
- pyvbmc: `C:\Users\luigi\Documents\GitHub\pyvbmc-wave7\pyvbmc\__init__.py`
- gpyreg: `C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`
- numpy 2.5.2, scipy 1.18.1, Python 3.12.6

Revisions:
- **gpyreg:** `git describe --tags` gives `v1.3.0` (`0186d89`).
- **MATLAB VBMC:** `396d649`.
- **Worktree:** HEAD moved from `0570d92a` to `df0f502b` during the session, through docs-only commits. `git diff a65b96f4 HEAD -- pyvbmc pyproject.toml` is empty, so the package code is still that of `a65b96f4`.

Scripts and logs are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\verify_O4`:
- `o4v_common.py`, a helper that includes an 80-digit `Decimal` nlZ with a hand-written Cholesky.
- `O4_F1_fixed_bound_gradient.py`, `O4_F1_fixed_with_smoothbox.py`, `O4_F1_pyvbmc_reach.py`, `O4_F1_tiny_range_gpyreg.py`.
- `O4_F2_prior_mass_upper_tail.py`, `O4_F3_nonfinite_prior_location.py`.
- `O4_Q1_noise_configs_fd.py`, `O4_Q1_rectified_fd.py`, `O4_Q1_lownoise_precision.py`.

Each `.py` has a `.log` beside it.

## 2. Headline

All three findings reproduce, and all three are **confirmed Python-only defects**: code in gpyreg with no counterpart in `gplite`. None of them fires in PyVBMC, on a noiseless target or on a noisy one.

- **F1 (O4-1) is wider than reported, in two ways.**
  - The NaN also survives when the fixed coordinate has a smooth-box prior (either family) whose box contains the fixed value. The smooth-box branches write the gradient only for coordinates below or above the box. The report says every family overwrites the NaN.
  - `log_posterior(compute_grad=True)` returns the NaN even on a GP with no prior at all. `fit` is affected only when some coordinate has a prior, because it adds the prior only when `no_prior` is False. The report does not state that condition.
  - F1 also contradicts the sheet's statement that `f_idx` "changes only what a caller who evaluates off a fixed value is told". The sheet's commit citations are half right: `0ca35b3` is where `f_idx` began to read the GP's bounds.
- **F3 (O4-3) is the root of the `dev/TODO.md` item on a NaN location, and it is wider.** It adds infinite locations. With the bounds that `fit` fills, an infinite `mu` gives NaN, not the `-inf` the report states. It is also the unhandled remainder of W6-17/G1-12, whose fix refused only `sigma`.
- **The report's answers to its first question hold.**
  - Gradients agree by finite differences in every configuration PyVBMC builds, including the level-1 multiplier.
  - In the low-noise representation, the analytic gradient matches an 80-digit reference where double-precision differences fail.

## 3. Ledger

| id | report | statement | verdict (class) | fires at defaults? | how verified | dating and recorded reason | already recorded? | note for disposition |
|---|---|---|---|---|---|---|---|---|
| O4-1 | F1 | `__compute_log_priors` writes `dlp[f_idx] = np.nan` for every coordinate with `lb == ub` (`gpyreg/gaussian_process.py:1716-1720`). Only the Gaussian and Student's t branches, and the smooth-box branches for a value outside the box, overwrite it. The NaN therefore survives for a fixed coordinate that has **no prior**, or that has a **smooth-box prior containing its value**. It reaches every `log_posterior(compute_grad=True)` (even with no priors set), and `fit`'s L-BFGS-B (`:1503`) whenever some coordinate has a prior, since `__gp_obj_fun` adds the prior only then. L-BFGS-B then stops within 0 or 1 iterations. `gplite_hypprior.m:25` starts `dlp` at zeros and has no branch for fixed coordinates, so a coordinate without a prior contributes 0. | confirmed Python-only defect. It also falsifies one sentence of a sheet entry (section 8). | No. **PyVBMC:** the only pair that comes out equal on a usable training set is `noise_log_scale`, for target range < `tol_gp_noise` ≈ 3.2e-3 (18 of 18 configurations collapse, levels 0, 1 and 2). That coordinate always carries its Student's t prior, so there is no NaN. The coordinates without a prior (the negquad location and log scale) collapse only for a zero-width input column. There the length-scale pair is `(-inf, -inf)` and `fit` raises L-BFGS-B's `KeyError` first. **gpyreg alone:** it fires when a caller fixes a coordinate without a prior while another has one (the configuration of `test_fitting_with_fixed_bounds`), or for targets with range < 1e-6 (the collapse of W6-38) with a prior elsewhere. | **`O4_F1_fixed_bound_gradient.py`** (D=2, negquad, noise fixed at log 1e-2, Gaussian prior on the length scales):<br>• At the fixed value, `log_posterior` gradient is NaN at index 3; `log_likelihood` gradient there is 18.69; the MATLAB transcription gives finite values.<br>• `fit(n_samples=0)`: nit=1, log posterior −43.25, largest free-gradient entry 359.<br>• Same fit with the pair widened by 1e-9: nit=93, log posterior 50.78. Hand L-BFGS-B on the free coordinates from the stalled point: 50.78.<br>• No priors at all: the fit is normal (nit=73), but `log_posterior` still returns the NaN.<br>• Toy L-BFGS-B with a NaN on a fixed variable: nit=0, "ABNORMAL".<br>• Test configuration with n_samples=0: nit=1, −16.029, against the widened pair's nit=6, −16.005.<br>**`O4_F1_fixed_with_smoothbox.py`:** smooth box with the value inside gives a NaN entry and a fit with nit=1, log lik −65.66. Gaussian or Student's t: nit=73, log lik 54.70.<br>**`O4_F1_tiny_range_gpyreg.py`:** range 5e-7, prior on the length scale: noise pair equal (−13.8), NaN at the noise entry.<br>**`O4_F1_pyvbmc_reach.py`:** 90 PyVBMC-built configurations over 5 training sets; NaN only for a shared coordinate, where `fit` raises `KeyError: (-inf, -inf)`. | gpyreg `6754f01` (2021-06-22) added the NaN with a "fixed" prior *type* that set `LB = UB` in `hyper_priors`. `0ca35b3` (2021-06-28) removed the type and made `f_idx` read the GP's bounds. `a2f8ddc` (2026-09-05) moved the masks into a cache, with no change of behavior. MATLAB `gplite_hypprior.m` has been unchanged since `918809e` (2019-07-15). The two never agreed. No recorded reason for the NaN. | No. The `-inf` off a fixed value is on the sheet (W6-24, G1-33); the NaN gradient is recorded nowhere. W6-38's collapsed noise pair is F1's second gpyreg trigger. W6-4 removed the equal-target trigger: equal targets now make no equal pair (0 of 18). | Options:<br>(a) write 0, as MATLAB effectively does, for fixed coordinates, or only for those the family branches leave unset;<br>(b) take fixed coordinates out of the L-BFGS-B problem;<br>(c) keep and document.<br>Either way the sheet entry needs correcting (section 8), and `test_fitting_with_fixed_bounds` needs an assertion that can see it. |
| O4-2 | F2 | `__recompute_normalization_constants` stores `cdf(ub) − cdf(lb)` (`:1592-1607`; the upper branches of the smooth-box CDFs are at `f_min_fill.py:300,337`). With both bounds far in the upper tail, both CDFs round to 1 and the mass is exactly 0. Then `log_norm = −inf` (`:1670`), `log_posterior = +inf` everywhere (`:1851`), and the fit's objective is `−inf` everywhere. The lower tail is exact. | confirmed Python-only defect, in a listed Python-only addition whose sheet claim ("neither the optimizer nor the slice sampler sees it") fails in this regime. | No. PyVBMC's priors (Student's t, ν=3) have masses 0.44–0.94 inside the filled bounds (levels 0/1/2, D=1/3). | **`O4_F2_prior_mass_upper_tail.py`:**<br>• N(0,1) on `mean_const`, bounds [9,10]: stored 0, true 1.1285e-19, `log_posterior` = inf. Mirror [−10,−9]: stored 1.1285e-19, exact.<br>• Threshold between 8.2σ (stored 1.11e-16, true 1.20e-16) and 8.3σ (stored 0, true 5.19e-17).<br>• Student's t ν=3, [1e6, 2e6]: 0 against 9.6e-19. Both smooth-box upper branches: 0.<br>• The fit then returns a point whose finite objective (log lik + log prior) is −85.21, against an optimum of 25.89 in the same box. | `64dc49d` (2021-06-29); smooth-box CDFs `b0c5cde` (2021-06-14). MATLAB never had a renormalization. No recorded reason. | No. W6-24 and the sheet call the constant invisible to the optimizer; TODO notes only that the renormalization is untested against the truncated densities. | Options: survival functions (or `logsf`/`logcdf` with log-diff-exp) for bounds above the centre; refuse bounds whose mass underflows; or leave it (the prior is grossly inconsistent with the bounds) and qualify the sheet sentence. |
| O4-3 | F3 | `set_priors` validates `sigma` only (the block ending at `:684`); no location is checked. Accepted inputs:<br>• Gaussian or Student's t with `mu = ±inf` or NaN;<br>• a smooth box (either family) with an infinite or NaN end.<br>`("smoothbox", (0, inf, 1))` fails `sb_idx`, falls into `g_idx` with `mu = NaN` and gives NaN. `("gaussian", (inf, 1))` gives `−inf` without finite bounds, but **NaN with the bounds `fit` fills**: the mass is 0, and `−inf − (−inf)`. `gplite_hypprior.m:35` reads a non-finite `mu` as no prior. | confirmed Python-only defect, in input validation. | No. Every prior of `_gp_hyp` has finite `mu` and `sigma`. | **`O4_F3_nonfinite_prior_location.py`:**<br>• All eight such priors are accepted; with filled bounds every one gives NaN; `gaussian (±inf, 1)` gives −inf with unset bounds.<br>• The control, an infinite `sigma`, is refused.<br>• A transcription of `gplite_hypprior` gives lp 0 (uniform).<br>• A fit with `("smoothbox", (0, inf, 1))`: nit=0, "ABNORMAL", fun NaN, log lik −492.2, against 72.5 without priors. | Location never validated (since `6754f01`/`b0c5cde`). The `sigma` refusal came in `836af5d` and was reshaped in `c4a441e` (both 2026-09-22, W6-17). MATLAB's `|` dates from `9dfca5a` (2019-06-17). The two never agreed. | Partly. W6-17/G1-12 named "non-finite `mu` without `a`/`b`" as the unhandled remainder, and its fix refused only `sigma`. TODO ("What the independent check of the wave-6 pass left for later") lists a NaN location. **Same defect, wider:** infinite locations and infinite smooth-box ends. | One ruling covers the TODO item too. Options: refuse a non-finite location as `sigma` is refused; read MATLAB's non-finite `mu` as no prior; or support half-open boxes. The sheet entry on `None` needs correcting (section 8). |

## 4. The check of the first-question answers

The report's answer (the gradient holds in every configuration PyVBMC builds) is confirmed.

- **PyVBMC-built GPs** (`_gp_hyp` on a live `VBMC` object; levels 0/1/2 give noise [1,0,0], [1,2,0], [1,1,0]; each mean function; D = 1 and 3; three random points each; `O4_Q1_noise_configs_fd.py`):
  - Per-component FD error is ≤ 1.3e-8 for the fit objective with priors, and ≤ 3.2e-8 for the log likelihood.
  - The level-1 multiplier (`scale_user_provided`, s2 = 1/n as `FunctionLogger` pools repeats) matches my dense `0.5·Σ e^{h} s2_i Q_ii` to ≤ 1.6e-15 relative.
  - Exact duplicate inputs with [1,2,0], [1,1,0] and [1,0,0]: ≤ 7.4e-10.
- **Rectified noise**, [1,0,1] and [1,2,1] with points on both sides of the threshold (`O4_Q1_rectified_fd.py`): ≤ 3e-8. PyVBMC never builds this configuration.
- **Low-noise representation** (`O4_Q1_lownoise_precision.py`). The claim that the analytic gradient is right and double-precision FD is what fails is **confirmed** by both routes:
  - *Just below the threshold, well conditioned.* At sn2 = 0.9e-6 and 0.99e-6 (`L_chol=False`), the analytic gradient agrees with FD to 3–4e-12 relative to the largest component, and nlZ equals a dense `slogdet` evaluation. Just above, at 1.01e-6 and 1.1e-6 (`L_chol=True`), both are equally close and continuous across the switch.
  - *80-digit reference.* cond(C) = 2.4e6: analytic error 1.4–1.6e-11, against 3e-7 for double FD. cond 4.2e9 with a repeated input: 1.6e-8, against 1.5e-4. cond 9.4e12: 9.2e-6, against 0.41 for double FD.
  - One nuance: the analytic gradient's own error grows with conditioning, up to 5e-5 per component at cond 1e13, and the reviewer's own log shows 1.4e-4 on small components. The report states its errors "of the max component", which is correct.
- "Unreachable at PyVBMC's defaults" holds: the noise floor `log(sqrt(1e-5))` gives min sn2 ≥ 1e-5 at every level.

## 5. Errors in the report

1. **F1, the families that overwrite the NaN.** "Where the fixed coordinate also has a Gaussian, Student-t or smooth-box prior, those branches overwrite the NaN" is false for a smooth box (either family) containing the fixed value (`O4_F1_fixed_with_smoothbox.py`).
2. **F1, the scope.** `fit` is hit only when `no_prior` is False. `log_posterior(compute_grad=True)` returns the NaN even on a GP without any prior. The report states neither.
3. **F1, "L-BFGS-B stops at its starting point".** It stops within 0 or 1 iterations; the reviewer's own second reproduction has nit=1.
4. **F1, the reach.** "The one bound pair `_gp_hyp` can collapse is `noise_log_scale`" is inexact. A zero-width input column also collapses the length-scale pair to `(-inf, -inf)` (it has a prior), and the report does not say that `fit` then fails with `KeyError` before any gradient is used.
5. **F1, the history.** `0ca35b3` is not unrelated: it is where `f_idx` began reading the GP's bounds, having read a "fixed" prior type's `LB`/`UB` in `6754f01`. The sheet's real omission is `64dc49d` (the renormalization) and `6754f01`.
6. **F3.** "`("gaussian", (inf, 1))` gives −inf everywhere" holds only without finite bounds. With the bounds `fit` fills, it is NaN.
7. **Test notes.** They miss `pyvbmc/testing/vbmc/test_gpyreg_derivatives_fd.py`, which checks by FD the mean-function gradients and the noise gradients (per-point, scaled, rectified) that the report says go unchecked. What is untested is their assembly in `__core_computation` (the `dsn2[:, i]·diag(Q)` branch) and the prior terms.

The line numbers and MATLAB citations I checked are right: `:1503`, `:1592-1607`, `:1670`, `:1716-1720`, `:1851`, `:1933`, `f_min_fill.py:300,337`, `gplite_core.m:79,93,96,102,193,226,229-238,244,246-247,259,279-284`, `gplite_hypprior.m:25,35,44-49,52-57`, `covfun`/`meanfun`/`noisefun`/`nlZ`/`train` spot checks. So are the claims that no MATLAB slice file changed after 2021-01-19 (last: `gplite_meanfun.m`, 2020-05-08) and that `test_gp_gradient_computations` is unseeded and never leaves the Cholesky representation.

## 6. Test notes worth acting on

- **`test_fitting_with_fixed_bounds`** asserts only that the fixed value is kept. With `n_samples=0` it could assert a small free-coordinate gradient at the result, and it could add a smooth-box case with the value inside. Act if O4-1 is fixed.
- **Likelihood gradient with per-point noise.** A gpyreg test of the assembled gradient with per-point noise (s2 with `scale_user_provided`, rectified), D = 1 and repeated inputs is cheap. The component gradients are already covered on PyVBMC's side.
- **Low-noise representation.** A test of nlZ and its gradient just below sn2 = 1e-6 on a well-conditioned set, where FD is reliable (case A above).
- **Seeding.** Seed `test_gp_gradient_computations`.
- **Tails and locations.** If O4-2 or O4-3 is fixed: an upper-tail renormalization test (joins the TODO note on truncated densities), and a `set_priors` refusal test for non-finite locations.

## 7. Defects on the MATLAB side

None new. I agree with the report: only the two already settled.

## 8. Sheet entries

**Correct:** "The hyperprior is renormalized to the bounds, and a fixed hyperparameter gets a prior". Proposed text:

> ### The hyperprior is renormalized to the bounds, and a hyperparameter with equal bounds is held at its value
> - Python: `gpyreg/gaussian_process.py: __recompute_normalization_constants` stores, for each coordinate with a prior and a finite bound, the probability the prior puts inside the bounds, computed as `cdf(ub) - cdf(lb)`, and `__compute_log_priors` subtracts the sum of their logs (`lp -= masks["log_norm"]`). Separately, the mask `f_idx` of the coordinates whose two bounds are equal sets the log prior to `-inf` where such a coordinate is not exactly at its bound, and writes NaN into that coordinate's entry of the prior's gradient; the branch of the coordinate's own prior replaces the NaN for a Gaussian or Student's t prior, and for a smooth box only where the value lies outside the box.
> - MATLAB: `gplite/gplite_hypprior.m` has neither: its priors are unnormalized over the bounds, and every coordinate contributes what its prior gives (zero value and zero gradient where it has none), whatever its bounds.
> - What differs: gpyreg's `log_posterior` differs from `gplite_nlZ`'s by the sum of the log truncation constants (0.747 in the case measured). The term is constant in `hyp`, so the optimizer and the slice sampler do not see it while each mass is representable; where both bounds lie far in the upper tail of a prior (beyond about 8.3 scales for a Gaussian), the mass rounds to 0 and `log_posterior` is `+inf` at every point (wave 7, row O4-2). The NaN of `f_idx` reaches every `log_posterior(compute_grad=True)` on a GP with a fixed coordinate that has no prior, or a smooth-box prior around its value, and the fit's objective whenever any coordinate has a prior, where L-BFGS-B stops within an iteration (wave 7, row O4-1). In PyVBMC the only pair that comes out equal on a usable training set is the noise's, for targets whose range is below `tol_gp_noise`, and the noise always carries its Student's t prior; a training set with a zero-width input column collapses the length-scale pair onto `(-inf, -inf)`, and the fit ends with L-BFGS-B's `KeyError` before any gradient is taken.
> - Why: the fixed prior came in `6754f01` (2021-06-22) as a prior type that set equal bounds; `0ca35b3` (2021-06-28) moved the bounds out of the priors, and `f_idx` has read the GP's bounds since. The renormalization came in `64dc49d` (2021-06-29), after the smooth-box families of `27f8d66` (2021-06-11). No commit records a reason; PyVBMC never compares `log_posterior` values across the two implementations (`verification/wave6.md`, W6-24).
> - Kind: Python-only addition.

**Correct:** "'No prior' is expressed by `None`, not by an infinite scale". Replace the sentence "A prior written MATLAB's way is refused (gpyreg 1.2.1 took it …)" with:

> A flat prior written MATLAB's documented way, `sigma = Inf`, is refused (gpyreg 1.2.1 took it for a Student's t, with a log posterior of `-inf` or NaN and a design column of NaN). The other half of MATLAB's test, a non-finite `mu` beside a finite `sigma`, is not refused: `set_priors` checks no location, and such a prior gives a log posterior of `-inf` without finite bounds and NaN with them (wave 7, row O4-3).

**Flag for the record:** row W6-24 of `verification/wave6.md` says "without effect on the optimizer or the sampler". That holds except in O4-2's far-tail regime.

## 9. Anything found beyond the report

- **The two widenings of F1** (the smooth-box-inside case, and `log_posterior` without any prior) are in row O4-1.
- **A zero-width input column in a PyVBMC-built GP makes `fit` raise `KeyError: (-inf, -inf)`**, measured in `O4_F1_pyvbmc_reach.py`.
  - The length-scale lower bound comes from the HPD widths, and the upper is gpyreg's `log(10·w)` because `upper_gp_length_factor` defaults to 0.
  - PyVBMC reaches it when a user passes at least `fun_eval_start` starting points that share one coordinate: the initial design is then `x0[:sample_count]` (`active_sample.py:127-185`). This path is read from the code, not run.
  - It is the mechanism of the TODO item on a single training point (entry 49 of `matlab_side_defects.md`), reached by a multi-point set.
  - The formula is shared with `gplite_covfun.m:105,121-124` (read from source; what `fmincon` does with it was not looked up).
  - G2-20's reach argument ("≥ `fun_eval_start` distinct points") does not cover distinct points that share a coordinate. A candidate for triage.
- **O4-2 has a measured consequence** beyond the value: the fit returns a poor point (−85.2 against 25.9).

## 10. `git status --porcelain`

- `pyvbmc-wave7`: empty.
- `gpyreg`: empty (`v1.3.0`).
- `../vbmc`: empty (`396d649`).

The `__pycache__` directories under the worktree predate this session (10:17) and are gitignored.
