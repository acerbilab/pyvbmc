# Wave 7 verification

The findings of wave 7 (the third readers O1 to O4, each combining the
internal and the comparison track on one critical numerical path; reports
under `../reviews/O<n>_third_reader.md`), verified on 2026-09-23 against
PyVBMC's package code at `a65b96f4` (the head of `dev-port-review` after the
wave-1 check was merged into it), gpyreg at `v1.3.0` (`0186d89`) and MATLAB
VBMC at `396d649`. Reviewers and verifiers read a worktree frozen at
`a65b96f4` on the branch `dev-port-review-w7` while the merged head was
gated in the main checkout; the commits the branch gained during the wave
are records alone, and `git diff a65b96f4 -- pyvbmc` stayed empty. Four
read-only Opus verifiers, one per slice, wrote `wave7_O1.md` to
`wave7_O4.md` (rows O<n>-m); their scripts are under `scripts/`
(`wave7_O<n>_*.py`) and their logs on the machine that ran them
(`dev/scripts/runs/LOCAL.md`, "Port correctness review"). No `optimize()`
run was made and no MATLAB was run: a statement about MATLAB is read from
the source at `396d649` or measured on a Python transcription of it, and
the verifier reports say which.

Part 1 holds the findings that a user meets at a shipped option: in a run,
in a run resumed from a saved file, or in a public method of the result.
Part 2 holds the others. The column "proposed" holds the orchestrator's
proposal; the column "PI" is for the PI's ruling. PyVBMC 1.5 takes only bug
fixes and unequivocal improvements, and a fix may make an interface
stricter but may not change behavior silently (plan, "Decisions").

## Part 1: findings a user meets at a shipped option

| id | from | statement | verdict (class) | reach | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W7-1 | O3 F2; O3-2 | `warp_gp_and_vp` re-expresses a `ConstantMean` and a `NegativeQuadratic` and raises `ValueError("Unsupported GP mean function for input warping.")` for a `ZeroMean` (`whitening.py:435-472`), which `VBMC` accepts for `gp_mean_fun="zero"` (`vbmc.py:3905`, documented in `advanced_vbmc_options.ini:108`). Nothing catches the error, and the instance is left half-warped: `optim_state` and the function logger are the warped copies, the VP, the GP, `hyp_dict` and the transformer are not. MATLAB numbers its mean functions (0 zero, 1 const, 4 negquad) and `misc/warp_gpandvp_vbmc.m:38` has `case 0`, commented "Warp constant mean", which reads a hyperparameter past the end of a zero-mean vector; `'const'` falls to `otherwise` and errors | zero mean: confirmed shared defect (both stop). Constant mean: confirmed MATLAB-side defect (PyVBMC warps it exactly) | Not at the defaults (`negquad`). With `gp_mean_fun="zero"` and every other option at its default, every run with `D >= 2` that reaches a warp stops there, noiseless or noisy (`warp_rotoscaling` on, `K >= 5`, `r_index < 3`, and the spacing of warps; with `warmup=False` the first warp needs no end of warm-up) | `wave7_O3_F2_zero_mean_warp.py`: the `ValueError` for `ZeroMean`; for `ConstantMean` and `NegativeQuadratic` the shift of `m0` equals the constant shift of the stored log joint (-3.3586, spread 3.6e-15) to 4.4e-16 | Fix. The options: (a) warp the length scales alone for a zero mean and let the refit, which follows every warp, absorb the constant shift of the stored log joint that a zero mean cannot follow; (b) refuse `"zero"` at construction (stricter), which also makes `CHANGELOG.md`'s "the mean functions that PyVBMC implements" true; (c) no warping for a zero-mean run. Proposed (a), with a warp test on a bounded state for the constant and the zero mean that checks the `m0` shift against the logger's; the MATLAB side goes into `matlab_side_defects.md` | |
| W7-2 | O3 F6; O3-6 | At a kept warp, `warp_input` deep-copies `optim_state` (`whitening.py:219`) and so breaks the link between `optim_state["hyp_dict"]` and `self.hyp_dict`; active sampling, which relinks them (`vbmc.py:1571`, `:1586`), is skipped in that iteration. The iteration's record (`:1941`) therefore holds the `hyp_dict` of the previous iteration, in the old space, and `VBMC.load(file, iteration=k)` pairs it with the warped `get_gp(k)` (`:3252-3253`), so a run resumed there starts its GP fit from pre-warp hyperparameters and slice widths. An undone warp is not affected: the restored `optim_state_old` relinks before the fit (the report's "one fit behind" for that case is wrong) | confirmed Python-only defect (the resume feature, `a5f6effb`) | Only a resume from a kept-warp iteration, or `load(file)` of a run that stopped on one (possible through `max_iter`, not through `max_fun_evals`, since a warp iteration makes no evaluation). An uninterrupted run is unaffected | `wave7_O3_F6_hyp_dict_at_warp.py` replays the statements of `optimize` around a warp with the package's `warp_input`, `warp_gp_and_vp` and `IterationHistory`: at a kept warp the record equals the previous iteration's `hyp_dict` (length scales 0.6, 0.5, 0.8 against the warped 2.35, 1.59, 1.93); at an undone warp it equals the iteration's fit | Fix: relink after the warp (`self.optim_state["hyp_dict"] = self.hyp_dict`). `optim_state["hyp_dict"]` is read only by `active_sample` after the relink and by `load`, so no number of an uninterrupted run moves; a test that the record of a kept-warp iteration holds the post-warp `hyp_dict`. The run that measures the effect on a resume (O3 verifier, §9) is made when the heavy slot is free | |
| W7-3 | O2 F1; O2-1 | `_pdf` sums the component densities in the linear domain (`variational_posterior.py:919-928`) and takes the log afterwards (`:994-1000`), as `vbmc_pdf.m:58-66`, `:107-110` does: where the log density is below about -745 (38.6 standard deviations for a unit Gaussian) `log_pdf` returns `-inf` and its gradient NaN (`dy / y` = 0/0), both finite in truth, and in the subnormal band before that the value loses up to 7e-4 relative and the gradient 2.6e-2. `vp.mode()` meets it: on a narrow posterior the first trial point of its optimizer, about one coordinate unit long, lands there, and the start (the best of the draws, or a component mean) is returned unrefined. L-BFGS-B (`orig_flag=True`, the default) reports success; BFGS (`orig_flag=False`) usually stops with "precision loss" and sometimes recovers. A NumPy `RuntimeWarning` ("invalid value encountered in subtract", or "in divide") is printed, which does not say that the refinement was skipped | the log density and its gradient: confirmed shared defect. The mode search: confirmed on the Python side; for `vbmc_mode.m` it needs MATLAB (its objective is `+Inf` there, with no gradient requested, and recovery depends on `fmincon`/`fminunc`) | No run: the in-run readers floor the log density at `log(realmin)` (`acq_fcn_log.py:42`, `active_importance_sampling.py:481`) or combine it in a log-sum-exp with finite terms (`:362-364`), and `mode` is called by nothing in `optimize()`, the result, S-VBMC, the exports or the PyMC adapter. A user meets it through `vp.log_pdf` in the far tails and through `vp.mode()` (in the FAQ and example 2, whose posterior is wide enough): the default call skips its refinement for unbounded variables whose posterior SD is below about 0.013 original units (0.005 to 0.017 for bounded ones), so whether it refines depends on the units of the parameters. The error is small: the start is 0.3 % to 8 % of the smallest component SD from the mode, at most 0.01 nats | `wave7_O2_F1_pdf_tail_underflow.py` (with a transcription of `vbmc_pdf.m`, the same pattern and identical finite values), `wave7_O2_F1_mode_narrow.py`, `_mode_bfgs_trace.py`, `_mode_lse_counterfactual.py` (the same SciPy calls from the same starts converge on a log-sum-exp density), `_mode_start_error.py`, `_mode_warning_origin.py` | Fix: the log density by log-sum-exp in `_pdf` under `log_flag`, applied only where the linear sum falls below the smallest normal number, so that every value that is normal today stays bit-identical and the values that change lie below the floor of the in-run readers (the `vp_pdf` oracle under `make_oracle_fixtures.py --check --exact`, and the seeded runs, confirm that nothing moves); this repairs the value, the gradient and `mode`. Tests of the log density and its gradient in the tails and of `mode` on a narrow posterior, for both values of `orig_flag`. The shared part goes into `matlab_side_defects.md`. Records to correct: the P7 first question in the plan (`port-correctness-review.md:1397-1402`, "never at such a point": the starts are not, the trial steps are), `reviews/P7_internal.md` Q1 and row Q1 of `wave5_P7.md` ("unreachable") | |
| W7-4 | O2 §1; O2-2 | `handle_0D_1D_input` reads a 1-D array as one point, so for a one-dimensional posterior a flat array of `N > 1` points is one point of `N` coordinates. `pdf` and `log_pdf` then raise an `IndexError` or `ValueError`, except in the transformed space with a finite `df`, where the Student-t branches take `D` from `x.shape[1]` (`variational_posterior.py:877`) and return one meaningless number without error | confirmed Python-only defect (interface) | A user calling `vp.pdf` on a flat array with `D = 1`; no package caller (all pass 2-D arrays) | `wave7_O2_D1_input_shape.py`, nine calls over both flags, both transforms and `df` of 3 and -3 | Fix, stricter: `_pdf` refuses an `x` whose width is not `vp.D`, with a message that says a column of points is expected; this closes the silent case. The alternative, reading a flat array as a column when `D = 1`, departs from the decorator's convention elsewhere | |
| W7-5 | O3 F5; O3-5 | After a warp the search box is the minimum and maximum of 1000 uniform draws of the old box, mapped and widened by a thousandth of the range (`whitening.py:333-347`; `warp_input_vbmc.m:143-148`), where the exact image of the box under the affine warp has half-width `abs(A) h`. The extent per coordinate falls short of the exact one more as `D` grows (median 0.985 at `D = 2`, 0.741 at 10, 0.590 at 20) | confirmed shared defect, of no consequence | At every kept warp with `D > 1`, at the defaults, noiseless or noisy; nothing measurable: the volume lost is at most about `2D/1001` of the old box, all in far corners, the posterior mean stays at least 8.7 marginal SDs from every face, and the loss does not compound across warps (after two warps, at most 1e-4 of the original box) | `wave7_O3_F5_search_box_after_warp.py` (the package's `warp_input` on the default box with a posterior-like covariance), `wave7_O3_F5_two_warps.py` | Leave it as it is in both: the exact box would widen the search box up to about 2.3 times per coordinate at `D = 20` and move default trajectories for a gain nothing measures. Into `matlab_side_defects.md` among the items left as they are in both | |
| W7-6 | O3 F4, Q2; O3-4 | Every bounded transform goes through `z = (x - a)/(b - a)`, so the distance to the upper bound is resolved only to `(b - a) eps/2`: a loss beyond the representation of `x` where `abs(b) < b - a`, worst at `b = 0` (on `[-1, 0]`, at `b - x = 1e-12` the logit loses 8e-7 forward and 9e-5 back). The lower bound keeps full precision, zero or not; the Student-t(4) inverse alone also loses `x - a` at a zero lower bound. The log Jacobian as a function of `u` is exact, so the stored log joint stays consistent with the point the GP sees. MATLAB evaluates the same expressions. This answers the question of wave 0 (row N2 rerun F3 of `wave0.md`), left to slices P8 and O3: MATLAB shares it | confirmed shared defect (precision) | Reachable at the defaults (probit) on a bounded problem only for points within about `1e-10 (b - a)` of such an upper bound, and visible only where a stored point is transformed again (a warp, `vp.pdf(orig_flag=True)`, S-VBMC) | `wave7_O3_F3_F4_precision.py` | Leave it as it is in both; into `matlab_side_defects.md` among the items left as they are in both. The Torch export computes the distance to each bound separately and is accurate there. Records to correct: the plan's statement of the wave-0 question (`port-correctness-review.md:554-556`, "loss of precision near a nonzero bound"), which receives the answer and the criterion (an upper bound with `abs(b) < b - a`) | |
| W7-7 | O3 F3; O3-3 | `_student4` (`parameter_transformer.py:636-643`) forms `q - 1` by cancellation near `z = 1/2`: the absolute error in `t` reaches 1.55e-8, and within about 6e-9 of the midpoint it returns 0. `warpvars_vbmc.m:265-269` has the same expression; the Torch export (`_torch.py:23-39`) uses a cancellation-free form | confirmed shared defect (precision) | Only with the Student-t(4) transform, which is not the default; a displacement of at most 1.6e-8 in `u` | `wave7_O3_F3_F4_precision.py`, `wave7_O3_F3_round_trip_fine.py` | Leave it as it is in both (non-default, negligible); into `matlab_side_defects.md` among the items left as they are in both | |

## Part 2: the other findings

No row of this part changes a number of a PyVBMC run or of a public method
of its result at any shipped option.

| id | from | statement | verdict (class) | reach | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W7-8 | O1 F1; O1-1 | With `optimize_sigma=False` and `optimize_lambd=True`, `_neg_elcbo` assigns θ through `vp.set_parameters` (`variational_optimization.py:1228`), which rescales `lambd` to unit root mean square and multiplies the stored `sigma` by the same factor (`variational_posterior.py:1237-1240`); `optimize_vp` reuses one posterior, so the frozen `sigma` is multiplied on every call, and `_vp_bound_loss` (`:615`) reads a log scale off by that factor with a gradient that ignores it. Once reached the optimization is ruined: one `optimize_vp` call with `K = 1` takes the frozen `sigma` from 0.78 to 34648. A mirror case, `sigma` free and `lambd` frozen at a root mean square other than 1, gives two values of the objective at one θ on direct calls. MATLAB assigns `exp(theta)` to a by-value copy (`misc/negelcbo_vbmc.m:33-48`), so its objective is a function of θ alone | confirmed port discrepancy (MATLAB right). The Python matched until `abf5c3ec` (2021-10-31), a shape fix that routed the assignment through `set_parameters` | No run at any option: every production path sets both flags true, no option names them, and neither S-VBMC nor anything else sets them. Only by assigning the public attribute by hand. MATLAB cannot reach it either | `wave7_O1_F1_frozen_sigma.py`, with a transcription of `negelcbo_vbmc.m` and `vpbndloss.m` in `wave7_O1_common.py` | Fix: rescale in `set_parameters` only when `sigma` and `lambd` are both optimized, the one case with a gauge freedom. The production path, where both are, is then bit-identical, and both the frozen case and its mirror give an objective that depends on θ alone. Tests: finite differences with `sigma` frozen and `lambd` free and a scale bound active, and equal values at one θ on repeated calls | |
| W7-9 | O1 F2; O1-2 | `_neg_elcbo(compute_var=None)` computes the variance only for a nonzero `beta` (`:1211-1212`) and returns `varF = 0.0` otherwise; its docstring says only "determined automatically". MATLAB's default, `beta ~= 0 \|\| nargout > 4` (`negelcbo_vbmc.m:16`), combines with `compute_grad = nargout > 1` (`:10`), so a call that takes `varF` with the default gradient stops in `gplogjoint.m:25-29`: the reviewer's statement that MATLAB computes the variance there is wrong | documentation only; the MATLAB default is a dormant MATLAB-side defect | None: every caller on both sides passes `compute_var` | `wave7_O1_F2_F3_defaults_docstrings.py` | The docstring: "computed if and only if `beta` is nonzero". A sheet entry (text in `wave7_O1.md`, §8), and entry 56 of `matlab_side_defects.md` for the MATLAB default | |
| W7-10 | O1 F3; O1-3 | Docstrings: `_initialize_full_elcbo` calls `D` "the dimension" (it is the length of θ) and `Ns` the entropy's sample count (it is the number of GP hyperparameter samples); `_neg_elcbo`'s `Ns` is a count per component; `entropy_alpha` is documented where the parameter is `_entropy_alpha`; `compute_var` is documented as a bool and takes 0, 1 and 2 | documentation only | n/a | the same script | Docstring edits | |
| W7-11 | O2 §2; O2-3 | In the transcription `papers/acerbi2018variational_appendix.md`, line 88 has `(σ_k λ)^2` where the derivative gives `(σ_l λ)^2`, and line 113 carries a factor `1/K^2`; the code, MATLAB and the derivation use `σ_l` and `w_j/N_s`. Only the transcription could be checked, not the published PDF | documentation only (the code is right) | n/a | read against the derivation; the code's gradient confirmed exactly (O2 verifier, §4) | A note beside each of the two lines in the transcription, saying what the code uses and that the published PDF was not checked | |
| W7-12 | O3 F1, Q1; O3-1 | The sheet entry "The gradient of the log Jacobian is not ported" says `warpvars_vbmc.m` returns that gradient. Its `'g'` action returns the matrix of derivatives of the coordinate-wise inverse map at the coordinates before the rotation, which is not the gradient of the log Jacobian, and its one caller, `vbmc_pdf.m:119`, follows an unconditional `error`. No PyVBMC computation needs the gradient: the log Jacobian enters only as data in the stored log joint, `vp.pdf` refuses original-space gradients, the original-space mode search takes no analytic gradient on either side, and the Torch export differentiates its own by autograd | documentation only (and one dead MATLAB line) | n/a | `wave7_O3_F1_log_jacobian_gradient.py`: a transcription of `'g'` matches the finite-difference derivatives of the inverse to 2e-8 and differs from the gradient of the log Jacobian by up to 10.5 | Replace the sheet entry (text in `wave7_O3.md`, §8); the dead line into `matlab_side_defects.md` | |
| W7-13 | O3 F7; O3-7 | The sheet's reason for the probit default cites `AGENTS.md`, which held the sentence "probit by default" until its rewrite and never a reason; the default comes from `6cee9bb5` (2022-11-24, "feat: change default bounded variable transform to probit") | documentation only | n/a | `git log -S`, `git show 6cee9bb5` | Correct the "Why" (text in `wave7_O3.md`, §8) | |
| W7-14 | O4 F1; O4-1 | `__compute_log_priors` writes NaN into the prior's gradient for every coordinate whose bounds are equal (`gpyreg/gaussian_process.py:1716-1720`); only the Gaussian and Student's t branches, and the smooth-box branches for a value outside the box, overwrite it. The NaN survives for a fixed coordinate without a prior or with a smooth box around its value, reaches every `log_posterior(compute_grad=True)` (even with no prior set), and reaches `fit` whenever some coordinate has a prior, where L-BFGS-B stops within one iteration (log posterior -43.25 against 50.78). `gplite_hypprior.m` starts its gradient at zeros and has no branch for fixed coordinates. The sheet entry "The hyperprior is renormalized to the bounds, and a fixed hyperparameter gets a prior" says the mask changes only what a caller evaluating off the fixed value is told, and half its commit citations are off | confirmed Python-only defect; the sheet entry does not hold as written | Not in PyVBMC: the one pair that comes out equal on a usable training set is the noise's, for a target range below `tol_gp_noise`, and it always carries its Student's t prior. In gpyreg: a caller who fixes a coordinate without a prior while another has one (the configuration of `test_fitting_with_fixed_bounds`), or a target range below 1e-6 with a prior elsewhere (the collapsed noise pair of W6-38) | `wave7_O4_F1_fixed_bound_gradient.py`, `_fixed_with_smoothbox.py`, `_tiny_range_gpyreg.py`, `_pyvbmc_reach.py` (90 PyVBMC-built configurations) | Fix in gpyreg: write 0 for a fixed coordinate that its prior's branch leaves unset, which is what MATLAB effectively does; an assertion in `test_fitting_with_fixed_bounds` that sees it, and a smooth-box case with the value inside. Replace the sheet entry (text in `wave7_O4.md`, §8). A gpyreg patch release, with no PyVBMC number moving | |
| W7-15 | O4 F2; O4-2 | `__recompute_normalization_constants` stores the prior's mass inside the bounds as `cdf(ub) - cdf(lb)` (`:1592-1607`, and the upper branches of the smooth-box CDFs at `f_min_fill.py:300`, `:337`): with both bounds far in the upper tail (beyond about 8.3 scales for a Gaussian) both round to 1, the mass is 0, and `log_posterior` is `+inf` everywhere; a fit then returns a poor point (-85.2 against 25.9). The lower tail is exact | confirmed Python-only defect, in the renormalization that the sheet lists; the sheet's "neither the optimizer nor the slice sampler sees it" fails in that regime, and so does row W6-24 | Not in PyVBMC: its Student's t priors have masses 0.44 to 0.94 inside the bounds it fills | `wave7_O4_F2_prior_mass_upper_tail.py` | Fix in gpyreg: the survival function in the upper tail, with a switch chosen so that the masses PyVBMC's priors produce stay bit-identical (checked by PyVBMC's oracles and seeded runs against the patched gpyreg, as for any gpyreg change); a test in the upper tail. The sheet entry qualified (in the text of W7-14), and row W6-24 of `wave6.md` flagged | |
| W7-16 | O4 F3; O4-3 | `set_priors` validates `sigma` alone: a Gaussian or Student's t prior with a location that is infinite or NaN, and a smooth box with an infinite or NaN end, are accepted, and the log posterior is then NaN with the bounds `fit` fills (`-inf` without finite bounds for an infinite Gaussian location). `gplite_hypprior.m:35` reads a non-finite location as no prior | confirmed Python-only defect (input validation) | Not in PyVBMC (every prior of `_gp_hyp` is finite). In gpyreg, a user who writes such a prior; a fit with `("smoothbox", (0, inf, 1))` ends with NaN | `wave7_O4_F3_nonfinite_prior_location.py` | Fix in gpyreg, stricter: refuse a non-finite location as a non-finite `sigma` is refused. This is the item of `TODO.md` on a NaN location beside a finite `sigma`, widened, and the remainder of W6-17 that its fix left: one ruling closes all three, and the `TODO.md` line goes with the fix. Correct the sheet entry on `None` (text in `wave7_O4.md`, §8) | |
| W7-17 | O4 verifier, §9 (outside the slices) | A training set with an input column of zero width makes gpyreg's recommended length-scale bounds `(-inf, -inf)`, and a PyVBMC-built GP's `fit` ends with L-BFGS-B's `KeyError`. PyVBMC reaches it, read from the code and not run, when a user passes at least `fun_eval_start` starting points that share one coordinate: the initial design is then `x0[:fun_eval_start]` (`active_sample.py:127-185`). The formula is shared with `gplite_covfun.m:105`, `:121-124`. It is the mechanism of the `TODO.md` item on a fit on a single training point, reached by a set of distinct points | candidate, measured on a PyVBMC-built GP; the path from `x0` read from the code | Only with such starting points | `wave7_O4_F1_pyvbmc_reach.py` | For triage together with the `TODO.md` item on a single training point. The options: a finite floor under the width in gpyreg's recommendations; a refusal at construction of starting points that leave a coordinate without spread; or a `TODO.md` line. A capped run from such starting points, when the heavy slot is free, settles the path | |

## The first questions

**O1, the configurations the finite-difference tests leave out.** Every
gradient of `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss` and
`_soft_bound_loss` holds there: the zero and constant means (never checked
before, by finite differences or by value), `K = 1`, `D = 1`, several
hyperparameter samples, a weight below the threshold with the weight
penalty active, and the variance at separated and wide components against
quadrature of `predict_full`; worst relative errors 2.5e-10 for the
gradient of the expected log joint and 1.3e-11 for that of the ELCBO. The
production configuration the reach of every O1 row rests on is confirmed:
`elcbo_beta` is fixed at 0, so the gradient is always available, and every
`K >= 2`, warm-up included, takes the Adam path with the Monte Carlo
entropy.

**O2, the configurations the entropy and density tests leave out.** Every
gradient holds, apart from the tail case of W7-3: the lower-bound entropy
by complex step in fifteen configurations and both parameterizations
(relative errors at most 2e-12), the Monte Carlo entropy against the exact
path derivative of an independent estimator (6e-15 absolute), including
production-sized calls whose canonical blocks split one component's
samples, and the density's gradient by finite differences and against the
formula.

**O3, the three questions.** The gradient of the log Jacobian: no path
needs it on either side (W7-12). The precision near a bound: shared with
MATLAB, and the criterion is an upper bound with `abs(b) < b - a`, not a
nonzero bound (W7-6). The re-expression at a warp, checked on a bounded
state, which the warp tests never use: exact for the stored inputs and log
joint (a constant shift `C = -log abs(det A)`), the mean's location and
constant, the VP's means and weights and the search cache; approximated as
MATLAB approximates it for the length scales, the mean's scales and the
VP's scales, which take marginal standard deviations; the zero mean
refused (W7-1).

**O4, the gradient of the marginal likelihood in the configurations
PyVBMC builds.** It holds for each mean function and each of the three
noise configurations, the multiplier of the recorded noise at level 1
included (1.6e-15 relative against a dense evaluation), with and without
the hyperprior, at `D = 1` and with repeated inputs. In the low-noise
representation, which no run reaches (the noise floor gives a smallest
noise variance of at least 1e-5), the analytic gradient matches an 80-digit
reference where double-precision finite differences fail.

## Defects on the MATLAB side

For `matlab_side_defects.md`, all from a reading of the source; nothing was
run in MATLAB.

- 56. `misc/negelcbo_vbmc.m:10`, `:16`, `:21` with `misc/gplogjoint.m:25-29`:
  the default `compute_var` combines with the default `compute_grad` so that
  a call that takes `varF` stops (W7-9). Dormant. Not shared.
- `misc/warp_gpandvp_vbmc.m:37-66`: the cases are off by one mean code; a
  constant-mean run stops at its first warp, a zero-mean run reads past the
  end of the hyperparameters (W7-1). Shared for the zero mean.
- `vbmc_pdf.m:107-110`: the log density is the log of a linear sum, `-Inf`
  with a NaN gradient where it underflows (W7-3). Shared unless W7-3 is
  fixed.
- `vbmc_pdf.m:119`: dead code after an unconditional `error`; it would
  subtract derivatives of the inverse map, not gradients of the log
  Jacobian (W7-12).
- `vbmc_pdf.m:41`: the width of `X` is not checked against `vp.D`; for
  `D = 1` a row of points gives one number (W7-4). A user's error under
  MATLAB's convention.
- Left as they are in both, if so ruled: the sampled search box after a
  warp (`warp_input_vbmc.m:143-148`, W7-5), the resolution at an upper
  bound (`warpvars_vbmc.m:106-108`, `:256-258`, `:442-459`, W7-6), the
  Student-t(4) forward cancellation (`:265-269`, W7-7).
- Needs MATLAB, not written up: whether `vbmc_mode.m` recovers from an
  infinite trial value (W7-3). The disposition of W7-3 does not depend on
  it.

## Sheet entries

To replace: "The gradient of the log Jacobian is not ported" (W7-12); "The
hyperprior is renormalized to the bounds, and a fixed hyperparameter gets a
prior" (W7-14, W7-15). To correct: the reason of "The default bounded
transform is probit, not logit" (W7-13); "'No prior' is expressed by
`None`, not by an infinite scale" (W7-16); in "The mode search starts from
draws of the posterior", the optimizer of `orig_flag=False`, which is BFGS
with the analytic gradient, not L-BFGS-B (O2 verifier, §8). To add: the
default of `compute_var` (W7-9). The texts are in the verifier reports,
§8.

## Test notes worth acting on

- `_gp_log_joint` with the zero and constant means, by finite differences
  and by value; finite-difference checks at `K = 1`, `D = 1` and with the
  weight penalty active; the exact penalty gradient in
  `test_neg_elcbo_retains_capped_small_weight_penalty`; an independent pin
  of the variance at separated components (O1 verifier, §6).
- `test_neg_elcbo_grad_fd_mc_entropy` barely constrains the entropy's own
  gradient: scaled by 5 it still passes (O1 verifier, §5.4).
- The exact frozen-density check of the Monte Carlo entropy, in place of
  the fixed-draw checks; the unasserted `check_grad` of
  `test_entmc_vbmc_nonoverlapping_mixture` would fail if asserted as
  written, though the code is right (O2-4, O2-6).
- A budget case whose canonical blocks split one component's samples in a
  gradient call, and agreement across subsets of the gradient flags
  (O2-8); a finite-difference test of the linear density's gradient at
  `K = 1`, `D = 1` and several rows that does not need Torch (O2-7).
- `test_inverse_type3_max_space` repeats the min-space case, so the upper
  saturation of probit and Student-t(4) is untested (O3-8); the
  `test_transform_*_largeN` tests build 12 identical rows (`10 ^ 6` is an
  exclusive or) (O3-9); a warp test for the constant and the zero mean on a
  bounded state (O3-10).
- In gpyreg: an assertion in `test_fitting_with_fixed_bounds` that sees
  W7-14; the assembled likelihood gradient with per-point noise, `D = 1`
  and repeated inputs; the low-noise representation just below its
  threshold, where finite differences are reliable; a seed for
  `test_gp_gradient_computations` (O4 verifier, §6).

## Errors in the reports

The verifier reports list them in full. Those that change a row: O1's
statement that MATLAB's default computes the variance (W7-9); O2's "no
warning reaches the user" (W7-3: a `RuntimeWarning` is printed; the
reviewer's scripts ran with warnings ignored); O3's "one fit behind" for an
undone warp (W7-2) and "the shrinkage compounds across warps" (W7-5); O4's
statement that every prior family overwrites the NaN of a fixed coordinate
(W7-14: a smooth box around the value does not) and its test note that no
test checks the component gradients (`pyvbmc/testing/vbmc/test_gpyreg_derivatives_fd.py`
does; the assembly in `__core_computation` is what is untested).

## Runs for when the heavy slot is free

- W7-2: a seeded `D = 3` run on a correlated Gaussian that reaches one kept
  warp at iteration `k`, saved and resumed at `k` against a resume at
  `k - 1` as the control (O3 verifier, §9).
- W7-17: a capped run from starting points that share one coordinate.
