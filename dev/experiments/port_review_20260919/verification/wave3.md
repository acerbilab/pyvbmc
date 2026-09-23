# Wave 3 verification

The findings of wave 3 (slices P5 and P8; reports under `../reviews/`),
verified on 2026-09-20 on `dev-port-review` at `befab5d` (the package as the
reviewers read it at `96cded7`), gpyreg at `9e70e6b`, against MATLAB VBMC at
`396d649`. Part 1 holds the seven findings that can change what a run
computes; part 2 holds the others. Two sources feed the ledger: the
orchestrator's own checks, for six rows of part 1, with the scripts named
in the rows; and two read-only Opus verifiers, one per slice, whose raw
reports are `wave3_P5.md` (rows P5-n) and `wave3_P8.md` (rows P8-n). The
scripts are under `scripts/`; the logs of the orchestrator's are kept on
the machine that ran them (`dev/scripts/runs/LOCAL.md`, "Port correctness
review"). Every row was searched for a recorded reason for the difference
in the code comments, the commit and pull-request messages, the porting
log, the documentation, `dev/` and the tests; a row says so where one was
found. The MATLAB lines involved all predate the Python lines, so no row is
a faithful port of an older MATLAB. The "proposed disposition" column is
the orchestrator's; the last column is the PI's.

Two populations of stored runs serve as evidence of how often a finding
fires. The *stored runs* are the 21 complete runs with their iteration
history under `dev/scripts/runs/gp_box_20260916/` and
`dev/scripts/runs/svbmc_pool_20260913/pool/` (15 of them two-dimensional,
three six-dimensional). The *population posteriors* are the 990 posteriors
that the runs of the population campaigns ended with before their final
boost (`dev/scripts/runs/population_*/results/*.boost.pkl`, entry `pre`),
23 problems of 2 to 15 dimensions.

## Part 1: findings that can change what a run computes

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed disposition | PI disposition |
|---|---|---|---|---|---|---|---|
| W3-1 | P5 internal F3, P5 comparison F1 | At uncertainty level 1 (`uncertainty_handling=True` without `specify_target_noise`) the GP gets the noise function of level 0. `vbmc.py:1091` sets `gp_noise_fun = [1, 2, 0]` as `misc/setupvars_vbmc.m:279` sets `[1 2]`, which in `gplite/gplite_noisefun.m:66-70`, `:186-194` means two hyperparameters and the variance `exp(2*h1) + exp(h2)*s2`. `train_gp` (`gaussian_process_train.py:96-105`) translates the 2 into `user_provided_add=False, scale_user_provided=True`, and gpyreg's `GaussianNoise` reads the scale flag only inside `if user_provided_add:` (`noise_functions.py:36-39`; its docstring says so), so the parameters come out `[1, 0, 0]`. The function logger records SD 1 per evaluation at this level and pools repeats to `1/sqrt(n_evals)`; that `s2` reaches the fit and is ignored, so a point evaluated four times weighs as much as one evaluated once. The prior that `_gp_hyp:415-419` sets for `noise_provided_log_multiplier` names a hyperparameter the model lacks and is dropped without a message, and MATLAB's starting value `hyp0(Ncov+2) = log(noisemult)` (`misc/gptrain_vbmc.m:165`) has no Python line. The noise must then be carried by the constant term, whose prior at this level is centred at `log(tol_gp_noise) = -5.76` with scale `log 10`: a noise SD of 1 lies 2.5 scales above the centre, where MATLAB centres the multiplier at 1 | confirmed port discrepancy; never matched. Both Python sites date from `aa45473` (2021-07-26, "added GP training for VBMC", #19), written against a gpyreg constructor that already nested the flag (`24592ac`, 2021-05-26); the MATLAB lines predate the port. The comment at `:348`, from the same commit, reads "This branch is not used and tested at the moment". No reason is recorded anywhere | no; every run with `uncertainty_handling=True` and no `specify_target_noise`, the mode to which W2-7 gave the noisy-target defaults. Levels 0 and 2 are translated correctly | `scripts/wave3_A1_level1_noise.py`: the three levels' noise functions beside the one MATLAB's flags name, and one `train_gp` call on a level-1 state built through `VBMC` (12 evaluations, one point evaluated four times): `gp.noise.parameters = [1, 0, 0]`, 9 hyperparameters where MATLAB has 10, `s2 = [0.25, 1, ...]` handed to the fit, one fitted noise variance for every row; both sources read; both histories traced | fix: translate the flag as MATLAB does, set the multiplier's starting value, and check that its prior lands. The scaled provided-noise path has never run in PyVBMC, so the fix is checked by a short seeded level-1 run in the orchestrator's heavy slot, and by a test that a level-1 GP has two noise hyperparameters and that its noise variance follows `s2`. No oracle moves (the fixtures are levels 0 and 2). Changelog line: level-1 runs change | fix now, with the level-1 run as its check; level 1 becomes a first question of the briefs of slices P3 and P4 |
| W3-2 | P5 comparison F5; P5 internal lists the line among those checked and found correct | The hyperparameter samples of past GPs that seed the starting points of the hyperparameter fit are taken from the iterations `range(ceil((n+1)/2) - 1, n)` of the history (`gaussian_process_train.py:139-142`), where `misc/gptrain_vbmc.m:39` has `ceil(numel(stats.gp)/2):numel(stats.gp)`, in 0-based terms `range(ceil(n/2) - 1, n)`. `n` is the same count on both sides at the call: the record of an iteration is written at its end on both sides (`vbmc.m:803`, `:1044`; `vbmc.py:1620`), and `IterationHistory.record` extends the array to the recorded iteration and no further. The two ranges agree for odd `n`; for even `n` Python leaves out the oldest GP of MATLAB's window (`n = 4`: `[2, 3]` against `[1, 2, 3]`). The internal reviewer read the line as the translation of a 1-based `ceil((n+1)/2):n` without reading MATLAB; the comparison reviewer is right | confirmed port discrepancy; never matched. The Python range is unchanged since `aa45473` (2021-07-26), under its comment "Be very careful with off-by-one errors compared to MATLAB in the range here"; the MATLAB line predates the port (`969c4ff`, 2019-12-04, and earlier). No reason recorded | yes. On the stored runs 361 of the 716 main training calls collect past samples (`init_N > 0`), and in 170 of them the window is one GP short, a GP of 8 samples each time. The samples only join the candidates from which the fit picks its starting points: the search differs, the model does not | `scripts/wave3_A_stored_runs.py` (the two index sets per call, on the stored runs); both sources read; both histories traced | fix, in one pass with W3-3, W3-4 and W3-5: all four change the starting points or the sampler widths of the hyperparameter fit at the defaults, so the pass moves the `gp_fit` oracle and the golden traces once, by the rule for a fix that moves default trajectories | fix, all five moving rows in one pass |
| W3-3 | P5 comparison F6 | When the pool of past samples holds more than `init_N/2` rows it is subsampled to `ceil(init_N/2)` rows (`gaussian_process_train.py:152-158`) where `misc/gptrain_vbmc.m:44` keeps `floor(Ninit/2)`: one row more whenever `init_N` is odd | confirmed port discrepancy. The first version passed the float `init_N / 2` to `np.random.choice`, which refuses it (`aa45473`); `9267816` (2021-08-25, "added VP optimization", #26) repaired that with `math.ceil`. MATLAB's `floor` predates the port. No reason recorded | yes, rarely: on the stored runs the pool is subsampled in 7 of the 361 collecting calls, late in the noisy two-dimensional Rosenbrock runs, and `init_N` was odd in all 7 | same script; both sources read; `git log -S` | fix with W3-2 | fix, with W3-2 |
| W3-4 | P5 comparison F7 | The iteration history records under `gp_hyp_full` the hyperparameter samples of the GP, the thinned set of `Ns` rows (`vbmc.py:1610`, `self.gp.get_hyperparameters(as_array=True)`), where MATLAB records the chain before thinning, `Ns * Thin` columns (`misc/gptrain_vbmc.m:65` `hypstruct.full = gpoutput.hyp_prethin`, passed to `savestats` at `vbmc.m:804` and stored at `:1041`; `gplite/gplite_train.m:314`, `:460`, `:482`). `train_gp` keeps the chain before thinning in `hyp_dict["full"]` (`:180`, gpyreg's `sampling_result["samples"]`), and the record does not use it. The recorded blocks feed the weighted covariance of `_get_hyp_cov`, which at the default `weighted_hyp_cov = True` sets the widths of the slice sampler: PyVBMC estimates it from a fifth of the draws MATLAB uses. Both estimate the same covariance, so the difference is sampling error in the widths, not a bias | confirmed port discrepancy; never matched (`aa45473` recorded `gp.get_hyperparameters(as_array=True)` from the start; MATLAB's record of the chain before thinning is from 2019, `d9822e3`). No reason recorded | yes, every iteration after the first in which the fit samples: on the stored runs all 732 recorded blocks of a sampling iteration hold `Ns_gp` rows, where MATLAB holds `5 * Ns_gp` | same script; both sources read; both histories traced | fix with W3-2, recording `hyp_dict["full"]`, together with the P5 row on the chain that `hyp_dict["full"]` keeps when the fit stops sampling (MATLAB replaces it with the single optimized vector), since the record would otherwise repeat the last chain drawn | fix, with W3-2 |
| W3-5 | P5 comparison F3 (the `mean_const` part; the other bounds of that finding are in part 2) | `_gp_hyp` writes the bounds of `mean_const` as the pair `(-inf, max(y_hpd) + delta_y)` (`gaussian_process_train.py:389`; `:382` for the constant mean). MATLAB builds `LB_gp` and `UB_gp` as NaN vectors and assigns the upper entry alone (`misc/gptrain_vbmc.m:174-188`); `gplite_train.m:120-127` fills what is still NaN from the full training set, `min(y)` for this lower bound (`gplite_meanfun.m:182`). gpyreg fills only NaN too (`gaussian_process.py:1163-1175`), so PyVBMC's `-inf` stays. `mean_const` has no hyperprior, and the space-filling design of the fit draws such a coordinate from the mixture of uniforms over `[LB, PLB, PUB, UB]` only when both bounds are finite, from the plausible box alone otherwise (`f_min_fill.py:119-139`, as `gplite/private/fminfill.m:73-82`). Every design point of that coordinate then lies in `[median(y), max(y)]`, and the optimizer and the sampler have no lower bound there | confirmed port discrepancy; never matched (`aa45473`). No reason recorded | yes, every fit with `init_N > 0` | `scripts/wave3_A5_mean_const_bound.py`: on a 20-point state the installed bounds are `(-inf, 5.36)` against the recommendation `(-0.51, 7.23)`; of 1024 design points none leaves the plausible box `[2.52, 3.36]` with `-inf`, and 7.3% do with MATLAB's convention (4.4% below, 2.9% above); both sources read; `git log -L` | fix with W3-2: `_gp_hyp` writes NaN where MATLAB leaves the entry unset, which also settles the noise upper bound and the overwritten `upper_gp_length_factor` bound of part 2 | fix, with W3-2 |
| W3-6 | P8 internal F7 | `warp_input` zeroes the entries of the posterior covariance whose correlation is at most `warp_roto_corr_thresh` (0.05) and takes the SVD of the result as its eigendecomposition (`whitening.py:154-182`). From `D = 3` on, a covariance with entries zeroed need not be positive semi-definite; the SVD then returns the absolute value of a negative eigenvalue and flips a column of `U`, and the transform misses unit variance in that direction. The reviewer's example reproduces: correlations `0.72, 0.72, 0.04` give eigenvalues `-0.018, 1, 2.018` after the threshold and the attained variances `1.01, 0.96, 0.097` | shared by both implementations and by the recipe of the paper: `misc/warp_input_vbmc.m:52-71` has the same threshold and the same `svd`, and `papers/acerbi2020variational_appendix.md` §B.2 prescribes it. Not observed: none of the 990 population posteriors, 718 of which have an entry zeroed, comes out indefinite, and the threshold removes at most 16% of the smallest eigenvalue (nothing, at the median); none of the 420 eligible recorded posteriors of the stored runs does either. It needs a nearly singular correlation matrix together with small entries placed against it | latent at the defaults (`warp_rotoscaling = True`) | `scripts/wave3_A6_threshold_population.py` (the covariance built as `whitening.py:142-151` does, per posterior); `scripts/wave3_A_stored_runs.py`; the example in `scripts/wave3_A5_mean_const_bound.py`; both sources and the paper read | the PI's, as a question of the algorithm. A guard that keeps the covariance as it was when the thresholded one is not positive definite would change no run of the population and removes the mis-scaling; leaving the recipe as it is and recording it as shared is the alternative | the guard: the covariance is kept as it was when the thresholded one is not positive definite; sheet entry |
| W3-7 | P5 internal F6, P5 comparison F3 (the noise part) | `_gp_hyp:374-375` writes the bounds of `noise_log_scale` as the pair `(log(tol_gp_noise), +inf)` under the comment "Increase minimum noise", where `misc/gptrain_vbmc.m:180` raises the lower entry alone and the upper one is filled with the recommendation `log(max(y) - min(y))`. The noise column of the fit's space-filling design is its Student-t prior truncated to the bounds, and the optimizer and the sampler lose the upper cap | confirmed port discrepancy (P5-6a); never matched (`aa45473`). Issue 99 (2022-09-19) names the confusion, "the element to set default bounds should be `np.nan`, not `np.inf`"; pull request 116 repaired it in gpyreg and in the covariance block of `_gp_hyp` and left this line | yes, every fit; small (the truncation constant moves from 1 to about 0.9992 on the verifier's state) | `wave3_P5.md`, row P5-6a | fix with W3-5, by the same change | fix, with W3-2 |

## Part 2: the other findings

Rows P5-n are in `wave3_P5.md` and rows P8-n in `wave3_P8.md`, the raw
reports of the two verifiers, which hold the supporting detail, the dating
of each row and the errors they found in the reviewers' reports.

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed disposition | PI disposition |
|---|---|---|---|---|---|---|---|
| W3-8 | P5 internal F1, P5 comparison F2 | `_gp_hyp` sets the upper bound of the length scales from `upper_gp_length_factor` (`:368-373`) and 26 lines later assigns the same dictionary key again, `(LB, nan)`, in the branch of the squared-exponential kernel, which is always taken (`:399-402`). The option has no effect at any value, is documented, and is not among the inert options, whose guard sees that it is read | confirmed port discrepancy (P5-1); matched MATLAB from `aa45473` (2021-07-26) until `5f09709` (2022-11-11, pull request 116) added the second assignment | no (the default is 0, as MATLAB's) | P5-1 | fix with W3-5, with a test that sets the option | as proposed |
| W3-9 | P5 internal F4 and F2, P5 comparison F9 and F8 | Three defects of the running covariance of the GP hyperparameters, `hyp_dict["run_cov"]`. The reset at the end of warm-up writes `hyp_dict["runcov"]`, a key nothing reads (`vbmc.py:1657`, under the MATLAB line it ports, `vbmc.m:831`). The guard of the update tests `shape[1] > 1`, the number of hyperparameters, where `misc/gptrain_vbmc.m:83` tests the number of samples. And when the fit does not sample, `hyp_dict["full"]` keeps the last chain drawn and the update goes on folding its covariance in, where MATLAB replaces the chain with the single optimized vector and clears the running covariance | confirmed port discrepancies (P5-2, P5-3, P5-4); the key was mistyped when the line was written (`4949c82`, 2021-11-04, the training module already had `run_cov`); the other two are from `aa45473`. No reason recorded | the writes happen in every run; no number changes at the defaults, where `weighted_hyp_cov = True` and `run_cov` is not read | P5-2, P5-3, P5-4 (`full2 is full1` after a second training call without sampling) | fix the three together: the guard and the stale chain depend on each other, and W3-4 records `hyp_dict["full"]` | as proposed |
| W3-10 | P5 internal F5, P5 comparison F14 | `np.min(a, b)` for `np.minimum(a, b)` in the bound of the output-dependent noise (`:427`); it raises `TypeError` | Python-only defect (P5-5), unreachable: nothing sets `gp_noise_fun[2]`, and `noise_shaping`, which is rejected at construction, would set index 1 | no | P5-5 | fix (two characters) | as proposed |
| W3-11 | P5 comparison F4 | The lower bounds of the length scales and of the output scale come from the high-posterior-density subset, where MATLAB leaves them to `gplite_train`, which uses the full training set | intentional difference, missing from the sheet, and the justification holds (P5-7): issue 99 and pull request 116, for training sets with an extreme range of values | yes, every fit | P5-7 | keep; sheet entry | as proposed |
| W3-12 | P5 internal F7 | Construction accepts twelve `gp_mean_fun` names, the list of `misc/setupvars_vbmc.m:287`, of which `train_gp` implements three; with one of the other nine the run fails in its first GP training, after the initial design has spent its evaluations. The error message misspells `negquad` as `egquad` and names four supported functions | confirmed port discrepancy of the validation (P5-8); the unported mean functions themselves are on the sheet | no | P5-8 | fix, stricter interface: construction rejects the nine names; the message corrected | as proposed |
| W3-13 | P5 comparison F12 and F13 | `gp_hyp_sampler` accepts seven values and gpyreg runs one: every value but `slicesample` raises inside `GP.fit` at some point of a run. Two lines of the branches that cannot complete also differ from MATLAB: with no usable covariance `covsample` keeps its name where `misc/get_GPTrainOptions.m:71-74` falls back to `slicesample` (the test `test_get_gp_training_options_samplers` asserts the differing value), and the burn-in of `slicelite` has the division by `log(gp_retrain_threshold)` inside the logarithm | confirmed port discrepancies in code that cannot complete a fit (P5-11, P5-12) | no | P5-11, P5-12 | stricter interface: construction rejects every value but `slicesample`, and the branches of the other samplers are removed, as the branch of `separate_search_gp` was (W2-15); sheet entry for the unported samplers | reject at construction and remove the branches |
| W3-14 | P5 comparison F10 | The number of design points of the hyperparameter fit is floored at 9 where `misc/get_GPTrainOptions.m:100` floors it at 0, which also switches off the design and the collection of past samples | confirmed port discrepancy (P5-9); the 9 is in the module's first commit and no reason for it is recorded anywhere | no; only with `max_fun_evals` above 1000, from about 1380 evaluations on, and never on the budget path | P5-9 | fix: the floor is 0 | fix: the floor is 0 |
| W3-15 | P5 comparison F11 | The retrain-threshold branch tests `iteration > 1` where the same function translates the same MATLAB predicate as `iteration > 0` | confirmed port discrepancy without effect (P5-10): the reliability index is infinite for the first two iterations on both sides, so the branch first fires at the same iteration | no | P5-10 | fix (one character), with a test at the boundary | as proposed |
| W3-16 | P5 comparison F17 | `_estimate_noise` and `get_hpd` order by `np.argsort(y)[::-1]`, whose order among equal values is neither MATLAB's stable descending sort nor its reverse, since NumPy's default sort is not stable; a tie at the cut puts another point in the subset. `get_hpd` also feeds the bounds of `_gp_hyp`, the search box of active sampling and the initialization of the posterior | confirmed port discrepancy (P5-15), stronger than reported | only with exactly equal log-density values at the cut, as a quantized likelihood gives | P5-15 (the subset differs from MATLAB's in 1148 of 2000 random integer-valued vectors) | fix: a stable descending order at both sites, after checking the stored states of the oracles for equal values | fix; no stored state of the oracles holds equal values (checked 2026-09-20), so no reference moves |
| W3-17 | P5 comparison F15 | `train_gp` has no counterpart of the `try`/`catch` with which `misc/gptrain_vbmc.m:37-50` falls back to the current hyperparameters when a recorded GP has another shape | confirmed port discrepancy, latent (P5-13): the model cannot change shape within a run since the branch of `separate_search_gp` was removed | no | P5-13 | leave | as proposed |
| W3-18 | P5 comparison F16 | The evaluation times are not carried onto the GP (`gp.t`) | not a defect (P5-14): nothing reads the field in MATLAB | no | P5-14 | sheet entry, one line | as proposed |
| W3-19 | P8 comparison F4, P8 internal F5 (second half) | `warp_input` re-transforms the rows of the function logger whose `X_flag` is true (`whitening.py:213-220`), where `misc/warp_input_vbmc.m:112-119` re-transforms every filled row. The rows that the trim at the end of warm-up deactivated keep `X` and `y` of the previous inference space beside a correct `X_orig` and `y_orig`. The duplicate scan of `_record` is the only reader of an inactive row; every computation indexes through `X_flag` | confirmed port discrepancy (P8-1); never matched (`4aa4ca9`, 2022-03-03, the first version of `warp_input`). No reason recorded | yes. The oracle fixture `corr_D5_warped`, a state at the default options after one warp, holds 5 inactive rows that are off by up to 8.9 in `X` and 11.6 in `y`. No number of a run changes; what changes is `function_logger.X` and `.y` as a user reads them, and whether a repeat at a trimmed input is recognized | P8-1 | fix, with a warp test on a logger that holds evaluations and an inactive row | fix |
| W3-20 | P8 internal F5 (first half) | A repeat at an input whose row was deactivated is pooled into that row, which stays inactive, so the evaluation is paid for and seen by nothing | confirmed shared defect (P8-2): `record` in `misc/funlogger_vbmc.m:220-247` scans every row and writes no flag either | no; it needs `max_repeated_observations > 0` together with `search_cache_frac > 0`, both 0 by default on both sides, the search cache being the one route by which a trimmed input can be proposed again | P8-2 | leave; listed as shared in `matlab_side_defects.md` | leave |
| W3-21 | P8 internal F4, P8 comparison F7 | In the duplicate branch of `_record` the running average of the evaluation time is taken before the test for an unknown time, which the new-row branch puts around the assignment; `add` defaults the time to NaN, so a repeat through `add` turns a measured time into NaN for good. MATLAB's `'add'` passes 0. Separately, `add` with an explicit time adds it to `total_fun_eval_time`, which MATLAB's `'add'` never touches; no caller in the package passes one | confirmed port discrepancy (P8-3). Nothing reads the per-row times on either side (`gp.t` has no reader); `total_fun_eval_time` is read by `FunctionLogger.__str__` alone, where MATLAB's feeds `output.overhead` | no; a repeat through `add` at a row with a measured time, which needs `f_vals` or `precomputed_evaluations` | P8-3 | fix the guard of the average; leave the total | as proposed |
| W3-22 | P8 internal F13, P8 comparison F8 | The logger returns a one-element array for a repeat and a scalar for a new point; both docstrings say `float`, and MATLAB returns a scalar in both branches | confirmed port discrepancy (P8-4), without effect today: the one consumer that would hand the array to gpyreg (`active_sample.py:808`) runs only for a first evaluation. The comparison report's statement that the value is discarded is wrong | no (a repeat needs `max_repeated_observations > 0`) | P8-4 | fix | as proposed |
| W3-23 | P8 internal F10 (second half), P8 comparison F5 | The public `ParameterTransformer` takes a variable with one finite bound as unbounded: identity map, inverse outside the support, constant log-Jacobian. MATLAB has log transforms for it (types 1 and 2 of `shared/warpvars_vbmc.m`). `VBMC` rejects such bounds on every route, as `misc/boundscheck_vbmc.m:138-143` does | intentional difference, missing from the sheet (P8-5). The port never had the transforms: `6a247e5` (2021-02-25) removed the type labels, under which the code already applied the identity | no; only a caller who builds the transformer directly | P8-5 | stricter interface: the constructor raises for a half-bounded variable; sheet entry for the unported types | as proposed |
| W3-24 | P8 internal F10 (first half) | The transformer keeps the caller's bound, scale and rotation arrays by reference, so a later change of such an array changes the transform; and its docstring documents the keyword `bounded_transform_type` where the signature has `transform_type`, which is what the API page publishes. The constructor's default `"logit"` is MATLAB's create-time default; the package default `"probit"` is the option's | Python-only defects (P8-6), the first latent: `VBMC` hands it freshly detached arrays | no | P8-6 | fix both | as proposed |
| W3-25 | P8 internal F1 | A transformer constructed with `scale` or `rotation_matrix` and with plausible bounds tighter than the hard bounds reads its centering constants off coordinates taken after the rotation and the rescaling, and applies them before: the plausible box no longer maps to `[-0.5, 0.5]`. The map stays a bijection with a consistent log-Jacobian | Python-only defect, latent (P8-7). MATLAB's create branch takes neither argument. No construction site of the package passes them; a warp installs them after construction and resets the centering, as MATLAB does | no | P8-7 | fix: the constants are derived before the rotation and the rescaling; a test asserts that the plausible box maps to `[-0.5, 0.5]`, which the present tests of `mu` and `delta` restate the constructor instead of asserting | fix: the centering is derived before the rotation |
| W3-26 | P8 internal F9 | `scale` is stored unchecked where `rotation_matrix` is validated, and `log_abs_det_jacobian` adds `log(scale)`: NaN for a negative entry, a singular map for a zero | shared formula (`shared/warpvars_vbmc.m:763-765`), Python-only exposure (P8-8): MATLAB gives a user no way to supply a scale, and a warp always produces a positive one | no | P8-8 | stricter interface: the constructor requires a finite positive `scale` of length `D` | as proposed |
| W3-27 | P8 internal F2 and F3 | `unscent_warp`, exported with a documented contract, truncates its sigma points when `x` has an integer dtype, and raises in its branch for a single-row `x` with a many-row `sigma`, because it reshapes to the shape `x` had before broadcasting. The package calls it with float arrays and with the other branch | Python-only defect and confirmed port discrepancy, both latent (P8-9a, P8-9b) | no | P8-9a, P8-9b | fix both | as proposed |
| W3-28 | P8 internal F6 | `FunctionLogger.add` without an SD records SD 1 whatever the uncertainty level, so with `specify_target_noise` and `f_vals`, which has no channel for an SD, the supplied values enter the GP with noise 1. `precomputed_evaluations` refuses the same case. MATLAB intends the same default and raises before reaching it (entry for `misc/funlogger_vbmc.m:159-162` below) | confirmed port discrepancy (P8-10): PyVBMC accepts and invents a value where MATLAB raises. The `f_vals` option is documented by six words | no | P8-10 | stricter interface: construction refuses `f_vals` with `specify_target_noise` and points to `precomputed_evaluations`; `add` without an SD raises at level 2 | refuse the combination at construction; `add` without an SD raises at level 2 |
| W3-29 | P8 internal F8 | The block of `warp_input` under the comment "Reset GP Hyperparameters" clears the running average of the variational moments | not a defect (P8-11): `misc/warp_input_vbmc.m:164-167` verbatim, comment included. What the reviewer suspected holds on both sides alike: after a warp, only `hyp_dict["hyp"]` is warped, and the hyperparameter statistics of before the warp (the running covariance, the last chain, the recorded GPs and chains) go on feeding the starting points and the sampler widths. The target density of the fit is unaffected | yes, on both sides | P8-11 | leave; listed as shared in `matlab_side_defects.md` | as proposed |
| W3-30 | P8 internal F11 | `warp_input` divides the log-Jacobian by the temperature and `_record` does not; MATLAB divides in both places and also divides the recorded value and its SD | confirmed port discrepancy, unreachable (P8-12): `temperature` is an inert option and never reaches `optim_state` | no | P8-12 | leave to whatever ports tempering; a line in the sheet's entry on tempering | as proposed |
| W3-31 | P8 internal F12 | The 0-D/1-D decorator returns a 2-D result for a 0-D input where it unwraps a 1-D one; `input_dims` is unbound if a patched argument has a default; one variable serves several patched arguments | Python-only (P8-13). The first is pinned behavior: eight tests named `test_0D_*` assert the `(1, 1)` result, against the internal report's statement that no test passes a 0-D input. The other two are latent: each of the five decorated methods has one patched argument, without a default | yes for a user who passes a 0-D input | P8-13 | leave; the decorator's docstring states what a 0-D input returns | leave; the docstring states it |
| W3-32 | P8 internal F14 | A logger built with `noise_flag=True` at uncertainty level 0 records SD 1 through `__call__` and NaN through `batch_call` | Python-only defect (P8-14), unreachable from `VBMC`, which sets the flag from the level | no | P8-14 | stricter interface: `FunctionLogger` rejects a flag and a level that disagree | as proposed |
| W3-33 | P8 comparison F1, F2 and F3 | Three differences in the floating-point evaluation of the bounded transforms; everything else is bit-identical to a transcription of `shared/warpvars_vbmc.m` on 20 000 points per configuration. The direct transform moves a unit-interval image that rounds to 0 or 1 to the adjacent number, where MATLAB returns an infinity; a default run reaches it when a warp re-transforms a stored point that the inverse had clamped beside a bound. The clamp of the inverse uses `nextafter` where MATLAB uses `eps(bound)`, one ulp apart at a bound that is a power of two. The inverse of `student4` groups one product differently, one ulp | the first is an intentional difference, missing from the sheet, with its reason in pull request 89 ("points strictly within the bounds in the original space should never map to points at infinity"); the other two are confirmed port discrepancies of one ulp (P8-15a, b, c). The commit message of `3e9d1c2`, "identically to MATLAB", is wrong for both of its changes: before it the clamp was MATLAB's | the first two rarely, on bounded problems; the third only with `bounded_transform="student4"` | P8-15a, b, c | keep all three; sheet entries. They answer the question on precision near a bound that wave 0 left to slice P8 | as proposed |
| W3-34 | P8 comparison F6 | The low-correlation mask zeroes `abs(corr) <= thresh` where MATLAB keeps `abs(corr) > thresh` and zeroes the rest; the two differ on NaN alone | confirmed port discrepancy, unreachable (P8-16): it needs a negative or non-finite variance, not a zero one as the report says | no | P8-16 | fix (one comparison) | as proposed |
| W3-35 | P8 comparison F9 | The pooling of a repeated noisy observation falls back to a scale-invariant form when the precisions overflow; the ordinary path is MATLAB's arithmetic bit for bit | intentional difference, missing from the sheet (P8-17); `6769a9a` (2026-09-16), recorded in `dev/plans/pymc-target-adapter.md` | only with an SD below about 1e-154 | P8-17 | keep; sheet entry | as proposed |
| W3-36 | P8 comparison F10 | The plausible bounds after a warp use `np.quantile`, whose convention differs from MATLAB's `quantile` | intentional in kind, the substitution the sheet records for gpyreg's bound recommendations, not recorded for this site (P8-18) | yes, every warp, by about 1e-4 | P8-18 | keep; the sheet entry names this site too | as proposed |

## Defects on the MATLAB side

Entries 15 to 20 of `matlab_side_defects.md`, all from a reading of the
source; nothing was run in MATLAB. The reviewers' claims that did not hold are in the two raw
reports.

| location | what the code does | consequence | PyVBMC |
|---|---|---|---|
| `misc/funlogger_vbmc.m:244` | The duplicate branch of `record` writes the pooled value to `optimState.y(optimState.Xn)` for `optimState.y(idx)` | The last filled row gets another point's value and the repeated row keeps its old one; both feed the GP and `ymax`. Only with `MaxRepeatedObservations > 0`, which no default sets, the noisy defaults included; then nearly every repeat | not shared: `self.y[idx]` since `2527c47` (2021-05-20) |
| `misc/funlogger_vbmc.m:159-162`, with `private/activesample_vbmc.m:388` and `misc/initdesign_vbmc.m:56` | The `'add'` action reads `varargin{2}` whenever the run is noisy, and both call sites pass the value alone | A noisy run with `Fvals`, or one that acquires a cached starting point with a value, raises an index error before the line that defaults the SD to 1 | not shared; see W3-28 for what PyVBMC does at level 2 |
| `misc/setupvars_vbmc.m:96`, with `misc/warp_gpandvp_vbmc.m:8-10` | `vp.temperature = NaN`, behind a guard that tests for empty | None: `misc/vpoptimize_vbmc.m:190` overwrites it before a warp can occur. The order of the loop is what saves it | no counterpart |
| `misc/gptrain_vbmc.m:19-25` | The branch on `BOWarmup` and its `else` call `vbmc_gphyp` with the same arguments | `BOWarmup` never switches the GP mean function, and `vbmc.m:825-828` restores one that never changed | no counterpart (the option is not ported) |
| `misc/get_GPTrainOptions.m:112` | The burn-in of `slicelite` divides by `log(GPRetrainThreshold)`, which is 0 at the default | In the branch's own region the quotient is minus infinity and the `max` returns 1: the burn-in is `Ns_gp` whatever the reliability index. The comparison report's "Inf or NaN" does not hold | the transcription differs and cannot run (W3-13) |
| `misc/gptrain_vbmc.m:33`, with `misc/get_GPTrainOptions.m:63` | For `covsample` the widths are built as an `Nhyp` by `Nhyp` covariance, and the caller discards widths whose number of elements differs from `Nhyp` | The covariance never reaches the sampler | shared (`train_gp:128-131`); moot once W3-13 rejects the value |

Shared by both implementations and left as they are: the repeat pooled into
a deactivated row (W3-20), the hyperparameter statistics of before a warp
that go on being used after it (W3-29), and `log(scale)` in the
log-Jacobian (W3-26). The thresholded covariance that need not be positive
semi-definite (W3-6) was shared too, with the paper's recipe; since
`61a7325`, on the PI's ruling, PyVBMC keeps the covariance as it was in that
case and no longer shares it.

## Sheet entries

Made with the fix pass (`9c29ce1`). Nine new entries: the lower bounds from
the high-posterior-density subset (W3-11); the hyperparameter samplers other
than slice sampling (W3-13); the evaluation times that are not carried onto
the GP (W3-18); the covariance the warp keeps when its threshold would leave
it indefinite (W3-6); the half-bounded transform types (W3-23); the nudge,
the clamp and the grouping in the bounded transforms (W3-33); the pooling
fallback (W3-35); the cached value that needs its SD (W3-28); the validation
of `scale` and of the logger's noise flag (W3-26, W3-32). Three extended
entries: NumPy's quantile convention, for the warp's plausible bounds
(W3-36); posterior tempering, for the divisor `_record` lacks (W3-30); the
mean functions, for the check at construction (W3-12).

## Test notes worth acting on

Both verifiers opened every test the reviewers cited; the full lists are in
their reports. With the fixes above:

- `test_gp_hyp` passes the original plausible bounds where `train_gp` takes
  the transformed ones, a box four times wider in that fixture, so the
  length-scale prior it builds is not the production one; it asserts two
  numbers of one prior and no bound. It is corrected before it is extended
  to the bounds (W3-5, W3-7, W3-8).
- `test_get_gp_training_options_samplers` asserts `"covsample"` where MATLAB
  answers `'slicesample'` (W3-13), and `test_get_gp_training_options_opts_N`
  holds `iter = 2` and `n_eff = 10` throughout, away from both boundaries
  (W3-14, W3-15).
- Nothing builds a GP at uncertainty level 1 (W3-1), calls `train_gp` twice
  (W3-9), or pins that the lean GP record is taken from the freshly fitted
  GP, on which the exactness of the restored factors rests.
- Every warp test runs on an unbounded problem and on a logger without
  evaluations, so the rewrite of the stored points never executes (W3-19).
- `test_init_type3_mu_all_params` and `test_init_type3_delta_all_params`
  build their expectation with the constructor they test (W3-25); `scale`
  has no validation test (W3-26); `test_add_no_f_sd` covers level 1 alone
  (W3-28); no test records a repeat with an unknown time (W3-21).
- `test_rotoscaling_rotation_2d`, `test_bounded_log_abs_det_jacobian_numerically`
  and `test_transform_bounded_and_unbounded` draw from the unseeded global
  generator.

## Fix commits

The PI ruled on all 36 rows on 2026-09-20. The fixes are on
`dev-port-review` after `fc8e561`, made by three Opus agents on worktrees
(reports `../fixes/wave3_agent_A.md`, `_B.md`, `_C.md`), reviewed and
cherry-picked by the orchestrator: 34 commits, the 32 of the agents landing
as 31, since the guard of W3-2 for a history that holds no GP was squashed
into the commit of its finding, and three made by the orchestrator
(`74414ce`, `894a353`, `83ea3d8`).

| row | commit | |
|---|---|---|
| W3-1 | `095c82c` | the noise function at uncertainty level 1 |
| W3-2 | `14a01e2` | the window of past GPs, with the guard for a history that holds none |
| W3-3 | `6e135f2` | `floor` in the subsample count |
| W3-4 | `1268d69` | `gp_hyp_full` records the chain before thinning |
| W3-5 | `720c416`, `894a353` | the lower bound of `mean_const`; the second parameter of the output-dependent noise, in its dead branch |
| W3-6 | `61a7325` | the guard of the warp's thresholded covariance |
| W3-7 | `992f7cb` | the upper bound of the noise |
| W3-8 | `de7a99b` | `upper_gp_length_factor` |
| W3-9 | `f954c63`, `bf081c2` | the key of the reset; the guard and the chain of a fit that does not sample |
| W3-10 | `dd41323` | `np.minimum` |
| W3-12 | `d9e3cc9` | `gp_mean_fun` at construction |
| W3-13 | `bee1866`, `894a353` | `gp_hyp_sampler` at construction and the branches removed; the `npv` block of `train_gp` |
| W3-14 | `3b8b26c` | the floor of the design schedule |
| W3-15 | `175629a` | `iteration > 0` |
| W3-16 | `0380c5d` | the stable descending order |
| W3-19 | `9ebaa48` | the warp rewrites every filled row |
| W3-21 | `490ea14` | the evaluation time of a repeat |
| W3-22 | `fb02de0` | the returned value is a float |
| W3-23 | `7b8cf6a` | a half-bounded variable is refused |
| W3-24 | `7c2f9b6`, `a385d74` | copies of the arrays; the keyword in the docstring |
| W3-25 | `b74c65c` | the centering before the rotation |
| W3-26 | `dea6b29` | `scale` validated |
| W3-27 | `1a204aa` | `unscent_warp` |
| W3-28 | `fbe25b6`, `e7e27cf`, `74414ce` | `add` at level 2; `f_vals` with `specify_target_noise` at construction; the test that combined them |
| W3-31 | `7f771b0`, `83ea3d8` | the decorator's docstring, and the names of its arguments there |
| W3-32 | `5aa4e89` | the logger's flag and level |
| W3-34 | `490daae` | the mask's comparison, moved into a helper of its own, `_drop_low_correlations`, that is tested directly: a correlation that is not a number needs a variance that is not positive or not finite, and the guard of W3-6 then keeps the covariance, so the rule cannot be observed through `warp_input` |
| test note | `23a69a9` | `test_gp_hyp` trains on the transformed plausible bounds |

Rows W3-11, W3-18, W3-33, W3-35 and W3-36 are sheet entries, W3-30 a line
in one, and W3-17, W3-20 and W3-29 are left as they are; `9c29ce1` holds the
sheet, the list of MATLAB-side defects and the changelog.

## Gates

Before the first cherry-pick, on `fc8e561`: the four seeded runs of
`scripts/wave2_fixpass_gate_runs.py` recorded, and the exact oracle check,
11 of 11. After agent B's batch and again after the whole first phase (every
commit but the five that move default trajectories, W3-2, W3-3, W3-4, W3-5
and W3-7): the four runs bit for bit against the record (92 arrays), the
exact oracle check, 11 of 11, and after the phase the whole
`pyvbmc/testing/vbmc` directory (637 passed, no reruns).

After the second phase the four runs all moved, as they must, and the
default suite failed in the oracles that the fixes move and nowhere else
(1588 passed, 58 skipped, 20 failed).

**The level-1 run** (`scripts/wave3_gate_level1_run.py`, W3-1's check: a
two-dimensional Gaussian target with noise of SD 1, 150 evaluations
allowed). Before the fix the GP had one noise hyperparameter, which settled
at an SD of about 1; ELBO 1.715 with SD 0.121 against a true log evidence of
1.838, posterior SDs 1.02 and 2.09 for 1 and 2, stable after 115
evaluations. After it the GP has two, the constant term falls to an SD of
about 0.02 and the multiplier settles at 1, so the recorded noise carries
the level, as in MATLAB's model; ELBO 1.755 with SD 0.123, posterior SDs
1.02 and 1.98, stable after 100 evaluations. Without repeated observations
every row has `s2 = 1` and the two models have one effective noise level
each, so the defect showed in the prior and with `max_repeated_observations`
above 0, not in a default level-1 run.

**The accuracy of the second phase.** The four seeded runs are a
fingerprint for comparing commits bit for bit and are no measure of
accuracy: their targets have no recorded truth, and the noiseless
Rosenbrock among them is the unscaled function, with a ridge ten times
narrower than that of the benchmark's `rosenbrock` (true log evidence
-1.3947, which PyVBMC misses by about 0.35 before and after the pass). On
that run the final ELBO moved by 0.054 against a reported SD of 0.0006,
which reflects the GP alone. A sweep of ten seeds on the same target
(`scripts/wave3_gate_sharp_rosenbrock_sweep.py`) gives a mean gap to the
truth of 0.339 before the phase and 0.395 after, the paired difference
0.056 with a standard error of 0.043. On benchmark targets with known truth
(`scripts/wave3_gate_benchmark_sweep.py`) the phase changes nothing that ten
and six seeds can show:

| target | code | error of the ELBO, median / mean / max | gsKL, median / mean | MMTV, mean | evaluations, mean |
|---|---|---|---|---|---|
| `rosenbrock_D2`, 10 seeds | before | 0.021 / 0.023 / 0.047 | 0.016 / 0.018 | 0.022 | 85.5 |
| | after | 0.018 / 0.025 / 0.078 | 0.014 / 0.025 | 0.025 | 87.0 |
| `cigar_D4`, 6 seeds | before | 0.006 / 0.009 / 0.020 | 0.0003 / 0.0024 | 0.013 | 126.7 |
| | after | 0.006 / 0.012 / 0.028 | 0.0004 / 0.0039 | 0.016 | 130.0 |

At the PI's request the same sweep was run on six more targets, a few
seeds each, against the code before the whole pass (`fc8e561`) where the
table above has the code before its second phase; the means:

| target | seeds | error of the ELBO, before / after | gsKL, before / after | evaluations, before / after |
|---|---|---|---|---|
| `banana_D2` | 3 | 0.048 / 0.043 | 0.186 / 0.129 | 85 / 83 |
| `halfnormal_D2` | 3 | 0.002 / 0.004 | 0.0002 / 0.0003 | 67 / 67 |
| `lumpy_D4` | 2 | 0.042 / 0.021 | 0.009 / 0.021 | 93 / 95 |
| `student_D4` | 2 | 0.023 / 0.038 | 0.017 / 0.043 | 95 / 110 |
| `corr_D5` | 2 | 0.031 / 0.003 | 0.009 / 0.0003 | 105 / 98 |
| `rosenbrock_D2_noise1` | 4 | 0.101 / 0.114 | 0.011 / 0.024 | 129 / 118 |

Every run is a good solution and the differences go both ways; on the
noisy target every error of the ELBO lies within two of its reported SDs,
about 0.11, and two of the four seeds are better after the pass. Eight
targets and 32 seeded pairs show no regression. The final release
benchmark, which regenerates the run pools, is the measure of the pass.

**The oracles that moved**, three and no other: `gp_fit` and
`gp_fit_history`, which rerun the hyperparameter fit, and the log prior of
`gp_nlZ`. W3-5 and W3-7 change the bounds that `_gp_hyp` installs, and the
fit draws its space-filling design inside them, so the design, the point
the chain starts from and the samples change; the default widths of the
slice sampler, the standard deviation of the design, change with it, which
W3-5 reaches as well as the starting points. W3-2 and W3-3 change the pool
of past samples of a fit with a populated history. The log prior moves by
the change of the log normalization of the noise prior, truncated at its
new upper bound, `log(1 - cdf_lb) - log(cdf_ub - cdf_lb)`, between 3e-4 and
5e-4, which was checked on the eight states; its gradient, the marginal
likelihood and its gradient are bit-identical. Of the three authentic
captures of the fit, the two whose fits draw no design (`init_N = 0`: the
reliability index was below the retraining threshold, so neither the bounds
nor the window enter) replay bit for bit, one of them with an even history,
and the noisy one, which draws 814 design points, moved
(`scripts/wave3_gate_probe_captures.py`). The three oracles were
re-baselined from the stored states with the generator's targeted mode
(`0ccaf76`), and the capture by a mode added for it, which keeps the
captured inputs and the portable references bit-identical (`c60834d`,
`2828fd3`); every fixture records the reason (PI, 2026-09-20). On
`2828fd3`: the exact oracle check, 11 of 11, and the whole default suite
(1608 passed, 58 skipped, no reruns).

**CI.** The branch smoke (Ubuntu, Python 3.12, with Torch) failed twice, each
time on one test, both described below: an S-VBMC test that the
orchestrator's default environment skips, and a test of the wave-2 pass
whose precondition rested on the course of a seeded run. The third smoke and
the full matrix, nine cells, are green on `92eb2cc`.

## Found during the fix pass

- `test_vectorized_initial_design_matches_scalar` gave both of its arms a
  cached value through `f_vals`, the arm whose target provides its noise
  included, which construction refuses since W3-28; the cached value stays
  in the noiseless arm (`74414ce`).
- The second parameter of the output-dependent noise had infinite bounds
  where MATLAB leaves them unset, the defect of W3-5 and W3-7 in the branch
  that nothing reaches; found by agent A (`894a353`).
- The window of W3-2 as MATLAB writes it reads the last entry of an empty
  array when the history holds no GP, which MATLAB's `~isempty(stats)`
  prevents and which the stand-in history of the `gp_fit` oracle produces;
  found by agent A and part of `14a01e2`.
- The docstring of `handle_0D_1D_input` named its arguments `kwarg` and
  `argpos`; found by agent C (`83ea3d8`). The description of `f_vals` says
  what it cannot carry (same commit).
- `pyvbmc/testing/whitening/` has no `__init__.py`, as `AGENTS.md` records,
  and `test_gp_hyp`, `test_rotoscaling_rotation_2d`,
  `test_bounded_log_abs_det_jacobian_numerically` and
  `test_transform_bounded_and_unbounded` draw from the unseeded global
  generator. Not acted on.
- The branch smoke failed on
  `test_svbmc_filters.py::test_infinite_bounds_have_to_agree_too`, which
  built a run bounded on one side in one variable, what W3-23 refuses, to
  show that S-VBMC rejects runs whose infinite bounds disagree. The test
  needs Torch, which the orchestrator's default environment lacks, so the
  local suite had skipped it. The run of the test is bounded on both
  sides in that variable, which shows the same, and the tests that need
  Torch or PyMC were run in their environments: 739 passed and 107
  passed. The plan's gate section lists them among the gates of a pass.
- The second smoke failed on
  `test_vbmc_warp_branch.py::test_warp_refit_sieve_is_sized_at_the_new_component_count`,
  a test of the wave-2 pass, with `assert 20 != 20`. The test drives a
  short seeded run and needs the posterior that enters the warp to have
  fewer components than the training set, which held only if the
  iteration before had pruned one; after the second phase the run on
  Ubuntu pruned none, while the one on Windows still did. The fixture's
  options bring about what the checks rely on: every component lighter
  than a fifth is pruned, and the warp needs no minimum of components and
  no reliability threshold (`92eb2cc`).
- `scripts/wave3_P8_logger.py`, the verifier's evidence for rows P8-10 and
  P8-14, builds loggers whose noise flag contradicts their level and adds
  values without an SD at level 2. The fixes of W3-32 and W3-28 refuse both,
  so the script shows the code as it was verified: on the fixed package it
  stops at its first `add` without an SD. Noted by agent C.
- The commits of the fix agents carried a `Claude-Session:` trailer, which
  the PI does not want; it was removed from the unpushed commits of the
  pass, and `AGENTS.md` says so for every later session.

## The independent check of the pass

Five fresh Opus reviewers, read-only, checked the pass on 2026-09-21 without
the context of the session that made it: the inputs of the hyperparameter
fit; its policy, the recorded chain and the option checks; slice P8; this
ledger, the user-facing records and the re-baselined fixtures; and the
consequences of the pass outside the lines it changed. What a reviewer had
rated uncertain was verified before anything was acted on. They found no
defect in what a run computes. Against a transcription of `vbmc_gphyp`,
every bound, prior and starting value that `_gp_hyp` installs is MATLAB's at
the three uncertainty levels, the lower bounds of W3-11 apart. A GP of
level 1 passes through `train_gp`, the warp, `reupdate_gp`, the lean record
and its restoration, the noise estimate, the candidate noise of active
sampling and the expected log joint, every reader taking the layout of the
hyperparameters from the model's own counts. A comparison of the fixtures
key by key, before the pass and after it, has `gp_fit`, `gp_fit_history`,
the log prior of `gp_nlZ` and the fit outputs of the one capture moved, and
every other array bit-identical.

In the code they found what ten commits follow up, each fix with a test
that fails on the code before it:

| commit | |
|---|---|
| `2ff2dfe` | the trim at the end of warm-up keeps the earlier of equal values (`private/vbmc_warmup.m:123`), the third site of W3-16 |
| `06bc8d5` | `determine_best_vp` ranks the earlier of two iterations with equal scores first (`misc/best_vbmc.m:36`, `:40`) |
| `027e972` | `get_hpd` orders integer values by value: the negation that W3-16 introduced wraps around for an unsigned zero and for the smallest value of a signed type |
| `9b5213d` | `load` checks the option values as construction does; `noise_shaping`, `gp_hyp_sampler` and `search_acq_fcn` were checked at construction alone, which agent B had left open |
| `a775bb6` | `batch_call` refuses a cached value at uncertainty level 2 before the target is called; since W3-28 it raised after recording the rows before it |
| `0dfc282` | a known evaluation time takes the place of an unknown stored average, the other direction of W3-21; sheet entry |
| `68f7984` | an `f_vals` of NaN alone, which supplies no value, passes the check of W3-28 |
| `b00fe40` | the bound statements of `_gp_hyp` carry the other half of their pair over, as their comment says; the test of the mean constant pins the values MATLAB's recommendation fills in |
| `791c51d` | the test of W3-19 runs on a bounded problem as well, where every row has its own log-Jacobian; `build_short` is defined once |
| `095c29e` | the description of `gp_mean_fun` names the three values; the comment on the noise switches of an oracle state |

In the records they found, and this check corrected:

- `CHANGELOG.md` had no entry for the four refusals of `ParameterTransformer`
  and `FunctionLogger` that its "Upgrading" list names; said of the level-1
  noise model that the difference shows with repeated observations, where
  every run at that level changes from its first fit; had no "Upgrading"
  line for `gp_hyp_full` and for three changed return values; and put the
  zero of the design schedule at 1000 evaluations for about 1380.
- Entry 18 of `matlab_side_defects.md`, after the P5 verifier's report,
  spoke of commented-out lines that `misc/gptrain_vbmc.m` does not have.
  The entry is corrected and the report carries a flag under its header.
- This ledger counted seven more targets, nine targets and 35 seeded pairs
  for the six, eight and 32 of its own tables, which the logs of the sweeps
  confirm, and the worklog 36 commits for 34; it listed five of the nine new
  sheet entries, and said nothing of the helper of W3-34 or of the evidence
  script that the fixes stopped.
- The capture re-baseline mode of the oracle generator was missing from
  `dev/README.md` and from the fixture plan.
- `AGENTS.md` gave one cause for the three moved oracles, and recorded
  neither the second noise hyperparameter of level 1 nor what `gp_hyp_full`
  holds.
- 83 of the 161 Python line citations of the known-differences sheet pointed
  at other lines. Each was carried to its present line through the history
  of the cited file, from the commit that last touched the citation, two of
  them by hand; where an entry names a symbol beside its citation, the
  symbol is at the cited line or around it. The sheet's entry on
  `noise_shaping` said that `load` does not apply the check, which `9b5213d`
  made false.

One statement of a reviewer did not hold: the repeat pooled into a
deactivated row (W3-20) gets no new route through the warp. The same input
still has to be proposed again, which needs the search cache; the rewrite of
the inactive rows only stops coordinates of an earlier space from hiding the
match.

Rulings (PI, 2026-09-21): the option checks run in `load`; the stable order
goes into the trim and into both rankings of `determine_best_vp`; a stored
oracle state at uncertainty level 1, which no fixture holds, is an item of
`TODO.md` for the time after the review's remaining fixes.

Left as they are:

- The test of W3-6 for a positive definite matrix is a Cholesky
  factorization without a tolerance, as the ruling words it.
- `hyp_dict["logp"]` is `None` after a fit that draws no samples, where
  MATLAB keeps a scalar that nothing reads.
- `_gp_hyp` takes the minimum with the cap before it rounds the number of
  samples, where `misc/gptrain_vbmc.m:321-327` rounds first; the two agree
  at the default caps.
- A file saved before the pass keeps an inert `runcov` key. A level-1 run
  saved before the pass continues with the new noise model: its recorded
  blocks, one hyperparameter short, are left out of the sampler widths
  without a message.
- The short run of `test_ending_warmup_clears_the_covariance_the_fit_reads`
  stays, the reset being visible only through the loop. The new whitening
  tests build their posterior without a generator of their own, and their
  assertions do not depend on the draws. The stand-in histories of the
  `gp_fit` oracles hold thinned blocks.

Gates on `095c29e`, the code of the check: the exact oracle check, 11 of 11
with nothing re-baselined; the four seeded runs bit for bit those recorded
on the wave-4 pass (92 arrays); the default suite, 1699 passed and 58
skipped, with one rerun, of `test_minimize_adam_matyas_with_noise`; the
Torch environment, 785 passed and 19 skipped; the PyMC
environment, 107 passed.

The rerun led to one more commit (PI: fix it before the push). The test drew
its gradient noise from NumPy's global state and failed 48 of 400 calls made
with that state seeded 0 to 399, about one run of the suite in eight, which
the reruns hid. The cause is its target: Matyas is `0.5 u^2 + 0.02 v^2`
across and along the line `x[0] = x[1]`, noise of SD 3 swamps the slope
along that valley, and the iterates random-walk there up to 2.8 while the
position across it stays within 0.13; the test asked for `|x| < 1` in both
directions. It draws from a generator of its own and bounds each direction,
with a margin that holds on every one of 500 seeds, 200 of them not used to
set the bounds (`b731fac`); the sphere test beside it, which never failed,
gets its own generator as well.
