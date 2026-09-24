# Wave 6 verification

The findings of wave 6 (slices G1 and G2, the two gpyreg slices, both tracks;
reports under `../reviews/`), verified on 2026-09-21 against gpyreg at
`fdbafdf`, whose code under `gpyreg/` is that of the pinned revision
`9e70e6b` (release 1.2.1), PyVBMC `dev-port-review` at `8c6e72eb` (package
code of `8af5daac`) and `gplite/` of MATLAB VBMC at `396d649`. Part 1 holds
the findings that can change what a run computes; part 2 holds the others.
Two sources feed the ledger: the orchestrator's own checks, for part 1, with
the scripts named in the rows; and two read-only Opus verifiers, one per
slice, whose raw reports are `wave6_G1.md` (rows G1-n, and G1-X for the
candidate from outside the slices) and `wave6_G2.md` (rows G2-n). The scripts
are under `scripts/` (`wave6_A*.py` and `wave6_per_column_patch.py` the
orchestrator's, `wave6_G1_*.py` and `wave6_G2_*.py` the verifiers', the
latter with the import of their transcription module renamed to the file as
it is kept); the logs of the orchestrator's are kept on the machine that ran
them (`dev/scripts/runs/LOCAL.md`, "Port correctness review"). No MATLAB was
run: a statement about MATLAB is read from the source at `396d649` or
measured on a Python transcription of it, and the verifiers' reports say
which.

gpyreg's history starts on 2021-05-24, and of the MATLAB files of the two
slices four changed after that: `fminfill.m` (`46b6f5e`, `1d1f20d`),
`slicesamplebnd.m` (`fcf2674`) and `gplite_pred.m` (`68a197b`). gpyreg has
all four (the first questions, below), so no row of this ledger is a
faithful port of an older MATLAB.

The column "proposed" holds the orchestrator's proposal and the column "PI"
the PI's ruling, all of 2026-09-22: on the five rows where the proposal left
alternatives (W6-1, W6-4, W6-8, W6-19, W6-21) the PI took the orchestrator's
recommendation of that day, and the others by taking the proposals as they
stood ("as proposed"). A fix to gpyreg goes to gpyreg on a branch and a pull
request of its own ("Fixes and gates" of the plan); "fix" in a row of part 2
means that, with a test seen to fail on the code before it. No row of part 2
changes a number of a PyVBMC run at any shipped option, so no fix of part 2
can move a PyVBMC gate.

## Part 1: findings that can change what a run computes

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W6-1 | G2 internal F6, G2 comparison F3 and F4, independently; verifier row G2-7 | gpyreg's bound recommendations take the statistics of the training inputs over all `N * D` entries of `X` where gplite takes them per column: the starting value of every length scale, `np.log(np.std(X, ddof=1))` (`covariance_functions.py:451`, and `:401` for the rational quadratic kernel) against `log(std(X))` (`gplite_covfun.m:126`), although `width`, nine lines above, has `axis=0`; and in the negative quadratic mean `w`, `min`, `max`, `median` and `std` (`mean_functions.py:488`, `:511-524`) against `gplite_meanfun.m:142`, `:220-230`, which gives the location and the scale of the mean one hard box, one plausible box and one starting value for all dimensions. The pooled statistics mix the locations of the columns with their spreads | confirmed port discrepancy, unintended; never matched. The pooled forms are in the first revision of the bound code (`16c441d`, 2021-05-31), no revision of gpyreg ever had the axis (the G2 verifier searched every revision of the three files), and the MATLAB lines date from 2019-06-17 and 2020-05-08. No commit, pull request or release note speaks of it; the comment at `mean_functions.py:522-523`, which the sheet's entry rests on, is about the normalization, and `ddof=1` is MATLAB's already | yes, every run. The two reports disagree on the reach and each has half (G2 verifier, by reading and by instrumenting `GP.fit`): `x0` enters `hyp_dict["hyp"]` while that is `None` (`gaussian_process_train.py:117-118`), the first GP training of a run, as `gptrain_vbmc.m:28` has it; `_gp_hyp` sets no bound of the mean's location and scale (`:371-416`), so `GP.fit` fills their hard bounds, and the plausible bounds of every hyperparameter, from its own recommendation on the whole training set at every fit (`gaussian_process.py:1157-1196`). The hard bounds confine the optimizer and the slice sampler; the plausible bounds shape the space-filling design and, with `init_N == 0`, are the default sampler widths | `scripts/wave6_A1_pooled_recommendations.py`, on the eight stored oracle states. In the two cigar states the transformed coordinates lie at about 0.9, 10.5, -48 and 52 with SDs of 0.5 to 2.5, and the pooled SD is 35.7: the starting length scales are 15 to 74 times gplite's, the mean's location starts at 8.2 in every dimension, and its hard box is `[-107, 113]` in every dimension where gplite's are `[-16, 17]`, `[7.6, 14.2]`, `[-55, -44]`, `[47, 62]`; in the other states the starting length scales differ by factors of 1.1 to 2.6. gpyreg's box always contains gplite's, and the stored fits use the room: samples with a mean location outside gplite's hard box, 5 of 7 (cigar), 6 of 8 (warped), 4 of 8 (noisy Rosenbrock), 1 of 10 (half-normal), none in the three states of the two-dimensional normal. `scripts/wave6_A1b_gp_fit_per_column.py`: the `gp_fit` oracle, whose start is the stored `hyp_dict` and not `x0`, reproduces its stored reference bit for bit in the eight states with gpyreg as it is, and with the statistics per column (`scripts/wave6_per_column_patch.py`, eight lines of the two helpers, in the process alone) gives other samples in the seven states that sample, by about the spread of the samples (cigar: the SD of the mean's location in the long dimension 15.5 for 8.6); the state that does not sample is identical. `scripts/wave6_A1c_gate_runs_per_column.py`: the four seeded runs of the gates differ from the record `wave5_gates/after_pass_23d962a1.npz` from the first evaluation after the initial design, the eleventh, in all four (82 of 92 arrays): 25 iterations for 21, 17 for 19, 32 for 30, 17 and 17; final ELBO -1.7959 for -1.7903, 1.2272 for 1.2315, -1.445 +- 0.101 for -1.635 +- 0.114, 1.356 +- 0.080 for 1.430 +- 0.081. The runs are a fingerprint and say nothing of accuracy | fix in gpyreg: `axis=0` in the statistics of `X` of the two helpers (the eight lines of the patch) and in the copy of the rational quadratic kernel, last and alone on the gpyreg branch, with a test of each recommendation on inputs whose columns differ in location and in spread. It moves every sampled GP fit of every run, so it is read for accuracy on the targets of `benchmark_targets.py` over several seeds before and after (`wave3_gate_benchmark_sweep.py`) before it is taken, and the oracles that run a fit (`gp_fit`, `gp_fit_history`, `active_sample_step`) are re-baselined on purpose with it, the reason recorded in the fixtures; the one regeneration of the golden references waits for it. The sheet's entry is replaced (below). The alternative, to keep the pooled form and record it as deliberate, has nothing in the record behind it | fix, as proposed; the benchmark sweep before and after is the gate that decides whether it is taken. Taken on the sweep, reopened when the independent check of the pass corrected the orchestrator's reading of it, and kept after the sweep was extended to ten seeds on four targets and after the probe of the mean's bounds (PI, 2026-09-22; "Gates") |
| W6-2 | G1 comparison F1 | The adapted widths of the slice sampler come from the sums of the second half of the burn-in. `slicesamplebnd.m:362` accumulates over `ii > burn/2` (`ii` from 1) and divides by `floor(burn/2)` (`:367-368`): with an odd burn-in that is one term more than the divisor, so `xx_sqsum/k - (xx_sum/k).^2` is no variance, and where it is negative the square root is complex and `:371` keeps the old widths of every coordinate. `slice_sample.py:562` accumulates `floor(burn/2)` terms for the same divisor | confirmed MATLAB-side defect; gpyreg is the consistent side, so an intentional difference in effect, missing from the sheet. The MATLAB lines predate gpyreg's history; `fcf2674` touched the bracket alone | yes, where the burn-in is odd: `thin * 3 = 15` in every fit that does not recompute the variational posterior, the burn-in of every stored oracle state, and `thin * gp_s_N` for an odd number of samples (`gaussian_process_train.py:607`, `:614`), as `get_GPTrainOptions.m:108` has it | `scripts/wave6_A2_burn_in_window.py`. Part 1 counts the terms: equal for an even burn-in, one more in MATLAB for an odd one (8 for 7 at 15). Part 2 installs MATLAB's window and its answer to a negative estimate by an edit of the source of `SliceSampler.sample`, in the process alone, and runs the `gp_fit` oracle on the stored states: at a burn-in of 16 the two give the same samples bit for bit in the seven states that sample, so the edit changes nothing else; at 15 MATLAB's estimate is negative in 5 to 11 of the 9 to 18 free coordinates in every one of the seven, so MATLAB keeps the widths of the shrinking phase, 1 to 74 times gpyreg's adapted ones, and the samples differ. Both are valid slice samplers: the widths change the chain, not the law it draws from | no fix. A sheet entry, and a line of `matlab_side_defects.md`. gpyreg's test of the sampler gets a case with an odd burn-in that pins the window (test note) | as proposed |
| W6-3 | G1 comparison M6; the closing note of the G1 verifier | gpyreg floors the variance estimate of W6-2 at zero (`slice_sample.py:574`), and a width of zero stops a coordinate for the rest of the chain: the bracket has no extent and the point is accepted where it stands. With a burn-in of 2 or 3 the window holds one iteration, the estimate is exactly zero in every coordinate, and the chain returns one point as many times as it is asked for samples, with no message | Python-only defect. MATLAB has no floor (a negative estimate discards the adaptation, W6-2), and at a burn-in of 3 its window holds two iterations | not at the shipped options: no estimate is near zero in the stored states (smallest adapted width 0.018), and the smallest burn-in of a run is `thin * 1 = 5`. A user's `gp_sample_thin = 1` makes the burn-in 3 in every fit that does not recompute the variational posterior (`gaussian_process_train.py:614`), and every hyperparameter sample of those fits the same point | `scripts/wave6_A2_burn_in_window.py`, part 3: a standard normal in two dimensions, burn-in of 2 and of 3, widths `[0, 0]` and 1 distinct sample of 6; of 4, 5 and 15, 6 of 6. The reach through `gp_sample_thin` is read, not run | fix in gpyreg: a coordinate whose estimate is not positive keeps the width it has, and a window of fewer than two iterations adapts nothing; a test at a burn-in of 3. Moves no run at the shipped options | as proposed |
| W6-4 | G1 internal F9; the G1 verifier's closing note, the G2 verifier's row G2-20 | The recommended bounds of the output scale and the upper bound of the noise come from the range of the targets, `height = max(y) - min(y)` (`covariance_functions.py:453-454`, `noise_functions.py:131`). With equal targets they are `-inf`, `ub = np.maximum(lb, ub)` (`gaussian_process.py:428`) leaves the pair `(-inf, -inf)`, and L-BFGS-B answers it with `KeyError: (-inf, -inf)` from `scipy/optimize/_lbfgsb_py.py:388` | the pair is shared, `gplite_covfun.m:130-131`, `gplite_noisefun.m:104` and `gplite_train.m:142` being the same formulas; what `fmincon` does with it was not looked up. That nothing on the way to the optimizer refuses or repairs the pair is Python's to decide | not on a problem with hard bounds, where the log Jacobian of the transform makes the training targets differ. Without hard bounds a log joint that is constant over the ten points of the initial design is enough, a plateau or a clipped floor of the likelihood for one: the run stops at its first GP training (`gaussian_process_train.py:170`) | `scripts/wave6_A3_constant_target.py`: gpyreg alone, equal targets, the `KeyError`; two values, the fit completes. PyVBMC at 25 evaluations: with hard bounds both a constant log joint and one that is constant over the initial design alone complete; without hard bounds both raise the `KeyError` | for the PI to weigh: gpyreg refuses a training set of equal targets in `fit` with a message that says so (the smaller change, and no departure from MATLAB), or takes a range of one for it, as both sides already do for a single target (`if np.size(y) <= 1`), so that such a run goes on. Either leaves every run that works today as it is | gpyreg takes a range of one for equal targets, as both sides do for a single target, with a warning; the run goes on |

## Part 2: the other findings

Rows G1-n are in `wave6_G1.md` and rows G2-n in `wave6_G2.md`, which hold the
supporting detail, the dating of each row, the search for a recorded reason
and the errors found in the reviewers' reports. No row of this part changes
a number of a PyVBMC run at any shipped option; where a row is reached at
all, by a run or by gpyreg's own defaults, the column says with what effect.

### Slice G1

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W6-5 | G1 internal F1, G1 comparison F3, independently | The masks of the hyperprior families have `df == 0 \| ~np.isfinite(df)` (`gaussian_process.py:1441`, `:1457`), which Python reads as `df == (0 \| ~np.isfinite(df))`: a prior with `df = inf` or NaN falls out of every mask and contributes nothing, while `__recompute_normalization_constants`, which has the test right (`:1398`, `:1405`), still subtracts its truncation constant. `gplite_hypprior.m:36` is `(df == 0 \| ~isfinite(df))`, and `gplite_nlZ.m:12` documents `nu = Inf` for a Gaussian prior. The slip of W5-4 | confirmed port discrepancy (G1-1); never matched (`16c441d`, 2021-05-31). The comments of `set_priors` ("Implicit flag for gaussian, is set to inf later") name a conversion that no line makes | no: `_gp_hyp` sets `student_t` priors with `df = 3`, `set_priors` writes a Gaussian as `df = 0`, and `fit` fills a NaN `df` with `df_base`. A caller who writes `hyper_priors["df"]` | G1-1: a Gaussian prior on `mean_const` gives `log_posterior - log_likelihood = -0.9189`, the whole Gaussian term, at `df = 0` and exactly 0 at `df = inf` | fix (the parentheses), with a test of the log prior against an independent density for each documented combination of `mu`, `sigma`, `df`, `a`, `b` | as proposed |
| W6-6 | G1 internal F2; G1 comparison M4 | The normalization constant of a smooth-box prior has one entry per smooth-box coordinate and multiplies the `sigma` of the coordinates on one side of the box (`:1479`, `:1483`, used at `:1541-1551`, `:1579-1601`): with two such coordinates on different sides the log prior is twice its value, with three or more the evaluation raises. `get_priors` raises on the same priors (`:466`, an `or` over an array where `:473` has `np.all`) | Python-only defects (G1-2, G1-32); the families have no MATLAB counterpart. The constants moved into the mask cache unsubscripted with `a2f8ddc` (2026-09-05). Two defects in one block: no caller has set such a prior | no: a user's smooth-box prior over a block of more than one hyperparameter | G1-2: `D = 2`, one coordinate below `a` and one on the plateau, -13.455 for -6.728; `D = 3` raises. G1-32 | fix both, with the smooth box of the test fixture on the length scales | as proposed |
| W6-7 | G1 internal F3, G1 comparison F2, G2 internal F3, G2 comparison F5: all four reviewers | The eigenvalue fallback of `__robust_cholesky` fixes the signs of the eigenvectors with `negidx = U[maxidx] < 0; U[negidx] *= -1` (`:2620-2622`), which indexes rows and flips single entries, where `gplite_rnd.m:97-99` flips columns: `T'T` is not the matrix. Beside it (both verifiers), `sp.linalg.eig` on a symmetric matrix need not return an orthogonal basis of a repeated eigenvalue and returns complex vectors that `np.real` cuts off, where MATLAB's `eig` takes the symmetric solver | confirmed port discrepancy (G1-3, G1-3b, G2-3); never matched (`009007b`, 2021-07-12) | no: `random_function` alone calls it. A gpyreg user who draws functions on a one-dimensional grid reaches it as a rule: the direct factorization failed in 96 of 96 cells of the G2 verifier's sweep at 100 points in one dimension and in none in two and three dimensions | G1-3: `max\|T'T - C\|` of 2.8 for a matrix of size 2.7, 9e-16 for a transcription of `robustchol`, and 0.42 from `eig` alone before any flip. G2-3: two identical test points drawn with correlation -0.08 for 1 | fix: the flip per column, or none, and `eigh`; a test of the static method on matrices that the direct factorization refuses | as proposed |
| W6-8 | G2 comparison F6; G2 internal F3, in passing | With a negative eigenvalue left above the tolerance `__robust_cholesky` returns a factor of zeros of the full shape (`:2635`), so `random_function` returns the predictive mean, with no message and `n` normals drawn; `gplite_rnd.m:110` returns an empty factor and `:61` then fails on the sizes (inferred). The tolerance, MATLAB's own, lets eigenvalues of rounding size through | the trigger is a confirmed shared defect, the silence Python-only (G2-4) | no. In the sweep: one dimension, 40 or more training points and a noise SD of 1e-3, 21 of 144 cells | G2-4: 35 eigenvalues above the tolerance, 5 of them negative, the smallest -3.5e-15; two draws from different generators equal and equal to the mean | for the PI: (a) raise where no factor can be built, which changes no draw that works; and (b), a departure from MATLAB, take an eigenvalue of rounding size for zero, so that a draw on a dense grid is a draw. (a) at the least | both: raise where no factor can be built, and an eigenvalue of rounding size counts as zero, with a relative tolerance and a raise beyond it |
| W6-9 | G1 internal F4 and M-I; G1 comparison M3 and M14 | The rank-one update tests `sqrt_arg <= 0` in the `L_chol` branch, warns and recomputes (`:885-899`); the low-noise branch divides by `v_star` with no test (`:920-928`), and `v_star` comes from a `predict` that clamps the latent variance at zero (`:2023`) | confirmed shared defect (G1-4, G1-25): `gplite_post.m:233-237` has no guard in either branch and `gplite_pred.m:120` the same clamp, so the guard of the other branch is gpyreg's own improvement; the internal report takes both for gpyreg's | no: the noise variance of a run is at least 1e-5 (`tol_gp_noise`), against the threshold 1e-6 of the representation. gpyreg alone recommends `log(1e-6)` as the lower bound of the noise | G1-4: a noise of `eps` and an exact copy of a training input, `alpha` off by 1.6e-2 to 5.2e5 relative as the output scale grows, no warning | fix: the low-noise branch recomputes where the variance is at or below what the clamp can produce, as the other branch does; tests of that branch, which no test enters | as proposed |
| W6-10 | G1 internal F5 | `fit` reads `options["sampler"]` (`:1126`) and documents `sampler_name` (`:1073-1075`); the documented name is ignored. The `"laplace"` clause at `:1221` has no effect, the value raising at `:1345` | Python-only defect (G1-5); `009007b` | no: PyVBMC writes the key that is read | G1-5 | fix: both names read, the documented one first; the clause goes | as proposed |
| W6-11 | G1 internal F6 | `log_likelihood` and `log_posterior` take the dictionary their docstrings offer through `hyperparameters_from_dict`, whose `(1, hyp_N)` result is then sliced along the first axis, and raise (`:1674-1675`, `:1706-1707`) | Python-only defect (G1-6); `b5e3ac4`, 2021-06-24 | no: PyVBMC calls neither | G1-6 | fix (the row of the result), with a test of the dictionary form | as proposed |
| W6-12 | G1 internal F7, G1 comparison F5, independently | `hyp = X0[0 : max(opts_N, 1), :]` is a view, `hyp[1, :] = xx[idx_best, :]` writes the low-noise start into row 1 of the design, and `widths_default = np.std(X0, ...)`, three lines on, is taken from the design so altered (`:1243`, `:1259`, `:1262`); `gplite_train.m:206-207` copies, and takes the widths before the write | confirmed port discrepancy (G1-7); never matched (`dbc91d6`, 2021-06-24) | not in PyVBMC (`opts_N` is 0 or 1 where samples are drawn, and with 2 `fit` returns before the widths are read); yes at gpyreg's own defaults (`opts_N = 3`, `init_N = 1024`, 10 samples) | G1-7: `hyp.base is X0`; the widths differ by 1.3e-1 relative at `init_N = 8`, 7.7e-3 at 256 | fix (`.copy()`). It moves the sampler widths of every fit at gpyreg's defaults: gpyreg's suite is the gate, no PyVBMC gate can move | as proposed |
| W6-13 | G1 internal F8, G1 comparison M13, independently | `update` takes the rank-one path from the shapes of its arguments (`:815-824`) without asking whether the posteriors hold their factors, and raises `TypeError` after `clean()`, whose docstring names `update` as the way back, and after `update(..., compute_posterior=False)` | confirmed port discrepancy (G1-8): `gplite_post` takes `update1` from its caller | no: PyVBMC rebuilds through `update(hyp=...)`, the full path | G1-8, both states | fix: the rank-one path asks for factors and falls through without them | as proposed |
| W6-14 | G1 internal F10 | `SliceSampler` copies `base_widths` before it replaces an infinite width by 10 and checks the widths after (`slice_sample.py:186`, `:188`, `:213-220`), so the infinity returns after the burn-in (`:590-592`) and the chain is NaN | confirmed shared defect (G1-9): `slicesamplebnd.m:168`, `:176`, `:198`, `:377` in the same order; the internal report has it as Python's | no: `fit` and the importance sampling pass finite widths | G1-9: 20 of 40 recorded entries not finite, and a normal return | fix: the copy after the replacement, as the docstring promises | as proposed |
| W6-15 | G1 internal F11 and M-J | A free parameter that never moved is recognized by `R` and `eff_N` not being finite, which holds only for a constant whose mean is exact: 0.3 gives `R = 0.9487`, `eff_N = 1.05` and `exit_flag = -1` where the docstring promises -3 (`:683-685`). The autocorrelations are paired from lag 0 where the references the docstring names pair from lag 1 (`:926-930`) | Python-only defects (G1-10, G1-26); both from `cec1f85` (release 1.2.1). MATLAB's `psrf` is in no file of the repository. The split of an odd number of samples, which the report lists with the pairing, is MATLAB's (`slicesamplebnd.m:506`) | no: `fit` and PyVBMC pass `diagnostics: False` | G1-10: the constants 0, 1, 0.5, 0.25, 2, -7 give NaN; 0.3 and 1e-3 give 0.9487. The two tests use 0 and 1 | fix the detection (the range of the samples); the pairing from lag 1 or the docstring; the two tests over several constants | as proposed |
| W6-16 | G1 internal F12, M-A, M-C; both reports' answers on `uuinv` | The docstring of `uuinv` names a mixture with equal weight on the two tails; the body gives the tails weight in proportion to their lengths (`f_min_fill.py:193-214`, `:218-250`). The comment "half of all starting points" (`:130-132`) holds for the whole design when no coordinate has a prior, every bound is finite and no coordinate is fixed by equal bounds, each coordinate without a prior putting the weight `w = 0.5 ** (1 / n_vars)` inside its plausible box (a fixed coordinate skips `uuinv` but counts in `n_vars`) (this row said "per coordinate", which the independent check of the pass found wrong, R3-1). The mark for `p` outside `[0, 1]` stands in one of three return paths (`:252-253`) | confirmed shared defects of the wording (G1-11, G1-17, G1-19): `fminfill.m:133-136`, `:77`, `:169` are the same, and the sentence on the mixture is gpyreg's first (`96f1090`, 2022-01-22; MATLAB's `1d1f20d`, 2022-01-30, same author). The body is right, and the same on both sides bit for bit | the wording nowhere; the length weighting it misdescribes in every fit with a design (the output scale gets 5.5% of the design below its plausible box and 1.8% above) | G1-11: the round trip against the distribution function of the body exact to 1e-16, against the docstring's off by up to 0.5; gpyreg against a transcription of `fminfill.m`'s `uuinv`, 0 | fix the three texts; `test_uuinv` then tests the specification as it stands. A line of `matlab_side_defects.md` | as proposed |
| W6-17 | G1 comparison F4 | The mask for no prior is `~isfinite(mu) & ~isfinite(sigma)` (`gaussian_process.py:1453`, and `f_min_fill.py:119`) where `gplite_hypprior.m:35` and `fminfill.m:73` have `\|`: MATLAB's documented flat prior, `sigma = Inf` (`gplite_nlZ.m:13`), is taken for a Student's t and gives a log posterior of `-inf`, or NaN once the bounds are set, and a design column of NaN | intentional difference with a defect left in it, missing from the sheet (G1-12): the `and` is what the smooth-box families need; the two cases it no longer takes are neither handled nor refused | no: every prior of `_gp_hyp` has a finite `mu` and `sigma`. The one way in from PyVBMC, plausible bounds of zero width, is refused at construction (G1 verifier) | G1-12 | stricter interface: `set_priors` refuses a `sigma` that is not finite and positive, with a message that names `None` as the way to no prior (with W6-22); sheet entry | as proposed |
| W6-18 | G1 comparison F6 | With a constant total noise and more than one noise hyperparameter (`scale_user_provided` without `s2`) the gradient loop takes `dsn2[i]`, a row, and raises (`:2812-2819`); `gplite_core.m:244` takes `dsn2(i)`, a linear index, and returns the wrong entry. `dsn2[0, i]` is meant | confirmed shared defect (G1-13) | no: uncertainty level 1 always has `s2` | G1-13: gpyreg raises; the finite-difference gradient is `[-3.996, 0]`, MATLAB's index gives 0.2707 for the 0 | fix, with a gradient test of that configuration; `matlab_side_defects.md` | as proposed |
| W6-19 | G1 comparison F7 and M9 | Neither the design loop of `f_min_fill` nor the optimizer loop of `fit` catches an exception of the objective, where `fminfill.m:104-110`, `gplite_train.m:276-296` and `gp_objfun` do, leaving `Inf`, the starting point and NaN. `np.argsort` is not stable where MATLAB's `sort` is, which shows only among tied values | confirmed port discrepancies (G1-14, G1-36); a NaN value is handled alike on both sides, a raised exception is not | no: nothing raises at the defaults, the noise floor keeping the factorization from failing ten times | G1-14 | for the PI: leave it loud, as it is, with a sheet entry; or take MATLAB's tolerance with a warning and a stable sort. The orchestrator proposes to leave it: a failed factorization that ends a run is seen, one that costs a start in silence is not | leave it loud; a sheet entry |
| W6-20 | G1 comparison, departures of `update` from `gplite_post.m`, and M14 | The sheet's entry on the rank-one update names one difference of state, the kept `sn2_mult`; a second is a new point whose noise falls below 1e-6, where the rank-one path keeps `L_chol = True` and a rebuild has `False` (the numbers agree to 1e-13). Its figure of 1e-15 holds for `L_chol` true, the one representation a run has | the entry is incomplete (G1-15, G1-38) | no | G1-15. The G1 verifier also settles the `sn2_mult` on which the two reports differ: forced to 100, the extension is exact for the stored multiplier (4e-16) and 0.75 in `alpha` from a rebuild that needs none, which is the difference the entry names | the entry amended in the verifier's words | as proposed |
| W6-21 | G1 internal M-B; the G1 verifier's addition | The option of the Metropolis step is read as `"metopolis_rnd"` (`slice_sample.py:238`), so the step never runs; and the flag is an `and` of the two options where `slicesamplebnd.m:189` has an `or` and then asserts both | Python-only defect (G1-18); the step's options never turned it on in gpyreg, while setting its three attributes directly did (one test did so, R3-3) | no: nothing sets either key | G1-18 | for the PI: the spelling corrected and the two keys documented, or the dead step removed. The orchestrator proposes the removal: no caller, no test, no documentation (one test did set the three attributes and ran the step, `test_serialized_sampler_continues_stream` at the pin; found by the independent check of the pass, R3-3, after which the PI kept the removal) | the step and its options removed |
| W6-22 | G1 internal M-D, M-E, M-F, M-H, M-K, M-L, M-M; G1 comparison M7, M11 | Inputs that are not checked, and texts against the code: `set_priors` and `set_bounds` take an unknown name in silence, the first against its docstring; `get_recommended_bounds` raises on a tuple, names the wrong argument in a message and repairs an inverted bound in silence; a negative `sigma` gives a prior from its absolute value and a design from its raw value; `fit` writes `df_base` into the GP's own priors, so a second `fit` with another value changes nothing; `nll` starts at `-inf` where `gplite_train.m:250` has `Inf`; `sp.special.gamma` overflows from a `df` of 343; `update` on a GP without hyperparameters completes with NaN factors (the report has it raising); a `thin` or `burn` that is not whole raises in `range` | Python-only defects, latent (G1-20, 21, 22, 24, 27, 28, 29, 34); the `abs` of `sigma` and `UB = max(LB, UB)` are MATLAB's as well | no | G1-9 of the verifier's scripts (`wave6_G1_9_api_minors.py`), one check each | fix as one commit of input checks and one of texts: an unknown name, a `sigma` that is not positive, a bound pair that is inverted and hyperparameters that are NaN are refused with a message; `df_base` fills a copy; `gammaln`; `+inf`; a count that is not whole is refused | as proposed |
| W6-23 | G1 comparison M2 | `hyp_dict["logp"]` is given `res["log_priors"]` (`gaussian_process_train.py:178`), zeros where no `log_prior` callable is given, where `gptrain_vbmc.m:66` stores the thinned log posterior, `res["f_vals"]` in gpyreg | Python-only defect of PyVBMC (G1-31), without effect: nothing reads the key, and MATLAB's one reader is commented out | every iteration of every run writes it; no reader | G1-31 | remove the key and its comment (it feeds a branch of `gplite_train.m` that is not ported). In PyVBMC, one commit | as proposed |
| W6-24 | G1 comparison M5 and M8 | Without a MATLAB counterpart and without a sheet entry: the two smooth-box hyperprior families; the renormalization of the prior over the bounds (`lp -= masks["log_norm"]`, `:1649`), a constant in the hyperparameters; the prior of a fixed hyperparameter, `-inf` off its value (`:1519-1523`); the default `hyp0` of `fit`, the current hyperparameters or the middle of the plausible box where `gplite_train.m:98` has zeros | intentional differences, missing from the sheet (G1-33, G1-35) | the renormalization in every fit, without effect on the optimizer or the sampler, except where both bounds of a prior lie far in its upper tail, where the stored mass is 0 and the log posterior infinite (`wave7.md`, W7-15) | G1-33: `log_norm = -0.747` in the case measured | sheet entries, in the verifier's words | as proposed |
| W6-25 | G1 internal M-G; G1 comparison M1, M12, and the fourth departure of `update` | `f_min_fill` with no more evaluations than provided points evaluates and returns the first `N` of them; `fit` takes SciPy's `res.x` without MATLAB's clamp one `eps` inside the bounds; `gplite_train.m:298` searches entries of `nll` that no optimization wrote; `gplite_post.m:250` leaves `gp.s2` shorter than `gp.X` after a rank-one update without `s2`, where gpyreg fills with zero | not defects of the port (G1-23, G1-30, G1-37, G1-16): the first is MATLAB's own (`46b6f5e`), the second without effect, the last two MATLAB's and avoided | no | G1-23, G1-30 | none; the last two to `matlab_side_defects.md` | as proposed |

### Slice G2

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W6-26 | G2 internal F1, G2 comparison F1, independently | `predict_full(add_noise=True)` adds `np.dot(np.eye(N_star), sn2_star) * sn2_mult` (`gaussian_process.py:1852`), a column broadcast over the rows when the noise varies by point: every entry of row `i` gains `sn2_star[i]` | Python-only defect (G2-1): `gplite_pred.m` returns variances alone. `3b39954`, 2021-07-02 | no: no production call of `predict_full`; the two test helpers pass neither `s2_star` nor `add_noise` | G2-1: the matrix is not symmetric, and symmetrized it has an eigenvalue of -0.03, so it is no covariance either way; the diagonal is right | fix (`np.diag`), with a test on noise that varies by point | as proposed |
| W6-27 | G2 internal F2, G2 comparison F2, independently | The variance of `quad` derives the scale of the factorization anew, `np.min(sn2) * sn2_mult` (`:2193-2196`), where `Posterior.sl` stores the one the factor was built with; after a rank-one update with a point of lower noise the two differ and the variance is clamped to `eps` | Python-only defect (G2-2), brought in by `cec1f85` (release 1.2.1), which added `Posterior.sl` and the recomputation in one commit | no: no caller of `quad`. A script that calls `quad(compute_var=True)` on the GP of a run reaches it, the run having taken rank-one updates | G2-2: 2.2e-16 for 8.9e-3, the mean right; with the stored `sl` the value agrees with a grid quadrature to 12 digits. gpyreg's `test_rank_one_update_with_heteroskedastic_noise` builds the state already | fix (the stored `sl`), with one `quad(compute_var=True)` in that test | as proposed |
| W6-28 | G2 internal F4, G2 comparison F7, independently | `RationalQuadraticARD.get_bounds_info` writes the plausible upper bound of the shape to index `D`, the output scale's, in a block whose other lines index `-1` (`covariance_functions.py:414`) | Python-only defect (G2-5); `9237c82`, 2022-11-04 | no: PyVBMC has the squared exponential alone | G2-5; `fit` clips the infinite bound that is left to the hard one, against the comparison report | fix, with W6-1's test of the recommendations | as proposed |
| W6-29 | G2 internal F5, G2 comparison F10 | The isotropic bounds take the log of the mean width where `gplite_covfun.m:114-118` has the mean of the log widths, and `np.min`, `np.max` of that scalar; the starting value is the pooled SD of W6-1 | confirmed port discrepancy (G2-6); `20730b5`, 2023-04-12. The four bounds are shifted up by one constant, `log(mean w) - mean(log w)`, not widened (against the comparison report), and MATLAB's branch does not bracket by the narrowest and the widest dimension (against the internal one) | no: the isotropic kernels are gpyreg's own, and MATLAB's identifier has bounds and no kernel | G2-6: a shift of 2.77 on widths that span a factor of eight | fix to MATLAB's means of logs, with W6-1, and the sheet's entry on the isotropic kernels completed | as proposed |
| W6-30 | G2 comparison F8 | The variance of `quad` is normalized by the minimum of the total training noise since `cec1f85`, where `gplite_quad.m:66-67` has the constant noise term alone and clamps to `eps` whenever the noise varies by point | intentional difference, missing from the sheet: gpyreg's repair of a MATLAB-side defect (G2-8), with pull request 49 and the release notes as its record | no | G2-8: 9.4377e-3 from gpyreg and from a grid quadrature, 2.2e-16 from a transcription of MATLAB's formula; equal under constant noise | sheet entry, in the verifier's words; `matlab_side_defects.md` | as proposed |
| W6-31 | G2 comparison F9; both answers to the first question | With `separate_samples=False`, `predict(return_lpd=True)` returns the log density of the moment-matched Gaussian; `gplite_pred.m` never pools `lp` and returns the matrix per sample | intentional difference, missing from the sheet (G2-9); in that form from `9b3d46b` (2022-06-02), and the docstring says so | no: no PyVBMC call passes `return_lpd` | G2-9: 0.09 nats from the mean of MATLAB's columns, 0.18 from the log of the mean density; per sample the two agree to 1e-13 | sheet entry; the docstring names the moment matching | as proposed |
| W6-32 | G2 comparison F11 | The length-scale gradient of the Matern kernel of degree 1 is `inf * 0` on the diagonal (`covariance_functions.py:221`, `:289`, and the isotropic kernel), so the gradient of the marginal likelihood is NaN and the kernel cannot be fitted. `gplite_covfun.m:198`, `:218` are the same, with the repair as a commented-out line below | confirmed shared defect (G2-10). `test_matern_isotropic_against_anisotropic` passes on it with `equal_nan=True`, against the report's note that no test meets it | no | G2-10: 15 NaN on the diagonal on both sides; degrees 3 and 5 exact | fix, with MATLAB's own line: the derivative there is zero. The comment that calls the NaN acceptable goes, and the test asserts finite values; `matlab_side_defects.md` | as proposed |
| W6-33 | the G2 verifier's own | In the low-noise branch of `__core_computation`, `pL` is computed from `L` (`:2761`) before the test `if L is None` (`:2770`), so ten failed factorizations end in `TypeError` and not in the `LinAlgError` that is meant | Python-only defect of a message (G2-11) | no | G2-11 | fix (the order) | as proposed |
| W6-34 | G2 internal M8, M10, M13; G2 comparison M1, M2, M3, M6, M7 | Inputs that are not checked: `quad` on a GP without data, after `clean()`, with a one-dimensional `mu` of length `D`, and with a mean function that is none of gpyreg's three; `_convert_shapes` refuses a NumPy integer and a 0-d array, takes a row for a column in silence and raises `AssertionError` with a message that lacks a space; `compute(compute_diag=True, compute_grad=True)` returns a pair that means nothing; `update(hyp=...)` on a GP without data stores a row of the wrong width, which `predict` then reads by offset | Python-only defects, latent (G2-19, 21, 24, 27, 30). Pull request 49 lists the checks of `update` as open | no | `wave6_G2_7_minor.py`; the exceptions are not the ones the two reports predict (`AxisError`, from the log of `tau`) | fix as one commit of input checks. The width check makes `test_predict_lpd` and its isotropic twin fail as they stand, so they are repaired with it (test notes) | as proposed |
| W6-35 | G2 internal M1, M2, M3, M11, M12, M14, M15; G2 comparison M4, M5 | Texts against the code: `predict_full` documents `add_noise` as defaulting to `True`; the summary of `predict` describes the noise-inclusive output, and no text says that the log density always has the noise; `AbstractKernel.compute` documents `(N,)` for the diagonal where every kernel returns `(N, 1)` and `predict` needs it; the nugget of `eps` with `constant_add=False` is not documented; the isotropic kernels have no page; a typo and a stale comment above `plot`. `predict_full` does not clamp its diagonal where `predict` and `gplite_pred.m:120` do: 242 negative entries, down to -0.11, on a nearly singular posterior | Python-only, of the documentation (G2-13, 14, 15, 22, 23, 25, 26) | documentation alone | G2-7 and G2-11 of the verifier's scripts | fix, one commit; `predict_full` documents that its diagonal is not clamped | as proposed |
| W6-36 | G2 internal M4, M5, M6, M7, M9; G2 comparison M8, M9, M10 | The variance over the hyperparameter samples has `ddof=1` in `predict` and in `quad`, behind `lcb_max`; `random_function(add_noise=True)` and `predict` without `y_star` leave out the noise terms they have no argument for; `s2_star` is ignored where the noise has no such feature; a degenerate training set gives infinite or inverted recommendations; `predict` raises for a log density without `y_star` where MATLAB returns an empty one; `Matern` has no default degree; every formula and gradient of the ported components agrees with MATLAB (kernels to 2e-15, the means exactly) | not defects (G2-12, 16, 17, 18, 20, 28, 29, 31 to 37): `gplite_pred.m:158`, `:160` and `gplite_quad.m:115` divide by `Ns - 1`, `gplite_rnd.m:67` and `gplite_noisefun.m:186-198` make the same omissions, and both sides repair the inverted pair with `UB = max(LB, UB)` (what is left of it is W6-4) | `ddof=1` yes, as MATLAB | G2-0 section E; `wave6_G2_7_minor.py` | none; two lines under the sheet's settled non-differences | as proposed |

### Outside the slices

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W6-37 | the orchestrator's candidate of 2026-09-21, from reading `load` (the worklog of the plan); verifier row G1-X | `VBMC.load(new_options=)` checks the names, updates the options and runs `_validate_option_values` (`vbmc.py:3197-3208`); it runs neither `Options.update_defaults` nor `_init_optim_state`. An option that is read at construction alone is taken and does nothing coherent, and the checks that live in `_init_optim_state` are passed by: `gp_mean_fun="nonsense"` is accepted where construction raises | Python-only defect, sharper than the candidate had it: a missing check beside the missing effect. W2-7 and W2-8 settled the order at construction and did not touch `load`, whose docstring says that the values are checked as at construction | whenever a user loads with such an option: `uncertainty_handling`, `specify_target_noise`, `gp_mean_fun`, `integer_vars`, `warmup`, `k_warmup`, `entropy_switch`, `active_search_bound`, `tol_bound_x` and the others that G1-X lists by reader | `scripts/wave6_G1_X_load_new_options.py`, on an object saved without a run: with `uncertainty_handling=True` the option is `True`, the level stays 0, the noise model of the GP `[1, 0, 0]`, `max_fun_evals` 200 for the 300 of construction, the search acquisition the noiseless one; `integer_vars=[True, False]` is accepted and the state keeps `[False, False]`; `max_iter` and `max_fun_evals`, the documented use, take effect | stricter interface: `load` refuses an option that is read at construction alone, with a message that says to construct a new object; the checks of `gp_mean_fun` and `integer_vars` move into `_validate_option_values`, so that both paths run them; the docstring corrected. In PyVBMC, one commit, with the changelog | as proposed |

### Found during the fix pass

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W6-38 | found by fix agent B while making W6-4 (`../fixes/wave6_agent_B.md`), fixed by fix agent A on the orchestrator's instruction; statement, verdict and reach corrected after the independent check of the pass (R1-2, R5-2) | The noise recommends the plausible pair `[0.5 * log(tol), log(std(y))]` (`noise_functions.py`, as `gplite_noisefun.m:105-106`), which is inverted for training targets whose standard deviation is below 1e-3. `fit` clips the plausible bounds into the hard bounds (`PLB = min(max(PLB, LB), UB)`, `PUB = max(min(PUB, UB), LB)`); with `LB <= UB`, which holds there, the two clips are one monotone map and keep the inverted pair inverted, unless the range of the targets is below 1e-6, where the hard pair collapses and takes both bounds with it. The space-filling design then fails `uuinv`'s ordering assertion | Python-only defect, latent: the inverted recommendation and the two clips are gplite's as well (`gplite_train.m:157-158`), and gplite draws its design from the inverted pair; gpyreg's `uuinv` alone asserts the order. As first written, this row placed the inversion in the clips and called both clips gpyreg's | no: PyVBMC gives the noise a Student's t prior, so its coordinate of the design takes the Student's t branch and never reaches `uuinv`. gpyreg alone reaches it, for targets whose standard deviation is below 1e-3 and whose range is above 1e-6, with a noise without a prior (agent A's test: range 8.2e-5, standard deviation 2.6e-5) | agent A's `test_fit_with_targets_of_a_tiny_range`, seen to fail with the `AssertionError` of `uuinv`; the mechanism, the check's probe on the pin and on the branch | fix, made as the last commit of the assembled branch: an inverted pair collapses onto its clipped upper bound, which the clip left inside the hard box. For the PI to strike before the pull request | keep (PI, 2026-09-22) |

## The first questions

**G1, `uuinv` and the three MATLAB commits of the slice.** gpyreg has all
three. Its `uuinv` and a line-by-line transcription of `fminfill.m`'s agree
to 0 over every set of bounds and every weight the G1 verifier tried, inputs
outside `[0, 1]` included; `46b6f5e` is carried by construction (`sX =
None`), and `fcf2674` by `slice_sample.py:452-453`. The fix of `uuinv` was
gpyreg's first: `00d9406` (2022-01-21) corrected the Python body nine days
before `1d1f20d`, by the same author. What the internal reviewer found, that
the docstring names another mixture than the body implements, holds on both
sides (W6-16). The design differs from MATLAB's in its source alone, SciPy's
Sobol engine, which the sheet has.

**G1, the rank-one extension of `update`.** Equal to a rebuild on the
enlarged training set at the same hyperparameters to 2.1e-15 in `alpha` and
1.1e-15 in the predictions with `L_chol` true, the one representation a run
has, at the three noise parameterizations, with and without `s2_new` and
with three hyperparameter samples (G1 verifier, after both reports). With
`hyp` passed `update` recomputes in full. The three states on which the
reports differ: a stored `sn2_mult` above one is producible by patching the
factorization, the extension is exact for it, and the distance from a rebuild
that needs no retry is the whole multiplier, the difference the sheet names
(W6-20); the low-noise representation is unguarded (W6-9); a new point whose
noise falls below 1e-6 keeps `L_chol` true, a difference of state and not of
value (W6-20). One thing neither report has: MATLAB's own rank-one extension
scales the new column by the new point's noise where the factor carries
`min(sn2) * sn2_mult`, and with noise that varies by point reproduces the
represented matrix to 0.116 relative, gpyreg's to 3e-16; it is latent in
VBMC, `gplite_post.m:76-79` refusing the path for a point that comes with
`s2`.

**G2, the predictive log-density.** Both reports are right on every point
(G2 verifier). `predict(return_lpd=True)` has the total predictive variance,
`max(s2, 0) + sn2_star * sn2_mult` (`gaussian_process.py:2032`): 1.8e-15
from the log density under the total variance, 0.8 to 4.9 nats from the one
under the latent variance. `predict` reproduces a transcription of
`gplite_pred.m` to 9.4e-14 over 120 configurations (five noise
parameterizations, one and three samples, the flags, `s2_star` given and
not), and bit for bit with a retried factorization at `sn2_mult` of 1e9 and
1e6 in the two representations of `L`. `add_noise` changes what is returned,
not the log density. The quantity entered gpyreg in that form with `9b3d46b`
(2022-06-02); MATLAB's line had the noise variance alone until `68a197b`
(2022-06-25), so MATLAB's fix brought MATLAB to gpyreg. Over the samples the
two differ in kind (W6-31). `predict_full` adds the same noise, wrongly
placed when it varies by point (W6-26); `quad` adds none.

## Defects on the MATLAB side

For `matlab_side_defects.md`, all from a reading of the source; nothing was
run in MATLAB, and the two verifier reports say for each what is read and
what is inferred from MATLAB's documented semantics. Only
`gplite/gplite_post.m:76-79` of the `gplite` directory is in the list so
far; the thirteen rows below are new.

| location | what the code does | consequence | gpyreg |
|---|---|---|---|
| `gplite/private/slicesamplebnd.m:362`, `:367-368`, `:371` | the sums of the burn-in statistics run over `ii > burn/2` and are divided by `floor(burn/2)` | with an odd burn-in, one term more than the divisor: the estimate is no variance, is negative where a coordinate's SD is small against its mean, and the complex root makes `:371` discard the adapted widths of every coordinate. VBMC's burn-in is `Thin*3 = 15` in the fits that do not recompute the posterior (W6-2) | not shared |
| `gplite/gplite_quad.m:66-67`, with `:101`, `:106` | the solves of the variance are normalized by the constant noise term, the factor by the minimum of the total noise (`private/gplite_core.m:82`) | with noise that varies by point the variance of the integral is clamped to `eps`; the mean is right (W6-30) | not since 1.2.1 |
| `gplite/private/gplite_core.m:244` | `dsn2(i)`, a linear index into an `N`-by-`Nnoise` array | with a constant total noise and two noise hyperparameters the gradient of the second is the first's at the second training point (W6-18) | gpyreg raises there |
| `gplite/gplite_post.m:230-232`, `:239` | the rank-one extension scales the new column, the diagonal and the new entry of `sW` by the new point's noise | with noise that varies by point the extended factor represents the matrix to 0.116 relative. Latent in VBMC (`:76-79`) | not shared |
| `gplite/gplite_covfun.m:198`, `:218-219` | the gradient of the Matern kernel of degree 1 is `Inf * 0` on the diagonal; the repair is commented out below | the gradient of the marginal likelihood is NaN and the kernel cannot be trained (W6-32) | shared |
| `gplite/gplite_rnd.m:102-105`, `:110`, `:61` | `robustchol` lets eigenvalues of rounding size through its tolerance and refuses the factor when one of them is negative | `gplite_rnd` fails on array sizes on a dense one-dimensional grid (W6-8) | the trigger shared; gpyreg returns the mean in silence |
| `gplite/private/slicesamplebnd.m:168`, `:176`, `:198`, `:377` | the base widths are copied before an infinite width is replaced, the check runs after | an infinite width returns after the burn-in and the chain is NaN (W6-14) | shared |
| `gplite/private/fminfill.m:133-136`, `:77`, `:169` | the comment of `uuinv` names equal weights for the two tails above a body that weights them by length; "half of all starting points" holds for the whole design when no coordinate has a prior and every bound is finite, each coordinate taking the weight `w = 0.5^(1/nvars)` (this row said "per coordinate", corrected with row W6-16); `p` outside `[0, 1]` is marked in one of three branches | a reader has the wrong design in hand (W6-16) | shared |
| `gplite/gplite_meanfun.m:142`, `:220-230`, `gplite/gplite_covfun.m:105` | `max`, `min`, `median` and `std` act along the row of a single training input | with one training point and `D > 1` the bounds come from the range across the dimensions. A direct call alone reaches it | the mean's bounds coincide there with gpyreg's pooled ones |
| `gplite/gplite_train.m:180`, `:250-256`, `:298` | `nll` keeps entries that no optimization wrote, and `min(nll)` searches them | none in practice | not shared |
| `gplite/gplite_post.m:250` | `gp.s2` is extended only when the new point comes with `s2` | `gp.s2` shorter than `gp.X` after a rank-one update without it | not shared (gpyreg fills with zero) |
| `gplite/private/slicesamplebnd.m:506`, and three more call sites under `utils/` | the diagnostics call `psrf`, which is in no file of the repository | the diagnostics fail without GPstuff on the path (inferred); `gplite_train` has them off | not shared |
| `gplite/gplite_train.m:142`, `gplite/gplite_covfun.m:130-131`, `gplite/gplite_noisefun.m:104` | equal targets give bounds of `-Inf`, and `UB = max(LB, UB)` leaves the pair | a pair `(-Inf, -Inf)` reaches `fmincon`; what it does with it was not looked up (W6-4) | the pair shared |

Read and found right, so that no later reader goes over them: the masks of
`gplite_hypprior.m:34-38`, MATLAB's `==` binding before `|`; the variance
clamp of `gplite_pred.m:120`; `gplite_pred.m`'s missing guard for an empty
`sn2_mult`, which `gplite_core.m:283` always sets.

## Sheet entries

Replaced: "NumPy quantile and standard-deviation conventions in GP bound
recommendations", now "NumPy's quantile convention in the GP mean-function
bound recommendations". Its half on the quantiles holds (`np.quantile`
against `quantile1.m`, +0.05 and -0.12 on a 30-point set); its half on the
standard deviation was wrong, the difference being the axis (W6-1).
Amended: the rank-one update (W6-20, with the guard of W6-9 as made); the
isotropic and rational quadratic kernels, whose MATLAB paragraph missed the
bounds branch of `isoflag` (W6-29). New: the window of the burn-in
statistics (W6-2, W6-3); the variance of `quad` (W6-30); the pooling of the
log density over the samples (W6-31); the two smooth-box hyperprior
families, the renormalization over the bounds with the prior of a fixed
hyperparameter, and "no prior is `None`, not an infinite scale", with the
coordinates of a block that have no prior (W6-24, W6-17); the exceptions of
the objective, left as they are by the ruling on
W6-19; the default starting point of `fit` (W6-24); equal targets (W6-4);
the negative eigenvalues of `random_function` (W6-7, W6-8); an infinite
width of the slice sampler (W6-14); the degree-1 Matern gradient (W6-32);
the noise gradient where the total noise is a scalar (W6-18); the removal
of the Metropolis step (W6-21). Lines under the settled
non-differences: `ddof=1` over the hyperparameter samples, the quiet
omissions of the noise function (W6-36), and `f_min_fill` with no more
evaluations than provided points (W6-25). The sheet states that its line
citations into gpyreg are at the pin and that the entries citing gpyreg by
function describe the branch. Written by the record scripts kept with
the wave's scratchpad copy (`dev/scripts/runs/LOCAL.md`): `records_sheet.py`
once W6-1 was first taken, `records_sheet_check.py` after the independent
check of the pass, `records_round_check_1.py` and `_2.py` after the check of
its fix round; its citations into `pyvbmc/` carried by
`refresh_citations.py`, one moved by hand (`1e6bf5e4`).

## Test notes worth acting on

The verifiers opened every test that the reviewers cite; their reports have
the full lists. What matters, all in gpyreg's suite:

- No test reads `PLB`, `PUB` or `x0` of any component, and
  `test_setting_bounds` compares the assembler with the components' own
  values: W6-1, W6-28 and W6-29 have no test of any kind.
- `test_predict_lpd` and its isotropic twin zero their `s2_star` and hand
  `update` a row of 12 hyperparameters for a GP of 11 (of 9, the twin), so
  the noise entry is read as the mean's constant; with `s2_star = 0.7` the
  tests' own formula is 5.9 and 15.2 nats from the returned value. They pin
  the log density at a total noise of `eps` alone, and they are repaired with
  the width check of W6-34.
- The tests around the rank-one update assert `L_chol`, use one
  hyperparameter sample and `sn2_mult = 1`; `wave6_G1_4_rank_one.py` has the
  states they leave out. `test_update_aligns_user_provided_noise` and
  `test_split_update` append several rows and never take the rank-one path,
  and the second is unseeded.
- `test_prior_mask_cache_follows_priors_and_bounds` compares a cached GP with
  an uncached one and can see a stale cache alone; the fixture of the priors
  puts each smooth box on a single hyperparameter, the one case W6-6 leaves
  right.
- `random_function` is tested for shapes, unseeded; `__robust_cholesky` has
  no test.
- The two tests of a frozen chain use the constants 0 and 1, the two for
  which W6-15 does not show; no test of the sampler has an odd burn-in, a
  small one, or a chain compared with a reference
  (`test_evaluations_stay_on_coordinate_line` is the model of such a test).
- `test_uuinv` asserts the masses of the body, all twelve pairs of bounds and
  weights, and a test written from the docstring fails on every set of bounds
  (against the internal report's count of three of four that would pass); it
  tests the specification once W6-16 corrects the text.
- `test_fitting_options` asserts nothing.
- Well tested, against the specification: `quad` with noise against
  Gauss-Hermite quadrature of `predict_full`, on a fresh posterior alone
  (W6-27 adds the other case); the cross-covariance interface;
  `_solve_triangular`; the batched means.

## Errors in the reports

The verifiers list them in full: thirteen in the internal G1 report and nine
in the comparison one, eight in the internal G2 report and twelve in the
comparison one. Those that change a row: the reach of the pooled
recommendations, where the comparison G2 report has `x0` in every fit and the
internal one misses the bounds that gpyreg fills (W6-1); three properties
that the internal G1 report reads as gpyreg's and that are MATLAB's (the
variance clamp before the rank-one update, the order of the base widths in
the sampler, the split of an odd number of samples), and its `update` on a
GP without hyperparameters, which does not raise and returns NaN factors
(W6-22); the isotropic bounds, shifted and not loosened, and their intent
(W6-29); the fallback of `__robust_cholesky`, the rule in one dimension and
never entered in two and three, and the zero factor, which hides the flip in
21 of 144 cells and not "in many cases" (W6-7, W6-8); the infinite plausible
bound of the rational quadratic kernel, which `fit` clips (W6-28); the
comparison G2 report's 4e-8 between the two representations of `L`, which is
its transcription's and 0 with `pL` formed as `gplite_core.m:98` forms it;
its note that no test meets the Matern gradient of degree 1, one tolerating
it by `equal_nan=True` (W6-32); `test_uuinv`, above. The line citations of
both G1 reports into `gaussian_process.py` have drifted, the internal one's
by up to 37 lines (`:1494` for `:1457`) and the comparison one's by up to 14;
`wave6_G1.md` gives the present line for every citation it uses, and its own
opening sentence, which has the internal report off by up to 18, is
contradicted by its list. The worklog entry of the wave said that the
citations of the comparison report hold, which was true of the five the
orchestrator had looked up; it is corrected there.

## Fix commits

Made on 2026-09-22. Three Opus agents on worktrees: A and B on worktrees of
gpyreg made by hand (`../gpyreg-port-review-A`, `-B`, branches
`port-review-wave6-A`, `-B`, cut from gpyreg's `main` at `fdbafdf`, the code
of the pin), A for `gaussian_process.py` and its tests, B for the other
modules, their tests and `docsrc/`; C on a harness worktree of PyVBMC
fast-forwarded to `b7622e51`. Their reports are `../fixes/wave6_agent_A.md`,
`wave6_agent_B.md` and `wave6_agent_C.md`. Each fix came with a test seen
to fail on the code before it, but for the agents' two commits of tests alone and
the three of documentation. The orchestrator reviewed each diff and
cherry-picked the commits: C's onto `dev-port-review`, B's and then A's onto
gpyreg's `port-review-wave6` (worktree `../gpyreg-port-review`), where they
applied without a conflict and have the hashes below. W6-1 was made last and
alone, by the orchestrator, after gate 1 below, with the seeding of
`test_fitting` before it; the PI's rulings of 2026-09-22 on the three
questions the pass had left open were to keep W6-38, to seed `test_fitting`
in this pass, and to leave the guard of W6-9 as made (the sheet's rank-one
entry describes it).

On `dev-port-review`:

| row | commit | |
|---|---|---|
| W6-23 | `4d3c1515` | `hyp_dict` no longer carries `logp`; the reference name removed from the three fit-history captures, whose sidecars carry an audit entry from the generator's rebaseline mode with every replayed output moved by zero (`0a6682fc`) |
| W6-37 | `e657eba8` | `load` refuses a construction-only option (`_CONSTRUCTION_ONLY_OPTIONS`, nineteen names, `bounded_transform` and `cache_size` beside the review's seventeen; `noise_shaping` is read in every iteration and is not one); the checks of `gp_mean_fun` and of the form of `integer_vars` moved into `_validate_option_values`, run before the names are weighed; the docstring and the FAQ. A saved run whose stored `integer_vars` has a form the current code refuses now fails to load (agent C's note 7; since `edc42739` and `b2d7c6e8` such a file opens, the option taking the mask the run was made with) |

On gpyreg's `port-review-wave6`, in order:

| row | commit | |
|---|---|---|
| W6-3 | `8dfbae4` | a coordinate whose burn-in variance estimate is not positive keeps its width; a window of fewer than two iterations adapts nothing; a test pins the window of the statistics (`floor(burn/2)` iterations) |
| W6-14 | `d07c849` | `base_widths` copied after an infinite width is replaced |
| W6-15 | `a29d0cf` | a frozen free parameter is recognized by its range, with `R` and `eff_N` NaN; the pairing of the autocorrelations kept at lag 0 and documented (agent B's note 3 compared it with BDA3 and Stan; the independent check of the pass found that comparison overstated, R3-8, and the docstring no longer makes it, `f1415de`); the two tests over several constants |
| W6-21 | `919742e` | the Metropolis step, its call sites, its options and attributes removed (the commit's message says that no test covered them; one did, R3-3) |
| W6-22, B's part | `9b282da` | a `thin` or `burn` that is not whole refused; `gammaln` in the two normalizers of `f_min_fill.py` |
| W6-16 | `3983eb3` | the docstring of `uuinv`, the comment on the half of the design, the mark of `p` outside `[0, 1]` in every return path |
| W6-4 | `c3e1901` | equal targets take a range of one and the standard deviation of a unit range (agent B's note 1: the range alone leaves the noise's plausible upper bound `log(std(y))` below its lower bound and the design fails), through `_target_spread` in the four helpers and the mean's, with one warning |
| W6-28 | `0ed8cfb` | the shape's plausible upper bound at `[-1]` |
| W6-29 | `4ac63a1` | the isotropic bounds take the means of the logs, `x0` the mean of the per-column log SDs |
| W6-32 | `7c382c4` | the degree-1 Matern gradient zero where two inputs coincide, both kernels; the tests run degree 1 and assert finite values |
| W6-34, B's part | `b291d21` | the kernels refuse `compute_diag=True` with `compute_grad=True` |
| W6-35, B's part | `97ac1c2` | the `(N, 1)` diagonal, the nugget, the page of the isotropic kernels |
| test note | `4415940` | the eight distributional tests of the slice sampler seeded (`test_multivariate_normal` failed once on the untouched worktree) |
| W6-26 | `f803be0` | `predict_full` adds the noise on the diagonal |
| W6-27 | `003d75d` | `quad` reads the stored `sl`; the assertion in the rank-one test |
| W6-33 | `e542794` | the `LinAlgError` of a failed factorization in both branches |
| W6-7 | `d96a602` | the sign flip per column and `eigh` |
| W6-8 | `2682a49` | a surviving negative eigenvalue within `10 * n * eps * max|D|` counts as zero, one beyond it raises; MATLAB's drop tolerance kept for which eigenvalues carry the matrix (agent A's first version widened it and moved every draw through the fallback; amended). Agent A's note 3: a covariance whose largest eigenvalue has collapsed with its negatives is not absorbed by a relative band and now raises with the value where it returned the mean |
| W6-9 | `4267466` | the low-noise rank-one branch warns and recomputes where `v_star <= sn2_eff`, the analogue of the other branch's test; agent A's note 4: a tiny positive latent variance still proceeds, and the branch stays inaccurate in that band, as the Cholesky branch does for a tiny positive `sqrt_arg` |
| W6-13 | `e4a5aac` | the rank-one path asks the posteriors for their factors |
| W6-5 | `1e19aa9` | the parentheses of the two masks; the comments of `set_priors`; a test of the log prior against the documented densities |
| W6-6 | `5a0ede8` | the normalization constants subscripted; `get_priors` with `np.all` |
| W6-17, W6-22 | `836af5d` | `set_priors` refuses a `sigma` that is not finite and positive, naming `None`; unknown names refused by `set_priors` and `set_bounds`; `get_recommended_bounds` takes any array_like, names `upper_bounds`, refuses a user pair given inverted (a recommended pair that comes out inverted is still collapsed, as `gplite_train.m:142` collapses it); `update` refuses unset hyperparameters by name; `fit` fills `df_base` into a copy, installed for the fit and restored in a `finally` (the objectives read `self.hyper_priors`; `git diff -w` is 84 lines) |
| W6-22 | `e0a057d` | `gammaln` in `__prior_masks`, the smooth-box-t normalizer the row's evidence cites; `nll` starts at `+inf` |
| W6-10 | `69b59f4` | `sampler_name` read first, `sampler` second; the `"laplace"` clause gone |
| W6-11 | `70a76d0` | the dictionary form of `log_likelihood` and `log_posterior` |
| W6-12 | `3f9d258` | the starting points a copy of the design |
| W6-18 | `ff1f14d` | `dsn2[0, i]` |
| W6-34 | `5ff9353` | `quad` refuses a GP without data or factors and an unsupported mean, and reads a one-dimensional measure as one of `D` dimensions (`gplite_quad.m:26`); `_convert_shapes` takes any number and 0-d array, checks the row count of `s2`, raises `ValueError`; `update(hyp=)` checks the width; `test_predict_lpd` and its twin repaired (a row of the right width, two samples that differ, `s2_star` not zeroed, the factor `np.pi` gone); its refusal of a `sigma` of one column, which 1.2.1 and gplite take, was withdrawn in `286b595` ("The check for substantial errors of the last round") |
| W6-35, W6-31 | `a077e70` | the docstrings of `predict`, `predict_full`, `quad` and `log_posterior` |
| test notes | `3faeaa9` | `test_split_update` seeded; `test_fitting_options` asserts |
| W6-38 | `caacbc1` | an inverted plausible pair collapses onto its clipped upper bound; kept by the PI's ruling (the commit's message gives the clip as the cause, which the independent check of the pass found wrong, R1-2; the row W6-38 and the comment of `a4f7969` give the mechanism) |
| test note | `8dec795` | `test_fitting` in both GP test files seeded (agent A's note 2); the fit lands within 0.042 of the generating hyperparameters, a margin of 0.46 to the tolerance of 0.5 (the commit's message calls the distance a margin; found by the independent check of the pass, R2-7) |
| W6-1 | `f76eca2` | `axis=0` in the statistics of `X` of the two helpers and of the rational quadratic kernel's copy, nine lines, and the comment beside the mean's `std` removed; four tests in `test_bounds_info.py` on inputs whose columns differ in location and spread, seen to fail on the pooled code |
| release notes | `dc2a930` | `docsrc/source/release_notes.rst`, a `1.3.0 (unreleased)` section, one point per change a user of 1.2.1 can see, from the agents' sentences |

Two things the agents found beyond the rows, for the record: gpyreg's
`test_fitting`, in both of A's test files, was an unseeded statistical test
that failed 2 of 6 runs on the untouched revision and 1 of 6 on A's branch
(A's note 2), seeded in this pass (`8dec795`); and `_convert_shapes`' type
annotation still names the narrower types (A's note 5), left as it stands.

## Gates

Gate 1, on the assembled branch (`caacbc1`, the 32 commits) and PyVBMC at
`28bf945f`, with `PYTHONPATH` naming `../gpyreg-port-review` and
`gpyreg.__file__` printed from it: the exact oracle check, 11 of 11 with
nothing re-baselined; the four seeded runs bit for bit the record after the
wave-5 pass, `wave5_gates/after_pass_23d962a1.npz` (92 arrays, 0 differ).
So the part-2 commits change no number of a PyVBMC run, as this ledger says.
gpyreg's suite on the assembled branch: 293 passed (222 passed and 1 failed
on the untouched worktree, the unseeded test). The focused tests of C's
files in PyVBMC: 388 passed, 15 skipped (the platform-bound oracles under
their skip), against gpyreg at the pin, before the branch was assembled; no
gate of the pass ran PyVBMC's suite against the branch, which the
independent check did (below). The logs are kept on the machine that ran
them
(`dev/scripts/runs/LOCAL.md`).

Gate 2, on the branch with W6-1 (`f76eca2`) and PyVBMC at `b2e858cd`, the
same way: gpyreg's suite on the working tree of the commit, 297 passed; the
exact oracle check, 1 of 11 fixtures ok, `gp_fit` and `gp_fit_history` moved
in the seven states that sample and the three fit-history captures moved,
`active_sample_step` and `gp_nlZ` bit for bit in every state, though the
plan had expected both to move, the second through `_install_hyperprior`,
which fills the mean's bounds from the recommendation; the four seeded runs
differ from `wave5_gates/after_pass_23d962a1.npz` from the first evaluation
after the initial design, 82 of 92 arrays, and are bit for bit the record
of the verifier's prototype, `wave6_A/A1c_per_column.npz`
(`gate2_compare_prototype_f76eca2.log`: 92 arrays, 0 differ; the runs in
`gate2_seeded_runs_w6_1_f76eca2.log`: 25, 17, 32 and 17 iterations; final
ELBO -1.7959, 1.2272, -1.445 +- 0.101 and
1.356 +- 0.080), so the commit is the patch the row was verified with.

The benchmark sweep (`wave3_gate_benchmark_sweep.py`, eight targets, the
same seeds before and after; before with gpyreg at the pin, after with the
branch at `f76eca2`, the added seeds at `dc2a930`, which differs from it in
the release notes alone; PyVBMC at `ac039528`, `b2e858cd` and `a3d4a70d`
across the runs, which compute the same, as gate 1's seeded runs and the
repeated seeds of corr_D5 show; the logs `w6_1_sweep_before_*`,
`w6_1_sweep_after_*`
and, for the added seeds, `w6_1_sweep_more_before_*` and
`w6_1_sweep_more_after_*` on the orchestrator's machine). It first ran
rosenbrock_D2 over 10 seeds, cigar_D4 over 6 and the other six targets over
3, and the PI took W6-1 on the orchestrator's summary of it. That summary
was wrong (the independent check of the pass, R5-1 and R5-16): it said that
four targets were better after on every mean, where two are, and that no
target degraded beyond its own seed spread, where on corr_D5, cigar_D4,
lumpy_D4 and the noisy Rosenbrock the worst seed after lay beyond the worst
before (corr_D5's ELBO error 0.048 against 0.0069 over three seeds; the
noisy Rosenbrock's gsKL 0.24 against 0.063). The PI reopened the ruling, and
those four targets were run to 10 seeds on both sides; corr_D5 ran all ten
anew, and its first three seeds reproduced both earlier sweeps exactly.
Means over the paired seeds, before -> after:

| target | seeds | abs(ELBO - lnZ) | gsKL | MMTV | seeds better after (ELBO / gsKL / MMTV) |
|---|---|---|---|---|---|
| rosenbrock_D2 | 10 | 0.0234 -> 0.0189 | 0.0240 -> 0.0190 | 0.0242 -> 0.0244 | 5 / 5 / 5 |
| cigar_D4 | 10 | 0.0043 -> 0.0085 | 0.0010 -> 0.0027 | 0.0096 -> 0.0135 | 2 / 5 / 5 |
| corr_D5 | 10 | 0.0157 -> 0.0208 | 0.0029 -> 0.0048 | 0.0097 -> 0.0115 | 3 / 3 / 4 |
| lumpy_D4 | 10 | 0.0379 -> 0.0450 | 0.0226 -> 0.0158 | 0.0277 -> 0.0268 | 5 / 5 / 6 |
| student_D4 | 3 | 0.0287 -> 0.0168 | 0.0349 -> 0.0306 | 0.0377 -> 0.0285 | 1 / 2 / 3 |
| banana_D2 | 3 | 0.0408 -> 0.0316 | 0.1397 -> 0.0956 | 0.0400 -> 0.0302 | 2 / 2 / 2 |
| halfnormal_D2 | 3 | 0.0040 -> 0.0030 | 0.0003 -> 0.0002 | 0.0196 -> 0.0198 | 2 / 1 / 1 |
| rosenbrock_D2_noise1 | 10 | 0.0971 -> 0.1004 | 0.0299 -> 0.0863 | 0.0339 -> 0.0517 | 4 / 4 / 4 |
| all | 59 | | | | 24 / 27 / 30 |

Over the 59 seeds the fix is better on 24 and worse on 35 by the ELBO
error, better on 27 and worse on 30 by gsKL (two tied), and better on 30
and worse on 29 by MMTV. Its means are worse on corr_D5, cigar_D4 and the
noisy Rosenbrock on all three metrics and on lumpy_D4 in the ELBO error;
the noisy Rosenbrock's gsKL and MMTV are carried by two seeds whose gsKL
(0.24, 0.31) and MMTV (0.11) no seed before reached. They are better on
student_D4 and banana_D2 on all three, on rosenbrock_D2 and half-normal in
the ELBO error and gsKL, and on lumpy_D4 in gsKL and MMTV. The losses are
thousandths of a nat of ELBO error and up to 0.06 of mean gsKL. The PI's
rule for the gate returns such a result to triage; the PI took the
per-column form to be the right one and asked why the pooled form scored
better. As first written, this paragraph said that the fix was better on
about as many seeds as it was worse and left out the losses on cigar_D4's
gsKL and MMTV, lumpy_D4's ELBO error and the noisy Rosenbrock's means (the
independent check of the fix round, F3-3).

The probe of the mean's bounds (`scripts/wave6_A1d_mean_bounds_probe.py`;
`scripts/wave6_A1d_read.py` prints every number below from the records
`wave6_A/A1d_pooled_1790104477.npz` and `A1d_per_column_1790104477.npz` on
the orchestrator's machine) repeats the sweep's runs on corr_D5, cigar_D4
and lumpy_D4 over seeds 1 to 4 and on the noisy Rosenbrock over seeds 1 to
4 and 9, with `GP.fit` wrapped to record, after each fit, the
hyperparameter vectors it returns, the number of training points, gplite's
per-column box of the mean computed from the training inputs, and the hard
bounds in effect; it draws nothing, and every run reproduces the sweep's
result for its seed. With the pooled bounds, the mean's location lies
outside gplite's per-column box in up to a third of the vectors of some
dimensions (cigar_D4 12 to 33%, the noisy Rosenbrock 25% in one dimension,
corr_D5 2 to 26%, lumpy_D4 1 to 8.5%), at the median 0.04 to 0.9 column
widths beyond it on three targets and 0.4 to 1.8 on cigar_D4, with a tail
to 1 to 3 widths (to 27 on cigar_D4). The mean's log scale leaves its box
less often, and only upward: in up to 9.3% of the vectors of a dimension on
cigar_D4, 5.9% on the noisy Rosenbrock, 3.5% on corr_D5 and 2.4% on
lumpy_D4. The vectors whose location lies outside are of two kinds. In
lumpy_D4, the noisy Rosenbrock and two dimensions of cigar_D4 the
quadratic is wide, 0.9 to 7 column widths at the median against 0.13 to
0.23 inside the box, and up to 26% of those vectors are wider than
gplite's cap of e^3 widths; its term changes by 0.05 to 3 nats across the
data, so the location is barely identified and drifts. In corr_D5 and the
other two dimensions of cigar_D4 the quadratic is narrow, 0.14 to 0.3
widths at the median (0.7 in one dimension of corr_D5), up to twice as
wide as inside the box, and falls by 15 to 90 nats across the data (3 in
that dimension): a one-sided slope with its peak just past the data. With
the per-column bounds the location does not crowd the edge of its box: the
outer 2% of the box holds at most 3.3% of the vectors, where the pooled
runs had up to a third outside it, and the displaced share lands within
the span of the data (corr_D5 dimension 0 from 47% to 60%, cigar_D4
dimension 0 from 46% to 64%). The length scales and the output scale sit
at their upper bounds, which are the same in both forms (`log(10 w_d)`,
`log(10 range(y))`), as often with either form: corr_D5's length scales
0.30 -> 0.30 and 0.36 -> 0.39 of the vectors, its output scale 0.37 ->
0.41; cigar_D4's length scales in two dimensions near half of the vectors
in both. The orchestrator's reading, which the probe does not test: the box
removes a region that the pooled fits reached with a differently shaped
quadratic, a separate mode of a misspecified mean, rather than truncating a
posterior that presses against it. The runs diverge after their first
iterations, so these shares mix different training sets, and the probe
does not say what separates the accuracies. PI's ruling of 2026-09-22:
W6-1 kept.

The targeted re-baselines, on the generating machine with BLAS
single-threaded, against the branch (`rebaseline_w6_1_f76eca2_*.log`):
`--rebaseline gp_fit --expect-moving gp_fit_history`, `--rebaseline
gp_fit_history`, then `--rebaseline-gp-fit-history` for each of the three
captures, each with its reason in the fixture's `.json`; the exact check
after them, 11 of 11. What moved: `gp_fit` and `gp_fit_history` in the
seven states that sample (`hyp` by 1.9e-6 in the two `normal_D2` states of
the history, whose columns lie alike, and by 6.6 to 29.5 elsewhere; nothing
in `normal_D2_singlesample`); of the captures, `early_sampled` and
`later_changing_ns` in their default sampler widths alone (by 0.097), their
fit bit for bit, and `noisy_nonuniform_weights` in its fit (`gp_hyp` by 77)
and its widths (0.031). The capture mode checks all three captures after
each rewrite, so the first two of its three runs wrote their capture and
then ended with the error that names the captures not yet re-baselined; the
third, and the exact check, passed. Committed on `dev-port-review` as
`3123135e`.

Gate 3, on the branch with the fix round of the independent check
(`d93f03c`, below) and PyVBMC at `326c7676`, with `PYTHONPATH` naming the
worktree and `gpyreg.__file__` printed: gpyreg's suite, 313 passed;
PyVBMC's suite against the branch, 1990 passed, 58 skipped; the exact
oracle check, 11 of 11; the four seeded runs, bit for bit gate 2's record
`after_w6_1_f76eca2.npz` (92 arrays, 0 differ), so the round changes no
number of a PyVBMC run. The logs are `gate3_*_d93f03c.log` on the
orchestrator's machine.

## The independent check of the pass

On the PI's instruction (`/doublecheck`, 2026-09-22) the pass was read by
five fresh Opus reviewers, read-only, on gpyreg `dc2a930` and PyVBMC
`a3d4a70d`: R1 the commits on the posterior, the rank-one update, the
prediction, `quad` and `random_function`; R2 the priors, the bounds of `fit`
and its interface, and the two commits of tests; R3 the other modules of
gpyreg, W6-1 and the release notes; R4 PyVBMC's commits of the pass, the
coupling between the branch and PyVBMC's use of it, and the changelog; R5
the records. They returned 7, 13, 11, 9 and 30 findings. Their reports and
probes are kept on the machine that ran them (`dev/scripts/runs/LOCAL.md`).

What had to be fixed:

- No gate of the pass had run PyVBMC's suite against the branch (R4-2).
  Run then against `dc2a930`: 1 failed, 1986 passed, 58 skipped. `set_priors`
  refused a block prior whenever a `sigma` in it was not finite and
  positive (W6-17, `836af5d`), and `_gp_hyp` writes the prior of
  `noise_rectified_log_multiplier` with one coordinate that has none, its
  location and scale NaN (R4-1, R2-2). No option reaches that branch in a
  run; its unit test failed.
- `random_function` raised `LinAlgError` on every grid inside the training
  data at small noise (R1-1). The band of W6-8 was measured against the
  largest eigenvalue of the predictive covariance, about 1e-5 there, while
  that matrix is a difference of terms of the size of the prior variance,
  whose rounding left eigenvalues near -5e-15.
- `get_priors` returned `None` for a prior whose degrees of freedom are
  NaN, so `set_priors(get_priors())` dropped it (R2-1).
- `load` refused a run saved by release 1.0.4 whose `integer_vars` has a
  form the option check refuses, such as one zero or one per variable,
  which 1.0.4 read as a mask; such a file could not be opened (R4-3).
- The orchestrator's summary of the benchmark sweep, on which the PI had
  taken W6-1, was wrong (R5-1, R5-16). "Gates" has the reread, the extended
  sweep, the probe of the mean's bounds and the PI's final ruling.
- Records the check found wrong, corrected where they stand: the
  mechanism, verdict and reach of W6-38 (R1-2, R5-2); the premise of W6-21
  (R3-3); the claim of W6-16 on half of the design (R3-1); the margin of the
  seeded `test_fitting` (R2-7); the comparison with BDA3 and Stan under
  W6-15 (R3-8); the account of gate 1 (R4-2); citations and claims of the
  sheet, its history phrasing, and five entries it lacked (R5-3 to R5-8,
  R5-10 to R5-13, R5-26); the changelog's gpyreg requirement, which named a version not yet
  released (R4-4, R5-22); entries 41 and 49 to 52 of
  `matlab_side_defects.md` (R5-24, R5-30).

The PI's rulings of 2026-09-22: `set_priors` takes a coordinate of a block
whose location and `sigma` are both NaN as having no prior and refuses any
other `sigma` that is not finite and positive; the band of `random_function`
is measured against the prior variance at the test points; `load` gives a
stored `integer_vars` that fails the check the mask the run was made with;
`get_priors` returns a prior with NaN degrees of freedom in a form
`set_priors` reads back, and a NaN `df` is documented; the removal of the
Metropolis step stands, with its record corrected; the fixture of the
priors puts its smooth boxes over blocks, and a scan test keeps the
construction-only options of `load` in step with the code; the sheet states
the revisions of its gpyreg citations and states facts where it narrated
the pass; the message of `8dec795` stands, its error flagged in "Fix
commits". The other findings were taken as the orchestrator proposed; those
left for later went into `TODO.md` ("What the independent check of the
wave-6 pass left for later"), and the PI ruled on them on 2026-09-23, after
the review closed: the ledger
`dev/results/2026-09-23-port-correctness-review.md`, "Findings ruled after
the close", gives each ruling and fix.

Fix agent D, on a worktree of gpyreg made by hand
(`../gpyreg-port-review-D`, branch `port-review-wave6-D`, from `dc2a930`),
made the gpyreg side, one commit per finding, with a test seen to fail
first for every change of behaviour (`../fixes/wave6_check_agent_D.md`). The
orchestrator reviewed the diffs and took the branch onto `port-review-wave6`
as a fast-forward, so the hashes stand, and made the PyVBMC side.

On gpyreg's `port-review-wave6`, in order:

| finding | commit | |
|---|---|---|
| R4-1, R2-2 | `c4a441e` | `set_priors` takes a coordinate of a block without a prior (NaN location and `sigma`) and names what is wrong with any other `sigma` that is not finite and positive |
| R1-1 | `97d2b1a` | `__robust_cholesky` takes the scale of the terms that formed the matrix; `random_function` passes the largest prior variance at its test points; a grid inside the data draws |
| R2-1 | `2e96c90` | `get_priors` reads a block's family from its coordinates that have a prior, and returns NaN degrees of freedom and partial blocks as `set_priors` reads them; `set_priors` and `fit`'s `df_base` document a NaN `df` |
| R2-6 | `ebc9dcd` | the smooth boxes of the prior fixture over blocks of three coordinates (tests alone; with W6-6 reverted, the ten tests that use the fixture fail) |
| R1-5 | `43bf232` | an unsupported type of `s2` raises `TypeError` |
| R3-5 | `fa2c70c` | targets with a NaN, or all of one infinity, are not reported as all equal |
| R3-11 | `ad685ef` | an infinite `thin` or `burn` refused with `ValueError` |
| R1-3 | `2d5796d` | the test of the low-noise duplicate fires the guard by construction (tests alone) |
| R1-6 | `4994bbe` | the tiny-range test asserts on the plausible pair the design receives (tests alone) |
| R1-2, R5-2, R5-14 | `a4f7969` | the mechanism of an inverted plausible pair, in `fit`'s comment and the test's docstring |
| R1-4, R2-4, R2-9, R2-11, R5-29 | `51e492f` | `predict`'s pooled variance, the `Raises` sections of `update`, `get_recommended_bounds` and `fit`, three comments, and the wording of `update`'s refusal of unset hyperparameters, a message no test pins |
| R2-5 | `7a48db8` | the comment in the test of the documented densities |
| R3-1 | `c36bada` | the weight of the plausible box in the space-filling design |
| R3-6, R3-8 | `f1415de` | the Notes of the effective sample size |
| R3-9 | `39ef329` | what the isotropic kernels take from each parent |
| R1-2, R1-3, R2-3, R3-2, R3-3, R3-4, R3-7, R5-9, R5-14, R5-15 | `d93f03c` | the release notes, true point by point, with the Upgrading lines of the new refusals |

On `dev-port-review`:

| finding | commit | |
|---|---|---|
| R4-3 | `edc42739` | `load` gives a stored `integer_vars` that fails the check the mask the run was made with; the changelog's sentence corrected |
| R4-5 | `326c7676` | a scan test: `_CONSTRUCTION_ONLY_OPTIONS` is exactly the set of options read only at construction or by a value check (tests alone) |

Found by agent D on the way and left for triage: `set_priors` takes a NaN
location beside a finite `sigma`, which gives a NaN log prior, the mirror of
R4-1 (`TODO.md`). Gate 3, under "Gates", is the gate of the round.

## The independent check of the fix round

On the PI's instruction (`/doublecheck`, 2026-09-22) the fix round of the
check was read by three fresh Opus reviewers, read-only: F1 fix agent D's
16 gpyreg commits and the release notes, F2 PyVBMC's `edc42739` and
`326c7676` with their changelog lines, F3 the records `262d5d3d` and
`1e6bf5e4` and the probe scripts. They returned 8, 8 and 27 findings. Their
reports and probes are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`).

What had to be fixed:

- `random_function` still raised `LinAlgError` whenever the posterior is
  in its low-noise representation (a smallest noise variance below 1e-6),
  inside the data and outside it (F1-1). The posterior holds the negative
  inverse of the training covariance there, and the predictive covariance
  formed from it carries a rounding that grows like the inverse of the
  noise: negative eigenvalues from -7e-11 to -2e-2 against a band of
  3e-13, at noise standard deviations from just below 1e-3 to 1e-6. A
  default fit on noiseless targets ends there. The band of the round
  reached the Cholesky representation alone, and the test sat two ulps
  above the switch between the two.
- Three records of the round were wrong: the reading of the sweep said the
  fix was better on about as many seeds as it was worse, where by the ELBO
  error it is worse on 35 of 59, and left out losses (F3-3); the probe
  paragraph said that the mean's log scale almost never left its box, where
  it does in up to 9.3% of the vectors of a dimension (F3-2); and the
  correction of W6-16 was not carried to the table of MATLAB-side defects
  and to entry 48 of `matlab_side_defects.md` (F3-1). All three are
  corrected where they stand.

The PI's rulings of 2026-09-22: `random_function` forms the covariance of a
low-noise posterior from a Cholesky factor; `load` gives the option the
run's mask wherever its reading differs from it, not only where the check
refuses it; the changelog names the gpyreg version it requires together
with the minimum version of `pyproject.toml`, after gpyreg's release, and
the sheet's header moves with the pin; the small findings no record held
go to `TODO.md`. The other findings were corrected as the orchestrator
proposed: the release notes (F1-2 to F1-4, F1-8, and the other changed
exception types that fix agent E found), the docstrings and comments (F1-5,
F1-7), the scan test and the comment of the tuple (F2-1, F2-2, F2-4, F2-5),
the `Raises` of `load`, the test of the all-zero form and the `logp` lines
(F2-6 to F2-8), and the records (F3-4 to F3-27; the probe's numbers are
printed by `scripts/wave6_A1d_read.py`).

Fix agent E, on a worktree of gpyreg made by hand
(`../gpyreg-port-review-E`, branch `port-review-wave6-E`, from `d93f03c`),
made the gpyreg side (`../fixes/wave6_check_round_agent_E.md`); the
orchestrator reviewed the diffs, added one commit of text, and took the
branch onto `port-review-wave6` as a fast-forward, so the hashes stand.

On gpyreg's `port-review-wave6`, in order:

| finding | commit | |
|---|---|---|
| F1-1 | `5db2255` | `random_function` factors the training covariance of a low-noise posterior, with its kernel, noise and multiplier, through the factorization and retry that compute the posterior, moved into a helper that `__core_computation` calls (its results bit for bit, over 51 arrays); the dense-grid test states its representation and covers the low-noise one, inside and outside the data; a test of two hyperparameter samples with and without `add_noise` |
| F1-5, F1-7 | `2938581` | the reading of a NaN `df` gplite shares, and the weight of the plausible box in the design |
| F1-1 to F1-4, F1-8 | `dd12db3` | the release notes: low-noise draws, the refusals of inputs 1.2.1 took and of inputs it failed on, `thin` and `burn` of any whole type, the remedy of `update` on unset hyperparameters, the change of the shape checks' exception, one dimension |
| F1-5, F1-8 | `ec0f085` | the orchestrator's: the rule for no prior and the smooth-box reading stated as gpyreg's own at the three sites left, and the other changed exception types in the Upgrading line |

On `dev-port-review`:

| finding | commit | |
|---|---|---|
| F2-3, F2-6 to F2-8 | `b2d7c6e8` | `load` gives a stored `integer_vars` the run's mask wherever its reading differs from it; the test covers an array of zeros and the list `[1]`; the `Raises` of `load`; the changelog's lines on `integer_vars` and `logp` |
| F2-1, F2-2, F2-4, F2-5 | `503562d7` | the scan takes any name that contains `options`, matches its set-aside reads by site and names the sites of a difference; the comment of the tuple states the rule (tests alone apart from the comment) |

Found by fix agent E on the way and left for triage (`TODO.md`): `predict`
and `predict_full` form the low-noise covariance from the same explicit
inverse, as `gplite_pred.m:95-96` does, so their variances there carry
the rounding (1.9e-3 against true variances of about 1e-12 inside the data
at a noise standard deviation of 1e-6), and the low-noise rank-one update
divides by such a variance; no PyVBMC run reaches the low-noise
representation unless `tol_gp_noise` is set below 1e-3, its noise
standard deviation being at least `tol_gp_noise`. And `fit` with a whole float `thin` still raises
`TypeError`.

Gate 4, on the branch with E's round (`ec0f085`) and PyVBMC at `503562d7`,
with `PYTHONPATH` naming the worktree and `gpyreg.__file__` printed:
gpyreg's suite, 319 passed; PyVBMC's suite against the branch,
1992 passed, 58 skipped; the exact oracle check, 11 of 11; the four seeded runs,
bit for bit gate 2's record `after_w6_1_f76eca2.npz` (92 arrays, 0 differ). The logs are `gate4_*_ec0f085.log` on the orchestrator's
machine.

## The check for substantial errors of the last round

On the PI's instruction (`/doublecheck`, 2026-09-23, "only substantial
errors") three fresh read-only Opus reviewers read the last round: G fix
agent E's gpyreg commits and the release notes, P PyVBMC's `b2d7c6e8` and
`503562d7`, R the records of `858742f1`. Their reports are kept on the
machine that ran them (`dev/scripts/runs/LOCAL.md`). P found none.

G found one, in a commit of the first round: the width check that `quad`
gained with W6-34 (`5ff9353`) refused a `sigma` of shape `(N, 1)` in more
than one dimension, which release 1.2.1 broadcast across the dimensions,
returning the integrals of the same widths written out per dimension, and
which `gplite_quad.m` broadcasts through `bsxfun`; and the release notes
said that 1.2.1 raised on a measure of the wrong width, where it raised
`IndexError` for a narrower `mu` alone. Fixed by the orchestrator in
`286b595`: `quad` tiles a `sigma` of one column and refuses a `mu` of
another width than `D` and a `sigma` of any width but one or `D`; the
release notes describe what 1.2.1 did; a test compares a one-column
`sigma` with its tiled form and failed before the fix. gpyreg's suite on
`286b595`, 320 passed (`gate5_gpyreg_suite_286b595.log`); no PyVBMC code
calls `quad`, so no gate of PyVBMC is concerned.

R found two: the plan's pickup point left out `dev-port-review-w1check`, a
parallel session's check of wave 1, cut at `326c7676` in the worktree
`../pyvbmc-w1check`, which its own plan entry says waits to be brought
onto `dev-port-review` and which shares seven files with it; and row W6-16
gave the condition of the half of the design without gpyreg's fixed
coordinates. Both are corrected.
