# Wave 5 verification

The findings of wave 5 (slices P7 and P9, both tracks, and the internal track
of P2; reports under `../reviews/`), verified on 2026-09-21 on
`dev-port-review`, whose package code is that of `831acef`, gpyreg at
`9e70e6b`, against MATLAB VBMC at `396d649`. The reviewers of P7 and P9 read
`f873556`; between the two revisions two files of their slices changed,
neither in a line a finding cites (`stats/get_hpd.py`, `027e972`, where the
rounding is now at line 38; `priors/product.py`, `b693e42`,
`Product._generic` alone). The reviewer of P2 read `831acef`. Part 1 holds
the findings that can change what a run computes; part 2 holds the others.
Two sources feed the ledger: the orchestrator's own checks, for part 1, with
the scripts named in the rows; and three read-only Opus verifiers, one per
slice, whose raw reports are `wave5_P7.md` (rows P7-n), `wave5_P9.md` (rows
P9-n) and `wave5_P2.md` (rows P2-n). The scripts are under `scripts/`; the
logs of the orchestrator's are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`, "Port correctness review"). Every MATLAB line
involved in part 1 predates the Python line, so no row there is a faithful
port of an older MATLAB.

The column "proposed" holds the orchestrator's proposal and the column "PI"
the PI's ruling, all of 2026-09-21: W5-25 while the verification was under
way, W5-14, W5-26 and W5-40 on the orchestrator's questions, and the others
by taking the proposals as they stood ("as proposed").

## Part 1: findings that can change what a run computes

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W5-1 | P7 internal F6, P7 comparison F3, independently | The balanced draw of `VariationalPosterior.sample` takes `floor(w*N)` draws from each component and draws the remainder by weight. `variational_posterior.py:641-644` corrects the weights of the remainder with `self.w * (repeats_extra - sum(w_extra))`, and the builtin `sum` of a `(1, K)` array returns its row, so the correction is applied element by element: the remainder is drawn with probabilities proportional to `e_k + w_k (R - e_k)`, `e_k` the fractional parts and `R` their total, where `vbmc_rnd.m:67-71` adds `w*delta_extra` with the scalar `delta_extra = N_extra - sum(w_extra)`, zero up to rounding, and draws in proportion to `e_k`. MATLAB's expected count of component `k` is `w_k N`; PyVBMC's is off by up to about one draw per component at any `N` | confirmed port discrepancy; never matched. The Python line is that of `42a0deef` (2021-03-14, "initial version of VariationalPosterior.sample()"); MATLAB's dates from `a00a609` (2018-08-29). No reason recorded; `np.sum` stands on the line above | yes, every balanced draw of a posterior with `K > 1`: the posterior and heavy-tailed candidates of every active-sampling step (`active_sample.py:968`, `:1065`), and the Monte Carlo moments in original space (`variational_posterior.py:1167`) behind `sKL` at every iteration, behind `kl_div`, `mtv` (`variational_posterior.py:1346`) and the moments a user asks for | `scripts/wave5_A1_balanced_remainder.py`: with `w = [0.7, 0.2, 0.1]` and `N = 13` the one remainder draw has probabilities `[0.41, 0.38, 0.21]` for MATLAB's `[0.1, 0.6, 0.3]`, and 20 000 repetitions of `vp.sample` give frequencies `[0.724, 0.183, 0.093]`, the coded expectation; equal weights, the case of the two tests, give the same law on both sides. On the `K = 50` weights of the MATLAB fixture the two remainder laws are 0.18 to 0.21 apart in total variation at `N = 100` to `1e5` (22 to 27 remainder draws), and the expected counts are off by up to 1.1 draws. What the corrected line moves (the method replaced in the process, same seeds): about 16 of 2048 rows of a draw; the acquired point of one active-sampling step on none of the seven stored states that have the oracle; no oracle reference (`scripts/wave5_A1c_oracles_corrected_sampler.py`: the exact check, 11 of 11); in the four seeded runs of the gates (`scripts/wave5_A1b_gate_runs_corrected_sampler.py`) `sKL` and `r_index` of all four, by up to 2.5e-5 and 5.9e-4, three runs being bit-identical otherwise, and the whole of `rosenbrock_D2_noisy` (30 iterations and 150 evaluations for 32 and 155; ELBO -1.635 +- 0.114 for -1.536 +- 0.111) | fix (`np.sum`), with a test on weights for which the two laws differ. It moves default trajectories now and then, so it belongs with the moving fixes that the one regeneration of the golden references waits for; its statistical effect is nil, so no benchmark sweep | fix, as proposed; a moving fix |
| W5-2 | P7 internal F8; the last minor observation of P7 comparison | `kl_div_mvn` (`stats/kl_div_mvn.py:34-39`) computes the two determinants, returns `(inf, inf)` when either is zero and otherwise takes `log(detq2 / detq1)`. The determinant of a well-conditioned covariance leaves the range of a double at moderate `D`: it is zero below an SD of about 8e-9 per coordinate at `D = 20` (1.6e-11 at `D = 15`), subnormal and good for six digits at 1e-8, and infinite above about 5e7, where the ratio is NaN | confirmed shared defect: `shared/mvnkl.m` has the same determinants and ratio, without the guard, and gives NaN or `Inf` in the same cases. The determinants are from the first port of `kl_div` (`fdb1e56d`, 2021-04-22, in `variational_posterior.py`; moved to `stats/kl_div_mvn.py` by `1b31ffc0`, 2021-06-09); the guard is Python's own (`fdb6e94f`, 2022-12-19, pull request 125) | no at ordinary scales. It is the default route of `sKL` (`kl_gauss=True`), on the covariance in original space, so it is reached by a problem of 15 to 20 parameters in units that make every posterior SD tiny or huge | `scripts/wave5_A2_kl_div_mvn_determinants.py`: against the closed form and the same formula with `slogdet`, which is exact in every case; `vp.kl_div(gauss_flag=True)` on two posteriors of `D = 20` with SD 1e-9 returns `[inf, inf]`, so `sKL = inf`, the reliability index is infinite and the run is never stable; with SD 1e9 it returns `[nan, nan]` and `max(0, nan)` at `vbmc.py:1542` makes `sKL = 0`, the value of a posterior that has stopped moving | fix: the log determinants from `slogdet` (or a Cholesky factor), infinite only for a singular matrix. Moves no number at ordinary scales beyond the last digits of `sKL`; to be checked against the seeded runs like any fix | as proposed |
| W5-3 | P7 internal F9 | `entmc_vbmc` draws samples from every component, takes `log(E @ wnf)` at them and multiplies the sum by the component's weight (`entmc_vbmc.py:217-218`, `:234`). With a weight of exactly zero and components far enough apart for the others' densities to underflow at its samples, that is `0 * (-inf)`: `H` is NaN, and the gradient is not finite from a weight of about 1e-322 | confirmed shared defect, latent: `ent/entmc_vbmc.m:63-67` has the same product. The lower bound `entlb_vbmc` does the same on the same posteriors, which the report does not say | no. A weight is exactly zero when its `eta` lies more than 745 below the largest (a gap of 745 leaves the smallest subnormal, 746 gives zero). Nothing bounds `eta` since the soft bound was removed, yet an ADAM step is of the order of `sgd_step_size = 0.005`, an optimization has at most `100 (2 + D)` of them, and a component below `tol_weight = 0.01`, 4.6 below the largest, is a candidate for pruning after each: a run cannot get there | `scripts/wave5_A3_entmc_zero_weight.py`: means 60 apart, both entropies finite down to `w = 1e-304`, gradients not finite at an `eta` gap of 740, `H` infinite at 745 and NaN from 746; with means 3 apart everything is finite at a weight of zero | leave; a line in `matlab_side_defects.md` under the shared ones | leave |
| W5-4 | P7 internal F1 and F10, P7 comparison F2; the two disagree on the reach | `kl_div(gauss_flag=False)` guards its densities with `q[q == 0 \| np.isinf(q)] = ...` (`variational_posterior.py:1492-1493`, `:1499-1500`), which Python reads as `q == (0 \| np.isinf(q))`: a zero is caught, an infinity never, where `vbmc_kldiv.m:75-76`, `:82-83` have `q == 0 \| ~isfinite(q)`, in which `==` binds first, and catch both and NaN. The zero guard caps the divergence of two posteriors that do not overlap near `-log(realmin)` without a message, on both sides. The density the guards protect is formed in original space as `y / exp(logJ)` (`:917-928`), which is infinite where `exp(logJ)` underflows although the quotient is representable; the log form subtracts | confirmed port discrepancy for the infinity (correct in `fdb1e56d`, 2021-04-22, as `np.logical_or(q1 == 0, np.isinf(q1))`; rewritten by `4a071d8c`, 2022-09-13, pull request 96, with no reason recorded), never matched for NaN; the cap is shared by design; the division is a Python-only defect, latent | no: `kl_gauss=False`, or a user's call. An infinite density then needs a posterior far wider than the bounded transform's scale: under a probit transform none among 1e5 draws up to an SD of 8 in the inference space, 8 at 10, a quarter of the draws at 60 (the comparison reviewer tried up to 6, the internal one 60: both are right) | `scripts/wave5_A4_kl_div_guards.py`: the masks on `[0, 1, inf, nan, 0.5]`; two unit Gaussians 80 apart return `[706.98, 706.98]` for a divergence of 3200; from an SD of 10 `kl_div` returns `[nan, nan]`; in `D = 2` at the image of `u = (-27.3, -27.3)` under a posterior of SD 4.5 the log density is 705.5, its exponential 2.4e306, and the linear form returns `inf` | fix the guard as MATLAB has it (`~np.isfinite`), with a test; fix the division by exponentiating the difference of the logs. No effect at the defaults | as proposed |
| W5-5 | P9 internal F6 | `VBMC` checks the dimension of a prior and nothing else: `Prior.support()` has no reader in `vbmc.py`. A prior narrower than the hard bounds makes the log joint `-inf` inside the box, and the run stops in the function logger with `FunctionLogger:InvalidFuncValue`, which names neither the prior nor the bounds. A prior wider than the bounds is left with its own normalization, so the evidence is that of the prior's density restricted to the box, not of a prior truncated to it and normalized again | not a defect of the computation, and shared by design: a MATLAB target that returns `-Inf` inside the bounds stops in `misc/funlogger_vbmc.m:121` alike, and the restriction to the box is what hard bounds mean on both sides. The Python-only part is that the prior object knows its support and the constructor does not look at it, while for a PyMC target the bounds must equal the model's support (`vbmc.py:93`, `:581`) | no: a user's prior whose support is not the box | `scripts/wave5_A5_prior_support_and_bounds.py`: `UniformBox(0, 1)` on the box `[0, 10]^2` with plausible bounds inside the support raises at evaluation 13, the third of the first active-sampling iteration; with `UniformBox(-100, 100)` the log joint carries `-2 log 200` | stricter interface, for the PI to weigh: refuse at construction a prior whose support does not cover the box (the case that stops a run), and say in the documentation of `prior` that the evidence is that of the prior as given, restricted to the hard bounds | as proposed: refused at construction, and the documentation |
| W5-6 | P7 internal F11, P7 comparison F7; wave 1 confirmed it as a minor observation of P2 and left it | `get_hpd` sizes its subset with the builtin `round` (`stats/get_hpd.py:38`), a half to even, where `misc/gethpd_vbmc.m:10` has MATLAB's `round`, a half away from zero. The builtin `round` also splits the search points among their sources (`active_sample.py:946`, `:964`, `:973`, `:981`, `:1026`) and `np.round` splits the high-posterior-density draws among six fractions (`:994`), each where `private/activesample_vbmc.m:565-605` has `round`. Five more sites of the builtin stand where MATLAB has `round` (`gaussian_process_train.py:506`, `:602`, `:631`, `:632`; `variational_optimization.py:46`); of their ties the orchestrator looked at one, `ns_gp_max / sqrt(N)`, which has a single one, at `N = 1024` | confirmed port discrepancy; never matched (`aa454736`, 2021-07-26; MATLAB `f62075b`, 2019-09-22) | not in a run that starts from no more points than its initial design takes, and in no gate. Every default call passes `hpd_frac = 0.8`, which has no tie for any `N` up to 2000 (the internal report's `N = 5, 15, 25` is wrong), and the default fractions of the search points are quarters of `ns_search = 8192`. A tie needs `hpd_search_frac > 0` (default 0), whose six fractions include the exact 0.1: one point fewer than MATLAB at `N = 5, 25, 45, ...` live points, and none at `N = 5` where MATLAB has one; or a user's `hpd_frac` (0.5: every fourth `N`). One tie needs no option: more starting points than the initial design takes leave a starting cache, and at an active-sampling step at which it leaves a number of random points that is 2 modulo 4, the quarter shares of the sieve are half-integers (8186 random points: 2047 candidates per quarter for 2046) | `scripts/wave5_A6_get_hpd_rounding.py`: the ties counted in floating point as the code computes them; `get_hpd` at `N = 5`, `hpd_frac = 0.1` returns no point; `np.round` splits differently from MATLAB's `round` for a third of the counts up to 8192 | fix the seven sites of the search and of `get_hpd`, and the five others with them, with the helper of `70ce067`, which rounds from the exact fractional part; `get_hpd`'s test on a tie. Moves no gate and no run that starts from one point; the tie of the starting cache is the one within reach of the shipped options | fix at all twelve sites |
| W5-7 | P2 internal F1 | A cached starting point that wins the sieve keeps its cache index, and its stored value is recorded, with no target call, at the candidate as `_get_search_points` clipped it into the search box (`active_sample.py:939`, `:1072`) and `active_sample` snapped it to the integer grid (`:383`, `:679-698`) | confirmed shared defect, latent: `private/activesample_vbmc.m:555`, `:637`, `:219` and `:388` clip, snap and add the stored value in the same order | no, and the clip is out of reach at the default `active_search_bound = 2`: the starting points lie in the plausible box, which is widened to hold them, the search box adds two widths on each side, and a warp replaces it by the bounding box of its own image, not of the training inputs as the report has it. The snap needs `integer_vars`, more starting points than the initial design takes, their values provided, and an off-grid coordinate among them | `scripts/wave5_A7_cached_value_moved_point.py`: on a state built so that the cached row is the one candidate, a cached `[50, 0]` with value -1250 is recorded at `[15, 0]`, where the target is -112.5, and a cached `[2.4, 0.1]` at `[2, 0.1]`, both without a target call; the corners of the plausible box of the eight stored states stay 0.385 box widths or more inside the search box through `warp_input` | fix cheaply or leave: a cached row that the clip or the snap moved loses its cache index, as the result of the local search already does (`:629`), so that the target is called. For the PI to weigh with the documented status of `integer_vars` (P2 verifier's rows) | fix: the stored value is not reused for a cached row that the clip or the snap moved; the target is called at the candidate and the row leaves the cache. The proposal left the choice between this and leaving it; the orchestrator takes the fix under the PI's general "as proposed", for the PI to strike before the push |
| W5-8 | P2 internal F6; the defect is row W3-20 of wave 3 (P8 internal F5), which the PI ruled to leave | `FunctionLogger._record` looks for an earlier evaluation of the same input over the whole array of inputs (`function_logger.py:685`), rows that the trim at the end of warm-up switched off included. An acquired point equal to such a row is evaluated and pooled into it; the row stays off, so the evaluation reaches neither the GP nor `y_max`, and `update1` being false the posterior is recomputed from an unchanged training set | confirmed shared defect, latent, as W3-20 has it: `misc/funlogger_vbmc.m:220` scans the whole of `optimState.X`, and nothing on either side switches a row back on. What the report adds is a second route to it. W3-20 names one, a trimmed input proposed again through the search cache (`search_cache_frac > 0` with `max_repeated_observations > 0`); the other is a grid | no. Equality in every coordinate, bit for bit, needs either the route of W3-20, a search cache that hands a trimmed training input back (`search_cache_frac > 0` with `max_repeated_observations > 0`), or a problem whose variables are all integers (a repeated observation is taken among the live rows alone); and the rows that the trim switches off are those of low value, which an acquisition seldom returns to | `scripts/wave5_A8_acquired_point_on_dead_row.py`: one step whose one candidate is the input of a row switched off: `func_count` 12 to 13, `Xn`, the live rows, the GP's rows and `y_max` unchanged, the row's `n_evals` 1 to 2, the row still off | leave, as W3-20 was ruled; the route through a grid is one more consequence of `integer_vars` for the documentation of W5-26. If the PI wants it closed after all: the duplicate scan over the live rows alone, or a hit on a row that is off switching it on | leave, as W3-20 |

## Part 2: the other findings

Rows P7-n are in `wave5_P7.md`, rows P9-n in `wave5_P9.md` and rows P2-n in
`wave5_P2.md`, which hold the supporting detail, the dating of each row, the
search for a recorded reason and the errors found in the reviewers' reports.
No row of this part changes a number of a run at the default options; where
a row is reached at the defaults at all, the column says with what effect.

### Slice P7

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W5-9 | P7 internal F2, P7 comparison F1, independently | `kl_div(samples=..., gauss_flag=True)` takes `np.mean(samples)`, one scalar over the `(N, D)` matrix, for the mean vector (`variational_posterior.py:1482`), where `vbmc_kldiv.m:63` has `mean(vp2,1)`; `kl_div_mvn` broadcasts the scalar into every coordinate | confirmed port discrepancy (P7-1); never matched (`fdb1e56d`, 2021-04-22). No reason recorded | no: no caller in the package, the documentation or the notebooks passes `samples=`; a user's call alone | P7-1: a posterior at `[10, -5, 2]` against 20 000 draws of itself returns `[56.1, 56.4]` for 3e-4 | fix (`axis=0`), with a test whose coordinates have different means | as proposed |
| W5-10 | P7 internal F7, P7 comparison F4, independently | The positivity check of `set_parameters(raw_flag=False)` slices `theta[-check_idx:]` with a `check_idx` that is already negative (`variational_posterior.py:1078-1090`), so it inspects the entries from `2K + D` on where the tail of length `2K + D` is meant: a negative `sigma` passes at `D = 2`, `K = 2`, a negative mean is refused at `D = 3`, `K = 4`, and whenever `optimize_mu` is off the slice is empty and nothing is checked (the verifier's addition) | Python-only defect (P7-2): `misc/rescale_params.m` has no such check; `8a4644bf`, 2021-03-17 | no: the package passes `raw_flag=True` everywhere; tests alone pass `False` | P7-2, the inspected entries printed for both reports' examples and for the sixteen flag combinations | fix the slice and its two edge cases, with tests that a negative `sigma` alone is refused and a vector with negative means is taken | as proposed |
| W5-11 | P7 internal F3 | `vp.mode()`, the documented default call, raises `AxisError` for every one-dimensional posterior (`.squeeze()` then `np.stack(..., axis=1)`, `variational_posterior.py:1269-1277`) | Python-only defect (P7-3a); `ba8116fa`, 2022-11-03 | no: nothing in the package calls `mode` | P7-3a, bounded and unbounded | fix, with a `D = 1` test | as proposed |
| W5-12 | P7 internal F4; third minor observation of P7 comparison | The box of the mode search is offset by an absolute `sqrt(eps)`, so bounds narrower than 3e-8 raise in SciPy; `x0` is clamped to the raw bounds while the optimizer gets the shrunken pair | the offset is shared by design (`vbmc_mode.m:39-40`); the clamp is a confirmed port discrepancy without effect (P7-3b): SciPy clips the start to the point that MATLAB's clamp produces | no | P7-3b | the clamp to the shrunken pair, one line, with W5-11; the offset left | as proposed |
| W5-13 | P7 internal F5 and F13; first minor observation of P7 comparison | The mode cache is returned whatever `n_opts` is passed and is cleared by `set_parameters` alone. `get_parameters` rescales `lambd`, `sigma` and the weights in place and leaves the cache, where `misc/rescale_params.m:39-40` removes `vp.mode` on every call; the comment at `variational_posterior.py:1021` says as much. PyVBMC also stores the mode always, MATLAB only when the second output is asked for | the `n_opts` part and direct assignment are shared by design (`vbmc_mode.m:18` ignores `nmax` alike); the missing clearing in `get_parameters` and the unconditional store are confirmed port discrepancies (P7-3c, P7-5). The rescaling has no numerical consequence: it preserves the distribution, and in a run it is a bitwise no-op, the posterior being in its canonical gauge after any `set_parameters` | the rescaling runs in every variational optimization, as a no-op; the cache is `None` throughout a run | P7-3c, P7-5, with the census of the ten MATLAB call sites of `get_vptheta` | fix: `get_parameters` clears the cache, and its docstring says that it normalizes the posterior; an explicit `n_opts` bypasses the cache | as proposed |
| W5-14 | P7 comparison F6 | The mode search is not MATLAB's: `ceil(sqrt(K))` optimizations, each from the best of 1e5 fresh draws, the component means joining the first, where `vbmc_mode.m:21-47` starts one at each of up to 20 component means and draws nothing. The draws come from the posterior's generator, which a run shares, so a call of `vp.mode()` moves every later draw of that stream, and the mode depends on the stream in its seventh digit | intentional difference, missing from the sheet (P7-3d): `ba8116fa` (2022-11-03, pull request 115, co-authored by the PI) repaired a first port that looped over the wrong axis, and the README of the module records the bug it answered. The stored MATLAB mode of the `K = 50` fixture is reproduced to 1e-4 | no | P7-3d | sheet entry, as the verifier words it. For the PI: whether `mode` should draw from a generator of its own, so that a call between two runs leaves the run's stream alone | sheet entry, and `mode` draws from a copy of the generator, as the diagnostic of the true posterior and the closing display line do |
| W5-15 | P7 internal F12 (a), (b), (d); the note at the end of P7 comparison's second answer | `sample` documents its index array as `N`-by-1 and returns `(N,)`, of float dtype for `K = 1`; `moments(orig_flag=True, cov_flag=True)` returns a 0-d covariance for `D = 1`; `sample`, and through it `kl_div(gauss_flag=False)` and `mtv`, refuse a float `N` such as `1e5` | confirmed port discrepancy, cosmetic, for the index array (MATLAB: an `N`-by-1 column); Python-only defects for the other two (P7-4a, b, d) | the 0-d covariance arises in every `D = 1` run, behind `sKL`, without effect (`kl_div_mvn` promotes it); the index array has no reader; every package caller passes an `int` | P7-4 | fix: the covariance through `np.atleast_2d`; a whole float accepted for `N`; the docstring of the index array brought to the code, `(N,)` of integer dtype in every branch, since a change of shape would reach users' scripts | as proposed |
| W5-16 | P7 internal F12 (c) | `pdf` takes a negative `df`, the product of univariate `t` densities, and `sample` raises NumPy's `shape < 0` for it | shared by design (P7-4c): `vbmc_pdf.m:87-102` has the family and `vbmc_rnd.m` does not | no: no caller uses a negative `df` | P7-4c | stricter interface: `sample` refuses a negative `df` with a message that says the family cannot be drawn from | as proposed |
| W5-17 | P7 internal F14, P7 comparison F8 | `kde_1d` is an independent implementation of Botev's estimator and no transcription of `shared/kde1d.m`: it bins a sample to the nearest grid point where `histc` takes the one at or below it, which puts MATLAB's estimate half a grid step low; it falls back on Scott's rule where MATLAB has `fminbnd`; it floors round-off at 0 where MATLAB has `eps`. Both sum to `n/(n-1)` on their grid, which the internal report takes for a Python defect | intentional difference, missing from the sheet (P7-6a to d); `522a901b` (2021-05-26, with the PI) replaced `scipy.stats.gaussian_kde`. The normalization is shared by design, and `mtv`, the one consumer, normalizes the density on both sides | every `mtv` call; the two normalized estimates are 2e-4 apart in total variation at `nkde = 2^13`, against 0.02 between either and the truth | P7-6, against a line-by-line transcription of `kde1d.m` | sheet entry, as the verifier words it; nothing to fix | as proposed |
| W5-18 | P7 comparison F5 | `vp.pdf(orig_flag=True)` on an integer array writes the transformed coordinates into a copy of integer dtype and evaluates the density at truncated coordinates (`variational_posterior.py:787`, `:801`); a list of integers does the same | Python-only defect (P7-8); from the first port of `pdf`. The factor of 21 in the report is its example's; the verifier's gives 1.25 | no: every internal caller passes floats, with `integer_vars` as without | P7-8 | fix: the input promoted to float, in one commit with the same trap in three priors (slice P9 below) | as proposed |
| W5-19 | P7 comparison F10 | `pdf` gives rows on or outside the original bounds a density of zero and transforms the others alone, where `vbmc_pdf.m:36-39` warps every row and returns NaN on a bound and a complex number outside it. A NaN coordinate takes the same branch and gets zero | intentional difference, missing from the sheet (P7-9); in the Python from the first port, pinned by `test_pdf_outside_bounds` | every `orig_flag=True` call under a bounded transform | P7-9 | sheet entry, which says that a NaN row gets zero; nothing to fix | as proposed |
| W5-20 | P7 comparison F11 | The sheet's entry on `qtrapz.m` says that the endpoint handling differs. `qtrapz(y)` is `sum(y) - 0.5*(y(1) + y(end))`, the trapezoid rule at unit spacing, and both sides multiply by the spacing afterwards | the sheet's entry is wrong (P7-10): the two agree bit for bit on short vectors and to 2e-14 on long ones (7e-15 at 1000 points, 2.1e-14 at 8192), by the order of the sum | n/a | P7-10 | the entry corrected, in the verifier's words | as proposed |
| W5-21 | second minor observation of P7 comparison | `VariationalPosterior(D, K, x0)` refuses a `(D, 1)` column: `x0.reshape(-1, 1)` at `:136` discards its result, and the docstring accepts "a single array" | Python-only defect (P7-11); `4c2d107c`, 2021-03-13 | no: the package passes a row | P7-11 | fix: the column accepted, as the dead line intends | as proposed |
| W5-22 | P7 internal F15; the headline of P7 comparison on the entropies | For `K = 1` without the Jacobian the two entropies return different weight gradients, 0 and `H - 1`; the weight gradient of the Monte Carlo entropy keeps a term that the others drop | not a defect (P7-7, P7-13): both conventions are MATLAB's (`ent/entlb_vbmc.m:45-47`, `ent/entmc_vbmc.m:96-101`), every production caller sets the Jacobian flag, under which both are zero, and the kept term estimates the derivative of the mixture's total mass, which is one for a weight and zero for the rest. The fix of `1b72896` is present: the Python agrees with a transcription of the fixed MATLAB block to 1.8e-15 and differs from the block before the fix by 1.45 | yes, every stochastic optimization, correctly | P7-7, P7-13 | none; recorded so that the third reader of O2 does not repeat the transcription | none |

### Slice P2

Every MATLAB line of this slice predates the port. The verifier read the
MATLAB counterparts, which the reviewer had not, and the records of wave 1 on
the same slice; the last paragraph of `wave5_P2.md` lists what wave 1 ruled
on, and no row below reopens one of those rulings. W5-24 bears on one: the
fix of wave 1 for `search_cache_frac > 0` made the first step work, and the
option still fails at the second.

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W5-23 | P2 internal F2 | With `acq_hedge=True`, `idx_acq` is never assigned (`active_sample.py:320-323`) and the first active-sampling step raises `UnboundLocalError` at `:408`, after the initial design. The option is declared, read, not registered as inert and not refused | confirmed port discrepancy, partly shared (P2-1): MATLAB assigns `idxAcq` only with more than one acquisition function, so its own default, one acquisition with `AcqHedge` on, fails alike; PyVBMC fails with any number. The branch has never assigned it in any revision (`4949c826`, 2021-11-04). The sheet records that the hedge is not ported, not that the option raises | no | P2-1 | stricter interface: the option refused at construction, as `noise_shaping` is, with a message that names the unported hedge; the option's description marked; the sheet's entry amended | as proposed |
| W5-24 | P2 internal F3; the verifier's P2-2e | The search cache. (a) The deletion of the acquired point from the search set is dead code; (b) the cache is written before it and keeps the acquired point in first place; (d) a row of the starting cache loses its cache index on the way through the search cache. (c) With `max_repeated_observations > 0` the training inputs join the search set and so the cache, and return from it without their repeat flag. (e) With the shipped fractions of the sieve, a `search_cache_frac` above 0.25 makes the second active-sampling step raise `ValueError`, the cache being empty at the first step and full from the second, with a message that prints as a tuple | (a), (b), (d) shared by design (P2-2a, b, d): `private/activesample_vbmc.m:230-242`, `:567`, `:633` are the same; (c) Python-only, a consequence of PyVBMC's own repeat mechanism (P2-2c); (e) Python-only guard (P2-2e), where MATLAB builds an oversized search set | no (`search_cache_frac = 0`); the dead deletion runs at every step without effect | P2-2a to e | stricter interface for (e): the fractions of the sieve checked at construction, and the message repaired; (c): the training rows left out of what is cached; (a), (b), (d) left, with a comment on the deletion | as proposed |
| W5-25 | P2 internal F4 | The Nelder-Mead branch calls SciPy without bounds, without `search_max_fun_evals` (SciPy's `200 D` takes its place) and with `tol=tol_fun`, which SciPy applies to the step as well as to the value (`active_sample.py:609-619`). On an unbounded variable the acquired point runs away (7e43 against a search box of 2.5); on a bounded one the acquisition's own mask stops it, against the report | confirmed port discrepancy in the three parts (P2-3a to c): MATLAB's `fmincon` has the bounds, `MaxFunEvals` and `TolFun` alone. The sheet records the substitution, and wave 1 fixed the same at `D = 1`. That the mask is inert on an unbounded variable, its thresholds being NaN, is shared by design (P2-3d; `misc/setupvars_vbmc.m:30-31`) | no (`search_optimizer = "cmaes"`) | P2-3 | either fix (the bounds, `maxfev` and `fatol` passed, with the first test of the branch) or remove the value: no default path has reached the branch since wave 1 took the one-dimensional search off it (`cb8a51d`; `v1.0.4` forced every `D = 1` run onto it), and no test runs it | remove the value and its code (2026-09-21): `search_optimizer` takes `"cmaes"` or `"none"`, anything else is refused at construction and in `load` with a message, and the branch, its sheet entry and its mention in the option's description go. To settle in the fix: `v1.0.4` wrote `"Nelder-Mead"` into the options of every one-dimensional run, so a saved one-dimensional object carries the value and `load` checks the stored options; the orchestrator proposes that `load` replaces it by `"cmaes"` for `D = 1`, where the option has no effect, and refuses it for `D > 1`, where a user set it, with a message that names `new_options`. The PI took the proposal for `load` on the same day |
| W5-26 | P2 internal F5 and the first-question answer | With `integer_vars` set, the initial design is evaluated off the grid (the provided `x0`, both designs, the cache rows the design takes), and every branch of the search delivers a point on it. The documents disagree: the FAQ says that integer parameters are not supported, the option's description says how to write the mask, the changelog says that such runs now complete, and none says that the first evaluations are off the grid | the design is shared by design (P2-4): `misc/initdesign_vbmc.m` never snaps, as W4-2 noted in passing. The documentation is a Python-only discrepancy (P2-5); MATLAB documents `IntegerVars` in one line | no; every run with `integer_vars`, from its first evaluation | P2-4, P2-5 | documentation: the FAQ answer, the option's description and the changelog's line say what `integer_vars` does and does not do. For the PI: whether an `x0` off the grid is refused when `integer_vars` is set, and whether the feature is documented as supported or as experimental | experimental, and documented as such in the FAQ, the option's description and the changelog: the search is snapped, the initial design and a provided `x0` are not, repeats on a grid cost evaluations. No refusal of an `x0` off the grid |
| W5-27 | P2 internal, last paragraph of the first-question answer | On a grid the search often returns an input that is in the training set; on a noiseless target the evaluation is pooled into the row, no training point is added, and the posterior is recomputed in full. Nothing on either side keeps an existing noiseless input out of the candidates | shared by design (P2-6) for the pooling; the update path is the deliberate difference the sheet already has ("The rank-one GP update is taken for a fresh observation, noisy or not") | no; routine with `integer_vars` | P2-6: two calls at the same input, `func_count` 12 to 14, no row added | none in the code; one sentence of the documentation of W5-26. A guard would be a change of the algorithm on both sides | none in the code; the sentence of W5-26 |
| W5-28 | P2 internal F7; wave 1 confirmed it as P2 F16 (b) and (c) and left it | The evaluations of the initial design are outside the `fun_time` timer, so the record of iteration 0 has none; the timers are not stopped in a `finally` | confirmed port discrepancy (P2-7a), diagnostic alone: nothing in the package reads `fun_time` of the main timer, and the replay leaves the timer record out of its verdict. The report's consequence for the missing `finally` does not follow (P2-7b): the timer is reset at the top of every iteration and the exception leaves `optimize` | every run, in a record that nothing reads | P2-7 | leave, as wave 1 did; or the three-line bracket if the PI wants the record complete | leave, as wave 1 did |
| W5-29 | P2 internal F8, F9, F10 | `recompute_var_post` saved and restored around a block that never writes it; two normalizations of the covariance and a literal 3 in branches of `_get_search_points`; the clip to the search box before the snap to the grid, and an expansion of the search bounds that ignores the integer step | not a defect of the port, each line being MATLAB's (P2-8, P2-9a, b, P2-10a, b): MATLAB's own assignment is commented out at `private/activesample_vbmc.m:58`; `cov(X)` and `cov(X_hpd,1)` stand at `:593`, `:596`, the 3 at `:618-619`; the clip precedes the snap at `:637`, `:219`; the comment that asks for other checks for integer variables is MATLAB's own, `:495`. The fallback box is unreachable, the search bounds being always finite | no | P2-8 to P2-10 | none | none |

### Slice P9

The MATLAB prior functions entered with `c387612` and `f3c05d6` (2022-10-26)
and have not changed; the Python priors entered whole with `2da98b5f`
(2023-02-08, pull request 128), and `v1.0.4` ships them as that commit wrote
them. Nothing in `vbmc.py` reads a prior's `support()`, `.a` or `.b`, no code
of a run calls `prior.sample`, and a run hands a prior float64 arrays alone,
so the rows of this slice are reached by a user's own call of a prior, but for W5-34, which a run with such a prior reaches, W5-37, which is construction, and W5-35, which is documentation.

| id | reports | statement | verdict (class) | fires at defaults? | how verified | proposed | PI |
|---|---|---|---|---|---|---|---|
| W5-30 | P9 internal F1, P9 comparison F1, independently | `SciPy.support()` reads `.a` and `.b` of the frozen distribution (`priors/scipy.py:69-70`), which SciPy sets from the standardized distribution and never moves by `loc` and `scale` (`rv_frozen.__init__`, read in SciPy 1.18.1); `Product` copies the box and prints it. `uniform(loc=2, scale=3)` reports `[0, 1]` and lives on `[2, 5]`; the unit-integral helper of the package's own tests returns 0.0 for it | Python-only defect (P9-1); `2da98b5f` | no: a user's call, `Product.__init__` and two tests | P9-1, with the list of every reader | fix (`distribution.support()`), with a shifted distribution in the unit-integral test | as proposed |
| W5-31 | P9 internal F2, P9 comparison F2, independently | `np.full_like(x, -np.inf)` in `trapezoidal.py:96`, `spline_trapezoidal.py:98` and `smooth_box.py:79` takes the dtype of the input. An integer input truncates the log density; in the two trapezoids the fill becomes the smallest integer, whose sum over the coordinates wraps: density one for an even number of coordinates out of the support, and `inf` for a mix (`[[20, 5, 5]]` in `D = 3`), which the reports do not have. The smooth box truncates and never wraps, against the internal report. A float32 input returns float32 | confirmed port discrepancy (P9-2): MATLAB's `-inf(size(x))` is a double whatever `x` is; never matched | no: every caller inside a run passes float64 (the verifier traced the five routes) | P9-2 | fix: the input made float64 once, in `Prior.log_pdf`, which also repairs the truncated value that `Product` inherits; a test of the output dtype for integer and float32 input. One commit with W5-18 | as proposed |
| W5-32 | P9 internal F3 | `UniformBox.log_pdf` marks a row as out of the support by `(x < a) \| (x > b)`, false at NaN, so a row with a NaN coordinate gets the full density, where the other three families give `-inf` and `SciPy` gives NaN | confirmed shared defect, latent (P9-3): `munifboxlogpdf.m:51` has the same two comparisons and returns the full density, and the other three MATLAB functions return `-Inf`, as a transcription run on the same row shows | no: inside a run the function logger refuses the point on the likelihood's value | P9-3 | for the PI, the change departing from MATLAB: `UniformBox` tests membership as its siblings do, so that a NaN coordinate has density zero in all four families; one line | as proposed: depart from MATLAB |
| W5-33 | P9 internal F4 | NaN and infinite constructor arguments pass the checks of every family, every comparison with NaN being false and an infinity satisfying a strict order. `UniformBox(lb, ub)` with the infinite hard bounds of an unbounded problem builds, has density zero everywhere, and the run stops at its first evaluation in the function logger. With a NaN pivot the rejection sampler accepts every proposal and draws uniformly on `[a, b]`, while the density is `-inf` outside `[v, b)` and NaN on it (the report has `-inf` everywhere) | confirmed shared defect, latent (P9-4): MATLAB's checks are the same comparisons, its two trapezoids have none, and a transcription of `mtrapezrnd` at a NaN pivot is uniform too | no: a user builds the prior | P9-4, the table of seven argument sets on both sides | stricter interface: each constructor refuses an argument that is not finite, with a message that names it. The strongest case of the slice | as proposed |
| W5-34 | P9 internal F5, P9 comparison F3, independently | `Product._log_pdf` calls `marginal.log_pdf(x[:, m], keepdims=False)` (`product.py:88`), and for a `UserFunction` marginal that is the user's own callable, which takes `x` alone: `TypeError`. A callable that tolerates the keyword is called once with the column of all `n` points, and its scalar return is given to every row. `Product.sample` handles such a marginal and has a test; the docstring of `prior`, the API page and `Product` itself offer a list of one-dimensional PyVBMC priors | Python-only defect (P9-5). The density half has raised since `2da98b5f`; at `v1.0.4` the sampling half worked, and `d02c517e` (2026-09-02) kept it working when the generator was threaded through, so the asymmetry is from 2023, against the comparison report | yes in a run whose `prior=` list holds a `UserFunction`: the object is built and the first evaluation of the target raises | P9-5, through `VBMC(..., prior=[UserFunction(...), UniformBox(0, 1)])` | fix: the marginal's callable applied row by row, and the existing test extended to `log_pdf`. No script of the last release can depend on the present behavior, which raises | as proposed |
| W5-35 | P9 internal F7, P9 comparison F6 and F7; one more from the verifier | Documented against the code, all on the API pages: `keepdims` of `Prior.log_pdf` and `Prior.pdf` described with shapes `(1, D)` and `(D,)` for `(n, 1)` and `(n,)`; the attributes of the four box families documented as `(1, D)` and stored as `(D,)`; `convert_to_prior` naming `sample_prior` twice, so that `log_prior` is not documented; the comment at `spline_trapezoidal.py:101` with the sign of the plateau term flipped against `msplinetrapezlogpdf.m:54` (the code is right); and the docstring of `log_prior` in `vbmc.py:150-154`, a sentence cut in the middle since 2023 | Python-only defects of the documentation; the comment a cosmetic port discrepancy (P9-6) | documentation alone | P9-6 | fix, one commit | as proposed |
| W5-36 | P9 internal F8, P9 comparison F5 | `Trapezoidal.log_pdf` warns of a division by zero at `x == a` where `SplineTrapezoidal` is silent, as MATLAB is for both; the spline's `np.seterr` pair has no `try`/`finally`, so an exception inside the loop leaves `divide="ignore"` installed. NumPy's error state is per thread, so the leak stays in the calling thread, against the comparison report | Python-only defect (P9-7) | no | P9-7 | fix: `np.errstate` in both classes | as proposed |
| W5-37 | P9 internal F9 | The guard that refuses a separate prior with a PyMC target leaves out `sample_prior`, and `sample_prior` given alone builds a prior without density, the target then being taken for the log joint | not a defect (P9-8): intended, and pinned by the two tests of the adapter written with the guard (`pyvbmc/testing/pymc/test_vbmc_binding.py:206-230`); `sample_prior` is documented as unused | at construction, without numerical effect | P9-8, with the two tests run in the PyMC environment | none | none |
| W5-38 | P9 internal F10 | `tile_inputs` compares its array arguments with each other and never with `size`, against its docstring, and `reshape` takes any array with the right number of elements: `UniformBox(np.zeros((2, 2)), np.ones((2, 2)), D=4)` builds | Python-only defect, latent (P9-9) | no | P9-9 | stricter interface: the check the docstring promises | as proposed |
| W5-39 | P9 internal F11 | A frozen univariate distribution with array-valued parameters, `norm(loc=[0, 10, 100])`, is taken for a one-dimensional `SciPy` prior: its sampler returns one draw from each of three normals and its density raises on reshape. `VBMC` refuses it by the dimension check unless the model has one dimension; `Product` builds with it and both its methods raise | Python-only defect, latent (P9-10) | no | P9-10 | stricter interface: refused at construction, with a message that names the parameter | as proposed |
| W5-40 | P9 comparison F4 | The two trapezoid classes refuse `a >= u`, `u >= v` and `v >= b`, where MATLAB's functions check nothing and compute `u == v`, the tent prior, and `v == b` correctly and normalized (checked by quadrature on a transcription); the spline computes `u == a` as well, and MATLAB's trapezoid gives NaN there. The two tests that pin the message use a strict violation and would pass a relaxed check, against the report | intentional difference, missing from the sheet (P9-11): documented in the class docstrings and tested, with no reason recorded for the limits that are legal | no | P9-11 | sheet entry. For the PI: whether `u == v` and `v == b` are accepted, which widens what is taken and can stop no script | keep the refusal; sheet entry |

## The first questions

**P7, where an underflowed log density reaches.** Both reports list the same
readers, and the P7 verifier checked the list on the present code, each
reader against its MATLAB counterpart (`wave5_P7.md`, "The check of the
first-question answers"). `pdf` takes the log of a linear sum, as
`vbmc_pdf.m:107-110` does, exact down to about -744. The three pointwise
acquisitions, the log acquisition and `fess` floor it at the log of the
smallest positive double as `acq/acqf_vbmc.m:7` and its siblings do, and both
sides discard the band between -708 and -744, so a log-sum-exp evaluation
would change nothing there while the floors stand. The mode search turns
`-inf` into `+inf` as `vbmc_mode.m` does and never starts there; `mtv` reads
no density; the entropies take the log of their own sums with no floor on
either side (W5-3); at the default `kl_gauss=True` the main loop reads
moments and no density. `kl_div(gauss_flag=False)` is the one reader that
neutralizes a zero, and it is W5-4. The one reader that the internal report
names as raising on an underflow, the proposal density of the importance
sampling, no longer does: since `799852a` (W4-7) such a row gets the log
weight `-inf`, which is what MATLAB's arithmetic produces. The orchestrator's
oracle check adds one observation: `make_oracle_fixtures.py --check` emits
the warning of a division of zero by zero at `variational_posterior.py:907`
(`dy = dy / y`), so a stored candidate set reaches the NaN gradient of the
log density that the internal report describes. No production caller asks
for that gradient but `mode`, which no package code calls.

**P7, the pair that `vp.sample` returns.** Every caller in the present
package unpacks it (the verifier lists 16 sites); the one that did not,
`fess`, is fixed (`dd3d1d3`, W4-5). Outside the remainder of the balanced
draw (W5-1) the sampler follows `vbmc_rnd.m` line by line and has the
mixture's law: the verifier's Kolmogorov-Smirnov checks at 2e5 draws pass
for `K = 1` and `K = 3`, Gaussian and heavy-tailed, and the scale of the
heavy-tailed draws is the multivariate `t`'s on both sides (`gamrnd` and
`Generator.gamma` both take shape and scale). One difference that neither
report records: with weights that do not sum to one MATLAB's `catrnd` draws
from the normalized weights and PyVBMC raises, in the plain draw through
`rng.choice` and in the balanced one through a broadcast error. It is
latent, `set_parameters` normalizing the weights.

**P9, whether every sampler draws from the density of its class.** Yes, on
both sides, and the P9 verifier re-derived it from the sources
(`wave5_P9.md`, "The check of the first-question answers"). The bounding
height of the two rejection samplers, the density at `(u + v)/2`, is the
maximum of the density for every valid argument (300 random argument sets per
class and three extreme ones); the mixture weights of the smooth-box sampler
are the exact masses of the two tails and the plateau; the normalizers
integrate to one with parameters that differ in every dimension; closed-form
distribution functions, checked against quadrature of the code's own
densities, pass Kolmogorov-Smirnov tests at 2e5 draws for PyVBMC's samplers
and for a transcription of MATLAB's, and the three p-values below 0.05 among
the fourteen are noise of the seed (40 seeds each, p-values uniform). The
four classes reproduce a transcription of the MATLAB log densities to 1e-13
with the same `-inf` sets. `_init_log_joint` combines prior and likelihood as
`lpostfun.m` does, the noisy branch included, in the caller's coordinates,
and no code of a run calls `prior.sample`. The exception is MATLAB's:
`msmoothboxrnd.m` for `D > 1` (below).

**P2, every path of a point with `integer_vars` set.** The P2 verifier
rebuilt the table of the report row by row from the code and from one check
per branch, and every row holds (`wave5_P2.md`, "The first-question
answer"). The search delivers a point on the grid in each of its four
branches, the acquisition snapping what it values and the snap being
idempotent; a chosen repeat is not snapped, on purpose, so that the logger
pools it; a cached starting point that the sieve draws is snapped, which is
W5-7. The whole initial design is evaluated off the grid, as in MATLAB, which
snaps the same three things (`private/activesample_vbmc.m:219`, `:248`,
`:325`) and never the design. The two objects that the report names as
transformers of the snap are one object on entry (`vbmc.py:1405-1408`).

## Defects on the MATLAB side

For `matlab_side_defects.md`, all from a reading of the source; nothing was
run in MATLAB, and the three verifier reports say for each what is read and
what is inferred from MATLAB's documented semantics. The first row is entry
1 of that file already, which the P9 verifier now measures on a
transcription; the twelve others are new.

| location | what the code does | consequence | PyVBMC |
|---|---|---|---|
| `shared/msmoothboxrnd.m:59`, `:66` | `a(idx)` and `b(idx)` with a logical column index into the `n`-by-`D` matrices of pivots read column 1 | For `D > 1` the Gaussian tails of every dimension sit at the pivots of the first: marginal means `[-4.50, -1.65, 5.85]` for `[-4.5, 0.5, 11.0]` on a transcription. `test/test_pdfs_vbmc.m:101` histograms the first column alone | not shared (`smooth_box.py:132`, `:138`) |
| `shared/mtrapezlogpdf.m:50`, `:57`, `:60` | `log(u-a)` is folded into the normalization and cancelled in every branch | NaN on the whole of `[u, b)` for `u == a`, and a sampler that degenerates to uniform there; the spline version has no such factor and is right | not shared; the class refuses `u == a` |
| `shared/mtrapezrnd.m:22`, `shared/msplinetrapezrnd.m:22` | `nargin < 3` for five declared inputs | The documented four-argument call errors at that line, on `isempty(n)` of an argument that was not given | no counterpart |
| `mtrapezlogpdf.m:40`, `msmoothboxlogpdf.m:27`, `:38`, `:47`, `munifboxrnd.m:33` | Error identifiers that name another function | cosmetic | no counterpart |
| `vbmc_mode.m:21-28`, `:35` | With more components than `nmax`, the starting means are ranked by the density at the transformed-space means read as original-space points; they are converted afterwards | With a transform that is not the identity and `K > 20`, the twenty starts are chosen at the wrong points (found by the P7 verifier) | not shared (`variational_posterior.py:1255-1258`) |
| `vbmc_mode.m:3-10` | The help text documents `VBMC_PDF(VP,ORIGFLAG)` for the signature `vbmc_mode(vp,nmax,origflag)` | A user who follows it passes `origflag` as `nmax` and polishes one basin | not shared |
| `vbmc_kldiv.m:44-52` | The analytical branch starts with `if origflag; error`, and `origflag` is 1 at `:34` | dead code; the documented `Ns = 0` mode does not exist | the behavior shared (`kl_div` raises for `N == 0`), the dead code not |
| `shared/kde1d.m:46-48` | `histc` credits a sample to the grid point at or below it | the estimate sits half a grid step low (W5-17) | not shared |
| `shared/kde1d.m:74`; `vbmc_rnd.m:84`, `:88` | an unused local; a redundant index | none | not shared |
| `private/activesample_vbmc.m:22-26`, `:153` | `idxAcq` is assigned by the hedge only with more than one acquisition function | `AcqHedge` with the default single acquisition errors at the first use (W5-23) | partly shared |
| `private/acqhedge_vbmc.m:55` | `hedge.g` without a semicolon | the hedge values printed at every iteration of a run with `AcqHedge` | no counterpart |
| `private/activesample_vbmc.m:627-633`, `:239` | fractions of the sieve that add up to more than the number of random points give a search set longer than its index vector | an index past the end for a winner in the surplus | not shared: PyVBMC raises at the second step (W5-24) |
| `private/activesample_vbmc.m:382-386` | the target call sits in a `try` with an empty `catch` | an error of the target is swallowed, and the next lines fail on variables that were never set | not shared |

Shared, for the list of shared defects there: the raw determinants of
`shared/mvnkl.m` (W5-2); `0 * (-Inf)` in both entropies at a weight of zero
(W5-3); the stored value of a cached starting point recorded at the clipped
and snapped candidate (W5-7); the full density of `munifboxlogpdf.m` at a
NaN coordinate (W5-32); NaN and infinite arguments passing the checks of the
prior functions, with a rejection sampler that turns uniform at a NaN pivot
(W5-33). The pooling into a row that is not live (W5-8) is there already.
`vbmc_pdf.m:112-124`, which returns an uncorrected gradient beside a
corrected density, is the basis of a sheet entry already; the P7 verifier
adds that no MATLAB caller reaches it.

## Sheet entries

New, in the verifiers' wording where they give one: the mode search
(W5-14); the mask of `pdf` on and outside the bounds, with the zero it gives
a NaN row (W5-19); `kde_1d` as a substituted implementation, with the
binning, the fallback, the floor and the shared `n/(n-1)` (W5-17); the
argument checks of the two trapezoid classes (W5-40). Corrected: `qtrapz`
(W5-20); the hedge, which is not inert and raises (W5-23);
`search_optimizer`, whose Nelder-Mead value goes (W5-25). Lines: weights
that do not sum to one in `sample`, under the entry on the sampler or the
settled non-differences; under the settled non-differences, the two
conventions of the weight gradient at `K = 1` (W5-22), and the lines of
`active_sample.py` that are MATLAB's own and read as defects (W5-29). Entries
that depend on a ruling: `UniformBox` at a NaN coordinate (W5-32), the
rounding sites (W5-6).

## Test notes worth acting on

The verifiers opened every test that the reviewers cite; their reports have
the full lists. What matters:

- The tests of three defects that two reviewers found independently are
  blind by construction: `test_sample_balance_*` with equal weights (W5-1),
  the two `samples=` tests of `kl_div` with one mean for every coordinate or
  `D = 1` (W5-9), `test_set_parameters_not_raw_negative_error` with a vector
  negative throughout (W5-10). So are the mode tests, in `D = 2` under an
  identity transform (W5-11, W5-12). Each fix brings the test that sees it.
- No sampler of a prior has a test of its distribution; the one check of
  moments touches `UniformBox(0, 1)` and a smooth box that is a standard
  normal. MATLAB's `test_pdfs_vbmc.m` histograms all four, the first column
  alone, which is why it misses the defect of `msmoothboxrnd.m`. The
  closed-form distribution functions of `scripts/wave5_P9_12_*.py` are a
  ready reference for a Kolmogorov-Smirnov test per family and per column.
- `test_vbmc_init.py` holds twelve `np.isclose(...)` statements without
  `assert` (the verifier walked the syntax tree: lines 835 to 1022); seven of
  them are the claim that the log joint is the likelihood plus the prior,
  which is asserted in one place, line 903, for the noiseless path with
  `prior=`.
- `prepare_gp_for_acq` of the oracles transcribes `active_sample.py:328-363`,
  so the eight acquisition oracles pin that block against a copy; the copy
  should call the production lines or say that it is a transcription.
- No test runs `acq_hedge=True`, a search cache through `active_sample`, the
  initial design with an integer variable, or `fun_time`; they are the
  surface of W5-23, W5-24, W5-26 and W5-28.
- `test_kldiv_mvn.py` has identity covariances alone (W5-2); `test_get_hpd`
  has no tie (W5-6); `test_kde1d.py` has no reference of any kind, and the
  transcription of `kde1d.m` in `scripts/wave5_P7_6_kde1d.py` is one.
- Refuted: the internal P7 report's note that no test has `eta` and the
  weights in step under the softmax Jacobian (`test_entmc_vbmc_matlab` does,
  against MATLAB); the comparison P9 report's statement that the tests of the
  trapezoids' argument check would not survive a relaxed check.

## Errors in the reports

The verifiers list them in full: eleven in the internal P7 report and nine in
the comparison one, fourteen across the two P9 reports, seven in the P2
report. Those that change a row: the reach of the rounding of `get_hpd`,
where the internal P7 report has ties at `N = 5, 15, 25` for the default
fraction, which has none (W5-6); three properties that the internal P7
report reads as Python's and that are MATLAB's (the `n/(n-1)` of `kde_1d`,
the absolute offset of the mode's box, the weight gradient at `K = 1`); the
warp, which the P2 report has resetting the search box from the training
inputs and which maps the box itself, so that the clip of W5-7 is out of
reach (the orchestrator's correction, from `scripts/wave5_A7_*`, not the
verifier's); the mask of the acquisition, which stops the Nelder-Mead search on a
bounded variable (W5-25); the consequence drawn from the missing `finally`
of the timers, which does not follow (W5-28); the dating of the
`UserFunction` asymmetry of `Product`, three years older than the comparison
P9 report has it (W5-34); the smooth box, which never wraps an integer fill,
and the mixed case that gives an infinite density (W5-31); the density at a
NaN pivot, NaN on `[v, b)` and not `-inf` everywhere (W5-33); NumPy's error
state, which is per thread (W5-36). Several line citations into `vbmc.py`,
`active_importance_sampling.py` and `test_vbmc_init.py` have drifted since
`f873556`; the citations into the files of the three slices held but for two: the
rounding of `get_hpd`, which `027e972` moved from line 36 to line 38, and
the `N < 1` return of `sample` in the internal P7 report, one line off.

## Fix commits

On `dev-port-review` after `a2104e6`, made on 2026-09-21. Three Opus agents
on worktrees cut at `a2104e6` made thirty (reports `../fixes/wave5_agent_A.md`,
`wave5_agent_B.md` and `wave5_agent_C.md`): twenty-six fixes, each with a
test written against the contract and seen to fail on the code before it,
two commits of documentation (`0fe371b`, `05e4934`) and two of tests alone;
`1ea2f6d` and `0fe371b` cover two rows each. The orchestrator reviewed each
diff and cherry-picked the commits, which applied without a conflict, the
sampler (`23d962a`, agent A's) last, and made two more: `a093f2e`, the last
row of the table, and `725e35e`, the changelog with the requirement of W5-5
in the FAQ and on the API page of the priors.

| row | commit | |
|---|---|---|
| W5-25 | `4cc09cc` | the `"Nelder-Mead"` value of `search_optimizer` and its branch removed; the option checked at construction and in `load`, which replaces a stored `"Nelder-Mead"` by `"cmaes"` for a problem of one dimension and refuses it otherwise |
| W5-23 | `06fd70e` | `acq_hedge=True` refused at construction and in `load` |
| W5-24 | `a25fc73` | the five fractions of the sieve checked at construction and in `load`; the message of the guard in `_get_search_points`; the training rows left out of the search cache; a comment on the deletion that nothing reads |
| W5-7 | `052c442` | the stored value of a cached starting point reused only for a candidate that is the cached point; the target called otherwise, and the row leaves the cache either way |
| W5-6 | `8e2977a` | `pyvbmc/stats/_rounding.py`, a half away from zero from the exact fractional part, at the twelve sites and in `_real2int`, whose results are bit for bit what they were |
| W5-26, W5-27 | `0fe371b` | the FAQ and the description of `integer_vars`: experimental, what is snapped and what is not, the repeats on a grid |
| W5-9 | `7749ddb` | the mean of the samples per coordinate in `kl_div(samples=...)` |
| W5-10 | `dfc3334` | the positivity check of `set_parameters(raw_flag=False)` on the entries that hold `sigma`, `lambd` and the weights, for the sixteen flag combinations |
| W5-11, W5-12 | `1ea2f6d` | the mode of a one-dimensional posterior; the starting point clamped to the box the search runs in |
| W5-13 | `eab21fa` | `get_parameters` clears the stored mode and documents its normalization; an explicit `n_opts` runs the search |
| W5-14 | `13cb245` | the candidates of the mode search drawn from a copy of the posterior with a copy of its generator |
| W5-15 | `6c38beb` | a 1-by-1 covariance from `moments` for one parameter; a whole float taken for `N`; the index array of `sample` flat and of integer dtype in every branch, the docstring brought to it |
| W5-16 | `a5e66d2` | a negative `df` refused by `sample` with the reason |
| W5-18 | `d40e9fd` | the working copy of `pdf` in float64 |
| W5-21 | `b503ae6` | a single starting point given as a column |
| W5-4 | `25a4bb4` | the guards of `kl_div(gauss_flag=False)` as MATLAB has them; the original-space density from the difference of the logs on the rows where the quotient fails, every other row bit for bit |
| W5-2 | `a37945a` | the log determinants of `kl_div_mvn` from `slogdet` |
| W5-30 | `24b07f4` | the support of a `SciPy` prior from the frozen distribution's `support()` |
| W5-31 | `5c05678` | a prior reads its input as float64, in `Prior.log_pdf` |
| W5-32 | `da12c24` | `UniformBox` tests membership, so a NaN coordinate has density zero |
| W5-33 | `4a43731` | the four box constructors refuse an argument that is not finite |
| W5-34 | `fa3df99` | `Product` applies a `UserFunction` marginal to one point at a time |
| W5-38 | `70b8f2d` | `tile_inputs` checks an array argument against `size` |
| W5-39 | `474b273` | a frozen univariate distribution with array-valued parameters refused |
| W5-36 | `a652cd8` | `np.errstate` in both trapezoids |
| W5-35 | `05e4934` | the documented shapes, `log_prior` in `convert_to_prior`, the sign in the spline's comment, the cut sentence of `log_prior` in `vbmc.py` |
| W5-5 | `a613fe3` | a prior whose support does not cover the hard bounds refused at construction; the docstring of `prior` |
| test notes | `48518c8`, `614cdfd` | a seeded Kolmogorov-Smirnov test of each box sampler against its distribution function, per column in `D = 3`, with a check that it sees a uniform sampler; the twelve statements of `test_vbmc_init.py` given their `assert` (two of them compared a pair with a pair and unpack it) |
| W5-1 | `23d962a` | `np.sum` in the remainder weights of the balanced draw, last and alone |
| test note | `a093f2e` | the docstring of `prepare_gp_for_acq` says that it transcribes the lines of `active_sample` and which oracle runs them (orchestrator) |

The sheet has an entry, new or amended, for rows W5-2, W5-5 with W5-33,
W5-38, W5-39 and W5-40 (the checks of the priors), W5-7, W5-14, W5-15 with
W5-16 (the interface of `sample`), W5-17, W5-19 with W5-4 and W5-18 (`pdf`),
W5-20, W5-23, W5-24, W5-25, W5-30 with W5-31 and W5-34 (the prior API) and
W5-32; W5-1, W5-22 and W5-29 are lines under its settled non-differences;
W5-3, W5-8, W5-28 and W5-37 are left as they are. W5-18 and W5-31, ruled as
one commit, are two, one by each agent whose file it was. The requirement of W5-5 is
in the FAQ answer on priors and on the API page of the priors as well
(orchestrator, with the records).

## Gates

The baseline is `wave3_doublecheck/after_095c29e0.npz`, the four seeded runs
of `scripts/wave2_fixpass_gate_runs.py` on the code of before the pass (the
commits between `095c29e` and `a2104e6` touch tests, records and
`Product._generic`).

Gate 1, on `a37945a`, the seventeen commits of agents C and A without the
sampler: the four runs bit for bit the baseline (92 arrays), and the exact
oracle check, 11 of 11. The log determinants (W5-2) move not even the last
digit of `sKL` in these runs. The twelve commits of agent B, which no run
without a prior reaches, came after it.

Gate 2, on `23d962a`, the thirty commits: the four runs bit for bit the
record that `scripts/wave5_A1b_gate_runs_corrected_sampler.py` made during
the verification with the corrected sampler and nothing else (92 arrays, 0
differ), so the pass moves what W5-1 moves and no more: `sKL` and `r_index`
of all four runs, by up to 2.5e-5 and 5.9e-4, and the whole of
`rosenbrock_D2_noisy` (30 iterations and 150 evaluations for 32 and 155). The
exact oracle check, 11 of 11 with nothing re-baselined: the oracles reach the
balanced draw, in `active_sample_step` on six states with `K > 1`, and the
acquired points are the same.

On `a093f2e`, the head of the code of the pass: the whole default suite, 1843
passed and 58 skipped with no reruns; with the Torch environment, the S-VBMC,
variational-posterior, prior and statistics directories and the modules of
the pass, 886 passed and 1 skipped, the seeded references of S-VBMC among
them, which the balanced draw of `SVBMC.sample(balance_flag=True)` could have
moved and does not; with the PyMC environment, the adapter's tests, 107
passed. Each environment printed `pyvbmc.__file__` of this checkout. The logs
and the two records of the seeded runs are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`).

W5-1 moves default trajectories, one of the four seeded runs from iteration
13 on (its 71st evaluation is the first that differs), with no statistical
effect (a component's count
changes by less than about one draw among thousands), so no sweep on
benchmark targets was run; the golden references and the run pools, which
describe the code of before the moving fixes of waves 1 to 3, are regenerated
once, after the review's remaining fixes, as the plan has it. No fix of the
pass moves an oracle reference. The changelog has the pass's lines, under
Changed, Fixed and Removed and in its "Upgrading from 1.0.4" list.

## Found during the fix pass

- Release 1.0.4 wrote `"Nelder-Mead"` into the options of every
  one-dimensional run, so a saved one-dimensional object carries the value
  that W5-25 removes, and `load` checks the stored options: `load` replaces
  it by `"cmaes"` there, where the option has no effect, before `new_options`
  is applied, so that an explicit value still wins and an explicit
  `"Nelder-Mead"` is refused. `test_vbmc_optimize.py` passed the value in a
  one-dimensional test and no longer does.
- One tie of W5-6 is within reach of the shipped options: with more starting
  points than the initial design takes, the starting cache is not empty at an
  active-sampling step, the number of random points can leave a remainder of
  two on division by four, and the quarter shares of the sieve then differ by
  one candidate from the old rounding (agent C). No gate has such a state.
- `scripts/wave5_P2_2_search_cache.py` and
  `wave5_P2_11_search_cache_frac_second_step.py` build `VBMC` objects with a
  `search_cache_frac` of 0.5 or 1.0 beside the shipped fractions, which
  construction refuses since `a25fc73`: they are records of the code of
  `831acef` and no longer run against the package.
- A density that does overflow a double is still infinite after W5-4: on the
  probit unit box with an SD of 30 in the inference space a tenth of the
  draws have one, and the corrected guard is what keeps `kl_div` finite
  there. The quotient and its recovery run under `np.errstate`, so such a
  density no longer emits NumPy's overflow warning (agent A).
- `mode(n_opts=k)` runs the search and stores its result, which a later
  `mode()` returns: the ruling's bypass is of the read (agent A).
- `_rebuild_log_joint`, which `load(new_options={"vectorized_target": ...})`
  reaches, calls `_init_log_joint` and so runs the check of W5-5 on a saved
  object (agent B). No shipped example is refused by that check. The FAQ's
  list of `uniform(loc=..., scale=...)` marginals needs W5-30, which came
  first, and the slack of `42c3942`: while the check compared exactly it
  refused that list for about a quarter of decimal bounds ("The
  independent check of the pass").
- The two statements of `test_vbmc_init.py` that compared the noisy log
  joint of the package with the test's own compared a pair with a pair; with
  their `assert` they unpack the value and the noise (agent B).

## The independent check of the pass

On the PI's instruction (`/doublecheck`, 2026-09-21) the pass was read by five
fresh Opus reviewers, read-only, on `0bf7963`: R1 the commits on the
variational posterior, R2 the priors, R3 the active sampling and the options,
with the reach of the pass, R4 this ledger against its sources, R5 the
user-facing texts and the records. They returned 13, 13, 6, 15 and 21
findings. Their reports and scripts are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`). `dev-port-review-w2check`, the fixes of the
independent check of wave 2, was merged as a fast-forward (`b1bab4d`) before
the round, and the round's worktrees were cut there.

Three findings had to be fixed.

- The check of W5-5 compared exactly (R2-1, R5-1, independently). A support
  computed as `loc + scale` ends a few units in the last place inside the
  bound it was built from (`uniform(loc=-20, scale=20.2).support()` ends at
  0.1999999999999993), so `VBMC` refused the FAQ's own
  `[uniform(loc=low, scale=high - low) for low, high in zip(LB, UB)]` for
  about a quarter of decimal bounds (R2: 20 000 random and 40 000 decimal
  pairs). The statement under "Found during the fix pass" that the FAQ's list
  "is taken because W5-30 came first" was therefore false when written
  (R2-8) and is corrected there; it holds since `42c3942`.
  `../fixes/wave5_agent_B.md`, a raw report, says the same and carries a
  flag in its header.
- "With one variable a bounded scalar search is used whatever the option
  holds" is false for `search_optimizer = "none"`, under which no local
  search runs (R5-2). The sentence came from the orchestrator's brief and
  stood in the option's description, the refusal of `"Nelder-Mead"`, the
  changelog and the sheet.
- W5-6 said that the rounding fires in no run at the defaults, against this
  ledger's own last section (R4-1); and the tie it describes did more than
  move a count (R3-1, R3-5): with a half going away from zero, three
  quarter-shares of two points are three points, and the sieve raised its
  `ValueError` part-way through a run, with the shipped fractions where two
  points were left to draw, and with a quarter for the search cache wherever
  the starting cache left 2 or 3 points modulo 4. The check of the fractions
  at construction (W5-24) did not close that. Release 1.0.4 has the same
  guard and rounds a half to even, which gives those two points no share, so
  there the error needs fractions that add up to one.

PI rulings (2026-09-21): a slack of `1e-9 * (ub - lb)` in the support check,
none at an infinite bound; the sieve caps each source at what is left, the
check at construction staying; `mode(n_opts=k)` neither reads nor writes the
stored mode; the sample count of `to_arviz` (part of R1-3) waits for slice
N2; W5-7 stays; the should-fix items are taken, with these optional ones: a
0-d `N` in `sample` (R1-8), the last `else` of the search chain naming the
option, the dead `optim_state["hedge"]` line, the `new_options` hint in the
two other refusals, the comment above the two `np.delete` calls and the
unseeded test (all R3-6), a scalar support broadcast (R2-9), `a` and `b` of
`SciPy` as float64 (R2-10), the helper in `test_gp_training_policy.py`
(R5-18), the dead `jacobian == 0` clause (R1-5).

| Finding | Commit | What it does |
|---|---|---|
| R2-1, R5-1, R2-9 | `42c3942` | the slack, where both hard bounds are finite; a support given as a scalar is broadcast to the bounds |
| R2-2, R2-10 | `cc909c8` | `SciPy` and `Product` read their support from what they hold at every call (`_support_box`), so a prior unpickled from a file of 1.0.4 reports the interval its distribution lives on and `load(new_options={"vectorized_target": ...})` no longer refuses it; `a` and `b` are read-only float64 properties |
| R2-3 | `a0bb6c2` | a `UserFunction` marginal may return a float or an array of one element; more values raise a message that names the marginal and the row |
| R2-5 | `b79a127` | `tile_inputs` judges an argument by its squeezed shape, so a row given with `squeeze=False` builds as it did in 1.0.4 |
| R2-7, R5-17 | `3468704` | `_check_finite`, a private name for a helper that is neither exported nor documented |
| R2-6, R5-16 | `02fc9e6` | the docstrings that `05e4934` left wrong, the `Returns` header of `Prior.pdf` among them |
| R5-2, R3-6 | `d66ddf2` | the description of `search_optimizer` and the refusal of `"Nelder-Mead"` say what the two values do; the last `else` of the search chain names the option and its values |
| R3-1 | `94638fe` | `capped_share`: each source takes its rounded share or what the sources before it left, in the order in which they are drawn, the variational posterior drawing the rest; the guard that raised is gone. Where no cap binds the draws are what they were: the agent compared `_get_search_points` before and after over five option sets at one and three variables, 24 of 24 arrays and the state of the generator after the draws byte for byte |
| R3-5, R5-18 | `3b07758` | a test of the tie that the shipped fractions reach (a starting cache that leaves ten points: 3, 3, 3 and 1); `_matlab_n_init` of `test_gp_training_policy.py` uses the helper |
| R3-4, R3-6 | `7d5c29a` | a test that a cached point on the integer grid keeps its stored value, with no target call; the unseeded `_state_with_gp` call is seeded |
| R1-1, R1-2, R5-14, R5-15 | `a188c31` | `mode(n_opts=k)` leaves the stored mode alone; the `Returns` section of `mode` parses; the docstrings of `mode` and `get_parameters` say what holds |
| R1-5, R1-8, R1-10, R3-6 | `bc0ab84` | `sample` takes any scalar that holds a whole number; the dead clause and the dead line are gone; the `x0` docstring covers the single element and the rows past the `K`-th; the two refusals say how a saved run is loaded; the comment above the `np.delete` calls is true |
| R5-8, R2-4 | `7500dba` | the descriptions of the five fractions give the range and the sum; that of `integer_vars` says that a prior given with `prior=` must cover the half-integer hard bounds |

The reports are `../fixes/wave5_check_agent_D.md` (priors; the first six
rows) and `wave5_check_agent_E.md`. Every fix came with a test seen to fail
on the code before it, but for `3b07758` and `7d5c29a`, which are tests of
code that was right, the two removals of dead code in `bc0ab84` and the
docstrings. The orchestrator reviewed the diffs and cherry-picked the
thirteen commits, which applied without a conflict.

The texts are the orchestrator's. `851fd17`: the FAQ on integer parameters
(R3-2, R3-3, R5-7, R2-4), the API page of the priors (R5-20) and the README
of the variational posterior (R1-4, R5-6). `7411919`: the corrections of R4
to this ledger (its fifteen findings, each in the row or the sentence it
names; R1-13 is R4-11) and of R5 to `matlab_side_defects.md` (R5-9,
R5-10). `09d0310`: the
changelog (R5-3, R5-4, R5-5, R5-12, R5-19, R1-9, R2-5, and the sentences of
the round). `f86d373`: the sheet (R5-2, R5-11) with the cap of the sieve and
the slack, entry 35 of the MATLAB-side list, which the rounded shares
extend to fractions that add up to at most one, and the citations carried
to the present lines. The changelog is written against 1.0.4, so it does
not follow the two agents' sentences where those describe the branch: the
sieve's error with the shipped fractions, and the row that `tile_inputs`
refused, never reached a release.

The round was then read by one more fresh Opus reviewer, read-only (R6), on
`f86d373` with this section as it stood; its report and scripts are kept
with those of R1 to R5. Nothing had to be fixed. It re-derived, from the code
and with probes of its own, that the sieve returns the number of points it
is asked for in every case it could build (a search cache of 0, 1, 2, 3 and
20 rows against a share of 4, the key absent, one to nine points, every
fraction forced to one) and draws what it drew wherever no cap binds; that a
point in the gap of the slack is never evaluated; that a property shadows
the `a` and `b` left in the dictionary of an older pickle, in `support()`,
`str` and `repr`, and that `pickle`, `dill` and `deepcopy` take the two
classes; that nothing tracked assigns to `prior.a` or `prior.b`; the
changelog's statements on 1.0.4, at the tag; and the sets of `Nrnd` at which
MATLAB's rounded shares overshoot (entry 35), up to 500. Five findings to
fix and six optional ones:

- R6-1: the in-flight entry of the worklog described a state that the
  cherry-picks had ended. It is replaced by the entry of the round.
- R6-2: `../fixes/wave5_agent_C.md` says twice that one variable is searched
  by a bounded scalar method whatever the option holds. A raw report: it
  carries a flag in its header, as agent B's does.
- R6-3: the sheet's entry on the fractions said without condition that the
  search cache keeps the acquired point in first place, against the comment
  that `bc0ab84` corrected in the code. Qualified: where a cache is kept,
  and unless the point is a training input.
- R6-4: the argument below on the other rounding sites gave the burn-in as
  `400 / sqrt(N)`. The code multiplies the thinning by the count of samples
  after it is rounded, so the burn-in is a whole number and its rounding
  meets no tie; corrected below, and the changelog no longer lists the
  burn-in among the quantities a tie can move.
- R6-5: R1-13 was carried under R4-11 alone. Named above.
- Taken of the optional ones: the notes of `_get_search_points` say that the
  search cache gives at most the rows it holds (R6-6), and the docstring of
  the support check names both things that keep a run off a hard bound, the
  acquisition search's `tol_bound_x` and the thousandth of the range of
  `_effective_bounds` (R6-9), in `5c4fc87`; the range of `D` that the
  argument below assumes (R6-10).
- Left: `a` and `b` of `SciPy` and `Product` will render twice on the API
  page of the priors, from the `Attributes` block of the class and as
  properties (R6-7); the `new_options` hint of the fractions' refusal names
  `search_cache_frac` whichever fraction is at fault (R6-8); the changelog's
  2e-11 is 1.67e-11 at one digit, as its 8e-9 is 8.27e-9 (R6-11).

Gates, once for the merge of `dev-port-review-w2check` and the round
together, on `f86d373`. The four seeded runs of
`scripts/wave2_fixpass_gate_runs.py` are bit for bit the record after the
pass, `after_pass_23d962a1.npz` (92 arrays, 0 differ), as they were on the
merged head `b1bab4d` before the round: neither moves a default trajectory.
The exact oracle check, 11 of 11 with nothing re-baselined. The whole
default suite, 1961 passed and 58 skipped with no reruns. With the Torch
environment, the S-VBMC, variational-posterior, prior and statistics
directories and the modules of the round, 1009 passed and 1 skipped; with
the PyMC environment, the adapter's tests, 107 passed. Each environment
printed `pyvbmc.__file__` of this checkout. None of the four seeded runs
is given a prior with `prior=`, so the slack and the support read at every
call rest on the tests of the suite; a seeded run with a prior is a
candidate for the gate of the release.
The two docstrings of `5c4fc87` came after these gates; the modules of the
two files were run again on it, 262 passed. The logs and the record of
the seeded runs are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`).

Two fixes followed, on the PI's word (2026-09-21), made by the orchestrator,
each with a test seen to fail on the code before it.

| Finding | Commit | What it does |
|---|---|---|
| R1-3, the part no ruling had covered | `6d492a2` | `moments` converted its count with `int()` before it called `sample`, so `vp.moments(N=2.5)` and `vp.kl_div(gauss_flag=True, N=2.5)`, which computes its moments through it, truncated the count where `sample`, `mtv` and `kl_div(gauss_flag=False)` refuse it. The count goes to `sample` as it is given; a whole number draws what it drew. As 1.0.4 had it (`variational_posterior.py:788` there) |
| noted by agent E | `6dcd027` | `cache_frac`, the share of the whole search set that the starting cache gives, is checked to lie in `[0, 1]` with the five fractions, at construction and in `load`, outside their sum. Nothing checked it, in 1.0.4 (`active_sample.py:699-703` there) or in MATLAB (`private/activesample_vbmc.m:552-554`): above one, with a starting cache of more rows than the search has candidates, the search set came out larger than asked; a negative one made `N_cache` negative, which as the end of a slice takes all but that many rows of the cache (read in the code, not run; MATLAB's `randperm` is given the negative count there, and what it does with it was not looked up) |

The changelog and the sheet have both. Gates on `6dcd027`: the four seeded
runs bit for bit the record after the pass once more (92 arrays, 0 differ),
the exact oracle check 11 of 11, and the modules of the files touched, 131 and
347 passed. No caller in the package hands `moments` or `kl_div` a count that
is not whole (`Nkl = int(1e5)`, `vbmc.py:1317`; `1e6` in the diagnostics).

Left as they are:

- R1-3, the part of slice N2, by the ruling: `to_arviz` refuses a whole
  float such as `1e3`, which `sample` takes.
- R1-6: the `0/0` row of `pdf`, where the transformed density and the
  Jacobian both underflow, is NaN in both branches. R1-7: `np.errstate`
  around the quotient hides NumPy's warning for a density that does overflow.
  R1-11: `mode()` returns the stored array itself, so a caller that writes
  into it writes into the store. R1-12: half of
  `test_kl_div_samples_mean_is_taken_per_coordinate` recomputes the
  implementation.
- R2-11: a 0-d array argument of a box prior is refused with a message that
  names a shape the caller did not pass. R2-12: only `UniformBox` points an
  unbounded problem to another prior. R2-13: the rejection samplers behind
  the seeded Kolmogorov-Smirnov test may draw differently on another
  platform (smallest p 0.268 against 1e-3 here).
- R5-13: the changelog's sentence on the balanced draws does not name
  `SVBMC.sample(balance_flag=True)`; S-VBMC is in no release and its seeded
  references did not move. R5-21: the sentence of `AGENTS.md` on
  `__deepcopy__` is true of `__deepcopy__` and does not say that `mode()`
  copies the generator on purpose.
- Noted by the agents of the round: a support of the wrong length raises
  NumPy's broadcast error and not the check's own message (D); a pickle of a
  `SciPy` or `Product` prior written from now on carries no `a` and `b` in
  its dictionary, so release 1.0.4 cannot read them from it (D);
  `tile_inputs` has no `Returns` section (D).

W5-6, completed. The row records that of the five sites outside the search
the orchestrator had looked at one. None of them has a tie at the shipped
options. The number of hyperparameter samples is `80 / sqrt(N)`, whose
half-integers are at `N = 1024` and `N = 25600`, and sampling stops at
`N >= 200 + 10 D`, which is at most 400 for the twenty variables PyVBMC is
meant for (the two meet at `D = 83`). The burn-in is the thinning, 5,
times that number after it is rounded, or times 3
(`gaussian_process_train.py:113`, `:607`, `:614`): a whole number, which
its rounding leaves alone. The number of starting
points of the hyperparameter search is `64 + 960 u^3` with `u = m / (10 r)`
for whole `m` and `r`, since the budget and `fun_eval_start` are multiples
of ten; a half-integer needs `240 t^3 / r^3` odd with `m = 5 t`, that is
`4 + 3 v(t) = 3 v(r)` for the powers of two in `t` and `r`, which no whole
numbers satisfy, and the nearest value is further from a half than rounding
can carry it. The bonus of components is `round(2)`. So the tie of the
starting cache is the one within reach of the shipped options, as the row
says.
