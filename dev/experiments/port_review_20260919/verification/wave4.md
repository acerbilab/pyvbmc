# Wave 4 verification

The findings of wave 4 (slices P3 and P4; reports under `../reviews/`),
verified on 2026-09-20 on `dev-port-review` at `f873556`, the revision the
reviewers read, gpyreg at `9e70e6b`, against MATLAB VBMC at `396d649`. Part 1
holds the three findings that stop a run or change what it computes; part 2
holds the others. Two sources feed the ledger: the orchestrator's own checks,
for part 1 and for row W4-10, with the scripts named in the rows; and two
read-only Opus verifiers, one per slice, whose raw reports are `wave4_P3.md`
(rows P3-n) and `wave4_P4.md` (rows P4-n). The scripts are under `scripts/`;
the logs of the orchestrator's are kept on the machine that ran them
(`dev/scripts/runs/LOCAL.md`, "Port correctness review"). Every row was
searched for a recorded reason for the difference in the code comments, the
commit and pull-request messages, the porting log, the documentation, `dev/`
and the tests; a row says so where one was found. Every MATLAB line involved
predates the Python line, so no row is a faithful port of an older MATLAB.

The PI ruled before the verification, on the dispositions that the
orchestrator recommended with the wave's report ("go ahead with verification
and fixes", 2026-09-20). The last column gives the disposition carried out;
where the verification changed what had been recommended, the row says so.

## The first question: uncertainty level 1

The four reports agree on the noise of the training points, of a candidate
and of a newly acquired point, and both verifiers checked the answers
against the sources on both sides (`wave4_P3.md` and `wave4_P4.md`, "The
first-question answers"): at uncertainty level 1 (`uncertainty_handling=True`
without `specify_target_noise`) the noise of a candidate point, of the
training points and of a newly acquired point is what MATLAB computes, and it
is consistent with the GP's own noise model.

- The function logger records SD 1 for an evaluation and pools repeats by
  precision, so `S = 1/sqrt(n_evals)`; the GP is fitted on the pooled
  variance `S**2`, and `active_sample.py:326-351` evaluates the noise
  function at `S**2 * n_evals`, which is 1 at every training point whatever
  its number of evaluations. Hence `sn2_new = exp(2*h0) + exp(h1)`, the
  variance of one new observation, as `private/activesample_vbmc.m:159-174`
  has it. On a state at level 1 the port agrees bit for bit with a
  transcription of the MATLAB lines in `sn2_new`, in the length scale and in
  the four pointwise acquisitions, and to 2e-14 in VIQR and IMIQR once `u` is
  the same (W4-8). The P3 verifier repeated the comparison with a multiplier
  `exp(h1) = 2.3`, because the reviewer's state had a multiplier of 1, at
  which a port that dropped the multiplier would pass as well
  (`scripts/wave4_P3_10_level1_vs_matlab.py`).
- The noise of a newly acquired point, `s2new = S[idx_new]**2`, is MATLAB's
  at each level. The path differs (W4-11): gpyreg extends the factors by
  rank one where `gplite/gplite_post.m:76-79` recomputes every posterior as
  soon as a noise variance is given. The two agree to about 1e-15 at levels
  1 and 2 and to 3e-13 at level 0, also for a new point whose noise lies
  below every existing one (`scripts/wave4_P4_5_rank_one_update.py`).
- Inside the importance sampling one quantity depends on the observation
  noise, the target density of IMIQR's MCMC step, and it is MATLAB's
  (W4-13). Here the reports part: the P4 internal report calls it a
  discrepancy, and the P3 comparison report says the port gets the noise
  of that prediction right, where the prediction adds the constant term
  alone on both sides.

Four properties that the answers raise are in both implementations as their
authors wrote them, and the algorithm papers ask for nothing else (W4-13,
W4-14, W4-15, W4-17). The ledger calls such a row "shared by design", a
class beside the plan's six: not a defect of either side, and no difference
between them. No test and no stored state below the GP fit is at
level 1; the fix pass adds a test of `sn2_new` at that level.

## Part 1: findings that stop a run or change what it computes

| id | reports | statement | verdict (class) | fires at defaults? | how verified | disposition |
|---|---|---|---|---|---|---|
| W4-1 | P4 comparison F1, P4 internal F1, independently | The MCMC step of `active_importance_sampling`, which IMIQR alone reaches, picks the starting point of each slice-sampling chain among the proposal samples of the first step by importance sampling-resampling. `active_importance_sampling.py:240-247` builds the log weights as a column, `(Na, 1)`, and takes `np.amax(ln_weights, axis=1)`, the maximum along the axis of length one, so every weight is `exp(0)` and the draw is uniform. `private/activeimportancesampling_vbmc.m:206-208` builds a row and takes `max(lnw,[],2)`, the maximum over the samples, and draws in proportion to `exp(lnw - max)`. The guard of the next line, written for "no sample has weight", raises `ValueError("Invalid value.")` as soon as one sample has weight zero | confirmed port discrepancy; never matched. The Python lines are those of the file's first commit, `e38351ab` (2022-05-23, "Noisy likelihoods", #79); the MATLAB lines date from `2a076c9` (2020-06-16). No reason recorded. The comparison report names `70325a34` as the file's first commit, which is its second | no: the noisy default is `AcqFcnVIQR`, which takes the branch without MCMC. Every active-sampling step of a run with `AcqFcnIMIQR` and `active_importance_sampling_mcmc_samples > 0` (the default, 100), once per GP hyperparameter sample | `scripts/wave4_A1_isr_weights.py` runs the function on the stored noisy state of the oracles with a generator that records the `p` of the draw: 8 chains, `p = 1/200` in each; under MATLAB's rule 67 to 101 of the 200 samples have weight below 1e-6 and the effective sample size is 1.5 to 42. What it costs, `scripts/wave4_A1b_isr_start_cost.py`: the IMIQR acquisition at 256 candidates, per hyperparameter sample, from the function as it is and from a copy with MATLAB's rule, same seed, against the acquisition from 20 000 importance samples of the first step, every vector centred over the candidates. On the stored state (`D = 2`, 100 pairs) the RMS error is 0.0187 nats with the uniform start and 0.0150 with the weighted one, paired difference +0.0037 (SE 0.0016); the regret of the candidate each estimate ranks best, measured on the reference, is 0.0089 with the uniform start and 0.0132 with the weighted one, paired difference -0.0043 (SE 0.0021). Both numbers sit below the noise of either estimate on that state, where the reference itself varies over the candidates by 0.001 to 0.014 nats. On the state a noisy run on `corr_D6` ends in after 130 evaluations the RMS error is 0.0008 for both. So the uniform start leaves the acquisition within its Monte Carlo error on the two states, and on the six-dimensional one the burn-in of 50 absorbs it altogether: the comparison reviewer's reading holds, the internal reviewer's does not | fix: the maximum over the samples, and the guard for the case that no sample has weight. Moves the `acq_AcqFcnIMIQR` oracle and no other |
| W4-2 | P3 internal F1 | `AbstractAcqFcn._real2int` indexes its input as `X[:, integer_vars]` (`abstract_acq_fcn.py:279-285`), and `active_sample.py:624` hands it the one-dimensional result of the local search (`cma.fmin`, the bounded scalar search and Nelder-Mead all return one), when that search improves on the sieve's best candidate. `ParameterTransformer.inverse` keeps the dimension of its input, so the call raises `IndexError`. MATLAB passes a row vector, which `xtemp(:,integervars)` indexes (`private/activesample_vbmc.m:325`, `misc/real2int_vbmc.m:7`) | confirmed port discrepancy; never worked. The call is from `4949c826` (2021-11-04, #30), the function from `9d1b6590` (2021-08-11, #25); both are in `v1.0.4` as they are today | no (`integer_vars` is empty). Every run that sets `integer_vars` with a search optimizer, the default `cmaes` included | `scripts/wave4_A2_integer_vars_run.py`: the bare call raises on a one-dimensional point and snaps a two-dimensional one; a seeded run on a two-dimensional target whose first variable is an integer raises `IndexError` at `active_sample.py:624` after the 10 evaluations of its initial design, and the same run with `search_optimizer="none"` completes, 31 of its 40 evaluated points on the integer grid (the initial design is not snapped, in MATLAB either). The one `active_sample` test with an integer variable, `test_repeated_observation_is_exact_with_integer_vars`, sets `search_optimizer` to `none`, so it never reaches the call; the internal report says that no test runs `active_sample` with an integer variable | fix, with a test through `active_sample` and a search optimizer |
| W4-3 | P3 internal F2; P3 comparison, minor observation 5 (P3-9e) | `string_to_acq` (`acquisition_functions/utilities.py:17-22`), which builds the acquisition named by a string of `options["search_acq_fcn"]`, splits the arguments on `","` and each on `"="` and strips trailing whitespace alone. A keyword whose name has a space before or after it raises `TypeError` (`"AcqFcnVIQR(0.9, loss='iqr_reduction')"`, `"AcqFcnVIQR( quantile=0.9)"`, `"AcqFcnVIQR(quantile = 0.9)"`), wider than the second and later arguments the report names; a value that holds `","` raises `SyntaxError`; and a value that holds `"="` is dropped without a message, the default taking its place | Python-only defect (MATLAB has `str2func`); `70325a34` (2022-06-02, #80). No reason recorded | no (the defaults are objects). A string with a keyword argument written with a space, inside the run, after the initial design | `scripts/wave4_A3_string_to_acq.py` (ten argument forms); `scripts/wave4_P3_9_minor_observations.py` for the values with `","` and `"="` | fix: the arguments are parsed as Python parses a call, and what cannot be parsed raises |

## Part 2: the other findings

Rows P3-n are in `wave4_P3.md` and rows P4-n in `wave4_P4.md`, which hold
the supporting detail, the dating of each row and the errors found in the
reviewers' reports.

| id | reports | statement | verdict (class) | fires at defaults? | how verified | disposition |
|---|---|---|---|---|---|---|
| W4-4 | P3 comparison F3 | `_real2int` rounds with `np.around`, a half to even, where `misc/real2int_vbmc.m:7` has `round`, a half away from zero | confirmed port discrepancy (P3-2); never matched (`9d1b6590`). A tie is reachable, against the report: the hard bounds of an integer variable sit at half-integers, so the midpoint of a box with an even number of levels is a half-integer, the probit transform maps it to 0 within rounding and back to the half-integer exactly, and a starting point there enters the sieve through the cache | no; with `integer_vars`, on an exact half-integer alone | P3-2 | fix, with W4-2; `test_real2int`, whose four comparisons lack their `assert` and whose input is the tie, asserts MATLAB's values |
| W4-5 | P4 comparison F3, P4 internal F4 | `fess(vp, gp, X)` with a number of samples for `X`, its documented default, raises: `vp.sample` returns a pair and the pair is used as the array | confirmed port discrepancy (P4-2), without a caller: MATLAB's one call of that form, the gate of `private/activesample_vbmc.m:440`, is not ported (`active_sample.py:736-737`; the sheet's entry on inert options) | no | P4-2 | fix |
| W4-6 | P4 comparison F4, P4 internal F3 | The branch that `acq_info["mcmc_importance_sampling"]` requests (`active_importance_sampling.py:86-126`) hands gpyreg's `SliceSampler` the matrix of walkers that MATLAB's ensemble sampler takes, and the sampler refuses a two-dimensional starting point. With one chain, its `burn_in = 0` and its last `Na` rows are consecutive draws, where MATLAB's are one draw of each of `Na` walkers. Its weights are `ln_y` alone, as MATLAB's are (`private/activeimportancesampling_vbmc.m:94-103`, outside the MCMC block; `:81` there discards the sampler's log-densities) | confirmed port discrepancy in its first two parts, shared by design in the third (P4-3). The branch has never run in any revision: gpyreg's check (`2bf7f10`, 2021-06-08) is older than the branch (`e38351ab`). Decision D5 of `dev/plans/latent-bug-fixes.md` is a decision not to delete hooks; that a user's acquisition can request this one is the sheet's statement, and it is false | no; a user's acquisition that sets the flag, when the fractional effective sample size falls below 0.9 | P4-3 | stricter interface: an acquisition that sets the flag is refused, at construction where `search_acq_fcn` holds the object and in `active_importance_sampling` for any other caller, with a message that says the branch is not ported; the body is removed and `active_importance_sampling_fess_thresh`, which it alone read, is registered as inert; the sheet's entry is rewritten |
| W4-7 | P4 comparison F5 | `active_sample_proposal_pdf` raises `ValueError("Invalid value.")` for a point at which every component of the proposal has density zero, where MATLAB's arithmetic gives NaN and `private/activeimportancesampling_vbmc.m:148` turns the weight into `-Inf` | confirmed port discrepancy, latent (P4-4): the proposal's own draws cannot reach such a point (20 000 draws stay far above the underflow; the density is zero from about 50 units off the posterior) | no | P4-4 | fix: the point gets weight zero |
| W4-8 | P3 comparison F2 | `u = norm.ppf(quantile)`, 0.6744897..., where `acq/acqviqr_vbmc.m:4` and `acq/acqimiqr_vbmc.m:4` have the literal `0.6745; % norminv(0.75)`. It is the whole difference between the port and MATLAB's formulas in VIQR (1.7e-5) and IMIQR (8e-5), with the same best candidate | intentional difference, missing from the sheet (P3-1). The port had the literal until `70325a34` added the `quantile` argument; the reason is in the body of pull request 80 | yes, every noisy run | P3-1 | keep; sheet entry |
| W4-9 | P3 comparison F4, P4 comparison F2, P4 internal F2 | `renormalize_weights` subtracts one log-sum-exp over the whole array of log weights; MATLAB stores them raw. The two readers of the weights are IMIQR and the `iqr_reduction` loss of VIQR; everything that is added to or compared with an acquisition value afterwards is unchanged by a constant (the variance regularization, the masks, the ranking, `f_val_optim < f_val_old`, the tolerances of the search) | confirmed port discrepancy without effect on any decision (P4-1a); from `70325a34` (2022-06-02, #80), the file's second commit. No reason recorded; the docstring of `AcqFcnVIQR` documents the constant, as `log N_s` where it is `log(Ns_gp * Na)` | yes; moves no candidate | P4-1a | keep; sheet entry; the docstring's constant corrected (`76ed8e9`) |
| W4-10 | P4 internal F2 (its argument on the MCMC branch) | In the MCMC branch each row of weights belongs to one GP hyperparameter sample and to a chain drawn from that sample's own unnormalized density, so the row carries the factor `1/Z_s`, `Z_s` being the sample's integrated interquantile range before the new point. The report takes the rows to weigh unevenly in the average over samples, and blames the joint normalization | not a defect, and shared (P4-1b): MATLAB's raw rows carry the same factor. Each row estimates `N * IQR_s(after the new point) / IQR_s(before)`, the relative reduction for its sample, and the acquisition averages those. The total weights of the rows do spread, over 17 nats on the stored noisy state (medians 17.0 with PyVBMC's uniform start and 17.5 with MATLAB's weighted one, 20 seeds, against 0.15 for the first step alone), yet each row's contribution to the acquisition sits at the same level within 0.05 nats: a chain that stays where the GP is uncertain has small weights `1/(2 sinh(u s))` and large terms `2 sinh(u s_pred)`. The verifier's statement that the spread is the noise of a chain started uniformly does not hold | yes for IMIQR with `Ns_gp > 1` | `scripts/wave4_A1c_isr_row_totals.py`; P4-1b for the derivation | none; a line under the sheet's settled non-differences |
| W4-11 | P4 comparison F6 | The sheet's entry on the rank-one GP update says that on a noisy target the noise variance goes into the rank-one update on both sides. `gplite/gplite_post.m:76-79` sets `update1 = false` whenever `s2` is given, so at levels 1 and 2 MATLAB recomputes every posterior and PyVBMC extends the factors by rank one | intentional difference whose description of MATLAB does not hold (P4-5). gpyreg's extension is the generalization of MATLAB's to unequal noise and equals a rebuild to about 1e-15. After a retried factorization the rank-one path keeps the stored `sn2_mult` where a rebuild derives it again; at the conditioning that produces a retry the predictions still agree to 2.4e-9 and their variances to 1.5e-8 | yes, every acquired point of a noisy run that is not a repeat; the numbers are the same | P4-5, `scripts/wave4_P4_5b_sn2_mult.py` | keep; the sheet's entry corrected |
| W4-12 | P4 comparison F7; P3 comparison, minor observation 4 | PyVBMC draws the search candidates before the importance samples, MATLAB after (`private/activesample_vbmc.m:208-218`); neither routine changes the state the other reads | confirmed port discrepancy of the random stream alone (P4-6; the P3 verifier, who reads a difference of the stream as no defect, has it as "not a defect", P3-9d); since `e38351ab`, not `c4035ff8` as the report has it | yes, every noisy step | P4-6 | keep; a line in the sheet's entry on randomness |
| W4-13 | P4 internal F5, P3 internal F3; part (c) of both P4 answers | The target density of IMIQR's MCMC step, `is_log_full`, predicts with `add_noise=True` and no `s2` at the test point, so at levels 1 and 2 the constant term alone is added, while the resampling weights and every other density use the latent variance | shared by design (P4-7): `log_isbasefun` takes the first two outputs of `gplite_pred`, the prediction with noise, and the stored MATLAB output of `compare_MATLAB/log_isbasefun.npz` matches the value with noise to 1.4e-4 and the latent one to 4.8e-2. The estimator is unbiased under either choice, the weights using the sampler's own log-densities. At the shipped noise floor the added variance is about 1e-5 | yes for IMIQR | P4-7 | none; a line under the sheet's settled non-differences |
| W4-14 | P3 internal F4 | `sn2_new` is the mean over the GP hyperparameter samples and enters each sample's own formula. At level 1, where the fitted multiplier carries the noise, the per-sample values spread (1.36 to 3.70 around 1.91 in a fit of 26 points) | shared by design (P3-3): `private/activesample_vbmc.m:174` and the three MATLAB acquisitions do the same, and §C.1 of the paper's appendix defines the noise of a candidate as one function of the input, that of its nearest training point | yes, every noisy run | P3-3 | none; a line under the settled non-differences. A noise per hyperparameter sample would be a change of the algorithm |
| W4-15 | P3 internal F5, P4 internal F7 | `sn2_new` leaves out the `sn2_mult` of a retried Cholesky factorization, which the prediction with noise and the GP update apply | shared by design (P3-4): `private/activesample_vbmc.m:172` calls the noise function without it, and `gplite_pred.m:121` and `gplite_post.m:208` apply it where gpyreg does. Unreachable at the shipped noise floor, under which the factorized matrix is the identity plus a positive semi-definite one | no | P3-4 | none |
| W4-16 | P4 internal F8 | `S**2 * n_evals` at a repeated input whose observations have different SDs (level 2) is the harmonic mean of their variances, the smallest of the plausible summaries | confirmed shared defect, latent (P4-8): MATLAB has the same expression and the same pooling | no; repeats with unequal SDs need `max_repeated_observations > 0`, duplicate rows of `x0` or `precomputed_evaluations` | P4-8 | leave; listed as shared in `matlab_side_defects.md` |
| W4-17 | P4 internal, adjacent observation | The hard bounds of the noise multiplier are `[1e-3, 1e3]` on the variance | shared by design (P4-9; `gplite/gplite_noisefun.m:114-115`). The report's upper limit for the noise SD does not follow, the constant term being free up to the range of the values; the lower limit, an SD of about 0.03 at level 1, holds on both sides | every level-1 fit | P4-9 | none. With the wave's report the orchestrator had proposed a sentence in the description of `uncertainty_handling` on the noise range that level 1 can fit; it is not written, the upper limit not being one and a noise SD below 0.03 being no case for level 1 |
| W4-18 | P3 internal F8 | `AcqFcnVIQR` and `AcqFcnIMIQR` assign `acq_info` without calling the base constructor, so `compute_var_log_joint`, which the base class documents, is absent | Python-only defect, latent (P3-5): every reader uses `.get`. MATLAB's `acqInfo` of the two lacks the field too | no | P3-5 | fix |
| W4-19 | P3 internal F9 | `quantile` is unchecked. With VIQR a value of 0.5 or less, or of 1, gives acquisitions that are all NaN or all `-realmax`, and the run takes the first candidate of every sieve without a message; with IMIQR the importance sampling raises an error that names nothing (the report has both fail silently) | Python-only defect (P3-6); `70325a34`. The paper takes the quantile in (0.5, 1) | no | P3-6 | stricter interface: both constructors refuse a quantile outside (0.5, 1) |
| W4-20 | P3 internal F7; P3 comparison F1 read the same lines and found no defect | In `_log_iqr_reduction` the clamped `s_pred` beside the unclamped `tau2 / (s_a + s_pred)` | not a defect (P3-7): `tau2 <= f_s2_a * s2 / (s2 + sn2)`, a factor below 1 wherever the noise is positive, which the noise floor ensures; of 409 600 entries on the stored noisy state none exceeds a ratio of 0.87 | no | P3-7 | none |
| W4-21 | all four reports | The sheet's entry on the removed VIQR losses says `AcqFcnVIQR` offers `loss="iqr"` alone. `iqr_reduction` is shipped, documented, tested and in the changelog, as the message of `fd9c7e8e` says; `var_reduction`, `sd_reduction` and `AcqFcnEIG` are gone | the sheet's entry is wrong (P3-8) | no | P3-8 | the entry corrected |
| W4-22 | P3 comparison, minor observations 1 to 3 | `np.maximum(acq, -realmax)` lets a NaN through where MATLAB's two-argument `max` returns `-realmax`; the log sums guard a row that is all `-inf`, where MATLAB reaches the same `-realmax` through a NaN; the quadrature branch of `acqwrapper_vbmc.m` for `vp.delta` is not ported | the first is a confirmed port discrepancy, latent (P3-9a): no shipped acquisition returns a NaN once W4-19 refuses the quantiles that did, and MATLAB's answer makes the point with the NaN the most attractive; the second is not a defect (P3-9b); the third is on the sheet (P3-9c) | no | P3-9a, b, c | leave; a line in the sheet for the first |

## Defects on the MATLAB side

Entries 21 to 24 of `matlab_side_defects.md` and one item of its shared
list, all from a reading of the source; nothing was run in MATLAB. The
reviewers listed six; the first, `misc/funlogger_vbmc.m:244`, is entry 15
already.

| location | what the code does | consequence | PyVBMC |
|---|---|---|---|
| `acq/acqviqr_vbmc.m:25-28` | The `'islogf'` branch adds `vp`, its second argument, which `log_isbasefun` passes as `[]` for VIQR (`importance_sampling_vp` is false) | The log-density of the MCMC target comes out empty. Reachable through `mcmc_importance_sampling` alone, which no MATLAB acquisition sets | not shared: `AcqFcnVIQR.is_log_full` returns the added term |
| `private/activeimportancesampling_vbmc.m:208-214`, with the local `catrnd` | The walker loop sets the weight of each drawn sample to zero; once every remaining weight is zero `catrnd` returns index 1 (no division by zero, against the report) | The remaining walkers all start at the first proposal sample. With `2(D+1)` walkers and the effective sample sizes of 1.5 to 2 that W4-1 measured, it is within reach | no counterpart: PyVBMC runs one chain |
| `gplite/gplite_post.m:76-79`, with `private/activesample_vbmc.m:481-484` | A requested rank-one update is turned into a full recomputation whenever `s2` is given, without a message | None for the numbers. The call site reads as if noisy observations were taken by rank one, which misled the sheet (W4-11) | not shared (W4-11) |
| `acq/acqviqr_vbmc.m:107-108`, `acq/acqimiqr_vbmc.m:93-94` | A row of the log sum that is all `-Inf` gives NaN, which `acqwrapper_vbmc.m:47` turns into `-realmax` | None: the value is the right one | the same value, reached without the NaN |
| `private/activesample_vbmc.m:165`, with the pooling of `misc/funlogger_vbmc.m` | `S.^2 .* nevals` at a repeated input with unequal SDs is the harmonic mean of the variances | W4-16 | shared |

The report's sixth, the size check of `misc/fess_vbmc.m:21-23`, does not
hold as stated: the check is dead whenever the second argument is a GP,
because the means are computed at the very points it counts, and live for a
matrix of means, on the path with a number of samples as on the other.

## Sheet entries

New: `u` from the quantile (W4-8); the normalized log weights (W4-9); the
NaN that `np.maximum` lets through (W4-22). Corrected: the removed VIQR
losses (W4-21); the rank-one GP update (W4-11); the dormant MCMC branch,
which becomes an entry on a branch that is not ported (W4-6). Brought to the
state of the code: `_real2int` (W4-2, W4-4), the slice sampler of IMIQR's
MCMC step (W4-1), the inert options (W4-6), and the line citations into the
files the pass changed and into `vbmc.py`. Lines: the order of the draws
(W4-12), in the entry on randomness; and, under the settled non-differences,
the noise in the target of IMIQR's MCMC step (W4-13), the candidate's noise
averaged over the hyperparameter samples, with `sn2_mult` (W4-14, W4-15),
and the relative reduction that each row of the MCMC branch estimates
(W4-10).

## Test notes worth acting on

Both verifiers opened every test the reviewers cited; the full lists are in
their reports. What the tests held at `f873556`, and what the pass did:

- `test_real2int` held four comparisons without `assert`, on the input at
  which NumPy's rounding and MATLAB's differ. They assert MATLAB's values
  (W4-4).
- `test_active_importance_sampling` asserts the shapes of what the function
  returns and no value; W4-1's test is the first of a value of the MCMC
  step. `test_fess` passed arrays alone; a test calls it with a number of
  samples (W4-5). The two `test_acq_info` of VIQR and IMIQR listed the keys
  they expected, and four sibling modules asserted
  `not acq_info.get("compute_var_log_joint")`, which a missing key passes;
  all seven assert the key and its value (W4-18). No test passed a quantile
  outside its range; both modules do (W4-19).
- `test_acq_log_f` built a VIQR with `importance_sampling_vp` set, computed
  its value, and overwrote it two lines later with that of a fresh instance;
  the dead lines are gone, and a comment says that what the test compares
  with MATLAB is assembled in the test.
- No test and no stored state below the GP fit was at uncertainty level 1:
  the one noisy oracle state is at level 2 with `S = 1` and single
  evaluations, the one case in which the two levels hand the noise function
  the same `s2`. The pass adds a test of `sn2_new` at level 1 with a
  repeated row. A stored oracle state at level 1 needs a level-1 variant of
  a benchmark target and is left for the PI to decide.

Not acted on:

- The smoothing of the variational posterior into the proposal
  (`active_importance_sampling.py`, the four groups of scales) has no
  reference of any kind: `test_active_sample_proposal_pdf` passes the
  posterior unsmoothed.
- `test_acq_fcn_viqr.py` and its IMIQR twin pin `u` by the implementation's
  own definition, and pass with MATLAB's literal as well.
- Both `test_complex__call__` scenarios build a GP with one hyperparameter
  sample, so the average of IMIQR and VIQR over several samples, the
  configuration of W4-10, is tested by the oracles alone.
- `_regularization.check_variance_regularization` builds its expected value
  with the expression of the code, so a wrong penalty would pass, and the
  equivalence of the penalty in the log and the plain acquisitions is not
  checked.

## Errors in the reports

The verifiers' reports list them in full, twelve for each slice. Those that
change a row: the file's first commit (W4-1); the test with an integer
variable (W4-2); the reach of the parser's defect (W4-3); the tie that the
comparison reviewer thought unreachable, and MATLAB's bound check, which
does not require half-integers (`matlab_side_defects.md`, entry 10) (W4-4);
IMIQR's failure with a bad quantile, which is loud (W4-19); and the cause of
the rows' spread, which both the P4 internal reviewer and the P4 verifier
misplace (W4-10). One that changes no row is worth keeping in mind for the
P3 comparison report's numbers: it ascribes a residual of 1.4e-14 in
`AcqFcnLog` to the order of a `log` and a `max`, which cannot produce one
(the verifier's own transcription is bit-identical there).

## Fix commits

On `dev-port-review` after `f873556`, made on 2026-09-20. One Opus agent on
a worktree made ten (report `../fixes/wave4_agent.md`), which the
orchestrator reviewed and cherry-picked; the orchestrator made the others,
among them the five that follow the independent check of the pass (below).

| row | commit | |
|---|---|---|
| W4-2 | `dbfe895` | `_real2int` takes a single point as well as an array of points |
| W4-4 | `3c088ab`, `70ce067` | a half away from zero; the rounding from the exact fractional part, by the orchestrator: `floor(abs(x) + 0.5)` sends the largest number below a half to one |
| W4-3 | `e5e6015`, `601555e` | `string_to_acq` parses the string as Python parses a call, literal arguments alone; its docstring says which name it looks up |
| W4-18 | `4554135` | VIQR and IMIQR call the base constructor |
| W4-19 | `0477a2f`, `c14a6b9` | the quantile is refused outside (0.5, 1); it is taken as a real scalar of any type, or an array with one element, by one check that both constructors share |
| W4-5 | `dd3d1d3` | `fess` with a number of samples |
| W4-7 | `799852a` | weight zero where the proposal has density zero |
| W4-6 | `1c23fda`, `44c1973` | the flag refused at construction and in `active_importance_sampling`, the branch removed, `active_importance_sampling_fess_thresh` inert; `search_acq_fcn` must be a list of acquisition objects or of strings, and its description says which strings are read |
| W4-1 | `53ad16a`, `2dc98ce` | the maximum over the samples, and the guard for the case that no sample has weight; the reference of `acq_AcqFcnIMIQR` re-baselined |
| W4-9 | `76ed8e9` | the constant in the docstring of `AcqFcnVIQR` |
| test notes | `9d5d682`, `315919a` | `sn2_new` at uncertainty level 1 with a repeated row, through one `active_sample` step, and the dead lines of `test_acq_log_f`; the state of the integer-variable search test seeded and built on the module's state helper |

`601555e` also corrects the descriptions of the two unused fESS thresholds
and the comment over the draw of the chain's starting point. Rows W4-8,
W4-9, W4-11, W4-12, W4-21 and W4-22 are sheet entries, W4-10, W4-13 and
W4-14 lines under the sheet's settled non-differences, and W4-15, W4-16,
W4-17 and W4-20 are left as they are. The sheet's entries on `_real2int`, on
the slice sampler of IMIQR's MCMC step and on the inert options were brought
to the state of the code as well.

## Gates

Before the first cherry-pick, on `f873556`: the four seeded runs of
`scripts/wave2_fixpass_gate_runs.py` recorded, and the exact oracle check,
11 of 11. After the eleven commits that change the package or its tests, on
`70ce067`: the four runs bit for bit against the record (92 arrays), and the
exact oracle check with one oracle moved and no other, `acq_AcqFcnIMIQR` on
`rosenbrock_D2_noise1_viqr`, the one stored state on which it is computed
(largest difference of a value 0.24), which W4-1 predicts: the chain of
every GP hyperparameter sample starts elsewhere. That oracle is also the
only one that reaches the proposal density of W4-7, VIQR sampling from the
variational posterior alone, so the two changes cannot be told apart there;
`scripts/wave4_A7_proposal_pdf_bitwise.py` tells them apart: the first step
of the importance sampling on the same state, MCMC switched off, returns the
same arrays bit for bit with the module of `f873556` and with the fixed one,
over five seeds. The reference was re-baselined from the stored state with
the generator's targeted mode, single-threaded, the reason in the fixture's
metadata (`2dc98ce`); the exact check then passes on 11 of 11.

On `2dc98ce`: the whole default suite, 1662 passed and 58 skipped with no
reruns; with the Torch environment, the S-VBMC and variational-posterior
directories and the modules of the pass, 752 passed and 1 skipped; with the
PyMC environment, the adapter's tests, 107 passed. No fix of the pass moves a
default trajectory, so no sweep on benchmark targets was run; W4-1 changes
runs with `AcqFcnIMIQR`, whose acquisition it leaves within its Monte Carlo
error on the two states measured (part 1, W4-1). The seeded run of
`scripts/wave4_A2_integer_vars_run.py` with an integer variable and the
default search optimizer, which raised after its initial design before the
pass, completes after it: 40 evaluations, 31 of them on the integer grid (9
of the 10 points of the initial design, which is not snapped, lie off it).
The changelog has the pass's lines, under Changed and Fixed and in its
"Upgrading from 1.0.4" list.

The five commits that follow the independent check (`76ed8e9` to `601555e`)
ran the focused tests of what they touch: the quantile, `acq_info` and
rounding tests (39 passed), `test_vbmc_init.py` with `test_options.py` (194),
the two tests of `315919a` three times over, and `test_options.py` with
`test_string_to_acq.py` and `test_active_importance_sampling.py` (109). On
`601555e` the four seeded runs are bit for bit those of `f873556` (92
arrays). On `11fb766`, which adds the records to that code: the exact oracle
check, 11 of 11; the whole default suite, 1687 passed and 58 skipped with no
reruns; the Torch environment, 777 passed and 1 skipped; the PyMC
environment, 107 passed.

**CI.** On `11fb766` the branch smoke (Ubuntu, Python 3.12, with Torch) and
the full matrix, nine cells, are green at the first attempt. The cells of
macOS and Ubuntu compare the re-baselined reference of `acq_AcqFcnIMIQR`,
computed on Windows, within the oracle's tolerance.

## The independent check of the pass

Four fresh Opus reviewers, read-only, checked the pass on 2026-09-21 without
the context of the session that made it: the commits of the
acquisition-function side (the one reviewer that ran tests), those of the
importance sampling, this ledger against the reports, the sources and the
logs, and the user-facing and cross-document records. They found no defect
in the twelve commits. In the records they found, and the orchestrator
corrected: the evidence this ledger gave for W4-7, the VIQR oracle, which
never reaches the changed function; the statement that the weighted start
ranks the best candidate no worse, where the log has it 0.0043 worse, within
noise; the dating of `renormalize_weights`; a disposition of W4-9 that had
not been carried out; and a changelog sentence on
`mcmc_importance_sampling` that held for some acquisitions alone. The code
follow-ups are the five commits above. One reviewer confirmed that `70ce067`
is needed beyond its two test values: the inverse transform returns the
largest number below a half, negated, for a coordinate saturated at the
lower bound -0.5 of an integer variable, which `floor(abs(x) + 0.5)` rounds
to -1, a point outside the box whose forward transform is NaN.

Left as they are, on the reviewers' own weighing: a restored acquisition
object does not pass through the constructor, so an object saved before the
pass keeps its `acq_info` and its quantile; `np.round` at
`active_sample.py:994` and the built-in `round` of `stats/get_hpd.py` round
a half to even where MATLAB rounds away from zero, which the P2 and P7
reviewers report; only the `cmaes` branch of the local search is exercised
with an integer variable; `fess` and `AcqFcnVIQR.is_log_full` have no caller
in the package.

## Found during the fix pass

- `floor(abs(x) + 0.5)`, the usual way to write a rounding away from zero,
  rounds its own sum: the largest number below a half goes to one, and an odd
  integer of 53 bits to the even integer above it. The orchestrator replaced
  it with the truncated part plus a comparison of the exact fractional part
  with a half (`70ce067`), with those values in the test.
- The probit transform maps the midpoint 4.5 of the box `[-0.5, 9.5]` to
  7.5e-17 on the machine that ran the pass, not to the exact 0 that
  `wave4_P3.md` reports, and back to 4.5 exactly, so the tie of W4-4 is
  reached all the same; the test transforms the midpoint instead of writing
  the literal (agent's report).
- `_validate_search_acq_fcn_option` runs at construction, as the checks of
  `noise_shaping` and `gp_hyp_sampler` beside it do; an acquisition that sets
  the flag and arrives through `load(new_options=)` is refused in
  `active_importance_sampling` instead. Not acted on.
