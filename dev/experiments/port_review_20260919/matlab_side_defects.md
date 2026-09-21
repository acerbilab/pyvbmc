# Defects found on the MATLAB side

Defects of MATLAB VBMC (`acerbilab/vbmc`, `master` at `396d649`) that the
port correctness review came across while comparing PyVBMC with it
(`dev/plans/port-correctness-review.md`). They are material for the MATLAB
repository; none of them is a task for PyVBMC. Every entry rests on a
reading of the MATLAB source, confirmed by the verification record it
cites. **Nothing here was run in MATLAB**, so an entry says what the code
reads as doing, and three entries (19, 21 and 22) mark the step that is
inferred. Paths are
relative to the MATLAB repository root. The list is brought up to date as
the review's waves are verified; it stands at the end of wave 4 and of its
fix pass.

## Defects

| # | Location | What the code does | Consequence | PyVBMC | Record |
|---|---|---|---|---|---|
| 1 | `shared/msmoothboxrnd.m:59`, `:66` | The Gaussian tails are drawn around `a(idx)` and `b(idx)`, a linear index into the first column, where line 72 of the same function uses `a(idx,d)` and `b(idx,d)` | For `D > 1` every dimension's tails are placed around the first dimension's bounds | `SmoothBox.sample` uses each dimension's own bounds | `reviews/M_comparison.md`, commit `c387612` |
| 2 | `private/vbmc_output.m:5-9` | `output.problemtype` tests `optimState.LB` and `optimState.UB`, which `misc/setupvars_vbmc.m:49-50` fills with the transformed bounds; the transform sends a finite bound to infinity | Every problem is reported as `'unconstrained'`. A reported field only | Not shared since the wave-2 fixes: `results["problem_type"]` tests the original bounds (`verification/wave2.md`, W2-12) | `verification/wave2_B_loop.md`, B-3a |
| 3 | `misc/warp_input_vbmc.m:8`, `:133` | The search bounds and the search cache, which live in the current inference space, are mapped back to the original space with `vp.trinfo` of the posterior handed in, which `vbmc.m:546-557` takes from `best_vbmc`; a posterior recorded before an earlier warp carries an older transform | After a second or later warp the search box can be grossly rescaled (8 to 12 times wider per coordinate in the reproduction on the Python side). Latent: it needs the best-ranked iteration to predate the previous warp | Not shared since the wave-2 fixes: `warp_input` inverts the search bounds and the search cache with the transform of the current inference space (W2-6) | `verification/wave2.md`, W2-6 |
| 4 | `private/vbmc_warmup.m:39`, `:87` | With `TolStableWarmup <= FunEvalsPerIter` the window `max(4,end-T+1):end` is empty on the first call that passes the guard of line 35, so `max` returns `[]`, the comparison yields an empty logical, and `StableCountFlag && ...` on line 87 receives it. That MATLAB raises there is inferred from the rules of `&&` | A run with a one-iteration stability window fails at its third iteration | Not shared since the wave-2 fixes: an empty window means no stability count yet (W2-14); until then PyVBMC raised in `np.amax` | `verification/wave2_B_loop.md`, B-4 |
| 5 | `private/activesample_vbmc.m:481`, with `misc/funlogger_vbmc.m:229-248` | `update1` is true for a repeated input on a noiseless target (`isempty(s2new)`), so the rank-one path appends a GP training row for an input that the logger pooled into an existing row without incrementing `Xn` | The GP holds a duplicate training row for that input until its next full update | Not shared: PyVBMC recomputes the posterior from the logger's training set in that case (sheet, "The rank-one GP update is taken for a fresh observation, noisy or not") | `verification/wave1_M_P2.md`, P2 F8 |
| 6 | `private/activesample_vbmc.m:380-392` | A cached starting point is deleted from the cache only in the branch that reuses a stored value; a cached row whose value is `NaN` is evaluated and left in the cache | The sieve can draw the same row again and spend a second evaluation on it. Only when more starting points are supplied than the initial design consumes | Not shared (sheet, "An acquired starting point leaves the cache whether or not it had a value") | `verification/wave1_M_P2.md` |
| 7 | `misc/vpoptimize_vbmc.m:236-237` | Pruning a component deletes it from the component axis of `I_sk` and from the last axis alone of `J_sjk`, which is declared `(Ns,K,K)` | `vp.stats.J_sjk` becomes `(Ns,K,K-1)`, inconsistent with its own `I_sk` | Not shared: both axes are pruned (sheet, "`vp.stats["J_sjk"]` is pruned on both component axes") | `verification/wave1_P6.md` |
| 8 | `vbmc.m:699`, with `vbmc.m:459` | The main loop sizes the sieve with `options.NSelbo` evaluated at `K`, which is assigned once, to `options.Kwarmup`, and never updated; `vbmc.m:584` and `misc/finalboost_vbmc.m:33` use `Knew` | The main loop always asks for `50*Kwarmup = 100` fast candidates, or 10 on an incremental iteration, however many components the posterior has | Not shared: PyVBMC evaluates the option at the current `vp.K` (sheet, "The sieve asks for candidates in proportion to the current `K`") | `verification/wave1_P6.md`, row "cmp F3" |
| 9 | `vbmc_pdf.m` | With `origflag` true and `logflag` false the density is divided by the transform's Jacobian and the gradient is returned as computed in the transformed space, uncorrected | The returned gradient is not the gradient of the returned density | Not shared: PyVBMC refuses both original-space gradient modes (sheet, "`vp.pdf` refuses original-space gradients instead of returning a wrong one") | `dev/plans/latent-bug-fixes.md`, question Q2 |
| 10 | `misc/setupvars_vbmc.m:19-22` | The bound check of an integer variable reads `~isfinite(LB(d)) && floor(LB(d)) ~= 0.5`; `floor` of any value differs from 0.5, so the test reduces to "the bound is not finite" | The half-integer placement of the bounds that the error message demands is never checked | Not shared: PyVBMC checks both | `verification/wave2_C_setup.md`, C-C12 |
| 11 | `misc/boundscheck_vbmc.m:27-30` | `idx = any(PLB == PUB)` reduces the row to one logical, so `PLB(idx) = LB(idx)` repairs the first coordinate only | When the plausible bounds are estimated from several starting points and a coordinate other than the first has zero width, that coordinate keeps equal plausible bounds | Not shared: every degenerate coordinate is repaired | `verification/wave2_C_setup.md`, C-C5 |
| 12 | `misc/setupoptions_vbmc.m:120-124` | In the `MaxFunEvals < MinFunEvals` branch the assignment reads `options.MinFunEvals = options.MinFunEvals;` | The warning announces a change of `MaxFunEvals` that is never applied | No counterpart of this branch: PyVBMC checks `max_fun_evals` and `max_iter` and raises `max_iter` to `min_iter` (W2-26), and does not compare `max_fun_evals` with `min_fun_evals` | `verification/wave2_C_setup.md`, C-C3 |
| 13 | `misc/setupoptions_vbmc.m:47` | The list of evaluated fields contains `'ConstrainedGPMean''FeatureTest'`, one string, since commit `2044530` (2021-06-18) dropped the comma | Neither option is evaluated; both stay the character vector `'no'`. Neither is read anywhere, so nothing follows | No counterpart | `reviews/M_comparison.md`; `verification/wave2_C_setup.md`, C-C4 |
| 14 | `misc/finalboost_vbmc.m:6`, with `misc/vbinit_vbmc.m:132-136` | With `VariableMeans` off the boost takes `Knew = max(MinFinalComponents, vp.K)` while the initialization keeps the posterior's own `mu0`, of `vp.K` columns, beside `Knew` weights and scales. The main loop avoids the mismatch by setting `vp.mu = gp.X'` and `Knew = size(vp.mu,2)` first (`vbmc.m:575-577`, `:690-692`); the boost does not. Read, not run | A fixed-means run that ends with fewer than `MinFinalComponents` components, as one stopped during warm-up does, would fail in the boost | Not shared since `73d2a81`: the boost places the components at the training inputs (`verification/wave2.md`, W2-29) | `verification/scripts/wave2_variable_means_boost.py` for the Python side |
| 15 | `misc/funlogger_vbmc.m:244` | The duplicate branch of `record` writes the pooled value of a repeated point to `optimState.y(optimState.Xn)`, the last filled row, for `optimState.y(idx)`, the row of the point; the new-row branch below it uses `Xn` rightly, since there `idx == Xn` | The last filled row takes another point's value and the repeated row keeps its old one; both are GP training targets (`misc/get_traindata_vbmc.m:7`) and feed `ymax`. Only with `MaxRepeatedObservations > 0`, which no default sets, those for noisy targets included; then at nearly every repeat, the repeated point being rarely the last one added | Not shared: `self.y[idx]` since `2527c47` (2021-05-20) | `verification/wave3_P8.md`, MATLAB-side defects, 1 |
| 16 | `misc/funlogger_vbmc.m:159-162`, with `private/activesample_vbmc.m:388` and `misc/initdesign_vbmc.m:56` | The `'add'` action reads `fsd = varargin{2}` whenever the run is noisy, and both call sites pass the value alone, so the line that defaults a missing SD to 1 is never reached | A noisy run with `Fvals`, or one that acquires a cached starting point that has a value, raises "Index exceeds the number of array elements" | Not shared: `add` takes the SD as an optional argument; at the level where the target provides its noise PyVBMC requires it, and refuses `f_vals` there (sheet, "A cached value of a target that provides its noise needs its SD") | `verification/wave3_P8.md`, MATLAB-side defects, 2 |
| 17 | `misc/setupvars_vbmc.m:96`, with `misc/warp_gpandvp_vbmc.m:8-10` | `vp.temperature` starts as NaN, and the warp guards on `~isempty`, which NaN passes | None as the loop stands: `misc/vpoptimize_vbmc.m:190` overwrites the field in the first iteration, before a warp can occur. The order of the loop is what prevents a NaN temperature from reaching every warped hyperparameter and weight | No counterpart | `verification/wave3_P8.md`, MATLAB-side defects, 4 |
| 18 | `misc/gptrain_vbmc.m:19-25` | The branch on `optimState.Warmup && options.BOWarmup` and its `else` call `vbmc_gphyp` with the same arguments; the commented-out lines beside them show the intent, a constant mean during that warm-up | `BOWarmup` never switches the GP mean function, and `vbmc.m:825-828` restores at the end of warm-up a mean function that never changed | No counterpart: the option is not ported (sheet) | `verification/wave3_P5.md`, MATLAB-side defects, 2 |
| 19 | `misc/get_GPTrainOptions.m:112` | The burn-in of the `slicelite` sampler divides by `log(options.GPRetrainThreshold)`, which is `log(1) = 0` at the default. Inferred from the rules of the arithmetic, not run: in the branch's own region `rindex < 1` the quotient is minus infinity, and `max(1, ceil(-Inf))` is 1 | At the default threshold the burn-in is `Ns_gp` whatever the reliability index, so the scaling the formula is there for never happens | No counterpart since 2026-09-20: slice sampling is the only sampler (sheet); the transcription removed then had the division inside the logarithm | `verification/wave3_P5.md`, MATLAB-side defects, 1 |
| 20 | `misc/gptrain_vbmc.m:33`, with `misc/get_GPTrainOptions.m:63` | For `GPHypSampler = 'covsample'` the widths are built as an `Nhyp` by `Nhyp` covariance, and the caller discards widths whose number of elements differs from `Nhyp` | The covariance never reaches `eissample_lite`: covariance sampling runs on the default widths | No counterpart since 2026-09-20 (as 19); the transcription had the same guard | `verification/wave3_P5.md`, MATLAB-side defects, 5 |
| 21 | `acq/acqviqr_vbmc.m:25-28`, with `private/activeimportancesampling_vbmc.m:345-352` | The `'islogf'` branch of VIQR adds `vp`, its second argument, to the log-density; `log_isbasefun` passes `[]` there for an acquisition whose `importance_sampling_vp` is false, as VIQR's is. Inferred from the rules of the arithmetic, not run: `[] + x` is `[]` | The target density of the MCMC refinement comes out empty. Reachable through `mcmc_importance_sampling` alone, which no MATLAB acquisition sets | Not shared: `AcqFcnVIQR.is_log_full` returns the added term, with a base of zero. The refinement itself is not ported (sheet) | `verification/wave4_P3.md` and `wave4_P4.md`, MATLAB-side defects |
| 22 | `private/activeimportancesampling_vbmc.m:208-214`, with the local `catrnd` (`:396-420`) | The loop that draws the walkers' starting points sets the weight of each drawn sample to zero and never renormalizes. Inferred, not run: once every remaining weight is zero, `catrnd` scales its uniform draw by a total of zero, every comparison is false and it returns index 1 | The remaining walkers all start at the first proposal sample. With `2(D+1)` walkers and the effective sample sizes of 1.5 to 2 measured on a stored state of PyVBMC, it is within reach | No counterpart: PyVBMC runs one chain from one starting point | `verification/wave4_P4.md`, MATLAB-side defects, 3; `verification/wave4.md`, W4-1 |
| 23 | `gplite/gplite_post.m:76-79`, with `private/activesample_vbmc.m:481-484` | A requested rank-one update becomes a full recomputation whenever a noise variance is given, without a message | None for the numbers. The call site reads as if noisy observations were taken by rank one | Not shared: gpyreg extends the factors by rank one with the noise variance, which equals the recomputation to rounding (sheet, the rank-one GP update) | `verification/wave4_P4.md`, row P4-5 |
| 24 | `acq/acqviqr_vbmc.m:107-108`, `acq/acqimiqr_vbmc.m:93-94` | A row of the log sum whose entries are all `-Inf` gives NaN, which `acq/acqwrapper_vbmc.m:47` turns into `-realmax` | None: the value is the right one for a candidate that leaves no interquantile range at any importance point | The same value, reached without the NaN (guards in the log sums) | `verification/wave4_P3.md`, MATLAB-side defects, 3 |

## Questionable, shared by both implementations

Not defects by a plain reading, but without a counterpart in the algorithm
papers; both are on the known-differences sheet under "Settled
non-differences".

- `misc/gplogjoint.m:404`: `varss = varFss + std(varF)` adds a standard
  deviation of the per-sample variances to a variance of the per-sample
  means.
- `utils/fminadam.m:48`, `:63`: `ftab(iter)` is the objective at the point
  the iteration starts from and `xtab(:,iter)` the point after the update,
  so the two tables, and their trailing averages, are offset by one.

Found in wave 3 and left as they are in both implementations
(`verification/wave3.md`, rows W3-20, W3-29 and W3-26):

- `misc/funlogger_vbmc.m:220-247`: the duplicate scan of `record` covers
  every row, and a repeat at an input whose row the end of warm-up
  deactivated is pooled into that row, which stays inactive: the evaluation
  is paid for and seen by nothing. It needs `MaxRepeatedObservations > 0`
  and a search cache that carries a training input across the trim.
- `misc/warp_input_vbmc.m:164-167`, under the comment "Reset GP
  hyperparameters", clears the running average of the variational moments,
  which nothing reads. After a warp only `hypstruct.hyp` is warped
  (`vbmc.m:559`); the running hyperparameter covariance, the last chain and
  the recorded GPs and chains of before the warp go on feeding the starting
  points and the sampler widths of the hyperparameter fit. The density the
  fit targets is unaffected.
- `shared/warpvars_vbmc.m:763-765` adds `log(scale)` to the log-Jacobian,
  where `log(abs(scale))` is the term; a warp always produces a positive
  scale. PyVBMC validates a scale given to its public constructor.

Found in wave 4 and left as it is in both implementations
(`verification/wave4.md`, row W4-16):

- `private/activesample_vbmc.m:165`: `S.^2 .* nevals`, the variance of one
  observation recovered from the pooled one, is the harmonic mean of the
  variances at a repeated input whose observations have different SDs, the
  smallest of the plausible summaries, so the noisy acquisitions take a new
  observation there to be more precise than the past ones were. It needs
  repeats with unequal SDs at one input.

The thresholded covariance of the warp, which need not be positive
semi-definite (`misc/warp_input_vbmc.m:52-71`, and the recipe of the 2020
paper's appendix B.2), is not shared since 2026-09-20: PyVBMC keeps the
covariance as it was in that case (sheet; `verification/wave3.md`, row
W3-6). It occurs in none of the 990 final posteriors of PyVBMC's benchmark
campaigns.

## Written or declared and never read

State and options that the MATLAB code sets and nothing reads, found while
checking whether PyVBMC's counterparts were live:

- `optimState.LastNonlinearWarping` (`private/vbmc_warmup.m:102`), its only
  occurrence.
- `optimState.RunMean`, `RunCov`, `LastRunAvg` (`vbmc.m:779-793`): updated
  every iteration, reset by `misc/warp_input_vbmc.m:165-167`, read by no
  computation, so `MomentsRunWeight` changes no computed number.
- `optimState.redoRotoscaling` (`vbmc.m:519`), `optimState.pruned`
  (`misc/setupvars_vbmc.m:207`), `optimState.iterList`
  (`misc/setupvars_vbmc.m:242-245`), `optimState.acqrand`
  (`private/activesample_vbmc.m:215`), `optimState.ProposalFcn`
  (`misc/setupvars_vbmc.m:186-190`), `trinfo.x0_orig`
  (`misc/setupvars_vbmc.m:45`).
- `options.OptimToolbox`: computed at `misc/setupoptions_vbmc.m:98-106` and
  read nowhere else in the repository.
- `defopts.AnnealedGPMean`, `defopts.ConstrainedGPMean`,
  `defopts.FeatureTest`: declared, with no reader since commit `2044530`.
- `utils/fminfill.m`: a copy of `gplite/private/fminfill.m` that nothing
  calls (the two call sites in `gplite/gplite_train.m` resolve to the
  private one).
