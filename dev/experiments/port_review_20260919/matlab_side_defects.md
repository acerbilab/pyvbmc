# Defects found on the MATLAB side

Defects of MATLAB VBMC (`acerbilab/vbmc`, `master` at `396d649`) that the
port correctness review came across while comparing PyVBMC with it
(`dev/plans/port-correctness-review.md`). They are material for the MATLAB
repository; none of them is a task for PyVBMC. Every entry rests on a
reading of the MATLAB source, confirmed by the verification record it
cites. **Nothing here was run in MATLAB**, so an entry says what the code
reads as doing, and one entry marks the step that is inferred. Paths are
relative to the MATLAB repository root. The list is brought up to date as
the review's waves are verified; it stands at the end of wave 2.

## Defects

| # | Location | What the code does | Consequence | PyVBMC | Record |
|---|---|---|---|---|---|
| 1 | `shared/msmoothboxrnd.m:59`, `:66` | The Gaussian tails are drawn around `a(idx)` and `b(idx)`, a linear index into the first column, where line 72 of the same function uses `a(idx,d)` and `b(idx,d)` | For `D > 1` every dimension's tails are placed around the first dimension's bounds | `SmoothBox.sample` uses each dimension's own bounds | `reviews/M_comparison.md`, commit `c387612` |
| 2 | `private/vbmc_output.m:5-9` | `output.problemtype` tests `optimState.LB` and `optimState.UB`, which `misc/setupvars_vbmc.m:49-50` fills with the transformed bounds; the transform sends a finite bound to infinity | Every problem is reported as `'unconstrained'`. A reported field only | Shared until fixed (`verification/wave2.md`, W2-12) | `verification/wave2_B_loop.md`, B-3a |
| 3 | `misc/warp_input_vbmc.m:8`, `:133` | The search bounds and the search cache, which live in the current inference space, are mapped back to the original space with `vp.trinfo` of the posterior handed in, which `vbmc.m:546-557` takes from `best_vbmc`; a posterior recorded before an earlier warp carries an older transform | After a second or later warp the search box can be grossly rescaled (8 to 12 times wider per coordinate in the reproduction on the Python side). Latent: it needs the best-ranked iteration to predate the previous warp | Shared until fixed (W2-6) | `verification/wave2.md`, W2-6 |
| 4 | `private/vbmc_warmup.m:39`, `:87` | With `TolStableWarmup <= FunEvalsPerIter` the window `max(4,end-T+1):end` is empty on the first call that passes the guard of line 35, so `max` returns `[]`, the comparison yields an empty logical, and `StableCountFlag && ...` on line 87 receives it. That MATLAB raises there is inferred from the rules of `&&` | A run with a one-iteration stability window fails at its third iteration | Shared until fixed (W2-14); PyVBMC raises in `np.amax` | `verification/wave2_B_loop.md`, B-4 |
| 5 | `private/activesample_vbmc.m:481`, with `misc/funlogger_vbmc.m:229-248` | `update1` is true for a repeated input on a noiseless target (`isempty(s2new)`), so the rank-one path appends a GP training row for an input that the logger pooled into an existing row without incrementing `Xn` | The GP holds a duplicate training row for that input until its next full update | Not shared: PyVBMC recomputes the posterior from the logger's training set in that case (sheet, "The rank-one GP update is taken for a fresh observation, noisy or not") | `verification/wave1_M_P2.md`, P2 F8 |
| 6 | `private/activesample_vbmc.m:380-392` | A cached starting point is deleted from the cache only in the branch that reuses a stored value; a cached row whose value is `NaN` is evaluated and left in the cache | The sieve can draw the same row again and spend a second evaluation on it. Only when more starting points are supplied than the initial design consumes | Not shared (sheet, "An acquired starting point leaves the cache whether or not it had a value") | `verification/wave1_M_P2.md` |
| 7 | `misc/vpoptimize_vbmc.m:236-237` | Pruning a component deletes it from the component axis of `I_sk` and from the last axis alone of `J_sjk`, which is declared `(Ns,K,K)` | `vp.stats.J_sjk` becomes `(Ns,K,K-1)`, inconsistent with its own `I_sk` | Not shared: both axes are pruned (sheet, "`vp.stats["J_sjk"]` is pruned on both component axes") | `verification/wave1_P6.md` |
| 8 | `vbmc.m:699`, with `vbmc.m:459` | The main loop sizes the sieve with `options.NSelbo` evaluated at `K`, which is assigned once, to `options.Kwarmup`, and never updated; `vbmc.m:584` and `misc/finalboost_vbmc.m:33` use `Knew` | The main loop always asks for `50*Kwarmup = 100` fast candidates, or 10 on an incremental iteration, however many components the posterior has | Not shared: PyVBMC evaluates the option at the current `vp.K` (sheet, "The sieve asks for candidates in proportion to the current `K`") | `verification/wave1_P6.md`, row "cmp F3" |
| 9 | `vbmc_pdf.m` | With `origflag` true and `logflag` false the density is divided by the transform's Jacobian and the gradient is returned as computed in the transformed space, uncorrected | The returned gradient is not the gradient of the returned density | Not shared: PyVBMC refuses both original-space gradient modes (sheet, "`vp.pdf` refuses original-space gradients instead of returning a wrong one") | `dev/plans/latent-bug-fixes.md`, question Q2 |
| 10 | `misc/setupvars_vbmc.m:19-22` | The bound check of an integer variable reads `~isfinite(LB(d)) && floor(LB(d)) ~= 0.5`; `floor` of any value differs from 0.5, so the test reduces to "the bound is not finite" | The half-integer placement of the bounds that the error message demands is never checked | Not shared: PyVBMC checks both | `verification/wave2_C_setup.md`, C-C12 |
| 11 | `misc/boundscheck_vbmc.m:27-30` | `idx = any(PLB == PUB)` reduces the row to one logical, so `PLB(idx) = LB(idx)` repairs the first coordinate only | When the plausible bounds are estimated from several starting points and a coordinate other than the first has zero width, that coordinate keeps equal plausible bounds | Not shared: every degenerate coordinate is repaired | `verification/wave2_C_setup.md`, C-C5 |
| 12 | `misc/setupoptions_vbmc.m:120-124` | In the `MaxFunEvals < MinFunEvals` branch the assignment reads `options.MinFunEvals = options.MinFunEvals;` | The warning announces a change of `MaxFunEvals` that is never applied | No counterpart of the block (W2-26 adds the validation) | `verification/wave2_C_setup.md`, C-C3 |
| 13 | `misc/setupoptions_vbmc.m:47` | The list of evaluated fields contains `'ConstrainedGPMean''FeatureTest'`, one string, since commit `2044530` (2021-06-18) dropped the comma | Neither option is evaluated; both stay the character vector `'no'`. Neither is read anywhere, so nothing follows | No counterpart | `reviews/M_comparison.md`; `verification/wave2_C_setup.md`, C-C4 |

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
