# Wave 1: the disposition of every finding, and the independent check of the pass

Wave 1 of the port review (2026-09-19) read slice M (the MATLAB changes
since the port), slice P6 on both tracks (the variational optimization, the
ELBO, `update_K`) and slice P2 on the comparison track (the initial design
and the active-sampling search). Its records are the four reports under
`../reviews/` (`M_comparison.md`, `P6_internal.md`, `P6_comparison.md`,
`P2_comparison.md`), the verifiers' ledgers `wave1_M_P2.md` and
`wave1_P6.md`, the pass over the options surface `wave1_options.md`, the
measurement `cmaes_side_by_side/`, and the worklog of
`dev/plans/port-correctness-review.md`. The two verifiers' ledgers are their
reports kept verbatim and have no column for the disposition; this file
gives it for every finding, and holds the independent check of the pass,
made on 2026-09-22.

## Dispositions

"This check" names the commits of the independent check below.

### Slice P6 (`wave1_P6.md`)

| finding | disposition | where |
|---|---|---|
| int F1 + cmp F1, type-2 starting widths | fixed | `be38570` |
| int F2 + cmp F6, stale `eta` | fixed; the same commit keeps the `(1, K)` shape of `eta` in the pruning loop, one of the verifier's own observations | `495a497` |
| int F13b + cmp F2, soft bounds never accumulate | ruled (PI): the box is rebuilt from the training inputs of each call, and the accumulation code, which never had a stored box to widen, is removed | `7dd2e3f`; worklog, 2026-09-19 rulings |
| int F5 + cmp F8, `_sieve` with no candidates | fixed | `51b01cb` |
| int F10 + cmp F9, `minimize_adam` writes into its start; the averaging window of a short run | fixed | `dccbc3d` |
| int F7 + cmp F4, the deterministic-entropy branch | the iterate is kept with a warning; the evaluation cap and the tolerance are a sheet entry | `04d15ff`; sheet, "The deterministic-entropy optimization is SciPy's BFGS"; this check, `43dfb7b`, for an iterate whose objective is NaN |
| int F3, Monte Carlo entropy where the exact one exists | exact for a one-component posterior; under `entropy_switch` shared with MATLAB and kept | `7331841`; sheet, "The full ELCBO evaluation uses the exact entropy of a one-component posterior" |
| int F4, F6, F13a | not defects; F6 documented | `d54266c` |
| int F8, F9 | intentional differences already on the sheet | sheet |
| int F11, F12 | differences between the paper and the code, shared with MATLAB; no action | — |
| cmp F3, the sieve's candidate count | sheet | "The sieve asks for candidates in proportion to the current `K`"; `../matlab_side_defects.md`, entry 8 |
| cmp F5, `Bandwidth` | sheet (unported) | "GP smoothing (`Bandwidth`, `vp.delta`) is not ported" |
| cmp F7, the `adaptive_k` keyword and its rounding | fixed: the keyword by wave 1, the rounding by wave 5 | `06f9b03`; `8e2977a` (`wave5.md`, W5-6) |
| cmp F10a, F10b, unstable sort and `argmin` of NaN | fixed, with the second unstable sort the verifier found | `81e21bf`; this check, `43dfb7b`, for the slots never evaluated and the midpoint |
| cmp sheet notes 1-2 | the sheet corrected | `583ef86` |
| cmp sheet note 3, the discarded write to the repository | fixed | `1564769` |
| verifier: type 3 draws its permutation only when the means are optimized | left for the PI (below); reached only with `variable_means=False` | — |
| verifier: `get_bounds` special case for `tol_weight == 0` | removed | `7dd2e3f` |
| verifier: `get_parameters` keeps the cached mode | fixed in wave 5 | `eab21fa` (`wave5.md`, W5-13) |

### Slices M and P2 (`wave1_M_P2.md`)

| finding | disposition | where |
|---|---|---|
| P2 F1, CMA-ES step size per coordinate | fixed | `4d1d984` |
| P2 F2 = M F1, the size of the initial design | fixed | `cb2d513` |
| P2 F3, F4, surplus starting points and the random count | fixed | `181a63b`; this check, `e985b95`, for their reuse after a warp |
| P2 F5, cma's noise handler | ruled (PI): dropped, after the side-by-side | `d617d99`; `cmaes_side_by_side/` |
| P2 F6, `search_cmaes_best` | ruled (PI): stays inert | `a63b17a`; sheet, "The `cma` package replaces `cmaes_modded.m`" |
| P2 F7, `search_cache_frac` on the first step | fixed; a value above 0.25 with the shipped fractions is refused since wave 5 | `9abeb10`; `wave5.md`, W5-24 |
| P2 F8 = M F3, the rank-one GP update | changed by `510a493`, which the header of `wave1_M_P2.md` corrects: MATLAB never takes the rank-one path on a noisy target; ruled (PI) in wave 4 to stay | `510a493`; `wave4.md`, W4-11; `../matlab_side_defects.md`, entries 5 and 23 |
| P2 F9, the one-dimensional search | fixed; the `"Nelder-Mead"` value removed in wave 5 | `cb8a51d`; `wave5.md`, W5-25 |
| P2 F10, `n_eff` | fixed for the name, and by this check for the refresh after every evaluation | `118626e`; this check, `6590ea4` |
| P2 F11, `noise_shaping` | refused when true | `bb6ab65`; this check, `497d15e`, for the message |
| P2 F12, fallback search bounds | fixed | `659e83c` |
| P2 F13, a failing local search | fixed | `795b3b0` |
| P2 F14, `active_sample_fess_thresh` | inert | `a63b17a` |
| P2 F15, the repository write | removed | `1564769` |
| P2 F16, the timers | (a) fixed; (b) and (c) left by a ruling of wave 5 | `a719de6`; `wave5.md`, W5-28 |
| P2 minor observations | the pointwise value `f_val_old`: recorded in the Stage 2 plan; rounding half to even: fixed in wave 5 (W5-6); the unstable `argsort` of the search set: left for the PI (below); the direct option calls: this check, `179406c`; the case-sensitive `"none"`: superseded by the check of `search_optimizer` values (W5-25); the order of the draws: the sheet's entry on randomness; the shape of `gp_length_scale`: no action; `skip_logger` left in the cache: fixed | `a7f323e` for the last |
| P2 §4 sheet notes | the `_selection_policy_callback` hook of `active_sample.py` is a developer seam, inert unless installed; no action. P2's statement that the sheet had no `vp_repo` entry is wrong: it had one | — |
| M F2, options MATLAB deleted in 2021 | inert | `a63b17a` |
| M F4, MATLAB's `samples` output struct | sheet (unported) | "MATLAB's `samples` output struct has no counterpart in `results`" |
| M F5, `warp_cov_reg` | fixed; checked at construction by this check | `1d99936`; this check, `cc0ef2f` |
| M sheet notes | the counterpart map corrected; the prediction's log density is slice G2's; the one-dimensional override as P2 F9 | `583ef86`; `wave6.md` |
| verifier items 1-2, `eval_initial_x` and `tolfunhist` | ruled (PI): cma's defaults stay | sheet, the `cma` entry |
| verifier item 3, MATLAB's duplicate row on a noiseless repeat | MATLAB-side defect | `../matlab_side_defects.md`, entry 5 |
| verifier item 4, the refresh of `Neff` after every evaluation | this check | `6590ea4` |
| verifier items 5-6 | no action | — |

The rows of `wave1_options.md` carry their commits; row O-6 (`gp_int_mean_fun`)
and `proposal_fcn` are registered as inert by this check (`375df34`).

## The independent check of the pass

Six fresh Opus reviewers, read-only, checked the pass on 2026-09-22 without
the context of the session that made it, on `dev-port-review` at
`a3d4a70d`: R1 the active-sampling search (the CMA-ES start, the noise
handler, the one-dimensional search, the fallback bounds, a failing local
search, the re-baseline of `active_sample_step`); R2 the initial design, the
search cache, the GP updates inside active sampling and the regularization
of the warp; R3 the variational optimization; R4 the options and the
selection of the returned posterior; R5 the three ledgers and the worklog
against their sources; R6 the changelog, the sheet and the other records,
and the consequences of the pass outside the lines it changed. Each judged
whether a fix was right when made, whether it was complete, and whether it
still holds after the passes of waves 2 to 6. Their raw reports and probes
are on the orchestrator's machine (`dev/scripts/runs/LOCAL.md`).

What holds. No later pass undid a wave-1 fix. `determine_best_vp` agrees
with a 1-based transcription of `best_vbmc.m` on all 19,520 constructed
histories without NaN, and each of its wave-1 fixes fails its test on the
code before it. The type-2 starting widths equal a transcription of
`vbinit_vbmc.m:25-32` exactly for every `D`, `K` and `K_new` tried. The first
generation of the CMA-ES search has the standard deviation `insigma` per
coordinate, as `cmaes_modded.m` starts it. `fun_eval_start` equals
`10*ceil((D+1)/10)` for `D` from 1 to 25. The search cache matches
`initdesign_vbmc.m` for a starting set longer than, as long as and shorter
than the design. The rank-one GP update passes the squared SD of the point
at every uncertainty level and equals a recomputation to rounding.
`dd89374` was made by the generator's targeted mode, and only the
`active_sample_step` references of the seven states that hold one moved.
The user documentation, the notebooks and the agent skill say nothing the
pass made false, and a run saved by 1.0.4 or before the pass loads.

In the code they found what these commits follow up, each fix with a test
that fails on the code before it, except where the test pins a contract the
old code already met:

| commit | |
|---|---|
| `43dfb7b` | `optimize_vp` selects among the full ELCBO evaluations that were made. The slots of the midpoints that the deterministic optimization, and a run with `elcbo_midpoint` off, never fill hold `+inf` and parameters of NaN, and the guard of `81e21bf` counted them: when every evaluation was NaN, `nanargmin` took an empty slot and returned a posterior of NaN without an error. It raises now. The best midpoint skips NaN, as MATLAB's `min` does (R3) |
| `f5eafa7` | `determine_best_vp` reads its scores as floats: in the object arrays of the history NaN is incomparable, so a NaN ELCBO or reliability index was ranked anywhere, and the look-back took a NaN at the start of its window. A NaN score ranks last, the look-back skips it, and when every candidate's ELCBO is NaN the last iteration considered is taken, in both branches (R4) |
| `e3c6e93`, `c469ba5` | `determine_best_vp()` called without its selection arguments uses the run's options (`rank_criterion` on by default), so a direct call returns the iteration the run selected; the test of `best_safe_sd` names the look-back branch it tested by default (R6) |
| `e985b95` | a cached starting point that nothing moved is recognized after a warp. The check of W5-7 compared the candidate with a new one-row transform of the point, and once a warp has set a rotation a one-row matrix product rounds differently from the sieve's many-row one (0 to 3 rows of 7 bitwise equal on the orchestrator's machine), so the stored value was dropped and the target called. The candidate is compared with the row the sieve made of the point before its clip (R2) |
| `6590ea4` | `N` and `n_eff` are refreshed after every evaluation of active sampling and after the initial design, as `misc/funlogger_vbmc.m:278-279` refreshes them; the GP refit inside active sampling, on at the noisy defaults, read them one evaluation behind. `118626e` had renamed the key and refreshed it at the top of each acquisition only, and its message named `optimize_vp` among the readers, which reads no `n_eff` (R2, R5). Moves noisy trajectories |
| `375df34` | `gp_int_mean_fun` and `proposal_fcn` are registered as having no effect: each was copied into a key of `optim_state` that nothing reads, so neither warned, and `load` refused both with advice that would not help (R4) |
| `2a3b8e8` | `load(new_options=...)` warns of an option without effect as construction does, which the changelog promised (R4, R6) |
| `497d15e` | the refusal of `noise_shaping=True` names the `new_options` that loads a saved run carrying it (R4, R6) |
| `e56da87` | a budget equal to the initial design no longer ends the first GP fit in a division by zero; `cb2d513` had moved that equality to `max_fun_evals = 20` for `D` from 10 to 19, a setting 1.0.4 ran (R2) |
| `179406c` | `ns_elbo` and `ns_ent_fine_active` are read through `Options.eval` in the updates inside active sampling, where a number given for either raised `TypeError`: P2's fourth minor observation, which no record had disposed of (R2) |
| `cc0ef2f` | `warp_cov_reg` is checked at construction and in `load`, and a function's result at the warp; `None`, a string, a complex number or NaN had failed deep in the first warp, and `True` was taken as 1 (R2) |
| `831b024` | the comparison inside active sampling scores a one-component posterior by its exact entropy, as its reported ELBO has been since `7331841`, so that both sides use one estimator (R3) |
| `b34a825` | `set_parameters` with a zero weight sets `eta` without a warning of a division by zero (R3) |
| `e71eab2`, `3bc079f` | the rank-one test asserts the variance at levels 1 and 2 with an SD other than 1; the end-to-end search test passes the run's generator to its own GP fit; the failure branch of the one-dimensional search, a zero entry of `insigma`, the fallback bounds and the one-dimensional interval are tested against MATLAB's expressions (R1, R2) |
| `893d0ba`, `5b7881c`, `1443b39`, `73fb89c`, `2d931a6`, `ea47ff0` | documentation: the fixture of the Gaussian-mixture tests, `_sieve` and `var_ss`, the descriptions of `best_frac_back`, `adaptive_k`, `warp_cov_reg`, `output_fcn`, `annealed_gp_mean` and `constrained_gp_mean`, four comments of the search, and the oracle harness on the legacy global state (R1, R3, R4, R6) |

The PI ruled on the code findings on 2026-09-22 as the orchestrator
proposed them, and read with them what the fix agents decided where a
ruling left a choice. In the ranking branch, too, the last iteration is
taken when every ELCBO is NaN, since the stable ranks would otherwise
cancel into the earliest. The cached point is compared with the sieve's own
row rather than through a mask of the clip and a flag of the snap, which
would have called the target for a point that the snap returns exactly to
the cached one. A zero span of the space-filling schedule asks for no
points at every count, including a count short of `fun_eval_start`, where
MATLAB's cubic goes to infinity. And the names given to `load` join the
options the user set, which only the printed summary reads.

In the records they found, and this check corrected:

- `wave1_M_P2.md`: the rank-one row, whose premise about MATLAB fails on a
  noisy target (`gplite_post.m:76-79`; wave 4 found it and corrected the
  sheet alone); the correction of M's cause for the one-dimensional search,
  itself wrong (the message of `4949c82`, and `cma` raising at `D = 1` with
  bounds); P2 F10 at the noisy defaults; the count of the minor
  observations; a MATLAB citation; the revision read. `wave1_P6.md`: four
  citations and a dating. `wave1_options.md`: the count of inert options,
  three short; the label of `output_fcn`; which options take their value as
  an argument; `load` and `noise_shaping`. The README of
  `cmaes_side_by_side/`: four figures, none of which changes its reading.
  Each correction is a flag in the header of the file.
- The worklog: the counts (35 commits after `6c151cb`, 23 rows, eight minor
  observations, four citations), the hash of the flaky-test remedy
  (`29c822d`, the fix agent's commit, for `e48fade` on the branch), the
  "7 of the 8 states" of the re-baseline (the eighth holds no reference),
  the fixes that move default trajectories (`510a493` and `118626e` added,
  `7331841` stated in full), the soft-bound figures, which did not match the
  retained log of `soft_bounds_trace.py` (2.08 and 10.4 times the width, not
  2.2 and 12; on the calls where the two boxes differ a mean 0.008 box
  widths outside and a log scale 0.007 above its bound, not 0.07 and 0.008;
  the largest overshoot, 0.14 box widths, falls at a warm-up call where the
  boxes coincide, so the ruling stands), the reports of the wave-1 fix
  agents, which were not kept, and the working rule on the package a script
  imports from a worktree.
- `CHANGELOG.md`: the "Upgrading" line for `noise_shaping=True`, refused at
  construction and in `load`; the one-dimensional search, which replaces
  1.0.4's unbounded Nelder-Mead; the selection of the returned posterior,
  which applies whenever the last iteration is not stable and also chooses
  the posterior a warp starts from; entries for the two wave-1 fixes a user
  of 1.0.4 notices that had none, the run that `RuntimeError` stopped when
  the deterministic optimizer did not converge and the keyword of a callable
  `adaptive_k`; the warning for options without effect; and the entries of
  the commits above.
- `known_differences.md`: the reason for the one-dimensional search; three
  settings of the `cma` entry (a degenerate `insigma`, the handling of the
  bounds, the stop on a diverging step); the entry on options without
  effect (MATLAB declares `OutputFcn` and `NonlinearScaling` with no reader,
  four names were missing); the surplus starting points, which need no
  target call only with `f_vals`; the first search set, which MATLAB leaves
  short when the search cache is empty; and the entries of the commits
  above. `matlab_side_defects.md`: entries 54 and 55, and the source of
  entry 6. `counterpart_map.md`: the preparatory report named by its file,
  and the `noise_shaping` row. The Stage 2 plan on MATLAB's noise handling.

The items the round left, and the PI's rulings on them of the same day, as
the orchestrator proposed them. Fixed, each with a test that fails on the
code before it but for the two texts:

| commit | |
|---|---|
| `3d71b00` | the search set is sorted stably when a search cache is kept, as MATLAB's `sort` is (`private/activesample_vbmc.m:231-233`), so that of tied candidates, as candidates snapped to one point of an integer grid are, the earlier is acquired and the cache keeps their order; P2's third minor observation, and a fourth site of the stable order the check of wave 3 put into three |
| `2d8448e` | `Options` built from the options of another run, an `Options` object or a dict copied from one, keeps a set of user options of its own; it took over the other's and added every option name to it, so building a second run from `vbmc.options` changed the first (older than wave 1) |
| `e4ffa60` | a CMA-ES search whose step size is not finite is not started, and fails as a search that raises does; cma does not return from such a start. No known path of a run produces one |
| `422f774` | `get_parameters(raw_flag=True)` gives a zero weight the raw parameter minus infinity without NumPy's warning, as `set_parameters` does for `eta` |
| `f18a6c1` | `hpd_frac` is checked at construction and in `load`: a real fraction in `(0, 1]` that leaves at least two points of the initial design. Of ten points, a value below 0.15 left none or one, and the first GP fit failed, on an empty array or on the slice sampler's zero widths (both measured) |
| `ba0234c` | the texts say that `options["f_vals"]` are log-joint values, the prior already added when a separate prior is given, as in 1.0.4 and as MATLAB's `Fvals` are values of the function VBMC is given, where the `precomputed_evaluations` argument takes log-likelihood values and adds the prior. Changing `f_vals` instead would move, without a word, the results of a script that passes log-joint values |
| `f5b7efc` | `dev/scripts/pymc_setup_probe.py` says that its `max(D, 10)` is the ordinary initial design of the code its experiment ran on; the experiment sets the option explicitly and its report rests on those counts, so the number stays |

Kept as they are:

- Type 3 of `_vb_init` draws its permutation only when the means are
  optimized, where `vbinit_vbmc.m:86` draws it always. PyVBMC's random
  stream is not comparable draw for draw with MATLAB's in any case (the
  sheet's entry on randomness), and the draw would be unused; the case
  needs `variable_means=False`.
- After a warp with integer variables, the snap's round trip through the
  rotated transform changes an on-grid cached point in its last bits, so the
  point is evaluated rather than given its stored value. It needs integer
  variables, surplus starting points with values and a warp at once; a
  tolerance in the comparison would reopen the question W5-7 settled, which
  moved points may keep a stored value, for one evaluation.
- The test that recomputes `INERT_OPTIONS` counts a quoted
  `options["name"]` in a string as a read and accepts any mapping whose name
  ends in `options`; it matches the registry exactly today, and an AST scan
  in its place waits until the test is touched for another reason.
- The source hashes of `dev/scripts/torch_vi_step.py` are stale, a guard of
  parked scaffolding that fails closed; they are refreshed if that
  scaffolding is taken up again.
- The porting log `pyvbmc/vbmc/README.md` has no entry for the differences
  of slice P2; the durable entries of the sheet are consolidated there at
  the end of the review.
- The evidence of the wave-1 gates ("every module suite touched", the
  exact oracle check) was not kept, apart from the log of the options tests;
  the gates of this check stand in for it.
- The accuracy of the refreshed counts (`6590ea4`), which move noisy
  trajectories, is read on a benchmark sweep of noisy targets once the
  branch is on `dev-port-review` ("The merge into `dev-port-review`",
  below).

Gates, on the branch of the check (`dev-port-review-w1check`, cut at
`326c7676`), with the package of that checkout and gpyreg `main`:

- the test files of every module the commits touch, 288 passed and 2
  skipped after the first round and 526 passed after the second, and the
  option tests again, 79 passed, after the last change to the descriptions;
- `make_oracle_fixtures.py --check --exact --against` a dump of the base
  `326c7676`: the eight state fixtures bit-identical; the three
  `gp_fit_history` fixtures, which that check compares with their stored
  references alone, replayed at the base and at the head, 36 arrays and
  none different. Their stored references were re-baselined for gpyreg's
  branch of wave 6 and are not reached with gpyreg `main`;
- the four seeded runs of `scripts/wave2_fixpass_gate_runs.py`, recorded at
  the base `326c7676` and at the head of the fixes: the two noiseless runs
  bit-identical, and the two noisy ones moved, as `6590ea4` predicts (and
  `831b024` where an old posterior has one component), 36 of the 92 arrays
  differing, all theirs. Both keep their numbers of
  iterations and evaluations; their final ELBOs go from -1.635 ± 0.114 to
  -1.722 ± 0.111 (the noisy Rosenbrock, whose warm-up now ends at iteration
  6 where it ended at 7) and from 1.430 ± 0.081 to 1.308 ± 0.079. The
  targets of these runs have no recorded truth, so the accuracy of the
  change was not measured;
- after the commits on the items the round left: the test files they
  touch, 613 passed and 2 skipped on `f18a6c1`, and the option tests, 81
  passed, after the texts that follow it; at `35606cb`, the exact oracle check
  against the dump of the base, the eight state fixtures bit-identical and
  the replays of the three `gp_fit_history` fixtures identical to the
  base's; and the four seeded runs bit for bit those of `2d931a6`, 92
  arrays and none different.

Not run on the branch: the whole suite, the Torch and PyMC environments,
the CI matrix, and the refresh of the sheet's Python line citations
(`refresh_citations.py`, 125 to carry and 14 to read by hand on the
branch). All but the CI matrix, which runs with the push, ran on the
merged code (below).

## The merge into `dev-port-review`

The branch was merged into `dev-port-review` on 2026-09-23 as `e86bbb1c`,
after wave 6 had finished there and gpyreg 1.3.0, the release that carries
wave 6's fixes to gpyreg, was installed. Two files conflicted: the sheet,
where both sides had appended entries at the end of one section and both
were kept, and the plan, where the branch's worklog entry was put in date
order and the pickup point it had carried from its cut was dropped. The
sheet's Python line citations were then carried to the merged code
(`68e37d3e`): 113 by `refresh_citations.py` and six by hand, among them the
entry on the integrated mean function, whose copy into `optim_state`
`375df34` removed.

Gates, on `68e37d3e`, whose package code is the merge's, with gpyreg 1.3.0
and BLAS single-threaded:

- the four seeded runs of `scripts/wave2_fixpass_gate_runs.py`, recorded
  first on `403fb678`, the head of `dev-port-review` before the merge,
  where they are bit for bit the last record of wave 6 (92 arrays, 0
  differ), and then on the merged head: the two noiseless runs
  bit-identical, the two noisy ones moved in 40 of the 92 arrays, all
  theirs. The noisy Rosenbrock run ends after 31 iterations and 155
  evaluations, where it ended after 32 and 165, with a final ELBO of
  -1.669 +- 0.105 for -1.445 +- 0.101; the noisy two-blob run keeps its 17
  iterations and 90 evaluations, 1.249 +- 0.082 for 1.356 +- 0.080. On
  gpyreg 1.2.1 the same commits moved 36 arrays and left both runs their
  counts ("Gates" above); under 1.3.0 the runs start from other
  trajectories, its GP fit differing from 1.2.1's on every run (W6-1 of
  `wave6.md`);
- the exact oracle check, 11 of 11;
- the default suite, 2073 passed and 58 skipped, without reruns; the tests
  of the optional integrations, 841 passed and 19 skipped in the Torch
  selection (`svbmc`, `variational_posterior`, `parameter_transformer`,
  `function_logger`, `whitening`) and 110 passed in the PyMC adapter's,
  with the same counts in the two environments that ran them until then
  and in the one environment, holding Torch, ArviZ and PyMC, that replaced
  both;
- a benchmark sweep of noisy targets (`scripts/wave3_gate_benchmark_sweep.py`),
  before on `403fb678` and after on the merged head (its added seeds on
  `a65b96f4`, which differs from `68e37d3e` in the plan alone), with the
  same seeds on both sides: `rosenbrock_D2_noise1` over 10 seeds, and
  `rosenbrock_D2_noise3` and `logreg_D5_noise3` over 5 and then, on the
  PI's request after the losses at 5 seeds, over 10.

Means over the paired seeds of the sweep, before -> after:

| target | seeds | abs(ELBO - lnZ) | gsKL | MMTV | evaluations | seeds better / worse after (ELBO, gsKL, MMTV) |
|---|---|---|---|---|---|---|
| `rosenbrock_D2_noise1` | 10 | 0.1004 -> 0.0862 | 0.0863 -> 0.0817 | 0.0517 -> 0.0490 | 126.0 -> 127.5 | 7/3, 7/3, 7/3 |
| `rosenbrock_D2_noise3` | 10 | 0.1511 -> 0.1763 | 0.4548 -> 0.1718 | 0.1449 -> 0.0917 | 174.0 -> 173.5 | 4/6, 6/4, 6/4 |
| `logreg_D5_noise3` | 10 | 0.3776 -> 0.1426 | 0.4255 -> 0.4067 | 0.1723 -> 0.1688 | 239.0 -> 252.5 | 10/0, 5/5, 5/5 |

Over the 30 seeds the merge is better on 21 and worse on 9 by the error of
the ELBO, and better on 18 and worse on 12 by gsKL and by MMTV; two runs
of `rosenbrock_D2_noise3` end without a stable solution on each side. The
losses: on `rosenbrock_D2_noise3` the mean error of the ELBO, by 0.025 nats
against a reported ELBO SD of about 0.3, worse on 6 of the 10 seeds, while
its largest error is smaller (0.52 -> 0.41); on `rosenbrock_D2_noise1`
seed 10, whose gsKL goes from 0.004 to 0.57, above the largest before
(0.31, seed 9, which goes to 0.010); and on `logreg_D5_noise3` 13.5 more
evaluations on average. At 5 seeds `logreg_D5_noise3` was worse after on 4
of 5 by gsKL and by MMTV (means 0.345 -> 0.438 and 0.144 -> 0.180), which
seeds 6 to 10 did not bear out. On `logreg_D5_noise3` and
`rosenbrock_D2_noise3` the worst seed after is better than the worst
before by every metric (gsKL 0.69 -> 0.59 and 1.35 -> 0.83). The sweep
measures the merge as a whole; of its commits, `6590ea4` and `831b024` are
the ones that move a noisy run. The code before the merge reproduces
exactly the numbers of `rosenbrock_D2_noise1` that the after-sweep of W6-1
recorded. PI's ruling of 2026-09-23: the merge taken as it is. The logs
are on the orchestrator's machine (`dev/scripts/runs/LOCAL.md`).
