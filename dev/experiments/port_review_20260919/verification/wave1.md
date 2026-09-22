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

Left for the PI:

- Type 3 of `_vb_init` draws its permutation only when the means are
  optimized, where `vbinit_vbmc.m:86` draws it always; with
  `variable_means=False` the random streams part.
- The sort of the search set's fast acquisition values
  (`active_sample.py`, `np.argsort(acq_fast)`) is not stable, where
  MATLAB's `sort` is; P2's third minor observation. The stable order went
  into the three sites that the check of wave 3 found.
- `Options(..., user_options=<an Options object>)` shares and changes the
  source's set of user options, so passing one run's options to a new `VBMC`
  changes the first (older than wave 1). The test that recomputes
  `INERT_OPTIONS` counts a quoted `options["name"]` in a string as a read and
  accepts any mapping whose name ends in `options`; it matches the registry
  exactly today.
- A non-finite entry of `insigma` starts a CMA-ES search that does not
  return; the comment says so, and no reachable path produces one. A tiny
  `hpd_frac` makes `train_gp` raise, and construction does not check it.
- `get_parameters(raw_flag=True)` warns of a division by zero for a zero
  weight. After a warp with integer variables, the snap's round trip through
  the rotated transform changes an on-grid cached point in its last bits, so
  the point is evaluated rather than given its stored value, as before the
  check.
- Whether `f_vals` include the prior when `prior=` is given is documented
  nowhere. `dev/scripts/pymc_setup_probe.py` hard-codes `max(D, 10)` as the
  design size, and the source hashes of `dev/scripts/torch_vi_step.py` are
  stale, a guard that fails closed.
- The porting log `pyvbmc/vbmc/README.md` has no entry for the differences
  of slice P2; the durable entries of the sheet are to be consolidated there
  at the end of the review.
- The evidence of the wave-1 gates ("every module suite touched", the
  exact oracle check) was not kept, apart from the log of the options tests.

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
  change was not measured.

Not run on the branch, which waits to be brought onto `dev-port-review`:
the whole suite, the Torch and PyMC environments, the CI matrix, and the
refresh of the sheet's Python line citations
(`refresh_citations.py`, 125 to carry and 14 to read by hand on the
branch), which is to run against the merged code.
