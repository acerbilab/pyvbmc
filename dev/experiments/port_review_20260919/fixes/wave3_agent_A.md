# Wave 3 fix pass, agent A: the inputs of the GP hyperparameter fit

Raw report of a fix agent, wave 3 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent, in
its own git worktree cut from `dev-port-review` at `fc8e561`, fixed the
wave-3 findings in `train_gp` and `_gp_hyp` and the two lines of the main
loop that go with them: first the ones that leave default trajectories
alone (W3-9, W3-10, W3-8, W3-1), then the ones that move them (W3-2, W3-3,
W3-4, W3-5, W3-7), one finding per commit with a test written against the
contract, running only the focused test files it touched. The commit
hashes in the report are those of the agent's worktree branch; the
orchestrator reviewed each diff and cherry-picked the commits onto
`dev-port-review`, where they carry other hashes (`../verification/wave3.md`
lists them) and where the report's commits 7 and 12 are one. The text below
the rule is the agent's final message, unedited.

---

# Wave-3 fix pass, agent A

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a6b0522f1ae007bbe`
Branch: `worktree-agent-a6b0522f1ae007bbe` (cut from `fc8e5617`; nothing pushed)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..fc8e5617`; `git log -1 --format=%h` printed `fc8e5617`.
- `python -c "import pyvbmc; print(pyvbmc.__file__)"` from the worktree root printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-a6b0522f1ae007bbe\pyvbmc\__init__.py`.
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. No standalone script was needed (every check ran through `python -m pytest` from the worktree root or a one-line `python -c`). `git status --porcelain` is empty; nothing untracked is left (`--ignored` shows `__pycache__` only).

## `git log --oneline fc8e561..HEAD`

```
ed18f423 fix(vbmc): collect no starting points from a history that holds no GP
b85045d1 fix(vbmc): leave the largest observation noise to the training set
7c74492f fix(vbmc): leave the smallest constant of the GP mean to the training set
adaa610e fix(vbmc): record the GP hyperparameters of an iteration before thinning
0b7de721 fix(vbmc): thin an oversized pool of starting points to half the design
a78ac06e fix(vbmc): start the hyperparameter fit from the later half of the recorded GPs
d9e02574 fix(vbmc): scale the reported noise of a noisy target without its own estimate
1e3e3a58 fix(vbmc): let the length-scale cap of its option reach the GP
cec88d6b fix(vbmc): take the elementwise minimum for the output-dependent noise bound
b8a4a244 fix(vbmc): keep the hyperparameters of the last fit in the GP summary
8a893a08 fix(vbmc): clear the GP hyperparameter covariance under the name that is read
123ca2b4 test(vbmc): train the GP of the hyperparameter test on the bounds it runs with
```

Diffstat over the range: 5 files, +628 / −25. Source files touched: `pyvbmc/vbmc/gaussian_process_train.py` (`train_gp` and `_gp_hyp` only), `pyvbmc/vbmc/vbmc.py` (two lines: the `gp_hyp_full` record and the warm-up reset).

## Commits in order

Phase 1 is 1–6, phase 2 is 7–12.

| # | hash | finding | what it does | files | tests |
|---|---|---|---|---|---|
| 1 | `123ca2b4` | test note of W3-5 | `test_gp_hyp` calls `train_gp` with `optim_state["plb_tran"]`/`["pub_tran"]` as `VBMC` does, not the user-space plausible bounds (four times wider in that fixture) | `test_gaussian_process_train.py` | changed `test_gp_hyp` (its two assertions kept) |
| 2 | `8a893a08` | W3-9 (1st part) | the end-of-warm-up reset writes `hyp_dict["run_cov"]`, the key every producer and reader uses; no stray key is inserted | `vbmc.py`, new `test_gp_train_fit_inputs.py` | +`test_ending_warmup_clears_the_covariance_the_fit_reads` |
| 3 | `b8a4a244` | W3-9 (2nd, 3rd) | `hyp_dict["full"]`/`["logp"]` written after every fit, from that fit's result (the optimized vector as one row when no sampling); the covariance guard counts samples (`shape[0]`) | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_a_fit_without_sampling_holds_the_optimized_hyperparameters` |
| 4 | `cec88d6b` | W3-10 | `np.minimum` for `np.min(a, b)` in the lower bound of the rectified output-dependent noise | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_output_dependent_noise_is_bounded_by_the_training_values` |
| 5 | `1e3e3a58` | W3-8 | each of the two statements on the length-scale bounds replaces its own entry and carries the other over, so `upper_gp_length_factor` reaches the fit; the dead `-inf` lower bound goes | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_the_longest_length_scale_the_option_allows_reaches_the_fit` |
| 6 | `d9e02574` | W3-1 | `gp_noise_fun[1] > 0` translates into gpyreg's `user_provided_add`, `== 2` into the scale; the multiplier starts at `log(noisemult)`; the comment about the untested branch removed | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_the_noise_model_follows_the_uncertainty_level`, +`test_a_noisy_target_without_its_own_noise_scales_the_pooled_noise`, +`test_the_noise_multiplier_starts_where_matlab_starts_it` |
| 7 | `a78ac06e` | W3-2 | the window of recorded GPs is `range(ceil(n/2) - 1, n)`; comment states the correspondence with MATLAB's 1-based `ceil(n/2):n` | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_the_starting_points_come_from_the_later_half_of_the_history` (n = 1…6) |
| 8 | `0b7de721` | W3-3 | an oversized pool is thinned to `floor(init_N/2)` | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_a_large_pool_of_starting_points_is_thinned_to_half_the_design` |
| 9 | `adaa610e` | W3-4 | `iteration_history["gp_hyp_full"]` records `hyp_dict["full"]`, the chain before thinning | `vbmc.py`, `test_vbmc_loop_state.py` | +`test_the_recorded_hyperparameters_are_the_chain_before_thinning`; module-scoped fixture `two_iterations` added and `test_running_moments_...` switched to it, so the module still makes one run |
| 10 | `7c74492f` | W3-5 | the lower bound of `mean_const` is NaN in both mean branches | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_the_constant_of_the_mean_keeps_a_lower_bound` |
| 11 | `b85045d1` | W3-7 | the upper bound of `noise_log_scale` is carried over (NaN) where the floor is raised | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_the_scale_of_the_observation_noise_keeps_an_upper_bound` |
| 12 | `ed18f423` | W3-2 (completes 7) | the start of the window is floored at the first record, so a history holding no GP collects nothing | `gaussian_process_train.py`, `test_gp_train_fit_inputs.py` | +`test_a_history_holding_no_gp_collects_no_starting_points` |

**Commit 12 belongs with commit 7 and should be squashed into it when landing.** `range(ceil(n/2) - 1, n)` is `range(-1, 0)` for `n = 0` and reaches for the last entry of an empty array; MATLAB is protected by `~isempty(stats)` (`misc/gptrain_vbmc.m:38`). No run trains a GP in a later iteration with no GP recorded, but the **oracle harness hands `train_gp` exactly that**: `pyvbmc/testing/oracles/_oracles.py:573` builds the `gp_fit` stand-in history with `"gp": np.array([], dtype=object)` and `optim_state["iter"] >= 1`, where commit 7 alone raises `IndexError`. I could not amend commit 7 in place (`git reset --hard` is refused in this worktree), hence the separate commit.

## Contract and test, per commit

1. **Test note (W3-5).** `train_gp`'s docstring types both bounds as transformed; `VBMC` passes `optim_state["plb_tran"]`/`["pub_tran"]` at both call sites. In the fixture `x0 = 3` lies outside the given plausible box, so the user-space bounds are widened to `[-1, 3]` against a transformed `[-0.5, 0.5]`.
2. **W3-9, the key.** `vbmc.m:831` (`hypstruct.runcov = []`) against `misc/get_GPTrainOptions.m:59`, which reads that field. Test: a four-iteration seeded run with the end of warm-up forced (`_check_warmup_end_conditions` patched, `stop_warmup_reliability = inf`) and `train_gp` wrapped at the `vbmc` module; the last warm-up fit must see a covariance and the first main-algorithm fit must see `None`, and `hyp_dict` must carry no key beyond `{hyp, warp, logp, full, run_cov}`. Before: the covariance survived.
3. **W3-9, the block and the guard.** `misc/gptrain_vbmc.m:65` assigns unconditionally, `gplite/gplite_train.m:465` makes the block the single optimized vector when `Ns = 0`, `:83` tests the number of samples. Test: two `train_gp` calls on one state, the second with `stop_sampling` set; the first must hold `gp_s_N * thin` rows and a covariance, the second exactly the fitted vector, `run_cov is None` and `logp is None`. Before the block fix: `(40, 9)` rows where `(1, 9)` was due; with the block fixed and the old guard: `run_cov` an all-NaN matrix.
   **What I chose for `logp`:** the log prior density of each row of `full`, which is what gpyreg's sampler returns, and `None` when there is no chain. gpyreg's `fit` reports no density for the optimized vector (MATLAB keeps `-nll` there), and nothing reads the field on either side, so I did not reach into the optimizer result for it. Stated in the commit body.
4. **W3-10.** `misc/gptrain_vbmc.m:235-236`. Test: a GP with the rectified noise term built by hand (nothing switches `gp_noise_fun[2]` on), `_gp_hyp` called on a real state; both entries checked against the MATLAB formulas. Before: `TypeError: 'numpy.float64' object cannot be interpreted as an integer`.
5. **W3-8.** `misc/gptrain_vbmc.m:177-179` assigns `UB_gp(1:D)` and `LB_gp(1:D)` into separate NaN vectors. Test: with `upper_gp_length_factor = 3` the installed and the gpyreg-filled upper bounds equal `log(3*(PUB-PLB))`; at the default the entry is NaN, the filled value is gpyreg's recommendation (different from the cap), and the lower bounds are the same in both runs. Before: `[nan, nan]` installed.
6. **W3-1.** `misc/setupvars_vbmc.m:279` sets `[1 2]`; `gplite/gplite_noisefun.m:66-70`, `:186-194` make that two hyperparameters and `exp(2*h1) + exp(h2)*s2`; `misc/gptrain_vbmc.m:165` starts the multiplier at `log(noisemult)`, `:151-159`/`:213-218` give the prior. Three tests: the three levels' `gp.noise.parameters` and hyperparameter names against `optim_state["gp_noise_fun"]` (a real fit each, sampling switched off); at level 1 the per-row noise follows `s2` and a point evaluated four times carries less noise than one evaluated once; the starting point and both hyperpriors, with and without `noise_size` (the default case alone would not have caught a missing starting value, since gpyreg's own `x0` for the multiplier is `log(1)` too). Before: the level-1 model had one noise hyperparameter, so the first two tests failed and the third raised.
7. **W3-2.** `misc/gptrain_vbmc.m:39`. Test: `n = 1…6` recorded GPs whose hyperparameters are all equal to their iteration index, `GP.fit` and `_get_gp_training_options` stood in for; the set of starting points must be the summary vector plus exactly `range(ceil(n/2) - 1, n)`. Before: the oldest GP of the window was missing for even `n`.
8. **W3-3.** `misc/gptrain_vbmc.m:44`. Test: six recorded GPs (a window of four) with `init_N = 5`; the number of starting points must be `floor(5/2) + 1 = 3`. Before: 4.
9. **W3-4.** `misc/gptrain_vbmc.m:65` with `vbmc.m:804`, `:1041`. Test, on the module's shared two-iteration run: for every recorded iteration the block has `Ns_gp * gp_sample_thin` rows, the same width as the GP's samples, and its every `thin`-th row is exactly what the GP kept. Before: the block was the thinned set.
10. **W3-5.** `misc/gptrain_vbmc.m:174-188` assigns the upper entry alone, `gplite/gplite_train.m:120-127` fills the rest. Test, for the negative-quadratic and the constant mean: the installed lower entry is NaN, the upper is MATLAB's value, and after gpyreg's fill the lower equals gpyreg's recommendation while the upper is still MATLAB's. Before: `-inf` installed and kept.
11. **W3-7.** `misc/gptrain_vbmc.m:180`. Test: the installed floor is `log(tol_gp_noise)` and above gpyreg's recommended floor, the upper entry is NaN, and after the fill it equals `log(max(y) - min(y))`, gpyreg's recommendation. Before: `+inf` installed and kept.
12. **W3-2, the empty history.** Test: `optim_state["iter"] = 1` with `iteration_history["gp"]` an empty object array; the only starting point is the summary vector. Before: `IndexError: index -1 is out of bounds for axis 0 with size 0`.

Every fix was seen to fail first, by reverting the source hunk (Edit, run, Edit back) or by running the new test on the unfixed code.

## Test commands run, and results

All from the worktree root with `PY = C:/Users/luigi/Documents/GitHub/pyvbmc/.venv/Scripts/python.exe`, the three thread variables, `-q -p no:cacheprovider`:

- `PY -m pytest pyvbmc/testing/vbmc/test_gp_train_fit_inputs.py` — **12 passed** (~5 s)
- `PY -m pytest pyvbmc/testing/vbmc/test_gaussian_process_train.py` — **12 passed** (~2 s)
- `PY -m pytest pyvbmc/testing/vbmc/test_vbmc_loop_state.py` — **5 passed** (~7 s)
- Final combined run of the three files at HEAD: **29 passed in 12.4 s**

No whole test directory, no oracle or golden command, no benchmark, no install. The one `optimize()` a test of mine runs is the four-iteration `D = 2` run of commit 2 (~4 s), and commit 9's test shares the module's existing two-iteration run instead of adding one.

## Changelog sentences (for a user of 1.0.4)

- **`d9e02574` (W3-1):** With `uncertainty_handling=True` and no target-provided noise, the GP now scales the noise it records for each evaluation by a fitted multiplier, as the algorithm prescribes, instead of ignoring it; such runs change from their first GP fit, and repeated evaluations of a point now weigh more than single ones.
- **`1e3e3a58` (W3-8):** The option `upper_gp_length_factor` now caps the GP input length scales as documented; at its default of 0 nothing changes.
- **`8a893a08` + `b8a4a244` (W3-9):** The running covariance of the GP hyperparameters is cleared at the end of warm-up and rebuilt only from fits that actually sampled, which changes the width of the hyperparameter sampler for runs with `weighted_hyp_cov=False`.
- **`a78ac06e` + `ed18f423` (W3-2), `0b7de721` (W3-3), `adaa610e` (W3-4), `7c74492f` (W3-5), `b85045d1` (W3-7)** (one sentence each, all "results differ slightly from 1.0.4 at the default options"):
  - The GP hyperparameter fit starts from the recorded GPs of the later half of the run's iterations, one iteration further back than before when an even number is recorded.
  - A pool of collected starting points larger than the fit's design is thinned to one point fewer than before.
  - Each iteration records the GP hyperparameters the fit drew before thinning, five times as many as before, which makes the covariance that sets the next sampler's widths less noisy.
  - The smallest constant of the GP mean function and the largest observation-noise scale are left to the bounds recommended from the training set, where they used to be unbounded; the fit's starting points and its search are confined accordingly.
  (`mean_const` and `noise_log_scale` can be one sentence together if that reads better.)

Commits `123ca2b4` (a test) and `cec88d6b` (unreachable code) are not user-visible.

## What I stopped on

Nothing blocked. Two things I did not do, by the rules:

- **The second hyperparameter of the rectified output-dependent noise** keeps `(-inf, +inf)` in `_gp_hyp` (`bounds["noise_rectified_log_multiplier"]`), where `misc/gptrain_vbmc.m:235-236` sets only the entries of the *first* one and leaves the second unset — the same defect as W3-5 and W3-7, in the same dead branch. It is not in my brief, and my W3-10 test checks only the two bounds MATLAB writes, so I left it. One character each (`np.nan`) if the orchestrator wants it closed with W3-5/W3-7.
- **The comment placement in `_gp_hyp`.** The sentence "Each statement here replaces one bound of a hyperparameter and carries the other one over…" landed inside the `if options["upper_gp_length_factor"] > 0:` block (commit 5), where it describes the whole bounds section. It reads correctly but would sit better just above `bounds = gp.get_bounds()`. Moving it touches the W3-8 hunk, so I left it for the orchestrator's review pass.

## What else I noticed

**The noise function with two hyperparameters and an `s2`, as asked (W3-1).** I read `_estimate_noise` and the callers; nothing mishandles it.
- `_estimate_noise` (`gaussian_process_train.py:872-879`) slices `hyp[cov_N : cov_N + noise_N]` with `noise_N` from `hyperparameter_count()` and passes the subset's `s2`, so it evaluates `exp(2*h1) + exp(h2)*s2` per row correctly.
- `vbmc.py:1330` (the warp-undo refit) and `:1473` (the loop) pass the state through; nothing there is noise-specific. `vbmc.py:1646` calls `reupdate_gp`, which only replaces `X`, `y`, `s2` and recomputes.
- `active_sample.py:746` (`train_gp`), `:759` and `:814` (`reupdate_gp`) likewise; the rank-one update at `:809` goes through `gpyreg`'s `GP.update`, which computes the new point's noise with `noise.compute(hyp_noise, X_new, y_new, s2_new)` — generic in the hyperparameter count.
- `active_sample.py:332-345` (the per-point noise for the acquisitions) and `whitening.py:309` also index through `hyperparameter_count()`. One thing worth knowing: `active_sample.py:337` evaluates the noise at `s2 = S**2 * n_evals`, the variance of a *single* observation, where the fit receives `S**2`, the pooled one. That is `private/activesample_vbmc.m:165` verbatim, so it is shared with MATLAB, not a discrepancy — but at level 1 it becomes visible for the first time, because the multiplier now multiplies it.

**Readers of `iteration_history["gp_hyp_full"]` (W3-4), and whether the change reaches them.**
- `pyvbmc/vbmc/gaussian_process_train.py:762` (`_get_hyp_cov`) — **affected, and this is the point of the fix**: the weighted covariance is estimated from `thin` times as many rows. It accepts a block of any row count ≥ 1 whose width matches the model, so both the new records and those of runs saved by earlier versions are read correctly.
- `pyvbmc/testing/oracles/_oracles.py:573` (`gp_fit`) and `:620` (`gp_fit_history`) — **not affected**: both build synthetic histories from the stored snapshot (`[hyp_prev] * n` and a hand-built ragged one), not from what `optimize()` records.
- `pyvbmc/testing/oracles/_gp_fit_history.py:126` (`capture_inputs`) and `:282` — **not affected now**; it copies `history["gp_hyp_full"]` when a *fixture is generated*, so a future regeneration would capture pre-thinning chains and change the `gp_fit` inputs.
- `pyvbmc/testing/oracles/test_oracles.py:154` — reads the synthetic ragged history; not affected.
- `dev/scripts/golden_trace.py:199` — **affected**: the stored `gp_hyp` array of a trace grows by the thinning factor. The golden references are regenerated after this pass anyway.
- `pyvbmc/testing/vbmc/test_gaussian_process_train.py:196, 260, 277, 290` — hand-built histories for `_get_hyp_cov`; not affected (all 12 tests in that file pass).

**Oracle expectations for the orchestrator.** Phase 2 (commits 7, 8, 10, 11 and the floor of 12) changes the starting points and the bounds of every fit, so `gp_fit` moves and `active_sample_step` moves with it. Phase 1 should leave every reference bit-identical at the default options, with one thing to watch: commit 3 changes `hyp_dict["full"]`, `["logp"]` and `["run_cov"]` **for a fit that does not sample**, and `_gp_fit_history.fit_outputs` writes `hyp_dict_full`, `hyp_dict_logp` and `hyp_dict_run_cov` into the `gp_fit` outputs only when they are not `None`. If any stored snapshot's fit has `gp_s_N == 0`, that oracle's output set changes (`full` becomes the one fitted row, `run_cov` and `logp` disappear). Every snapshot I know of is an early-iteration state that samples, so I expect no movement, but it is the one phase-1 place where `--check --exact` could speak up.

**Cherry-pick order.** The tests share helpers in `pyvbmc/testing/vbmc/test_gp_train_fit_inputs.py`: `build_trained_state`, `training_data`, `fit_the_gp` and `build_short` come with commit 2, `default_gp` and `install_hyperparameters` with commits 4–5 (`install_hyperparameters` gains its `hyp0` return in commit 6, which also updates its two call sites), and `RecordedGP`, `record_marked_gps` and `capture_starting_points` with commit 7. Commits 8 and 12 use commit 7's helpers, and 10 and 11 use commit 5's. Landing them out of order needs care; in the given order they apply cleanly.

**A test-adequacy note for the ledger.** `test_gaussian_process_train.py::test_gp_hyp` draws its training points from the unseeded global generator (`np.random.rand`). I left that alone (outside the one change my brief asked for there), but it is the only test in that file that is not deterministic.
