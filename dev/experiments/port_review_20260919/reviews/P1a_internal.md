# P1a — main loop, warmup, termination, final boost, internal-correctness track

Raw reviewer report, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent,
fresh context, read-only on the repositories; it did not read the MATLAB
source. Its check scripts ran in the session scratchpad and are not
retained. The text below is the agent's final message, unedited.

---

# Slice P1a, internal-correctness track — review report

Repository `C:\Users\luigi\Documents\GitHub\pyvbmc`, branch `dev-port-review` at
`51451dc`. No MATLAB source was read. All line numbers are at this checkout.

(Note: I was unable to save `REPORT.md` — the harness refuses report files from
subagents — so the full text is here. The check scripts referenced below are in
`C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\b8a4c56e-a31c-47b4-87d0-66a2c9861ded\scratchpad\port_review\P1a_internal\`.)

## 1. Coverage

### Read completely, line by line

- `pyvbmc/vbmc/vbmc.py: optimize` (1114–1942), including the continuation
  block (1161–1181), the warping branch (1231–1381), the active-sampling and
  GP-training call sites (1383–1493), the variational-optimization call site
  and the `update_K` call (1495–1546), the finalize block (1548–1598), the
  recording into `iteration_history` (1600–1625, 1815–1823), the warmup and
  termination blocks (1627–1671), the output-warping threshold block
  (1673–1700), the display and plotting blocks (1702–1813), and the
  post-loop selection, boost, messages and result construction (1825–1942).
- `_check_warmup_end_conditions` (1943–2023), `_setup_vbmc_after_warmup`
  (2025–2089).
- `_check_termination_conditions` (2091–2206), `_compute_reliability_index`
  (2208–2264), `_check_gp_sampling_stop` (2266–2274),
  `_is_gp_sampling_finished` (2276–2329), `_ensure_gp_sampling_history`
  (2331–2401), `_recompute_lcb_max` (2403–2408).
- `final_boost` (2412–2565) with `_validate_final_boost_tolerance` (2584),
  `_is_valid_final_boost_score` (2600) and `_accept_final_boost_candidate`
  (2613).
- `determine_best_vp` (2626–2745), `get_gp` (2747–2793),
  `_compute_true_diagnostic` (3061–3091), `_create_result_dict` (3119–3178),
  `_log_evaluation_budget_summary`, `_log_column_headers`,
  `_setup_logging_display_format`.
- `pyvbmc/vbmc/option_configs/basic_vbmc_options.ini` and
  `advanced_vbmc_options.ini` in full.
- `pyvbmc/vbmc/iteration_history.py` in full (the store the loop writes to).
- `dev/experiments/port_review_20260919/known_differences.md` in full.
- Papers: `acerbi2018variational_appendix.md` §B.4 (reliability index and
  long-term stability), `acerbi2020variational_appendix.md` §B.1–B.2
  (modified reliability index, variational whitening schedule);
  `acerbi2018variational_main.md` p. 235.
- `pyvbmc/vbmc/README.md` (porting log), `docsrc/source/api/classes/vbmc.rst`,
  `docsrc/source/faq.md` (the display, troubleshooting, resume and
  reproducibility sections).

### Read as an interface (signature, returns, and the state written/read)

- `active_sample` (`pyvbmc/vbmc/active_sample.py`), only its return contract
  and the `optim_state` keys it touches around the call site.
- `train_gp`, `_lean_gp`, `_restore_gp_posteriors`, `reupdate_gp`,
  `_get_gp_training_options` (only the `optim_state["max_fun_evals"]`
  reader at `gaussian_process_train.py:624`).
- `optimize_vp` and `update_K` (`pyvbmc/vbmc/variational_optimization.py`),
  read far enough to establish what they mutate on the input posterior and
  that the returned posterior is a fresh copy.
- `warp_input` and `warp_gp_and_vp` (`pyvbmc/whitening/whitening.py`), read
  for what the warping call site hands them and what they leave in
  `optim_state`.
- `gpyreg.GP.predict` / `predict_full` (`../gpyreg/gaussian_process.py`), to
  settle what `y_star`/`s2_star` do when `add_noise=False`.
- `VariationalPosterior.kl_div`, `moments`, `Options`, `pyvbmc/vbmc/options.py`
  (`INERT_OPTIONS`, `_warn_inert_options`), `pyvbmc/timer/timer.py`,
  `pyvbmc/vbmc/_runtime_tips.py`.

### Not reached

- The bodies of `active_sample`, `train_gp`, `optimize_vp`, the acquisition
  functions and the whitening arithmetic (other slices).
- `VBMC.__init__`, the bounds setup, `_init_log_joint`, `save`, the full body
  of `load` (P1b). `_init_optim_state` was read in full, but only to
  establish the state my slice starts from; I report on it only where the
  loop's reads and its writes disagree.
- No MATLAB source (track boundary).

### Checks run

All in the scratch directory; `pyvbmc.__file__` was printed and confirmed to
be this checkout on every run.

1. `check1.py` — built a fresh `VBMC` (D=5) with `entropy_switch=True` and
   inspected `optim_state`. Output: `'entropy_force_switch' in optim_state:
   False`, `optim_state.get('entropy_force_switch') = None`,
   `options['entropy_force_switch'] = 0.8`, and the product the loop forms
   raises `TypeError: unsupported operand type(s) for *: 'NoneType' and
   'int'`. Settles **F1**. The same script evaluated the `run_mean`/`run_cov`
   condition of `optimize` on a populated state: it returns 5 (truthy) for
   D=5 and is truthy for every D ≥ 1. Settles **F2**.
2. `check2.py` — loaded `pyvbmc/testing/vbmc/test_vbmc_save_static.pkl`
   read-only with `dill`. Confirmed that `last_run_avg == N` at every one of
   the 7 recorded iterations (the running-average branch never ran, **F2**),
   that `lcb_max_vec` is `[]` in every record (**F6**), and that every
   recorded posterior carries `stats["stable"] = False`.
3. `check3.py` — evaluated the GP-training initial-design schedule of
   `gaussian_process_train.py:620–638` with a stale versus a refreshed
   `max_fun_evals`. At `n_eff = 700`, stale limit 300 gives `init_N = 9`,
   refreshed limit 1000 gives `init_N = 91`. Settles the magnitude of **F7**.
4. `check4.py` — `VBMC.load` of the static pickle, with and without
   `new_options={"max_fun_evals": 999}`. Output: `optim_state['max_fun_evals']
   = 40` against `options['max_fun_evals'] = 999`. Settles **F7**. The same
   check refuted a suspicion I had raised: after `load`,
   `vbmc.function_logger`, `vbmc.optim_state`, `vbmc.vp` and `vbmc.gp` are
   *not* the iteration-history entries (the history truncation loop goes
   through `IterationHistory.__setitem__`, which deep-copies), so a continued
   run does not mutate the records of earlier iterations.
5. `check5.py` / `check6.py` — called `_check_termination_conditions` on a
   constructed D=3 instance with `max_iter=1`, `max_fun_evals=20`,
   `func_count=1000`. It reports `is_finished=False` for iteration indices
   0, 1 and 2 and `True` from index 3, i.e. 4 iterations at `min_iter=3`, and
   it overrides both `max_iter` and `max_fun_evals` to do so. Settles **F4**.
6. `check7.py` — built bounded, half-bounded and unbounded instances under
   both `bounded_transform` values and evaluated the `problem_type` test of
   `_create_result_dict`. `lb_tran`/`ub_tran` are `±inf` in every case, so
   the answer is `"unconstrained"` in every case. Settles **F3**.
7. `check8.py` — confirmed that the 7 posteriors in the static pickle's
   history each carry their own `ParameterTransformer` object (7 distinct
   ids), so a posterior recorded before a warp keeps the pre-warp transform.
   Supports **F8**.
8. `check9.py` — scanned the two shipped `.ini` files for declared option
   names that never appear as `options[...]`, `options.get(...)` or
   `options.eval(...)` anywhere in the package outside `options.py` and the
   test suite. Result: `entropy_force_switch` (only as an `optim_state` key)
   and `temperature` (only as an `optim_state` key); `diagnostics` appears
   only as a local sampler-options key in
   `active_importance_sampling.py:437`. Settles **F9**.
9. `check10.py` — called `_check_warmup_end_conditions` with
   `tol_stable_warmup=5`, `fun_evals_per_iter=5` and a three-entry history.
   Raises `ValueError: zero-size array to reduction operation maximum which
   has no identity`. Settles **F10**.

No `optimize()` run, no test-suite run, no writes into the repository.

### Verified correct, no finding

Stated explicitly, so that the absence of a finding is informative.

- **Reliability index against the papers.** `_compute_reliability_index`
  (2208) computes exactly the three features of
  `papers/acerbi2018variational_appendix.md:340–366`:
  `|E[ELBO(t)] − E[ELBO(t−1)]|/Δ_SD`, `sqrt(V[ELBO(t)])/Δ_SD`, and
  `sKL/Δ_KL` with `sKL = ½(KL(q_t‖q_{t−1}) + KL(q_{t−1}‖q_t))` formed at
  1556–1565, and averages them. `Δ_KL = tol_skl = 0.01·sqrt(D)` matches the
  paper. The noise-adapted `Δ_SD` at 2222–2229 is algebraically
  `min[1, max[0.1, sqrt(Δ_SD^base·σ_hpd)]]` with `Δ_SD^base = tol_sd = 0.1`,
  which is Eq. S10 of `papers/acerbi2020variational_appendix.md:141–147`
  (`sqrt(sn/0.1)·0.1 = sqrt(0.1·sn)`), then clamped by `max(tol_sd, ·)` and
  `min(·, 10·tol_sd)`. The `iter < 2` early return (2214) is the 0-based
  counterpart of MATLAB's `< 3`, as its comment states.
- **The stability termination window.** `iteration + 1 >= tol_stable_iters`
  (2166) guarantees the slice `[iteration − tol_stable_iters + 1 : iteration]`
  (2172–2176) starts at index ≥ 0 and holds exactly `tol_stable_iters − 1`
  past iterations, excluding the current one, as the comment says. The
  acceptance count `tol_stable_iters − floor(tol_stable_iters ·
  tol_stable_excpt_frac) − 1` (2179–2186) is reachable (9 of 11 at the
  shipped defaults) and degenerates correctly to "no exceptions" at
  `tol_stable_excpt_frac = 0`. The entropy-switch branch (2169–2173) consumes
  the stability event without declaring the iteration stable, and the
  post-warp delay `iteration − last_successful_warping >= tol_stable_iters/3`
  (2176–2178) is well-defined at the `-inf` initial value.
- **`tol_stable_iters` selection.** `tol_stable_entropy_iters` when the
  entropy switch is on, otherwise `ceil(tol_stable_count /
  fun_evals_per_iter)` (2124–2132) — the quantity the option comments
  describe.
- **GP-sampling stop.** `_check_gp_sampling_stop` (2266) is called after the
  iteration is recorded, so `N` and `var_ss` for the current iteration are
  available; its guards agree with the ones `_is_gp_sampling_finished`
  re-checks (2283–2291). The weights at 2316–2322 are
  `0.5·onehot(last) + 0.5·normalize(exp(−(N_last − N_i)/10))`, the
  max-subtraction is a no-op that cannot change them (the maximum is always
  the last entry), and every path that cannot form finite weights or a
  finite history returns `False` rather than stopping. This matches the
  "Settled non-differences" entry on the GP-sampling criterion.
  `_ensure_gp_sampling_history` (2331) backfills `N` from the recorded
  `optim_state` and then from the recorded `FunctionLogger.Xn + 1`, leaves a
  `None` where neither is usable, and restores `check_keys`.
- **`final_boost` acceptance arithmetic.** `_accept_final_boost_candidate`
  (2613) computes `min(dE, dE − 5·dS) > −tol`, which is the exact minimum
  over `b ∈ [0,5]` of the linear function `b ↦ dE − b·dS`, as the sheet's
  P1a entry describes. `_validate_final_boost_tolerance` rejects booleans,
  non-reals, non-finite values and negatives; `_is_valid_final_boost_score`
  requires both scores finite and `elbo_sd >= 0`. The four combinations of
  candidate/pre validity are all handled and the rejected path returns the
  untouched pre-boost copy with `changed_flag = False`. The boost-only
  options are set on a `copy.deepcopy(self.options)` with `force=True`, so
  the instance's frozen options are not modified; `max_iter_stochastic =
  np.inf` is consumed safely by `min(10000, ...)` at
  `variational_optimization.py:279`.
- **`determine_best_vp`.** Both call sites (1261, 1826) pass
  `rank_criterion`, `best_safe_sd` and `best_frac_back`; the ranking branch
  reads the object-dtype history through `np.asarray` and converts the flags
  with `dtype=bool` before using them as a mask; the four rank columns are
  each a permutation of `1..max_idx+1` with the best scoring 1 (recency
  reversed at 2668, ELCBO descending at 2695–2697, reliability ascending at
  2700–2701); the non-stable penalty and the look-back window both use
  `max_idx + 1`, the number of iterations. The fallback branch starts at the
  last stable iteration when there is one and otherwise
  `max(0, max_idx − ceil(n_iterations · frac_back))`. This is the corrected
  code described in the sheet, and I found no defect in the correction.
- **`get_gp`.** It deep-copies the record before restoring the factors
  (2793), so the stored record keeps its lean form; a record that already
  carries factors is returned unchanged (`_restore_gp_posteriors` only acts
  when some `alpha` is `None`); the index guard rejects negatives, indices
  past the end and `None` entries with the message the docstring promises.
  `_lean_gp` builds its fresh `posteriors` on a shallow copy without
  disturbing the live GP, and `IterationHistory.record` deep-copies what it
  is given.
- **`_compute_true_diagnostic`.** It validates the shapes of `true_mean` and
  `true_cov`, treats empty and non-finite values as "no diagnostic", and
  isolates the draws by replacing the shared generator on the working copy
  with a deep copy of it (3086–3087) — the only place in the loop that must
  not advance the run's stream, and it does not.
- **Randomness.** `optimize` contains no `np.random` call. Every draw goes
  through `vp.rng` (which is `vbmc.rng`): `vp.kl_div` → `moments` → `sample`,
  `vp.sample` in the output-warping block (1684), `warp_input`'s two
  `vp.rng.random` calls, `train_gp(rng=self.rng)` at 1300 and 1474, and
  `optimize_vp`. The startup tip uses a private `random.Random` instance
  (`_runtime_tips.py:15`), as `optimize`'s docstring says. The two
  `_get_random_state` snapshots (1817, 1941) and `results["rng_state"]`
  (3157) are all taken after the last draw.
- **Iteration counter.** `self.iteration` starts at `-1` (429) and is
  incremented at the top of the loop, so `optim_state["iter"]` is the 0-based
  index of the current iteration; `iteration_history` arrays grow to exactly
  `iteration + 1` entries, which the slices in `_check_warmup_end_conditions`
  and `determine_best_vp` rely on. `max_iter` is correctly compared against
  `iteration + 1` (2116). On a continuation the counter resumes from the
  stored value and `max_iter`/`max_fun_evals` are totals, as
  `docsrc/source/faq.md` describes.
- **`update_K` call site.** `update_K` (1503–1505) runs before
  `recompute_var_post` is consumed (1512–1524), so it sees the flag that
  describes the current iteration; it reads `iteration_history[...][-1]`,
  which at that point is the previous iteration; `optim_state["vp_K"]` is
  refreshed from `self.vp.K` after `optimize_vp` (1535), so the two never
  drift. `N_fastopts` is evaluated at the current `vp.K` on purpose (settled).
  The `optimize_mu = False` branch bypasses `update_K` and takes `K` from the
  GP training set, as intended.
- **Warping branch bookkeeping.** `vp_old` is copied before the branch, so
  the end-of-iteration `sKL` compares against the pre-warp posterior; the
  undo (1369–1381) restores `vp`, `gp`, `optim_state`, `function_logger` and
  `hyp_dict` from copies taken before `warp_input`, which also restores
  `last_successful_warping`, `skip_active_sampling` and `recompute_var_post`,
  and then applies the double `warping_count` penalty and re-stamps
  `last_warping`. The transformer identity invariant is restored immediately
  afterwards at 1384–1389 on both paths. The undo test itself
  (`elbo < elbo_old + warp_tol_improvement` or `elbo_sd > elbo_sd_old ·
  warp_tol_sd_multiplier + warp_tol_sd_base`) is the conjunction its comment
  describes, and compares like with like: both `elbo_old` and the post-warp
  `elbo` describe the last iteration's posterior. The warp gate matches
  `papers/acerbi2020variational_appendix.md:194` in the two respects it
  states: the increasing interval `warp_every_iters · max(1, warping_count)`
  and the reliability precondition `r_index < warp_tol_reliability = 3`.
- **Post-warmup trimming.** In `_setup_vbmc_after_warmup`, rows past `Xn`
  are `NaN` and so are excluded by `(y_max − y_orig) < threshold`; the
  `n_keep_min = D + 1` fallback sorts with non-finite values mapped to
  `-inf`, caps at `Xn + 1` and indexes correctly through the `(N,1)` shape;
  the final `logical_and` with `X_flag` prevents a trimmed row from being
  revived. Because `y_max` never decreases and the second threshold is
  smaller than the false-alarm one, a row excluded by an earlier trim cannot
  be counted toward `n_keep_min` at a later one. `reupdate_gp` is called
  right after the trim (1650), and `optim_state["N"]` deliberately keeps
  counting trimmed rows, consistent with the sheet.
- **Warmup/termination ordering.** `_check_termination_conditions` records
  `r_index` and `stable` for the current iteration before
  `_setup_vbmc_after_warmup` reads `r_index[iteration]` (2036) and before
  `warmup` is recorded (1671), and the comment at 1668–1670 correctly
  explains why the `warmup` record must come after the block.
- **Result fields other than the two below.** `func_count` is the literal
  fresh-call count as `docsrc/source/api/classes/vbmc.rst:86` promises;
  `best_iter`, `r_index` and `train_set_size` index the history at
  `idx_best`; `convergence_status` and `success_flag` are both
  `iteration_history["stable"][idx_best]` and so cannot disagree; `message`
  is the last termination message; `elbo`/`elbo_sd` are the returned
  posterior's own statistics, boosted when the boost was accepted;
  `evaluation_budget` appears only when a charge was made. `overhead = NaN`
  and the generator-snapshot `rng_state` are settled sheet entries.
- **Logging.** The three display formats have 9, 9 and 8 fields and the
  three call sites pass 9, 9 and 8 arguments; `_log_column_headers` selects
  the matching header under the same conditions.

## 2. Findings

### F1. The forced entropy switch reads `entropy_force_switch` from `optim_state`, where it is never written, and crashes the run
- Location: `pyvbmc/vbmc/vbmc.py:1224-1230` (the read at `:1226`); option
  declared at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:211`;
  MATLAB: "no counterpart read" (internal track)
- Category: cross-module
- Proposed classification: suspected defect
- Confidence: high
- The block is

  ```python
  if self.optim_state.get("entropy_switch") and (
      self.function_logger.func_count
      >= self.optim_state.get("entropy_force_switch")
      * self.optim_state.get("max_fun_evals")
  ):
  ```

  `entropy_force_switch` is an *option*, not a piece of optimization state:
  `_init_optim_state` (900–1112) never writes it, and no other module writes
  it either (a package-wide scan for the name finds this line alone). The
  read therefore returns `None` and the multiplication raises. Every sibling
  in the same block reads its option correctly through `self.options.get`;
  `optim_state["max_fun_evals"]` is a deliberate copy of the option
  (`:1001`), but there is no such copy for `entropy_force_switch`. The
  correct read is `self.options.get("entropy_force_switch")` (0.8 by
  default): "force switch to stochastic entropy at this fraction of total
  fcn evals".
- Consequence if real: the whole entropy-switch feature is unusable. With
  `options={"entropy_switch": True}` and `D >= det_entropy_min_d` (5),
  `optim_state["entropy_switch"]` is `True` (set at `:988-992`), and the
  first pass through the loop raises `TypeError: unsupported operand type(s)
  for *: 'NoneType' and 'int'` at iteration 0, before any work is done. The
  run does not produce a wrong answer; it does not run at all. It does not
  fire at default options (`entropy_switch = False`), and it cannot fire for
  `D < 5` because the flag is forced off there. It also means the intended
  behaviour — switching from the deterministic to the stochastic entropy
  once 80% of the budget is spent — has never run: the only other place that
  clears the flag is the stability branch at `:2169-2172`.
- Suggested reproduction: already run (`check1.py`). Building
  `VBMC(f, x0, ..., options={"entropy_switch": True})` at `D=5` gives
  `optim_state['entropy_switch'] = True`,
  `'entropy_force_switch' in optim_state: False`, and forming the product
  the loop forms raises `TypeError: unsupported operand type(s) for *:
  'NoneType' and 'int'`. A one-line confirmation of the whole path would be
  a two-iteration `optimize()` with that option at `D=5`.
- Test adequacy: no. No test sets `entropy_switch=True` on an object that
  then enters `optimize()`. `test_vbmc_loop_termination.py:166`
  (`test_vbmc_is_finished_stability_entropy_switch`) and `:55` set
  `optim_state["entropy_switch"] = True` but call
  `_check_termination_conditions` directly, which never reaches line 1226.
  `test_variational_optimization.py:589` sets the flag in a hand-built
  `optim_state` dict. Nothing exercises the main loop with the switch on.

### F2. The running average of the variational posterior's moments is never taken: a misplaced parenthesis makes the reset condition always true
- Location: `pyvbmc/vbmc/vbmc.py:1580-1582` (the condition), `:1586-1597`
  (the unreachable branch); option declared at
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:189`; MATLAB: "no
  counterpart read" (internal track)
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- The code is

  ```python
  if len(self.optim_state.get("run_mean")) == 0 or len(
      self.optim_state.get("run_cov") == 0
  ):
  ```

  The second test should be `len(self.optim_state.get("run_cov")) == 0`. As
  written, `run_cov == 0` is an elementwise comparison producing a `(D, D)`
  boolean array, whose `len` is `D >= 1` and therefore always truthy. The
  `or` makes the whole condition true at every iteration, so the reset
  branch always runs and the exponentially weighted update at 1586–1597 is
  dead code. What the block is supposed to do is stated by its own option:
  `moments_run_weight = 0.9`, "Weight of previous trials (per trial) for
  running avg of variational posterior moments", combined as
  `w = moments_run_weight ** (N − last_run_avg)`.
- Consequence if real: `optim_state["run_mean"]` and
  `optim_state["run_cov"]` hold the current iteration's moments rather than
  a running average, and `optim_state["last_run_avg"]` always equals
  `optim_state["N"]`, so the exponent `Nnew` would be 0 even if the branch
  were reached. The option `moments_run_weight` has no effect at any
  setting. No other module reads the three keys
  (`pyvbmc/whitening/whitening.py:244-246` only resets them), so no
  numerical result of a run changes today; what is wrong is the recorded
  state — the `optim_state` of every iteration in `vbmc.iteration_history`
  reports an instantaneous moment where the field is documented as a running
  average — and the latent trap that any future reader of
  `run_mean`/`run_cov` gets an unsmoothed quantity. It fires at the default
  options, every iteration, for every run.
- Suggested reproduction: already run. `check1.py` evaluates the condition
  on a populated state and prints `5` (truthy) for `D = 5`, and truthy for
  `D ∈ {1, 2, 10}`. `check2.py` reads
  `pyvbmc/testing/vbmc/test_vbmc_save_static.pkl` and shows
  `last_run_avg == N` at all seven recorded iterations (10/10, 15/15, …,
  40/40), which is only possible if the reset branch ran every time.
- Test adequacy: no. No test reads `optim_state["run_mean"]`, `run_cov` or
  `last_run_avg`, and `moments_run_weight` appears in no test. Note that
  `test_options.py:183`
  (`test_inert_options_are_the_declared_options_nothing_reads`) does not
  catch the dead option either, because the name does occur in a
  `self.options.get(...)` call — inside the unreachable branch.

### F3. `results["problem_type"]` is always `"unconstrained"`: it tests the transformed bounds, which are always infinite
- Location: `pyvbmc/vbmc/vbmc.py:3127-3132`; MATLAB: "no counterpart read"
  (internal track)
- Category: cross-module
- Proposed classification: suspected defect
- Confidence: high
- The test is `np.all(np.isinf(optim_state["lb_tran"])) and
  np.all(np.isinf(optim_state["ub_tran"]))`. `lb_tran` and `ub_tran` are the
  *hard bounds mapped into the inference space* (`:908-915`), and the whole
  point of `ParameterTransformer` is that a bounded coordinate is carried to
  the whole real line: the probit map sends `lb` to `norminv(0) = -inf` and
  `ub` to `norminv(1) = +inf`, and the logit map does the same. So both
  arrays are `±inf` for a bounded problem exactly as they are for an
  unbounded one, and the `"bounded"` branch at `:3132` is unreachable. The
  field is meant to report whether the *user's* problem is bound
  constrained, which is a question about the original-space bounds; those
  are available in the same dictionary as `optim_state["lb_orig"]` and
  `optim_state["ub_orig"]` (`:895-896`).
- Consequence if real: `results["problem_type"]` misreports every
  bound-constrained run as unconstrained. It is a reported field only — no
  algorithmic decision reads it — but it is part of the public results dict,
  and it fires at the default options for every problem with a finite bound,
  which is the common case in the example notebooks.
- Suggested reproduction: already run (`check7.py`). For `D = 2` with
  `lb = -5`, `ub = 5` under both `bounded_transform` settings the script
  prints `lb_tran=[-inf -inf] ub_tran=[inf inf] -> problem_type=
  'unconstrained'`; the half-bounded and unbounded cases print the same.
- Test adequacy: no. `test_vbmc_optimize.py:469` (`test_optimize_results`)
  asserts `results["problem_type"] == "unconstrained"` for a problem that
  *is* unconstrained — the one case in which the expression happens to be
  right. No test constructs a bounded problem and checks the field.

### F4. `min_iter` is compared against the 0-based iteration index, so a run always performs one more iteration than the minimum, and the minimum silently overrides `max_iter` and `max_fun_evals`
- Location: `pyvbmc/vbmc/vbmc.py:2194-2200`; contrast `:2116`; MATLAB: "no
  counterpart read" (internal track)
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high (for the off-by-one); medium (for whether the override of
  the maxima is intended)
- Within the same method, the maximum is counted correctly and the minimum
  is not. `iteration` is the 0-based index of the current iteration, so the
  number of iterations completed is `iteration + 1`. The maximum test is
  `iteration + 1 >= self.options.get("max_iter")` (`:2116`), which stops
  after exactly `max_iter` iterations. The minimum test is
  `iteration < self.options.get("min_iter")` (`:2196`), which blocks
  termination until `iteration >= min_iter`, that is until `min_iter + 1`
  iterations have run. Written in the same units as the maximum it should be
  `iteration + 1 < min_iter`. The option's own documentation is "Min number
  of iterations" (`advanced_vbmc_options.ini:154-155`), a count, and the
  companion `min_fun_evals` on the same line is correctly compared against a
  count (`func_count`).
  A second, separable consequence of the same lines: in the ordinary
  (non-budget) path the guard is `below_minimum and (not self._budget_active
  or ...)`, whose second operand is trivially true, so `below_minimum`
  unconditionally clears `is_finished_flag` — including when the run has
  already exceeded `max_iter` or `max_fun_evals`.
- Consequence if real: every run does `min_iter + 1 = D + 1` iterations at
  the defaults rather than `D`, spending `fun_evals_per_iter` extra target
  evaluations whenever the run would otherwise have terminated at exactly
  `min_iter` iterations. This only bites when something else wants to stop
  that early — a small `max_fun_evals`, a small `max_iter`, or a
  quickly-stable run at `D >= tol_stable_iters`. The override effect is
  larger and easier to hit: a user who sets `max_iter=1` at `D=3` gets 4
  iterations, and a user who sets a small `max_fun_evals` keeps sampling
  past it until `min_iter` is satisfied, which contradicts "Max number of
  iterations" / "Max number of target fcn evals" in
  `basic_vbmc_options.ini:7-10`.
- Suggested reproduction: already run (`check6.py`). With `D=3`
  (`min_iter = 3`, `min_fun_evals = 15`), `max_iter=1`, `max_fun_evals=20`,
  `func_count=1000` and a stubbed reliability index,
  `_check_termination_conditions` returns `is_finished=False` at iteration
  indices 0, 1 and 2 and `True` from index 3 — four iterations for
  `min_iter = 3`, with both maxima long exceeded.
- Test adequacy: no. `test_vbmc_loop_termination.py:78`
  (`test_vbmc_check_termination_conditions_prevent_early_termination`) pins
  the current behaviour by construction: it sets `min_iter=101`,
  `optim_state["iter"]=100` and asserts "not terminated", which is exactly
  the implementation's `100 < 101` and says nothing about how many
  iterations `min_iter=101` should buy. This is a test that mirrors the
  implementation rather than the specification.

### F5. `results["iterations"]` reports the index of the last iteration, one less than the number of iterations
- Location: `pyvbmc/vbmc/vbmc.py:3134`; MATLAB: "no counterpart read"
  (internal track)
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: medium
- `output["iterations"] = self.optim_state["iter"]` copies the 0-based
  iteration index. A run that performed `n` iterations (indices `0 … n−1`)
  reports `n − 1`. The neighbouring `output["func_count"]` is a literal
  count, and the loop control that the field mirrors treats `iteration + 1`
  as the count (`:2116`), so the dict is internally inconsistent: with
  `max_iter = 5` a run that stops on the iteration limit reports
  `iterations = 4`. `best_iter` (`:3133`) is correctly an index, and is
  named as one.
- Consequence if real: a reported field only, off by one, always. Anyone
  comparing `results["iterations"]` against `options["max_iter"]`, or
  summing iterations across runs, is off by one per run. It fires at the
  default options for every run.
- Suggested reproduction: read `vbmc.iteration_history["iter"]` and
  `results["iterations"]` after any run. On the stored
  `test_vbmc_save_static.pkl`, `len(iteration_history["iter"]) == 7` while
  `optim_state["iter"] == 6` (`check2.py`).
- Test adequacy: no. `test_vbmc_optimize.py:506` asserts only
  `"iterations" in results`.

### F6. `recompute_lcb_max` defaults to `True` but `_recompute_lcb_max` is an unimplemented stub, and the `optim_state` key it fills is read nowhere
- Location: `pyvbmc/vbmc/vbmc.py:2403-2408` (the stub), `:1641-1645` (the
  call), option at
  `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:308-309`; MATLAB:
  "no counterpart read" (internal track)
- Category: defaults
- Proposed classification: suspected defect (unported feature reachable at
  the default option)
- Confidence: high (the facts); medium (on how much the missing
  recomputation changes a run)
- `_recompute_lcb_max` carries `# ToDo: Recompute_lcb_max needs to be
  implemented.` and returns `np.array([])`. The loop stores the transpose of
  that empty array in `optim_state["lcb_max_vec"]` whenever the default-on
  option `recompute_lcb_max` is set, and nothing anywhere in the package
  reads `lcb_max_vec` (a package-wide grep finds the write at `:1644` and no
  read). The warmup end test instead uses the per-iteration `lcb_max` values
  recorded at the time (`:1975`), each computed from *that* iteration's GP
  (`:1567-1572`). The option's documented meaning is "Recompute LCB max for
  each iteration based True current GP estimate", and
  `pyvbmc/vbmc/README.md` maps `_recompute_lcb_max()` to MATLAB's
  `recompute_lcbmax.m`, so the feature is declared, mapped and enabled by
  default, and does nothing.
- Consequence if real: the second warmup end condition ("no substantial
  improvement of max fcn value in recent iters",
  `_check_warmup_end_conditions:1979-1993`) compares stale LCB maxima
  computed under successively different GP fits rather than one consistent
  set recomputed under the current GP. Early-iteration GPs are poorly fitted
  and their LCB maxima can be far from what the current GP implies, so the
  improvement measure is noisier than intended, in the direction of both
  ending and prolonging warmup wrongly. It fires at the default options for
  every run. Note also that this option escapes the `INERT_OPTIONS` registry
  (see the P1b sheet entry) because it *is* read — the read just gates a
  stub.
- Suggested reproduction: already run (`check2.py`): every recorded
  `optim_state` from iteration 1 onward in
  `pyvbmc/testing/vbmc/test_vbmc_save_static.pkl` carries
  `lcb_max_vec = []`. `grep -rn lcb_max_vec pyvbmc` shows the write and no
  read.
- Test adequacy: no. No test calls `_recompute_lcb_max` or inspects
  `lcb_max_vec`. The four `test_check_warmup_end_conditions_*` tests in
  `test_vbmc_loop_termination.py:396-468` install an `lcb_max` history by
  hand, which is the path that would still be taken if the recomputation
  existed, so they could not notice.

### F7. Continuing a run with a larger `max_fun_evals` leaves `optim_state["max_fun_evals"]` stale, collapsing the GP-training initial design to its floor
- Location: `pyvbmc/vbmc/vbmc.py:1161-1181` (the continuation block, which
  refreshes the key only under `if self._budget_active:` at `:1171`);
  written at `:1001`; read at
  `pyvbmc/vbmc/gaussian_process_train.py:624-626`; MATLAB: "no counterpart
  read" (internal track)
- Category: state/caching
- Proposed classification: suspected defect
- Confidence: high (the staleness); medium (that the intended semantics is
  "track the option")
- `optim_state["max_fun_evals"]` is set once at construction as a copy of
  the option, with the comment "Copy maximum number of fresh function
  evaluations, used by some schedules and acquisition functions". On a
  continuation, `self.optim_state` is restored from the recorded state
  (`:1168-1170`) and the copy is refreshed only when the opted-in budget
  path is active (`:1171-1181`); `VBMC.load` does the same (`:2992`). The
  documented way to continue a run is
  `VBMC.load("fit.pkl", new_options={"max_fun_evals": 1000})`
  (`docsrc/source/faq.md`, "How do I save and continue a run?"), which is
  the ordinary, non-budget path. After it, `_check_termination_conditions`
  reads the *live* option (`:2102-2106`) and lets the run go to 1000
  evaluations, while `_get_gp_training_options` reads the *stale* copy and
  believes the horizon is the old budget. The two readers of the same
  quantity disagree.
- Consequence if real: the GP hyperparameter fit's multi-start design size
  is a cubic schedule in `x = (n_eff − fun_eval_start) / (min(max_fun_evals,
  1000) − fun_eval_start)`, floored at 9. Past `x ≈ 1.3` the cubic turns
  negative and the floor takes over, so for the whole continued portion of
  the run the GP fit restarts from 9 points instead of the scheduled
  number. Measured (`check3.py`) for a run whose first budget was 300 and
  which is continued to 1000: at `n_eff = 500` the stale schedule gives
  `init_N = 9` where the refreshed one gives 188; at `n_eff = 700`, 9
  against 91. A worse hyperparameter fit degrades every downstream stage.
  It fires whenever a run is continued with a raised `max_fun_evals`, which
  is the documented resume workflow, and needs no non-default option.
- Suggested reproduction: already run (`check4.py`):
  `VBMC.load(static_pkl, new_options={"max_fun_evals": 999})` gives
  `optim_state['max_fun_evals'] = 40` against `options['max_fun_evals'] =
  999`. `check3.py` evaluates the resulting schedule.
- Test adequacy: no. `test_vbmc_optimize.py:553`
  (`test_vbmc_resume_optimization`) is the one resume test and it is a good
  one — it asserts that a split run reproduces a continuous run — but it
  raises `max_iter`, not `max_fun_evals`, and it round-trips with
  `dill.dump`/`dill.load` rather than `VBMC.load`, so it never changes the
  option that the stale copy shadows.

### F8. The warping branch hands `warp_input` the best recorded posterior, whose parameter transformer need not be the current one
- Location: `pyvbmc/vbmc/vbmc.py:1261-1287` (the `determine_best_vp` call
  and the `warp_input` call); consumed at
  `pyvbmc/whitening/whitening.py:215-234`; contrast
  `pyvbmc/vbmc/vbmc.py:1288-1290`, which correctly passes `self.vp` to
  `warp_gp_and_vp`; MATLAB: "no counterpart read" (internal track)
- Category: cross-module
- Proposed classification: unsure (possibly a defect in both)
- Confidence: medium
- `vp_tmp` comes from `determine_best_vp()` over the whole recorded history,
  and each recorded posterior carries its own `ParameterTransformer` copy
  (verified: 7 distinct objects in the stored run, `check8.py`). After a
  warp, posteriors recorded *before* that warp keep the pre-warp transform.
  `warp_input(vp_tmp, ...)` uses `vp_tmp.parameter_transformer` for two
  different jobs. Un-rotating `vp_tmp`'s own covariance
  (`whitening.py:136-148`) is correct, because that covariance lives in
  `vp_tmp`'s space. But the same transformer is then used as the inverse leg
  of `warpfun` (`whitening.py:215-216`) and applied to
  `optim_state["lb_search"]`, `optim_state["ub_search"]` (`:219-229`) and
  `optim_state["search_cache"]` (`:232-234`) — quantities that live in the
  *current* inference space, the one `self.vp.parameter_transformer`
  defines. When the two transforms differ, these points are inverted with
  the wrong map before being pushed into the new space. The GP training
  inputs are safe because they are recomputed from `X_orig`
  (`whitening.py:203-206`), and `warp_gp_and_vp` is correct because the call
  site passes it `self.vp` (`:1289`); the asymmetry between the two call
  sites is what exposes the mismatch.
- Consequence if real: after a second or later warp whose
  `determine_best_vp` picks an iteration from before the previous warp, the
  active-sampling search box and the cached search points are mapped through
  a stale rotation/scaling. The search box is then wrong by that rotation,
  which can exclude the region the posterior actually occupies or inflate
  the box, and the cached candidates are wrong points. It cannot fire before
  the second warp; it needs `warp_rotoscaling` (default `True`),
  `vp.K >= warp_min_k` (5), `r_index < 3` and at least
  `warp_every_iters · warping_count` iterations between warps, so it is a
  long-run, default-options event rather than a common one.
- Suggested reproduction: a seeded run long enough for two warps, with a
  probe that records `id(vp_tmp.parameter_transformer)` against
  `id(self.vp.parameter_transformer)` at `:1266` — or, more cheaply, drive
  `warp_input` twice on a hand-built state in which the posterior handed in
  carries a transformer with `R_mat = I` while `optim_state["lb_search"]`
  was produced under a non-identity `R_mat`, and compare the resulting
  search box with the one obtained by passing the current posterior. I did
  not run this (it needs either a full run or a fabricated multi-warp
  state); the object-identity half is settled by `check8.py`.
- Test adequacy: no. No test drives two warps; `pyvbmc/testing/whitening/`
  exercises `warp_input` with a single posterior whose transformer is the
  current one by construction.

### F9. `temperature` is a declared option that nothing reads, contrary to the P7 sheet entry — which is also what makes the two omitted `vptrain2real` calls harmless
- Location: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:256-257`;
  the only occurrences of the name in the package are
  `pyvbmc/whitening/whitening.py:196-197` and `:291-292`, which read
  `optim_state`, not `options`; the commented-out calls are at
  `pyvbmc/vbmc/vbmc.py:1349` and `:1541`; MATLAB: "no counterpart read"
  (internal track)
- Category: defaults
- Proposed classification: suspected defect (a dead option presented as
  live), and a contradiction of a sheet entry
- Confidence: high
- The P7 sheet entry "Posterior tempering (`vbmc_power`, `vptrain2real`) is
  not ported" states: "The option `temperature = 1` exists
  (`advanced_vbmc_options.ini:256`) and is read only by
  `pyvbmc/whitening/whitening.py:196`, `:291`." Those two lines read
  `optim_state.get("temperature")` and `vbmc.optim_state.get("temperature")`.
  Nothing ever writes `optim_state["temperature"]`: `_init_optim_state`
  (900–1112) does not, and a package-wide scan finds no other writer. So
  both reads always return `None`, both branches always take `T = 1`, and
  the option `temperature` is read nowhere at all. Setting
  `options={"temperature": 2}` changes nothing and warns about nothing — it
  is not in `INERT_OPTIONS` (`pyvbmc/vbmc/options.py:21-46`), so
  `_warn_inert_options` stays silent. The registry test does not catch it
  because `test_options.py:200-205` looks for the quoted name anywhere in
  the package text, and `"temperature"` does occur — as an `optim_state`
  key. The same false positive shields `diagnostics`
  (`advanced_vbmc_options.ini:26-27`), whose only occurrence in the package
  is the unrelated local key `sampler_opts["diagnostics"]` at
  `pyvbmc/vbmc/active_importance_sampling.py:437`.
- Consequence if real: two user-settable options are silently inert and
  unflagged, one of them (`temperature`) advertised in its own comment as
  "Temperature for posterior tempering (allowed values T = 1234)". The
  positive side of the same fact answers the question I was asked to judge
  about the two commented-out `vptrain2real` calls: since no reachable
  option setting can make the temperature anything but 1, and since
  `vp_real` is used only for `stats["elbo"]`, `stats["elbo_sd"]`
  (`:1350-1352`, `:1542-1544`) and `_compute_true_diagnostic`, which all
  read the posterior's own statistics, **leaving the calls out is harmless
  at every reachable setting** — more strongly so than the sheet's "at the
  default `temperature = 1`" phrasing suggests, because a non-default
  temperature cannot be reached at all.
- Suggested reproduction: already run (`check9.py`), which scans the two
  `.ini` files for names that never appear as `options[...]`,
  `options.get(...)` or `options.eval(...)` outside `options.py` and the
  test suite, and reports `entropy_force_switch` and `temperature` (plus
  `diagnostics`, found by the follow-up grep).
- Test adequacy: no — and worse, `test_options.py:183`
  (`test_inert_options_are_the_declared_options_nothing_reads`) claims in
  its docstring to be exactly this check while its text scan cannot
  distinguish an option read from a same-named dictionary key.

### F10. `_check_warmup_end_conditions` raises on an empty slice when the stable-warmup window is one iteration
- Location: `pyvbmc/vbmc/vbmc.py:1960-1972`; MATLAB: "no counterpart read"
  (internal track)
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: high
- `max_now = np.amax(elcbo_vec[max(3, len(elcbo_vec) − tol_stable_warmup_iters):])`
  assumes the hard-coded lower bound 3 (the "ignore first two iterations"
  rule) always leaves something in the slice. The guard above it is
  `iteration > tol_stable_warmup_iters`, which only guarantees
  `len(elcbo_vec) >= tol_stable_warmup_iters + 2`. When
  `tol_stable_warmup_iters == 1` — that is, whenever
  `tol_stable_warmup <= fun_evals_per_iter` — the first call has
  `len(elcbo_vec) == 3` and the slice `[3:]` is empty, and `np.amax` of an
  empty array raises. At the shipped defaults (`tol_stable_warmup = 15`,
  `fun_evals_per_iter = 5`) the window is 3 and the slice is never empty.
- Consequence if real: a run configured with a short stable-warmup window
  (for example `tol_stable_warmup=5` at the default `fun_evals_per_iter=5`,
  or the default `tol_stable_warmup=15` with `fun_evals_per_iter=15`) dies
  with `ValueError: zero-size array to reduction operation maximum which has
  no identity` at iteration 2, in the middle of the loop, losing the run. It
  does not fire at default options.
- Suggested reproduction: already run (`check10.py`): a D=2 instance with
  `tol_stable_warmup=5`, `fun_evals_per_iter=5`, `optim_state["iter"]=2` and
  a three-entry history raises exactly that `ValueError`.
- Test adequacy: no. The four `test_check_warmup_end_conditions_*` tests
  (`test_vbmc_loop_termination.py:396-468`) all use the default
  `tol_stable_warmup` with histories long enough to avoid the edge.

### Minor observations

- **The recorded posteriors never carry the stability flag.**
  `self.vp.stats["stable"]` is assigned at `pyvbmc/vbmc/vbmc.py:1636`, after
  the deep copy of the posterior was recorded at `:1622-1625`. Every entry
  of `vbmc.iteration_history["vp"]` therefore keeps the `False` that
  `optimize_vp` sets (`variational_optimization.py:403`), whatever
  `iteration_history["stable"]` says. `determine_best_vp:2744` repairs the
  one posterior it returns, so no result is wrong; only a user inspecting
  the history sees the inconsistency. Confirmed on the stored run
  (`check2.py`).
- **`self.vp.mu = self.gp.X.T` in the warping branch (`:1319`) lacks the
  `.copy()` its main-loop twin has (`:1500`)**, so with
  `variable_means=False` the posterior's means are a transposed view of the
  GP's training inputs. The two lines should not differ.
- **`final_boost` leaves two permanent marks on the live state**:
  `self.optim_state["warmup"] = False` (`:2509`) and
  `self.optim_state["entropy_alpha"] = 0` (`:2517`) are not restored,
  including when the boosted posterior is rejected. Harmless at the end of
  `optimize` (a continuation re-reads `optim_state` from the history), but
  `final_boost` is a documented public method and a mid-run call would
  silently end warmup.
- **`final_boost`'s docstring types `elbo` and `elbo_sd` as
  `VariationalPosterior`** (`:2427-2431`); they are floats.
- **With `do_final_boost=False`, `self.vp` is the iteration-history entry
  itself**, not a copy (`determine_best_vp:2741` returns
  `self.iteration_history.get("vp")[idx_best]`, and `:1826` assigns it
  directly). The returned value is deep-copied (`:1942`), but `vbmc.vp` and
  the history entry alias afterwards.
- **Write-only `optim_state` entries** left by the loop and its setup:
  `redo_roto_scaling` (`:1214`), `pruned` (`:985`, never updated from the
  loop's local `pruned`), `vp_repo` (`:1663`), `lcb_max_vec` (`:1644`),
  `iter_list` (`:1028-1032`). The declared history key `data_trim_list`
  (`:473`) is never recorded. None of these changes a result; they are
  leftovers that a reader can mistake for live state.
- **Two options that no `.ini` declares are read in the loop**:
  `self.options.get("warp_nonlinear")` (`:1243`) and
  `self.options.get("varactivesample")` (`:1409`, `:1438`). Both return
  `None`, so the warping gate reduces to `warp_rotoscaling` and the
  `sys.exit("Function currently not supported")` at `:1443` is unreachable.
  Consistent with the sheet's "Nonlinear input warping is not ported" and
  "Variational active sampling is not ported" entries, but a misspelling of
  either name would be just as silent.
- **`self.gp.predict(self.gp.X, self.gp.y, self.gp.s2, add_noise=False)`**
  (`:1568-1570`) passes `y_star` and `s2_star`, which
  `gpyreg/gaussian_process.py:2034-2036` uses only when `add_noise` or
  `return_lpd` is set. The call is correct but reads as if the observed
  values entered the prediction.

## 3. Test adequacy notes

**Tests that mirror the implementation rather than the specification.**

- `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:78`,
  `test_vbmc_check_termination_conditions_prevent_early_termination`. It
  sets `min_iter=101`, `optim_state["iter"]=100` and asserts "not
  terminated" — a restatement of `100 < 101`. A specification-level test
  ("a run with `min_iter=k` performs at least `k` iterations", counted from
  `len(iteration_history["iter"])`) would have caught F4. The companion
  `..._max_iter` test at `:55` is in the same shape but happens to pin the
  correct convention, which is precisely why the two conventions could
  diverge unnoticed.
- `pyvbmc/testing/vbmc/test_options.py:183`,
  `test_inert_options_are_the_declared_options_nothing_reads`. Its docstring
  states the specification ("exactly the declared options that no module of
  the package reads"), but the implementation is a regex for the quoted name
  anywhere in the package source, which cannot tell an option read from a
  same-named `optim_state` key or local dictionary key. Three declared
  options pass it while being dead or misread (`temperature`, `diagnostics`,
  `entropy_force_switch`; F1 and F9), and two more pass it while being read
  only from unreachable code (`moments_run_weight`, F2; `recompute_lcb_max`,
  F6). Matching `options[...]` / `options.get(...)` / `options.eval(...)`
  instead would close most of the gap.
- `pyvbmc/testing/vbmc/test_vbmc_optimize.py:469`, `test_optimize_results`.
  It asserts membership for most fields (`"iterations" in results`) and
  checks `problem_type` only on the one input for which the expression is
  accidentally right. F3 and F5 both sit in fields this test touches.
- `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:396-468`, the four
  `test_check_warmup_end_conditions_*` tests. They install `elbo`,
  `elbo_sd`, `lcb_max` and `func_count` histories by hand at the default
  options and assert the boolean outcome. They pin the current slicing but
  exercise neither the window edge (F10) nor the role the recomputed LCB
  maxima were meant to play (F6).

**Good models.**

- `pyvbmc/testing/vbmc/test_vbmc_finalboost.py:270`,
  `test_final_boost_endpoint_rule_matches_dense_continuum`. It states the
  rule the code is a reduction of ("`dE − b·dS > −tol` for every `b` in
  `[0,5]`") and checks the implementation against a dense grid of `b`,
  rather than re-deriving `min(dE, dE − 5·dS)`. That is the shape a test of
  a closed-form reduction should have.
- `pyvbmc/testing/vbmc/test_vbmc_optimize.py:553`,
  `test_vbmc_resume_optimization`. It asserts an invariant a user can state
  (a split run equals a continuous one) and then checks a structural
  property with a reason attached (every history entry describes its own
  iteration; the GP records are lean and `get_gp` restores them). Its only
  gap is the option it varies (F7).
- `pyvbmc/testing/vbmc/test_vbmc_determine_best_vp.py:40`,
  `test_determine_best_vp_receives_the_option_values`, and `:210`/`:234`,
  which check that the look-back window and the rank penalty count
  iterations. These test what the options are supposed to mean rather than
  the expressions that implement them.
- `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:291`,
  `test_gp_sampling_history_compatibility_and_stable_weights`, and `:329`,
  `test_gp_sampling_history_missing_data_does_not_stop`. They state the
  weighting rule and the "never stop on missing data" contract directly.

## 4. Sheet notes

- **P7, "Posterior tempering (`vbmc_power`, `vptrain2real`) is not ported".**
  The entry says the option `temperature` "is read only by
  `pyvbmc/whitening/whitening.py:196`, `:291`". Those two lines read
  `optim_state["temperature"]`, a key nothing ever writes, so the *option*
  is read nowhere and its value cannot reach the whitening code. The entry's
  conclusion is unaffected and in fact strengthened — the commented-out
  `vptrain2real` calls are harmless at every reachable setting, not only at
  the default — but the stated mechanism ("a temperature other than 1 would
  be silently wrong outside the whitening code") does not hold as written: a
  temperature other than 1 cannot be set at all. See F9.
- **P1b, "Declared options that nothing reads".** The entry says "A test
  recomputes the set by scanning the package for reads, so a newly dead
  option, or a newly read registered one, fails it." The scan
  (`test_options.py:200-205`) matches the quoted option name anywhere in the
  package text, not a read of the options object, so a dead option whose
  name coincides with an `optim_state` key or a local dictionary key does
  not fail it. `temperature` and `diagnostics` are both in that position
  today. See F9.
- **P1a, "Final boost is guarded; MATLAB accepts its result
  unconditionally".** Accurate as far as the guard goes. Worth adding, if
  the sheet is meant to describe the state the boost leaves behind: the
  boost-only changes are confined to a deep copy of the options, but
  `self.optim_state["warmup"]` and `self.optim_state["entropy_alpha"]` are
  set on the live dictionary and are not restored on either the accepted or
  the rejected path (`pyvbmc/vbmc/vbmc.py:2509`, `:2517`).
