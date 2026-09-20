# Wave 2 verification, group B: loop, termination, results

Raw report of a verifier, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent,
read-only on the repositories, verified the wave-2 findings on the loop,
the termination checks and the result fields (reports
`../reviews/P1a_internal.md` and `../reviews/P1a_comparison.md`), without
`optimize()` runs. Its scripts are kept as `scripts/wave2_B_*.py`. The
text below the rule is the agent's final message, unedited; `wave2.md`
holds the consolidated ledger and the PI's dispositions.

One correction by the orchestrator, from the run the agent was not allowed
to make (`scripts/wave2_two_option_runs.py`). Row B-5 places the failure
of `separate_search_gp=True` in the training call that follows the search
GP's, at the concatenation of `gaussian_process_train.py:135-149`. The run
fails one step earlier, inside the search GP's own training call
(`vbmc.py:1419`), at `gaussian_process_train.py:201`: the running average
of the hyperparameter covariance combines the constant-mean fit's 5 by 5
covariance with the 9 by 9 `run_cov` that the shared `hyp_dict` carries
from the main model (`ValueError: operands could not be broadcast together
with shapes (5,5) (9,9)`, iteration 1, `D = 2`). The cause is the one the
row gives, the shared hyperparameter struct. The same script confirms row
B-1 inside the loop: `entropy_switch=True` at `D = 5` raises the
`TypeError` at `vbmc.py:1226` before any iteration is recorded.

---

# Wave-2 verification, group B (P1a loop / termination / results)

Verified 2026-09-20 on `dev-port-review` at `d6c3827` against MATLAB VBMC at `396d649`.
Scripts (each named for its finding, with a docstring saying what it settles) are in
`C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\b8a4c56e-a31c-47b4-87d0-66a2c9861ded\scratchpad\port_review\verify_wave2_B\`:
`f1_entropy_force_switch.py`, `f2_run_moments_guard.py`, `f3_result_fields.py`,
`f4_warmup_empty_slice.py`, `f5_separate_search_gp.py`, `f6_sieve_warp_branch.py`.
No `optimize()` run, no test suite, nothing written in `pyvbmc`, `../gpyreg` or `../vbmc`.
`pyvbmc.__file__` printed by every script:
`C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`.

## Ledger

| id | reports | statement | verdict (class) | fires at defaults? | how verified | dating and rationale |
|---|---|---|---|---|---|---|
| B-1 | P1a int F1, P1a cmp F5, P1b int F2, P1b cmp F2 | `optimize` reads `optim_state["entropy_force_switch"]` (`vbmc.py:1226`), which `_init_optim_state` and every other module never write, so with `entropy_switch=True` and `D >= det_entropy_min_d` (5) the first iteration raises `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`; `vbmc.m:524-525` reads `options.EntropyForceSwitch` (0.8) | **confirmed port discrepancy**; the whole forced-switch feature has never run | no (`entropy_switch = False` on both sides); fires on the first iteration of any run with `entropy_switch=True` and `D >= 5` | `f1_entropy_force_switch.py`: at D=5 `optim_state['entropy_switch'] = True`, key absent, `.get` → `None`, the loop's own expression raises; at D=2 the flag is forced off and the expression short-circuits. `options.get("entropy_force_switch") = 0.8` | Python: `ec94eba` 2021-06-08 Marlon "feat: add skeleton of vbmc.optimize()" wrote the read against `optim_state`; never matched MATLAB, whose option read dates from `224e333` 2018-07-09. Rationale: none found (grep of `pyvbmc/**/*.md`, `dev/**`, the `.ini` comment, `pyvbmc/vbmc/README.md`, the sheet; `gh search issues/prs acerbilab/pyvbmc "entropy_force_switch"`, `"entropy switch"` → no hits) |
| B-2 | P1a int F2, P1a cmp F7, P1b int F5 | `if len(run_mean) == 0 or len(run_cov == 0):` (`vbmc.py:1580-1582`): the `== 0` is inside `len(...)`, so the second operand is `D` and the guard is true every iteration; the exponentially weighted update at `:1586-1597`, the only reader of `moments_run_weight`, is dead. MATLAB `vbmc.m:781` is `isempty(RunMean) || isempty(RunCov)` | **confirmed port discrepancy** (a Python typo; the recorded `run_mean`/`run_cov` are instantaneous moments where MATLAB's are smoothed). No algorithmic quantity depends on them on **either** side — see the note below on where `moments_run_weight` is inert | yes, every run, from the second iteration | `f2_run_moments_guard.py`: `len(np.eye(D) == 0) == D` for D in 1,2,3,5,10; on `test_vbmc_save_static.pkl` `last_run_avg == N` at all 7 recorded iterations (10/10 … 40/40) and `run_mean[0]` swings (+0.0200 → −0.2535 between iterations 3 and 4) | Python: `ec94eba` 2021-06-08 Marlon, as written today; only reformatted since (`343b9c7` 2022-10-04). MATLAB block last changed `c6d723f` 2018-10-03. Rationale: none found (no comment, no `dev/` note, no test, no sheet entry) |
| B-3a | P1a int F3, P1a cmp F8 | `results["problem_type"]` tests `optim_state["lb_tran"]/["ub_tran"]`, the *transformed* hard bounds, which the probit/logit map sends to ±inf for a bounded variable exactly as for an unbounded one, so the `"bounded"` branch (`vbmc.py:3127-3132`) is unreachable | **confirmed shared defect**: `private/vbmc_output.m:5-9` tests `optimState.LB/UB`, which `misc/setupvars_vbmc.m:49-50` fills with `warpvars_vbmc(LB,'dir',trinfo)` — the transformed bounds, never reassigned anywhere else in the MATLAB tree. Both toolboxes always report "unconstrained" | yes, for every bound-constrained run (a reported field only; nothing reads it on either side) | `f3_result_fields.py`: bounded, mixed and unbounded D=2 problems under both `bounded_transform` settings all give `lb_tran=[-inf -inf] ub_tran=[inf inf] → 'unconstrained'`, while `lb_orig`/`ub_orig` in the same dict carry the user's bounds | Python: `c477a7b` 2021-11-01 Marlon "feat: log iterations and return result dict (#31)" created the field, testing the transformed bounds from the start (then named `optim_state["lb"]`; `lb_tran` since `9706c53` 2022-09-19). It has always matched MATLAB. Rationale: none found. Docs: no passage defines the field; the shipped notebook 3 output prints `'problem_type': 'unconstrained'` for an unbounded problem |
| B-3b | P1a int F5, P1a cmp F9 | `results["iterations"] = optim_state["iter"]` (`vbmc.py:3134`) is the 0-based index of the last iteration; `private/vbmc_output.m:10` copies MATLAB's 1-based `optimState.iter`, which is the count | **confirmed port discrepancy**, one less than the number of iterations, always | yes, every run (reported field only) | `f3_result_fields.py`: on the stored run `len(iteration_history['iter']) == 7`, values `[0..6]`, `optim_state['iter'] == 6` | Same commit `c477a7b` 2021-11-01; never matched MATLAB. Rationale: none found. Docs: `optimize`'s docstring says only "A dictionary with additional information about the VBMC run"; no `.rst`/`.md` defines the field; notebook 3 prints `'iterations': 6` beside `'best_iter': 5` |
| B-4 | P1a int F10 | `_check_warmup_end_conditions` computes `max_now = np.amax(elcbo_vec[max(3, len-T):])` with `T = ceil(tol_stable_warmup/fun_evals_per_iter)`; when `T == 1` the first call that passes the guard has `len == 3` and the slice is empty, so `np.amax` raises `ValueError: zero-size array to reduction operation maximum which has no identity` | **confirmed shared defect**. `private/vbmc_warmup.m:39` is `max(elcbo_vec(max(4,end-T+1):end))`, which is empty in exactly the same state (1-based `iter = 3`, history 3). MATLAB's `max([])` is `[]`, `[] - scalar` is `[]`, `[] < thresh` is a 0×0 logical, and `:87` then evaluates `StableCountFlag && …`, whose left operand must be convertible to a logical scalar — MATLAB raises there instead. *Inferred from MATLAB's documented `&&` semantics; not executed (no MATLAB here).* PyVBMC's index arithmetic is otherwise a faithful translation | no (defaults 15/5 give `T = 3`); fires when `tol_stable_warmup <= fun_evals_per_iter`, at 0-based iteration 2 only | `f4_warmup_empty_slice.py`: (5,5) and (15,15) at iteration 2 raise the `ValueError`; (5,5) at iteration 3 returns False; (15,5) at iterations 4 and 5 return False. The script also tabulates both toolboxes' index sets for history 3/4/5 at `T = 1`, showing both `max_now` sets empty at history 3 | Python: present form since `e38351a` 2022-05-23 Bobby-Huggins "Noisy likelihoods (#79)" (which replaced the negative-index slices and dropped `initial=0` from `max_before`); the empty-`max_now` exposure is older (`35be58b` 2021-07-07, `d407462` 2021-08-27). MATLAB lines unchanged since `d6f1188` 2019-09-20. Rationale: none found beyond the comment "NB: Take care with MATLAB 'end' indexing and off-by-one errors" |
| B-5 | P1a cmp F6 | With `separate_search_gp=True` the constant-mean search GP is trained through the **main** `self.hyp_dict` and reassigns it (`vbmc.py:1419`), and copies the constant-mean noise estimate into `optim_state["sn2_hpd"]` (`:1430`). MATLAB keeps a second struct (`vbmc.m:471 hypstruct_search`, used only at `:643`) and captures two outputs, discarding the `optimState` that `misc/gptrain_vbmc.m:103` wrote | **confirmed port discrepancy**; the option is unusable in PyVBMC | no (`separate_search_gp = False` on both sides); with it on, the failure is in iteration 1 (the first odd iteration) | `f5_separate_search_gp.py`: const mean gives 5 hyperparameters at D=2 (D+3) and negquad 9 (3D+3); 6 vs 12 at D=3. `gaussian_process_train.py:135` builds `hyp0 = np.empty((0, 5))` from the overwritten `hyp_dict["hyp"]` and `:143-149` concatenates the recorded negquad GPs → `ValueError: … along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 9`; the model-change guard at `:159-160` sits after it. Both sides' call sites read; the odd/even iteration mapping (`iteration % 2 == 1` against `mod(iter,2)==0`) is a correct port | Python: `aa45473` 2021-07-26 Mikko Aarnos "feat: added GP training for VBMC (#19)" gave the lines their present meaning (`ec94eba` 2021-06-08 already shared the single hyperparameter struct). MATLAB's `hypstruct_search` since `d9822e3` 2019-09-17; never matched. Rationale: none found; `grep separate_search_gp pyvbmc/testing/` → nothing |
| B-6 | P1a cmp F10 | The warp-undo refit sizes its sieve at `self.vp.K` (`vbmc.py:1326-1328`) where `vbmc.m:584` uses `Knew`, the value both computed two lines above; `Knew` is passed to `optimize_vp` but not to `ns_elbo` | **confirmed port discrepancy**, and the sheet's P6 entry ("PyVBMC's warp branch and final boost agree with MATLAB's") is wrong for the warp branch. `final_boost` does agree (`vbmc.py:2493` evaluates `ns_elbo` at `K_new`, as `misc/finalboost_vbmc.m:33`) | no (`variable_means = True` → `vp.optimize_mu` True after warm-up → `Knew == vp.K`); differs only with `variable_means=False` | `f6_sieve_warp_branch.py`: `variable_means` default True, `vp.optimize_mu` True; with it False and `ns_elbo = 50*K`, PyVBMC asks for 100 candidates at `vp.K = 2` where MATLAB asks for `50 × n_train` (1500 at 30 training points, 3000 at 60) | Python: `2df0d7e` 2022-09-20 Bobby-Huggins "`vbmc.K`/`vp.K` fixes (#101)" replaced `self.K` with `self.vp.K` at this site and at the main-loop site; before that it was `self.K` (= `k_warmup`), which also differed from `Knew`. MATLAB's `Knew` here since `d2989b1` 2021-01-12, a week before the port began. **Partial recorded rationale** (issue #98, see below) — it argues for `Knew`, not for `vp.K` |

### Minor observations (P1a internal and P1a comparison), one row each

| id | reports | statement | verdict (class) | fires at defaults? | how verified | dating and rationale |
|---|---|---|---|---|---|---|
| B-M1 | P1a int mo-1 | The recorded posteriors never carry the stability flag: `self.vp.stats["stable"]` is set at `vbmc.py:1636`, after the deep copy recorded at `:1622` | **not a defect** (faithful port): MATLAB's `savestats` stores `stats.vp(iter) = vp` at `vbmc.m:803`/`:1043`, by value, and `:811` sets `vp.stats.stable` afterwards — the same ordering | yes on both sides; harmless (`determine_best_vp` sets the flag on the posterior it returns) | both sources read | Python: `9267816` 2021-08-25. The reviewer is right about the facts and right that no result is wrong; it is not a difference from MATLAB |
| B-M2 | P1a int mo-2, P1a cmp mo-5 | `self.vp.mu = self.gp.X.T` in the warp branch (`:1319`) lacks the `.copy()` its main-loop twin has (`:1500`), so `vp.mu` is a transposed view of `gp.X` | **Python-only defect**, latent (MATLAB's `vp.mu = gp.X'` copies by value); no in-place write to `vp.mu` exists today (`set_parameters` rebinds at `variational_posterior.py:1093`) | no — only with `variable_means=False` | both sources read; searched every write to `self.mu` | `359dd90` 2022-08-19 Bobby-Huggins "Benchmark bugfixes (#86)", bullet "fix: Small reference vs. copy issues in vmbc.py", added `.copy()` to the main-loop site and left the twin. That is the recorded rationale for the fix, and none for the omission. **A real defect, though a dormant one** |
| B-M3 | P1a int mo-3, P1a cmp mo-4 | `final_boost` leaves `optim_state["warmup"] = False` (`:2509`) and `optim_state["entropy_alpha"] = 0` (`:2517`) on the live instance, including when the boost is rejected | **confirmed port discrepancy**: `misc/finalboost_vbmc.m:40`, `:48` write the same two fields on a by-value `optimState` that dies with the call | yes in the sense that the writes always happen (`do_final_boost = True`); harmless at the end of `optimize()`, wrong for a mid-run call of the public method | both sources read | Python: `3bfb485` 2021-07-06 Marlon "feat: perform a final boost of variational components (#14)". Rationale: none found. **A real defect** (a documented public method silently ends warm-up) |
| B-M4 | P1a int mo-4 | `final_boost`'s docstring types `elbo` and `elbo_sd` as `VariationalPosterior` (`:2426-2431`); they are floats | **Python-only defect** (documentation) | n/a | read | Same commit lineage as B-M3. A leftover, not a behaviour defect |
| B-M5 | P1a int mo-5, P1a cmp mo-3 | With `do_final_boost=False`, `self.vp` *is* the iteration-history entry: `determine_best_vp` returns `self.iteration_history.get("vp")[idx_best]` (`:2741`) and writes `vp.stats["stable"]` into it (`:2744`); `:1826` assigns it to `self.vp` | **confirmed port discrepancy**, latent: MATLAB's `misc/best_vbmc.m:79` writes into a by-value copy, so the history is untouched. Two consequences — the alias (only when the boost is off) and the repair of one history entry (always) | no: `do_final_boost = True` by default (a PyVBMC-only option, already on the sheet) and `final_boost` deep-copies at `:2446` | both sources read | Python: `0c3b0ef` 2021-07-06 Marlon "feat: determine best VP in VBMC (#13)". Rationale: none found. **A real defect, dormant at the defaults** |
| B-M6 | P1a int mo-6 | Write-only `optim_state` entries `redo_roto_scaling` (`:1214`), `pruned` (`:985`), `vp_repo` (`:1663`), `lcb_max_vec` (`:1644`), `iter_list` (`:1028-1032`), and the declared history key `data_trim_list` (`:473`) that is never recorded | **mostly not a defect** (faithful ports), with one correction to the report: `redoRotoscaling` (`vbmc.m:519`), `optimState.pruned` (`setupvars_vbmc.m:207`) and `iterList` (`setupvars_vbmc.m:242-245`) are write-only in MATLAB too. `lcb_max_vec` is the already-ruled W2-2. **`vp_repo` is different**: MATLAB reads it at `misc/vpsieve_vbmc.m:60-66` under `options.VariationalInitRepo` (default `'no'`), so PyVBMC's reset-only write is the residue of an unported feature already registered in `INERT_OPTIONS`. The unrecorded history key `data_trim_list` is a Python-only leftover (MATLAB's `savestats` has no such field) | n/a | greps over both trees | Leftovers, not defects; worth one sheet line at most |
| B-M7 | P1a int mo-7 | `self.options.get("warp_nonlinear")` (`:1243`) and `self.options.get("varactivesample")` (`:1409`, `:1438`) name options that no `.ini` declares, so both return `None` | **confirmed** (reading + grep of both `.ini` files): the warping gate reduces to `warp_rotoscaling` and the `sys.exit("Function currently not supported")` at `:1443` is unreachable | n/a | grep | Consistent with the sheet's two "not ported" entries; the risk the reviewer names (a misspelling would be equally silent) is real. Not a defect today |
| B-M8 | P1a int mo-8 | `self.gp.predict(self.gp.X, self.gp.y, self.gp.s2, add_noise=False)` (`:1568-1570`) passes `y_star`/`s2_star`, which `gpyreg/gaussian_process.py:2024-2036` uses only under `add_noise` or `return_lpd` | **not a defect**, and a faithful port: `vbmc.m:756` is `[~,~,fmu,fs2] = gplite_pred(gp,gp.X,gp.y,gp.s2)`, passing the same arguments and taking the latent outputs | n/a | both sources read | Readability only |
| B-M9 | P1a cmp mo-1 | The final "finalize" line is printed only under `changed_flag` (`vbmc.py:1839`), where MATLAB prints under `new_final_vp_flag = (idx_best ~= iter) \|\| changedflag` (`vbmc.m:889`, `:895`, `:897`, `:908`); and its `sKL` compares against the `vp_old` captured at the *start* of the last iteration (`:1215`) where `vbmc.m:884` captures `vp_old = vp` *after* the loop | **confirmed port discrepancy**, display only (nothing records `sKL` after the loop on either side); PyVBMC has no counterpart of `new_final_vp_flag` at all | yes (a run whose returned posterior comes from an earlier iteration and whose boost was a no-op prints nothing, and the printed `sKL` mixes the last iteration's update with the boost) | both sources read | Python: `c477a7b` 2021-11-01. Rationale: none found. **A real, if cosmetic, defect** |
| B-M10 | P1a cmp mo-2 | `vp.gp` is never attached to the returned posterior; `vbmc.m:890-891` does `gp = stats.gp(idx_best); vp.gp = gp;` | **not a defect**: the only MATLAB consumer is `vbmc_rnd.m:46-49`'s `balanceflag == 'gp'` branch, and `known_differences.md:238-247` already records `gpsample_vbmc.m` as unported | n/a | grep over both trees | Consequence of a recorded unported feature; a cross-reference on the sheet would be optional |
| B-M11 | P1a cmp mo-6 | The warp re-warps only the active rows: `pyvbmc/whitening/whitening.py:203-209` selects `function_logger.X_flag`, where `misc/warp_input_vbmc.m:112-119` re-warps rows `1:optimState.Xn` regardless of the flag, so rows deactivated by a warm-up trim keep stale transformed coordinates | **confirmed difference**; belongs to slice P8, recorded here only | yes whenever a warp follows a trim | both sources read | Not assessed further, per the brief |
| B-M12 | P1a cmp mo-7 | The entropy-switch stability branch (`vbmc.py:2169`) tests only `entropy_switch`; `private/vbmc_termination.m:80` tests `optimState.EntropySwitch && isfinite(options.EntropyForceSwitch)` | **confirmed port discrepancy**, unreachable today because of B-1; with `entropy_force_switch = inf` MATLAB would allow termination where PyVBMC would turn the switch off and continue | no (unreachable) | both sources read | Python: `f1e6a0b` 2021-06-30 Marlon "feat: add loop termination methods to VBMC (#10)"; MATLAB's `isfinite` test predates the port. Rationale: none found. **A real defect, in the same repair as B-1** |
| B-M13 | P1a cmp mo-8 | String and formatting divergences: `problem_type` uses `"bounded"` where MATLAB uses `'boundconstraints'`; `best_iter` is 0-based where `output.bestiter` is 1-based; PyVBMC's finalize line has a `cache_active` branch (`:1858-1872`) that `vbmc.m:908-913` lacks; MATLAB's `stats.timer(iter).totalruntime = toc(t0)` (`:880`) has no Python counterpart | **confirmed**, all four; the first is unreachable (B-3a), the second is consistent with PyVBMC's own 0-based `iteration_history`, the third is a superset of MATLAB's in-loop format, the fourth is a genuine omission (only the comment at `vbmc.py:1600` survives) | yes for the `totalruntime` omission | both sources read; grep for `total_runtime` in `pyvbmc/` finds only the comment | `"bounded"` since `85ba77b` 2022-10-06 (the snake_case rename of the result keys). Cosmetic; the `totalruntime` omission is the only one worth a sheet line |

## Detail

### B-1 — the forced entropy switch

`pyvbmc/vbmc/vbmc.py:1224-1230`

```python
if self.optim_state.get("entropy_switch") and (
    self.function_logger.func_count
    >= self.optim_state.get("entropy_force_switch")
    * self.optim_state.get("max_fun_evals")
):
```

`vbmc.m:524-528`

```matlab
if optimState.EntropySwitch && ...
        optimState.funccount >= options.EntropyForceSwitch*options.MaxFunEvals
    optimState.EntropySwitch = false;
```

`_init_optim_state` copies `entropy_switch` (`:988`), forces it off for `D < det_entropy_min_d` (`:991-992`), copies `max_fun_evals` (`:1001`) and `entropy_alpha` (`:1035`), but never `entropy_force_switch`. The only two occurrences of the name in the tree are the read at `:1226` and the declaration `entropy_force_switch = 0.8` at `advanced_vbmc_options.ini:211`.

Reproduction (`f1_entropy_force_switch.py`):

```
D=2, options['entropy_switch'] = True
  optim_state['entropy_switch'] = False        <- forced off below D=5
  loop expression -> False
D=5, options['entropy_switch'] = True
  optim_state['entropy_switch'] = True
  'entropy_force_switch' in optim_state: False
  optim_state.get('entropy_force_switch') = None
  options.get('entropy_force_switch') = 0.8
  optim_state['max_fun_evals'] = 350
  loop expression raises: TypeError unsupported operand type(s) for *: 'NoneType' and 'int'
```

Both reports are right that the whole feature is unreachable: the only other write to the flag is the clearing at `:2172`, and `tol_stable_entropy_iters` (`:2124-2125`) can therefore never take effect either.

Scope note: the one-line read is not the whole repair — `vbmc.py:2169` must also gain MATLAB's `isfinite(options.EntropyForceSwitch)` test (B-M12), and no test enters the loop with the switch on, so a fix needs a new test rather than an edit to an existing one.

### B-2 — the running average of the variational moments

The guard and MATLAB's are quoted in the ledger. Confirmations:

- `len(np.eye(D) == 0) == D` for D ∈ {1,2,3,5,10}, so the guard is true from the second iteration onward (at the first, `run_mean == []` makes it true anyway).
- On `test_vbmc_save_static.pkl`, every recorded `optim_state` has `last_run_avg == N` (10/10, 15/15, …, 40/40), so `Nnew` would be 0 and `wRun` 1 even if the `else` branch ran; `run_mean[0]` jumps from `+0.0200` (iteration 3) to `−0.2535` (iteration 4), i.e. no smoothing.

**Where `moments_run_weight` is inert.** Nothing reads `run_mean`, `run_cov` or `last_run_avg` for a computation on either side. In PyVBMC the only other touches are the reset at `pyvbmc/whitening/whitening.py:244-246`; in MATLAB the only other touches are `misc/setupvars_vbmc.m:199-201` and the reset at `misc/warp_input_vbmc.m:165-167`. So `MomentsRunWeight`/`moments_run_weight` changes no number the algorithm computes **in either toolbox**. The difference is in the reported state: MATLAB returns the smoothed `optimState` as its 7th output (`vbmc.m:1`), PyVBMC exposes `vbmc.optim_state` and the per-iteration copies in `iteration_history`, and those hold the instantaneous moments. One further small difference, should the branch be revived: the reset writes `mubar.reshape(1, -1)`, a row, where `vbmc.m:782` writes `mubar(:)`, a column.

Scope note: fixing the parenthesis makes the `else` branch live for the first time; it has never executed, so the fix should come with a test of the branch itself. No golden trace or oracle records these keys (`dev/scripts/golden_trace.py` records only `optim_state["N"]`), so nothing needs re-baselining.

### B-3 — the two result fields

MATLAB, `private/vbmc_output.m:4-10`:

```matlab
output.function = func2str(optimState.fun);
if all(isinf(optimState.LB)) && all(isinf(optimState.UB))
    output.problemtype = 'unconstrained';
else
    output.problemtype = 'boundconstraints';
end
output.iterations = optimState.iter;
```

`misc/setupvars_vbmc.m:49-50` is the only assignment to `optimState.LB`/`.UB` anywhere in the MATLAB tree, and it stores `warpvars_vbmc(LB,'dir',trinfo)`. For the default `BoundedTransform = 'logit'` (type 3, `shared/warpvars_vbmc.m:104-110`) the lower bound gives `z = 0`, `log(0/(1-0)) = -Inf`; for probit (type 12, `:254-260`) `-sqrt(2)*erfcinv(0) = -Inf`; the identity branch leaves ±Inf for an unbounded variable. So MATLAB's test is always true as well: **shared defect**, and the P1a internal report's "no counterpart read" framing (internal track) understates it while P1a comparison F8's "suspected defect in both" is right.

`f3_result_fields.py` output (both `bounded_transform` settings identical):

```
bounded    lb_tran=[-inf -inf] ub_tran=[inf inf] -> 'unconstrained'  (lb_orig=[-5. -5.], ub_orig=[5. 5.])
mixed      lb_tran=[-inf -inf] ub_tran=[inf inf] -> 'unconstrained'  (lb_orig=[-5. -inf], ub_orig=[5. inf])
unbounded  lb_tran=[-inf -inf] ub_tran=[inf inf] -> 'unconstrained'
```

(PyVBMC rejects variables bounded on one side only, `pyvbmc/vbmc/_bounds.py:282`, so the "mixed" case is one bounded and one unbounded coordinate.)

**What the documentation says.** Nothing defines either field: `optimize`'s docstring says only "results : dict — A dictionary with additional information about the VBMC run"; `_create_result_dict`'s docstring is one line; no `.rst`/`.md` under `docsrc/source/` documents the dict. The only published values are the two `format_dict(results)` outputs in `examples/pyvbmc_example_3_diagnostics_and_saving.ipynb` (`'problem_type': 'unconstrained'`, `'iterations': 6` with `'best_iter': 5`, then `'iterations': 25` with `'best_iter': 25`), i.e. the 0-based value is what users see. So neither field has a documented contract that a fix would contradict.

Tests: `pyvbmc/testing/vbmc/test_vbmc_optimize.py:505` asserts `results["problem_type"] == "unconstrained"` for a genuinely unbounded problem — the one input on which the expression is right, so the assertion survives a fix. `test_vbmc_init.py:503-504` asserts `lb_tran == -inf` for a bounded problem; that is a correct statement about the transform and must not be changed.

Scope note for B-3a: a fix must also settle the string, since PyVBMC says `"bounded"` where MATLAB says `'boundconstraints'` (B-M13); the sheet has no entry for either field.

### B-4 — the empty slice in the warm-up end check

PyVBMC (`vbmc.py:1960-1972`) and MATLAB (`private/vbmc_warmup.m:35-42`) translate to the same index sets: MATLAB's 1-based `max(4, end-T+1):end` is PyVBMC's 0-based `[max(3, len-T):]`, and `3:max(3,end-T)` is `[2:max(3, len-T)]`. The guards line up too (`iter > T+1` 1-based ≡ `iteration > T` 0-based), so both first evaluate the block with a three-entry history.

At `T = 1` both `max_now` slices are empty (script output, "index sets" table: history 3 → python `[]`, matlab `[]`). PyVBMC raises there. MATLAB's `max([])` returns `[]`, the subtraction and comparison propagate the empty, and `StableCountFlag` reaches `vbmc_warmup.m:87` as a 0×0 logical, where `&&` requires an operand convertible to a logical scalar — MATLAB raises at that line. (This last step is reasoned from MATLAB's `&&` semantics; I could not execute MATLAB. If the PI wants certainty about the MATLAB failure mode rather than the fact that MATLAB also reaches an empty `max`, that single step is "needs MATLAB".)

Reproduction:

```
tol_stable_warmup=  5 fun_evals_per_iter=  5  T=1  iteration=2 (history 3)  -> raises ValueError: zero-size array …
tol_stable_warmup=  5 fun_evals_per_iter=  5  T=1  iteration=3 (history 4)  -> returned False
tol_stable_warmup= 15 fun_evals_per_iter= 15  T=1  iteration=2 (history 3)  -> raises ValueError
tol_stable_warmup= 15 fun_evals_per_iter=  5  T=3  iteration=4 (history 5)  -> returned False
```

Report correction: P1a internal F10 says the guard "only guarantees `len(elcbo_vec) >= tol_stable_warmup_iters + 2`" — right — but treats the raise as PyVBMC's alone; MATLAB reaches the same empty window, so the class is shared, not Python-only. Its statement that the defaults are safe (window 3) is confirmed.

Scope note: the repair is one expression, but the empty case has to be decided (MATLAB's intent is "no stable count yet", i.e. `stable_count_flag = False`), and no test covers a short `tol_stable_warmup`; the four `test_check_warmup_end_conditions_*` tests use constant vectors of length 101.

### B-5 — the separate search GP

MATLAB, `vbmc.m:471` and `:638-648`:

```matlab
gp = [];        hypstruct = [];     hypstruct_search = [];
...
if mod(iter,2) == 0
    meantemp = optimState.gpMeanfun;
    optimState.gpMeanfun = 'const';
    [gp_search,hypstruct_search] = gptrain_vbmc(hypstruct_search,optimState,stats,options);
    optimState.gpMeanfun = meantemp;
```

`gptrain_vbmc`'s signature is `[gp,hypstruct,Ns_gp,optimState]`, and its `:103` writes `optimState.sn2hpd = estimate_GPnoise(gp)`. Capturing two outputs discards both the fourth output and any effect on the main `hypstruct`.

PyVBMC, `vbmc.py:1415-1431`, passes `self.hyp_dict` into `train_gp`, rebinds it from the return, and writes `self.optim_state["sn2_hpd"] = sn2_hpd` at `:1430`. After that line `hyp_dict["hyp"]`, `["full"]`, `["logp"]` and `["run_cov"]` all describe a constant-mean GP.

The downstream failure, confirmed by shapes: at D=2 the constant-mean model has 5 hyperparameters and the negative-quadratic 9 (6 and 12 at D=3). `gaussian_process_train.py:135` builds `hyp0 = np.empty((0, hyp_dict["hyp"].T.shape[0]))` — width 5 — and `:143-149` concatenates the recorded GPs' `(Ns, 9)` blocks:

```
concatenate raises: ValueError all the input array dimensions except for the concatenation
axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the
array at index 1 has size 9
```

The guard that exists for a model change, `if hyp0.shape[1] != np.size(gp.hyper_priors["mu"]): hyp0 = None` (`:159-160`), is reached only after the concatenation. The search GP's own `train_gp` call survives (there `hyp0` is width 9 and the guard does reset it); the failure is the *next* `train_gp`, in the same iteration — either the main one at `vbmc.py:1474`, or, if `active_sample_gp_update` is enabled (default False), the in-sampling refit at `active_sample.py:742-755`. The reviewer's "the run reaches its second iteration" is right if iterations are counted 1-based; in PyVBMC's 0-based terms it is iteration 1, the first odd one.

Two independent defects here, both absent from MATLAB: the shared hyperparameter struct, and the `sn2_hpd` write that MATLAB deliberately drops by capturing two outputs (`pyvbmc/vbmc/active_sample.py:744` reads `sn2_hpd` during the search).

Scope note: a fix needs a second `hyp_dict` carried on the instance (construction, `load`/resume and the recorded `optim_state["hyp_dict"]` all touch it), not only the two lines.

### B-6 — the sieve size in the warp branch

`vbmc.py:1317-1328`:

```python
if not self.vp.optimize_mu:
    self.vp.mu = self.gp.X.T
    Knew = self.vp.mu.shape[1]
else:
    Knew = self.vp.K
N_fastopts = math.ceil(self.options.eval("ns_elbo", {"K": self.vp.K}))
```

`vbmc.m:575-584` is the same block with `Nfastopts = ceil(evaloption_vbmc(options.NSelbo,Knew))`. `Knew` *is* passed to `optimize_vp` at `:1341`, so only the sieve size differs.

With `variable_means=False` and `ns_elbo = 50*K` (identical to `defopts.NSelbo = '@(K) 50*K'`):

```
vp.K | training set | python N_fastopts | matlab N_fastopts
   2 |           30 |               100 |              1500
   2 |           60 |               100 |              3000
   5 |           80 |               250 |              4000
  10 |          120 |               500 |              6000
```

**Recorded rationale, partial.** `2df0d7e` fixes issue #98, "`self.K` is always `self.options.get("kwarmup")`":

> `self.K` is never updated. Therefore the following `N_fastopts` is a fixed number. **`Knew` should be used instead.** But it seems not affect much performance. […] Also, we should either update `self.K` at each iteration or just delete it

and lacerbi's reply:

> As far as I can tell, `K` is "the number of components in the current variational posterior". So perhaps we should get it from the `variational_posterior` and no reason to keep a duplicate?

The PR body (#101) says only "Fixes issue #98 by getting rid of `vbmc.K`, since the number of variational components `K` is an attribute of `vp`." So the *deliberate* decision was to stop using the stale `k_warmup`; the issue names `Knew` as the intended value, and the implementation substituted `vp.K` at both sites, which coincides with `Knew` everywhere except the `optimize_mu = False` arm of the warp branch. In other words, the sheet's P6 entry has a genuine rationale for the main loop and none for the warp branch.

Scope note: the sheet entry "The sieve asks for candidates in proportion to the current `K`" (`known_differences.md:1050-1070`) and the statement in `verification/wave1_P6.md` ("PyVBMC's warp branch (`:1322`) and `final_boost` (`:2484`) both agree with their MATLAB counterparts") both need correcting whatever the PI decides.

## Errors found in the reviewers' reports

1. **P1a internal F3** classifies `results["problem_type"]` as a PyVBMC defect; it is shared with MATLAB (`private/vbmc_output.m:5-9` tests the transformed `optimState.LB`/`.UB`). The internal track did not read MATLAB, so this is a framing limitation rather than a misreading; P1a comparison F8 has it right.
2. **P1a internal F10** treats the empty-slice raise as PyVBMC-specific. MATLAB reaches the same empty `max` in the same configuration; the class is a shared defect.
3. **P1a internal mo-1** ("recorded posteriors never carry the stability flag") is a faithful port, not a divergence: `savestats` (`vbmc.m:803`, `:1043`) records `vp` before `:811` sets `vp.stats.stable`.
4. **P1a internal mo-6** lists `vp_repo` among write-only leftovers without noting that MATLAB reads it (`misc/vpsieve_vbmc.m:60-66`, gated by `VariationalInitRepo`, default off). `redo_roto_scaling`, `pruned` and `iter_list` are write-only in MATLAB too, which the report does not say.
5. **P1a internal mo-8** flags the `gp.predict(gp.X, gp.y, gp.s2, add_noise=False)` call as reading oddly; it is character-for-character MATLAB's `gplite_pred(gp,gp.X,gp.y,gp.s2)` (`vbmc.m:756`). Not a divergence.
6. **Dating imprecision (minor).** P1a comparison F5 dates the MATLAB entropy-switch block "predates `2044530`, 2021-06-18" and P1b comparison F2 says the lines "last changed in `cc56c7c` (2018-08-05)". `git log -L 524,528:vbmc.m` attributes the last content change to `224e333` (2018-07-09, "added entropy switch options"). Both conclusions (predates the port) stand.
7. **P1a comparison F6** says the crash comes "in its second iteration"; in PyVBMC's 0-based numbering it is iteration 1. It also omits that the in-sampling refit (`active_sample.py:742-755`) would fail first if `active_sample_gp_update` were on.

Everything else I checked in these two reports — line numbers, quoted code, MATLAB references, consequences — held up.

## Recorded rationale: where I looked

For every row: the surrounding comments; `git log`/`git log -L`/`git log -S` messages on `pyvbmc/vbmc/vbmc.py`; `pyvbmc/vbmc/README.md` (the porting log — it maps `_check_warmup_end_conditions`/`_setup_vbmc_after_warmup` to `vbmc_warmup.m` and nothing more) and the other `pyvbmc/*/README.md`; `docsrc/source/` and the shipped notebooks; `dev/` in full, including §9 "Latent bugs and cleanups found along the way" of `dev/2026-09-02-modernization-discussion.md` (none of these findings appears there) and `known_differences.md`; the test suite; and `gh search issues`/`gh search prs` plus `gh issue view` / `gh pr view` on `acerbilab/pyvbmc` (authenticated as `lacerbi`). The only rationale found anywhere is issue #98 / PR #101 for B-6, quoted above; for B-1, B-2, B-3a, B-3b, B-4, B-5 and every minor row except B-M2 (`359dd90`'s "Small reference vs. copy issues" bullet, which explains the twin that *was* fixed): none found.
