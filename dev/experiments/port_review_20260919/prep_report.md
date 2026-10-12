# Preparatory agent — report

Wave 0 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories. It wrote
`known_differences.md` (67 entries plus 7 settled non-differences) and
`counterpart_map.md` (124 rows, one per `.m` file of `../vbmc` at
`396d649` except 17 demo, plot, example, test and installer files listed
at its end) in this directory. The text below is the agent's final message
about them, unedited: the mismatches between the counterpart map and the
plan's slice table, and the points where the repositories sharpen or
contradict the plan's reference facts. The orchestrator applies these to
the plan before the comparison-track waves.

---

### 1. Counterpart in a different slice's Python files than the table assigns

| MATLAB file | Table's slice | Python counterpart | Slice that owns that Python file | Suggested action |
| --- | --- | --- | --- | --- |
| `misc/fess_vbmc.m` | P1a (`vbmc.py`) | `pyvbmc/vbmc/active_importance_sampling.py:442: fess` | P4 | Move to P4. Its MATLAB callers are `private/activeimportancesampling_vbmc.m` (P4) and `private/activesample_vbmc.m` (P2); nothing in `vbmc.m` calls it. |
| `private/updateK.m` | P1a (`vbmc.py`) | `pyvbmc/vbmc/variational_optimization.py:19: update_K` | P6 | Either move to P6, or add `update_K` to P1a's Python file list explicitly. `vbmc.py: optimize` is the caller, so P1a reads the call site but not the function. |
| `misc/get_traindata_vbmc.m` | P2 (`active_sample.py`) | `pyvbmc/vbmc/gaussian_process_train.py:794: _get_training_data` | P5 | Move to P5, or name the function in P2's Python column. |
| `misc/gpreupdate.m` | P2 (`active_sample.py`) | `pyvbmc/vbmc/gaussian_process_train.py:938: reupdate_gp` | P5 | Same. `active_sample.py` calls it; the code is in a P5 file. |
| `misc/real2int_vbmc.m` | P8 | `pyvbmc/acquisition_functions/abstract_acq_fcn.py:261: AbstractAcqFcn._real2int` | P3 | Move to P3, or add the function to P8's Python column. It is a static method of the acquisition base class, called from `active_sample.py`. |
| `misc/vptrain2real.m` | P7 | none; the call is commented out at `pyvbmc/vbmc/vbmc.py:1344` and `:1536` | P1a holds the only trace | Mark *unported* in P7, and note the two commented-out call sites for the P1a reviewer. At the default `temperature = 1` MATLAB's function is the identity, so nothing is lost at defaults. |

### 2. MATLAB file with a Python counterpart that no slice names

| MATLAB file | Python counterpart | Suggested slice |
| --- | --- | --- |
| `private/vbmc_output.m` | `pyvbmc/vbmc/vbmc.py:3097: _create_result_dict` | P1a (whose Python column already says "the result dict"; only the MATLAB column omits the file) |
| `misc/vbmc_gphyp.m` | `pyvbmc/vbmc/gaussian_process_train.py:279: _gp_hyp` | P5. Note: the `.m` file is **zero bytes** at `396d649`; the function is a local subfunction of `misc/gptrain_vbmc.m:109`. |
| `utils/evalbool.m` | `pyvbmc/vbmc/options.py` (`.ini` values are Python literals; MATLAB's `'yes'`/`'no'` strings have no analogue) | P1b |
| `utils/sq_dist.m` | `pyvbmc/acquisition_functions/abstract_acq_fcn.py:288: AbstractAcqFcn._sq_dist` | P3. G2 names only `gplite/private/sq_dist.m`, whose counterpart is `scipy.spatial.distance.cdist` in gpyreg. The two MATLAB files are identical copies but their Python counterparts are in different repositories. |
| `utils/quantile1.m`, `gplite/private/quantile1.m` | `np.quantile` in `gpyreg/mean_functions.py:500`, `:511`, `:518` | G2 (the quantile convention differs; see the known-differences sheet) |
| `utils/slicesamplebnd.m` | `gpyreg/slice_sample.py: SliceSampler` (duplicate of `gplite/private/slicesamplebnd.m`) | G1, as a duplicate |
| `utils/fastkmeans.m` | none — and that absence is why `misc/initdesign_vbmc.m` is only partly ported | P2, marked *unported* |
| `shared/qtrapz.m` | inlined in `pyvbmc/variational_posterior/variational_posterior.py: mtv` via `scipy.integrate.trapezoid` | P7 |
| `vbmc_plot.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.plot` (uses the `corner` package) | P7 if plotting is in scope, or add it to the out-of-scope list; the current plan does neither. |
| `gplite/gplite_plot.m` | `gpyreg/gaussian_process.py:2259: GP.plot` | G2 if plotting is in scope; same remark. |
| `utils/fminfill.m` | none (VBMC's own copy; gpyreg ports `gplite/private/fminfill.m`) | See §4, item 3: slice M's commit table assumes a Python surface that does not exist. |
| `utils/covcma.m`, `utils/slicelite.m`, `utils/eissample_lite.m` | none | Explicitly *unported* under P2, P5 and P4 respectively, so the comparison reviewers do not hunt for them. `utils/covcma.m`'s only MATLAB call site is commented out; `slicelite` is one of the five unported GP hyperparameter samplers of `misc/get_GPTrainOptions.m`; `utils/eissample_lite.m` duplicates `gplite/private/eissample_lite.m`, already named in P4. |

### 3. Python module that no slice covers

Only one, excluding tests and the modules the plan lists as out of scope:

- **`pyvbmc/__init__.py`** — it resolves `pyvbmc.SVBMC` lazily (so that `import pyvbmc` never imports Torch) and performs the eager imports of `matplotlib.pyplot`, `cma` and `imageio` that `AGENTS.md` warns about. No slice names it. Suggested: add to N1 (the lazy S-VBMC resolution) or to P1b.

Everything else under `pyvbmc/` and `gpyreg/gpyreg/` falls inside a slice's glob (`acquisition_functions/*.py`, `entropy/*.py`, `stats/*.py`, `priors/*.py`, `svbmc/*.py`, `pymc/*.py`, `calibration/*.py`) or is listed as out of scope. The remaining uncovered files are `__init__.py` shims and `gpyreg/_version.py` (generated). Note that `pyvbmc/priors/__init__.py` is *not* a trivial shim — its import order is load-bearing — but it is inside P9's glob.

### 4. Counterparts in the table to qualify

1. **P2, `misc/proposal_vbmc.m`.** The table lists it without a marker, but it has no Python counterpart: `pyvbmc/vbmc/vbmc.py:963` stores the literal string `"@(x)proposal_vbmc"` in `optim_state["proposal_fcn"]` and nothing reads it. In MATLAB, `optimState.ProposalFcn` is likewise set in `misc/setupvars_vbmc.m:186-189` and read nowhere. Mark *unported (unused in MATLAB too)* so the P2 reviewer spends no time on it.
2. **P2, `utils/cmaes_modded.m`.** Correctly marked as substituted by the `cma` package. Worth adding that PyVBMC also subclasses `cma`'s noise handler (`_BatchedNoiseHandler` in `active_sample.py`) and passes a `randn` callback bound to the run's generator, so the substitution extends to the re-evaluation policy and the population draws.
3. **Slice M, commit `bd38b48`.** The table describes it as "`utils/fminfill.m` update (VBMC's own copy, plus `acq/acqbald_vbmc.m`)". `acq/acqbald_vbmc.m` does not exist at the comparison revision — it was deleted in `ee7ad00` — and `utils/fminfill.m` has no Python counterpart, because gpyreg's `f_min_fill.py` ports `gplite/private/fminfill.m`. So `bd38b48` has no Python surface to check. The related commit `1d1f20d` (the `uuinv` fix in `gplite/private/fminfill.m`) *does*: gpyreg's reference copy already carries that fix, so the live question is whether `f_min_fill.py` implements the fixed mixture-of-uniforms inverse CDF.

### 5. Facts in "Reference revisions" that the repositories sharpen or contradict

**Verified as stated:** PyVBMC `dev-next` at `f91fdf0`; gpyreg `main` at `9e70e6b` (2026-09-14, "docs: date the 1.2.1 release notes"); MATLAB `master` at `396d649` with a clean working tree; 68 MATLAB commits since 2021-01-19; the six gplite files and their line counts (119 / 32 / 20 / 11 / 4 / 2); `gplite_intmeanfun.m` absent from gpyreg's copy; the "Missing port" comment for the integrated mean function in `gaussian_process_train.py` (at `:961` "Missing port: intmean part", and `:421` for the mixture-of-quadratics hyperprior).

Five refinements:

1. **The non-`intmeanfun` difference in gpyreg's `gplite` copy is one line, not a set of "later MATLAB edits".** Comparing file by file while ignoring whitespace *and* blank lines:
   - `gplite_post.m` (32), `gplite_train.m` (4) and `private/gplite_core.m` (119) differ **only** in `intmeanfun` code.
   - `private/fminfill.m`'s 2 lines are two blank lines. gpyreg's copy already contains the `1d1f20d` `uuinv` rewrite, so this file is not a "later MATLAB edit" the copy is missing.
   - `private/slicesamplebnd.m`'s 11 lines are **documentation comments only** (the `OPTIONS.Thin` and `OPTIONS.Burnin` paragraphs). No code.
   - `gplite_pred.m`'s 20 lines are `intmeanfun` code **plus exactly one substantive line**, the GP log-density fix of commit `68a197b` (2022-06-25):
     - gpyreg's copy: `lp(:,s) = -(ystar-ymu(:,s)).^2./(sn2_star*sn2_mult)/2-log(2*pi*sn2_star*sn2_mult)/2;`
     - target `396d649`: `lp(:,s) = -0.5*(ystar-ymu(:,s)).^2./ys2(:,s) - 0.5*log(2*pi*ys2(:,s));`

   So the entire non-`intmeanfun`, non-cosmetic gap between gpyreg's reference copy and the comparison target is that single predictive log-density line. Slices M and G2 should be pointed straight at it: does `GP.predict`'s `log_p` use the total predictive variance `ys2` or the noise term alone?
   The plan's file list also omits three files that differ by one blank line (`gplite_qpred.m`, `gplite_rnd.m`, `private/eissample_lite.m`) and `gplite_demo.m` (20 lines, a demo); gpyreg's copy carries one extra file, `gplite_demo_new.m`. All cosmetic.

2. **`misc/vbmc_gphyp.m` is a zero-byte tracked file at `396d649`.** A reviewer told to compare it will find nothing. The GP hyperprior setup is the local subfunction `vbmc_gphyp` inside `misc/gptrain_vbmc.m:109`, whose counterpart is `gaussian_process_train.py:279: _gp_hyp`.

3. **gpyreg's `AGENTS.md` §"Relation to the MATLAB reference" is wrong on three mappings.** Every gpyreg reviewer receives that file through the harness, so the brief should warn about it:
   - "`gplite_quad.m` ... not ported" — it **is** ported, as `gpyreg/gaussian_process.py:2074: GP.quad` ("Bayesian quadrature for a Gaussian Process"). The plan's G2 row is right to list it.
   - "`gplite_sample.m` to `slice_sample.py`" — `slice_sample.py` is the port of `gplite/private/slicesamplebnd.m`. `gplite_sample.m` (drawing MCMC samples from the log density a GP represents) has **no** Python counterpart.
   - "`gplite_fmin.m` to `f_min_fill.py`" — `f_min_fill.py` is the port of `gplite/private/fminfill.m` (the Sobol space-filling design for the hyperparameter optimizer). `gplite_fmin.m` (global optimization of a GP-represented density from multiple starts) has **no** Python counterpart.
   - Minor: "Isotropic kernels are a Python-only addition" is true in substance but `gplite_covfun.m` does declare the `seiso` identifier in its hyperparameter-count switch (`Ncov = 2`, `covbase = 0`); it simply has no compute branch, so the kernel is unusable in MATLAB.

4. **The porting log `pyvbmc/vbmc/README.md` names two functions that no longer exist.** It lists `_reupdate_gp()` (the function is the public `reupdate_gp` in `gaussian_process_train.py:938`) and `_get_hpd()` under `gaussian_process_train.py` (the function moved to `pyvbmc/stats/get_hpd.py: get_hpd`). Its statement "sample(): vbmc_rnd.m — gp_sample is missing" is correct and corresponds to `vbmc_rnd.m:49`. Every reviewer is handed this log as a specification source, so the brief should say it is dated.

5. **PyVBMC's first commit.** The plan cites `d974e20` (2021-01-19). The repository's first commit is `53329d9` "Initial commit" on the same day; `d974e20` is the first code commit ("feat: initial commit of first version of code port"). Immaterial to the dating rule.

### 6. Observations for the orchestrator (not sheet entries, not mismatches)

These came out of the mechanical work and are questions for reviewers, not settled differences, so they are deliberately absent from the sheet.

1. **Four PyVBMC options correspond to MATLAB options deleted in 2021.** `double_gp`, `empirical_gp_prior`, `integrate_gp_mean` and `gp_stochastic_step_size` (all `False` by default, `advanced_vbmc_options.ini:313`, `:235`, `:305`, `:135`) existed in `vbmc.m` before commit `2044530` (2021-06-18, "cleanup: removed many unused or experimental features") and were removed there. PyVBMC was ported around that boundary and kept them. Whether they do anything, and whether they should stay, is a slice-M / P1b question.

2. **Option-name comparison.** Normalizing case and underscores, `vbmc.m`'s `defopts` block declares 180 options and PyVBMC's two `.ini` files declare 183. After matching the two renamings (`AlwaysRefitVarPost` ↔ `always_refit_vp`, `DetEntTolOpt` ↔ `det_entropy_tol_opt`):
   - **MATLAB only (12):** `AltMCEntropy`, `Bandwidth`, `BOWarmup`, `ELCBOWeight`, `FeatureTest`, `gpOutwarpFun`, `OptimToolbox`, `RepeatedAcqDiscount`, `RetryMaxFunEvals`, `VarActiveSample`, `VarParamsBack`, `WarpNonlinear`. Each is covered by an entry of the known-differences sheet except `Bandwidth` and `FeatureTest`, which no source document explains; they are a P1b/M question.
   - **Python only (15):** `do_final_boost`, `double_gp`, `empirical_gp_prior`, `gp_stochastic_step_size`, `integrate_gp_mean`, `log_file_level`, `log_file_mode`, `log_file_name`, `ns_gp_max_active`, `performance_calibration`, `print_iteration_header`, `record_full_history_details`, `show_tips`, `tol_elcbo_boost`, `vectorized_target`.
   I did not compare the 168 shared options' *values*; that is a P1b task, and the sheet records only the default differences a source document explains. Spot checks of the search-related defaults (`ns_search`, `heavy_tail_search_frac`, `mvn_search_frac`, `hpd_search_frac`, `box_search_frac`, `search_cache_frac`, `search_optimizer`, `max_repeated_observations`) found them identical.

3. **Six "Missing port" markers remain in `active_sample.py`** (`:327`, `:349`, `:564`, `:609`, `:652`, `:665`), referring to `private/activesample_vbmc.m` lines 185-205 (the per-evaluation algorithmic cost model `t_algoperfuneval`), 330-364 (the repeated-observation comparison), 388-402 (the acquisition debug table) and 425-432 (a commented-out search-bound tightening). The sheet covers the cost model and the repeated-observation block; the other two are debug/commented-out MATLAB code with nothing to port, which the P2 reviewer can confirm in a minute.

4. **`gp_int_mean_fun = 0` is a live but dead option** in PyVBMC (`advanced_vbmc_options.ini:111`, copied to `optim_state["int_mean_fun"]` at `vbmc.py:1100`, read nowhere). Recorded under the sheet's first entry.

## Counts

- Known-differences sheet: **67 entries** + **7 settled non-differences**.
- Counterpart map: **124 rows** (+ 17 excluded demo/plot/example/test/installer files listed separately; 124 + 17 = 141 `.m` files).

## Working-rule compliance

All three repositories are unmodified. `git status --porcelain` in `../gpyreg` and `../vbmc` is empty; in `pyvbmc` only the orchestrator's own expected changes are present (`dev/README.md`, `dev/TODO.md`, `dev/plans/port-correctness-review.md`, `dev/experiments/port_review_20260919/`). No test suite, `optimize()` run or install was executed; only `ls`, `find`, `grep`, `diff`, `git log/show/cat-file` and two short read-only Python snippets over already-extracted text files.
