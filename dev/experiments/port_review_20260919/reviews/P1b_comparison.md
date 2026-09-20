# P1b — setup, options, defaults, bounds, history, save/load, MATLAB-comparison track

Raw reviewer report, wave 2 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus agent,
fresh context, read-only on the repositories; it compared the Python code
with MATLAB VBMC at `396d649`. Its check scripts ran in the session
scratchpad and are not retained. The text below is the agent's final
message, unedited.

---

# Slice P1b, MATLAB-comparison track — report

PyVBMC `dev-port-review` at `51451dc`; MATLAB VBMC `master` at `396d649`; gpyreg `main` at `9e70e6b` (read only as an interface, for the GP noise-function spec).

(Note: the harness refused the request to write `REPORT.md` into the scratch directory — "Subagents should return findings as text, not write report files" — so the full report is delivered here instead. The check scripts and the generated defaults table remain at `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\b8a4c56e-a31c-47b4-87d0-66a2c9861ded\scratchpad\port_review\P1b_comparison\`.)

---

## 1. Coverage

### 1.1 Files read

| MATLAB file | Read completely | Python counterpart | Read completely |
| --- | --- | --- | --- |
| `vbmc.m` lines 1–560 (help, `defopts`, argument handling, setup before the loop) and 1021–1153 (`savestats`, `add2path`, `initFromVP`, `robustSampleFromVP`, `timer_init`, `getAcqInfo`) | yes | `pyvbmc/vbmc/vbmc.py`: class docstring + `__init__` (`:62`–`:492`), `target` (`:494`), `_normalized_hard_bound` (`:502`), `_resolve_pymc_target` (`:516`), `_validate_initialization_cost` (`:595`), `_precomputed_*` (`:610`, `:637`), `_initialize_precomputed_evaluations` (`:648`), `_validate_initial_fresh_budget` (`:796`), `_fresh_evaluations_for_batch` (`:821`), `_bounds_check` (`:831`), `_init_optim_state` (`:851`), `save` (`:2795`), `load` (`:2834`), `_optim_state_record` (`:3035`), `_get_random_state` (`:3056`), `_set_random_state` (`:3093`), `_create_result_dict` (`:3119`), `_init_logger` (`:3255`), `_init_log_joint` (`:3336`), `_validate_*` (`:3418`, `:3429`, `:3435`, `:2571`, `:2584`), `_rebuild_log_joint` (`:3453`) | yes |
| `misc/setupoptions_vbmc.m` (179 lines) | yes | `pyvbmc/vbmc/options.py` (437 lines) and the two `.ini` files | yes |
| `misc/setupvars_vbmc.m` (312 lines) | yes | `VBMC.__init__` + `_init_optim_state` | yes |
| `misc/boundscheck_vbmc.m` (143 lines) | yes | `pyvbmc/vbmc/_bounds.py` (294 lines) | yes |
| `misc/evaloption_vbmc.m` (9 lines) | yes | `Options.eval` (`options.py:306`) | yes |
| `utils/evalbool.m` (24 lines) | yes | no counterpart (`.ini` values are Python literals) | n/a |
| `lpostfun.m` (23 lines) | yes | `VBMC._init_log_joint` (`vbmc.py:3336`) | yes |
| `vbmc_diagnostics.m` | header + signature only (confirmed unported, see below) | none | n/a |
| `private/vbmc_output.m` (30 lines), read only to check `rng_state`/`overhead` | yes | `_create_result_dict` | yes |
| `acq/acqwrapper_vbmc.m`, `acq/acqf_vbmc.m`, `acq/acqflog_vbmc.m` (read only to judge the default-acquisition difference, F5) | yes | `pyvbmc/acquisition_functions/abstract_acq_fcn.py:60-188`, `acq_fcn_log.py` | partial (P3's slice) |
| `private/vbmc_termination.m:1-30`, `:90-103` (read only for the `MinIter`/`MaxIter` semantics) | partial | `vbmc.py:2091-2206` (P1a's slice) | partial |
| `gplite/gplite_noisefun.m:1-45` (interface only) | header only | — | — |

Also read completely on the Python side: `pyvbmc/vbmc/iteration_history.py`, `pyvbmc/rng.py`, `pyvbmc/__init__.py`, both `option_configs/*.ini`.

**`vbmc_diagnostics.m` is unported.** It is a cross-run convergence diagnostic (`[exitflag,best,idx_best,stats] = vbmc_diagnostics(vp_array,...)`). No file under `pyvbmc/` implements it; the modules that contain the string "diagnostics" are the slice-sampler option in `pyvbmc/vbmc/active_importance_sampling.py:437`, the declared-but-unread `diagnostics` option, and unrelated PyMC/calibration text. This matches the sheet entry. No further time spent.

**Precomputed-evaluation and PyMC helpers leave the MATLAB-equivalent path untouched when unused** (checked line by line, as instructed): `_resolve_pymc_target` returns its arguments unchanged when `log_density` is not a `PyMCTarget` (`vbmc.py:534-545`); `_initialize_precomputed_evaluations` returns immediately for `None` (`:651-652`); `_validate_initial_fresh_budget` and `_fresh_evaluations_for_batch` return immediately when `_budget_active` is false (`:798-799`, `:823-824`); and `optim_state["max_fun_evals"]` is `self._effective_max_fun_evals`, which equals `options["max_fun_evals"]` when `initialization_cost` is zero (`:352-354`, `:1001`). `optim_state` gains the keys `max_fun_evals_total`, `initialization_cost`, `budget_active` only in the opted-in case (`:1002-1005`). These have no MATLAB counterpart and are on the sheet. Confirmed; no further time spent.

### 1.2 The complete defaults table

MATLAB has 180 `defopts` entries; PyVBMC declares 183 options across the two `.ini` files. Every expression that depends on `D` was evaluated on both sides for `D = 1, 2, 5, 9, 10, 15, 20, 30`, and every option declared as a function was compared as a function at `K` (or `N`) `= 1, 2, 3, 7, 10, 50`. MATLAB's `nvars` and `D` are the same quantity (`misc/setupoptions_vbmc.m:6`).

"agrees" below means the values coincide at every tested `D` (and at every tested argument, for the function-valued options).

| MATLAB option (vbmc.m line) | MATLAB default | Python option (.ini line) | Python default | Verdict |
| --- | --- | --- | --- | --- |
| `Display` (158) | `iter` | `display` (basic 3) | `"iter"` | agrees (accepted *values* differ; minor obs. M8) |
| `Plot` (159) | `off` | `plot` (basic 7) | `False` | agrees |
| `MaxIter` (160) | `50*(2+nvars)` | `max_iter` (basic 9) | `50*(2+D)` | agrees |
| `MaxFunEvals` (161) | `50*(2+nvars)` | `max_fun_evals` (basic 11) | `50*(2+D)` | agrees |
| `FunEvalsPerIter` (162) | `5` | `fun_evals_per_iter` (basic 13) | `5` | agrees |
| `TolStableCount` (163) | `60` | `tol_stable_count` (basic 15) | `60` | agrees |
| `RetryMaxFunEvals` (164) | `0` | none | none | MATLAB-only, unported, on the sheet (P1a) |
| `MinFinalComponents` (165) | `50` | `min_final_components` (basic 17) | `50` | agrees |
| `SpecifyTargetNoise` (166) | `no` | `specify_target_noise` (basic 19) | `False` | agrees |
| `UncertaintyHandling` (193) | `[]` | `uncertainty_handling` (adv 3) | `[]` | default agrees; **semantics differ, F4**; **overrides differ, F1** |
| `IntegerVars` (194) | `[]` | `integer_vars` (adv 5) | `[]` | default agrees; **semantics differ, F3** |
| `NoiseSize` (195) | `[]` | `noise_size` (adv 7) | `[]` | agrees (validation missing, minor obs. M1) |
| `MaxRepeatedObservations` (196) | `0` | `max_repeated_observations` (adv 9) | `0` | agrees |
| `RepeatedAcqDiscount` (197) | `1` | none | none | MATLAB-only, unported, on the sheet (P2) |
| `FunEvalStart` (198) | `10*ceil((D+1)/10)` | `fun_eval_start` (adv 11) | `10 * ceil((D + 1) / 10)` | agrees |
| `SGDStepSize` (199) | `0.005` | `sgd_step_size` (adv 13) | `0.005` | agrees |
| `SkipActiveSamplingAfterWarmup` (200) | `no` | `skip_active_sampling_after_warmup` (adv 15) | `False` | agrees |
| `RankCriterion` (201) | `yes` | `rank_criterion` (adv 17) | `True` | agrees |
| `TolStableEntropyIters` (202) | `6` | `tol_stable_entropy_iters` (adv 19) | `6` | agrees |
| `VariableMeans` (203) | `yes` | `variable_means` (adv 21) | `True` | agrees |
| `VariableWeights` (204) | `yes` | `variable_weights` (adv 23) | `True` | agrees |
| `WeightPenalty` (205) | `0.1` | `weight_penalty` (adv 25) | `0.1` | agrees |
| `Diagnostics` (206) | `off` | `diagnostics` (adv 27) | `False` | same meaning; dead on both sides (**F10**: not registered inert in Python) |
| `OutputFcn` (207) | `[]` | `output_fcn` (adv 29) | `[]` | agrees; dead on both sides (sheet: inert) |
| `TolStableExcptFrac` (208) | `0.2` | `tol_stable_excpt_frac` (adv 31) | `0.2` | agrees |
| `Fvals` (209) | `[]` | `f_vals` (adv 33) | `[]` | agrees |
| `OptimToolbox` (210) | `[]` | none | none | MATLAB-only, on the sheet (see sheet note S4) |
| `ProposalFcn` (211) | `[]` | `proposal_fcn` (adv 35) | `None` | agrees in meaning (`[]`/`None` both select the default proposal) |
| `NonlinearScaling` (212) | `on` | `nonlinear_scaling` (adv 37) | `True` | same meaning; dead on both sides (sheet: inert) |
| `SearchAcqFcn` (213) | `@acqf_vbmc` | `search_acq_fcn` (adv 39) | `[AcqFcnLog()]` | **DIFFERS, F5, not on the sheet** |
| `NSsearch` (214) | `2^13` | `ns_search` (adv 41) | `2 ** 13` | agrees |
| `NSent` (215) | `@(K) 100*K.^(2/3)` | `ns_ent` (adv 43) | `lambda K : 100 * K ** (2 / 3)` | agrees |
| `NSentFast` (216) | `0` | `ns_ent_fast` (adv 45) | `0` | agrees |
| `NSentFine` (217) | `@(K) 2^12*K` | `ns_ent_fine` (adv 47) | `lambda K : 2 ** 12 * K` | agrees |
| `NSentBoost` (218) | `@(K) 200*K.^(2/3)` | `ns_ent_boost` (adv 53) | `lambda K : 200 * K ** (2 / 3)` | agrees |
| `NSentFastBoost` (219) | `[]` | `ns_ent_fast_boost` (adv 55) | `[]` | agrees |
| `NSentFineBoost` (220) | `[]` | `ns_ent_fine_boost` (adv 57) | `[]` | agrees |
| `NSentActive` (221) | `@(K) 20*K.^(2/3)` | `ns_ent_active` (adv 59) | `lambda K : 20 * K ** (2 / 3)` | agrees |
| `NSentFastActive` (222) | `0` | `ns_ent_fast_active` (adv 61) | `0` | agrees |
| `NSentFineActive` (223) | `@(K) 200*K` | `ns_ent_fine_active` (adv 63) | `lambda K : 200 * K` | agrees |
| `NSelbo` (224) | `@(K) 50*K` | `ns_elbo` (adv 65) | `lambda K : 50 * K` | agrees (the *argument passed* differs; on the sheet, P6) |
| `NSelboIncr` (225) | `0.1` | `ns_elbo_incr` (adv 67) | `0.1` | agrees |
| `ElboStarts` (226) | `2` | `elbo_starts` (adv 69) | `2` | agrees |
| `NSgpMax` (227) | `80` | `ns_gp_max` (adv 71) | `80` | agrees |
| `NSgpMaxWarmup` (228) | `8` | `ns_gp_max_warmup` (adv 73) | `8` | agrees |
| `NSgpMaxMain` (229) | `Inf` | `ns_gp_max_main` (adv 75) | `np.inf` | agrees |
| `WarmupNoImproThreshold` (230) | `20 + 5*nvars` | `warmup_no_impro_threshold` (adv 79) | `20 + 5 * D` | agrees |
| `WarmupCheckMax` (231) | `yes` | `warmup_check_max` (adv 81) | `True` | agrees |
| `StableGPSampling` (232) | `200 + 10*nvars` | `stable_gp_sampling` (adv 83) | `200 + 10 * D` | agrees |
| `StableGPvpK` (233) | `Inf` | `stable_gp_vp_k` (adv 85) | `np.inf` | agrees |
| `StableGPSamples` (234) | `0` | `stable_gp_samples` (adv 87) | `0` | agrees |
| `GPSampleThin` (235) | `5` | `gp_sample_thin` (adv 89) | `5` | agrees |
| `GPTrainNinit` (236) | `1024` | `gp_train_n_init` (adv 91) | `1024` | agrees |
| `GPTrainNinitFinal` (237) | `64` | `gp_train_n_init_final` (adv 93) | `64` | agrees |
| `GPTrainInitMethod` (238) | `rand` | `gp_train_init_method` (adv 95) | `"rand"` | agrees |
| `GPTolOpt` (239) | `1e-5` | `gp_tol_opt` (adv 97) | `1e-5` | agrees |
| `GPTolOptMCMC` (240) | `1e-2` | `gp_tol_opt_mcmc` (adv 99) | `1e-2` | agrees |
| `GPTolOptActive` (241) | `1e-4` | `gp_tol_opt_active` (adv 101) | `1e-4` | agrees |
| `GPTolOptMCMCActive` (242) | `1e-2` | `gp_tol_opt_mcmc_active` (adv 103) | `1e-2` | agrees |
| `TolGPVar` (243) | `1e-4` | `tol_gp_var` (adv 105) | `1e-4` | agrees |
| `TolGPVarMCMC` (244) | `1e-4` | `tol_gp_var_mcmc` (adv 107) | `1e-4` | agrees |
| `gpMeanFun` (245) | `negquad` | `gp_mean_fun` (adv 109) | `"negquad"` | agrees |
| `gpIntMeanFun` (246) | `0` | `gp_int_mean_fun` (adv 111) | `0` | agrees (feature unported; sheet) |
| `KfunMax` (247) | `@(N) N.^(2/3)` | `k_fun_max` (adv 113) | `lambda N : N ** (2 / 3)` | agrees |
| `Kwarmup` (248) | `2` | `k_warmup` (adv 115) | `2` | agrees |
| `AdaptiveK` (249) | `2` | `adaptive_k` (adv 117) | `2` | agrees |
| `HPDFrac` (250) | `0.8` | `hpd_frac` (adv 119) | `0.8` | agrees |
| `ELCBOImproWeight` (251) | `3` | `elcbo_impro_weight` (adv 121) | `3` | agrees |
| `TolLength` (252) | `1e-6` | `tol_length` (adv 123) | `1e-6` | agrees |
| `CacheSize` (253) | `500` | `cache_size` (adv 125) | `500` | agrees |
| `CacheFrac` (254) | `0.5` | `cache_frac` (adv 127) | `0.5` | agrees |
| `StochasticOptimizer` (255) | `adam` | `stochastic_optimizer` (adv 129) | `"adam"` | agrees |
| `TolFunStochastic` (256) | `1e-3` | `tol_fun_stochastic` (adv 131) | `1e-3` | agrees |
| `MaxIterStochastic` (257) | `100*(2+nvars)` | `max_iter_stochastic` (adv 133) | `100 * (2 + D)` | agrees |
| `TolSD` (258) | `0.1` | `tol_sd` (adv 137) | `0.1` | agrees |
| `TolsKL` (259) | `0.01*sqrt(nvars)` | `tol_skl` (adv 139) | `0.01 * np.sqrt(D)` | agrees |
| `TolStableWarmup` (260) | `15` | `tol_stable_warmup` (adv 141) | `15` | agrees |
| `VariationalSampler` (261) | `malasample` | `variational_sampler` (adv 143) | `"malasample"` | agrees (inert in Python; sheet) |
| `TolImprovement` (262) | `0.01` | `tol_improvement` (adv 145) | `0.01` | agrees |
| `KLgauss` (263) | `yes` | `kl_gauss` (adv 147) | `True` | agrees |
| `TrueMean` (264) | `[]` | `true_mean` (adv 149) | `[]` | agrees |
| `TrueCov` (265) | `[]` | `true_cov` (adv 151) | `[]` | agrees |
| `MinFunEvals` (266) | `5*nvars` | `min_fun_evals` (adv 153) | `5 * D` | agrees |
| `MinIter` (267) | `nvars` | `min_iter` (adv 155) | `D` | value agrees; **comparison is off by one, F9** |
| `HeavyTailSearchFrac` (268) | `0.25` | `heavy_tail_search_frac` (adv 157) | `0.25` | agrees |
| `MVNSearchFrac` (269) | `0.25` | `mvn_search_frac` (adv 159) | `0.25` | agrees |
| `HPDSearchFrac` (270) | `0` | `hpd_search_frac` (adv 161) | `0` | agrees |
| `BoxSearchFrac` (271) | `0.25` | `box_search_frac` (adv 163) | `0.25` | agrees |
| `SearchCacheFrac` (272) | `0` | `search_cache_frac` (adv 165) | `0` | agrees |
| `AlwaysRefitVarPost` (273) | `no` | `always_refit_vp` (adv 167) | `False` | agrees (name differs) |
| `Warmup` (274) | `on` | `warmup` (adv 169) | `True` | agrees |
| `WarmupOptions` (275) | `[]` | `warmup_options` (adv 171) | `[]` | agrees (inert in Python; sheet) |
| `StopWarmupThresh` (276) | `0.2` | `stop_warmup_thresh` (adv 173) | `0.2` | agrees |
| `WarmupKeepThreshold` (277) | `10*nvars` | `warmup_keep_threshold` (adv 175) | `10 * D` | agrees |
| `WarmupKeepThresholdFalseAlarm` (278) | `100*(nvars+2)` | `warmup_keep_threshold_false_alarm` (adv 177) | `100 * (D + 2)` | agrees |
| `StopWarmupReliability` (279) | `100` | `stop_warmup_reliability` (adv 179) | `100` | agrees |
| `SearchOptimizer` (280) | `cmaes` | `search_optimizer` (adv 181) | `"cmaes"` | agrees (accepted values differ; sheet, P2) |
| `SearchCMAESVPInit` (281) | `yes` | `search_cmaes_vp_init` (adv 183) | `True` | agrees |
| `SearchCMAESbest` (282) | `no` | `search_cmaes_best` (adv 185) | `False` | agrees (inert in Python; sheet, P2) |
| `SearchMaxFunEvals` (283) | `500*(nvars+2)` | `search_max_fun_evals` (adv 187) | `500 * (D + 2)` | agrees |
| `MomentsRunWeight` (284) | `0.9` | `moments_run_weight` (adv 189) | `0.9` | agrees |
| `GPRetrainThreshold` (285) | `1` | `gp_retrain_threshold` (adv 191) | `1` | agrees |
| `ELCBOmidpoint` (286) | `on` | `elcbo_midpoint` (adv 193) | `True` | agrees |
| `GPSampleWidths` (287) | `5` | `gp_sample_widths` (adv 195) | `5` | agrees |
| `HypRunWeight` (288) | `0.9` | `hyp_run_weight` (adv 197) | `0.9` | agrees |
| `WeightedHypCov` (289) | `on` | `weighted_hyp_cov` (adv 199) | `True` | agrees |
| `TolCovWeight` (290) | `0` | `tol_cov_weight` (adv 201) | `0` | agrees |
| `GPHypSampler` (291) | `slicesample` | `gp_hyp_sampler` (adv 203) | `"slicesample"` | agrees |
| `CovSampleThresh` (292) | `10` | `cov_sample_thresh` (adv 205) | `10` | agrees |
| `DetEntTolOpt` (293) | `1e-3` | `det_entropy_tol_opt` (adv 207) | `1e-3` | agrees (meaning of the number differs; sheet, P6) |
| `EntropySwitch` (294) | `off` | `entropy_switch` (adv 209) | `False` | agrees |
| `EntropyForceSwitch` (295) | `0.8` | `entropy_force_switch` (adv 211) | `0.8` | value agrees; **never reaches the reader, F2** |
| `DetEntropyAlpha` (296) | `0` | `det_entropy_alpha` (adv 213) | `0` | agrees |
| `UpdateRandomAlpha` (297) | `no` | `update_random_alpha` (adv 215) | `False` | agrees |
| `AdaptiveEntropyAlpha` (298) | `no` | `adaptive_entropy_alpha` (adv 217) | `False` | agrees (inert in Python; sheet) |
| `DetEntropyMinD` (299) | `5` | `det_entropy_min_d` (adv 219) | `5` | agrees |
| `TolConLoss` (300) | `0.01` | `tol_con_loss` (adv 221) | `0.01` | agrees |
| `BestSafeSD` (301) | `5` | `best_safe_sd` (adv 223) | `5` | agrees |
| `BestFracBack` (302) | `0.25` | `best_frac_back` (adv 225) | `0.25` | agrees |
| `TolWeight` (303) | `1e-2` | `tol_weight` (adv 227) | `1e-2` | agrees |
| `PruningThresholdMultiplier` (304) | `@(K) 1/sqrt(K)` | `pruning_threshold_multiplier` (adv 229) | `lambda K : 1 / np.sqrt(K)` | agrees |
| `AnnealedGPMean` (305) | `@(N,NMAX) 0` | `annealed_gp_mean` (adv 231) | `lambda N,NMAX: 0` | agrees; dead on both sides (sheet) |
| `ConstrainedGPMean` (306) | `no` | `constrained_gp_mean` (adv 233) | `False` | agrees; dead on both sides (sheet); MATLAB never even evaluates it (minor obs. M4) |
| `TolGPNoise` (307) | `sqrt(1e-5)` | `tol_gp_noise` (adv 237) | `np.sqrt(1e-5)` | agrees |
| `GPLengthPriorMean` (308) | `sqrt(D/6)` | `gp_length_prior_mean` (adv 239) | `np.sqrt(D / 6)` | agrees |
| `GPLengthPriorStd` (309) | `0.5*log(1e3)` | `gp_length_prior_std` (adv 241) | `0.5 * np.log(1e3)` | agrees |
| `UpperGPLengthFactor` (310) | `0` | `upper_gp_length_factor` (adv 243) | `0` | agrees |
| `InitDesign` (311) | `plausible` | `init_design` (adv 245) | `"plausible"` | agrees |
| `gpQuadraticMeanBound` (312) | `yes` | `gp_quadratic_mean_bound` (adv 247) | `True` | agrees |
| `Bandwidth` (313) | `0` | none | none | MATLAB-only, unported, on the sheet (P6) |
| `FitnessShaping` (314) | `no` | `fitness_shaping` (adv 249) | `False` | agrees |
| `OutwarpThreshBase` (315) | `10*nvars` | `out_warp_thresh_base` (adv 251) | `10 * D` | agrees |
| `OutwarpThreshMult` (316) | `1.25` | `out_warp_thresh_mult` (adv 253) | `1.25` | agrees |
| `OutwarpThreshTol` (317) | `0.8` | `out_warp_thresh_tol` (adv 255) | `0.8` | agrees |
| `Temperature` (318) | `1` | `temperature` (adv 257) | `1` | value agrees; **never reaches a reader, F6** |
| `SeparateSearchGP` (319) | `no` | `separate_search_gp` (adv 259) | `False` | agrees |
| `NoiseShaping` (320) | `no` | `noise_shaping` (adv 261) | `False` | agrees (rejected when true; sheet) |
| `NoiseShapingThreshold` (321) | `10*nvars` | `noise_shaping_threshold` (adv 263) | `10 * D` | agrees (inert in Python; sheet) |
| `NoiseShapingFactor` (322) | `0.05` | `noise_shaping_factor` (adv 265) | `0.05` | agrees (inert in Python; sheet) |
| `AcqHedge` (323) | `no` | `acq_hedge` (adv 267) | `False` | agrees |
| `AcqHedgeIterWindow` (324) | `4` | `acq_hedge_iter_window` (adv 269) | `4` | agrees (inert in Python; sheet) |
| `AcqHedgeDecay` (325) | `0.9` | `acq_hedge_decay` (adv 271) | `0.9` | agrees (inert in Python; sheet) |
| `ActiveVariationalSamples` (326) | `0` | `active_variational_samples` (adv 273) | `0` | agrees (inert in Python; sheet) |
| `ScaleLowerBound` (327) | `yes` | `scale_lower_bound` (adv 275) | `True` | agrees (inert in Python; sheet) |
| `ActiveSampleVPUpdate` (328) | `no` | `active_sample_vp_update` (adv 277) | `False` | agrees (override condition differs, F1) |
| `ActiveSampleGPUpdate` (329) | `no` | `active_sample_gp_update` (adv 279) | `False` | agrees (override condition differs, F1) |
| `ActiveSampleFullUpdatePastWarmup` (330) | `2` | `active_sample_full_update_past_warmup` (adv 281) | `2` | agrees |
| `ActiveSampleFullUpdateThreshold` (331) | `3` | `active_sample_full_update_threshold` (adv 283) | `3` | agrees |
| `VariationalInitRepo` (332) | `no` | `variational_init_repo` (adv 285) | `False` | agrees (inert in Python; sheet) |
| `SampleExtraVPMeans` (333) | `0` | `sample_extra_vp_means` (adv 287) | `0` | agrees (inert in Python; sheet) |
| `OptimisticVariationalBound` (334) | `0` | `optimistic_variational_bound` (adv 289) | `0` | agrees (inert in Python; sheet) |
| `ActiveImportanceSamplingVPSamples` (335) | `100` | `active_importance_sampling_vp_samples` (adv 291) | `100` | agrees |
| `ActiveImportanceSamplingBoxSamples` (336) | `100` | `active_importance_sampling_box_samples` (adv 293) | `100` | agrees |
| `ActiveImportanceSamplingMCMCSamples` (337) | `100` | `active_importance_sampling_mcmc_samples` (adv 295) | `100` | agrees (MATLAB keeps it a char and `eval`s it at the call site, `private/activeimportancesampling_vbmc.m:40-51`; equivalent) |
| `ActiveImportanceSamplingMCMCThin` (338) | `1` | `active_importance_sampling_mcmc_thin` (adv 297) | `1` | agrees |
| `ActiveSamplefESSThresh` (339) | `1` | `active_sample_fess_thresh` (adv 299) | `1` | agrees (inert in Python; sheet) |
| `ActiveImportanceSamplingfESSThresh` (340) | `0.9` | `active_importance_sampling_fess_thresh` (adv 301) | `0.9` | agrees |
| `ActiveSearchBound` (341) | `2` | `active_search_bound` (adv 303) | `2` | agrees |
| `TolBoundX` (342) | `1e-5` | `tol_bound_x` (adv 307) | `1e-5` | agrees |
| `RecomputeLCBmax` (343) | `yes` | `recompute_lcb_max` (adv 309) | `True` | agrees |
| `BoundedTransform` (344) | `logit` | `bounded_transform` (adv 311) | `"probit"` | DIFFERS, on the sheet (P8) |
| `WarpEveryIters` (345) | `5` | `warp_every_iters` (adv 315) | `5` | agrees |
| `IncrementalWarpDelay` (346) | `yes` | `incremental_warp_delay` (adv 317) | `True` | agrees |
| `WarpTolReliability` (347) | `3` | `warp_tol_reliability` (adv 319) | `3` | agrees |
| `WarpRotoScaling` (348) | `yes` | `warp_rotoscaling` (adv 321) | `True` | agrees (name differs) |
| `WarpCovReg` (349) | `0` | `warp_cov_reg` (adv 323) | `0` | agrees |
| `WarpRotoCorrThresh` (350) | `0.05` | `warp_roto_corr_thresh` (adv 325) | `0.05` | agrees |
| `WarpMinK` (351) | `5` | `warp_min_k` (adv 327) | `5` | agrees |
| `WarpUndoCheck` (352) | `yes` | `warp_undo_check` (adv 329) | `True` | agrees |
| `WarpTolImprovement` (353) | `0.1` | `warp_tol_improvement` (adv 331) | `0.1` | agrees |
| `WarpTolSDMultiplier` (354) | `2` | `warp_tol_sd_multiplier` (adv 333) | `2` | agrees |
| `WarpTolSDBase` (355) | `1` | `warp_tol_sd_base` (adv 335) | `1` | agrees |
| `WarpNonlinear` (359) | `off` | none | none | MATLAB-only, unported, on the sheet (P8) |
| `ELCBOWeight` (360) | `0` | none | none | MATLAB-only, unported, on the sheet (P6) |
| `VarParamsBack` (361) | `0` | none | none | MATLAB-only, unported, on the sheet (P6) |
| `AltMCEntropy` (362) | `no` | none | none | MATLAB-only, unported, on the sheet (P7) |
| `VarActiveSample` (363) | `no` | none | none | MATLAB-only, unported, on the sheet (P2) |
| `FeatureTest` (364) | `no` | none | none | MATLAB-only, unported (map, `vbmc.m` row). **Read nowhere in MATLAB** — see below |
| `BOWarmup` (365) | `no` | none | none | MATLAB-only, unported, on the sheet (P1a) |
| `gpOutwarpFun` (366) | `[]` | none | none | MATLAB-only, unported, on the sheet (G) |
| none | none | `do_final_boost` (adv 49) | `True` | Python-only, on the sheet |
| none | none | `double_gp` (adv 313) | `False` | Python-only (MATLAB option deleted by `2044530`), on the sheet |
| none | none | `empirical_gp_prior` (adv 235) | `False` | Python-only (same), on the sheet |
| none | none | `gp_stochastic_step_size` (adv 135) | `False` | Python-only (same), on the sheet |
| none | none | `integrate_gp_mean` (adv 305) | `False` | Python-only (same), on the sheet |
| none | none | `log_file_level` (basic 25) | `"iter"` | Python-only, on the sheet |
| none | none | `log_file_mode` (basic 27) | `"a"` | Python-only, on the sheet |
| none | none | `log_file_name` (basic 23) | `None` | Python-only, on the sheet |
| none | none | `ns_gp_max_active` (adv 77) | `np.inf` | Python-only, on the sheet |
| none | none | `performance_calibration` (adv 339) | `"cached"` | Python-only, on the sheet |
| none | none | `print_iteration_header` (basic 29) | `None` | Python-only, on the sheet |
| none | none | `record_full_history_details` (adv 337) | `False` | Python-only, on the sheet |
| none | none | `show_tips` (basic 5) | `True` | Python-only, on the sheet |
| none | none | `tol_elcbo_boost` (adv 51) | `0.1` | Python-only, on the sheet (P1a) |
| none | none | `vectorized_target` (basic 21) | `False` | Python-only, on the sheet |

**Answer to the question left from the slice table: `FeatureTest` is read nowhere in MATLAB, and is not even evaluated.** `defopts.FeatureTest` (`vbmc.m:364`) is the only assignment; there is no other occurrence of the name in the repository except inside `misc/setupoptions_vbmc.m:47`, where the cell literal reads `'ConstrainedGPMean''FeatureTest'`. In MATLAB a doubled quote inside a single-quoted string is an escaped quote, so that element is the one string `ConstrainedGPMean'FeatureTest`, and neither `ConstrainedGPMean` nor `FeatureTest` appears in `evalfields`. Both therefore stay the raw character vectors `'no'` after `setupoptions_vbmc`, and nothing reads either.

### 1.3 `setupoptions_vbmc.m` post-default work against PyVBMC

| MATLAB step | Lines | PyVBMC counterpart | Verdict |
| --- | --- | --- | --- |
| Fill absent/empty fields from `defopts`, record which the user supplied in `updated` | `:10-16` | `Options.__init__` (`options.py:93-98`), `useroptions` set | equivalent |
| Strip `% comment` and trailing blanks from char options | `:19-26` | `.ini` `# description` lines above each value (`options.py:401-437`) | equivalent |
| `eval` the string options listed in `evalfields`, falling back to `evalbool` | `:29-81` | `.ini` values `eval`'d with `D` bound (`options.py:174`); `options=` dict verbatim | on the sheet ("layered options"); see M4 for the MATLAB quoting typo |
| Wrap `SearchAcqFcn` in a cell | `:84-90` | `.ini` already declares a list | equivalent |
| Determine `OptimToolbox` | `:98-106` | none | on the sheet; dead in MATLAB too (sheet note S4) |
| `MaxFunEvals`, `MaxIter` must be positive integers | `:109-114` | none | **F8** |
| `MaxIter < MinIter` → raise `MaxIter` | `:115-119` | none | **F8** |
| `MaxFunEvals < MinFunEvals` → warn (assignment is a MATLAB no-op) | `:120-124` | none | minor obs. M3 |
| `UncertaintyHandling` defaults to `SpecifyTargetNoise` | `:127-129` | implicit: `uncertainty_handling_level` is 2 whenever `specify_target_noise` (`vbmc.py:1016-1017`) | equivalent in effect |
| `NoiseSize > 0` | `:131-133` | none | minor obs. M1 |
| error if `~UncertaintyHandling && SpecifyTargetNoise` | `:135-137` | none | minor obs. M2 |
| warn if `SpecifyTargetNoise && ~isempty(NoiseSize)` | `:139-141` | none | minor obs. M2 |
| **if `UncertaintyHandling`:** `MaxFunEvals *= 1.5`, `TolStableCount *= 1.5`, `ActiveSampleGPUpdate = true`, `ActiveSampleVPUpdate = true`, `SearchAcqFcn = {@acqviqr_vbmc}`, each only when not user-supplied | `:144-163` | `Options.update_defaults` (`options.py:100-112`), gated on **`specify_target_noise`** | **F1** |
| Assemble `CMAESopts` | `:166-178` | `pyvbmc/vbmc/active_sample.py:541-578` | on the sheet (P2), ruled on |

`misc/evaloption_vbmc.m` against `Options.eval` (`options.py:306-328`): both return a non-callable value unchanged and call a callable with the supplied argument. PyVBMC passes the argument **by keyword** (`self.get(key)(**params)`), MATLAB positionally; every shipped default declares the matching parameter name (`K`, `N`), and the 18 Python call sites all pass `{"K": ...}`/`{"N": ...}` matching the 21 MATLAB `evaloption_vbmc` call sites for the ported options. Equivalent for shipped defaults; see minor obs. M7. `utils/evalbool.m` has no Python counterpart (the `.ini` values are Python literals); on the sheet.

### 1.4 `optimState` field table (`misc/setupvars_vbmc.m` against `_init_optim_state`)

| MATLAB `optimState` field (setupvars line) | Python `optim_state` key (`vbmc.py` line) | Same initial value / meaning? |
| --- | --- | --- |
| `integervars` (`:15-17`) | `integer_vars` (`:878-881`) | value agrees when unset; **meaning differs, F3** |
| `LB_orig`, `UB_orig`, `PLB_orig`, `PUB_orig` (`:26-29`) | `lb_orig`, `ub_orig`, `plb_orig`, `pub_orig` (`:895-898`) | yes (Python stores copies) |
| `LBeps_orig`, `UBeps_orig` (`:30-31`) | `lb_eps_orig`, `ub_eps_orig` (`:899-905`) | yes (both give `NaN` for an unbounded coordinate) |
| `LB`, `UB`, `PLB`, `PUB` (transformed, `:49-52`) | `lb_tran`, `ub_tran`, `plb_tran`, `pub_tran` (`:909-921`) | yes |
| `Cache.X_orig`, `Cache.y_orig` (`:55-63`) | `cache["x_orig"]`, `cache["y_orig"]` (`:869-870`) | yes, including the size-mismatch error |
| `trinfo` (`:86`) | none — the shared `self.parameter_transformer` | equivalent by construction |
| `iter = 0` (`:147`) | `iter = -1` (`:926`) | equivalent (0-based vs 1-based, documented at `:924-925`) |
| `sn2hpd = Inf` (`:150`) | `sn2_hpd = np.inf` (`:930`) | yes |
| `Cache.active` (`:153`) | `cache_active` (`:873-875`) | yes (Python also ORs in precomputed rows, `:726-728`) |
| `LastWarping = -Inf` (`:156`) | `last_warping = -np.inf` (`:933`) | yes |
| `LastSuccessfulWarping = -Inf` (`:159`) | `last_successful_warping = -np.inf` (`:937`) | yes |
| `WarpingCount = 0` (`:162`) | `warping_count = 0` (`:940`) | yes |
| `StopSampling` (`:165-169`) | `stop_sampling` (`:943-946`) | yes |
| `RecomputeVarPost = true` (`:172`) | `recompute_var_post = True` (`:949`) | yes |
| `Warmup`, `LastWarmup` (`:175-180`) | `warmup`, `last_warmup` (`:952-956`) | yes |
| `WarmupStableCount = 0` (`:183`) | `warmup_stable_count = 0` (`:960`) | yes |
| `ProposalFcn` (`:186-190`) | `proposal_fcn` (`:963-966`) | placeholder string; `proposal_vbmc` unported (map). Read nowhere on either side |
| `R = Inf` (`:193`) | `R = np.inf` (`:969`) | yes |
| `SkipActiveSampling = false` (`:196`) | `skip_active_sampling = False` (`:972`) | yes |
| `RunMean`, `RunCov`, `LastRunAvg` (`:199-201`) | `run_mean`, `run_cov`, `last_run_avg` (`:976-979`) | yes |
| `vpK = K` (`:204`) | `vp_K` (`:982`) | yes |
| `pruned = 0` (`:207`) | `pruned = 0` (`:985`) | yes |
| `EntropySwitch` + `nvars < DetEntropyMinD` override (`:210-214`) | `entropy_switch` (`:988-992`) | yes |
| `TolGPVar` (`:217`) | `tol_gp_var` (`:995`) | yes |
| `MaxFunEvals` (`:220`) | `max_fun_evals` (`:1001`) | yes when no budget is active |
| `VarianceRegularizedAcqFcn = true` (`:223`) | `variance_regularized_acq_fcn = True` (`:1009`) | yes |
| `SearchCache = []` (`:226`) | `search_cache = []` (`:1012`) | yes |
| `UncertaintyHandlingLevel` (`:230-236`) | `uncertainty_handling_level` (`:1016-1021`) | value agrees for the shipped defaults; **truth test differs, F4** |
| `hedge` when `AcqHedge` (`:239`) | `hedge` (`:1024-1025`) | yes (feature unported; sheet) |
| `iterList.{u,fval,fsd,fhyp}` (`:242-245`) | `iter_list{"u","f_val","f_sd","fhyp"}` (`:1028-1032`) | yes (name spelling differs) |
| `delta = Bandwidth*(PUB-PLB)` (`:247`) | **none** | `Bandwidth` unported; on the sheet (P6) |
| `temperature` + validation `T∈{1,2,3,4}` (`:250-255`) | **none** | **F6** |
| `entropy_alpha` (`:258`) | `entropy_alpha` (`:1035`) | yes |
| `vp_repo = []` (`:261`) | **not initialized** (commented out at `:1037-1038`; written at `:1663`) | feature unported (sheet); minor obs. M10 |
| `RepeatedObservationsStreak = 0` (`:264`) | `repeated_observations_streak = 0` (`:1041`) | yes |
| `DataTrimList = []` (`:267`) | `data_trim_list = []` (`:1044`) | yes |
| `LB_search`, `UB_search` (`:270-272`) | `lb_search`, `ub_search` (`:1046-1057`) | yes, same formula |
| `gpCovfun = 1` (`:276`) | `gp_cov_fun = 1` (`:1061`) | yes |
| `gpNoisefun` `[1 0]/[1 2]/[1 1]` (`:277-284`) | `gp_noise_fun` `[1,0,0]/[1,2,0]/[1,1,0]` (`:1063-1077`) | equivalent (`gplite_noisefun.m:20-22` pads the third slot with 0) |
| `gpMeanfun` + the 12-name validation (`:285-291`) | `gp_mean_fun` + the identical 12-name list (`:1079-1100`) | yes |
| `intMeanfun` (`:292`) | `int_mean_fun` (`:1101`) | yes (feature unported; sheet) |
| `gpOutwarpfun` + string decoding (`:293-300`) | **none** | output warping unported; on the sheet (G) |
| `OutwarpDelta` (`:303-307`) | `out_warp_delta` (`:1104-1110`) | equivalent (the `gpOutwarpfun` disjunct is vacuous in Python) |
| `DefaultWarnings.singularMatrix` (`:311-312`) | **none** | MATLAB-specific warning suppression; PyVBMC uses local `np.errstate` |

**Python `optim_state` keys with no MATLAB counterpart in setupvars:** `max_fun_evals_total`, `initialization_cost`, `budget_active` (Python-only budget feature, on the sheet). Keys created later in `optimize()` that MATLAB also creates later (`N`, `n_eff`, `lcb_max`, `lcb_max_vec`, `hyp_dict`, `redo_roto_scaling`, `active_importance_sampling`, `gp_length_scale`, `gp_mala_step_size`, `t_algo_per_fun_eval`, `var_log_joint_samples`, `cov_log_joint_components`) were not compared here (they belong to P1a/P2/P5). MATLAB's `optimState` fields written by `funlogger_vbmc` (`X`, `y`, `S`, `Xn`, `X_flag`, `nevals`, `ymax`, `Xmax`, `N`, `Neff`, `funccount`, `cachecount`, `X_orig`, `y_orig`, `funevaltime`) live on PyVBMC's `FunctionLogger` object (P8's slice).

### 1.5 `savestats` against what PyVBMC records

MATLAB `vbmc.m:1021-1053` plus `private/vbmc_termination.m:64-65`, `:95`. PyVBMC `vbmc.py:1602-1625`, `:1671-1673`, `:1775-1777`, `:1817-1823`, `:2139-2142`, `:2191`.

| MATLAB `stats` field | PyVBMC history key | Note |
| --- | --- | --- |
| `iter`, `N`, `elbo`, `elbo_sd`, `sKL`, `sKL_true`, `pruned`, `warmup`, `timer`, `vp` | `iter`, `N`, `elbo`, `elbo_sd`, `sKL`, `sKL_true`, `pruned`, `warmup`, `timer`, `vp` | same |
| `Neff` | `n_eff` | same |
| `funccount` | `func_count` | same |
| `gpSampleVar` | `var_ss` | same |
| `gpNsamples` | `Ns_gp` | same |
| `gpHypFull` | `gp_hyp_full` | same |
| `lcbmax` | `lcb_max` | same |
| `gp` (`gplite_clean(gp)`) | `gp` (`_lean_gp`) | deliberate; on the sheet |
| `rindex`, `elcbo_impro`, `stable` | `r_index`, `elcbo_impro`, `stable` | same |
| `cachecount` | — | available as `function_logger.cache_count` (the whole logger is recorded) |
| `vpK` | — | available as the recorded `vp.K` |
| `gpNoise_hpd` = `sqrt(sn2hpd)` | — | available as `optim_state["sn2_hpd"]` (recorded) |
| `outwarp_threshold` | — | available as `optim_state["out_warp_delta"]` (recorded); feature unported |
| `t` (per-iteration wall time) | — | the `timer` object is recorded instead |
| — | `optim_state`, `random_state`, `function_logger`, `logging_action` | Python-only; `optim_state` omits the importance samples (on the sheet) |
| — | `data_trim_list` | declared key, never recorded (minor obs. M9) |

### 1.6 `lpostfun.m` against `_init_log_joint`

`lpostfun.m:12-22` returns `y = llike(x)` and adds `lprior(x)` when a prior is given; with two outputs it takes `[y,s] = llike(x)` and leaves the prior noiseless. `VBMC._init_log_joint` (`vbmc.py:3336-3416`) builds exactly this: `log_likelihood(theta) + log_prior(theta)` at uncertainty level 0/1 (`:3410-3411`), and `ll + log_prior(theta), noise_est` at level 2 (`:3404-3406`). Equivalent. (The vectorized branch, `:3358-3400`, is the Python-only feature on the sheet.) MATLAB's `lpostfun` is a user-facing example rather than something `vbmc.m` calls; PyVBMC builds the joint internally, which the map records.

### 1.7 MATLAB code unreachable at the defaults, and not reported

- `misc/setupvars_vbmc.m:101-142` — the whole "Import prior function evaluations" block is commented out.
- `misc/setupvars_vbmc.m:294-300` — `gpOutwarpfun` string decoding: at `defopts.gpOutwarpFun = '[]'` the branch maps to `[]`.
- `misc/setupvars_vbmc.m:247` (`delta`) is all zeros at `Bandwidth = 0`, so every consumer reduces to the unsmoothed formula (already on the sheet).
- `vbmc.m:434-444` — the `WarmupOptions` copy is a no-op at `defopts.WarmupOptions = '[]'`.
- `vbmc.m:418-424`, `:447-450`, `:1082-1125` — `initFromVP` / `robustSampleFromVP` run only when `x0` is a variational posterior (map: unported).
- `misc/setupoptions_vbmc.m:160-162` — the `TolStableWarmup` doubling is commented out.
- `vbmc.m:496` — the `BOWarmup` header branch (option default off, unported).
- `misc/setupoptions_vbmc.m:98-106` — `OptimToolbox` is computed but read nowhere else in the repository (see sheet note S4).

### 1.8 Checks run

All scripts are in `…\scratchpad\port_review\P1b_comparison\`; the interpreter printed `C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc\__init__.py`.

1. `extract_opts.py` — extracted the 180 `defopts` names and the 183 `.ini` names; established the MATLAB-only / Python-only sets.
2. `defaults_table.py` / `emit_table.py` — evaluated every MATLAB default expression (transcribed to Python) and every `.ini` default for `D = 1,2,5,9,10,15,20,30`, and the function-valued options at `K,N = 1,2,3,7,10,50`. Result: the five rows that differ are `Diagnostics` (`'off'` vs `False`, same meaning), `ProposalFcn` (`[]` vs `None`, same meaning), `NonlinearScaling` (`'on'` vs `True`, same meaning), `SearchAcqFcn` (**F5**) and `BoundedTransform` (on the sheet). Everything else agrees at every tested `D`.
3. `unread_opts.py` — scanned the package for a genuine options read (`options[...]`, `options.get(...)`, `options.eval(...)`) of each declared name. 27 declared options are never read through an options object; 22 are registered in `INERT_OPTIONS`; the other three are `diagnostics` (**F10**), `entropy_force_switch` (**F2**) and `temperature` (**F6**).
4. `check1.py` — built three `VBMC` objects (default, `uncertainty_handling=[1]`, `specify_target_noise=True`) and printed the five options MATLAB overrides for noisy targets. Settles **F1**: the level-1 run keeps `max_fun_evals=250`, `tol_stable_count=60`, both `active_sample_*_update=False` and `search_acq_fcn=[AcqFcnLog()]`, while the level-2 run gets 375, 90, `True`, `True`, `[AcqFcnVIQR()]`.
5. `check2.py` — settles **F2** (`optim_state.get("entropy_force_switch")` is `None`, and `None * 250` raises `TypeError`), **F7** (scalar bounds raise `ValueError: ('Bounds must match problem dimension D=%d.', 5)` for `D = 5` and are accepted only for `D = 1`), and the start of **F3**.
6. `check3.py`, `check4.py` — settle **F3**: `integer_vars=[0,2]` (a list) marks *every* variable integer; `np.array([0,2])` raises `IndexError`; only a length-`D` mask (`np.array([True,False,True])` or `np.array([1,0,1])`) behaves as intended.
7. `check5.py` — settles **F4**: `uncertainty_handling` set to `True`, `1`, `False` or `None` raises `TypeError: object of type ... has no len()`, while `[0]` and `[False]` select uncertainty level 1.

No test suite, no `optimize()` run, nothing installed; the longest script ran for a few seconds.

### 1.9 Compared and found equivalent (stated so that silence is informative)

- **`boundscheck_vbmc.m` against `_bounds.py`, line by line.** The plausible-bound estimate from a multi-row `X0` (`:17-25` ↔ `_bounds.py:102-112`), the fallback to hard bounds for `N0 == 1` (`:35-39` ↔ `:123-134`), the finite-plausible-bounds test (`:54-56` ↔ `:160-165`), the real-valued test (`:59-61` ↔ `:68-77`), the fixed-variable test (`:64-68` ↔ `:180-190`), the distinct-plausible-bounds test (`:71-74` ↔ `:193-198`), the `X0`-inside-hard-bounds error (`:77-80` ↔ `:201-206`), the *entire* effective bounds computation including `realmin` and the infinity carry-over (`:83-96` ↔ `_effective_bounds`, `_bounds.py:9-38`), the clamp of `X0` inside the effective bounds (`:99-103` ↔ `:213-219`), the permissive ordering test (`:106-110` ↔ `:222-232`), the plausible-bound repair (`:113-119` ↔ `:235-248`), the expansion of the plausible box around `X0` (`:122-127` ↔ `:252-261`), the strict ordering test (`:130-134` ↔ `:264-274`) and the half-bounded test (`:139-143` ↔ `:278-286`) all agree, in the same order, with the same errors and warnings. Two comparisons use `<`/`>` where MATLAB uses `<=`/`>=`, but only inside a warning condition whose repair is then a no-op (minor obs. M6). `NaN` starting points pass through both unchanged, so the "midpoint of the plausible box" replacement fires identically.
- **`x0` handling.** Missing `x0` with missing plausible bounds raises on both sides (`vbmc.m:407-410` ↔ `vbmc.py:282-290`); otherwise `x0` becomes all-`NaN` with the shape of `PLB` (`vbmc.m:411` ↔ `vbmc.py:292`), and the midpoint replacement fires when *any* entry is non-finite (`setupvars_vbmc.m:7-12` ↔ `vbmc.py:391-396`), after the bounds check on both sides. Several rows are kept as the starting cache on both sides; a row on a hard bound is moved inside; a row outside the plausible box expands the box.
- **The variational posterior's initial state** (`setupvars_vbmc.m:78-99` ↔ `VariationalPosterior.__init__`, `pyvbmc/variational_posterior/variational_posterior.py:120-164`): `w = 1/K`, `sigma = 1e-3`, `lambda = 1`, and `mu` built by tiling the rows of `x0` to `K` columns and adding `1e-6 * randn(D,K)` — the tiling order matches MATLAB's `repmat`/`(1:K,:)'` for both the one-row and the many-row case. `optimize_sigma`/`optimize_lambda` are `True` on both sides. MATLAB sets `optimize_weights = false` during warm-up in setupvars; PyVBMC leaves it `True` at construction and `optimize_vp` sets it `False` while `optim_state["warmup"]` is on (`variational_optimization.py:151-152`), before any reader; the post-warm-up assignment from `variable_means`/`variable_weights` matches (`vbmc.py:418-420`, `:1651-1655` ↔ `setupvars_vbmc.m:89-95`, `vbmc.m` warm-up exit).
- **The expanding search bounds** `LB_search`/`UB_search` (`setupvars_vbmc.m:270-272` ↔ `vbmc.py:1046-1057`): same formula, same `max`/`min` clamps, same `ActiveSearchBound = 2`.
- **`Options.eval` against `evaloption_vbmc.m`**, and all 18 Python call sites against the 21 MATLAB ones for the ported options.
- **`save`/`load`/`rng.py` against the sheet.** `_get_random_state` (`vbmc.py:3056-3059`) returns `{"generator": deepcopy(bit-generator state)}`; `_create_result_dict` puts a fresh snapshot in `results["rng_state"]` (`:3160`) after the post-loop draws, where MATLAB's `private/vbmc_output.m:22` stores the global `rng`; `_set_random_state` (`:3093-3117`) takes the `"generator"` entry and ignores any legacy companion entry. `get_rng` (`pyvbmc/rng.py:23-25`) derives the generator from NumPy's global legacy state when `seed is None`. All as the sheet describes. MATLAB has no save/load. `results["overhead"] = np.nan` (`vbmc.py:3159`) against MATLAB's `vbmc.m:939`, which does fill it — the sheet's entry is accurate (note that `private/vbmc_output.m:21` alone would have suggested otherwise).
- **`IterationHistory`**: keys are fixed at construction and `record` refuses an unknown one (`iteration_history.py:43-49`, `:100-104`); every stored value is deep-copied (`:49`, `:110`); `_expand_array` (`:112-127`) grows the object array without re-copying. MATLAB grows `stats.<field>(iter)` in place with no copying; PyVBMC's copying is what makes the recorded `vp`/`gp`/logger snapshots independent, and is the documented Python convention.
- **`pyvbmc/__init__.py`**: no MATLAB counterpart; the lazy `SVBMC`/`PyMCTarget` resolution matches `AGENTS.md`. `VBMC.__init__` imports `pyvbmc.pymc._target`, which does not import PyMC at module level (the default virtual environment, which has no PyMC, constructs `VBMC` fine).

### 1.10 MATLAB history checked

- `misc/boundscheck_vbmc.m`: last touched `969c4ff` (2019-12-04) — unchanged since before the port.
- `misc/setupvars_vbmc.m`: last touched `f3e5d76` (2022-07-06) and `74b046e` (2022-06-25), both about the bounded transform (already tracked on the sheet and in the map). The integer-variable block (`:14-24`) and the temperature block (`:249-256`) both last changed in `28f4fee` (2019-07-21), before the port.
- `misc/setupoptions_vbmc.m`: `a9615ba` (2022-07-23), `2044530` (2021-06-18), `f3f8b80` (2021-03-13, adding `skipextra_flag` and the `isfield` guards), `a5240d2` (2021-02-02). The noisy-override block (`:143-163`) last changed in `2a076c9` (2020-06-16), before the port.
- `vbmc.m:213` (`SearchAcqFcn`): last changed in `0b51bd0` (2020-02-29), which set it to `@acqf_vbmc`; unchanged since, i.e. before the port.
- `vbmc.m:523-528` (the forced entropy switch): last changed `cc56c7c` (2018-08-05).
- `misc/evaloption_vbmc.m`: `fc19a5a` (2019-11-24). `utils/evalbool.m`: `c4bde11` (2018-10-02). `lpostfun.m`: `2a076c9` (2020-06-16).

So every MATLAB line behind a finding below predates PyVBMC's first commit (2021-01-19): none of these are cases of MATLAB moving after the port.

---

## 2. Findings

### F1. The noisy-target option overrides fire only for `specify_target_noise`, not for `uncertainty_handling`
- Location: `pyvbmc/vbmc/options.py:100-112` (`Options.update_defaults`), called from `pyvbmc/vbmc/vbmc.py:311`; MATLAB: `misc/setupoptions_vbmc.m:144-163`
- Category: defaults (with a control-flow consequence)
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): the MATLAB block last changed in `2a076c9` (2020-06-16), before PyVBMC's first commit. PyVBMC's `update_defaults` was introduced already gated on `specify_target_noise` by `70325a3` (2022-06-02, PR #80 "Noisy likelihoods"), so the two never agreed for level-1 runs.
- What the code does, what it should do, and why. MATLAB applies five overrides when `options.UncertaintyHandling` is truthy — `MaxFunEvals = ceil(1.5*…)`, `TolStableCount = ceil(1.5*…)`, `ActiveSampleGPUpdate = true`, `ActiveSampleVPUpdate = true`, `SearchAcqFcn = {@acqviqr_vbmc}` — each only when the user did not supply that field. `UncertaintyHandling` is truthy for **both** noisy modes: for user-provided noise (level 2, because `setupoptions_vbmc.m:127-129` copies `SpecifyTargetNoise` into it when it is empty) and for inferred noise (level 1, set by the user). PyVBMC's `update_defaults` tests `self.get("specify_target_noise")` only, so a run configured as `options={"uncertainty_handling": [...]}` — the documented way to get uncertainty level 1 (`vbmc.py:1018-1019`, and `pyvbmc/testing/vbmc/test_vbmc_init.py:474`) — gets none of the five.
- Consequence if real: an inferred-noise run has two thirds of MATLAB's evaluation budget (`50*(2+D)` instead of `ceil(75*(2+D))`), a stability window of 60 instead of 90 function evaluations, no GP or variational update after each active sample, and the noiseless prospective-uncertainty acquisition instead of VIQR. On a noisy target VIQR versus the noiseless acquisition is the difference the 2020 noisy-VBMC paper is about, so the effect is large for that class of problem. It fires whenever a user selects uncertainty level 1 and does not set the five options by hand; it does not fire at the shipped defaults (level 0) nor for level 2.
- Suggested reproduction: already run (`check1.py`): three `VBMC` constructions, printing the five options. Output — default: `250 / 60 / False / False / [AcqFcnLog]`; `uncertainty_handling=[1]`: `250 / 60 / False / False / [AcqFcnLog]` (level 1); `specify_target_noise=True`: `375 / 90 / True / True / [AcqFcnVIQR]` (level 2).
- Test adequacy: no. `pyvbmc/testing/vbmc/test_options.py:117-161` (`test_init_with_specify_target_noise`) exercises only `specify_target_noise=True` and asserts exactly what the implementation does; it never constructs a level-1 run. `test_vbmc_init.py:474` and `:608` build level-1 runs but assert only the resulting level.

### F2. `optim_state["entropy_force_switch"]` is never set, so the forced entropy switch raises `TypeError`
- Location: `pyvbmc/vbmc/vbmc.py:1224-1230` (read), `:988-992` (`_init_optim_state`, where the key is not written); option declared at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:211`; MATLAB: `vbmc.m:524-528`, default `vbmc.m:295`
- Category: state/caching (with a control-flow consequence)
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): the MATLAB lines last changed in `cc56c7c` (2018-08-05), long before the port.
- What the code does, what it should do, and why. MATLAB reads the *option*: `if optimState.EntropySwitch && optimState.funccount >= options.EntropyForceSwitch*options.MaxFunEvals` (`vbmc.m:524-525`). PyVBMC reads `self.optim_state.get("entropy_force_switch")` (`vbmc.py:1226`), but `_init_optim_state` copies only `entropy_switch` into `optim_state` (`:988`); no module ever writes `optim_state["entropy_force_switch"]` (grep over the package finds a single occurrence, the read itself). The key is therefore always absent and `.get` returns `None`, so the right operand of `and` evaluates `None * self.optim_state.get("max_fun_evals")`. Python's `and` short-circuits, so this is reached exactly when `optim_state["entropy_switch"]` is true. The correct read is `self.options["entropy_force_switch"]`.
- Consequence if real: any run with `options={"entropy_switch": True}` and `D >= det_entropy_min_d` (5) aborts with `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` on the first iteration of `optimize()`. The deterministic-entropy mode is unreachable for `D >= 5`, which is precisely the regime the option exists for (`DetEntropyMinD = 5`). At the shipped defaults (`entropy_switch = False`) nothing fires, and `entropy_switch` is also forced off for `D < 5` (`vbmc.py:991-992`), which is why the defect has survived. Beyond the crash, the intended behavior — force the switch to the stochastic entropy once `funccount >= 0.8 * MaxFunEvals` — never happens.
- Suggested reproduction: already run (`check2.py`), without calling `optimize()`: construct `VBMC(..., D=5, options={"entropy_switch": True})`; `optim_state["entropy_switch"]` is `True`, `optim_state.get("entropy_force_switch")` is `None`, and the product raises `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`. A full confirmation is one short `optimize()` run with those options.
- Test adequacy: no. `pyvbmc/testing/vbmc/test_vbmc_init.py:572-582` (`test_vbmc_optimstate_entropy_switch`) checks only that `optim_state["entropy_switch"]` is set correctly; nothing constructs a run that reaches `vbmc.py:1224`. The end-to-end runs in `test_vbmc_optimize.py` all use default options.

### F3. `integer_vars` is read as a length-`D` mask where MATLAB reads a list of indices; a Python list silently marks every variable integer
- Location: `pyvbmc/vbmc/vbmc.py:877-892` (`_init_optim_state`); option declared at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:4-5` ("Array with indices of integer variables"); MATLAB: `misc/setupvars_vbmc.m:14-24`
- Category: indexing/shape (and defaults, since the documented value form is unusable)
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): the MATLAB block last changed in `28f4fee` (2019-07-21), before the port.
- What the code does, what it should do, and why. MATLAB: `optimState.integervars = false(1,nvars); optimState.integervars(options.IntegerVars) = true;` — `IntegerVars` holds 1-based **indices** (the help text and the `defopts` comment both say so; `[1 3]` marks variables 1 and 3). PyVBMC:
  ```
  optim_state["integer_vars"] = np.full(self.D, False)
  if len(self.options.get("integer_vars")) > 0:
      integeridx = self.options.get("integer_vars") != 0
      optim_state["integer_vars"][integeridx] = True
  ```
  `integeridx` is a *mask*, not indices, so the option is read as a length-`D` indicator vector. Worse, when the value is a plain Python `list` — which is the type of the declared default `[]` and the natural thing for a user to pass — `list != 0` is not elementwise: it is the scalar `True`. Indexing a NumPy array with a 0-dimensional boolean `True` selects the whole array, so `optim_state["integer_vars"][True] = True` marks **every** variable as an integer variable, and `self.lower_bounds[:, True]` then checks every coordinate's bounds. There is no error and no warning.
- Consequence if real: with `options={"integer_vars": [0, 2]}` at `D = 3` and half-integer bounds throughout, all three variables are treated as integer variables: `AbstractAcqFcn._real2int` (`pyvbmc/acquisition_functions/abstract_acq_fcn.py:261`) rounds every coordinate of every acquisition candidate to the integer grid, so the search can never propose a non-integer value in the continuous coordinates and the posterior is fitted to a lattice. With bounds that are not all at half integers the run instead raises a misleading "Hard bounds of integer variables need to be set at +/- 0.5 points" error naming coordinates the user never declared. `np.array([0, 2])` raises `IndexError` (mask length 2 against `D = 3`). Only a length-`D` NumPy array (`np.array([True, False, True])` or `np.array([1, 0, 1])`) does what the user means. The feature is off by default (`integer_vars = []`), so nothing fires at the shipped defaults.
- Suggested reproduction: already run (`check3.py`, `check4.py`). With `D = 3`, `lb = -0.5`, `ub = 10.5` in every coordinate and `options={"integer_vars": [0, 2]}`, `optim_state["integer_vars"]` comes back `[True True True]` where MATLAB's `IntegerVars = [1 3]` gives `[True False True]`. `np.array([0,2])` → `IndexError: boolean index did not match indexed array along axis 0; size of axis is 3 but size of corresponding boolean dimension is 2`.
- Test adequacy: no. `pyvbmc/testing/vbmc/test_vbmc_init.py:397-424` (`test_vbmc_optimstate_integer_vars`) passes `np.array([1, 0, 0])` and asserts the mask interpretation, i.e. it encodes the implementation. Under MATLAB's convention `[1 0 0]` is not even a legal index vector (index 0). The test never passes a list and never passes a genuine index vector.

### F4. `uncertainty_handling` is tested by length, not by truth value
- Location: `pyvbmc/vbmc/vbmc.py:1016-1021`; option declared at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:2-3` ("Explicit noise handling (0: none; 1: unknown noise level; 2: user-provided noise)"); MATLAB: `misc/setupvars_vbmc.m:230-236` with `misc/setupoptions_vbmc.m:127-129` and `utils/evalbool.m`
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): `setupvars_vbmc.m:230-236` and `evalbool.m` both predate the port (`evalbool.m` last touched `c4bde11`, 2018-10-02).
- What the code does, what it should do, and why. MATLAB's `UncertaintyHandling` is a boolean flag: `'[]'` means "decide from `SpecifyTargetNoise`", and `evalbool` maps `'yes'/'on'/'true'/'1 '` to 1 and `'no'/'off'/'false'/'0 '` to 0; `setupvars_vbmc.m:232` then tests `elseif options.UncertaintyHandling` — a truth test. PyVBMC tests `elif len(self.options.get("uncertainty_handling")) > 0` — a *length* test on a container. Consequences: `uncertainty_handling=True`, `=1`, `=False` and `=None` all raise `TypeError: object of type 'bool' has no len()` (etc.) during construction, so neither the boolean form MATLAB accepts nor the numeric form the option's own description advertises (`0`, `1`, `2`) can be used; and `[0]`, `[False]`, `[0.0]` — the natural ways to write "off" in the container form — all select uncertainty level 1, i.e. noisy inference.
- Consequence if real: a user following the option's documentation (`uncertainty_handling: 1`) gets a `TypeError` at construction. A user writing `uncertainty_handling: [0]` intending "no noise handling" silently gets an inferred-noise GP (`gp_noise_fun = [1, 2, 0]`, `vbmc.py:1066-1068`), a noisy `FunctionLogger`, and — combined with F1 — none of MATLAB's noisy budget adjustments. The only working spelling is "any non-empty container", e.g. `[1]` or the string `'yes'` (which works by accident, because a non-empty string has positive length). At the shipped default `[]` the behavior is correct.
- Suggested reproduction: already run (`check5.py`). Output: `True → TypeError`, `1 → TypeError`, `'yes' → level 1`, `[] → level 0`, `[0] → level 1`, `[False] → level 1`, `[1] → level 1`, `np.array([]) → level 0`, `False → TypeError`, `None → TypeError`.
- Test adequacy: no. `pyvbmc/testing/vbmc/test_vbmc_init.py:468-484` and `:604-613` pass `[3]` and `[]` only, so they pin the container-length behavior rather than the specification; no test passes a boolean or a numeric level.

### F5. The default search acquisition function is `AcqFcnLog`, where MATLAB's is `acqf_vbmc`
- Location: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:39` (`search_acq_fcn = [AcqFcnLog()]`); MATLAB: `vbmc.m:213` (`defopts.SearchAcqFcn = '@acqf_vbmc'`)
- Category: defaults
- Proposed classification: possibly intentional (and missing from the known-differences sheet)
- Confidence: high that the defaults differ; high that the change was deliberate
- History (comparison track): MATLAB set this default to `@acqf_vbmc` in `0b51bd0` (2020-02-29) and has not changed it since — so it did not move after the port. PyVBMC matched it (`searchacqfcn = ["@acqf_vbmc"]`, then `[AcqFcn()]`) until commit `e20d081` (2022-09-21, PR #102 "Acq fcn log"), whose message reads "refactor: Minor changes to AcqFcnLog, for numerical stability. / refactor: Default to AcqFcnLog instead of AcqFcn."
- What the code does, what it should do, and why. `acq/acqf_vbmc.m:11` computes `acq = -vtot .* exp(fbar-z) .* p`; `acq/acqflog_vbmc.m:18` computes `acq = -(log(vtot) + fbar - z + log(p))`. Writing `u = vtot*exp(fbar-z)*p > 0`, the two are `-u` and `-log u`, monotone transformations of each other, so in exact arithmetic they have the same minimizer; the variance regularization is applied consistently in each parameterization (`acq/acqwrapper_vbmc.m:40-44`, mirrored at `pyvbmc/acquisition_functions/abstract_acq_fcn.py:166-175`). The difference is numerical: `exp(fbar-z)` underflows to 0 far from the mode, so `acqf` returns exactly 0 over large regions and the search cannot rank candidates there, while the log form keeps full resolution. PyVBMC's default is therefore a deliberate, defensible improvement — but it is a default that differs from MATLAB's, and the sheet does not record it.
- Consequence if real: every default PyVBMC run selects its active-sampling points with a different (though order-equivalent) acquisition function from every default MATLAB run. In regions where `acqf` underflows the two make genuinely different choices, so trajectories diverge; elsewhere the choices agree up to rounding. It fires on every run at the default options.
- Suggested reproduction: read `advanced_vbmc_options.ini:39` against `vbmc.m:213`; already done by the defaults comparison (`defaults_table.py`, row `SearchAcqFcn`). A numerical demonstration would evaluate both acquisitions on a stored oracle fixture and compare the argmin and the number of exact ties (not run: the acquisition numerics are P3's).
- Test adequacy: `pyvbmc/testing/vbmc/test_options.py:158-159` asserts that the default is an `AcqFcnLog`, so the current default is pinned — as an implementation fact, not as agreement with MATLAB. Nothing would flag the divergence from `defopts.SearchAcqFcn`.

### F6. `optim_state["temperature"]` is never set, so the `temperature` option reaches no reader and MATLAB's validation is absent
- Location: `pyvbmc/vbmc/vbmc.py:851-1112` (`_init_optim_state`, which never writes the key); readers `pyvbmc/whitening/whitening.py:196-199`, `:291-294`; option declared at `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:257`; MATLAB: `misc/setupvars_vbmc.m:249-255`
- Category: state/caching
- Proposed classification: port discrepancy (the *feature* is unported and on the sheet; the dead `optim_state` read and the missing validation are not)
- Confidence: high
- History (comparison track): the MATLAB block last changed in `28f4fee` (2019-07-21), before the port.
- What the code does, what it should do, and why. MATLAB validates the option (`if round(T) ~= T || T > 4 || T < 1` → `error('vbmc:PosterioTemperature')`) and stores `optimState.temperature = T`. PyVBMC copies neither the validation nor the value; the two places written to honor a temperature read `optim_state.get("temperature")`, a key nothing ever sets, and fall back to `T = 1`.
- Consequence if real: none at the default `temperature = 1`, which is the only value PyVBMC supports (posterior tempering is unported; sheet, P7). The practical effects are that `options={"temperature": 2}` is accepted silently and has no effect anywhere (MATLAB would accept it and temper the whole run), and that a value outside `{1,2,3,4}` is not rejected. It also means the sheet's statement that the option "is read only by `whitening.py:196`, `:291`" is not accurate (see sheet note S1).
- Suggested reproduction: already run (`check1.py`): `"temperature" in v.optim_state` is `False` for every construction; and `unread_opts.py` reports `temperature` among the options with no options-object read.
- Test adequacy: no test constructs a run with a non-default `temperature`, and `test_options.py`'s inert-option guard does not cover it (see F10).

### F7. Scalar hard and plausible bounds are rejected, although the class docstring (following MATLAB) says they are replicated
- Location: `pyvbmc/vbmc/_bounds.py:136-157` (via `pyvbmc/vbmc/vbmc.py:831-849`); docstring claim at `pyvbmc/vbmc/vbmc.py:102-103` ("If scalars, the bound is replicated in each dimension"); MATLAB: `misc/boundscheck_vbmc.m:6-10`
- Category: indexing/shape
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): `misc/boundscheck_vbmc.m` last changed in `969c4ff` (2019-12-04), before the port; the scalar expansion has been there throughout.
- What the code does, what it should do, and why. MATLAB expands a scalar bound to a full row (`if isscalar(LB); LB = LB*ones(1,nvars); end`, and likewise for `UB`, `PLB`, `PUB`), which the `vbmc.m` help documents at lines 15-16. `_normalize_bounds` has no such step: it goes straight to `np.atleast_1d(...)` and `reshape((1, D))`, which succeeds for `D = 1` and raises `ValueError("Bounds must match problem dimension D=%d.", D)` for any other `D`. `VBMC`'s own docstring repeats MATLAB's promise.
- Consequence if real: `VBMC(f, x0, -10, 10, plb, pub)` — the form the docstring advertises and the form a MATLAB user would transcribe — raises at construction for `D > 1`. It never produces a wrong number; it refuses a documented input. It fires at the default options whenever a caller uses scalar bounds.
- Suggested reproduction: already run (`check2.py`): scalar `lb=-10.0`, `ub=10.0` at `D = 5` raises `ValueError: ('Bounds must match problem dimension D=%d.', 5)`; the same call at `D = 1` succeeds and gives `lower_bounds = [[-10.]]`.
- Test adequacy: no. `pyvbmc/testing/vbmc/test_vbmc_init.py` builds bounds as `(1, D)` arrays throughout; no test passes a scalar bound.

### F8. MATLAB's `MaxFunEvals`/`MaxIter` validation and the `MaxIter < MinIter` correction have no Python counterpart
- Location: `pyvbmc/vbmc/vbmc.py:297-333` and `pyvbmc/vbmc/options.py:100-112` (nothing validates these); MATLAB: `misc/setupoptions_vbmc.m:109-124`
- Category: defaults
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): the MATLAB checks predate the port (`setupoptions_vbmc.m` gained its current shape in `2a076c9`, 2020-06-16; `f3f8b80`, 2021-03-13, moved them behind `skipextra_flag` without changing them).
- What the code does, what it should do, and why. MATLAB errors when `MaxFunEvals` or `MaxIter` is not a positive integer, and raises `MaxIter` to `MinIter` with a warning when the user asked for fewer iterations than the minimum. PyVBMC performs neither check: a non-integer or non-positive `max_iter`/`max_fun_evals` is accepted, and `max_iter < min_iter` is left as given. Because PyVBMC does keep MATLAB's "prevent early termination" rule (`vbmc.py:2193-2201` ↔ `private/vbmc_termination.m:97-101`), the low `max_iter` is overridden at run time instead of at setup time, so the run still reaches `min_iter` — the end state is nearly the same, but the user gets no warning that their `max_iter` was ignored, and `max_iter = 0` or `2.5` passes silently.
- Consequence if real: small. `max_fun_evals = 0` yields a run that terminates immediately (the min-evaluation rule then keeps it alive until `min_fun_evals`); a fractional `max_iter` compares fine and just makes the stopping point non-obvious; `max_iter < min_iter` produces a silent override where MATLAB warns. Never fires at the defaults (`50*(2+D) >> D` and `50*(2+D) >> 5*D`).
- Suggested reproduction: construct `VBMC(..., options={"max_iter": 0})` and observe no error and no warning; compare with `setupoptions_vbmc.m:112-114`, which errors. Not run (it needs an `optimize()` call to show the run-time consequence).
- Test adequacy: no test supplies an invalid `max_iter`/`max_fun_evals`.

### F9. `min_iter` is compared against the 0-based iteration index, so PyVBMC runs one iteration more than MATLAB before it may stop
- Location: `pyvbmc/vbmc/vbmc.py:2194-2196` (with `self.optim_state["iter"] = self.iteration`, `:1213`, 0-based from `self.iteration = -1`, `:429`); option `min_iter = D` (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:155`); MATLAB: `private/vbmc_termination.m:98-99`, with `optimState.iter = iter` 1-based at `vbmc.m:515-516`; default `vbmc.m:267`
- Category: control flow
- Proposed classification: port discrepancy
- Confidence: high
- History (comparison track): `private/vbmc_termination.m:97-101` is unchanged since well before the port (the file's guard block has not moved in the window checked).
- What the code does, what it should do, and why. MATLAB forbids termination while `optimState.iter < options.MinIter`, where `iter` is the 1-based index and hence also the number of completed iterations. PyVBMC writes `iteration < self.options.get("min_iter")` with the 0-based `iteration`, so for the same number `n` of completed iterations it tests `n-1 < min_iter` instead of `n < min_iter`, requiring `min_iter + 1` iterations. Note that the *same function* converts correctly for the other two iteration-count comparisons: `iteration + 1 >= self.options.get("max_iter")` (`:2116`, matching `iter >= options.MaxIter` at `private/vbmc_termination.m:16`) and `iteration + 1 >= tol_stable_iters` (`:2148`). The `min_fun_evals` half of the same condition is a count on both sides and agrees.
- Consequence if real: a run that would otherwise stop early performs one extra iteration, i.e. `fun_evals_per_iter = 5` extra target evaluations, and returns a posterior from one iteration later. It bites only when the `min_iter` rule is the binding constraint, that is when the stability criterion is met before iteration `min_iter`; with `tol_stable_iters = ceil(60/5) = 12` at the defaults, that means `D >= 12`. It fires at the default options for such `D`. (This line sits in `_check_termination_conditions`, which belongs to slice P1a; it is reported here because it is the semantics of this slice's `min_iter` default. The P1a reviewer may have found it independently.)
- Suggested reproduction: a short seeded run at `D = 15` with `options={"min_iter": 3, "max_iter": 50}` and a target that converges immediately, counting the iterations performed; or simply read the two lines together. Not run (it needs an `optimize()` call).
- Test adequacy: no. `pyvbmc/testing/vbmc/test_vbmc_optimize.py`'s runs use default `min_iter = D` with small `D`, where the stability criterion binds first.

### F10. `diagnostics` is a declared option nothing reads, is not registered in `INERT_OPTIONS`, and the guard test cannot see that
- Location: `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:26-27`; `pyvbmc/vbmc/options.py:21-46` (`INERT_OPTIONS`, which omits it); `pyvbmc/testing/vbmc/test_options.py:183-205`; MATLAB: `vbmc.m:206`
- Category: defaults
- Proposed classification: port discrepancy (in the inert-option bookkeeping, not in the algorithm)
- Confidence: high
- History (comparison track): `defopts.Diagnostics` is declared at `vbmc.m:206` and read nowhere in the MATLAB repository either (the `sampleopts.Diagnostics` hits are the slice/ensemble samplers' own option), so the option is dead on both sides; only the Python registration is at issue.
- What the code does, what it should do, and why. The sheet's entry "Declared options that nothing reads" states that `INERT_OPTIONS` holds exactly the declared options no module reads, and that a test recomputes the set so a newly dead option fails it. Scanning the package for a genuine options read (`options[...]`, `options.get(...)`, `options.eval(...)`) finds **27** such options, not 22: besides the registered 22, `diagnostics`, `entropy_force_switch` (F2) and `temperature` (F6) are read nowhere. `test_inert_options_are_the_declared_options_nothing_reads` does not catch them because it looks for the *quoted name anywhere in the package text*: `"diagnostics"` matches `sampler_opts["diagnostics"] = False` in `pyvbmc/vbmc/active_importance_sampling.py:437`, `"entropy_force_switch"` matches its own dead read at `vbmc.py:1226`, and `"temperature"` matches the dead `optim_state` reads in `whitening.py`.
- Consequence if real: `options={"diagnostics": True}` is accepted with no effect and no warning, where the twenty-two registered names warn. The guard test's contract ("a newly dead option fails here") does not hold, which is how F2 and F6 stayed invisible. No numerical effect.
- Suggested reproduction: already run (`unread_opts.py`), which lists the 27 options with no options-object read and names the coincidental string match for each of the three unregistered ones.
- Test adequacy: the guard test exists but its matching rule is too loose; see §3.

### Minor observations

- **M1.** `misc/setupoptions_vbmc.m:131-133` errors when `NoiseSize` is supplied non-positive; PyVBMC has no such check. Harmless in practice: `pyvbmc/vbmc/gaussian_process_train.py:342-350` clamps with `max(options["noise_size"], min_noise)` where `min_noise = tol_gp_noise`.
- **M2.** `misc/setupoptions_vbmc.m:135-141` errors when `SpecifyTargetNoise` is set while `UncertaintyHandling` is explicitly off, and warns when both `SpecifyTargetNoise` and `NoiseSize` are given. PyVBMC has neither; its level is derived from `specify_target_noise` first (`vbmc.py:1016-1017`), so the contradictory configuration cannot arise, and `noise_size` is simply ignored at level 2.
- **M3.** `misc/setupoptions_vbmc.m:123` reads `options.MinFunEvals = options.MinFunEvals;` inside the `MaxFunEvals < MinFunEvals` branch — a self-assignment, so the warning fires and the value it announces ("Changing the value of OPTIONS.MaxFunEvals") is never applied. A MATLAB-side defect with no Python counterpart to inherit it.
- **M4.** `misc/setupoptions_vbmc.m:47` contains `'ConstrainedGPMean''FeatureTest'`, one string rather than two, so neither `ConstrainedGPMean` nor `FeatureTest` is in `evalfields` and both remain the character vector `'no'` after setup. Neither is read, so nothing follows; it is the reason `FeatureTest` is not merely unread but unevaluated.
- **M5.** `misc/boundscheck_vbmc.m:27` computes `idx = any(PLB == PUB)`, which reduces a row vector to a scalar logical; `PLB(idx) = LB(idx)` then repairs only the *first* coordinate. `pyvbmc/vbmc/_bounds.py:114-117` compares elementwise and repairs every degenerate coordinate. A MATLAB-side indexing defect that PyVBMC does not share; it can fire only when the plausible bounds are estimated from a multi-row `X0` with a zero-width coordinate.
- **M6.** Two bound-repair *warnings* differ in strictness. `misc/boundscheck_vbmc.m:99` tests `x0 <= LB_eff || x0 >= UB_eff` where `_bounds.py:213` tests `x0 < lower_effective or x0 > upper_effective`; and `misc/boundscheck_vbmc.m:113` tests `~(LB_eff < PLB & PUB < UB_eff)` where `_bounds.py:235` tests `lower_effective > plausible_lower_bounds or plausible_upper_bounds > upper_effective`. In both cases the difference is the exact-equality boundary, where the repair MATLAB then performs is a no-op, so only the warning differs.
- **M7.** `Options.eval` calls a callable option with keyword arguments (`options.py:326`), MATLAB's `evaloption_vbmc` positionally. All shipped defaults use the expected parameter name; a user-supplied callable whose parameter is named otherwise raises `TypeError` where MATLAB would work.
- **M8.** The accepted `Display`/`display` values differ: MATLAB `iter/notify/final/off` (`vbmc.m:388-399`), PyVBMC `off/iter/full` (`vbmc.py:3274-3279`). Both default to the iteration level and both fall back to it for an unrecognized value. Python-only interface choice, not on the sheet.
- **M9.** `data_trim_list` is declared as an `IterationHistory` key (`vbmc.py:473`) but never recorded; the list lives only in `optim_state` (`:1044`, `:2059-2061`), which is itself recorded. MATLAB likewise keeps `DataTrimList` in `optimState` only. Dead declaration, no behavioral effect.
- **M10.** `optim_state["vp_repo"]` is written at `vbmc.py:1663` but never initialized (`:1037-1038` keeps the initialization commented out) and never read; MATLAB initializes it at `setupvars_vbmc.m:261` and reads it in `misc/vpsieve_vbmc.m`, which is unported (sheet, P6).
- **M11.** `trinfo.x0_orig` (`misc/setupvars_vbmc.m:45`) is the only occurrence of that field in the MATLAB repository: it is written and never read. PyVBMC's `ParameterTransformer` correctly has no counterpart.
- **M12.** PyVBMC's integer-variable bound check (`vbmc.py:882-892`) rejects infinite bounds *and* bounds that are not at half integers. MATLAB's (`misc/setupvars_vbmc.m:19-22`) reads `(~isfinite(LB(d)) && floor(LB(d)) ~= 0.5) || ...`; since `floor` of any real is an integer, `floor(x) ~= 0.5` is always true, so the condition reduces to "the bound is not finite" and the half-integer requirement its own error message states is never checked. PyVBMC implements the documented intent and is strictly stricter — a MATLAB-side defect the port did not inherit.

---

## 3. Test adequacy notes

Tests of this slice that mirror the implementation rather than the specification:

- `pyvbmc/testing/vbmc/test_vbmc_init.py:397-424` (`test_vbmc_optimstate_integer_vars`) passes `np.array([1, 0, 0])` and asserts the resulting mask. That value is not a legal `IntegerVars` vector in MATLAB (index 0), so the test states the Python convention as if it were the specification and is blind to F3.
- `pyvbmc/testing/vbmc/test_vbmc_init.py:468-484` and `:604-613` pin `uncertainty_handling=[3]` → level 1 and `[]` → level 0. The literal `[3]` only makes sense under the container-length reading; a specification-driven test would have tried `True`, `1` and `0` and found F4.
- `pyvbmc/testing/vbmc/test_options.py:117-161` (`test_init_with_specify_target_noise`) checks the five noisy overrides for the one flag that triggers them in Python. It never builds a level-1 run, so F1 is invisible to it. Its title ("Turning on specify_target_noise should adjust defaults") describes the implementation, where the MATLAB specification is "turning on uncertainty handling adjusts defaults".
- `pyvbmc/testing/vbmc/test_options.py:183-205` (`test_inert_options_are_the_declared_options_nothing_reads`) is the right idea with a rule that is too weak: it searches the package text for the quoted option name anywhere, so an unrelated dictionary key with the same spelling (`sampler_opts["diagnostics"]`) or a dead read of a same-named `optim_state` key (`optim_state["entropy_force_switch"]`, `optim_state["temperature"]`) counts as a reader. Matching an options access (`options[...]`, `options.get(...)`, `options.eval(...)`) instead would have flagged F2, F6 and F10.
- `pyvbmc/testing/vbmc/test_vbmc_init.py:572-582` (`test_vbmc_optimstate_entropy_switch`) checks that `entropy_switch` is copied and suppressed for small `D`, which is exactly the part that works, and never exercises the reader that is broken (F2).

Tests that are good models in this slice:

- `pyvbmc/testing/vbmc/test_options.py:164-180` parametrizes `fun_eval_start` over `D` and compares against an independently written expression, which is how the whole defaults table ought to be checked. Extending exactly this pattern to every `D`-dependent default would be a real MATLAB-parity gate.
- `pyvbmc/testing/vbmc/test_vbmc_init.py`'s bounds tests build a hierarchy of bad-bounds cases and assert the *error identity* (the `vbmc:...` tag carried over from MATLAB) rather than the code path, which is specification-shaped.
- `pyvbmc/testing/vbmc/test_options.py:208-246` (the inert-option warning tests) test observable user-facing behavior — a warning naming the option — rather than internal state.

---

## 4. Sheet notes

- **S1 (entry "Posterior tempering (`vbmc_power`, `vptrain2real`) is not ported", Slice P7).** The entry says the option `temperature = 1` "is read only by `pyvbmc/whitening/whitening.py:196`, `:291`". Those two lines read `optim_state["temperature"]`, a key that `_init_optim_state` never writes (F6), so the *option* is read nowhere at all and both sites always take the `T = 1` fallback. The entry's substance (tempering unported, identity at the default) stands; its statement about where the option is read does not.
- **S2 (entry "Declared options that nothing reads", Slice P1b).** The entry describes `INERT_OPTIONS` as the declared options nothing reads and cites a test that recomputes the set. Three more declared options are read nowhere through an options object and are not registered: `diagnostics`, `entropy_force_switch` and `temperature` (F10, F2, F6). The entry is therefore incomplete as a statement about the package, and the test it cites cannot close the gap in its present form.
- **S3 (no entry).** The default search acquisition function differs (`@acqf_vbmc` in MATLAB, `[AcqFcnLog()]` in PyVBMC, changed deliberately by `e20d081` on 2022-09-21 "for numerical stability"). This looks like a settled, deliberate difference and belongs on the sheet; at present nothing records it and the only MATLAB-versus-Python default flagged in the sheet is the bounded transform (F5).
- **S4 (entry "The space-filling design and the optimizer are SciPy", Slices G1/G2).** The entry says MATLAB's `defopts.OptimToolbox` "chooses between toolbox and fallback optimizers at runtime". At the comparison revision it does not: `OptimToolbox` occurs only at `vbmc.m:210`, in the `evalfields` list, and in `misc/setupoptions_vbmc.m:98-106` where it is computed. Nothing reads it afterwards in the VBMC repository, so it is dead in MATLAB too. The conclusion (no Python counterpart is needed) is unaffected.
- **S5 (entry "`results['rng_state']` is a generator snapshot", Slice P1a).** The entry says PyVBMC "never touches a global stream". That is true of a run, but `_set_random_state` (`vbmc.py:3105-3111`), on the oldest save format, calls `np.random.set_state` twice — restoring the saved global state, deriving a generator from it, and then leaving the global state at the saved value. The method's own docstring says as much; a reader comparing the sheet with the code may want the qualification "during a run".
- **S6 (counterpart map, `vbmc.m` row).** The row lists `FeatureTest` among the unported experimental features. Worth recording that it is not merely unported: it is read nowhere in MATLAB and, because of the quoting typo at `misc/setupoptions_vbmc.m:47`, never even evaluated (M4). Same for `ConstrainedGPMean`, which the sheet already lists as declared-with-no-reader.
- **S7 (no entry).** MATLAB's `boundscheck_vbmc.m:27-33` repairs only the first degenerate coordinate because `any` collapses the comparison to a scalar, while PyVBMC repairs all of them (M5); and MATLAB's integer-variable bound check is vacuous in its half-integer half while PyVBMC's is not (M12). Both are places where PyVBMC deliberately does not reproduce a MATLAB defect, and neither is recorded.
