# MATLAB-to-Python counterpart map

Every `.m` file of the MATLAB VBMC repository at revision `396d649`, with
its Python counterpart in PyVBMC or in gpyreg (`main`, `9e70e6b`), the slice
of the review plan that names the MATLAB file, and a status. The map was
built against PyVBMC `f91fdf0`. Its Python line numbers were refreshed on
2026-09-19 against `80204f7` on `dev-port-review`, the last of the wave-1
fixes, and those of the eleven rows that the wave-2 pass changed
(`boundscheck_vbmc.m`, `proposal_vbmc.m`, `setupoptions_vbmc.m`,
`setupvars_vbmc.m`, `vptrain2real.m`, `warp_gpandvp_vbmc.m`,
`warp_input_vbmc.m`, `recompute_lcbmax.m`, `vbmc_output.m`,
`vbmc_termination.m` and `vbmc_warmup.m`) on 2026-09-21, after the
independent check of that pass; the line numbers of the other rows are
those of `80204f7`.

Demos, plots, examples, tests and `install.m` are excluded; they are listed
at the end so that nothing is silently dropped.

**Paths.** Python paths beginning `pyvbmc/` are relative to the PyVBMC
repository root, paths beginning `gpyreg/` to the gpyreg repository root.
MATLAB paths are relative to the MATLAB VBMC repository root.

**Status.** `ported` (a Python counterpart implements it), `partial` (some of
it is ported), `unported` (no Python counterpart), `removed` (PyVBMC had one
and it was taken out, retained on a branch), `substituted` (a third-party
library does the job).

**Slice.** The slice of the plan's table that names the MATLAB file, or
`none`. Where two slices name the same file, both are given; `O*` entries are
third readers layered on top.

| MATLAB file | Python counterpart(s) (path:function) | Slice in the plan | Status | Note |
| --- | --- | --- | --- | --- |
| `vbmc.m` | `pyvbmc/vbmc/vbmc.py: VBMC.__init__`, `optimize`, `_init_optim_state`, `_log_column_headers`, `_setup_logging_display_format` | P1a, P1b | partial | The `defopts` block is split over `pyvbmc/vbmc/option_configs/*.ini`. The experimental features listed in it (`AcqHedge`, `BOWarmup`, `VarActiveSample`, `WarpNonlinear`, `RetryMaxFunEvals`, `AltMCEntropy`, `VarParamsBack`, `gpOutwarpFun`, `gpIntMeanFun`, `Bandwidth`, `FeatureTest`, `OptimToolbox`) are unported. `initFromVP` is unported (`pyvbmc/variational_posterior/README.md`). |
| `lpostfun.m` | `pyvbmc/vbmc/vbmc.py: _init_log_joint`, `_rebuild_log_joint` | P1b | ported | MATLAB's is an example of a user-written log posterior; PyVBMC builds the log joint from `log_likelihood` and `prior`. |
| `vbmc_diagnostics.m` | none | P1b | unported | |
| `vbmc_isavp.m` | `isinstance(..., VariationalPosterior)` at the call sites | none | unported | No file is needed in Python. |
| `vbmc_kldiv.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.kl_div` | P7 | ported | |
| `vbmc_mode.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.mode` | P7 | ported | MATLAB commit `11737f5` (2021-03-27) constrained the starting points to the bounds; slice M checks whether PyVBMC followed. |
| `vbmc_moments.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.moments` | P7 | ported | |
| `vbmc_mtv.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.mtv` | P7 | ported | Uses `scipy.integrate.trapezoid` in place of `shared/qtrapz.m`. |
| `vbmc_pdf.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.pdf` | P7, O2 | partial | Original-space gradients raise `NotImplementedError` (known-differences sheet, P7). |
| `vbmc_plot.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.plot` | none | substituted | Uses the `corner` package instead of `utils/cornerplot.m`/`utils/kde2d.m`. |
| `vbmc_power.m` | none | P7 | unported | Posterior tempering. |
| `vbmc_rnd.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.sample` | P7 | partial | The `balanceflag == 'gp'` branch (`vbmc_rnd.m:49`, calling `gpsample_vbmc`) is unported. |
| `acq/acqwrapper_vbmc.m` | `pyvbmc/acquisition_functions/abstract_acq_fcn.py: AbstractAcqFcn.__call__` | P3 | ported | |
| `acq/acqeig_vbmc.m` | none in the package; `pyvbmc/acquisition_functions/acq_fcn_eig.py` on `retain/experimental-acquisitions` at `fa6922f` | P3 | removed | Removed 2026-09-14 (PI decision 2026-09-13). |
| `acq/acqf_vbmc.m` | `pyvbmc/acquisition_functions/acq_fcn.py: AcqFcn` | P3 | ported | |
| `acq/acqflog_vbmc.m` | `pyvbmc/acquisition_functions/acq_fcn_log.py: AcqFcnLog` | P3 | ported | |
| `acq/acqfsn2_vbmc.m` | `pyvbmc/acquisition_functions/acq_fcn_noisy.py: AcqFcnNoisy` | P3 | ported | |
| `acq/acqimiqr_vbmc.m` | `pyvbmc/acquisition_functions/acq_fcn_imiqr.py: AcqFcnIMIQR` | P3 | ported | |
| `acq/acqus_vbmc.m` | `pyvbmc/acquisition_functions/acq_fcn_vanilla.py: AcqFcnVanilla` | P3 | ported | |
| `acq/acqviqr_vbmc.m` | `pyvbmc/acquisition_functions/acq_fcn_viqr.py: AcqFcnVIQR` | P3 | partial | The integrated-mean-function branches are unported (`:293`, `:360`). The experimental `loss` variants were removed 2026-09-14. MATLAB commit `a9615ba` (2022-07-23) changed VIQR's speed; slice M checks it. |
| `ent/entlb_vbmc.m` | `pyvbmc/entropy/entlb_vbmc.py: entlb_vbmc` | P7, O2 | ported | |
| `ent/entmc_vbmc.m` | `pyvbmc/entropy/entmc_vbmc.py: entmc_vbmc` | P7, O2 | ported | Vectorized 2026-09-05 (`dev/plans/stage2-entmc.md`), draw order preserved. MATLAB commit `1b72896` (2021-06-23) fixed the weight gradient; slice M checks it. |
| `ent/entub_vbmc.m` | none | P7 | unported | |
| `misc/best_vbmc.m` | `pyvbmc/vbmc/vbmc.py: determine_best_vp` | P1a | ported | |
| `misc/boundscheck_vbmc.m` | `pyvbmc/vbmc/vbmc.py: _bounds_check`; `pyvbmc/vbmc/_bounds.py` | P1b | ported | Scalar bounds are replicated since the wave-2 fixes (`477226b`). PyVBMC repairs every degenerate plausible coordinate where `:27-30` repairs the first (known-differences sheet, P1b). |
| `misc/check_quadcoefficients_vbmc.m` | none | P2 | unported | Integrated mean function. |
| `misc/evaloption_vbmc.m` | `pyvbmc/vbmc/options.py: Options.eval` | P1b | ported | Referenced in a comment at `pyvbmc/vbmc/active_sample.py:316`. |
| `misc/fess_vbmc.m` | `pyvbmc/vbmc/active_importance_sampling.py:442: fess` | P1a | ported | The Python function lives in a P4 file, not in `vbmc.py`; its MATLAB callers are `private/activeimportancesampling_vbmc.m` and `private/activesample_vbmc.m`. See REPORT.md. |
| `misc/finalboost_vbmc.m` | `pyvbmc/vbmc/vbmc.py: final_boost`, `_validate_final_boost_tolerance`, `_is_valid_final_boost_score`, `_accept_final_boost_candidate` | P1a | ported | PyVBMC adds the acceptance guard and zeroes the boost-only weight penalty (known-differences sheet, P1a). |
| `misc/funlogger_vbmc.m` | `pyvbmc/function_logger/function_logger.py: FunctionLogger` | P8 | ported | `finalize()` is explicit, not called by `optimize`. `batch_call` and `precomputed_evaluations` are Python-only. |
| `misc/get_GPTrainOptions.m` | `pyvbmc/vbmc/gaussian_process_train.py: _get_gp_training_options`, `_get_hyp_cov` | P5 | partial | Only the `slicesample` sampler branch is ported; `npv`, `mala`, `slicelite`, `splitsample` and `covsample` are not. |
| `misc/get_traindata_vbmc.m` | `pyvbmc/vbmc/gaussian_process_train.py:794: _get_training_data` | P2 | partial | The Python function lives in a P5 file. Its `noiseshaping_vbmc` call is unported. See REPORT.md. |
| `misc/get_vptheta.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.get_parameters` | P7 | ported | |
| `misc/gethpd_vbmc.m` | `pyvbmc/stats/get_hpd.py: get_hpd` | P7 | ported | Called from `pyvbmc/vbmc/gaussian_process_train.py` and `pyvbmc/vbmc/active_sample.py` as well. |
| `misc/gplogjoint.m` | `pyvbmc/vbmc/variational_optimization.py:1352: _gp_log_joint` | P6, O1 | partial | `compute_var == 2` (diagonal variance approximation) and the variance gradient are unported. |
| `misc/gplogjoint_weights.m` | none | P6 | unported | Weight-only fast path; PyVBMC always uses the general routine. |
| `misc/gpreupdate.m` | `pyvbmc/vbmc/gaussian_process_train.py:938: reupdate_gp` | P2 | partial | The Python function lives in a P5 file. The `check_quadcoefficients_vbmc` call and the integrated-mean-function branch are unported. See REPORT.md. |
| `misc/gpsample_vbmc.m` | none | P5 | unported | "Missing port: sample for GP for debug (not used)" (`gaussian_process_train.py:205`). |
| `misc/gptrain_vbmc.m` | `pyvbmc/vbmc/gaussian_process_train.py: train_gp`, `_gp_hyp`, `_estimate_noise`, `_meanfun_name_to_mean_function`, `_cov_identifier_to_covariance_function` | P5 | partial | The local subfunction `vbmc_gphyp` (`gptrain_vbmc.m:109`) is `_gp_hyp`. Output-warping and integrated-mean hyperpriors are unported. |
| `misc/initdesign_vbmc.m` | `pyvbmc/vbmc/active_sample.py:41: active_sample` (the `gp is None` branch) | P2 | partial | The k-means clustering of surplus starting points (`utils/fastkmeans.m`) is unported: PyVBMC takes the first `sample_count` rows and leaves the rest in the starting cache, where MATLAB leaves the rows its clustering did not choose (known-differences sheet, P2). The MATLAB file last changed in `1e17fb8` (2019-09-13); commit `46b6f5e` (2021-05-26) touches only `gplite/private/fminfill.m` and `utils/fminfill.m`, whose counterpart is `gpyreg/f_min_fill.py`. |
| `misc/intkernel.m` | none | P3 | unported | Only caller is `acq/acqeig_vbmc.m`, itself removed. |
| `misc/negelcbo_vbmc.m` | `pyvbmc/vbmc/variational_optimization.py:1103: _neg_elcbo` | P6, O1 | partial | The eta soft bound is deliberately removed and the caller's `theta` is not mutated; the weight-only branches are unported (known-differences sheet, P6). |
| `misc/noiseshaping_vbmc.m` | none (the `noise_shaping*` options exist and are inert) | P3 | unported | Default off on both sides, and `VBMC.__init__` rejects `noise_shaping=True` (known-differences sheet, P5). |
| `misc/proposal_vbmc.m` | none; the option value is the placeholder string `"@(x)proposal_vbmc"` (`pyvbmc/vbmc/vbmc.py:1057`) | P2 | unported | `optimState.ProposalFcn` is set in `misc/setupvars_vbmc.m` but read nowhere in MATLAB either. |
| `misc/real2int_vbmc.m` | `pyvbmc/acquisition_functions/abstract_acq_fcn.py:261: AbstractAcqFcn._real2int` | P8 | ported | The Python function lives in a P3 file and snaps its input in place. See REPORT.md. |
| `misc/rescale_params.m` | `pyvbmc/variational_posterior/variational_posterior.py: VariationalPosterior.set_parameters` | P7 | ported | |
| `misc/setupoptions_vbmc.m` | `pyvbmc/vbmc/options.py: Options` (`validate_run_limits` for `:109-119`, `update_defaults` for `:127-163`); `pyvbmc/vbmc/option_configs/basic_vbmc_options.ini`, `advanced_vbmc_options.ini` | P1b | partial | The defaults themselves come from `vbmc.m`'s `defopts` block, not from this file. The checks of `MaxFunEvals` and `MaxIter` and the noisy defaults at either uncertainty-handling level are ported since the wave-2 fixes (`17badf9`, `0ba7680`); the `MinFunEvals` branch (`:120-124`) and `OptimToolbox` (`:98-106`) have no counterpart. |
| `misc/setupvars_vbmc.m` | `pyvbmc/vbmc/vbmc.py:936: _init_optim_state`; `VBMC.__init__` for the transformer, the transformed `x0` and the initial variational posterior (`:65`, `:82`) | P1b | ported | MATLAB commits `74b046e`/`f3e5d76` (2022) changed the bounded transform here; slice M checks it. `integer_vars` takes a boolean mask or 0-based indices where `:14-17` reads 1-based indices, `Temperature` (`:249-255`) is not ported, and the initial variational means are built from the transformed `x0` since the wave-2 fixes (`8cd4bbc`); known-differences sheet, P1b and P7. |
| `misc/testpdf.m` | none | none | unported | Test helper (excluded category; listed because it sits in `misc/`). |
| `misc/vbinit_vbmc.m` | `pyvbmc/vbmc/variational_optimization.py:860: _vb_init` | P6 | ported | The dormant type-3 frozen-sigma case follows a Python-only convention (known-differences sheet, P6). |
| `misc/vbmc_gphyp.m` | `pyvbmc/vbmc/gaussian_process_train.py:279: _gp_hyp` | none | ported | The `.m` file is zero bytes at `396d649`; the function is a local subfunction of `misc/gptrain_vbmc.m:109`. |
| `misc/vpbndloss.m` | `pyvbmc/vbmc/variational_optimization.py:532: _vp_bound_loss` | P6, O1 | partial | The eta block is deliberately inert (known-differences sheet, P6). |
| `misc/vpbounds.m` | `pyvbmc/variational_posterior/variational_posterior.py:296: VariationalPosterior.get_bounds` | P7 | partial | PyVBMC computes the box from the training inputs of the call; MATLAB accumulates it across iterations (known-differences sheet, P7). |
| `misc/vpoptimize_vbmc.m` | `pyvbmc/vbmc/variational_optimization.py:91: optimize_vp`, `:418 _initialize_full_elcbo`, `:452 _eval_full_elcbo` | P6 | ported | |
| `misc/vpoptimizeweights_vbmc.m` | none | P6 | unported | Its only MATLAB call site (`vbmc.m:717`) is itself commented out. |
| `misc/vpsample_vbmc.m` | none | P6 | unported | Variational-parameter sampling. |
| `misc/vpsieve_vbmc.m` | `pyvbmc/vbmc/variational_optimization.py:705: _sieve` | P6 | partial | `vp_repo` and `ELCBOWeight` are unported. |
| `misc/vptrain2real.m` | none; the call is commented out at `pyvbmc/vbmc/vbmc.py:1438`, `:1599` | P7 | unported | Only active for `vp.temperature` in `{2,3,4,5}`; identity at the default temperature. |
| `misc/warp_gpandvp_vbmc.m` | `pyvbmc/whitening/whitening.py:338: warp_gp_and_vp` | P8, O3 | partial | Rotoscaling only; the output-warping line is a comment (`whitening.py:381-383`). |
| `misc/warp_input_vbmc.m` | `pyvbmc/whitening/whitening.py:143: warp_input` | P8 | partial | Nonlinear warping unported. MATLAB commits `f9c04bc`/`a5240d2` (2021-02-02) changed warp robustness; slice M checks it. PyVBMC maps the search state back with the transform of the current inference space where `:8`, `:133` use the handed posterior's (known-differences sheet, P8). |
| `private/acqhedge_vbmc.m` | none | P2 | unported | `acq_hedge = True` leaves `idx_acq` unset in `pyvbmc/vbmc/active_sample.py:320-323`. |
| `private/activeimportancesampling_vbmc.m` | `pyvbmc/vbmc/active_importance_sampling.py:10: active_importance_sampling`, `:333 active_sample_proposal_pdf`, `:409 get_mcmc_opts`, `:497 renormalize_weights` | P4 | partial | The sampler is substituted (see `gplite/private/eissample_lite.m`); the integrated-mean-function branch is unported; the MCMC branch is a dormant hook. |
| `private/activesample_vbmc.m` | `pyvbmc/vbmc/active_sample.py:41: active_sample`, `:877 _get_search_points` | P2 | partial | Seven "Missing port" markers remain (`:343`, `:365`, `:632`, `:677`, `:722`, `:735`, `:785`). MATLAB commit `68a197b` (2022-06-25) added the `slicesample` branch of the acquisition search here, which is unported (known-differences sheet, P2). |
| `private/recompute_lcbmax.m` | `pyvbmc/vbmc/vbmc.py:2492: _recompute_lcb_max` | P1a | ported | Ported on 2026-09-20 (`8e591ff`); until then the function returned an empty array that nothing read, and this row said "ported" all the same (`verification/wave2.md`, W2-2). |
| `private/updateK.m` | `pyvbmc/vbmc/variational_optimization.py:20: update_K` | P1a | ported | The Python function lives in a P6 file. See REPORT.md. |
| `private/vbmc_output.m` | `pyvbmc/vbmc/vbmc.py:3353: _create_result_dict` | none | partial | `overhead` is `NaN`; `rng_state` is a generator snapshot; `problem_type` tests the original bounds where `:5-9` tests the transformed ones. See REPORT.md and the known-differences sheet, P1a. |
| `private/vbmc_termination.m` | `pyvbmc/vbmc/vbmc.py:2177: _check_termination_conditions`, `:2297 _compute_reliability_index`, `:2355 _check_gp_sampling_stop`, `:2365 _is_gp_sampling_finished`, `:2420 _ensure_gp_sampling_history` | P1a | ported | |
| `private/vbmc_warmup.m` | `pyvbmc/vbmc/vbmc.py:2011: _check_warmup_end_conditions`, `:2104 _setup_vbmc_after_warmup` | P1a | ported | The recent-improvement window, the preference for the recomputed LCB maxima and the warping clocks follow MATLAB since the wave-2 fixes (`c918d12`, `8e591ff`, `fb8a12e`). |
| `shared/kde1d.m` | `pyvbmc/stats/kde_1d.py:144: kde_1d` | P7 | ported | |
| `shared/msmoothboxlogpdf.m` | `pyvbmc/priors/smooth_box.py: SmoothBox._log_pdf` | P9 | ported | |
| `shared/msmoothboxpdf.m` | `pyvbmc/priors/prior.py: Prior.pdf` over `pyvbmc/priors/smooth_box.py: SmoothBox._log_pdf` | P9 | ported | |
| `shared/msmoothboxrnd.m` | `pyvbmc/priors/smooth_box.py: SmoothBox.sample` | P9 | ported | |
| `shared/msplinetrapezlogpdf.m` | `pyvbmc/priors/spline_trapezoidal.py: SplineTrapezoidal._log_pdf` | P9 | ported | |
| `shared/msplinetrapezpdf.m` | `pyvbmc/priors/prior.py: Prior.pdf` over `pyvbmc/priors/spline_trapezoidal.py: SplineTrapezoidal._log_pdf` | P9 | ported | |
| `shared/msplinetrapezrnd.m` | `pyvbmc/priors/spline_trapezoidal.py: SplineTrapezoidal.sample` | P9 | ported | |
| `shared/mtrapezlogpdf.m` | `pyvbmc/priors/trapezoidal.py: Trapezoidal._log_pdf` | P9 | ported | |
| `shared/mtrapezpdf.m` | `pyvbmc/priors/prior.py: Prior.pdf` over `pyvbmc/priors/trapezoidal.py: Trapezoidal._log_pdf` | P9 | ported | |
| `shared/mtrapezrnd.m` | `pyvbmc/priors/trapezoidal.py: Trapezoidal.sample` | P9 | ported | |
| `shared/munifboxlogpdf.m` | `pyvbmc/priors/uniform_box.py: UniformBox._log_pdf` | P9 | ported | |
| `shared/munifboxpdf.m` | `pyvbmc/priors/prior.py: Prior.pdf` over `pyvbmc/priors/uniform_box.py: UniformBox._log_pdf` | P9 | ported | |
| `shared/munifboxrnd.m` | `pyvbmc/priors/uniform_box.py: UniformBox.sample` | P9 | ported | |
| `shared/mvnkl.m` | `pyvbmc/stats/kl_div_mvn.py:4: kl_div_mvn` | P7 | ported | |
| `shared/qtrapz.m` | inlined in `pyvbmc/variational_posterior/variational_posterior.py: mtv` via `scipy.integrate.trapezoid` | none | substituted | See REPORT.md. |
| `shared/warpvars_vbmc.m` | `pyvbmc/parameter_transformer/parameter_transformer.py: ParameterTransformer` (`__call__`, `inverse`, `log_abs_det_jacobian`) | P8, O3 | partial | The gradient of the log Jacobian is unported. MATLAB commits `74b046e`/`f3e5d76` (2022) changed the bounded transform and added the `probit` name for `norminv`; both are reflected in the Python (`parameter_transformer.py:109-114` accepts either name, and the bounded type is set before `mu`/`delta` are computed). The two defaults differ: `defopts.BoundedTransform = 'logit'` (`vbmc.m:344`) against `bounded_transform = "probit"` (known-differences sheet, P8). |
| `utils/cmaes_modded.m` | `cma` package, called from `pyvbmc/vbmc/active_sample.py:572-578` | P2 | substituted | The search runs without cma's noise handler, matching MATLAB's `Noise.on = 0`. The settings the two packages express differently are listed in the known-differences sheet, P2. |
| `utils/covcma.m` | none | none | unported | Its only MATLAB call site (`private/activesample_vbmc.m:601`) is commented out. |
| `utils/eissample_lite.m` | none (duplicate of `gplite/private/eissample_lite.m`) | none | unported | PyVBMC uses gpyreg's `SliceSampler`; see the P4 slice. |
| `utils/evalbool.m` | `pyvbmc/vbmc/options.py` (`.ini` values are Python literals) | none | substituted | MATLAB's `'yes'`/`'no'` strings have no Python analogue. See REPORT.md. |
| `utils/fastkmeans.m` | none | none | unported | Used by `misc/initdesign_vbmc.m`; PyVBMC takes the first `sample_count` points instead. See REPORT.md. |
| `utils/fminadam.m` | `pyvbmc/vbmc/minimize_adam.py:8: minimize_adam` | P6 | ported | |
| `utils/fminfill.m` | `gpyreg/f_min_fill.py: f_min_fill` (VBMC's own copy has no separate Python counterpart) | none | unported | Identical in purpose to `gplite/private/fminfill.m`; MATLAB commit `bd38b48` (2022-01-30) updated this copy alone. See REPORT.md. |
| `utils/malasample_vbmc.m` | none | P6 | unported | |
| `utils/quantile1.m` | `np.quantile` at the call sites | none | substituted | Different quantile convention (known-differences sheet, G2). |
| `utils/slicelite.m` | none | none | unported | The `slicelite` GP hyperparameter sampler is not ported (`gpyreg/gaussian_process.py:1121` records "Not used since no slicelite"). |
| `utils/slicesample_vbmc.m` | none | P6 | unported | |
| `utils/slicesamplebnd.m` | `gpyreg/slice_sample.py: SliceSampler` (duplicate of `gplite/private/slicesamplebnd.m`) | none | ported | |
| `utils/softbndloss.m` | `pyvbmc/vbmc/variational_optimization.py:654: _soft_bound_loss` | P6 | ported | |
| `utils/sq_dist.m` | `pyvbmc/acquisition_functions/abstract_acq_fcn.py:288: AbstractAcqFcn._sq_dist` (duplicate of `gplite/private/sq_dist.m`) | none | ported | gpyreg uses `scipy.spatial.distance.cdist` instead. See REPORT.md. |
| `utils/unscent_warp.m` | `pyvbmc/whitening/whitening.py:7: unscent_warp` | P8 | ported | |
| `gplite/gplite_clean.m` | `gpyreg/gaussian_process.py:996: GP.clean` | G1 | ported | Never called by PyVBMC. |
| `gplite/gplite_covfun.m` | `gpyreg/covariance_functions.py` (`SquaredExponential`, `Matern`, `RationalQuadraticARD`); `gpyreg/isotropic_covariance_functions.py` | G2, O4 | partial | `RationalQuadraticARD` and the isotropic classes are Python-only. |
| `gplite/gplite_fmin.m` | none | none | unported | Global optimization of a GP-represented density. gpyreg's `AGENTS.md` maps this to `f_min_fill.py`, which is wrong. See REPORT.md. |
| `gplite/gplite_hypprior.m` | `gpyreg/gaussian_process.py: set_priors`, `get_priors`, `__compute_log_priors`, `__prior_masks`, `__recompute_normalization_constants` | G1 | ported | |
| `gplite/gplite_intmeanfun.m` | none | G2 | unported | Absent from gpyreg's reference copy of `gplite`. |
| `gplite/gplite_meanfun.m` | `gpyreg/mean_functions.py` (`ZeroMean`, `ConstantMean`, `NegativeQuadratic`) | G2, O4 | partial | 3 of MATLAB's 24 mean functions. |
| `gplite/gplite_nlZ.m` | `gpyreg/gaussian_process.py:1656: log_likelihood`, `:1681 log_posterior`, `:1713 __compute_nlZ`, `:1733 __gp_obj_fun` | G1, O4 | ported | |
| `gplite/gplite_noisefun.m` | `gpyreg/noise_functions.py: GaussianNoise` | G2, O4 | ported | |
| `gplite/gplite_post.m` | `gpyreg/gaussian_process.py:763: GP.update`; `gpyreg/gaussian_process.py:2891: Posterior` | G1 | partial | The integrated-mean-function branches are unported. |
| `gplite/gplite_pred.m` | `gpyreg/gaussian_process.py:1856: GP.predict`, `:1754 predict_full` | G2 | partial | The integrated-mean-function branches are unported. MATLAB commit `68a197b` (2022-06-25) fixed the GP log density here; gpyreg's reference copy `gpyreg/matlab/gplite/gplite_pred.m:108` predates it, but gpyreg's Python code carries the fix — `GP.predict(..., return_lpd=True)` divides by the total predictive variance (`gaussian_process.py:2035-2064`). Nothing under `pyvbmc/` requests `return_lpd`. |
| `gplite/gplite_qpred.m` | none | G2 | unported | `gpyreg/gaussian_process.py:2258` records the gap. |
| `gplite/gplite_quad.m` | `gpyreg/gaussian_process.py:2074: GP.quad` | G2 | ported | gpyreg's `AGENTS.md` lists this as not ported; it is. See REPORT.md. |
| `gplite/gplite_rnd.m` | `gpyreg/gaussian_process.py:2515: GP.random_function` | G2 | ported | |
| `gplite/gplite_sample.m` | none | G2 | unported | Sampling from the log density a GP represents. gpyreg's `AGENTS.md` maps this to `slice_sample.py`, which is wrong. See REPORT.md. |
| `gplite/gplite_train.m` | `gpyreg/gaussian_process.py:1021: GP.fit` | G1 | partial | Only the `slicesample` hyperparameter sampler is ported. The integrated-mean-function lines are unported. |
| `gplite/outwarp_negpow.m` | none | G2 | unported | |
| `gplite/outwarp_negpowc1.m` | none | G2 | unported | |
| `gplite/outwarp_negscaledpow.m` | none | G2 | unported | |
| `gplite/private/derivcheck.m` | `gpyreg/testing/test_utils.py: check_grad` (numdifftools); `pyvbmc/testing/_check_grad.py` | O4 | substituted | |
| `gplite/private/eissample_lite.m` | none; `gpyreg/slice_sample.py: SliceSampler` stands in at the PyVBMC call sites | P4 | substituted | |
| `gplite/private/fminfill.m` | `gpyreg/f_min_fill.py: f_min_fill` | G1 | ported | SciPy `stats.qmc.Sobol` replaces MATLAB's own fill; `uuinv` is the mixture-of-uniforms inverse CDF. MATLAB commit `1d1f20d` (2022-01-30) fixed `uuinv`, and gpyreg's reference copy already carries the fix. |
| `gplite/private/gplite_core.m` | `gpyreg/gaussian_process.py:2639: GP.__core_computation` | G1, O4 | partial | The integrated-mean-function block (119 lines of the difference against gpyreg's reference copy) is unported. |
| `gplite/private/quantile1.m` | `np.quantile` in `gpyreg/mean_functions.py` | none | substituted | Different quantile convention. See REPORT.md. |
| `gplite/private/slicesamplebnd.m` | `gpyreg/slice_sample.py: SliceSampler` | G1 | ported | gpyreg's reference copy differs from the target only in documentation comments (Thin/Burnin). |
| `gplite/private/sq_dist.m` | `scipy.spatial.distance.cdist` in `gpyreg/covariance_functions.py`; `pyvbmc/acquisition_functions/abstract_acq_fcn.py: AbstractAcqFcn._sq_dist` on the VBMC side | G2 | substituted | |

## Excluded files

Demos, plots, examples, tests and the installer, listed so that the exclusion
is explicit:

`install.m`, `rosenbrock_test.m`, `vbmc_examples.m`,
`private/vbmc_demo2d.m`, `private/vbmc_iterplot.m`, `private/vbmc_plot2d.m`,
`gplite/gplite_demo.m`, `gplite/gplite_plot.m`, `gplite/gplite_test.m`,
`gplite/outwarp_test.m`, `shared/warpvars_vbmc_test.m`,
`test/runtest_vbmc.m`, `test/test_pdfs_vbmc.m`, `utils/cornerplot.m`,
`utils/kde2d.m`, `utils/ibslike.m`, `utils/psycho_gen.m`.

Two of these have Python counterparts worth knowing about:
`gplite/gplite_plot.m` → `gpyreg/gaussian_process.py:2259: GP.plot`
(partial: the sigma and quantile branches are disabled), and
`utils/cornerplot.m`/`utils/kde2d.m` → the `corner` package used by
`VariationalPosterior.plot`.
`misc/testpdf.m` is a test helper but sits in `misc/`, so it appears in the
main table.
