# VBMC subpackage developing notes

These notes are used for keeping track of ToDos and porting information.

## active_sample.py

### Matlab references:
- active_sample(): [activesample_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/private/activesample_vbmc.m)
- active_sample() (part where gp is None): [initdesign_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/initdesign_vbmc.m)
- _get_search_points: getSearchPoints in [activesample_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/private/activesample_vbmc.m)

## iteration_history.py

### Porting status

- docstring of record_iteration() (depends on the other objects as well).     
- implement all elements of record_iteration() (depends on the other objects as well).

### Matlab references
-  function save_stats in: [vbmc.m](https://github.com/lacerbi/vbmc/blob/master/vbmc.m) 

## vbmc.py and related functions

### Porting status
- in _boundscheck the plausible bounds are set equal to the hard bounds, which is not admissible in VBMC (Fix by Luigi required)
- check the documentation of vbmc init and the class in general
- integrate the basic options somehow
- init a GP (wait for GPLite to be completed)
- experimental features (listed in [vbmc.m](https://github.com/lacerbi/vbmc/blob/master/vbmc.m) have not been ported yet)
 - [acqhedge_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/private/acqhedge_vbmc.m) has not been ported yet as it is considered to be experimental
- the warping in [vbmc.m](https://github.com/lacerbi/vbmc/blob/master/vbmc.m) is part of the 2020 paper and is well tested. We will implement it later as VBMC can run without it.
- in variational_optimization.py the compute vargrad is untested
- in variational_optimization.py the the vp_repo has not been ported for now (In is used in MATLAB in [vpsieve_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/vpsieve_vbmc.m) and filled in active_sampling). (Note by Luigi: In practice, even in MATLAB this part seems to be implemented lazily and then it's not used. I am not sure why I did not work on this more; probably the variational optimization was working fine and couldn't be bothered to fix this properly (it seems it could help).)
- We have to think about what should happen when somebody calls  vbmc.optimize twice, and stuff like that.

### Matlab references:
- initialization:
     - parts of the __init__(): [vbmc.m](https://github.com/lacerbi/vbmc/blob/master/vbmc.m)
     - parts of the __init__(): [setupvars_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/setupvars_vbmc.m)
     - _boundscheck(): [boundscheck_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/boundscheck_vbmc.m)
- VBMC loop (optimize(): loop in [vbmc.m](https://github.com/lacerbi/vbmc/blob/master/vbmc.m)):
     - Active Sampling:
          - see: :doc:[../functions/active_sample[
          - _reupdate_gp(): [gpreupdate.m](https://github.com/lacerbi/vbmc/blob/master/misc/gpreupdate.m) (Wait for gpyreg progress)
     - Gaussian Process training:
          - train_gp(): [gptrain_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/gptrain_vbmc.m)
          - _gp_hyp(): [gptrain_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/gptrain_vbmc.m)
          - _get_gp_training_options(): [get_GPTrainOptions.m](https://github.com/lacerbi/vbmc/blob/master/misc/get_GPTrainOptions.m)
          - _get_hyp_cov(): [get_GPTrainOptions.m](https://github.com/lacerbi/vbmc/blob/master/misc/get_GPTrainOptions.m)
          - _get_hpd(): [gethpd_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/gethpd_vbmc.m)
          - _get_training_data(): [get_traindata_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/get_traindata_vbmc.m)
          - _estimate_noise(): [gptrain_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/gptrain_vbmc.m)
     - Variational optimization / training of VP:
          - update_K(): [updateK.m](https://github.com/lacerbi/vbmc/blob/master/private/updateK.m)
          - optimize_vp(): [vpoptimize_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/vpoptimize_vbmc.m)
          - _initialize_full_elcbo(): [vpoptimize_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/vpoptimize_vbmc.m)
          - _eval_full_elcbo(): [vpoptimize_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/vpoptimize_vbmc.m)
          - _vp_bound_loss(): [vpbndloss.m](https://github.com/lacerbi/vbmc/blob/master/misc/vpbndloss.m)
          - _soft_bound_loss(): [softbndloss.m](https://github.com/lacerbi/vbmc/blob/master/utils/softbndloss.m)
          - _sieve(): [vpsieve_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/vpsieve_vbmc.m)
          - _vbinit(): [vbinit_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/vbinit_vbmc.m)
          - _negelcbo(): [negelcbo_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/negelcbo_vbmc.m)
          - _gplogjoint(): [gplogjoint.m](https://github.com/lacerbi/vbmc/blob/master/misc/gplogjoint.m)
     - Loop termination:
          - _check_warmup_end_conditions(): [vbmc_warmup.m](https://github.com/lacerbi/vbmc/blob/master/private/vbmc_warmup.m)
          - _setup_vbmc_after_warmup(): [vbmc_warmup.m](https://github.com/lacerbi/vbmc/blob/master/private/vbmc_warmup.m)
          - _recompute_lcbmax(): [recompute_lcbmax.m](https://github.com/lacerbi/vbmc/blob/master/private/recompute_lcbmax.m)
          - _is_finished(): [vbmc_termination.m](https://github.com/lacerbi/vbmc/blob/master/private/vbmc_termination.m)
          - _compute_reliability_index(): [vbmc_termination.m](https://github.com/lacerbi/vbmc/blob/master/private/vbmc_termination.m)
          - _is_gp_sampling_finished(): [vbmc_termination.m](https://github.com/lacerbi/vbmc/blob/master/private/vbmc_termination.m)
- Finalizing:
     - finalboost(): [finalboost_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/finalboost_vbmc.m)
     - determine_best_vp(): [best_vbmc.m](https://github.com/lacerbi/vbmc/blob/master/misc/best_vbmc.m)
     - _create_result_dict(): [vbmc_output.m](https://github.com/lacerbi/vbmc/blob/master/private/vbmc_output.m)
