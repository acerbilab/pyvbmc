====
VBMC
====

Porting status
==============
.. todo::
     - in _boundscheck the plausible bounds are set equal to the hard bounds, which is not admissible in VBMC (Fix by Luigi required)
     - check the documentation of vbmc init and the class in general
     - integrate the basic options somehow
     - init a GP (wait for GPLite to be completed)

Matlab references of ported methods
-----------------------------------
- initialization:
     - parts of the __init__(): `vbmc.m <https://github.com/lacerbi/vbmc/blob/master/vmbc.m>`_
     - parts of the __init__(): `setupvars_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/setupvars_vbmc.m>`_
     - _boundscheck(): `boundscheck_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/boundscheck_vbmc.m>`_
- VBMC loop (optimize(): loop in `vbmc.m <https://github.com/lacerbi/vbmc/blob/master/vmbc.m>`_):
     - Active Sampling:
          - _activesample(): `activesample_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/private/activesample_vbmc.m>`_
          - _reupdate_gp(): `gpreupdate.m <https://github.com/lacerbi/vbmc/blob/master/misc/gpreupdate.m>`_
     - Gaussian Process training:
          - _train_gp(): `gptrain_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/gptrain_vbmc.m>`_
     - Variational optimization / training of VP:
          - _updateK(): `updateK.m <https://github.com/lacerbi/vbmc/blob/master/private/updateK.m>`_
          - _optimize_vp(): `vpoptimize_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/vpoptimize_vbmc.m>`_
     - Loop termination:
          - _check_warmup_end(): `vbmc_warmup.m <https://github.com/lacerbi/vbmc/blob/master/private/vbmc_warmup.m>`_
          - _recompute_lcbmax(): `recompute_lcbmax.m <https://github.com/lacerbi/vbmc/blob/master/private/recompute_lcbmax.m>`_
          - _is_finished(): `vbmc_termination.m <https://github.com/lacerbi/vbmc/blob/master/private/vbmc_termination.m>`_
- Finalizing:
     - _finalboost(): `finalboost_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/finalboost_vbmc.m>`_
     - _determine_best_vp(): `best_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/best_vbmc.m>`_


Python documentation
====================
.. autoclass:: pyvbmc.vbmc.VBMC
   :members: