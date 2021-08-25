====
vbmc
====

gaussian_process_train
======================

Matlab references
-----------------
- train_gp(): `gptrain_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/gptrain_vbmc.m>`_
- _gp_hyp(): `gptrain_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/gptrain_vbmc.m>`_
- _get_gp_training_options(): `get_GPTrainOptions.m <https://github.com/lacerbi/vbmc/blob/master/misc/get_GPTrainOptions.m>`_
- _get_hyp_cov(): `get_GPTrainOptions.m <https://github.com/lacerbi/vbmc/blob/master/misc/get_GPTrainOptions.m>`_
- _get_hpd(): `gethpd_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/gethpd_vbmc.m>`_
- _get_training_data(): `get_traindata_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/get_traindata_vbmc.m>`_
- _estimate_noise(): `gptrain_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/gptrain_vbmc.m>`_

train_gp
--------

.. autofunction:: pyvbmc.vbmc.train_gp

variational_optimization
========================

Matlab references
-----------------
- update_K(): `updateK.m <https://github.com/lacerbi/vbmc/blob/master/private/updateK.m>`_
- optimize_vp(): `vpoptimize_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/vpoptimize_vbmc.m>`_
- _initialize_full_elcbo(): `vpoptimize_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/vpoptimize_vbmc.m>`_
- _eval_full_elcbo(): `vpoptimize_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/vpoptimize_vbmc.m>`_
- _vp_bound_loss(): `vpbndloss.m <https://github.com/lacerbi/vbmc/blob/master/misc/vpbndloss.m>`_
- _soft_bound_loss(): `softbndloss.m <https://github.com/lacerbi/vbmc/blob/master/utils/softbndloss.m>`_
- _sieve(): `vpsieve_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/vpsieve_vbmc.m>`_
- _vbinit(): `vbinit_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/vbinit_vbmc.m>`_
- _negelcbo(): `negelcbo_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/negelcbo_vbmc.m>`_
- _gplogjoint(): `gplogjoint.m <https://github.com/lacerbi/vbmc/blob/master/misc/gplogjoint.m>`_

optimize_vp
-----------

.. autofunction:: pyvbmc.vbmc.update_K
.. autofunction:: pyvbmc.vbmc.optimize_vp
