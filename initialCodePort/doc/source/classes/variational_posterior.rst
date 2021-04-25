====================
VariationalPosterior
====================

Porting status
==============
.. todo::
     - vbmc_plot: `vbmc_plot.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_plot.m>`_
     - vbmc_power: `vbmc_power.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_power.m>`_
     - _robustSampleFromVP: robustSampleFromVP in `vbmc.m <https://github.com/lacerbi/vbmc/blob/master/vbmc.m>`_
     - _init__(): completely missing so far (mock for test exists)

Matlab references of ported methods
-----------------------------------
- get_parameters(): `get_vptheta.m <https://github.com/lacerbi/vbmc/blob/master/misc/get_vptheta.m>`_ ✓
- kldiv(): `vbmc_kldiv.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_kldiv.m>`_ ✓
- mode(): `vbmc_mode.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_mode.m>`_ ✓
- moments(): `vbmc_moments.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_moments.m>`_ ✓
- mtv(): `vbmc_mtv.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_mtv.m>`_  ✓
- pdf(): `vbmc_pdf.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_pdf.m>`_  ✓
- sample(): `vbmc_rnd.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_rnd.m>`_
   - gp_sample is missing
- set_parameters(): `rescale_params.m <https://github.com/lacerbi/vbmc/blob/master/misc/rescale_params.m>`_ ✓

Known issues:
-----------------------------------
- mtv(): known issue with vps of more than one mode with high variance between the modes (due to the bandwidth computation of  `scipy.stats.gaussian_kde <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_)

Python documentation
====================
.. autoclass:: variational_posterior.VariationalPosterior
   :members: