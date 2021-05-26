====================
VariationalPosterior
====================

Porting status
==============
.. todo::
     - vbmc_plot: `vbmc_plot.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_plot.m>`_ (more important)
     - vbmc_power: `vbmc_power.m <https://github.com/lacerbi/vbmc/blob/master/vbmc_power.m>`_ (less important for now)
     - _robustSampleFromVP: robustSampleFromVP in `vbmc.m <https://github.com/lacerbi/vbmc/blob/master/vbmc.m>`_

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


Python documentation
====================
.. autoclass:: variational_posterior.VariationalPosterior
   :members: