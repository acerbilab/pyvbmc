=================================
``VariationalPosterior.to_arviz``
=================================

.. automethod:: pyvbmc.VariationalPosterior.to_arviz

The returned xarray DataTree has one ``posterior`` group and one chain. Every
parameter is represented as a scalar variable over the ``chain`` and ``draw``
dimensions. Names default to ``x_0``, ..., ``x_{D-1}``, or can be supplied
through ``var_names``.

The method draws once from the posterior and advances ``vp.rng``. The samples
are independent draws from a variational approximation. MCMC convergence
diagnostics on this single-chain export do not assess approximation quality;
use posterior summaries and plots instead.

See :ref:`ArviZ DataTree` for a short example and installation details in
:ref:`optional integrations`.
