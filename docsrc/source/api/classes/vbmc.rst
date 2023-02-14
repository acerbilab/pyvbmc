========
``VBMC``
========

.. note::

  The ``VBMC`` class implements the Variational Bayesian Monte Carlo (VBMC) algorithm.

  VBMC computes a variational approximation of the full posterior and a lower
  bound on the log normalization constant (log marginal likelhood or log model evidence)
  for a provided unnormalized log posterior.

  To perform inference, first initialize a ``VBMC`` object and then call ``vbmc.optimize()`` on the instance.

  By default VBMC assumes noiseless evaluations of the log posterior, but noisy likelihoods can also be handled. See :ref:`PyVBMC Example 6: Noisy log-likelihood evaluations` for more details.

  See below for more details on the ``VBMC`` class methods and interface. The primary entry-points for users are the ``VBMC`` class, which initializes the algorithm, and the :ref:`\`\`VariationalPosterior\`\`` class, which represents the returned variational solution. The :ref:`Basic options` may also be useful.

.. autoclass:: pyvbmc.vbmc.VBMC
   :members:
.. autofunction:: pyvbmc.vbmc.train_gp
.. autofunction:: pyvbmc.vbmc.update_K
.. autofunction:: pyvbmc.vbmc.optimize_vp
