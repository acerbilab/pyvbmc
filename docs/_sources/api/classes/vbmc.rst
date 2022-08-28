====
VBMC
====

.. note::

  The ``VBMC`` class implements the Variational Bayesian Monte Carlo (VBMC) algorithm.

  VBMC computes a variational approximation of the full posterior and a lower
  bound on the log normalization constant (log marginal likelhood or log model evidence)
  for a provided unnormalized log posterior.

  To perform inference, first initialize a ``VBMC`` object and then run ``optimize()``.

  The current version of VBMC only supports noiseless evaluations of the log posterior.
  We are currently working on implementing VBMC with noisy likelihoods.

  See below for the ``VBMC`` class methods and interface.
  For now, the only methods of interest for users are the ``VBMC`` constructor and ``optimize()``.


.. autoclass:: pyvbmc.vbmc.VBMC
   :members:
.. autofunction:: pyvbmc.vbmc.train_gp
.. autofunction:: pyvbmc.vbmc.update_K
.. autofunction:: pyvbmc.vbmc.optimize_vp
