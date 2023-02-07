======
Priors
======

.. note::
  By default, PyVBMC assumes that the first argument to ``VBMC(log_joint, ...)`` is a function which returns the log-joint (i.e., the sum log-likelihood and log-prior). However, you may instead pass a function which returns the log-likelihood as a first argument, and supply the prior separately. In this case, the prior may be represented as a function which returns the log-density of the prior using the keyword argument ``log_prior``. ::

    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, log_prior=log_prior)

  Alternatively, using the keyword ``prior``, you can specify one of

  #. a PyVBMC prior object,
  #. an appropriate continuous ``scipy.stats`` distribution, or
  #. a list of univariate continuous ``scipy.stats`` distributions, which are treated as independent priors for each parameter :math:`\theta_{i}`.

  ::

    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, log_prior=scipy.stats.multivariate_normal(mu, cov))

  For more details on (1), see the documentation below as well as :ref:`PyVBMC Example 5: Prior distributions`. For more details on (2) and (3), see the documentation on :ref:`\`\`SciPy\`\` priors` below (``prior`` keyword arguments of these types be passed to this class for conversion).

``Prior`` base class
====================
These methods are common to all of the subclasses that follow.

.. autoclass:: pyvbmc.priors.Prior
  :members:

``SciPy`` priors
================

.. autoclass:: pyvbmc.priors.SciPy
  :special-members: __init__
  :members:
  :exclude-members: sample

Bounded priors
==============

``UniformBox``
--------------

.. autoclass:: pyvbmc.priors.UniformBox
  :special-members: __init__
  :members:
  :exclude-members: sample

``Trapezoidal``
---------------

.. autoclass:: pyvbmc.priors.Trapezoidal
  :special-members: __init__
  :members:
  :exclude-members: sample

``SplineTrapezoidal``
---------------------

.. autoclass:: pyvbmc.priors.SplineTrapezoidal
  :special-members: __init__
  :members:
  :exclude-members: sample

Unbounded priors
================

``SmoothBox``
-------------

.. autoclass:: pyvbmc.priors.SmoothBox
  :special-members: __init__
  :members:
  :exclude-members: sample
