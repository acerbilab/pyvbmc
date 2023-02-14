======
Priors
======

.. note::
  By default, PyVBMC assumes that the first argument to ``VBMC`` is a function which returns the log-joint (i.e., the sum of log-likelihood and log-prior). However, you may instead pass a function which returns the log-likelihood as a first argument, and supply the prior separately. Using the keyword ``log_prior``, you may pass a function (of a single argument) which returns the log-density of the prior given a point::

    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, log_prior=log_prior_function)

  Alternatively, using the keyword ``prior``, you may pass one of the following:

  #. a PyVBMC prior imported from ``pyvbmc.priors`` that matches the dimension of the model,
  #. an appropriate continuous ``scipy.stats`` distribution that matches the dimension of the model, or
  #. a list of univariate (i.e., one-dimensional) PyVBMC priors and/or continuous ``scipy.stats`` distributions, which are treated as independent priors for each parameter $\theta_{i}$. In this case the length of the list should equal the dimension of the model.

  ::

    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=UniformBox(0, 1, D=2))
    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=scipy.stats.multivariate_normal(mu, cov))
    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=[UniformBox(0, 1), scipy.stats.norm()])

  For more details on (1), see the documentation below as well as :ref:`PyVBMC Example 5: Prior distributions`. For more details on (2), (3), and using a function as a ``log_prior``, see the documentation on :ref:`\`\`SciPy\`\` priors`, :ref:`\`\`Product\`\` priors`, and :ref:`\`\`UserFunction\`\` priors` below. (keyword arguments of these types will be converted to instances of these classes).

``Prior`` base class
====================
These methods are common to all of the subclasses that follow.

.. autoclass:: pyvbmc.priors.Prior
  :members:

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

Other priors and functions
==========================

``Product`` priors
------------------

When a user passes a list of ``Prior`` instances or ``scipy.stats`` distributions, they are converted to a ``Product`` prior with a corresponding marginal distributions for each item in the list.

.. autoclass:: pyvbmc.priors.Product
  :special-members: __init__
  :members:
  :exclude-members: sample

``SciPy`` priors
----------------

To standardize the interface, when a user provides a ``scipy.stats`` distribution as a prior, it is wrapped in a ``SciPy`` prior distribution

.. autoclass:: pyvbmc.priors.SciPy
  :special-members: __init__
  :members:
  :exclude-members: sample

``UserFunction`` priors
-----------------------

To standardize the interface, when a user provides a function as a log-prior it is wrapped in a ``UserPrior`` class.

.. autoclass:: pyvbmc.priors.UserFunction
  :special-members: __init__
  :members:
  :exclude-members: sample, pdf

Utility functions
-----------------

.. autofunction:: pyvbmc.priors.convert_to_prior

.. autofunction:: pyvbmc.priors.tile_inputs

.. autofunction:: pyvbmc.priors.is_valid_scipy_dist
