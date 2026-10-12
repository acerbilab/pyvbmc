======
Priors
======

.. note::
  By default, PyVBMC expects the first argument to ``VBMC`` to return the log
  joint: the sum of the log likelihood and log prior. To pass a log-likelihood
  function instead, supply the prior separately. The ``log_prior`` argument
  accepts a function of one point that returns its prior log density::

    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, log_prior=log_prior_function)

  Alternatively, ``prior`` accepts:

  #. a PyVBMC prior from ``pyvbmc.priors`` with the model dimension;
  #. a compatible continuous ``scipy.stats`` distribution with the model
     dimension; or
  #. a list of one-dimensional PyVBMC priors or continuous ``scipy.stats``
     distributions, one per parameter. The list entries are treated as
     independent.

  ::

    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=UniformBox(0, 1, D=2))
    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=scipy.stats.multivariate_normal(mu, cov))
    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=[UniformBox(0, 1), scipy.stats.norm()])

  The prior support must cover the hard bounds ``lb`` and ``ub``; the bounds
  may coincide with the ends of the support. For example,
  ``UniformBox(0, 1, D=2)`` permits hard bounds on or inside the unit square.
  Within the hard bounds, VBMC uses the prior as given and does not renormalize
  it.

  See :ref:`PyVBMC Example 5: Prior distributions` and the ``SciPy``,
  ``Product`` and ``UserFunction`` sections below for details. PyVBMC wraps
  these inputs in the corresponding prior classes.

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

When a user passes a list of ``Prior`` instances or ``scipy.stats``
distributions, PyVBMC converts it to a ``Product`` prior with one marginal
distribution per item.

.. autoclass:: pyvbmc.priors.Product
  :special-members: __init__
  :members:
  :exclude-members: sample

``SciPy`` priors
----------------

When a user provides a ``scipy.stats`` distribution as a prior, PyVBMC wraps
it in a ``SciPy`` prior.

.. autoclass:: pyvbmc.priors.SciPy
  :special-members: __init__
  :members:
  :exclude-members: sample

``UserFunction`` priors
-----------------------

To standardize the interface, when a user provides a function as a log-prior it is wrapped in a ``UserFunction`` prior.

.. autoclass:: pyvbmc.priors.UserFunction
  :special-members: __init__
  :members:
  :exclude-members: sample, pdf

Utility functions
-----------------

.. autofunction:: pyvbmc.priors.convert_to_prior

.. autofunction:: pyvbmc.priors.tile_inputs

.. autofunction:: pyvbmc.priors.is_valid_scipy_dist
