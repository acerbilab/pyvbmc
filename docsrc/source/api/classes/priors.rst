======
Priors
======

.. note::
  By default, PyVBMC assumes that the first argument to ``VBMC`` is a function which returns the log-joint (i.e., the sum log-likelihood and log-prior). However, you may instead pass a function which returns the log-likelihood as a first argument, and supply the prior separately::

    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior)

  In this case, ``log_likelihood`` should be a function which returns the log-likelihood of your model given a point, and ``prior`` should be one of:

  #. a function of a single argument which returns the log-density of the prior given a point,
  #. a PyVBMC prior (one of the classes detailed here),
  #. an appropriate continuous ``scipy.stats`` distribution, or
  #. a list of one-dimensional PyVBMC priors and/or continuous ``scipy.stats`` distributions, which are treated as independent priors for each parameter :math:`\theta_{i}`.

  For more details on (2), see the documentation below as well as :ref:`PyVBMC Example 5: Prior distributions`. For more details on (1), (3), and (4), see the documentation on :ref:`\`\`UserFunction\`\` priors`, :ref:`\`\`SciPy\`\` priors`, and :ref:`\`\`Product\`\` priors` below (``prior`` keyword arguments of these types will be converted to instances of these classes).

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
