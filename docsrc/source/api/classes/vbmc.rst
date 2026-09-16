:tocdepth: 2

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

  The iteration history (``vbmc.iteration_history``) keeps each iteration's Gaussian process surrogate as its training data and hyperparameters, without the posterior factors; ``vbmc.get_gp(iteration)`` returns the surrogate of an iteration with its posterior restored, ready for prediction.

Reusing evaluations and charging initialization work
----------------------------------------------------

Use ``precomputed_evaluations`` to give VBMC target evaluations that are
already available when the run starts. These observations are independent of
the starting point and plausible box: they neither become additional rows of
``x0`` nor change the plausible bounds. Pass ``(X, y)`` for an exact target or
one with unknown noise, where ``X`` has shape ``(N, D)`` in the target's
original input coordinates and ``y`` has shape ``(N,)``::

  vbmc = VBMC(
      log_density,
      x0,
      lower_bounds,
      upper_bounds,
      plausible_lower_bounds,
      plausible_upper_bounds,
      precomputed_evaluations=(X, y),
  )

The values in ``y`` must be outputs of ``log_density`` at the corresponding
rows of ``X``. When ``prior`` or ``log_prior`` is supplied separately,
``log_density`` and ``y`` contain log-likelihood values; VBMC adds the
deterministic log-prior once. Precomputed points must lie strictly inside the
hard bounds, but they need not lie inside the plausible box.

For a target with user-provided noise estimates, set
``options["specify_target_noise"]`` and pass ``(X, y, y_sd)``. ``y_sd`` has
shape ``(N,)`` and contains positive standard deviations of the log-density
observations, not uncertainties in ``X``. Repeated rows of ``X`` remain
independent noisy observations and are pooled by inverse-variance weighting::

  vbmc = VBMC(
      noisy_log_density,
      x0,
      lower_bounds,
      upper_bounds,
      plausible_lower_bounds,
      plausible_upper_bounds,
      precomputed_evaluations=(X, y, y_sd),
      options={"specify_target_noise": True},
  )

Precomputed observations do not count as fresh target calls. By default they
also carry no cost for the current run, which is appropriate for evaluations
obtained during earlier work. If preparing this run incurred target or
derivative work, pass its nonnegative integer function-equivalent charge as
``initialization_cost``. The charge is independent of the number of retained
observations and counts once against the total ``max_fun_evals`` allowance::

  vbmc = VBMC(
      log_density,
      x0,
      lower_bounds,
      upper_bounds,
      plausible_lower_bounds,
      plausible_upper_bounds,
      precomputed_evaluations=(X, y),
      initialization_cost=13,
      options={"max_fun_evals": 100},
  )

Here VBMC can make at most 87 fresh target calls.
``results["func_count"]`` remains the literal number of fresh calls made by
VBMC. When ``initialization_cost`` is positive,
``results["evaluation_budget"]`` additionally reports the
function-equivalent ``limit``, ``initialization``, ``new_calls`` and ``used``
total. Zero-cost runs retain the ordinary result and display without this
additional budget report.

Performance calibration
-----------------------

New runs reuse compatible machine-local chunk settings produced by an
explicit :doc:`../functions/calibrate` campaign. If no compatible record
exists, they use historical defaults. Optimization never starts a campaign.
Use the ``performance_calibration`` option to select ``"cached"`` (default),
``"off"``, or an explicit :doc:`calibration_profile`.

Settings are fixed before the first relevant kernel call and survive final
boost and save/resume. A resolved save rejects load-time ``"cached"`` or
different explicit settings; start a new run to use a different profile.

Startup tips
------------

With ``show_tips=True`` (default), a new run may print one tip before the
iteration headings. Tips appear on the first eligible start and every third
eligible start thereafter, in shuffled order, with no repeats within a Python
session. Resumed or continued runs do not show another tip. A calibration
reminder takes priority, including one already shown by an early call on that
run's ``vbmc.vp``. Quiet starts, disabled tips and calibration reminders do
not advance the tip cadence. Set ``show_tips=False`` to disable tips, or
``display="off"`` to suppress both tips and calibration reminders. These
settings belong in the ``options`` dictionary passed to ``VBMC``.

.. autoclass:: pyvbmc.VBMC
   :exclude-members: optimize, save, load
   :members:

   .. autofunction:: pyvbmc.VBMC.optimize
   .. autofunction:: pyvbmc.VBMC.save
   .. autofunction:: pyvbmc.VBMC.load

Additional functions
--------------------

.. autofunction:: pyvbmc.vbmc.optimize_vp
.. autofunction:: pyvbmc.vbmc.train_gp
.. autofunction:: pyvbmc.vbmc.update_K
