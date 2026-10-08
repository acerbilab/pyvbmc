:tocdepth: 2

=========
``SVBMC``
=========

.. note::

  The ``SVBMC`` class implements Stacking Variational Bayesian Monte Carlo
  (S-VBMC): it combines the variational posteriors of several completed
  VBMC runs on the same problem into one *stacked posterior* by optimizing
  the stacked ELBO with respect to the mixture weights only. The runs'
  components and their stored expected log-joint statistics stay fixed, so
  stacking needs no further evaluations of the model. Use it when repeated
  VBMC runs on the same model and data disagree, or each captures part of a
  multimodal posterior.

  S-VBMC needs the optional ``torch`` extra (see :doc:`../../installation`);
  constructing an ``SVBMC`` object without it raises an ``ImportError``.
  Importing PyVBMC does not import torch.

Basic usage
-----------

Run VBMC several times on the same target, collect the returned posteriors,
and stack them:

.. code-block:: python

   from pyvbmc import VBMC, SVBMC

   vps = []
   for seed in range(5):
       vbmc = VBMC(log_joint, x0[seed], lb, ub, plb, pub, seed=seed)
       vp, results = vbmc.optimize()
       vps.append(vp)

   stacked = SVBMC(vps, seed=0)
   stacked.optimize()

   samples = stacked.sample(10000)          # (10000, D), original space
   print(stacked.elbo)                       # headline stacked ELBO
   print(stacked.elbo_sd)                    # uncertainty of the raw estimate
   print(stacked.elbo_details["raw"])
   print(stacked.elbo_details["shrunk_two_level"])

:ref:`PyVBMC Example 7: Stacking the posteriors of several runs (S-VBMC)`
walks through this on a bimodal target.

.. The S-VBMC runtime tip noisy_elbo (pyvbmc/svbmc/_tip_catalog.py) links
   this label.

.. _svbmc-elbo-reporting:

ELBO reporting
--------------

``stacked.elbo`` is the headline estimate of the stacked ELBO, chosen by the
noise status of the stack:

- For a stack treated as noiseless, it is ``raw``, the ELBO from the fresh
  final evaluation at the selected weights.
- If any retained run is treated as noisy, it is ``shrunk_two_level``, the
  two-level shrinkage estimate at the same weights and with the same final
  entropy, whatever the ``version`` of the optimization. Where that
  calculation is numerically undefined, the headline is ``capped_I_median``
  instead, and ``optimize()`` warns.

The choice affects the reported value alone: the weights, and so the stacked
posterior and its samples, come from the same optimization whatever the
headline.

The weights maximize the weighted sum of the components' expected log-joints,
each estimated by its run's GP, plus the entropy of the mixture. On a noisy target,
this optimization favors components whose estimates came out high and adds
an upward bias to the raw ELBO. Here bias is measured against the stack's
ELBO computed with the exact log joint, which is itself a lower bound on the
log evidence.

The two-level shrinkage estimate targets this selection bias. It first
shrinks component expected log-joints toward the mean within each run, using
their GP uncertainty, and then shrinks the runs' levels toward their common
mean. It can also remove some bias inherited from the input VBMC runs, but it
does not make the headline unbiased: miscalibrated GP uncertainty and
different run-level biases can leave an error in either direction. The raw
ELBO can likewise fall below the stack's exact ELBO when the input runs are
pessimistic.

On our benchmarks with every component weight optimized, the remaining
stacking optimism and the inherited bias removed by shrinkage roughly
balanced, so noisy-stack headlines were about as biased as the input runs.
With ``version="ns"``, stacking adds no weight-selection bias; shrinkage
still applies and can move the headline below the raw value.

``stacked.elbo_details`` contains the complete report:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Meaning
   * - ``raw``
     - ELBO from the fresh evaluation at the selected weights; the headline
       of a noiseless stack.
   * - ``capped_I_median``
     - ELBO with the expected log-joint capped at the component median; the
       headline of a noisy stack whose shrinkage is unavailable.
   * - ``capped_E_median``
     - ELBO with the expected log-joint capped at the retained-run median.
   * - ``naive``
     - Baseline that gives every retained run equal total weight and keeps
       that run's original internal component weights. It inherits errors in
       the input runs and is not a bound on the optimized ELBO.
   * - ``shrunk_two_level``
     - Empirical-Bayes estimate that first shrinks component expected
       log-joints within each retained run using their full GP covariance,
       then shifts each run's components by the change that shrinkage makes
       to the run's level; the headline of a noisy stack. It uses the
       selected stacking weights and the same final entropy as ``raw``.
       ``None`` means finite input statistics led
       to a numerically undefined calculation;
       :meth:`~pyvbmc.svbmc.SVBMC.optimize` emits a ``RuntimeWarning`` and
       leaves every other report value available.
   * - ``shrinkage_noise_share``
     - Diagnostic ratio of estimated noise to component spread, averaged over
       runs using their selected mixture masses. It can exceed one and is
       ``None`` when shrinkage is unavailable or no run with positive spread
       has positive selected mass.
   * - ``headline_method``
     - The key of the headline: ``"raw"`` for a noiseless stack,
       ``"shrunk_two_level"`` for a noisy stack, or ``"capped_I_median"``
       for a noisy stack whose shrinkage is unavailable.
   * - ``cap_amount``
     - Reduction that the median cap applied to the headline, ``raw`` less
       ``capped_I_median``. It is zero unless the headline is
       ``capped_I_median``, even when a diagnostic cap would bind.
   * - ``entropy_sd``
     - Monte Carlo contribution to the uncertainty of ``raw``.
   * - ``gp_sd``
     - GP quadrature contribution, treating retained runs as independent.
   * - ``raw_sd``
     - Combined raw-estimate uncertainty, equal to ``stacked.elbo_sd``.
   * - ``noisy``
     - Whether the stack is treated as noisy.
   * - ``noise_status_source``
     - Per-retained-run provenance: ``"recorded"``, ``"inferred"`` or
       ``"override"``.

``stacked.elbo_sd`` combines the stratified entropy estimator's Monte Carlo
variance with the GP quadrature uncertainty at the selected weights. It
describes the ``raw`` value. It excludes selection bias and does not
propagate the shrinkage or the median cap, so it must not be interpreted as
defining a calibrated confidence interval for the headline of a noisy stack.

A composite posterior
---------------------

Each VBMC run works in its own transformed parameter space (bounded
parameters, and possibly a rotation and rescaling learned during the run).
The components of different runs therefore live in incompatible spaces and
are only comparable after mapping back to the original space. The stacked
posterior keeps the input posteriors as they are and draws from each through
its own transform. Do not combine the component means and covariances of
different runs directly, and do not read them as a single mixture: use
``sample()`` for every estimate and plot.

.. _svbmc-helpers:

Helpers
-------

:mod:`pyvbmc.svbmc.utils` holds two helpers from the S-VBMC workflow:
``overlay_corner_plot`` draws several sample sets on one corner plot, to
compare the individual runs with the stacked posterior, and
``find_init_bounds`` returns the box to draw VBMC starting points from
given the bounds and plausible bounds of a problem.

Differences from the standalone ``svbmc`` package
-------------------------------------------------

The implementation derives from the standalone S-VBMC package
(``acerbilab/svbmc``, version 0.1.1), which is deprecated in its favor: use
``pyvbmc.SVBMC`` for new and existing code. The package's BSD 3-Clause
license notice ships with the subpackage, ``pyvbmc.svbmc``. The method and
the optimizer of the weights are the same, and on our benchmark the two
reach the same weights. What differs:

- It is faster. The work that the components of one run share, its parameter
  transform above all, is done once per run rather than once per component.
- The reported ELBO is more precise. The corrections that bring the expected
  log-joint of each run to the original parameter space are computed once,
  when the object is created, exactly or by numerical quadrature, where the
  standalone package estimated them by Monte Carlo at every step. After the
  optimization the ELBO is evaluated again with more draws
  (``optimize(n_samples_final=...)``), and it comes with an uncertainty,
  ``elbo_sd``.
- ``elbo`` is a number, the headline described under
  :ref:`svbmc-elbo-reporting`, which for a noisy stack is a shrinkage
  estimate that the standalone package does not compute. The entries of the
  standalone package's ``elbo`` dictionary are in ``elbo_details``, under the
  names in the table below.
- ``seed`` takes the place of ``testing``: every random draw comes from the
  generator of the ``SVBMC`` object, and the input posteriors are neither
  modified nor advanced.
- ``sample(n)`` returns exactly ``n`` draws. The standalone package rounded
  the share of each run separately and could return a few more or fewer.
- Results are float64, problems with one variable work, and malformed inputs
  raise errors.

Code written for the standalone package's dictionary-valued ``elbo`` can be
migrated as follows:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Standalone ``svbmc``
     - ``pyvbmc.svbmc``
   * - ``stacked.elbo["estimated"]``
     - ``stacked.elbo_details["raw"]``
   * - ``stacked.elbo["debiased_I_median"]``
     - ``stacked.elbo_details["capped_I_median"]``
   * - ``stacked.elbo["debiased_E_median"]``
     - ``stacked.elbo_details["capped_E_median"]``

Citation
--------

Silvestrin, F., Li, C., & Acerbi, L. (2025). Stacking Variational Bayesian
Monte Carlo. *Transactions on Machine Learning Research*.
`arXiv:2504.05004 <https://arxiv.org/abs/2504.05004>`_,
`TMLR <https://openreview.net/forum?id=M2ilYAJdPe>`_. Please cite it
together with the VBMC and PyVBMC papers when you use S-VBMC.

.. autoclass:: pyvbmc.svbmc.SVBMC
   :members:

.. automodule:: pyvbmc.svbmc.utils
   :members:
