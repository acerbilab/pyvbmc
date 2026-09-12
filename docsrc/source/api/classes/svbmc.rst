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

:ref:`PyVBMC Example 7: Stacking the posteriors of several runs (S-VBMC)`
walks through this on a bimodal target.

Construction filters the input runs: a run is discarded when VBMC did not
mark it as converged (``vp.stats["stable"]``) or when the standard deviation
of the expected log-joint of one of its components reaches ``s_max``. An
error is raised when fewer runs survive than ``M_min`` requires (a count
when greater than 1, a proportion of the input runs otherwise). ``optimize()`` works
in place and returns ``None``: the optimized weights are in ``stacked.w``,
the entropy estimate in ``stacked.entropy``, the headline ELBO in the scalar
``stacked.elbo`` and its reported uncertainty in ``stacked.elbo_sd``. The
``version`` argument selects optimizing every component weight
(``"all-weights"``, the default), one weight per run
(``"posterior-only"``), or no optimization at all (``"ns"``, naive
stacking). ``optimize()`` uses 20 entropy draws per component during
optimization, an Adam learning rate of 0.1 and at most 500 steps by default.
It then performs a fresh evaluation at the selected weights with
``n_samples_final=100`` draws per component. ``n_samples_final`` is
keyword-only and must be an integer of at least 2.

ELBO reporting
--------------

For a stack treated as noiseless, ``stacked.elbo`` is the raw ELBO from the
fresh final evaluation. If any retained run is treated as noisy, the headline
is ``capped_I_median``: its expected log-joint is capped at the median value
of the retained components. This cap reduces one source of optimistic bias;
it does not remove every source of bias.

``stacked.elbo_details`` contains the complete report:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Meaning
   * - ``raw``
     - Uncapped ELBO from the fresh evaluation at the selected weights.
   * - ``capped_I_median``
     - ELBO with the expected log-joint capped at the component median.
   * - ``capped_E_median``
     - ELBO with the expected log-joint capped at the retained-run median.
   * - ``naive``
     - Baseline that gives every retained run equal total weight and keeps
       that run's original internal component weights. It inherits errors in
       the input runs and is not a bound on the optimized ELBO.
   * - ``headline_method``
     - ``"raw"`` for a noiseless stack or ``"capped_I_median"`` for a noisy
       stack.
   * - ``cap_amount``
     - Reduction applied to the headline. It is zero for a noiseless stack,
       even when a diagnostic cap would bind.
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
describes the uncapped ``raw`` value. It excludes selection bias and does not
propagate the median cap, so it must not be interpreted as defining a
calibrated confidence interval for a capped headline.

By default, the keyword-only constructor argument ``noisy=None`` reads the
noise status recorded by each retained posterior. Older posteriors without
that metadata are classified as noisy when ``vp.stats["elbo_sd"] > 0.1``;
``noise_status_source`` then records ``"inferred"`` because this fallback can
misclassify a run. Pass ``noisy=True`` or ``noisy=False`` to override the
entire stack; the source for every retained run is then ``"override"``.

The constructor defaults are ``s_max=np.sqrt(5)``, ``M_min=2/3``, ``seed=None``,
``show_tips=True`` and ``noisy=None``. Guidance tips appear only when
``show_tips`` is true and INFO logging is enabled. Setting ``show_tips=False``
suppresses tips while leaving progress and applied-cap diagnostics available.

Code written for the previous dictionary-valued ``elbo`` API can be migrated
as follows:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Previous expression
     - Current expression
   * - ``stacked.elbo["estimated"]``
     - ``stacked.elbo_details["raw"]``
   * - ``stacked.elbo["debiased_I_median"]``
     - ``stacked.elbo_details["capped_I_median"]``
   * - ``stacked.elbo["debiased_E_median"]``
     - ``stacked.elbo_details["capped_E_median"]``

Use the scalar ``stacked.elbo`` when the automatically selected headline is
the desired result.

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

Sampling
--------

``sample(n)`` returns ``n`` independent draws from the stacked posterior in
the original space: the number of draws from each run is multinomial in the
runs' total weights, the component within a run is chosen at random by
weight, and the rows are shuffled, so any subset of rows is itself a valid
sample. ``sample(n, balance_flag=True)`` instead splits the draws across
runs and components in proportion to the weights, every count within one
draw of its exact share: a stratified sample with lower variance for
expectations, not an independent one.

All numerical draws of an ``SVBMC`` object come from its own generator, set
by the ``seed`` argument; the input posteriors are never modified or advanced.
As for ``VBMC``, ``seed=None`` derives the generator from NumPy's global
state, so ``np.random.seed`` before construction still fixes a run.

Helpers
-------

:mod:`pyvbmc.svbmc.utils` holds two helpers from the S-VBMC workflow:
``overlay_corner_plot`` draws several sample sets on one corner plot, to
compare the individual runs with the stacked posterior, and
``find_init_bounds`` returns the box to draw VBMC starting points from
given the bounds and plausible bounds of a problem.

Citation
--------

Silvestrin, F., Li, C., & Acerbi, L. (2025). Stacking Variational Bayesian
Monte Carlo. *Transactions on Machine Learning Research*.
`arXiv:2504.05004 <https://arxiv.org/abs/2504.05004>`_,
`TMLR <https://openreview.net/forum?id=M2ilYAJdPe>`_. Please cite it
together with the VBMC and PyVBMC papers when you use S-VBMC.

The implementation is the S-VBMC package (``acerbilab/svbmc``) integrated
into PyVBMC; its BSD 3-Clause license notice ships with the subpackage.

.. autoclass:: pyvbmc.svbmc.SVBMC
   :members:

.. automodule:: pyvbmc.svbmc.utils
   :members:
