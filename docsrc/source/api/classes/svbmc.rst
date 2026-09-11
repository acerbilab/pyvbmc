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
   print(stacked.elbo["estimated"])         # optimized stacked ELBO
   print(stacked.elbo["debiased_I_median"]) # debiased estimate

Construction filters the input runs: a run is discarded when VBMC did not
mark it as converged (``vp.stats["stable"]``) or when the standard deviation
of the expected log-joint of one of its components reaches ``s_max``. An
error is raised when fewer than ``M_min`` runs survive. ``optimize()`` works
in place and returns ``None``: the optimized weights are in ``stacked.w``,
the entropy estimate in ``stacked.entropy`` and the estimated and debiased
stacked ELBO values in ``stacked.elbo``. The ``version`` argument selects
optimizing every component weight (the default), one weight per run, or no
optimization at all (naive stacking).

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
runs and components in exact proportion to the weights, a stratified sample
with lower variance for expectations that is not an independent sample.

All randomness of an ``SVBMC`` object comes from its own generator, set by
the ``seed`` argument; the input posteriors are never modified or advanced.
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
