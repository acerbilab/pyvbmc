***************
Getting started
***************

The best way to get started with PyVBMC is via the tutorials and worked examples.
In particular, start with :ref:`PyVBMC Example 1: Basic usage` and continue from there.

If you are already familiar with approximate inference methods, you can find a summary usage below.

Summary usage
=============

The typical usage pipeline of PyVBMC follows four steps:

1. Define the model, which defines a target log density (i.e., an unnormalized log posterior density);
2. Setup the parameters (parameter bounds, starting point);
3. Initialize and run the inference;
4. Examine and visualize the results.

PyVBMC is not concerned with how you define your model in step 1, as long as you can provide an (unnormalized) target log density.
Running the inference in step 3 only involves a couple of lines of code:

.. code-block:: python

  from pyvbmc import VBMC
  # ...
  vbmc = VBMC(target, x0, LB, UB, PLB, PUB)
  vp, results = vbmc.optimize()

with input arguments:

- ``target``: the target (unnormalized) log density — often an unnormalized log posterior. ``target`` takes as input a parameter vector and returns the log density at the point. The returned log density must return a *finite* real value, i.e. non `NaN` or `-inf`. See the :labrepos:`VBMC FAQ <vbmc/wiki#how-do-i-prevent-vbmc-from-evaluating-certain-inputs-or-regions-of-input-space>` for more details;
- ``x0``: the starting point of the inference in parameter space;
- ``LB`` and ``UB``: hard lower and upper bounds for the parameters (can be ``-inf`` and ``inf``, or bounded);
- ``PLB`` and ``PUB``: *plausible* lower and upper bounds, that is a box that ideally brackets a region of high density of the target.

The outputs are:

- ``vp``: a ``VariationalPosterior`` object which approximates the true target density;
- ``results``: a ``dict`` with additional information. Important keys are:
  - ``"elbo"``: the estimated lower bound on the log model evidence (log normalization constant);
  - ``"elbo_sd"``: the standard deviation of the estimate of the ELBO (*not* the error between the ELBO and the true log model evidence, which is generally unknown).

The ``vp`` object can be manipulated in various ways, see the :ref:`\`\`VariationalPosterior\`\`` class documentation.

See the examples for more detailed information. The :ref:`Basic options` may also be useful.

Bring a torch or JAX model into PyVBMC
======================================

The default target interface calls one point at a time. It accepts a
one-dimensional NumPy parameter vector and returns one finite scalar. A model
implemented in torch can be adapted without adding torch objects to PyVBMC.
PyVBMC does not install either modelling framework; install the one used by
your model separately::

  import numpy as np
  import torch
  from pyvbmc import VBMC
  from scipy.stats import norm

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  observations_t = torch.tensor(
      [0.4, -0.2, 0.7], dtype=torch.float64, device=device
  )

  def torch_log_likelihood(x):
      theta_t = torch.as_tensor(x, dtype=torch.float64, device=device)
      value_t = -0.5 * torch.sum((observations_t - theta_t[0]) ** 2)
      return np.asarray(
          value_t.detach().cpu().numpy(), dtype=np.float64
      ).item()

  x0 = np.array([0.0], dtype=np.float64)
  lb, ub = np.array([-10.0]), np.array([10.0])
  plb, pub = np.array([-2.0]), np.array([2.0])
  vbmc = VBMC(
      torch_log_likelihood,
      x0,
      lb,
      ub,
      plb,
      pub,
      prior=norm(loc=0.0, scale=2.0),
  )

``detach().cpu().numpy()`` completes device execution and returns the scalar
to the host. This also makes PyVBMC's target timing cover the framework work.
The explicit ``torch.float64`` matches PyVBMC's numerical precision.

For JAX, enable 64-bit values before importing ``jax.numpy`` or constructing
the model. ``jax.device_get`` completes the computation and moves its result
to the host::

  from jax import config
  config.update("jax_enable_x64", True)

  import jax
  import jax.numpy as jnp
  import numpy as np

  observations_j = jnp.array([0.4, -0.2, 0.7], dtype=jnp.float64)

  def jax_log_likelihood(x):
      theta_j = jnp.asarray(x, dtype=jnp.float64)
      value_j = -0.5 * jnp.sum((observations_j - theta_j[0]) ** 2)
      return np.asarray(
          jax.device_get(value_j), dtype=np.float64
      ).item()

Vectorized targets
------------------

Set ``options={"vectorized_target": True}`` when the likelihood can evaluate
a batch. PyVBMC collects all initial-design rows whose values were not supplied
and evaluates them in one target call. All subsequent target calls still
contain one point, with shape ``(1, D)``. The input is always a NumPy array in
original coordinates and the target must return a finite NumPy array with
shape ``(N,)`` or ``(N, 1)``. PyVBMC does not probe a target to determine
whether it supports batches. A scalar return is invalid even when ``N`` is
one.

The torch adapter above becomes::

  def torch_vectorized_log_likelihood(x):
      theta_t = torch.as_tensor(x, dtype=torch.float64, device=device)
      values_t = -0.5 * torch.sum(
          (observations_t[None, :] - theta_t[:, :1]) ** 2, dim=1
      )
      return np.asarray(
          values_t.detach().cpu().numpy(), dtype=np.float64
      )

  vbmc = VBMC(
      torch_vectorized_log_likelihood,
      x0,
      lb,
      ub,
      plb,
      pub,
      prior=norm(loc=0.0, scale=2.0),
      options={"vectorized_target": True},
  )

The equivalent JAX adapter is::

  def jax_vectorized_log_likelihood(x):
      theta_j = jnp.asarray(x, dtype=jnp.float64)
      values_j = -0.5 * jnp.sum(
          (observations_j[None, :] - theta_j[:, :1]) ** 2, axis=1
      )
      return np.asarray(jax.device_get(values_j), dtype=np.float64)

A separately supplied prior keeps its scalar interface. PyVBMC evaluates it
one point at a time and combines it with the batched likelihood; custom prior
functions therefore do not need to accept a batch.

For a target with user-provided noise estimates, set
``"specify_target_noise": True`` and return a pair of arrays. Each array must
have shape ``(N,)`` or ``(N, 1)``; the standard deviations must be finite and
positive. Do not combine the two outputs into an ``(N, 2)`` array. For
example::

  def torch_noisy_log_likelihood(x):
      values = torch_vectorized_log_likelihood(x)
      sds = np.full(values.shape, 0.1, dtype=np.float64)
      return values, sds

  def jax_noisy_log_likelihood(x):
      values = jax_vectorized_log_likelihood(x)
      sds_j = jnp.full(values.shape, 0.1, dtype=jnp.float64)
      sds = np.asarray(jax.device_get(sds_j), dtype=np.float64)
      return values, sds

  noisy_options = {
      "vectorized_target": True,
      "specify_target_noise": True,
  }

Pass ``noisy_options`` as the ``options`` argument when constructing
``VBMC`` with either noisy adapter.

Cached initial values can be mixed with evaluations through
:meth:`~pyvbmc.function_logger.FunctionLogger.batch_call`; ``NaN`` marks the
rows that still need evaluation. Results and cache indices retain input row
order.

Use a fitted posterior downstream
=================================

Torch distribution
------------------

:meth:`~pyvbmc.VariationalPosterior.to_torch` returns an independent torch
distribution snapshot::

  posterior_t = vp.to_torch()
  torch.manual_seed(7)
  samples_t = posterior_t.sample((1000,))

  point_t = samples_t[0].detach().clone().requires_grad_(True)
  gradient_t = torch.autograd.grad(posterior_t.log_prob(point_t), point_t)[0]

The default export uses original parameter coordinates, CPU tensors, and
``torch.float64``, regardless of torch's global defaults. Pass
``orig_flag=False`` for PyVBMC's unbounded internal coordinates, or specify
``dtype=torch.float32`` and ``device=...`` explicitly when needed. The export
copies all parameters: later changes to either object do not affect the other.
Conversion makes no random draws and does not alter NumPy's or torch's random
state; sampling the returned distribution uses torch's random generator.

For bounded parameters, ``log_prob`` accepts finite values strictly inside
the hard bounds. Support validation rejects exact-bound and outside values.
Samples remain strictly inside representable bounds. The distribution has
event shape ``(D,)`` and accepts arbitrary leading sample dimensions.

ArviZ DataTree
--------------

:meth:`~pyvbmc.VariationalPosterior.to_arviz` draws independent samples into
the current ArviZ DataTree format::

  import arviz as az

  posterior_data = vp.to_arviz(
      n_samples=2000, var_names=["location"]
  )
  summary = az.summary(posterior_data, group="posterior")
  axes = az.plot_dist(posterior_data, group="posterior")

The result contains one ``posterior`` group, one chain, and one scalar
variable per parameter. Names default to ``x_0``, ..., ``x_{D-1}``. Pass
``orig_flag=False`` to export samples in internal coordinates.

This method advances ``vp.rng`` in the same way as
:meth:`~pyvbmc.VariationalPosterior.sample`. The draws are independent samples
from a variational approximation, so MCMC convergence diagnostics computed on
them do not measure the quality of that approximation. Use posterior summaries
and plots, and assess PyVBMC convergence from its own results and repeated
runs.
