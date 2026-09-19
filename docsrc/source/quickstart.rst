***************
Getting started
***************

The best way to get started with PyVBMC is via the tutorials and worked examples.
In particular, start with :ref:`PyVBMC Example 1: Basic usage` and continue from there.

Optionally, we recommend running ``pyvbmc.calibrate()`` once to tune performance
for your machine (memory, processor, etc.); it takes tens of seconds and can
make PyVBMC run faster.
Saved calibration settings are reused automatically by future runs (see the
:doc:`calibration guide <api/functions/calibrate>`).

If you are already familiar with approximate inference methods, you can find a summary usage below.

Summary usage
=============

The typical usage pipeline of PyVBMC follows four steps:

1. Define the model, which defines a target log density (i.e., an unnormalized log posterior density);
2. Set up the parameters (parameter bounds, starting point);
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

- ``target``: the target (unnormalized) log density — often an unnormalized log posterior. ``target`` takes as input a parameter vector and returns the log density at the point. The returned log density must be a *finite* real value, i.e. neither ``NaN`` nor ``+/-inf``. See the :ref:`FAQ <faq-how-do-i-prevent-vbmc-from-evaluating-certain-inputs-or-regions-of-input-space>` for more details;
- ``x0``: the starting point of the inference in parameter space;
- ``LB`` and ``UB``: hard lower and upper bounds for the parameters (can be ``-inf`` and ``inf``, or bounded);
- ``PLB`` and ``PUB``: *plausible* lower and upper bounds, that is a box that ideally brackets a region of high density of the target.

You can also pass a log-likelihood as ``target`` and supply its prior
separately with ``prior=`` or ``log_prior=``. For model evidence, include
the likelihood's normalization constants and use a normalized prior; see
the :ref:`FAQ <faq-how-do-i-specify-the-prior-to-vbmc>`.

The outputs are:

- ``vp``: a ``VariationalPosterior`` object which approximates the true target density;
- ``results``: a ``dict`` with additional information. Important keys are:
  - ``"elbo"``: the estimated lower bound on the log model evidence (log normalization constant);
  - ``"elbo_sd"``: the standard deviation of the estimate of the ELBO (*not* the error between the ELBO and the true log model evidence, which is generally unknown).

The ``vp`` object can be manipulated in various ways, see the :ref:`\`\`VariationalPosterior\`\`` class documentation.

See the examples for more detailed information. The :ref:`Basic options` may also be useful.

For noisy likelihoods, see :ref:`PyVBMC Example 6: Noisy log-likelihood evaluations` and
the :ref:`FAQ <faq-noisy-target-function>` for the target interface and
noise options.

PyVBMC occasionally prints a tip when a new run starts. Tips appear at most
once each within a Python session; restarting Python resets their history.
Pass ``options={"show_tips": False}`` to ``VBMC`` to disable tips. This leaves
the performance-calibration reminder enabled; ``options={"display": "off"}``
suppresses both along with ordinary optimization output.

Reproducible runs
=================

Pass ``seed=`` to control the random generator used by a VBMC instance
and its variational posterior. The same integer seed reproduces a run
with the same target, setup and numerical environment. Reusing the same
integer seed repeats PyVBMC's random stream. For independent runs, leave
``seed`` unset or use different seeds; comparing their results helps assess convergence
(see :ref:`PyVBMC Example 4: Multiple runs as validation`).

The seed can also be a NumPy ``Generator``. PyVBMC advances that generator,
so reusing it continues its random stream.

If your likelihood function uses random simulations, ``VBMC(seed=...)``
does not control the randomness inside that function. To reproduce the
whole analysis, also set the simulator's seed before starting the run.
For example, with NumPy, create ``sim_rng = np.random.default_rng(123)``
once outside your target function, then use that generator for its random
draws (e.g., ``sim_rng.normal(...)`` instead of ``np.random.normal(...)``).
If you use another simulation library, use its seed or generator argument.
Let the simulator draw fresh random numbers on each evaluation; do not
reset its seed inside the target function.

The :ref:`FAQ <faq-i-have-been-running-vbmc-from-the-same-starting-point-but-i-get-different-results-each-time-is-something-wrong>`
explains reproducibility and the behavior when ``seed`` is omitted.

Bring a torch or JAX model into PyVBMC
======================================

The default target interface calls one point at a time. It accepts a
one-dimensional NumPy parameter vector and returns one finite scalar. A model
implemented in Torch can be connected with a small user-written wrapper that
keeps framework objects outside PyVBMC.
The example below uses independent Gaussian observations with unit standard
deviation and a Gaussian prior on their common mean. Install the modelling
framework used by your model separately::

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
      value_t = -0.5 * torch.sum(
          (observations_t - theta_t[0]) ** 2 + np.log(2 * np.pi)
      )
      return np.asarray(
          value_t.detach().cpu().numpy(), dtype=np.float64
      ).item()

  x0 = np.array([0.0], dtype=np.float64)
  lb, ub = np.array([-np.inf]), np.array([np.inf])
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
      value_j = -0.5 * jnp.sum(
          (observations_j - theta_j[0]) ** 2 + jnp.log(2 * jnp.pi)
      )
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

The Torch wrapper above becomes::

  def torch_vectorized_log_likelihood(x):
      theta_t = torch.as_tensor(x, dtype=torch.float64, device=device)
      values_t = -0.5 * torch.sum(
          (observations_t[None, :] - theta_t[:, :1]) ** 2
          + np.log(2 * np.pi),
          dim=1,
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

The equivalent JAX wrapper is::

  def jax_vectorized_log_likelihood(x):
      theta_j = jnp.asarray(x, dtype=jnp.float64)
      values_j = -0.5 * jnp.sum(
          (observations_j[None, :] - theta_j[:, :1]) ** 2
          + jnp.log(2 * jnp.pi),
          axis=1,
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
``VBMC`` with either noisy wrapper.

To reuse cached target values, pass ``precomputed_evaluations=(X, y)`` to
``VBMC``. Each value in ``y`` must be the output of the target supplied to
``VBMC`` at the corresponding row of ``X``. When a prior is supplied
separately, the target and ``y`` contain the log-likelihood and PyVBMC adds the
prior; when the target is a log joint, ``y`` contains that log joint. See
:ref:`Reusing evaluations and charging initialization work` for shapes,
noisy observations and budget accounting.

For a complete Torch fit, an equivalent short JAX target, Torch posterior
predictions and density gradients, and ArviZ summaries, see
:ref:`PyVBMC Example 9: Torch and JAX models and posterior exports`.

Bring a PyMC model into PyVBMC
==============================

Install the optional integration on Python 3.12 or newer with
``python -m pip install "pyvbmc[pymc]"``. A :class:`~pyvbmc.pymc.PyMCTarget`
provides the flat target and setup that PyVBMC needs. This model has a named
coefficient vector and a positive observation scale:

.. code-block:: python

  import numpy as np
  import pymc as pm
  from pyvbmc import PyMCTarget, VBMC
  rng = np.random.default_rng(7)
  x_data = np.linspace(-1.0, 1.0, 30)
  design = np.column_stack((np.ones(x_data.size), x_data))
  observations = design @ np.array([0.5, 1.0])
  observations += rng.normal(0.0, 0.7, size=x_data.size)
  coords = {
      "observation": np.arange(x_data.size),
      "coefficient": ["intercept", "slope"],
  }
  with pm.Model(coords=coords) as model:
      X = pm.Data("X", design, dims=("observation", "coefficient"))
      beta = pm.Normal("beta", 0.0, 2.0, dims="coefficient")
      sigma = pm.HalfNormal("sigma", 1.0)
      mean = pm.Deterministic("mean", X @ beta, dims="observation")
      pm.Normal("y", mean, sigma, observed=observations, dims="observation")
  target = PyMCTarget(model, seed=7)
  print(target)
  vbmc = VBMC(
      target,
      seed=7,
      options={"max_fun_evals": target.setup_cost + 60},
  )
  vp, results = vbmc.optimize()
  posterior_data = target.to_arviz(vp, n_samples=1000)
  posterior_data = pm.compute_deterministics(
      posterior_data, model=model, extend_dataset=True
  )
  predictions = pm.sample_posterior_predictive(
      posterior_data, model=model, var_names=["y"], random_seed=7
  )

``print(target)`` lists each model variable, its shape, its flat VBMC
coordinates, hard and plausible bounds, and its start. Here ``beta`` becomes
``beta[intercept]`` and ``beta[slope]``. The positive ``sigma`` keeps PyMC's
log transform, so its VBMC coordinate is ``sigma_log__`` and the target
includes the corresponding Jacobian.

.. code-block:: text

  variable | shape | VBMC coordinates | kept transform | hard bounds | plausible bounds | start
  beta | (2,) | beta[intercept], beta[slope] | - | [[-inf -inf], [inf inf]] | [[-0.09967958  0.24424391], [0.5182185 1.2769416]] | [0.20926946 0.76059275]
  sigma | () | sigma_log__ | LogTransform | [[-inf], [inf]] | [[-0.96092139], [-0.18161372]] | [-0.57126755]

The adapter computes a mode and a Laplace plausible box by default. Unusable
curvature falls back to prior widths; a mode outside the prior's location
interval produces a diagnostic and is not moved. Finite values obtained during
setup are saved in ``target.setup_evaluations`` and reused by ``VBMC(target)``.
They supplement the ordinary initial design without becoming extra starting
points or changing the plausible box. ``target.setup_cost`` charges the work
once against the
total ``max_fun_evals`` budget, while importing the cached values adds no
second charge. ``results["func_count"]`` remains the number of fresh calls
made by VBMC. Pass ``setup_budget=`` to cap preparation before it runs, or
supply ``start=`` and ``plausible_bounds=`` mappings in model coordinates to
override the automatic choices. The extra 60 calls above keep the example
short; use an adequate budget and validate repeated runs for real inference.

Supported free variables are continuous and float64, with recognized real-line
support or PyMC's standard log, log-odds or interval transform. Unknown support,
suppressed or custom transforms, discrete variables, support bounds that
depend on another random variable, and a log-transformed variable whose
density does not reach down to zero are rejected. Custom likelihood operations
remain usable when their PyTensor graph provides the density; without
gradients, setup uses the initial point and prior quantiles.

Construction freezes the data and dimensions used for inference. Later
``pm.set_data`` calls on ``model`` therefore do not alter the fitted target;
build a new ``PyMCTarget`` to refit changed data. The original model remains
available as ``target.model`` for the structured export, deterministics and
posterior prediction shown above. Export the posterior fitted with this same
target and inference snapshot. ``PyMCTarget(seed=)`` controls setup draws;
``VBMC(seed=)`` independently controls the run. Saving a run also saves its
adapter and models; load it only with compatible PyMC and PyTensor versions.
See :doc:`api/classes/pymc_target` and
:ref:`PyVBMC Example 8: Fitting a PyMC model` for the full workflow.

Use a fitted posterior downstream
=================================

Stacking several runs
---------------------

:doc:`Stacking Variational Bayesian Monte Carlo (S-VBMC) <api/classes/svbmc>`
(`Silvestrin et al., 2025 <https://arxiv.org/abs/2504.05004>`__) combines the posteriors of several completed
VBMC runs on the same model and data into a stacked posterior, without
further model evaluations. This often improves the approximation to the true posterior
by leveraging information from independent runs. See
:ref:`PyVBMC Example 7: Stacking the posteriors of several runs (S-VBMC)`
for a worked example. S-VBMC requires the optional ``torch`` extra; the
:doc:`installation guide <installation>` also covers dependencies for
the posterior exports below.

Torch distribution
------------------

:meth:`~pyvbmc.VariationalPosterior.to_torch` returns an independent torch
distribution snapshot::

  import torch

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

The complete workflow in
:ref:`PyVBMC Example 9: Torch and JAX models and posterior exports` uses this
distribution for psychometric probability predictions and density gradients.

ArviZ DataTree
--------------

:meth:`~pyvbmc.VariationalPosterior.to_arviz` draws independent samples into
the current ArviZ DataTree format::

  import arviz as az

  posterior_data = vp.to_arviz(
      n_samples=2000, var_names=["location"]
  )
  summary = az.summary(posterior_data, group="posterior", kind="stats")
  axes = az.plot_dist(posterior_data, group="posterior")

The result contains one ``posterior`` group and one chain. By default, each
parameter is a scalar variable named ``x_0``, ..., ``x_{D-1}``; the structured
form below supports vector and matrix variables. Pass ``orig_flag=False`` to
export samples in internal coordinates.

For a three-dimensional posterior whose first two columns form a vector,
preserve the variable shape and labels with a structured export::

  # Here vp has D == 3.
  structured_data = vp.to_arviz(
      n_samples=2000,
      variables={"beta": (2,), "sigma": ()},
      dims={"beta": ["coef"]},
      coords={"coef": ["intercept", "slope"]},
  )

Here ``beta`` has dimensions ``(chain, draw, coef)`` and ``sigma`` has
dimensions ``(chain, draw)``. Variables consume consecutive posterior columns
in mapping order.

This method advances ``vp.rng`` in the same way as
:meth:`~pyvbmc.VariationalPosterior.sample`. The draws are independent samples
from a variational approximation, so MCMC convergence diagnostics computed on
them do not measure the quality of that approximation. Use posterior summaries
and plots, and assess PyVBMC convergence from its own results and repeated
runs.
