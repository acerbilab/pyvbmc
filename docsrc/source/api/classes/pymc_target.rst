:tocdepth: 2

================
``PyMCTarget``
================

.. note::

  ``PyMCTarget`` adapts a supported PyMC model to PyVBMC, including the
  model's log joint, variable transforms, bounds, starting point and plausible
  box. Install the optional integration with
  ``python -m pip install "pyvbmc[pymc]"`` on Python 3.12 or newer; see
  :doc:`../../installation`. Importing ``pyvbmc`` or ``pyvbmc.pymc`` does not
  import PyMC or PyTensor.

PyMC adapter at a glance
------------------------

Construct the model normally, then pass one adapter object directly to
:class:`~pyvbmc.VBMC`:

.. code-block:: python

   import pymc as pm
   from pyvbmc import PyMCTarget, VBMC

   with pm.Model() as model:
       location = pm.Normal("location", 0.0, 2.0)
       scale = pm.HalfNormal("scale", 1.0)
       pm.Normal("y", location, scale, observed=observations)

   target = PyMCTarget(model, seed=7)
   vbmc = VBMC(
       target,
       seed=7,
       options={"max_fun_evals": target.setup_cost + 100},
   )
   vp, results = vbmc.optimize()

``VBMC(target)`` obtains ``x0``, hard and plausible bounds, the target
callable, setup observations and their function-equivalent cost from the same
object. ``str(target)`` displays the layout and numerical bounds before a run
starts. The read-only ``vbmc.target`` property retains the adapter for export
after fitting or loading a saved run.

Explicit ``VBMC`` arguments can override the starting point, plausible bounds,
observations and initialization charge. These starting points and bounds use
the adapter's flat coordinates; the adapter constructor's ``start`` and
``plausible_bounds`` mappings use model coordinates. Hard-bound overrides
must equal ``target.lb`` and ``target.ub`` in those flat coordinates, and a
separate ``prior`` or ``log_prior`` is rejected
because the PyMC model already includes its prior.

PyMC coordinate layout and hard bounds
--------------------------------------

Free variables follow the supplied model's ``model.free_RVs`` order. Each
variable is flattened in C order; ``coordinate_names`` uses declared PyMC
coordinate labels when available. The adapter's ``flatten`` and ``unflatten``
methods operate on PyMC value-variable names. ``to_model_variables`` and
``from_model_variables`` cross between flat VBMC points and mappings over the
original model-variable names.

Variables with no transform remain in model coordinates. Variables whose
coordinates all have two finite bounds also become model coordinates, with
those limits as VBMC's hard bounds. A variable whose coordinates are all
one-sided keeps its PyMC transform, travels in an unbounded value coordinate
such as ``scale_log__``, and contributes the transform Jacobian to the target.
This gives VBMC a fully unbounded or two-sided box while preserving the PyMC
model's density.

Explicit ``start=`` and ``plausible_bounds=`` values are mappings over every
free-variable name in model coordinates. Every value must be strictly inside
the recorded support. The adapter maps retained transforms to value space and
applies the same final normalization used by ordinary ``VBMC`` construction.

PyMC setup and plausible-region accounting
-------------------------------------------

By default, the adapter searches for a mode and builds a Laplace box of three
marginal standard deviations. A non-positive-definite Hessian or an unusable
marginal curvature takes its width from prior draws. A mode outside the
prior's 0.5%--99.5% location interval is reported as a diagnostic and is not
moved. If the graph has no gradient, the adapter uses the model's initial
point and the 5%--95% prior quantiles instead. Supply ``start=`` or
``plausible_bounds=`` to override the corresponding automatic step.

The default ``setup_budget`` is ``20 + 5 * D`` function-equivalent
evaluations. It caps work before it is performed; a smaller positive budget
shortens the mode search when the selected route still fits. The Hessian costs
``D`` units, and every attempted density or value-gradient call counts even
when its result is rejected or non-finite. ``setup_cost`` is the actual total,
while ``plausible_info`` separates target calls, Hessian calls and cost, the
selected route, and any fallback diagnostics.

Finite setup observations are available in ``setup_evaluations`` and are
automatically reused by ``VBMC(target)`` through the general
``precomputed_evaluations`` interface. They supplement the ordinary initial
design without changing ``x0`` or the plausible box. Importing them consumes
no fresh calls; ``setup_cost`` is charged once against
``options["max_fun_evals"]``. Consequently ``results["func_count"]`` remains
the literal number of fresh calls made by VBMC, and a positive setup charge
adds ``results["evaluation_budget"]`` with the total, initialization, fresh
and used counts. The maximum fresh-call allowance is the configured total
minus ``setup_cost``.

Passing ``precomputed_evaluations=None`` to ``VBMC(target, ...)`` disables
reuse while retaining the setup charge. Passing ``initialization_cost=0``
retains reuse but excludes setup from that run's budget, for example when
accounting for shared preparation separately across several fits. Reusing a
target does not consume its observations: each fit charges its setup cost by
default. The individual fields remain available for manually constructing
``VBMC(target.log_joint, ...)``; that form requires explicit observation and
cost arguments and leaves ``vbmc.target`` as ``None``.

Structured return to the PyMC model
-----------------------------------

Export variational draws with their model names, shapes, dimensions and
coordinates, then use PyMC's downstream functions. Use the posterior fitted
with this target and inference snapshot: the dimension check cannot establish
that a posterior belongs to the same model or coordinate layout. After loading
a run, use ``loaded.target.to_arviz(loaded.vp)``.

.. code-block:: python

   posterior_data = target.to_arviz(vp, n_samples=2000)
   posterior_data = pm.compute_deterministics(
       posterior_data, model=target.model, extend_dataset=True
   )
   predictions = pm.sample_posterior_predictive(
       posterior_data,
       model=target.model,
       var_names=["y"],
       random_seed=7,
   )

The result is an ArviZ DataTree whose ``posterior`` group records
``parameter_space="model"``. Export advances ``vp.rng`` exactly as
``vp.sample(n_samples)`` does. ``target.model`` is the original prediction
model, so PyMC's ordinary ``pm.set_data`` workflow can change compatible
covariates before prediction.

PyMC model-scope limits
-----------------------

Free variables must be continuous float64 variables with one of PyMC's
standard log, log-odds or interval transforms, or an explicitly recognized
proper distribution on the whole real line. Unknown free-variable support is
rejected even when no default transform is registered. A custom likelihood
operation is allowed when its model graph provides a compatible density;
missing derivatives select the documented setup fallbacks. This is distinct
from an unknown custom free-variable distribution.

PyTensor must use ``floatX=float64``; value variables and compiled density,
gradient and Hessian outputs must also be float64. Some PyMC expressions
retain a narrower constant dtype: for a uniform-only model, use explicit
``np.float64`` bounds, for example
``pm.Uniform("p", np.float64(0), np.float64(1))``. The adapter checks the
compiled output dtype rather than casting a lower-precision result.

The adapter rejects discrete or float32 free variables, suppressed or custom
transforms, interval limits that depend on another random variable, and a
single variable whose coordinates mix one-sided bounds with two-sided or
unbounded bounds. The log density must be finite strictly inside the hard box.
The adapter returns ``-inf`` only on or outside that box, where VBMC does not
evaluate. Use proper, normalized priors when interpreting the ELBO as a bound
on model evidence.

Persistence of adapted PyMC runs
--------------------------------

Target construction freezes the model data, shared numeric graph inputs,
dimensions and coordinate metadata used for inference. Later changes to the
original model do not change target values or setup observations; construct a
new target to refit changed data. The original model remains available for
posterior prediction.

Saving a :class:`~pyvbmc.VBMC` instance retains the adapter, its fixed
inference snapshot, original prediction model, setup observations and budget
charge. Loading recompiles PyTensor functions and can take several seconds.
Load the file only with compatible PyMC and PyTensor versions, then use
``loaded.target`` for export and prediction.

.. autoclass:: pyvbmc.pymc.PyMCTarget
   :members:

.. autoexception:: pyvbmc.pymc.UnsupportedModel
   :no-members:
