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
