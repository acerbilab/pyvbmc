***************
Getting started
***************

The best way to get started with ``pyvbmc`` is via the tutorials and worked examples.
In particular, start with `Example 1: Basic usage <_examples/pyvbmc_example_1.html>`_, and continue from there.

If you are already familiar with approximate inference methods, you can find a summary usage below.

Summary usage
=============

The typical usage pipeline of ``pyvbmc`` follows four steps:

1. Define the model, which defines a target log density (i.e., an unnormalized log posterior density);
2. Setup the parameters (parameter bounds, starting point);
3. Initialize and run the inference;
4. Examine and visualize the results.

``pyvbmc`` is not concerned with how you define your model in step 1, as long as you can provide an (unnormalized) target log density.
Running the inference in step 3 only involves a couple of lines of code:

.. code-block:: python

  from pyvbmc.vbmc import VBMC
  # ...
  vbmc = VBMC(target, x0, LB, UB, PLB, PUB)
  vp, elbo, elbo_sd, _, _ = vbmc.optimize()

with input arguments:

- ``target``: the target (unnormalized) log density — often an unnormalized log posterior. ``target`` takes as input a parameter vector and returns the log density at the point;
- ``x0``: the starting point of the inference in parameter space;
- ``LB`` and ``UB``: hard lower and upper bounds for the parameters (can be ``-inf`` and ``inf``, or bounded);
- ``PLB`` and ``PUB``: *plausible* lower and upper bounds, that is a box that ideally brackets a region of high density of the target.

The outputs are:

- ``vp``: a ``VariationalPosterior`` object which approximates the true target density;
- ``elbo``: the estimated lower bound on the log model evidence (log normalization constant);
- ``elbo_sd``: the standard deviation of the estimate of the ``elbo`` (*not* the error between the ``elbo`` and the true log model evidence, which is generally unknown).

The ``vp`` object can be manipulated in various ways, see the `VariationalPosterior <api/classes/variational_posterior.html>`_ class documentation.

See the examples for more detailed information.
