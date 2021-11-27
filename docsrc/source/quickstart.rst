***************
Getting started
***************

The typical usage pipeline of ``pyvbmc`` follows four steps:

1. Define your model (log likelihood and log prior) and the target log density;
2. Setup the parameters (parameter bounds, starting point);
3. Initialize and run the inference;
4. Examine and visualize the results.

``pyvbmc`` does not care how your model is defined in step 1, as long as you can provide an (unnormalized) target log density.
Running the inference in step 3 only involves a couple of lines of code:

.. code-block:: python
  
  from pyvbmc.vbmc import VBMC 
  # ...
  vbmc = VBMC(target, x0, LB, UB, PLB, PUB)
  vp, elbo, elbo_sd, _, _ = vbmc.optimize()

with input arguments:

- ``target``: the target (unnormalized) log density — often an unnormalized log posterior;
- ``x0``: the starting point of the inference;
- ``LB`` and ``UB``: hard lower and upper bounds for the parameters (can be ``inf``);
- ``PLB`` and ``PUB``: *plausible* lower and upper bounds, that is a box that ideally brackets a region of high density of the target.

The outputs are:

- ``vp``: a ``VariationalPosterior`` object which approximates the true target density;
- ``elbo``: the estimated lower bound on the log model evidence (log normalization constant);
- ``elbo_sd``: the standard deviation of the estimate of the ``elbo`` (*not* the error between the ``elbo`` and the true log model evidence, which is generally unknown).

The ``vp`` object can be manipulated in various ways, see the `VariationalPosterior <api/classes/variational_posterior.html>`_ class documentation.

Next steps
==========

The best way to get started with ``pyvbmc`` is via the worked examples. 
In particular, start with `Example 1: Basic usage <_examples/pyvbmc_example_1.html>`_, and continue from there.