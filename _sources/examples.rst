********
Examples
********

The examples below progress from basic setup to complete model-integration and
post-processing workflows. The corresponding Jupyter notebooks are available
:mainbranch:`in the repository <examples/>`, with generated plain scripts
:mainbranch:`under examples/scripts <examples/scripts>`.

The three integration examples have optional dependencies:

- :ref:`PyVBMC Example 7: Stacking the posteriors of several runs (S-VBMC)`
  combines completed runs and requires ``pyvbmc[torch]``.
- :ref:`PyVBMC Example 8: Fitting a PyMC model` covers model setup, fitting,
  structured export and posterior prediction. It requires Python 3.12 or
  newer and ``pyvbmc[pymc]``.
- :ref:`PyVBMC Example 9: Torch and JAX models and posterior exports` fits a
  two-parameter psychometric curve with a Torch target, shows the equivalent
  short JAX target wrapper, then makes posterior probability predictions and
  density-gradient calculations in Torch and summaries in ArviZ. The complete
  notebook requires Python 3.12 or newer and ``pyvbmc[torch,arviz]``. Install
  JAX separately only to run its JAX section; the Torch sections do not depend
  on JAX. A CPU is sufficient for all three examples.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :glob:

   _examples/*
