====================
VariationalPosterior
====================

.. note::

  The main output of a PyVBMC run is a ``VariationalPosterior`` object ``vp``.

  The variational posterior can be queried and manipulated via several methods, such as:

  -  ``moments``: compute mean and covariance matrix of the variational posterior;
  -  ``pdf``: evaluate the variational posterior density at a point;
  - ``plot``: plot the variational posterior as a corner plot (1D and 2D marginals);
  - ``sample``: draw random samples from the variational posterior.

  There are also methods to compare variational posteriors:

  - ``kldiv``: compute the Kullback-Leibler divergence between two posteriors;
  - ``mtv``: compute the marginal total variation distance between two posteriors.

  See below for the full ``VariationalPosterior`` class methods and interface.


.. autoclass:: pyvbmc.variational_posterior.VariationalPosterior
   :members:
