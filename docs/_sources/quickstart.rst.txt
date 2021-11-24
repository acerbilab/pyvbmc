***************
Getting started
***************

The ``pyvbmc`` interface is quite simple. 

.. code-block:: python

  vbmc = VBMC(target, x0, LB, UB, PLB, PUB)
  vp, elbo, elbo_sd, _, _ = vbmc.optimize()

To be continued...
