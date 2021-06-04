====
VBMC
====

Porting status
==============
.. todo::
     - in _boundscheck the plausible bounds are set equal to the hard bounds, which is not admissible in VBMC (Fix by Luigi required)
     - check the documentation of vbmc init and the class in general
     - integrate the basic options somehow
     - init a GP (wait for GPLite to be completed)

Matlab references of ported methods
-----------------------------------
- vbmc: `vbmc.m <https://github.com/lacerbi/vbmc/blob/master/vmbc.m>`_
- parts of the init: `setupvars_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/setupvars_vbmc.m>`_
- _boundscheck: `boundscheck_vbmc.m <https://github.com/lacerbi/vbmc/blob/master/misc/boundscheck_vbmc.m>`_

Python documentation
====================
.. autoclass:: pyvbmc.vbmc.VBMC
   :members: