============
VBMC options
============
Pass option values through the ``options`` argument of :class:`pyvbmc.VBMC`.
Basic options control common aspects of a run. Advanced options expose
specialized algorithm settings and are rarely needed.

Basic options
=============

.. include:: ./../../../../pyvbmc/vbmc/option_configs/basic_vbmc_options.ini
   :literal:
   :class: wrap

Integer-valued variables
========================

Integer variables are experimental. Set ``integer_vars`` to either a Boolean
mask with one entry per variable or a sequence of zero-based variable indices.
Place each hard bound half an integer outside the allowed range. For example,
use bounds ``-0.5`` and ``10.5`` for the integers 0 through 10. A prior passed
through ``prior`` must cover these hard bounds; the corresponding uniform prior
is ``UniformBox(-0.5, 10.5)``.

VBMC snaps points found by the active-sampling search to the integer grid. It
does not snap the initial design, including ``x0``. The first
``fun_eval_start`` evaluations may therefore be off-grid unless the supplied
starting points fill the whole initial design. On a grid, the search may also
select an input evaluated earlier. For a noiseless target, that call spends an
evaluation without adding an observation. For a noisy target, repeated calls
can refine the estimate at that input; ``max_repeated_observations`` limits
consecutive repeated measurements. A repeated measurement at an off-grid
point from the initial design stays at that point.

Advanced options
================

.. include:: ./../../../../pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini
   :literal:
   :class: wrap
