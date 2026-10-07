=============
``calibrate``
=============

Calibrate performance on this machine
-------------------------------------

Run calibration explicitly when you want PyVBMC to tune its performance for
your machine:

.. code-block:: python

   import pyvbmc

   profile = pyvbmc.calibrate()

The campaign uses synthetic inputs and its own random generators. It does
not evaluate your model or consume the inference random stream. It usually
takes tens of seconds; run it while the machine is otherwise quiet. Progress,
the outcome and the cache location are printed unless ``verbose=False``.

Each call requests a fresh campaign. If another calibration is already
active, or the measurements cannot be completed or validated, ``calibrate``
returns the best compatible saved or in-process profile, or the standard
profile when none is available. ``profile.status`` records the outcome and
``profile.source`` says where the settings came from. A failed campaign does
not replace a valid saved record. Keeping the standard settings is also a
successful outcome when the measurements find no reliable improvement.

To recalibrate, call ``pyvbmc.calibrate()`` again. Calibration never starts
automatically during import, posterior evaluation, optimization or save/load.

Using the result
----------------

New VBMC runs use a compatible cached profile by default. This works the
same way in Python scripts and Jupyter notebooks; the notebook kernel must
use a compatible numerical environment. You may also provide the returned
profile explicitly:

.. code-block:: python

   vbmc = pyvbmc.VBMC(
       log_density, x0, lower_bounds, upper_bounds,
       plausible_lower_bounds, plausible_upper_bounds,
       options={"performance_calibration": profile},
   )

Set ``performance_calibration="off"`` in the options dictionary to use the
standard chunk settings. Missing or incompatible cache records use the same
settings, and the first such run of a Python session suggests calibrating
unless ``display="off"``.

Each run keeps one profile through optimization, final boost and save/resume.
Recalibration affects future unresolved runs, not an existing resolved run.
Calling a PDF or entropy kernel on ``vbmc.vp`` before optimization resolves
its profile early. Standalone ``VariationalPosterior`` objects use standard
settings unless constructed with ``calibration=profile`` or
``calibration="cached"``.

Storage and reproducibility
---------------------------

Profiles are stored in the operating system's per-user cache directory under
``pyvbmc/calibration/v1``. On Windows this is normally beneath
``%LOCALAPPDATA%``; macOS uses ``~/Library/Caches`` and Linux uses the XDG cache
root. Set ``PYVBMC_CACHE_DIR`` to override the PyVBMC cache root. The returned
profile's ``cache_path`` identifies its JSON record, including detailed
measurement results, when persistence succeeded. It is ``None`` for a result
that could not be saved. If the cache is unwritable, the returned profile
remains usable in the current process.

Reuse checks the machine and numerical environment, including library
versions, numerical backend and thread settings.

The profile contains three fixed budgets: PDF, entropy with gradients and
entropy values. Calibration changes chunk sizes only; it does not change
sample counts, inference tolerances or numerical results. The same seed gives
the same trajectory with calibrated or standard settings. A saved run retains
its profile, including when loaded on another machine.

.. autofunction:: pyvbmc.calibrate

See also :doc:`../classes/calibration_profile` and
:doc:`../options/vbmc_options`.
