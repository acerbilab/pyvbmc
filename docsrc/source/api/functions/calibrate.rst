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
not evaluate your model or consume the inference random stream. It prints
progress, the outcome and the cache location. The returned profile contains
the settings; the saved report contains detailed measurements.
Use ``verbose=False`` to suppress routine output. Keeping the
existing settings is a successful outcome when no reliable gain is found.

On completion, the message states where the results were saved, using the
actual full path for your machine. For example (with the path abbreviated):

.. code-block:: text

   Calibration finished in 43 seconds.
   The standard settings performed well; no reliable improvement was found.
   Results saved to: <cache-root>/calibration/v1/<fingerprint>.json
   Future runs will use these settings automatically.
   To recalibrate, run pyvbmc.calibrate() again.

The report filename is also available as ``profile.cache_path`` (relative
if the cache override is relative). The displayed location reflects any
``PYVBMC_CACHE_DIR`` override. If saving fails, the message identifies the
attempted location and explains that the settings are usable in the current
process but were not saved.

Expect a duration in the tens of seconds, depending on your machine.
Thirty seconds is an estimate, not a cutoff: the campaign completes its
measurements and validation. A generous watchdog stops unexpectedly long
campaigns. Run it when the machine is otherwise quiet for useful timings.

Every call starts a fresh campaign. To recalibrate, call
``pyvbmc.calibrate()`` again. Calibration never starts automatically during
import, posterior evaluation, optimization or save/load.

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

Set ``performance_calibration="off"`` in the options dictionary to use
historical chunk settings. Missing or incompatible cache records also use
these defaults, with a calibration suggestion unless ``display="off"``.
There is no unexpected campaign after a software upgrade or cache deletion.

Each run keeps one profile through optimization, final boost and save/resume.
Recalibration affects future unresolved runs, not an existing resolved run.
Calling a PDF or entropy kernel on ``vbmc.vp`` before optimization resolves
its profile early. Standalone ``VariationalPosterior`` objects use historical
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
that could not be saved.

Reuse checks the machine and numerical environment, including library
versions, numerical backend and thread settings. An unwritable cache still
allows use of the returned profile and reuse within the process. An
interrupted or unsuccessful campaign preserves the previous valid record.

The profile contains three fixed budgets: PDF, entropy with gradients and
entropy values. It does not change sample counts or inference tolerances.
Entropy chunking changes floating-point addition order, so different profiles
can produce different trajectories even with the same seed. Save the profile
with the run to reproduce its settings; saved runs retain their profile on
another machine as well.

.. autofunction:: pyvbmc.calibrate

See also :doc:`../classes/calibration_profile` and
:doc:`../options/vbmc_options`.
