=====================
``check_for_updates``
=====================

Check for a newer release
-------------------------

Ask PyPI whether a newer release of PyVBMC is available:

.. code-block:: python

   import pyvbmc

   check = pyvbmc.check_for_updates()

The function prints one line. When PyPI has a newer release, the line names
it and gives the command that installs it, for example:

.. code-block:: text

   PyVBMC 1.6.0 is available; you have 1.5.0. Update with: python -m pip install --upgrade pyvbmc

The command follows the installer recorded with your installation:
``python -m pip install --upgrade pyvbmc`` for pip,
``conda update --channel=conda-forge pyvbmc`` for conda (the conda-forge
package can follow PyPI by a few days), and both when the installer is
another or unknown. Otherwise the line says that the installed version is the
latest release, or newer than the latest release on PyPI (as before a release
reaches PyPI); that it is a development version, or unknown, with the latest
release beside it; or that PyPI could not be reached, with the reason. A
network failure raises no error.

The returned named tuple gives the same answer to a script:
``check.installed`` is the installed version, ``check.latest`` the latest
release on PyPI (``None`` when PyPI could not be read), and
``check.update_available`` is ``True`` or ``False``, or ``None`` when either
version is unknown or the installed version is a development version.

Network access
--------------

PyVBMC contacts PyPI only when you call this function, and makes no other
network request. The request names the installed version of PyVBMC and
nothing else about your installation, and the call writes nothing to disk.

The old-release reminder
------------------------

When a new run starts in an interactive session and the installed release is
more than a year old, ``VBMC`` suggests calling this function, at most three
times for each installed version. The reminder makes no network request: it
compares the release date shipped with PyVBMC with the date of the run. The
:ref:`FAQ <faq-how-do-i-know-whether-a-newer-version-of-pyvbmc-exists>` says
when it appears and how to turn it off.

.. autofunction:: pyvbmc.check_for_updates

See also :doc:`../classes/vbmc` (its section on startup tips).
