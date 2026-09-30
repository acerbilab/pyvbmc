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
another or unknown. The other messages, the returned named tuple, which gives
a script the same answer, and the handling of network failures, which raise
no error, are described below.

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

See also the :ref:`startup tips <Startup tips>` of ``VBMC``.
