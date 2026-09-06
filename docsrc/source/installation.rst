************
Installation
************

PyVBMC is available via ``pip`` and ``conda-forge`` and requires Python 3.10
or newer.

Basic installation
==================

Install PyVBMC with pip::

  python -m pip install pyvbmc

or with Conda::

  conda install --channel=conda-forge pyvbmc

Optional integrations
=====================

Torch
-----

The :meth:`~pyvbmc.VariationalPosterior.to_torch` export requires torch 2.7
or newer. For a CPU-only installation, install torch from its official CPU
wheel index first, then install the PyVBMC extra::

  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python -m pip install "pyvbmc[torch]"

Installing torch first ensures that pip selects a CPU wheel. If your model or
export will run on an accelerator, follow the `torch installation selector
<https://pytorch.org/get-started/locally/>`_ for your platform before
installing ``pyvbmc[torch]``.

With Conda, install the named packages directly::

  conda install --channel=conda-forge pyvbmc pytorch

ArviZ
-----

The :meth:`~pyvbmc.VariationalPosterior.to_arviz` export uses the current
ArviZ DataTree format and requires Python 3.12 or newer. Install it with::

  python -m pip install "pyvbmc[arviz]"

PyVBMC itself continues to support Python 3.10 and newer. On Python 3.10 or
3.11, pip does not install the ArviZ dependencies because of their Python
version marker, and ``to_arviz`` is unavailable. With Conda, install the
corresponding packages directly::

  conda install --channel=conda-forge pyvbmc arviz arviz-base

The bracketed ``torch`` and ``arviz`` names are pip dependency groups; they
are not Conda package names. Both integrations are optional and loaded on demand.

Examples and tests
==================

Install Jupyter to view the examples. You can skip this step if your Conda
environment already has Jupyter, but an unrelated ``jupyter`` executable on
your path can cause import errors::

  conda install jupyter

If Python 3.11 produces an ``UnsatisfiableError``, install Jupyter from
``conda-forge``::

  conda install --channel=conda-forge jupyter

Open the example notebooks with::

  python -m pyvbmc

Example 2 uses ``plotly`` for one interactive figure. Install it with::

  python -m pip install "pyvbmc[examples]"

Run PyVBMC's internal tests after installing the test dependencies with::

  python -m pip install "pyvbmc[test]"
  pytest --pyargs pyvbmc --reruns=3

The ``--reruns=3`` argument retries a failed test up to three times because
some tests are stochastic. The complete suite can take 20--30 minutes or
longer, depending on the hardware.

To install directly from the latest source, see the :ref:`installation
instructions for developers`.
