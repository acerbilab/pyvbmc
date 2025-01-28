************
Installation
************

PyVBMC is available via ``pip`` and ``conda-forge``.

1. Install with::

     python -m pip install pyvbmc

   or::

     conda install --channel=conda-forge pyvbmc

   PyVBMC requires Python version 3.10 or newer.

2. (Optional): Install Jupyter to view the examples. You can skip this step if you're working from a Conda environment which already has Jupyter, but be aware that if the wrong ``jupyter`` executable is found on your path then import errors may arise. ::

     conda install jupyter

   If you are running Python 3.11 and get an ``UnsatisfiableError`` you may need to install Jupyter from ``conda-forge``::

     conda install --channel=conda-forge jupyter

   The example notebooks can be accessed by running ::

     python -m pyvbmc

You can run PyVBMC's internal tests with ::

  pytest --pyargs pyvbmc --reruns=3

The `--reruns=3` argument allows re-trying a failed test up to 3 times, as many of PyVBMC's tests are stochastic in nature. Note that the complete test suite may take a significant amount of time (20-30 minutes or more, depending on your hardware).

If you wish to install directly from latest source code, please see the :ref:`installation instructions for developers`.
