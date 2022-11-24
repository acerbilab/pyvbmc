************
Installation
************

PyVBMC is not yet available on ``pip`` / ``conda-forge``, but can be installed in a few steps:

1. Clone the PyVBMC and GPyReg GitHub repos locally::

      git clone https://github.com/acerbilab/pyvbmc
      git clone https://github.com/acerbilab/gpyreg

   (PyVBMC depends on :labrepos:`GPyReg <gpyreg>`, which is a package for lightweight Gaussian process regression in Python.)
2. (Optional) Create a new environment in ``conda`` and activate it. We recommend using Python 3.9 or newer, but older versions *might* work::

      conda create --name pyvbmc-env python=3.9
      conda activate pyvbmc-env

3. Install the packages::

      cd ./gpyreg
      pip install -e .
      cd ../pyvbmc
      pip install -e .

4. Install ``jupyter`` to view the examples (you can skip this step if you're working from a ``conda`` environment which already has ``jupyter``)::

      conda install jupyter
