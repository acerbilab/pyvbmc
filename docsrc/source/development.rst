********************************************
Instructions for developers and contributors
********************************************

PyVBMC is the port of the MATLAB VBMC algorithm to Python 3.x (development has targeted version 3.9 and up).

The reference code is the :labrepos:`MATLAB toolbox <vbmc>`.

The documentation is available at: https://acerbilab.github.io/pyvbmc/

How to install and run the package (temporary)
##############################################

PyVBMC is not yet available on ``pip`` / ``conda-forge``. The suggested manual installation instructions can be found on the :ref:`Installation` page.

We are using the dependencies listed in ``requirements.txt``. Please list all used dependencies there.
For convenience, we also have a temporary installer in ``setup.py``. Also list the used dependencies there.

The necessary packages can be installed with `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_ or `pip <https://pypi.org/project/pip/>`_.

Alternative installation commands
---------------------------------

These are alternative ways to install the required dependencies::

    conda env create --file environment.yml

or::

    pip install -i requirements.txt

The ``environment.yml`` seems not to work properly in some setups (e.g., Windows), which is something to be investigated.

Coding conventions
##################

We try to follow common conventions whenever possible. Some useful reading:

- `PEP 8 -- Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_
- `Code style in The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`_

These basic rules should be followed to ensure coherence and to make it easy for third parties to contribute. In the following, we list more detailed conventions. Please read carefully if you are contributing to PyVBMC.

Code formatting
---------------

The code is formatted using `Black <https://pypi.org/project/black/>`_ with a line length of 79, with the help of pre-commit hooks. To install and use::

    pip install pre-commit
    pre-commit install
    pre-commit run -a  # run for all files, optional

After installation, when you try to commit the staged files, git will automatically check the files and modify them for meeting the requirements of the hooks in ``.pre-commit-config.yaml``. The settings of the hooks are specified in ``pyproject.toml``. You need to restage the file if it gets modified by the hooks.

If you want, you can also check with Pylint for more detailed errors and warnings (although Pylint seems to raise many false positives).

Docstrings
----------

The docstrings are generated following the `NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
There are add-ons to generate docstring blueprints using IDE's.

- See an example for a correct docstring from NumPy `here <https://numpydoc.readthedocs.io/en/latest/example.html>`__.
- In PyVBMC, the ``VariationalPosterior`` class can be taken as an example of (mostly) correct docstring structure, see :mainbranch:`here <variational_posterior/variational_posterior.py>`.
  - In particular, see how the single quotes and double quotes are used; the math notation is used; full stops are added at the end of each sentence, etc.

Code documentation
------------------

The documentation is currently hosted on :doc:`github.io <index>`. We build the PyVBMC documentation using `Sphinx <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_. The source code of the documentation is in the :mainbranch:`docsrc folder <docsrc>` and the build version is in the :labrepos:`gh-pages <pyvbmc/tree/gh-pages>` branch. When the documentation is re-built, it should be pushed to the ``gh-pages`` instead of committing it on the ``main`` branch.

To setup an existing PyVBMC repository for building documentation, please follow the steps below:

1. One-time setup:

   a. Remove the ``docs/`` folder from the root of your existing PyVBMC repo, if it is present.
   b. From the root of the PyVBMC repo, run::

       git clone -b gh-pages --single-branch https://github.com/acerbilab/pyvbmc docs

      This will clone *only* the ``gh-pages`` branch inside ``docs/``, so that changes to the docs can now be pushed directly to ``gh-pages`` from within ``docs/``.
2. From the ``main`` branch render new documentation::

    cd /docsrc (navigate to documentation source folder)
    make github  (this builds the doc and copies the build version to ./docs)

   (If you are using Windows, run ``.\make.bat github`` with ``cmd`` instead.)
3. Change into the ``docs/`` directory::

     cd ../docs

4. Commit the new documentation and push. `github.io <https://acerbilab.github.io/pyvbmc/>`_ will detect the changes and rebuild the website (possibly after a few minutes). Only documentation that was built from the ``main`` branch should be committed to ``gh-pages``.

If it seems that the documentation does not update correctly (e.g., items not appearing in the sidebar or table of content), try deleting the ``./docs`` folder and the cached folder ``./docsrc/_build`` before compiling the documentation. There is a command for that::

    make clean

(If you are using Windows, run ``.\make.bat clean`` with ``cmd`` instead.)

General structure
.................

For each new class, function, etc. a ``.rst`` file needs to be created in an appropriate folder. The folder names are arbitrary, for now we have ``functions``, ``classes``, etc.
The ``.rst`` file contains the text in `reStructuredText format <https://en.wikipedia.org/wiki/ReStructuredText>`_, a lightweight markup language with special commands that tell Sphynx where to compile the documentation, for example::

    .. autoclass:: pyvbmc.vbmc.VBMC
      :members:

Refer to existing documentation for an overview of the file structure. So far the documentation includes the following:

- Status of the port (what is missing?);
- Reference to the respective file of the original :labrepos:`MATLAB <vbmc>` implementation;
- Known issues (if something is currently suboptimal in PyVBMC);
- The documentation of the Python code (generated from the docstrings).

For each new file, a link needs to be added manually to the :mainbranch:`index page <docsrc/source/index.rst>`.
Please keep the documentation up to date. (Sphinx logs possible issues when compiling the documentation.)

Exceptions
----------

Currently, the aim is to use the standard Python exceptions whenever it is sensible.
Here is a list of those `exceptions <https://docs.python.org/3/library/exceptions.html>`_.

``git`` commits
---------------

Commits follow the `conventional commits <https://www.conventionalcommits.org/en/v1.0.0/>`_ style. This makes it easier to collaborate on the project. A cheat sheet is can be found `here <https://cheatography.com/albelop/cheat-sheets/conventional-commits/>`__

Please do not submit pull requests with unfinished code or code which does not pass all tests. Work on feature branches whenever possible and sensible. All PR's must be approved by another developer before being merged to the main branch. `Read this <https://martinfowler.com/bliki/FeatureBranch.html>`_ ::

    git checkout -b <new-feature>
    [... do stuff and commit ...]
    git push -u origin <new-feature>
    [... when finished created pull request on github ...]

If you switch to an existing branch using ``git checkout``, remember to ``pull`` before making any change as it is not done automatically.

Modules and code organization
-----------------------------

We have decided against general util/misc modules for now. This means that general-purpose functions should be included in a fitting existing module or in their own module. This approach encourages us to keep functions general and coherent to their scope. Furthermore, it improves readability for new collaborators. See some reading about that `here <https://breadcrumbscollector.tech/stop-naming-your-python-modules-utils/>`__.

Testing
-------

The testing is done using ``pytest`` with unit tests for each class in the respective folder.
Tests can be run with::

    pytest test_filename.py
    pytest
    pytest --reruns 5 --cov=. --cov-report html:cov_html

The final command creates an html folder with a full report on coverage -- double-check it from time to time. Some tests are stochastic and occasionally fail: Tests can be automatically rerun by specifying e.g. ``--reruns 3``.

A few comments about testing:

- Testing is mandatory!
- Please try to keep the total runtime of the tests minimal for the task at hand.
- As a good practice, please rerun all tests before major commits and pull requests (might take a while, but it is worth it to avoid surprises).
- A nice way of proceeding is 'test first': write a test first, make it fail, write the code until the test is passed.
- Many methods are tested against test cases produced with the original :labrepos:`MATLAB implementation <vbmc>`.
- The ``pytest-mock`` library is very useful for testing. It allows you to replace parts of your system under test with mock objects and make assertions about how they have been used. (Perhaps we should switch to ``unittest.mock`` in the future, which is part of the Python standard library.)
- We should look into automating tests with GitHub actions.
