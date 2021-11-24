# PyVBMC

This repository contains the port of the VBMC algorithm to Python 3.x. 
The original source is the [MATLAB toolbox](https://github.com/lacerbi/vbmc).

## General coding conventions

We try to follow common conventions whenever possible.

Some useful readings:

- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Code style in The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/style/)

Please note that we are developing in a way to enable third parties to maintain and use the algorithm. Some rules have to be followed to ensure coherence and coordination, but please start a discussion when something does not seem sensible.

## How to install and run the package (temporary)

We are using the dependencies listed in `requirements.txt`. Please list all used dependencies there.
For convenience, we also have a temporary installer in `setup.py`. Also list the used dependencies there.

The necessary packages can be installed with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or pip.

The most stable way to install and run `pyvbmc` at the moment is:

1. Clone/update the `pyvbmc` GitHub repo locally.
2. Create a new environment in conda: `conda create --name pyvbmc-dev python=3.9`
3. Activate the environment: `conda activate pyvbmc-dev`
4. From the `pyvbmc` folder, run: `pip install -e .`
5. Install Jupyter notebook: `conda install jupyter`

If the list of requirements subsequently changes, you will only need to rerun `pip install -e .`.

### `gpyreg` package

To run `pyvbmc` you will also need the `gpyreg` package, a lightweight Gaussian process regression library that we wrote specifically for `pyvbmc`.
For now, since the package is not in a `conda` or `pip` package repository, you need to run the additional steps:

- Clone `gpyreg` from its [GitHub repo](https://github.com/lacerbi/gpyreg). 
- Install `gpyreg` in the `pyvbmc-dev` environment running `pip install -e .` from the `gpyreg` folder.


### Alternative installation commands

These are alternative ways to install the required dependencies:

```
conda env create --file environment.yml
```
or

```
pip install -i requirements.txt
```


## More detailed conventions

Please read these as well!

### Code formatting

The code is formatted using [Black](https://pypi.org/project/black/) with a line length of 79.

If you want, you can also check with pylint for more excessive errors. (Although pylint seems to raise many false positives.)

### Decorators

Try to evaluate if pre- and postprocessing in a function can be generalized with a decorator. One example is the handling of arrays of shape (N, ) to (N, 1) with the [handle_1D_decorator](./decorators/handle_1D_input.py) in the decorator module.


### Docstrings

The docstrings are generated following the [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).
There are add-ons to generate docstring blueprints using IDEs.

See an example for a correct docstring [here](https://numpydoc.readthedocs.io/en/latest/example.html).

### Documentation

The documentation is currently hosted on [github.io](https://lacerbi.github.io/pyvbmc/). We build the pyvbmc documentation using [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html). The source code of the documentation is in the [docsrc folder](./docsrc) and the build version is in the [docs folder](./docs).
From there new documentation can be compiled using the following commands:

1) Merge main branch into feature branch (bring the branch up to date with whatever changes were done in main):
```
git checkout main
git pull
git checkout <feature_branch>
git merge master
```

2) Make sure that everything works, e.g. by running tests.
3) Render new documentation:
```
cd /docsrc (navigate to documentation source folder)
make github  (this builds the doc and copies the build version to ./docs)
```
(If you are using Windows, run `.\make.bat github` with `cmd` instead.)

4) Commit the new documentation.
5) Create a new pull request.
6) When the pull request is merged, [github.io](https://lacerbi.github.io/pyvbmc/) detects changes and rebuilds the documentation.


#### General structure

For each new class, function, etc. a `.rst` file needs to be created in an appropriate folder. The folder names are arbitrary, for now we have `functions`, `classes`, etc.
The `.rst` file contains the text in [reStructuredText format](https://en.wikipedia.org/wiki/ReStructuredText), a lightweight markup language with special commands that tell Sphynx where to compile the documentation, for example:

```
.. autoclass:: pyvbmc.vbmc.VBMC
   :members:
```

Refer to existing documentation for an overview of the file structure. So far the documentation includes the following:

- Status of the Port (what is missing?);
- Reference to the respective file of the original [MATLAB](https://github.com/lacerbi/vbmc) implementation;
- Known issues (if something is currently suboptimal in pyvbmc);
- The documentation of the Python code (generated from the docstrings).

For each new file, a link needs to be added manually to the [index page](https://github.com/lacerbi/pyvbmc/blob/main/docsrc/source/index.rst).
Please keep the documentation up to date. (Sphinx logs possible issues when compiling the documentation.)


### Exceptions

Currently, the aim is to use the standard Python exceptions whenever it is sensible.
Here is a list of those [exceptions](https://docs.python.org/3/library/exceptions.html).

### Git commits

The git commits are following the [conventional commits convention](https://www.conventionalcommits.org/en/v1.0.0/). This makes it easier to collaborate on the project. A cheat sheet is can be found [here](https://cheatography.com/albelop/cheat-sheets/conventional-commits/)

Please do not commit broken code (red tests, not finished) on the master branch, work on feature branches whenever possible and sensible. [Read this](https://martinfowler.com/bliki/FeatureBranch.html)

```
git checkout -b <new-feature>
[... do stuff and commit ...]
git push -u origin <new-feature>
[... when finished created pull request on github ...]
```

If you switch to an existing branch using `git checkout`, remember to `pull` before making any change as it is not done automatically.

### Modules and code organization

We have decided against general util/misc modules for now. This means that general-purpose functions should be included in a fitting existing module or in their own module. The reason for this is to force us to think about the generalization of a function and prevent incohesion of those general collections. Furthermore, it improves readability for new collaborators. See some reading about that [here](https://breadcrumbscollector.tech/stop-naming-your-python-modules-utils/). One example of this are the decorators that are included in the decorator module.

### Testing

The testing is done using `pytest` with unit tests for each class in the respective folder. 
Tests can be run with:

```
pytest test_filename.py
pytest
pytest --reruns 5 --cov=. --cov-report html:cov_html
```

The final command creates an html folder with a full report on coverage -- double-check it from time to time. Since some tests are stochastic and occasionally fail (occasional failures for stochastic parts are fine, we rerun a failed test up to five times with `--reruns 5`).

A few comments about testing:

- Testing is mandatory!
- Please try to keep the total runtime of the tests minimal for the task at hand.
- As a good practice, please rerun all tests before major commits and pull requests (might take a while, but it is worth it to avoid surprises).
- A nice way of proceeding is `test first': write a test first, make it fail, write the code until the test is passed.
- Many methods are tested against test cases produced with the original [MATLAB](https://github.com/lacerbi/vbmc) implementation.
- The `pytest-mock` library is very useful for testing. It allows you to replace parts of your system under test with mock objects and make assertions about how they have been used. (Perhaps we should switch to `unittest.mock` in the future, which is part of the Python standard library.)
- Things to look into in the future: We should perhaps automatize tests with GitHub actions.
