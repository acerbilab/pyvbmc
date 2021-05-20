# PyVBMC
This works as a repository to port the VBMC algorithm to Python3.

## General conventions
We try to follow common conventions whenever possible.

Some useful reading hints regarding that:

- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Code style in The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/style/)

## Third party libraries
We are using the dependencies listed in the requirements.txt. Please list all libraries there.

They can be installed with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or pip.

```
conda env create --file environment.yml
```

```
pip install -i requirements.txt
```

## Docstrings

The docstrings are generated following the numpy format. There are libaries to generate docstring blueprints using IDEs.

## Code formatting

The code is formatted using [Black](https://pypi.org/project/black/) with a line length of 79.

## Git commits

The git commits are following the [conventional commits convention](https://www.conventionalcommits.org/en/v1.0.0/). This makes it easier to collaborate on the project. A cheat sheet is can be found [here](https://cheatography.com/albelop/cheat-sheets/conventional-commits/)

Please do not commit broken code (red tests, not finished) on the master branch, work on feature branches whenever possible and sensible. [Read this](https://martinfowler.com/bliki/FeatureBranch.html)

## Testing

The testing is done using pytest with unit tests for each class in the respective folder.
Most methods are also tested against testcases produced with the original [matlab](https://github.com/lacerbi/vbmc) implementation.

They can be run with (occasionally looked the coverage):

```
pytest
pytest --cov=. --cov-report html:cov_html
```

## Exceptions

Currently, the aim is to use the standard python exceptions whenever it is sensible. Here is a list of those [exceptions](https://docs.python.org/3/library/exceptions.html).

Please use fixtures when sensible and try to keep the total runtime of the tests as low as possible.

## Modules and code organization

We have decided against general util/misc modules for now. This means that general purpose functions should be included in a fitting existing module or in their own module. The reason for this is to force us to think about the generalization of a function and prevent incohesion of those general collections. Furthermore, it improves readability for new collaborators. See some reading about that [here](https://breadcrumbscollector.tech/stop-naming-your-python-modules-utils/). One example for this are the decorators that are included in the decorator module.

## Decorators

Try to evaluate if pre- and postprocessing in a function can be generalized with a decorator. One example is the handling of arrays of shape (N, ) to (N, 1) with the handle_1D_decorator in the decorator module.
