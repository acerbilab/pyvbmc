# PyVBMC
This works as a repository to port the VBMC algorithm to Python3.

## General conventions
We try to follow common conventions whenever possible.

Some useful reading hints regarding that:

- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Code style in The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/style/)

## Third party libraries
So far we are using the following libraries:

*tbd: how to lock dependencies*

- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [pytest](https://docs.pytest.org/en/stable/) (for testing)

## Docstrings

The docstrings are generated following the numpy format. There are libaries to generate docstring blueprints using IDEs.

## Code formatting

The code is formatted using [Black](https://pypi.org/project/black/) with a line length of 79.

## Git commits

The git commits are following the [conventional commits convention](https://www.conventionalcommits.org/en/v1.0.0/). This makes it easier to collaborate on the project. A cheat sheet is can be found [here](https://cheatography.com/albelop/cheat-sheets/conventional-commits/)

*tbd: branches and commiting of broken code*

## Testing

The testing is done using pytest with unit tests for each class in the respective folder.
Most methods are also tested against testcases produced with the original [matlab](https://github.com/lacerbi/vbmc) implementation.

They can be run with:

```
pytest
```

## Exceptions

Currently, the aim is to use the standard python exceptions whenever it is sensible. Here is a list of those [exceptions](https://docs.python.org/3/library/exceptions.html).

Please use fixtures when sensible and try to keep the total runtime of the tests as low as possible.

## Render class diagrams
The diagrams can be rendered using pyreverse which is a part of [pylint](https://pypi.org/project/pylint/)

```
pyreverse -o png -f ALL entropy/* gaussian_process/* variational_posterior/* vbmc/* 
```