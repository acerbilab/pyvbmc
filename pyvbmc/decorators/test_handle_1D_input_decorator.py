import numpy as np

from pyvbmc.decorators import handle_1D_input


def test_1D_kwarg():
    class Foo:
        @handle_1D_input(kwarg="x", argpos=0)
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = np.ones(10)
    assert x.shape == Foo().bar(x=x).shape


def test_1D_posarg():
    class Foo:
        @handle_1D_input(kwarg="x", argpos=0)
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = np.ones(10)
    assert x.shape == Foo().bar(x).shape


def test_1D_ignoring_2D():
    class Foo:
        @handle_1D_input(kwarg="x", argpos=0)
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = np.ones((10, 20))
    assert x.shape == Foo().bar(x).shape


def test_1D_return_scalar():
    class Foo:
        @handle_1D_input(kwarg="x", argpos=0, return_scalar=True)
        def bar(self, x):
            return np.sum(x)

    x = np.ones(10)
    assert np.ndim(Foo().bar(x)) == 0


def test_1D_return_multiple_returns():
    class Foo:
        @handle_1D_input(kwarg="x", argpos=0)
        def bar(self, x):
            if np.ndim(x) > 1:
                return x, x
            return None

    x = np.ones(10)
    res = Foo().bar(x)
    assert x.shape == res[0].shape
    assert x.shape == res[1].shape


def test_1D_return_multiple_returns_scalar():
    class Foo:
        @handle_1D_input(kwarg="x", argpos=0, return_scalar=True)
        def bar(self, x):
            y = np.sum(x)
            return y, y

    x = np.ones(10)
    res = Foo().bar(x)
    assert np.ndim(res[0]) == 0
    assert np.ndim(res[1]) == 0