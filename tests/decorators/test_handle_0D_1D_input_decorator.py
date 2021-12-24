import numpy as np

from pyvbmc.decorators import handle_0D_1D_input


def test_1D_kwarg():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = np.ones(10)
    assert x.shape == Foo().bar(x=x).shape


def test_1D_posarg():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = np.ones(10)
    assert x.shape == Foo().bar(x).shape


def test_1D_ignoring_2D():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = np.ones((10, 20))
    assert x.shape == Foo().bar(x).shape


def test_1D_return_scalar():
    class Foo:
        @handle_0D_1D_input(
            patched_kwargs=["x"], patched_argpos=[0], return_scalar=True
        )
        def bar(self, x):
            return np.sum(x)

    x = np.ones(10)
    assert np.ndim(Foo().bar(x)) == 0


def test_1D_return_multiple_returns():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
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
        @handle_0D_1D_input(
            patched_kwargs=["x"], patched_argpos=[0], return_scalar=True
        )
        def bar(self, x):
            y = np.sum(x)
            return y, y

    x = np.ones(10)
    res = Foo().bar(x)
    assert np.ndim(res[0]) == 0
    assert np.ndim(res[1]) == 0


def test_0D_kwarg():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = 1
    assert (1, 1) == Foo().bar(x=x).shape


def test_0D_posarg():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x):
            if np.ndim(x) > 1:
                return x
            return None

    x = 1
    assert (1, 1) == Foo().bar(x).shape


def test_0D_return_scalar():
    class Foo:
        @handle_0D_1D_input(
            patched_kwargs=["x"], patched_argpos=[0], return_scalar=True
        )
        def bar(self, x):
            return np.sum(x)

    x = 1
    assert np.ndim(Foo().bar(x)) == 0


def test_0D_return_multiple_returns():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x):
            if np.ndim(x) > 1:
                return x, x
            return None

    x = 1
    res = Foo().bar(x)
    assert (1, 1) == res[0].shape
    assert (1, 1) == res[1].shape


def test_0D_return_multiple_returns_scalar():
    class Foo:
        @handle_0D_1D_input(
            patched_kwargs=["x"], patched_argpos=[0], return_scalar=True
        )
        def bar(self, x):
            y = np.sum(x)
            return y, y

    x = 1
    res = Foo().bar(x)
    assert np.ndim(res[0]) == 0
    assert np.ndim(res[1]) == 0


def test_0D_multiple_kwarg():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x, y):
            if np.ndim(x) > 1:
                return x, y
            return None

    x = 1
    y = 2
    res = Foo().bar(x=x, y=y)
    assert (1, 1) == res[0].shape
    assert np.ndim(res[1]) == 0
    assert np.all(x == res[0])
    assert y == res[1]
    res2 = Foo().bar(y=y, x=x)
    assert (1, 1) == res2[0].shape
    assert np.ndim(res2[1]) == 0
    assert np.all(x == res2[0])
    assert y == res2[1]


def test_0D_multiple_posarg():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x, y):
            if np.ndim(x) > 1:
                return x, y
            return None

    x = 1
    y = 2
    res = Foo().bar(x, y)
    assert (1, 1) == res[0].shape
    assert np.ndim(res[1]) == 0
    assert np.all(x == res[0])
    assert y == res[1]


def test_0D_kwarg_posarg_mixed():
    class Foo:
        @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
        def bar(self, x, y):
            if np.ndim(x) > 1:
                return x, y
            return None

    x = 1
    y = 2
    res = Foo().bar(x, y=y)
    assert (1, 1) == res[0].shape
    assert np.ndim(res[1]) == 0
    assert np.all(x == res[0])
    assert y == res[1]
