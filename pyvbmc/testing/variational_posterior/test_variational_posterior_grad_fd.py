"""Finite-difference checks of the hand-derived input gradient of
``VariationalPosterior.pdf`` (used by ``VariationalPosterior.mode``).

Stage 0 of the modernization plan (see
``dev/2026-09-02-modernization-discussion.md``, section 10).
"""

import numpy as np
import pytest

from pyvbmc.testing import check_grad
from pyvbmc.variational_posterior import VariationalPosterior


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """``VariationalPosterior.__init__`` draws from the global ``np.random``
    state; leave the stream as we found it for the rest of the suite."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _vp():
    D, K = 3, 2
    vp = VariationalPosterior(D, K)
    vp.mu = np.array([[0.0, 0.8], [0.2, -0.5], [-0.3, 0.4]])
    vp.sigma = np.array([[0.7, 1.1]])
    vp.lambd = np.array([[1.0], [0.8], [1.3]])
    vp.w = np.array([[0.3, 0.7]])
    return vp


def test_pdf_grad_fd():
    vp = _vp()
    x0 = np.array([0.3, -0.1, 0.2])

    def f(x):
        return vp.pdf(x.reshape(1, -1), orig_flag=False)[0, 0]

    def grad(x):
        _, dy = vp.pdf(x.reshape(1, -1), orig_flag=False, grad_flag=True)
        return dy.ravel()

    assert grad(x0).shape == x0.shape
    assert check_grad(f, grad, x0, rtol=1e-6, atol=1e-9)


def test_log_pdf_grad_fd():
    vp = _vp()
    x0 = np.array([0.3, -0.1, 0.2])

    def f(x):
        return vp.pdf(x.reshape(1, -1), orig_flag=False, log_flag=True)[0, 0]

    def grad(x):
        _, dy = vp.pdf(
            x.reshape(1, -1), orig_flag=False, log_flag=True, grad_flag=True
        )
        return dy.ravel()

    assert grad(x0).shape == x0.shape
    assert check_grad(f, grad, x0, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("log_flag", [False, True])
def test_pdf_grad_fd_one_component_several_rows(log_flag):
    """One component in one dimension, evaluated at several points at once:
    each row's value depends on its own point alone, so the Jacobian of
    the column of values is the diagonal of the returned gradients."""
    vp = VariationalPosterior(1, 1, np.zeros((1, 1)), rng=20260923)
    vp.mu = np.array([[0.4]])
    vp.sigma = np.array([[0.7]])
    vp.lambd = np.array([[1.3]])
    vp.w = np.ones((1, 1))
    x0 = np.array([-1.7, -0.3, 0.4, 0.9, 2.6])

    def f(x):
        return vp.pdf(
            x.reshape(-1, 1), orig_flag=False, log_flag=log_flag
        ).ravel()

    def grad(x):
        _, dy = vp.pdf(
            x.reshape(-1, 1),
            orig_flag=False,
            log_flag=log_flag,
            grad_flag=True,
        )
        assert dy.shape == (x.size, 1)
        return np.diag(dy.ravel())

    assert check_grad(f, grad, x0, rtol=1e-6, atol=1e-9)
