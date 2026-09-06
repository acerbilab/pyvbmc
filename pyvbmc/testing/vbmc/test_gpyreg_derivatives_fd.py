"""Finite-difference checks of the gpyreg gradients PyVBMC's GP layer uses.

PyVBMC hard-wires one Gaussian process model: a squared-exponential ARD
kernel, a negative-quadratic mean, and Gaussian noise made of a constant term
plus an optional user-provided term (scaled or not) plus an optional
rectified-linear output-dependent term. Every hyperparameter gradient of that
model is written out by hand in gpyreg, and PyVBMC's own numerics rest on
them. These checks pin them from the side that depends on them: gpyreg tests
its kernels and its marginal-likelihood gradient, but its mean and noise
functions have no gradient checks of their own, and the noise switch
combinations exercised here are the ones ``vbmc/gaussian_process_train.py``
selects from ``optim_state["gp_noise_fun"]``. The kernel is checked too, as a
cross-check that the conventions of this module match gpyreg's own.

Every check perturbs one hyperparameter at a time by central differences and
compares against the gradient the ``compute(..., compute_grad=True)`` call
returns, in the layout that call uses.
"""

import numpy as np
import pytest
from gpyreg.covariance_functions import SquaredExponential
from gpyreg.mean_functions import ConstantMean, NegativeQuadratic, ZeroMean
from gpyreg.noise_functions import GaussianNoise

# Central differences of quantities of order one keep about eight digits.
# The absolute tolerance, scaled by the size of the gradient being checked,
# is what compares the entries that are exactly zero.
FD_STEP = 1e-6
RTOL = 1e-6
ATOL = 1e-8

SEEDS = [0, 3, 42]
DIMENSIONS = [2, 5]
N = 20

# (constant_add, user_provided_add, scale_user_provided, rectified_linear)
NOISE_CASES = {
    "constant": (True, False, False, False),
    "constant_user": (True, True, False, False),
    "constant_user_scaled": (True, True, True, False),
    "constant_rectified": (True, False, False, True),
    "constant_user_rectified": (True, True, False, True),
    "constant_user_scaled_rectified": (True, True, True, True),
}


def _fd_grad(f, hyp, h=FD_STEP):
    """Central-difference derivative of `f` in every entry of `hyp`.

    Returns one array per hyperparameter, each shaped like the value of `f`.
    """
    grads = []
    for k in range(hyp.size):
        plus = hyp.copy()
        plus[k] += h
        minus = hyp.copy()
        minus[k] -= h
        grads.append((f(plus) - f(minus)) / (2 * h))
    return grads


def _assert_close(finite_difference, analytic, what):
    scale = max(1.0, np.max(np.abs(analytic)))
    assert np.allclose(
        finite_difference,
        analytic,
        rtol=RTOL,
        atol=ATOL * scale,
    ), (
        f"{what}: largest difference "
        f"{np.max(np.abs(np.asarray(finite_difference) - analytic))}"
    )


def _data(rng, D):
    return rng.standard_normal((N, D))


@pytest.mark.parametrize("D", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_negative_quadratic_mean_gradient(seed, D):
    rng = np.random.default_rng(seed)
    mean = NegativeQuadratic()
    X = _data(rng, D)

    hyp = np.concatenate(
        (
            rng.normal(0.0, 1.0, size=1),  # mean_const
            rng.normal(0.0, 0.5, size=D),  # mean_location
            rng.normal(0.0, 0.3, size=D),  # mean_log_scale
        )
    )
    m, dm = mean.compute(hyp, X, compute_grad=True)
    assert m.shape == (N,)
    assert dm.shape == (N, mean.hyperparameter_count(D))

    for k, grad in enumerate(_fd_grad(lambda h: mean.compute(h, X), hyp)):
        _assert_close(grad, dm[:, k], f"NegativeQuadratic hyperparameter {k}")


@pytest.mark.parametrize("D", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_constant_mean_gradient(seed, D):
    rng = np.random.default_rng(seed)
    mean = ConstantMean()
    X = _data(rng, D)

    hyp = rng.normal(0.0, 1.0, size=1)
    m, dm = mean.compute(hyp, X, compute_grad=True)
    assert m.shape == (N,)
    assert dm.shape == (N, 1)

    for k, grad in enumerate(_fd_grad(lambda h: mean.compute(h, X), hyp)):
        _assert_close(grad, dm[:, k], f"ConstantMean hyperparameter {k}")


@pytest.mark.parametrize("D", DIMENSIONS)
def test_zero_mean_gradient(D):
    mean = ZeroMean()
    X = _data(np.random.default_rng(SEEDS[0]), D)

    hyp = np.zeros(0)
    m, dm = mean.compute(hyp, X, compute_grad=True)
    assert m.shape == (N,)
    assert np.all(m == 0.0)
    # No hyperparameters, hence nothing to differentiate.
    assert mean.hyperparameter_count(D) == 0
    assert np.size(dm) == 0
    assert _fd_grad(lambda h: mean.compute(h, X), hyp) == []


def _noise_threshold(y):
    """A threshold halfway between the two middle values of `y`.

    The rectified-linear term is not differentiable where ``y`` crosses the
    threshold, so the threshold is placed in the widest gap the middle of the
    sample offers: about half the points are then active, and no point sits
    close enough for a finite-difference step to switch it.
    """
    ordered = np.sort(y.ravel())
    middle = ordered.size // 2
    return 0.5 * (ordered[middle - 1] + ordered[middle])


@pytest.mark.parametrize("D", DIMENSIONS)
@pytest.mark.parametrize("case", list(NOISE_CASES))
@pytest.mark.parametrize("seed", SEEDS)
def test_gaussian_noise_gradient(seed, case, D):
    constant, user, scaled, rectified = NOISE_CASES[case]
    rng = np.random.default_rng(seed)
    noise = GaussianNoise(
        constant_add=constant,
        user_provided_add=user,
        scale_user_provided=scaled,
        rectified_linear_output_dependent_add=rectified,
    )
    X = _data(rng, D)
    y = rng.uniform(-2.0, 2.0, size=(N, 1))
    s2 = rng.uniform(0.05, 0.5, size=(N, 1)) if user else None

    hyp = [rng.normal(-1.0, 0.5, size=1)]  # noise_log_scale
    if scaled:
        hyp.append(rng.normal(0.0, 0.3, size=1))  # provided log multiplier
    if rectified:
        threshold = _noise_threshold(y)
        margin = np.min(np.abs(y - threshold))
        assert margin > 100 * FD_STEP
        assert np.any(y < threshold)  # The rectified term is active
        hyp.append(np.array([threshold]))
        hyp.append(rng.normal(0.0, 0.3, size=1))  # rectified log multiplier
    hyp = np.concatenate(hyp)
    assert hyp.size == noise.hyperparameter_count()

    sn2, dsn2 = noise.compute(hyp, X, y, s2, compute_grad=True)
    rows = N if (user or rectified) else 1
    assert dsn2.shape == (rows, hyp.size)
    assert np.size(sn2) == (N if (user or rectified) else 1)

    grads = _fd_grad(lambda h: noise.compute(h, X, y, s2), hyp)
    for k, grad in enumerate(grads):
        _assert_close(
            np.broadcast_to(np.atleast_1d(np.squeeze(grad)), (rows,)),
            dsn2[:, k],
            f"GaussianNoise {case} hyperparameter {k}",
        )


@pytest.mark.parametrize("D", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_squared_exponential_kernel_gradient(seed, D):
    rng = np.random.default_rng(seed)
    kernel = SquaredExponential()
    X = _data(rng, D)

    hyp = np.concatenate(
        (
            rng.normal(0.0, 0.3, size=D),  # log lengthscales
            rng.normal(0.0, 0.3, size=1),  # log output scale
        )
    )
    K, dK = kernel.compute(hyp, X, compute_grad=True)
    assert K.shape == (N, N)
    assert dK.shape == (N, N, kernel.hyperparameter_count(D))

    for k, grad in enumerate(_fd_grad(lambda h: kernel.compute(h, X), hyp)):
        _assert_close(
            grad, dK[:, :, k], f"SquaredExponential hyperparameter {k}"
        )
