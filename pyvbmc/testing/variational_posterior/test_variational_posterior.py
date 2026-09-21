import importlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import kl_div_mvn
from pyvbmc.variational_posterior import VariationalPosterior


def get_matlab_vp():
    path = Path(__file__).parent.joinpath("vp-test.npz")
    with np.load(path, allow_pickle=False) as fixture:
        vp = VariationalPosterior(2, 2, np.array([[5]]))
        vp.D = fixture["D"][0, 0]
        vp.K = fixture["K"][0, 0]
        vp.w = fixture["w"]
        vp.mu = fixture["mu"]
        vp.sigma = fixture["sigma"]
        vp.lambd = fixture["lambd"]
        vp.optimize_lambd = fixture["optimize_lambd"][0, 0] == 1
        vp.optimize_mu = fixture["optimize_mu"][0, 0] == 1
        vp.optimize_sigma = fixture["optimize_sigma"][0, 0] == 1
        vp.optimize_weights = fixture["optimize_weights"][0, 0] == 1
    vp.parameter_transformer = ParameterTransformer(vp.D)
    return vp


def test_constructor_takes_one_starting_point_in_any_layout():
    """A single starting point of `D` elements starts every component,
    whether it is given as a row, a flat array or a column."""
    D, K = 3, 4
    seed = 20260921
    point = np.array([1.0, -2.0, 0.5])

    row = VariationalPosterior(D, K, point.reshape(1, -1), rng=seed)
    flat = VariationalPosterior(D, K, point.copy(), rng=seed)
    column = VariationalPosterior(D, K, point.reshape(-1, 1), rng=seed)

    assert row.mu.shape == (D, K)
    assert np.allclose(row.mu, point.reshape(-1, 1), atol=1e-5)
    assert np.array_equal(flat.mu, row.mu)
    assert np.array_equal(column.mu, row.mu)


def test_sample_n_lower_1():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    x, i = vp.sample(0)
    assert x.shape == (0, 3)
    assert i.shape == (0,)
    assert np.issubdtype(i.dtype, np.integer)


@pytest.mark.parametrize("K", [1, 2])
@pytest.mark.parametrize("balance_flag", [False, True])
def test_sample_index_array_has_one_integer_per_row(K, balance_flag):
    """The second return value of ``sample`` holds, for each drawn row,
    the index of the component that generated it."""
    N = 11
    vp = VariationalPosterior(3, K, np.array([[5]]))
    vp.rng = np.random.default_rng(20260921)

    x, i = vp.sample(N, balance_flag=balance_flag)

    assert x.shape == (N, 3)
    assert i.shape == (N,)
    assert np.issubdtype(i.dtype, np.integer)


def test_sample_takes_a_whole_number_given_as_a_float():
    """``1e5``, the documented default of ``kl_div`` and ``mtv``, is a
    float; a count with a fractional part is refused."""
    vp = VariationalPosterior(2, 2, np.array([[5]]))
    vp.rng = np.random.default_rng(20260921)

    x, i = vp.sample(1e3)
    assert x.shape == (1000, 2)
    assert i.shape == (1000,)

    with pytest.raises(ValueError, match="whole number"):
        vp.sample(2.5)


def test_sample_refuses_a_negative_df():
    """A negative ``df`` asks for the product of univariate ``t``
    densities, which ``pdf`` evaluates and ``sample`` cannot draw from."""
    vp = VariationalPosterior(2, 2, np.array([[5]]))
    vp.rng = np.random.default_rng(20260921)

    assert vp.pdf(np.full((1, 2), 5.0), df=-3) > 0

    with pytest.raises(ValueError, match="product of univariate t"):
        vp.sample(10, df=-3)


def test_kl_div_and_mtv_take_a_whole_number_given_as_a_float():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp.rng = np.random.default_rng(20260921)
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp2.rng = np.random.default_rng(20260922)

    assert np.all(np.isfinite(vp.kl_div(vp2=vp2, N=1e4, gauss_flag=False)))
    assert np.all(np.isfinite(vp.mtv(vp2=vp2, N=1e4)))


def test_sample_default():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = int(1e6)
    x, i = vp.sample(N)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    assert 0 in i
    assert 1 in i


def test_sample_balance_no_extra():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = 10
    x, i = vp.sample(N, balance_flag=True)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert np.all(counts == N / 2)


def test_sample_balance_extra():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, balance_flag=True)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert np.all(np.isin(counts, np.array([N // 2, N // 2 + 1])))


def test_sample_one_k():
    vp = VariationalPosterior(3, 1, np.array([[5]]))
    N = 11
    x, i = vp.sample(N)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == N


def test_sample_one_k_df():
    vp = VariationalPosterior(3, 1, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, df=20)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == N


def test_sample_df():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    N = int(1e4)
    x, i = vp.sample(N, df=20)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    assert 0 in i
    assert 1 in i


def test_sample_no_orig_flag():
    vp = VariationalPosterior(3, 1, np.array([[5]]))
    N = 11
    x, i = vp.sample(N, orig_flag=False)
    assert np.all(x.shape == (N, 3))
    assert np.all(i.shape[0] == N)
    _, counts = np.unique(i, return_counts=True)
    assert counts[0] == N


def test_pdf_default_no_orig_flag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y = vp.pdf(x, orig_flag=False)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((N, 1)), rtol=1e-12, atol=1e-14
        )
    )
    log_y = vp.log_pdf(x, orig_flag=False)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y,
            np.log(0.002396970183585) * np.ones((N, 1)),
            rtol=1e-12,
            atol=1e-14,
        )
    )


def test_pdf_grad_default_no_orig_flag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    y, dy = vp.pdf(x, orig_flag=False, grad_flag=True)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(
        np.isclose(
            y, 0.002396970183585 * np.ones((N, 1)), rtol=1e-12, atol=1e-14
        )
    )
    assert dy.shape == x.shape
    assert np.isscalar(dy[0, 0])
    assert np.all(
        np.isclose(
            dy, 9.58788073433898 * np.ones((N, 3)), rtol=1e-12, atol=1e-14
        )
    )
    log_y, dlog_y = vp.log_pdf(x, orig_flag=False, grad_flag=True)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y,
            np.log(0.002396970183585) * np.ones((N, 1)),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert dlog_y.shape == x.shape
    assert np.isscalar(dlog_y[0, 0])
    assert np.all(
        np.isclose(
            dlog_y * y,
            9.58788073433898 * np.ones((N, 3)),
            rtol=1e-12,
            atol=1e-14,
        )
    )


def test_pdf_grad_log_flag_no_orig_flag():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    log_y, dlog_y = vp.pdf(x, orig_flag=False, log_flag=True, grad_flag=True)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y,
            np.log(0.002396970183585 * np.ones((N, 1))),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert dlog_y.shape == x.shape
    assert np.isscalar(dlog_y[0, 0])
    assert np.all(
        np.isclose(
            dlog_y,
            9.58788073433898 * np.ones((N, 3)) / np.exp(log_y),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    log_y_2, dlog_y_2 = vp.log_pdf(x, orig_flag=False, grad_flag=True)
    assert np.all(log_y_2 == log_y)
    assert np.all(dlog_y_2 == dlog_y)


def test_pdf_grad_orig_flag(mocker):
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.ones((N, D)) * 4.996
    transform = mocker.patch.object(
        ParameterTransformer,
        "__call__",
        side_effect=AssertionError("the transform should not be called"),
    )
    with pytest.raises(NotImplementedError, match="original space"):
        vp.pdf(x, grad_flag=True)
    with pytest.raises(NotImplementedError, match="original space"):
        vp.log_pdf(x, grad_flag=True)
    transform.assert_not_called()


def test_pdf_df_real_positive():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.99, 4.996], [10, 10])[:, np.newaxis] * np.ones((1, D))
    y = vp.pdf(x, orig_flag=False, df=10)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 0.01378581338784, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 743.0216137262, rtol=1e-12, atol=1e-14))
    log_y = vp.log_pdf(x, orig_flag=False, df=10)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(
            log_y[:10], np.log(0.01378581338784), rtol=1e-12, atol=1e-14
        )
    )
    assert np.all(
        np.isclose(log_y[10:], np.log(743.0216137262), rtol=1e-12, atol=1e-14)
    )


def test_pdf_df_real_negative():
    N = 20
    D = 3
    vp = VariationalPosterior(D, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * 5
    x = np.repeat([4.995, 4.99], [10, 10])[:, np.newaxis] * np.ones((1, D))
    y = vp.pdf(x, orig_flag=False, df=-2)
    assert y.shape == (N, 1)
    assert np.isscalar(y[0, 0])
    assert np.all(np.isclose(y[:10], 362.1287964795, rtol=1e-12, atol=1e-14))
    assert np.all(np.isclose(y[10:], 0.914743278670, rtol=1e-12, atol=1e-14))
    log_y = vp.log_pdf(x, orig_flag=False, df=-2)
    assert log_y.shape == (N, 1)
    assert np.isscalar(log_y[0, 0])
    assert np.all(
        np.isclose(log_y[:10], np.log(362.1287964795), rtol=1e-11, atol=1e-13)
    )
    assert np.all(
        np.isclose(log_y[10:], np.log(0.914743278670), rtol=1e-11, atol=1e-13)
    )


def test_pdf_heavy_tailed_pdf_gradient():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    x = np.ones((1, 3))
    with pytest.raises(NotImplementedError):
        vp.pdf(x, df=300, grad_flag=True)
    with pytest.raises(NotImplementedError):
        vp.pdf(x, df=-300, grad_flag=True)
    with pytest.raises(NotImplementedError):
        vp.log_pdf(x, df=300, grad_flag=True)
    with pytest.raises(NotImplementedError):
        vp.log_pdf(x, df=-300, grad_flag=True)


def test_pdf_outside_bounds():
    D = 2
    lb = np.ones((1, D)) * -3
    ub = np.ones((1, D)) * 3
    x0 = np.array([[2, 2], [-2, -2]])
    parameter_transformer = ParameterTransformer(D, lb, ub)

    vp = VariationalPosterior(D, 2, x0, parameter_transformer)
    vp.sigma = np.ones((1, 2))

    # outside or on bounds should be 0
    assert vp.pdf(lb, orig_flag=True) == 0
    assert vp.pdf(lb - 1e-3, orig_flag=True) == 0
    assert vp.pdf(ub, orig_flag=True) == 0
    assert vp.pdf(ub + 1e-3, orig_flag=True) == 0

    # inside should be more than 0
    assert vp.pdf(lb + 0.5, orig_flag=True) > 0
    assert vp.pdf(ub - 0.5, orig_flag=True) > 0

    # outside or on bounds should be -inf
    assert vp.log_pdf(lb, orig_flag=True) == -np.inf
    assert vp.log_pdf(lb - 1e-3, orig_flag=True) == -np.inf
    assert vp.log_pdf(ub, orig_flag=True) == -np.inf
    assert vp.log_pdf(ub + 1e-3, orig_flag=True) == -np.inf

    # inside should be finite
    assert np.all(np.isfinite(vp.log_pdf(lb + 0.5, orig_flag=True)))
    assert np.all(np.isfinite(vp.log_pdf(ub - 0.5, orig_flag=True)))


def test_pdf_whole_coordinates_given_as_integers():
    """A point whose coordinates are whole numbers has the same density
    however it is spelled: the transformed coordinates are computed in
    full precision, not truncated to the caller's integer type."""
    D = 2
    vp = VariationalPosterior(
        D,
        2,
        np.array([[3.0, 4.0]]),
        ParameterTransformer(D, np.zeros((1, D)), np.full((1, D), 10.0)),
    )
    vp.sigma = np.ones((1, 2))

    reference = vp.pdf(np.array([[3.0, 4.0]]))

    assert reference > 0
    assert vp.pdf(np.array([[3, 4]])) == reference
    assert vp.pdf([[3, 4]]) == reference
    assert vp.pdf(np.array([3, 4])) == vp.pdf(np.array([3.0, 4.0]))


def test_pdf_duplicate_log_flag():
    D = 2
    lb = np.ones((1, D)) * -3
    ub = np.ones((1, D)) * 3
    x0 = np.array([[2, 2], [-2, -2]])
    parameter_transformer = ParameterTransformer(D, lb, ub)

    vp = VariationalPosterior(D, 2, x0, parameter_transformer)
    vp.sigma = np.ones((1, 2))

    with pytest.raises(TypeError) as err:
        y = vp.log_pdf(lb + 0.5, log_flag=True)
    assert (
        "got multiple values for keyword argument 'log_flag'"
        in err.value.args[0]
    )


def test_set_parameters_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    theta_size = D * K + 2 * K + D
    rng = np.random.default_rng()
    theta = rng.random(theta_size)
    vp.optimize_weights = True
    vp.set_parameters(theta)
    assert vp.mu.shape == (D, K)
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    lamb = np.exp(theta[D * K + K : D * K + K + D])
    nl = np.sqrt(np.sum(lamb**2) / D)
    assert vp.sigma.shape == (1, K)
    assert np.all(
        vp.sigma == np.exp(theta[D * K : D * K + K]).reshape(1, -1) * nl
    )
    assert vp.lambd.shape == (D, 1)
    assert np.all(vp.lambd == np.array([lamb]).reshape(-1, 1) / nl)
    assert vp.w.shape == (1, K)
    w = np.exp(theta[-K:] - np.amax(theta[-K:]))
    w = w.reshape(1, -1) / np.sum(w)
    assert np.all(vp.w == w)


def test_set_parameters_not_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    theta_size = D * K + 2 * K + D
    rng = np.random.default_rng()
    theta = rng.random(theta_size)
    vp.optimize_weights = True
    vp.set_parameters(theta, raw_flag=False)
    assert vp.mu.shape == (D, K)
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    lamb = theta[D * K + K : D * K + K + D]
    nl = np.sqrt(np.sum(lamb**2) / D)
    assert vp.sigma.shape == (1, K)
    assert np.all(vp.sigma == theta[D * K : D * K + K].reshape(1, -1) * nl)
    assert vp.lambd.shape == (D, 1)
    assert np.all(vp.lambd == np.array([lamb]).reshape(-1, 1) / nl)
    assert vp.w.shape == (1, K)
    w = theta[-K:]
    w = w.reshape(1, -1) / np.sum(w)
    assert np.all(vp.w == w)


def test_set_parameters_not_raw_negative_error():
    """The entries that hold ``sigma``, ``lambd`` and the weights must be
    positive. The means are left positive here, so that the refusal can
    only come from those entries."""
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta_size = D * K + 2 * K + D
    rng = np.random.default_rng(20260921)
    theta = rng.random(theta_size)
    theta[D * K :] *= -1
    with pytest.raises(ValueError, match="must be positive"):
        vp.set_parameters(theta, raw_flag=False)


@pytest.mark.parametrize("D, K", [(3, 4), (2, 2)])
@pytest.mark.parametrize(
    "optimize_mu, optimize_sigma, optimize_lambd, optimize_weights",
    list(itertools.product([True, False], repeat=4)),
)
def test_set_parameters_not_raw_checks_the_constrained_entries(
    D, K, optimize_mu, optimize_sigma, optimize_lambd, optimize_weights
):
    """With ``raw_flag=False`` every entry that holds ``sigma``, ``lambd``
    or a weight is required to be positive, and those entries alone: the
    means are unconstrained, and a vector that carries none of the three
    leaves nothing to check."""
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_mu = optimize_mu
    vp.optimize_sigma = optimize_sigma
    vp.optimize_lambd = optimize_lambd
    vp.optimize_weights = optimize_weights

    blocks = []
    if optimize_mu:
        blocks.append(np.full(D * K, -1.0))
    n_unconstrained = int(sum(block.size for block in blocks))
    if optimize_sigma:
        blocks.append(np.full(K, 0.5))
    if optimize_lambd:
        blocks.append(np.full(D, 2.0))
    if optimize_weights:
        blocks.append(np.full(K, 1.0 / K))
    theta = np.concatenate(blocks) if blocks else np.array([])

    # Negative means are taken.
    vp.set_parameters(theta.copy(), raw_flag=False)

    # A single negative scale or weight is refused, wherever it sits.
    for idx in range(n_unconstrained, theta.size):
        negative = theta.copy()
        negative[idx] = -negative[idx]
        with pytest.raises(ValueError, match="must be positive"):
            vp.set_parameters(negative, raw_flag=False)


def test_get_parameters_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=True)
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    assert np.all(
        np.isclose(
            vp.sigma.flatten(),
            np.exp(theta[D * K : D * K + K]),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    assert np.all(
        vp.lambd.flatten() == np.exp(theta[D * K + K : D * K + K + D])
    )
    assert np.all(vp.w.flatten() == np.exp(theta[-K:]))


def test_get_parameters_not_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=False)
    assert np.all(
        vp.mu[: D * K] == np.reshape(theta[: D * K], (D, K), order="F")
    )
    assert np.all(vp.sigma.flatten() == theta[D * K : D * K + K])
    assert np.all(vp.lambd.flatten() == theta[D * K + K : D * K + K + D])
    assert np.all(vp.w.flatten() == theta[-K:])


def test_get_set_parameters_roundtrip():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=True)
    vp.set_parameters(theta, raw_flag=True)
    theta2 = vp.get_parameters(raw_flag=True)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_get_set_parameters_roundtrip_no_mu():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_mu = False
    theta = vp.get_parameters(raw_flag=True)
    vp.set_parameters(theta, raw_flag=True)
    theta2 = vp.get_parameters(raw_flag=True)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


def test_get_set_parameters_delete_mode():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    theta = vp.get_parameters(raw_flag=True)
    vp._mode = np.ones(D)
    assert hasattr(vp, "_mode")
    vp.set_parameters(theta, raw_flag=True)
    assert vp._mode is None


def test_get_set_parameters_roundtrip_non_raw():
    K = 2
    D = 3
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = True
    theta = vp.get_parameters(raw_flag=False)
    vp.set_parameters(theta, raw_flag=False)
    theta2 = vp.get_parameters(raw_flag=False)
    assert theta.shape == theta2.shape
    assert np.all(theta == theta2)


@pytest.mark.parametrize("raw_flag", [True, False])
@pytest.mark.parametrize("optimize_weights", [True, False])
def test_set_parameters_eta_matches_weights(raw_flag, optimize_weights):
    """``eta`` is the unbounded (softmax) parametrization of ``w``."""
    K = 3
    D = 2
    vp = VariationalPosterior(D, K, np.array([[5]]))
    vp.optimize_weights = optimize_weights
    vp.w = np.array([[0.2, 0.3, 0.5]])
    vp.eta = np.full((1, K), 7.0)
    theta = vp.get_parameters(raw_flag=raw_flag)

    vp.set_parameters(theta, raw_flag=raw_flag)

    assert vp.eta.shape == (1, K)
    softmax = np.exp(vp.eta - np.amax(vp.eta))
    softmax /= np.sum(softmax)
    assert np.allclose(softmax, vp.w, rtol=0, atol=1e-12)


def test_set_parameters_reference_regression():
    K = 2
    D = 2
    vp = VariationalPosterior(D, K)
    theta = vp.get_parameters().copy()
    theta[0] = -1e-7
    vp.set_parameters(theta)

    assert vp.mu[0, 0] == -1e-7

    # Make sure we don't get accidental reference to theta in the VP.
    theta[0] = -2e-7
    assert vp.mu[0, 0] == -1e-7


def test_moments_orig_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    mubar, sigma = vp.moments(N=int(1e6), cov_flag=True)
    x2, _ = vp.sample(N=int(1e6), orig_flag=True, balance_flag=True)
    assert mubar.shape == (1, 3)
    assert np.all(np.isclose(mubar, np.mean(x2, axis=0)))
    assert sigma.shape == (3, 3)
    assert np.all(np.isclose(sigma, np.cov(x2.T)))


def test_moments_orig_flag_one_dimensional_covariance():
    """The covariance of one parameter is a 1-by-1 matrix, as it is for
    every other number of parameters."""
    vp = VariationalPosterior(1, 2, np.array([[5]]))
    vp.rng = np.random.default_rng(20260921)

    mubar, cov = vp.moments(N=int(1e4), orig_flag=True, cov_flag=True)

    assert mubar.shape == (1, 1)
    assert cov.shape == (1, 1)


def test_moments_no_orig_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp.mu = np.ones((3, 2)) * [1, 4]
    mubar, sigma = vp.moments(N=1e6, cov_flag=True, orig_flag=False)
    assert mubar.shape == (1, 3)
    assert sigma.shape == (3, 3)
    assert np.all(mubar == 2.5)
    sigma2 = np.ones((3, 3)) * 2.25 + np.eye(3) * 1e-3**2
    assert np.all(sigma == sigma2)


def test_moments_no_orig_flag_2():
    # A second test with more unusual (non-ones) vp.lambd
    D = 6
    K = 3
    vp = VariationalPosterior(D, K)
    vp.mu = np.linspace(-3, 3, D * K).reshape([D, K], order="F")
    vp.sigma = np.atleast_2d(np.array(range(2, 5)))
    vp.lambd = np.atleast_2d(np.array(range(3, 9))).T
    vp.w = np.atleast_2d(np.array(range(1, 4)))
    vp.w = vp.w / np.sum(vp.w)

    mubar, sigma = vp.moments(N=1e6, cov_flag=True, orig_flag=False)
    path = Path(__file__).parent.joinpath(
        "test_moments_no_orig_flag_2_MATLAB.npz"
    )
    with np.load(path, allow_pickle=False) as matlab:
        assert mubar.shape == (1, 6)
        assert sigma.shape == (6, 6)
        assert np.allclose(mubar, matlab["mubar"])
        assert np.allclose(sigma, matlab["sigma"])


def test_moments_no_cov_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    mubar = vp.moments(N=1e6, orig_flag=False)
    assert mubar.shape == (1, 3)


def test_mode_exists_already():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp._mode = np.ones(3)
    mode2 = vp.mode()
    assert np.all(mode2 == vp._mode)


def test_mode_is_recomputed_when_n_opts_is_given():
    """A stored mode answers a call that gives no ``n_opts``; a call that
    gives one runs the search."""
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp.sigma = np.ones((1, 2))
    vp.rng = np.random.default_rng(20260921)
    stale = np.full(3, 42.0)
    vp._mode = stale.copy()

    assert np.all(vp.mode() == stale)

    computed = vp.mode(n_opts=1)
    assert np.allclose(computed, 5.0, atol=1e-3)


def test_mode_leaves_the_random_stream_alone():
    """The candidates of the mode search come from a copy of the
    posterior's generator, so a call neither advances the stream a run
    shares nor depends on where that stream stands."""
    vp = get_matlab_vp()
    vp.rng = np.random.default_rng(20260921)
    state = vp.rng.bit_generator.state

    first = vp.mode(n_opts=2)

    assert vp.rng.bit_generator.state == state
    second = vp.mode(n_opts=2)
    assert np.array_equal(first, second)

    # The posterior draws as if the mode had never been computed.
    after = vp.sample(5, orig_flag=False)[0]
    vp.rng = np.random.default_rng(20260921)
    assert np.array_equal(vp.sample(5, orig_flag=False)[0], after)


def test_get_parameters_clears_the_stored_mode():
    """``get_parameters`` normalizes the parameters in place and drops the
    stored mode, as ``misc/rescale_params.m:39-40`` does."""
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    vp._mode = np.ones(3)

    vp.get_parameters()

    assert vp._mode is None


def test_mode_no_orig_flag():
    vp = get_matlab_vp()
    assert np.all(
        np.isclose([0.0540, -0.1818], vp.mode(orig_flag=False), atol=1e-4)
    )


def test_mode_orig_flag():
    vp = get_matlab_vp()
    assert np.all(
        np.isclose([0.0540, -0.1818], vp.mode(orig_flag=True), atol=1e-4)
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_mode_one_dimensional(bounded):
    """``mode()`` of a one-dimensional posterior returns one point of the
    original space, where the density is at its highest."""
    if bounded:
        parameter_transformer = ParameterTransformer(
            1, np.array([[-3.0]]), np.array([[3.0]])
        )
        grid = np.linspace(-2.99, 2.99, 4001).reshape(-1, 1)
    else:
        parameter_transformer = ParameterTransformer(1)
        grid = np.linspace(-3.0, 4.0, 4001).reshape(-1, 1)
    vp = VariationalPosterior(1, 1, np.zeros((1, 1)))
    vp.parameter_transformer = parameter_transformer
    vp.mu = np.array([[0.7]])
    vp.sigma = np.array([[0.5]])
    vp.lambd = np.ones((1, 1))
    vp.w = np.ones((1, 1))
    vp.rng = np.random.default_rng(20260921)

    mode = vp.mode()

    assert mode.shape == (1,)
    best = grid[np.argmax(vp.pdf(grid))]
    assert np.isclose(mode, best, atol=1e-2)
    assert vp.pdf(mode) >= vp.pdf(best)
    if not bounded:
        # Of a single component, the mode is its mean.
        assert np.isclose(mode, 0.7, atol=1e-4)


def test_mode_starts_inside_the_box_it_searches(mocker):
    """The starting point is clamped to the box handed to the optimizer,
    the original bounds shrunk by ``sqrt(eps)`` (``vbmc_mode.m:39-41``)."""
    D = 2
    lb, ub = np.zeros((1, D)), np.ones((1, D))
    vp = VariationalPosterior(D, 1, np.zeros((1, D)))
    vp.parameter_transformer = ParameterTransformer(D, lb, ub)
    vp.mu = np.full((D, 1), -40.0)
    vp.sigma = np.array([[1.0]])
    vp.lambd = np.ones((D, 1))
    vp.w = np.ones((1, 1))
    vp.rng = np.random.default_rng(20260921)

    vp_module = importlib.import_module(
        "pyvbmc.variational_posterior.variational_posterior"
    )
    starting_points = []
    original_minimize = vp_module.minimize

    def record(*args, **kwargs):
        starting_points.append(np.array(kwargs["x0"], copy=True))
        return original_minimize(*args, **kwargs)

    mocker.patch.object(vp_module, "minimize", record)
    vp.mode()

    offset = np.sqrt(np.finfo(float).eps)
    assert len(starting_points) == 1
    x0 = starting_points[0]
    assert np.all(x0 >= lb.ravel() + offset)
    assert np.all(x0 <= ub.ravel() - offset)


def test_mtv_not_enough_arguments():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.mtv()


def test_mtv_vp_identical():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[1, 0]])
    mtv = vp1.mtv(vp2)
    assert np.isclose(0, mtv, atol=1e-2)


def test_mtv_vp_no_overlap():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0, 1]])
    mtv = vp1.mtv(vp2)
    assert np.isclose(1, mtv, atol=1e-2)


def test_mtv_sample_identical():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[1, 0]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(0, mtv, atol=1e-2)


def test_mtv_sample_no_overlap():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 100000]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0, 1]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(1, mtv, atol=1e-2)


def test_mtv_sample_some_overlap():
    vp1 = VariationalPosterior(1, 1, np.array([[5]]))
    vp1.mu = np.zeros((1, 1))
    vp1.sigma = np.array([[1]])
    vp2 = VariationalPosterior(1, 2, np.array([[5]]))
    vp2.mu = np.array([[0, 10000]])
    vp2.sigma = np.ones((1, 2))
    vp2.w = np.array([[0.5, 0.5]])
    samples, _ = vp2.sample(int(1e5))
    mtv = vp1.mtv(samples=samples, N=int(1e5))
    assert np.isclose(0.5, mtv, atol=1e-2)


def test_kl_div_missing_params():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kl_div()


def test_kl_div_no_gaussianflag_and_samples():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kl_div(samples=np.ones(3), gauss_flag=False)


def test_kl_div_two_vp_identical_gauss_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    kl_divs = vp.kl_div(vp2=vp, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kl_divs, atol=1e-4))


def test_kl_div_two_vp_identical_samples_gauss_flag():
    vp = VariationalPosterior(3, 2, np.array([[5]]))
    samples, _ = vp.sample(int(1e5))
    kl_divs = vp.kl_div(samples=samples, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kl_divs, atol=1e-3))


def test_kl_div_two_vp_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.zeros((1, 1))
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.ones((1, 1)) * 10
    vp2.sigma = np.ones((1, 1))
    kl_divs = vp.kl_div(vp2=vp2, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(50, kl_divs, atol=5e-1))


def test_kl_div_two_vp_samples_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.ones((1, 1)) * 0.5
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    samples, _ = vp2.sample(int(1e6))
    kl_divs = vp.kl_div(samples=samples, gauss_flag=True, N=int(1e6))
    assert np.all(np.isclose(np.ones(2) * 0.1244, kl_divs, atol=1e-2))


def test_kl_div_samples_mean_is_taken_per_coordinate():
    """The moments of the given samples are those of the sample matrix:
    one mean per coordinate, as ``mean(vp2,1)`` in ``vbmc_kldiv.m:63``."""
    D = 3
    seed = 20260921
    N = int(2e4)
    vp = VariationalPosterior(D, 1, np.zeros((1, D)))
    vp.parameter_transformer = ParameterTransformer(D)
    vp.mu = np.array([[10.0], [-5.0], [2.0]])
    vp.sigma = np.ones((1, 1))
    vp.lambd = np.ones((D, 1))
    vp.w = np.ones((1, 1))

    # Samples of the posterior itself: both divergences are near zero.
    vp.rng = np.random.default_rng(seed)
    samples, _ = vp.sample(N)
    kl_divs = vp.kl_div(samples=samples, N=N, gauss_flag=True)
    assert np.all(kl_divs < 1e-2)

    # Samples of a shifted Gaussian: the divergence of the posterior's
    # moments from the sample moments, coordinate by coordinate.
    shifted = samples + np.array([0.5, -1.0, 2.0])
    vp.rng = np.random.default_rng(seed + 1)
    kl_divs = vp.kl_div(samples=shifted, N=N, gauss_flag=True)
    vp.rng = np.random.default_rng(seed + 1)
    q1mu, q1sigma = vp.moments(N, True, True)
    expected = kl_div_mvn(
        q1mu, q1sigma, np.mean(shifted, axis=0), np.cov(shifted.T)
    )
    assert np.array_equal(kl_divs, np.maximum(0, expected))
    assert np.all(kl_divs > 1e-2)


def test_kl_div_two_vp_identical_no_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    kl_divs = vp.kl_div(vp2=vp, gauss_flag=False, N=int(1e6))
    assert np.all(np.isclose(np.zeros(2), kl_divs, atol=1e-4))


def test_kl_div_two_vp_no_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    vp2 = VariationalPosterior(1, 1, np.array([[5]]))
    vp.mu = np.ones((1, 1)) * 10
    vp.sigma = np.ones((1, 1))
    vp2.mu = np.zeros((1, 1))
    vp2.sigma = np.ones((1, 1))
    kl_divs = vp.kl_div(vp2=vp2, gauss_flag=False, N=int(1e6))
    assert np.all(np.isclose(50, kl_divs, atol=5e-1))


def test_kl_div_no_samples_gauss_flag():
    vp = VariationalPosterior(1, 1, np.array([[5]]))
    with pytest.raises(ValueError):
        vp.kl_div(vp, gauss_flag=True, N=0)


def test_soft_bounds_1():
    D = 2
    K = 1
    vp = VariationalPosterior(D, K)
    assert vp.bounds is None

    # use a fake options struct
    options = {
        "tol_con_loss": 0.01,
        "tol_weight": 1e-2,
        "weight_penalty": 0.1,
        "tol_length": 1e-6,
    }

    # Make up some fake data.
    X = np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T

    theta_bnd = vp.get_bounds(X, options)

    assert vp.bounds is not None
    assert np.all(vp.bounds["mu_lb"] == 0)
    assert np.all(vp.bounds["mu_ub"] == 1)
    assert np.all(vp.bounds["lnscale_lb"] == np.log(options["tol_length"]))
    assert np.all(vp.bounds["lnscale_ub"] == 0)
    assert vp.bounds["eta_lb"] == np.log(0.5 * options["tol_weight"])
    assert vp.bounds["eta_ub"] == 0

    assert theta_bnd["tol_con"] == options["tol_con_loss"]
    assert theta_bnd["weight_threshold"] == max(
        1 / (4 * K), options["tol_weight"]
    )
    assert theta_bnd["weight_penalty"] == options["weight_penalty"]


def test_soft_bounds_follow_the_training_inputs():
    """The box is a function of the training inputs of the call, so a
    later call on a narrower set gives the narrower box."""
    D = 2
    options = {
        "tol_con_loss": 0.01,
        "tol_weight": 1e-2,
        "weight_penalty": 0.1,
        "tol_length": 1e-6,
    }
    wide = np.array([np.linspace(-5.0, 5.0, 10)] * D).T
    narrow = np.array([np.linspace(-1.0, 2.0, 10)] * D).T

    vp = VariationalPosterior(D, 2)
    vp.get_bounds(wide, options)
    after_narrowing = vp.get_bounds(narrow, options)
    fresh = VariationalPosterior(D, 2).get_bounds(narrow, options)

    assert np.array_equal(after_narrowing["lb"], fresh["lb"])
    assert np.array_equal(after_narrowing["ub"], fresh["ub"])
    assert np.array_equal(vp.bounds["mu_lb"], np.min(narrow, axis=0))
    assert np.array_equal(vp.bounds["mu_ub"], np.max(narrow, axis=0))


def test_soft_bounds_2():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)

    options = {
        "tol_con_loss": 0.01,
        "tol_weight": 1e-2,
        "weight_penalty": 0.1,
        "tol_length": 1e-6,
    }
    base_path = Path(__file__).parent
    X = np.loadtxt(open(base_path.joinpath("X.txt"), "rb"), delimiter=",")
    path = Path(__file__).parent.joinpath("mu.txt")
    vp.mu = np.loadtxt(open(base_path.joinpath("mu.txt"), "rb"), delimiter=",")

    theta_bnd = vp.get_bounds(X, options)

    bnd_lb = np.loadtxt(
        open(base_path.joinpath("bnd_lb.txt"), "rb"), delimiter=","
    )
    assert np.allclose(theta_bnd["lb"], bnd_lb)

    bnd_ub = np.loadtxt(
        open(base_path.joinpath("bnd_ub.txt"), "rb"), delimiter=","
    )
    assert np.allclose(theta_bnd["ub"], bnd_ub)

    assert theta_bnd["tol_con"] == 0.0100
    assert theta_bnd["weight_threshold"] == 0.1250
    assert theta_bnd["weight_penalty"] == 0.1000


def test_plot():
    """
    This is a really naive test of the plotting as everything else is
    complicated.
    """
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    test_title = "Test title"
    fig = vp.plot(title=test_title)
    assert fig._suptitle.get_text() == test_title
    assert len(fig.axes) == D * D


def test__str__and__repr__():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    assert "num. components = 2" in vp.__str__()
    assert "self.K = 2" in vp.__repr__()
