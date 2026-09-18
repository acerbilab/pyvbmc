"""The quasi-Monte Carlo importance nodes of the VIQR estimate
(``active_importance_sampling_qmc``): the node draw, the component
ordering with its fallbacks, and the switch inside
``active_importance_sampling``."""

import importlib
import os.path

import numpy as np
import pytest
from scipy.stats import qmc

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.testing.vbmc.test_active_importance_sampling import _scenario
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.active_importance_sampling import (
    active_importance_sampling,
    qmc_component_order,
    qmc_vp_nodes,
)
from pyvbmc.vbmc.options import Options

# ``pyvbmc.vbmc`` re-exports the function under the module's name, so the
# module itself is reached through the import system.
ais = importlib.import_module("pyvbmc.vbmc.active_importance_sampling")

N_NODES = 96


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """``VariationalPosterior.__init__`` draws its seed from the global
    ``np.random`` state; leave that state as each test found it."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _options(D, **user_options):
    """Options from both ``.ini`` files, as ``VBMC`` builds them."""
    vbmc_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "vbmc"
    )
    basic = os.path.join(vbmc_path, "option_configs", "basic_vbmc_options.ini")
    advanced = os.path.join(
        vbmc_path, "option_configs", "advanced_vbmc_options.ini"
    )
    options = Options(
        basic, evaluation_parameters={"D": D}, user_options=user_options
    )
    options.load_options_file(advanced, evaluation_parameters={"D": D})
    return options


def _vp(D, K, mu, w=None, sigma=None, lambd=None):
    """A VP with the given parameters (``mu`` given as ``(K, D)``)."""
    vp = VariationalPosterior(D=D, K=K)
    vp.mu = np.asarray(mu, dtype=float).T.reshape(D, K)
    vp.w = (
        np.full((1, K), 1.0 / K)
        if w is None
        else np.asarray(w, dtype=float).reshape(1, K)
    )
    vp.sigma = (
        np.ones((1, K))
        if sigma is None
        else np.asarray(sigma, dtype=float).reshape(1, K)
    )
    vp.lambd = (
        np.ones((D, 1))
        if lambd is None
        else np.asarray(lambd, dtype=float).reshape(D, 1)
    )
    return vp


def _bump_integral(vp, c, ell):
    """Exact ``E_vp[exp(-|x - c|^2 / (2 ell^2))]`` for a diagonal mixture."""
    w = vp.w.ravel()
    mu = vp.mu.T
    s2 = (vp.lambd.reshape(1, -1) * vp.sigma.T) ** 2
    var = s2 + ell**2
    per_component = np.prod(
        np.sqrt(ell**2 / var) * np.exp(-((mu - c) ** 2) / (2 * var)), axis=1
    )
    return float(w @ per_component)


def _counts(comp, K):
    return np.bincount(comp, minlength=K)


# --- the node draw ---------------------------------------------------------


def test_nodes_shape_dtype_finite_and_seeded():
    vp = _vp(
        3,
        4,
        [[0, 0, 0], [1, -1, 2], [-2, 1, 0], [0.5, 0.5, -1]],
        w=[0.4, 0.3, 0.2, 0.1],
        sigma=[1.0, 0.5, 2.0, 0.8],
        lambd=[1.0, 0.5, 2.0],
    )
    X, comp = qmc_vp_nodes(vp, N_NODES, np.random.default_rng(11))
    assert X.shape == (N_NODES, 3) and X.dtype == np.float64
    assert comp.shape == (N_NODES,) and np.all(np.isfinite(X))
    X_again, comp_again = qmc_vp_nodes(vp, N_NODES, np.random.default_rng(11))
    assert np.array_equal(X, X_again) and np.array_equal(comp, comp_again)
    X_other, __ = qmc_vp_nodes(vp, N_NODES, np.random.default_rng(12))
    assert not np.array_equal(X, X_other)
    # Successive draws from one generator give fresh scrambles, and the
    # draw advances the generator's state, so a restored state replays it.
    rng = np.random.default_rng(13)
    first, __ = qmc_vp_nodes(vp, N_NODES, rng)
    state = rng.bit_generator.state
    second, __ = qmc_vp_nodes(vp, N_NODES, rng)
    assert not np.array_equal(first, second)
    rng.bit_generator.state = state
    replayed, __ = qmc_vp_nodes(vp, N_NODES, rng)
    assert np.array_equal(second, replayed)


def test_component_counts_are_within_two_of_proportional():
    """The first 96 points of a scrambled Sobol' sequence are a balanced
    block of 64 and a balanced block of 32; each block puts one point in
    every dyadic cell of its resolution, so a slab of width ``w`` holds
    ``96 w`` points up to two per block, four in all. Monte Carlo counts
    have standard deviation 4.5 for the half-weight component."""
    w = np.array([0.5, 0.3, 0.2])
    vp = _vp(2, 3, [[0, 0], [3, 0], [0, 3]], w=w)
    for seed in range(20):
        __, comp = qmc_vp_nodes(vp, N_NODES, np.random.default_rng(seed))
        assert np.all(np.abs(_counts(comp, 3) - N_NODES * w) <= 4)


def test_small_weight_component_is_never_forced_a_node():
    vp = _vp(2, 3, [[0, 0], [3, 0], [0, 3]], w=[0.99, 0.005, 0.005])
    counts = np.array(
        [
            _counts(qmc_vp_nodes(vp, N_NODES, np.random.default_rng(s))[1], 3)
            for s in range(40)
        ]
    )
    assert counts[:, 1:].max() <= 2
    assert (counts[:, 1:] == 0).any()
    # About 96 * 0.005 = 0.48 nodes per small component on average.
    assert 0.2 < counts[:, 1:].mean() < 0.9


def test_one_component_more_components_than_nodes_and_one_dimension():
    vp1 = _vp(3, 1, [[0.5, -0.5, 1.0]], sigma=[0.7])
    X, comp = qmc_vp_nodes(vp1, N_NODES, np.random.default_rng(0))
    assert np.all(comp == 0) and np.all(np.isfinite(X))
    rng = np.random.default_rng(1)
    K = 150
    vp_many = _vp(
        2, K, rng.uniform(-3, 3, size=(K, 2)), w=rng.dirichlet(np.ones(K))
    )
    X, comp = qmc_vp_nodes(vp_many, N_NODES, np.random.default_rng(2))
    assert X.shape == (N_NODES, 2) and np.all(np.isfinite(X))
    assert _counts(comp, K).max() <= 2 + int(
        np.ceil(N_NODES * vp_many.w.max())
    )
    vp_1d = _vp(1, 3, [[-1.0], [0.0], [2.0]], sigma=[0.3, 0.5, 0.2])
    X, __ = qmc_vp_nodes(vp_1d, N_NODES, np.random.default_rng(3))
    assert X.shape == (N_NODES, 1) and np.all(np.isfinite(X))


def test_unbiased_and_less_variable_than_monte_carlo():
    rng = np.random.default_rng(2024)
    D, K = 3, 4
    vp = _vp(
        D,
        K,
        rng.uniform(-2, 2, size=(K, D)),
        w=rng.dirichlet(2.0 * np.ones(K)),
        sigma=rng.uniform(0.5, 1.5, size=K),
        lambd=[1.0, 0.5, 2.0],
    )
    c = vp.mu.T[np.argmax(vp.w)] + 0.5
    ell = 1.0
    exact = _bump_integral(vp, c, ell)
    f = lambda X: np.exp(-np.sum((X - c) ** 2, axis=1) / (2 * ell**2))
    n_rep = 300
    est_qmc = np.array(
        [
            f(qmc_vp_nodes(vp, N_NODES, np.random.default_rng(s))[0]).mean()
            for s in range(n_rep)
        ]
    )
    vp.rng = np.random.default_rng(7)
    est_mc = np.array(
        [
            f(vp.sample(N_NODES, orig_flag=False)[0]).mean()
            for _ in range(n_rep)
        ]
    )
    se_qmc = est_qmc.std(ddof=1) / np.sqrt(n_rep)
    assert abs(est_qmc.mean() - exact) < 4 * se_qmc
    rmse_qmc = np.sqrt(np.mean((est_qmc - exact) ** 2))
    rmse_mc = np.sqrt(np.mean((est_mc - exact) ** 2))
    assert rmse_qmc < 0.8 * rmse_mc


def test_endpoints_of_the_unit_cube_give_finite_nodes(monkeypatch):
    vp = _vp(2, 3, [[0, 0], [1, 1], [2, 2]])
    corners = np.zeros((128, 3))
    corners[64:] = 1.0
    monkeypatch.setattr(
        qmc.Sobol, "random_base2", lambda self, m: corners[: 2**m]
    )
    X, comp = qmc_vp_nodes(vp, N_NODES, np.random.default_rng(0))
    assert np.all(np.isfinite(X))
    assert np.all(comp[:64] == qmc_component_order(vp)[0])


# --- the component order -----------------------------------------------------


def test_one_or_two_components_keep_their_index_order():
    assert np.array_equal(qmc_component_order(_vp(2, 1, [[3, 3]])), [0])
    vp2 = _vp(2, 2, [[5, 5], [-5, -5]], w=[0.2, 0.8])
    assert np.array_equal(qmc_component_order(vp2), [0, 1])


def test_order_follows_the_leading_axis_with_a_fixed_sign():
    direction = np.array([1.0, 2.0, -1.0])
    t = np.array([3.0, -1.0, 0.0, 2.0])
    mu = t[:, None] * direction + 0.05 * np.array(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, -1, 0]]
    )
    vp = _vp(3, 4, mu)
    order = qmc_component_order(vp)
    # The axis has its largest entry (the second) positive, so the
    # projection increases with t.
    assert np.array_equal(order, np.argsort(t))
    vp_mirror = _vp(3, 4, -mu)
    assert np.array_equal(qmc_component_order(vp_mirror), order[::-1])


def test_order_measures_the_means_in_units_of_lambd():
    # Along the first coordinate the means are 1 apart with lambd 0.01 (a
    # hundred widths); along the second 10 apart with lambd 100 (a tenth
    # of a width) and in the opposite order. The raw spread is larger
    # along the second coordinate; the standardized spread along the
    # first, which must decide the order.
    mu = np.array([[0.0, 20.0], [1.0, 10.0], [2.0, 0.0]])
    vp = _vp(2, 3, mu, lambd=[0.01, 100.0])
    assert np.array_equal(qmc_component_order(vp), [0, 1, 2])


def test_coincident_means_fall_back_to_the_component_scale():
    vp = _vp(2, 3, [[1, 1], [1, 1], [1, 1]], sigma=[0.5, 0.1, 0.3])
    assert np.array_equal(qmc_component_order(vp), [1, 2, 0])


def test_tied_leading_eigenvalues_fall_back_to_a_coordinate_axis():
    # Four equal-weight components on the corners of a square: the two
    # eigenvalues are equal, so the order follows the first coordinate,
    # ties by index.
    vp = _vp(2, 4, [[-1, -1], [1, -1], [1, 1], [-1, 1]])
    assert np.array_equal(qmc_component_order(vp), [0, 3, 1, 2])


def test_failed_or_non_finite_eigendecomposition_falls_back(monkeypatch):
    mu = np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, -1.0, 0.5]])
    vp = _vp(3, 3, mu, w=[0.5, 0.3, 0.2])
    # The largest weighted variance of the standardized means is along the
    # second coordinate, so the fallback sorts by it: 3.0 > 0.0 > -1.0.
    expected = [2, 0, 1]

    def raising(a):
        raise np.linalg.LinAlgError("no convergence")

    monkeypatch.setattr(np.linalg, "eigh", raising)
    assert np.array_equal(qmc_component_order(vp), expected)

    def non_finite(a):
        return np.full(a.shape[0], np.nan), np.full(a.shape, np.nan)

    monkeypatch.setattr(np.linalg, "eigh", non_finite)
    assert np.array_equal(qmc_component_order(vp), expected)


# --- the switch inside active_importance_sampling -------------------------


def test_switch_off_keeps_the_monte_carlo_draw(monkeypatch):
    vp, gp, __ = _scenario()
    options = _options(vp.D, active_importance_sampling_mcmc_samples=10)
    assert options["active_importance_sampling_qmc"] is False

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("qmc_vp_nodes called with the switch off")

    monkeypatch.setattr(ais, "qmc_vp_nodes", must_not_be_called)
    active_is = active_importance_sampling(vp, gp, AcqFcnVIQR(), options)
    assert active_is["X"].shape == (10, vp.D)


def test_options_saved_before_the_switch_existed_take_the_mc_path():
    vp, gp, __ = _scenario()
    options = _options(vp.D, active_importance_sampling_mcmc_samples=10)
    del options["active_importance_sampling_qmc"]
    del options["active_importance_sampling_qmc_samples"]
    active_is = active_importance_sampling(vp, gp, AcqFcnVIQR(), options)
    assert active_is["X"].shape == (10, vp.D)


def test_unknown_component_order_mode_is_refused(monkeypatch):
    vp, gp, __ = _scenario()
    options = _options(
        vp.D,
        active_importance_sampling_qmc=True,
        active_importance_sampling_qmc_samples=N_NODES,
    )
    monkeypatch.setattr(ais, "_QMC_COMPONENT_ORDER", "random")
    with pytest.raises(ValueError, match="_QMC_COMPONENT_ORDER"):
        active_importance_sampling(vp, gp, AcqFcnVIQR(), options)


def test_switch_on_draws_qmc_nodes_and_leaves_the_rest_unchanged(
    monkeypatch,
):
    vp, gp, __ = _scenario()
    D = vp.D
    options_qmc = _options(
        D,
        active_importance_sampling_qmc=True,
        active_importance_sampling_qmc_samples=N_NODES,
    )
    vp.rng = np.random.default_rng(5)
    active_qmc = active_importance_sampling(vp, gp, AcqFcnVIQR(), options_qmc)
    assert active_qmc["X"].shape == (N_NODES, D)
    assert active_qmc["X"].dtype == np.float64
    expected, __ = qmc_vp_nodes(vp, N_NODES, np.random.default_rng(5))
    assert np.array_equal(active_qmc["X"], expected)
    # Equal node weights: VIQR's base density is constant, so after
    # normalization every log weight is the same value.
    ln_w = active_qmc["ln_weights"]
    assert ln_w.shape[1] == N_NODES
    np.testing.assert_allclose(ln_w, ln_w.flat[0], rtol=0, atol=1e-12)

    # Everything downstream of the draw is the production path: feeding
    # the same nodes through the Monte Carlo path gives identical output.
    options_mc = _options(D, active_importance_sampling_mcmc_samples=N_NODES)
    monkeypatch.setattr(
        VariationalPosterior,
        "sample",
        lambda self, N, orig_flag=True, **kwargs: (expected, None),
    )
    active_mc = active_importance_sampling(vp, gp, AcqFcnVIQR(), options_mc)
    for key in ("X", "f_s2", "ln_weights", "K_Xa_X", "C_tmp"):
        assert np.array_equal(active_qmc[key], active_mc[key]), key


def test_index_order_mode_keeps_the_vp_component_order(monkeypatch):
    vp, gp, __ = _scenario()
    vp.K = 3
    vp.mu = np.array([[2.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    vp.w = np.array([[0.2, 0.5, 0.3]])
    vp.sigma = np.ones((1, 3))
    options = _options(
        vp.D,
        active_importance_sampling_qmc=True,
        active_importance_sampling_qmc_samples=N_NODES,
    )
    assert not np.array_equal(qmc_component_order(vp), np.arange(3))
    monkeypatch.setattr(ais, "_QMC_COMPONENT_ORDER", "index")
    vp.rng = np.random.default_rng(9)
    active_is = active_importance_sampling(vp, gp, AcqFcnVIQR(), options)
    expected, __ = qmc_vp_nodes(
        vp, N_NODES, np.random.default_rng(9), order=np.arange(3)
    )
    assert np.array_equal(active_is["X"], expected)
