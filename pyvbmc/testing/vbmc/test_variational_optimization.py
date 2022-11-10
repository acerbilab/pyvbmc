from pathlib import Path

import gpyreg as gpr
import numpy as np
from scipy.stats import multivariate_normal, norm

from pyvbmc.stats import kl_div_mvn
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import Options
from pyvbmc.vbmc.variational_optimization import (
    _gp_log_joint,
    _neg_elcbo,
    _soft_bound_loss,
    _vp_bound_loss,
    optimize_vp,
    update_K,
)


def setup_options(D: int, user_options: dict = None):
    if user_options is None:
        user_options = {}

    basic_path = Path(__file__).parent.parent.parent.joinpath(
        "vbmc", "option_configs", "basic_vbmc_options.ini"
    )
    options = Options(
        basic_path, evaluation_parameters={"D": D}, user_options=user_options
    )
    advanced_path = Path(__file__).parent.parent.parent.joinpath(
        "vbmc", "option_configs", "advanced_vbmc_options.ini"
    )
    options.load_options_file(advanced_path, evaluation_parameters={"D": D})
    return options


def test_soft_bound_loss():
    D = 3
    x1 = np.zeros((D,))
    slb = np.full((D,), -10)
    sub = np.full((D,), 10)

    L1 = _soft_bound_loss(x1, slb, sub)
    assert np.isclose(L1, 0.0)

    L2, dL2 = _soft_bound_loss(x1, slb, sub, compute_grad=True)
    assert np.isclose(L2, 0.0)
    assert np.allclose(dL2, 0.0)

    x2 = np.zeros((D,))
    x2[0] = 15.0
    x2[1] = -20.0
    L3 = _soft_bound_loss(x2, slb, sub)
    assert np.isclose(L3, 156250.0)

    L4, dL4 = _soft_bound_loss(x2, slb, sub, compute_grad=True)
    assert np.isclose(L4, 156250.0)
    assert np.isclose(dL4[0], 12500.0)
    assert np.isclose(dL4[1], -25000.0)
    assert np.allclose(dL4[2:], 0.0)


def test_update_K():
    D = 2

    # Dummy option state and iteration history.
    optim_state = {"vp_K": 2, "n_eff": 10, "warmup": True}
    iteration_history = {}

    # Load options
    options = setup_options(D)

    # Check right after start up we do nothing.
    assert update_K(optim_state, iteration_history, options) == 2

    # Similar, but check new branch runs.
    optim_state["warmup"] = False
    optim_state["iter"] = 0
    assert update_K(optim_state, iteration_history, options) == 2

    # Check that nothing is done right after warmup.
    optim_state["iter"] = 5
    optim_state["recompute_var_post"] = True
    iteration_history = {
        "elbo": np.array([-0.0996, 0.0094, -0.021, -0.0280, 0.0368]),
        "elbo_sd": np.array([0.0081, 0.0043, 0.0009, 0.0011, 0.0012]),
        "warmup": np.array([True, True, True, True, False]),
        "pruned": np.zeros((5,)),
        "r_index": np.array([np.inf, np.inf, 0.1773, 0.1407, 0.3420]),
    }
    assert update_K(optim_state, iteration_history, options) == 2

    # Check adding a new component.
    optim_state["iter"] = 7
    optim_state["recompute_var_post"] = True
    iteration_history = {
        "elbo": np.array(
            [-0.0996, 0.0094, -0.021, -0.0280, 0.0368, 0.0239, 0.0021]
        ),
        "elbo_sd": np.array(
            [0.0081, 0.0043, 0.0009, 0.0011, 0.0012, 0.0008, 0.0000]
        ),
        "warmup": np.array([True, True, True, True, False, False, False]),
        "pruned": np.zeros((7,)),
        "r_index": np.array(
            [np.inf, np.inf, 0.1773, 0.1407, 0.3420, 0.1422, 0.1222]
        ),
    }
    assert update_K(optim_state, iteration_history, options) == 3

    # Check that allowing bonus works.
    optim_state["recompute_var_post"] = False
    assert update_K(optim_state, iteration_history, options) == 5

    # Check that if we pruned recently we don't do anything.
    iteration_history["pruned"][-1] = 1
    assert update_K(optim_state, iteration_history, options) == 2


def test_gp_log_joint():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    base_path = Path(__file__).parent
    vp.mu = np.loadtxt(open(base_path.joinpath("mu.txt"), "rb"), delimiter=",")
    vp.eta = vp.eta.flatten()
    vp.lambd = vp.lambd.flatten()

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = np.loadtxt(open(base_path.joinpath("X.txt"), "rb"), delimiter=",")
    y = np.loadtxt(
        open(base_path.joinpath("y.txt"), "rb"), delimiter=","
    ).reshape((-1, 1))
    hyp = np.loadtxt(open(base_path.joinpath("hyp.txt"), "rb"), delimiter=",")
    gp.update(X_new=X, y_new=y, hyp=hyp)

    G, dG, varG, dvarG, var_ss, I_sk, J_sjk = _gp_log_joint(
        vp, gp, False, True, True, True, True
    )

    assert np.isclose(G, -0.461812484952867)
    assert dG is None
    assert np.isclose(varG, 6.598768992700180e-05)
    assert dvarG is None
    assert np.isclose(var_ss, 1.031705745662353e-04)

    G, dG, varG, dvarG, var_ss = _gp_log_joint(
        vp, gp, True, True, True, False, False
    )
    matlab_dG = np.loadtxt(
        open(base_path.joinpath("dG_gp_log_joint.txt"), "rb"), delimiter=","
    )
    assert np.allclose(dG, matlab_dG)
    assert np.isclose(G, -0.461812484952867)


def test_neg_elcbo():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    base_path = Path(__file__).parent
    vp.mu = np.loadtxt(open(base_path.joinpath("mu.txt"), "rb"), delimiter=",")

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = np.loadtxt(open(base_path.joinpath("X.txt"), "rb"), delimiter=",")
    y = np.loadtxt(
        open(base_path.joinpath("y.txt"), "rb"), delimiter=","
    ).reshape((-1, 1))
    hyp = np.loadtxt(open(base_path.joinpath("hyp.txt"), "rb"), delimiter=",")
    gp.update(X_new=X, y_new=y, hyp=hyp)

    options = {
        "tol_con_loss": 0.01,
        "tol_weight": 1e-2,
        "weight_penalty": 0.1,
        "tol_length": 1e-6,
    }
    theta_bnd = None  # vp.get_bounds(gp.X, options, K)
    theta = vp.get_parameters()

    F, dF, G, H, varF, dH, varG_ss, varG, varH, I_sk, J_sjk = _neg_elcbo(
        theta, gp, vp, 0.0, 0, False, True, theta_bnd, 0.0, True
    )

    assert np.isclose(F, 11.746298071422430)
    assert dF is None
    assert np.isclose(G, -0.461812484952867)
    assert np.isclose(H, -11.284485586469563)
    assert np.isclose(varF, 6.598768992700180e-05)
    assert dH is None
    assert np.isclose(varG, 6.598768992700180e-05)
    assert varH == 0.0

    F, dF, G, H, varF = _neg_elcbo(
        theta, gp, vp, 0.0, 0, True, False, theta_bnd, 0.0, False
    )
    matlab_dF = np.loadtxt(
        open(base_path.joinpath("dF.txt"), "rb"), delimiter=","
    )

    assert np.allclose(dF, matlab_dF)


def test_vp_bound_loss():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    base_path = Path(__file__).parent
    vp.mu = np.loadtxt(open(base_path.joinpath("mu.txt"), "rb"), delimiter=",")

    options = {
        "tol_con_loss": 0.01,
        "tol_weight": 1e-2,
        "weight_penalty": 0.1,
        "tol_length": 1e-6,
    }
    X = np.loadtxt(open(base_path.joinpath("X.txt"), "rb"), delimiter=",")
    theta = vp.get_parameters()
    theta_bnd = vp.get_bounds(X, options, K)

    L, dL = _vp_bound_loss(vp, theta, theta_bnd)

    assert L == 0.0
    assert np.all(dL == 0.0)

    theta[-1] = 1.0
    L, dL = _vp_bound_loss(vp, theta, theta_bnd, tol_con=0.01)

    assert np.isclose(L, 178.1123635679098)
    assert np.isclose(dL[-1], 356.2247271358195)
    assert np.all(dL[:-1] == 0.0)


def test_vp_optimize_1D_g_mixture():
    """
    Test that the VP is being optimized to the 1D Gaussian Mixture ground truth.
    """

    D = 1

    # fit GP to mixture logpdf
    X = np.linspace(-5, 5, 200)
    mixture_logpdf = lambda x: np.log(
        0.5 * norm.pdf(x, loc=-2, scale=1) + 0.5 * norm.pdf(x, loc=2, scale=1)
    )
    y = mixture_logpdf(X)
    X = np.reshape(X, (-1, 1))
    y = np.reshape(y, (-1, 1))
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(),
    )
    gp.fit(X, y)

    # optimize new VP
    vp = VariationalPosterior(D=D, K=2)
    optim_state = dict()
    optim_state["warmup"] = True
    optim_state["entropy_switch"] = False

    options = setup_options(D, {})
    vp, _, _ = optimize_vp(options, optim_state, vp, gp, 100, 2)

    # ELBO should be equal to the log normalization constant of the distribution
    # that is 0 for a normalized density
    assert np.abs(vp.stats["elbo"]) < 1e-2 * 5

    # compute kl_div between gaussian mixture and vp
    vp_samples, _ = vp.sample(int(10e6))
    vp_mu = np.mean(vp_samples)
    vp_sigma = np.std(vp_samples)

    mixture_samples = np.concatenate(
        (
            norm.rvs(loc=-2, scale=1, size=int(10e6 // 2)),
            norm.rvs(loc=2, scale=1, size=int(10e6 // 2)),
        )
    )
    mixture_mu = np.mean(mixture_samples)
    mixture_sigma = np.std(mixture_samples)
    assert np.all(
        np.abs(kl_div_mvn(mixture_mu, mixture_sigma, vp_mu, vp_sigma))
        < 1e-3 * 1.25
    )


def test_vp_optimize_2D_g_mixture():
    """
    Test that the VP is being optimized to the 2D Gaussian Mixture ground truth.
    """
    D = 2

    # fit GP to mixture logpdf
    X, Y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    Z = np.array(
        [
            [[X[i, j], Y[i, j]] for j in range(X.shape[1])]
            for i in range(X.shape[0])
        ]
    )
    mixture_logpdf = lambda x: np.log(
        0.5 * multivariate_normal.pdf(x, mean=[-2, 0], cov=1)
        + 0.5 * multivariate_normal.pdf(x, mean=[2, 0], cov=1)
    )
    y = mixture_logpdf(Z)

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(),
    )
    Z = Z.reshape(-1, 2)
    gp.fit(Z, y.reshape(-1, 1))

    # optimize new VP
    vp = VariationalPosterior(D=D, K=2)
    optim_state = dict()
    optim_state["warmup"] = True
    optim_state["entropy_switch"] = False

    options = setup_options(D, {})
    vp, _, _ = optimize_vp(options, optim_state, vp, gp, 100, 2)

    # ELBO should be equal to the log normalization constant of the distribution
    # that is 0 for a normalized density
    assert np.abs(vp.stats["elbo"]) < 1e-1

    # compute kl_div between gaussian mixture and vp
    vp_samples, _ = vp.sample(int(10e6))
    vp_mu = np.mean(vp_samples)
    vp_sigma = np.std(vp_samples)

    mixture_samples = np.concatenate(
        (
            multivariate_normal.rvs(mean=[-2, 0], cov=1, size=int(10e6 // 2)),
            multivariate_normal.rvs(mean=[2, 0], cov=1, size=int(10e6 // 2)),
        )
    )
    mixture_mu = np.mean(mixture_samples)
    mixture_sigma = np.std(mixture_samples)
    assert np.all(
        np.abs(kl_div_mvn(mixture_mu, mixture_sigma, vp_mu, vp_sigma)) < 1e-2
    )


def test_vp_optimize_deterministic_entropy_approximation():
    """
    Test that the VP is being optimized to the 1D Gaussian Mixture ground truth.
    Do this with the deterministic entropy approximation.
    """
    D = 1

    # fit GP to mixture logpdf
    X = np.linspace(-5, 5, 200)
    mixture_logpdf = lambda x: np.log(
        0.5 * norm.pdf(x, loc=-2, scale=1) + 0.5 * norm.pdf(x, loc=2, scale=1)
    )
    y = mixture_logpdf(X)
    X = np.reshape(X, (-1, 1))
    y = np.reshape(y, (-1, 1))
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(),
    )
    gp.fit(X, y)

    # optimize new VP
    vp = VariationalPosterior(D=D, K=2)
    optim_state = dict()
    optim_state["warmup"] = True
    optim_state["entropy_switch"] = True

    options = setup_options(D, {})
    vp, _, _ = optimize_vp(options, optim_state, vp, gp, 10, 1)

    # ELBO should be equal to the log normalization constant of the distribution
    # that is 0 for a normalized density
    assert np.abs(vp.stats["elbo"]) < 0.25

    # compute kl_div between gaussian mixture and vp
    vp_samples, _ = vp.sample(int(10e6))
    vp_mu = np.mean(vp_samples)
    vp_sigma = np.std(vp_samples)

    mixture_samples = np.concatenate(
        (
            norm.rvs(loc=-2, scale=1, size=int(10e6 // 2)),
            norm.rvs(loc=2, scale=1, size=int(10e6 // 2)),
        )
    )
    mixture_mu = np.mean(mixture_samples)
    mixture_sigma = np.std(mixture_samples)
    assert np.all(
        np.abs(kl_div_mvn(mixture_mu, mixture_sigma, vp_mu, vp_sigma))
        < 1e-4 * 1.25
    )
