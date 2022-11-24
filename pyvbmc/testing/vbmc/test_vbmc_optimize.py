from pathlib import Path

import dill
import numpy as np
import scipy as sp

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import *


def wrap_with_test(method, vbmc):
    def wrapper(*args, **kwargs):
        """Wraps vbmc._check_termination_conditions.

        To be run at the end of every iteration:
        """
        assert vbmc.vp.K == vbmc.optim_state["vp_K"]
        assert vbmc.parameter_transformer == vbmc.vp.parameter_transformer
        assert (
            vbmc.parameter_transformer
            == vbmc.function_logger.parameter_transformer
        )
        return method(*args, **kwargs)

    return wrapper


def test_vbmc_optimize_rosenbrock():
    D = 2

    def llfun(x):
        if x.ndim == 2:
            return -np.sum(
                (x[0, :-1] ** 2.0 - x[0, 1:]) ** 2.0
                + (x[0, :-1] - 1) ** 2.0 / 100
            )
        else:
            return -np.sum(
                (x[:-1] ** 2.0 - x[1:]) ** 2.0 + (x[:-1] - 1) ** 2.0 / 100
            )

    prior_mu = np.zeros((1, D))
    prior_var = 3**2 * np.ones((1, D))
    lpriorfun = lambda x: -0.5 * (
        np.sum((x - prior_mu) ** 2 / prior_var)
        + np.log(np.prod(2 * np.pi * prior_var))
    )

    plb = prior_mu - 3 * np.sqrt(prior_var)
    pub = prior_mu + 3 * np.sqrt(prior_var)
    x0 = prior_mu.copy()

    vbmc = VBMC(llfun, x0, None, None, plb, pub, log_prior=lpriorfun)
    # Patch in the modified method:
    vbmc._check_termination_conditions = wrap_with_test(
        vbmc._check_termination_conditions, vbmc
    )
    vbmc.optimize()


def run_optim_block(
    f,
    x0,
    lb,
    ub,
    plb,
    pub,
    ln_Z,
    mu_bar,
    options=None,
    noise_flag=False,
    log_prior=None,
):
    if options is None:
        options = {}

    # options["max_fun_evals"] = 100
    options["plot"] = False
    if noise_flag:
        options["specify_target_noise"] = True

    vbmc = VBMC(f, x0, lb, ub, plb, pub, options=options, log_prior=log_prior)
    # Patch in the modified method:
    vbmc._check_termination_conditions = wrap_with_test(
        vbmc._check_termination_conditions, vbmc
    )
    vp, results = vbmc.optimize()
    elbo = results["elbo"]

    vmu = vp.moments()
    err_1 = np.sqrt(np.mean((vmu - mu_bar) ** 2))
    err_2 = np.abs(elbo - ln_Z)

    return err_1, err_2


def test_vbmc_multivariate_normal(return_results=False):
    D = 6
    x0 = -np.ones((1, D))
    # Be careful about -2 and -2.0!
    plb = np.full((1, D), -2.0 * D)
    pub = np.full((1, D), 2.0 * D)
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    lnZ = 0
    mu_bar = np.zeros((1, D))
    f = (
        lambda x: np.sum(-0.5 * (x / np.array(range(1, np.size(x) + 1))) ** 2)
        - np.sum(np.log(np.array(range(1, np.size(x) + 1))))
        - 0.5 * np.size(x) * np.log(2 * np.pi)
    )

    err_1, err_2 = run_optim_block(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        lnZ,
        mu_bar,
    )

    assert err_1 < 0.5
    assert err_2 < 0.5
    if return_results:
        return err_1, err_2


def test_vbmc_multivariate_half_normal(return_results=False):
    D = 2
    x0 = -np.ones((1, D))
    plb = np.full((1, D), -6.0)
    pub = np.full((1, D), -0.05)
    lb = np.full((1, D), -D * 10.0)
    ub = np.full((1, D), 0.0)
    lnZ = -D * np.log(2)
    mu_bar = -2 / np.sqrt(2 * np.pi) * np.array(range(1, D + 1))
    f = (
        lambda x: np.sum(-0.5 * (x / np.array(range(1, np.size(x) + 1))) ** 2)
        - np.sum(np.log(np.array(range(1, np.size(x) + 1))))
        - 0.5 * np.size(x) * np.log(2 * np.pi)
    )
    print(f(np.array([1.5, np.pi])))

    err_1, err_2 = run_optim_block(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        lnZ,
        mu_bar,
    )

    assert err_1 < 0.5
    assert err_2 < 0.5
    if return_results:
        return err_1, err_2


def test_vbmc_correlated_multivariate_normal(return_results=False):
    D = 3
    x0 = 0.5 * np.ones((1, D))
    plb = np.full((1, D), -1.0)
    pub = np.full((1, D), 1.0)
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    lnZ = 0.0
    mu_bar = np.reshape(np.linspace(-0.5, 0.5, D), (1, -1))

    err_1, err_2 = run_optim_block(
        cigar,
        x0,
        lb,
        ub,
        plb,
        pub,
        lnZ,
        mu_bar,
    )

    assert err_1 < 0.5
    assert err_2 < 0.5
    if return_results:
        return err_1, err_2


def test_vbmc_correlated_multivariate_normal_2(return_results=False):
    D = 3
    x0 = 0.5 * np.ones((1, D))
    plb = np.full((1, D), -1.0)
    pub = np.full((1, D), 1.0)
    lb = np.full((1, D), -4.0)
    ub = np.full((1, D), 4.0)
    lnZ = 0.0
    mu_bar = np.reshape(np.linspace(-0.5, 0.5, D), (1, -1))

    err_1, err_2 = run_optim_block(
        cigar,
        x0,
        lb,
        ub,
        plb,
        pub,
        lnZ,
        mu_bar,
    )

    assert err_1 < 0.5
    assert err_2 < 0.5
    if return_results:
        return err_1, err_2


def test_vbmc_uniform(return_results=False):
    D = 1
    x0 = 0.5 * np.ones((1, D))
    plb = np.full((1, D), 0.05)
    pub = np.full((1, D), 0.95)
    lb = np.full((1, D), 0.0)
    ub = np.full((1, D), 1.0)
    lnZ = 0
    mu_bar = 0.5 * np.ones((1, D))
    f = lambda x: 0

    options = {"search_optimizer": "Nelder-Mead"}
    err_1, err_2 = run_optim_block(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        lnZ,
        mu_bar,
        options,
    )

    assert err_1 < 0.5
    assert err_2 < 0.5
    if return_results:
        return err_1, err_2


def test_vbmc_multivariate_half_normal_noisy(return_results=False):
    D = 2
    noise_scale = 0.5
    x0 = -np.ones((1, D))
    plb = np.full((1, D), -6.0)
    pub = np.full((1, D), -0.05)
    lb = np.full((1, D), -D * 10.0)
    ub = np.full((1, D), 0.0)
    lnZ = -D * np.log(2)
    mu_bar = -2 / np.sqrt(2 * np.pi) * np.array(range(1, D + 1))
    f = lambda x: (
        np.sum(-0.5 * (x / np.array(range(1, np.size(x) + 1))) ** 2)
        - np.sum(np.log(np.array(range(1, np.size(x) + 1))))
        - 0.5 * np.size(x) * np.log(2 * np.pi)
        + noise_scale * np.random.normal(),
        noise_scale,
    )
    options = {
        "specify_target_noise": True,
        "search_acq_fcn": [AcqFcnVIQR(), AcqFcnIMIQR(), AcqFcnNoisy()],
    }

    err_1, err_2 = run_optim_block(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        lnZ,
        mu_bar,
        options=options,
    )

    assert err_1 < 0.5
    assert err_2 < 0.5
    if return_results:
        return err_1, err_2


def noisy_cigar(x, noise_scale=0.4):
    return cigar(x) + noise_scale * np.random.normal(), noise_scale


def cigar(x):
    """
    Benchmark log pdf -- cigar density.
    """

    if x.ndim == 1:
        x = np.reshape(x, (1, -1))

    D = np.size(x)
    mean = np.reshape(np.linspace(-0.5, 0.5, D), (1, -1))
    if D == 1:
        R = 10.0
    elif D == 2:
        R = np.array(
            [
                [0.438952107785021, -0.898510460190134],
                [0.898510460190134, 0.438952107785021],
            ]
        )
    elif D == 3:
        R = np.array(
            [
                [-0.624318398571926, -0.0583529832968072, -0.778987462379818],
                [0.779779849779334, 0.0129117551612018, -0.625920659873738],
                [0.0465824331986329, -0.998212510399975, 0.0374414342443664],
            ]
        )
    elif D == 4:
        R = np.array(
            [
                [
                    0.530738877213611,
                    -0.332421458771,
                    -0.617324087669642,
                    0.476154584925358,
                ],
                [
                    -0.455283846255008,
                    -0.578972039590549,
                    0.36136334497906,
                    0.57177314523957,
                ],
                [
                    -0.340852417262338,
                    -0.587449365484418,
                    -0.433532840927373,
                    -0.592260203353068,
                ],
                [
                    -0.628372893431681,
                    0.457373582657411,
                    -0.548066400653999,
                    0.309160368031855,
                ],
            ]
        )
    elif D == 5:
        R = np.array(
            [
                [
                    -0.435764067736038,
                    0.484423029373161,
                    0.0201157836447536,
                    -0.195133090987468,
                    0.732777208934001,
                ],
                [
                    -0.611399063990897,
                    -0.741129629736756,
                    -0.0989871013229956,
                    0.187328571370887,
                    0.178962612288509,
                ],
                [
                    -0.0340226717227732,
                    0.234931965418636,
                    -0.886686220684869,
                    0.394077369749064,
                    -0.0462601570752127,
                ],
                [
                    0.590564625513463,
                    -0.400973629067102,
                    -0.304445169104938,
                    -0.33203681171552,
                    0.536207298122059,
                ],
                [
                    0.293899205038366,
                    0.00939792298771627,
                    0.333012903768423,
                    0.813194727889496,
                    0.375967653898291,
                ],
            ]
        )
    elif D == 6:
        R = np.array(
            [
                [
                    -0.254015072891056,
                    -0.0684032463717124,
                    -0.693077090686521,
                    0.249685438636409,
                    0.362364372413356,
                    -0.506745230203393,
                ],
                [
                    -0.207777316284753,
                    0.369766206365964,
                    0.57903494069884,
                    -0.0653147667578752,
                    0.122468089523108,
                    -0.682316367390556,
                ],
                [
                    -0.328071435400004,
                    0.364091738763865,
                    -0.166363836395589,
                    0.380087224558118,
                    -0.766382695507289,
                    -0.0179075049327736,
                ],
                [
                    -0.867667731196277,
                    -0.0332627076729128,
                    0.069771482765022,
                    -0.25333031036676,
                    0.206274179664928,
                    0.366678275010647,
                ],
                [
                    -0.0206482741639122,
                    -0.229074515307431,
                    -0.237811602709101,
                    -0.777821502080957,
                    -0.426607324185607,
                    -0.321782626441153,
                ],
                [
                    -0.177201636197285,
                    -0.820030251267824,
                    0.308647597698151,
                    0.346038046252171,
                    -0.204470893859678,
                    -0.198332931405751,
                ],
            ]
        )
    else:
        raise Exception("Unsupported dimension!")

    ell = np.ones((D,)) / 100
    ell[-1] = 1
    cov = np.dot(R.T, np.dot(np.diag(ell**2), R))

    y = mvnlogpdf(x, mean, cov)  # + mvnlogpdf(x, prior_mean, prior_cov)
    s = 0
    return y


def mvnlogpdf(x, mu, sigma):
    d = np.size(x)
    X0 = x - mu

    # Make sure sigma is a valid covariance matrix.
    R = sp.linalg.cholesky(sigma)

    # Create array of standardized data, and compute log(sqrt(det(Sigma)))
    xRinv = np.linalg.solve(R.T, X0.T).T
    log_sqrt_det_sigma = np.sum(np.log(np.diag(R)))

    # The quadratic form is the inner product of the standardized data.
    quad_form = np.sum(xRinv**2)

    y = -0.5 * quad_form - log_sqrt_det_sigma - d * np.log(2 * np.pi) / 2
    return y


def test_optimize_results(mocker):
    """
    Test that result dict is being recorded correctly.
    """
    D = 3
    vbmc = VBMC(
        lambda x: np.sum(x),
        np.ones((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
        options={"max_iter": 1},
    )
    mocker.patch(
        "pyvbmc.vbmc.VBMC._check_termination_conditions",
        return_value=(True, "test message", True),
    )
    mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp", return_value=(vbmc.vp, None, None)
    )
    mocker.patch(
        "pyvbmc.vbmc.VBMC.final_boost", return_value=(vbmc.vp, -2, 1, False)
    )

    vbmc.iteration_history["stable"] = list()
    vbmc.iteration_history["stable"].append(False)
    vbmc.iteration_history["r_index"] = list()
    vbmc.iteration_history["r_index"].append(2)
    vbmc.vp.stats = dict()
    vbmc.vp.stats["entropy"] = 1
    vbmc.vp.stats["elbo"] = -2
    vbmc.vp.stats["elbo_sd"] = 0
    __, results = vbmc.optimize()
    assert "function" in results
    assert results["problem_type"] == "unconstrained"
    assert "iterations" in results
    assert "func_count" in results
    assert "best_iter" in results
    assert "train_set_size" in results
    assert "components" in results
    assert results["r_index"] == 2
    assert results["convergence_status"] == "no"
    assert np.isnan(results["overhead"])
    assert "rng_state" in results
    assert results["algorithm"] == "Variational Bayesian Monte Carlo"
    assert "version" in results
    assert results["message"] == "test message"
    assert "elbo" in results
    assert "elbo_sd" in results


def _test_optimize_reproducibility():
    # 1D case with Nelder-Mead optimizer
    result = {"err_1": [], "err_2": []}
    for i in range(2):
        np.random.seed(42)
        err_1, err_2 = test_vbmc_uniform(return_results=True)
        result["err_1"].append(err_1)
        result["err_2"].append(err_2)

    for k, v in result.items():
        assert v[0] == v[1]

    # Multi-dimensional case with cmaes optimizer
    result = {"err_1": [], "err_2": []}
    for i in range(2):
        np.random.seed(42)
        err_1, err_2 = test_vbmc_multivariate_half_normal(return_results=True)
        result["err_1"].append(err_1)
        result["err_2"].append(err_2)

    for k, v in result.items():
        assert v[0] == v[1]


def test_vbmc_resume_optimization():
    D = 2
    seed = np.random.randint(0, 100)

    def llfun(x):  # Rosenbrock, as before
        if x.ndim == 2:
            return -np.sum(
                (x[0, :-1] ** 2.0 - x[0, 1:]) ** 2.0
                + (x[0, :-1] - 1) ** 2.0 / 100
            )
        else:
            return -np.sum(
                (x[:-1] ** 2.0 - x[1:]) ** 2.0 + (x[:-1] - 1) ** 2.0 / 100
            )

    prior_mu = np.zeros((1, D))
    prior_var = 3**2 * np.ones((1, D))
    lpriorfun = lambda x: -0.5 * (
        np.sum((x - prior_mu) ** 2 / prior_var)
        + np.log(np.prod(2 * np.pi * prior_var))
    )

    plb = prior_mu - 3 * np.sqrt(prior_var)
    pub = prior_mu + 3 * np.sqrt(prior_var)
    x0 = prior_mu.copy()

    # First run for 8 iterations:
    np.random.seed(seed)
    options = {"max_iter": 8}
    vbmc_1 = VBMC(
        llfun,
        x0,
        None,
        None,
        plb,
        pub,
        log_prior=lpriorfun,
        options=options,
    )
    vbmc_1._check_termination_conditions = wrap_with_test(
        vbmc_1._check_termination_conditions, vbmc_1
    )
    np.random.seed(seed + 1)
    vp_1, results_1 = vbmc_1.optimize()
    elbo_1 = results_1["elbo"]
    elbo_sd_1 = results_1["elbo_sd"]
    success_flag_1 = results_1["success_flag"]

    # Then run for 4, save, load, run for 4 more:
    options = {"max_iter": 4, "do_final_boost": False}
    np.random.seed(seed)
    vbmc_2 = VBMC(
        llfun,
        x0,
        None,
        None,
        plb,
        pub,
        log_prior=lpriorfun,
        options=options,
    )
    vbmc_2._check_termination_conditions = wrap_with_test(
        vbmc_2._check_termination_conditions, vbmc_2
    )
    np.random.seed(seed + 1)
    vbmc_2.optimize()

    base_path = Path(__file__).parent
    file_path = base_path.joinpath("vbmc_test.pkl")
    with open(file_path, "wb") as f:
        dill.dump(vbmc_2, f)
    with open(file_path, "rb") as f:
        vbmc_2 = dill.load(f)
    vbmc_2.options.__setitem__("max_iter", 8, force=True)
    vbmc_2.options.__setitem__("do_final_boost", True, force=True)
    vp_2, results_2 = vbmc_2.optimize()
    elbo_2 = results_2["elbo"]
    elbo_sd_2 = results_2["elbo_sd"]
    success_flag_2 = results_2["success_flag"]

    assert success_flag_1 == success_flag_2
    assert elbo_1 == elbo_1
    assert elbo_sd_1 == elbo_sd_2
