import gpyreg as gpr
import numpy as np
import pytest
import scipy.stats as sps

from pyvbmc.acquisition_functions import AcqFcnIMIQR
from pyvbmc.acquisition_functions.utilities import string_to_acq
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import active_importance_sampling

from ._look_ahead import gauss_hermite_reference, weighted_samples_reference
from ._regularization import check_variance_regularization
from ._scenario import look_ahead_scenario, options, prepare_optim_state


@pytest.fixture(autouse=True)
def _seeded_global_rng():
    """``VariationalPosterior.__init__`` derives ``vp.rng`` from the global
    ``np.random`` state and the target of ``test_simple__call__`` draws
    from it, so seed that stream for every test here and restore it
    after, leaving later modules where they would have been. (The MCMC
    step of ``active_importance_sampling`` drew from it too until
    2026-09-10; it now draws from ``vp.rng``.)"""
    state = np.random.get_state()
    np.random.seed(0)
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("ns_gp", [1, 2])
@pytest.mark.parametrize("M", [1, 3])
def test_imiqr_variance_regularization_batch_matches_pointwise(ns_gp, M):
    check_variance_regularization(AcqFcnIMIQR(), ns_gp, M)


def test_acq_info():
    acqf = AcqFcnIMIQR()
    assert acqf.acq_info["importance_sampling"]
    assert not acqf.acq_info["importance_sampling_vp"]
    assert acqf.acq_info["log_flag"]
    assert np.isclose(sps.norm.cdf(acqf.u), 0.75)

    # Test handling of string input for SearchAcqFcn:
    acqf2 = string_to_acq("AcqFcnIMIQR")
    acqf3 = string_to_acq("AcqFcnIMIQR()")
    assert acqf.u == acqf2.u == acqf3.u

    acqf4 = AcqFcnIMIQR(quantile=0.666)
    acqf5 = string_to_acq("AcqFcnIMIQR(quantile=0.666)")
    acqf6 = string_to_acq("AcqFcnIMIQR(0.666)")
    assert acqf4.u == acqf5.u == acqf6.u
    assert np.isclose(sps.norm.cdf(acqf4.u), 0.666)


def test_simple__call__():
    D = 2
    epsilon = 1e-6

    def ltarget(theta):  # Standard MVN with s2 est. propto dist. from origin
        return (
            sps.multivariate_normal(mean=np.zeros((D,)), cov=np.eye(D)).logpdf(
                theta
            )
            + np.sqrt(np.linalg.norm(theta)) * np.random.normal(),
            np.linalg.norm(theta) + epsilon,
        )

    # Train GP on "dummy" points far away from I.S. points.
    # This way the predictive variance at I.S. points will approximate the
    # GP prior variance.
    X = np.array([[50.0, -50.0], [-50.0, 50.0]])
    lls = np.array([ltarget(x) for x in X])
    y = lls[:, 0].reshape(-1, 1)
    s2 = lls[:, 1].reshape(-1, 1)

    # Choose a GP with a short lengthscale,
    hyp = np.array(
        [
            [
                # Covariance
                -4.0,
                -4.0,  # log ell
                1.0,  # log sf2
                # Noise
                0.0,  # log std. dev. of noise
                # Mean
                -(D / 2) * np.log(2 * np.pi),  # MVN mode
                0.0,
                0.0,  # Mode location
                0.0,
                0.0,  # log scale
            ]
        ]
    )
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)
    # gp.plot(lb=np.array([-5.0, -5.0]), ub=np.array([5.0, 5.0]))

    # Acquisition function evaluation point:
    X_eval = np.zeros((1, D))

    vp = VariationalPosterior(D, 1)  # VP with one component
    vp.mu = np.zeros((D, 1))
    vp.sigma = np.ones((1, 1))  # VP is standard normal

    ## Setup acquisition function and necessary preliminaries:

    acqimiqr = AcqFcnIMIQR()
    optim_state = prepare_optim_state(gp, s2)
    vbmc_options = options(D, 100)

    optim_state["active_importance_sampling"] = active_importance_sampling(
        vp, gp, acqimiqr, vbmc_options
    )

    # Test IMIQR Acquisition Function Values:
    # Should be close to log(2 * sinh(0.6745 * e)), because tau^2 approx= 0,
    # so s_pred^2 approx= f_s2.

    result = acqimiqr(
        X_eval[0], gp, vp, function_logger=None, optim_state=optim_state
    )
    u = sps.norm.ppf(0.75)
    # print(result)
    # print(np.log(np.sinh(u * np.exp(1))))
    assert np.isclose(np.log(2 * np.sinh(u * np.exp(1))), result, atol=1e-4)


def test_complex__call__():
    """The IMIQR of five candidates against its definition on the shared
    scenario (``_scenario.look_ahead_scenario``), whose GP mean is the
    standard-normal target exactly, so that the density IMIQR integrates
    against, exp(f_mu), is the VP.

    Two checks. The acquisition's value is recomputed from its own
    importance samples and normalized weights with the GP's full
    covariances, an exact check of the look-ahead formula. Then the
    estimate is compared with the integral under the VP, by Gauss-Hermite
    quadrature, at the precision the importance-sampling scheme delivers
    here: with 8000 MCMC samples the effective sample size is about 3300,
    but the samples that carry the weight (the low-noise half, where the
    sampler's sinh factor is small and the weight 1/sinh large) are
    visited in correlated stretches of the single slice-sampling chain,
    so the self-normalized estimate scatters by 2 % across seeds (twelve
    seeds: mean +0.7 %, largest error 4.3 %, measured 2026-09-10). The
    run is seeded, so the check is deterministic on a platform; the 10 %
    tolerance covers the measured spread with margin so that a platform
    whose rounding sends the chain elsewhere passes too. (Before
    2026-09-10 the reference was a 40x40 grid on [-30, 30],
    3 to 4 % high, and the GP's hyperparameter array was laid out for a
    noise hyperparameter its noise function did not have, so the mean was
    not the target; the two together left a 5 % tolerance to a coin
    toss.)"""
    gp, s2, vp, X_eval, u = look_ahead_scenario()
    D = gp.D
    N_eval = X_eval.shape[0]

    acqimiqr = AcqFcnIMIQR()
    optim_state = prepare_optim_state(gp, s2)
    vbmc_options = options(D, 8000)
    optim_state["active_importance_sampling"] = active_importance_sampling(
        vp, gp, acqimiqr, vbmc_options
    )

    result = np.exp(
        acqimiqr(X_eval, gp, vp, function_logger=None, optim_state=optim_state)
    ).reshape((N_eval,))

    # The noise the acquisition assumes for the hypothetical observation
    # (from the nearest training inputs) enters the reference too; the
    # estimator itself is not what these checks pin.
    sn2_new = acqimiqr._estimate_observation_noise(X_eval, gp, optim_state)
    imiqr = lambda s: 2 * np.sinh(u * s)  # the interquantile range

    # 1. Exact: the value from the stored importance samples and weights.
    active_is = optim_state["active_importance_sampling"]
    weights = np.exp(active_is["ln_weights"][0])
    assert np.isclose(np.sum(weights), 1.0)
    from_samples = weighted_samples_reference(
        gp, active_is["X"][0], weights, X_eval, sn2_new, imiqr
    )
    assert np.allclose(result, from_samples, rtol=1e-10)

    # 2. Statistical: the integral under the VP.
    imiqr_ref = gauss_hermite_reference(gp, vp, X_eval, sn2_new, imiqr)
    assert np.allclose(imiqr_ref, result, rtol=0.1)
