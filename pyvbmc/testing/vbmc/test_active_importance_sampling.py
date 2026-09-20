import copy
import os.path
from sys import float_info

import gpyreg as gpr
import numpy as np
import pytest
import scipy.stats as sps

from pyvbmc.acquisition_functions import AcqFcnIMIQR, AcqFcnVIQR
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.active_importance_sampling import (
    active_importance_sampling,
    active_sample_proposal_pdf,
    fess,
)
from pyvbmc.vbmc.options import Options


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """``VariationalPosterior.__init__`` draws its seed from the global
    ``np.random`` state and ``test_draws_come_from_vp_rng_only`` seeds that
    state; leave it as each test found it, so later modules see the
    stream they would have seen."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _scenario():
    """A two-component VP, a two-sample GP on 15 points and options with
    small sample counts (thinned MCMC), as ``test_active_importance_sampling``
    has always used."""
    D = 3
    K = 2
    vp = VariationalPosterior(D=D, K=K)
    vp.mu = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).T
    vp.w = np.array([[0.7, 0.3]])
    vp.lambd = np.ones(vp.lambd.shape)
    gp_means = np.arange(-5, 5).reshape((5, 2), order="F") * np.pi

    X = np.arange(-7, 8).reshape((5, 3), order="F")
    y = np.array(
        [
            sps.multivariate_normal.logpdf(
                x,
                mean=np.zeros(
                    D,
                ),
            )
            for x in X
        ]
    ).reshape((-1, 1))
    hyp = np.array(
        [
            # Covariance
            -2.0,
            -3.0,
            -4.0,  # log ell
            1.0,  # log sf2
            # Noise
            0.0,  # log std. dev. of noise
            # Mean
            -(D / 2) * np.log(2 * np.pi),  # MVN mode
            0.0,
            0.25,
            0.5,  # Mode location
            -0.5,
            0.0,
            0.5,  # log scale
        ]
    )
    hyp = np.vstack([hyp, 2 * hyp])
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
        ),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)

    # Initialize options:
    user_options = {
        "active_importance_sampling_mcmc_samples": 10,
        "active_importance_sampling_mcmc_thin": 2,
        "active_importance_sampling_vp_samples": 11,
        "active_importance_sampling_box_samples": 12,
    }
    pyvbmc_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "vbmc",
        )
    )
    basic_path = pyvbmc_path + "/option_configs/basic_vbmc_options.ini"
    vbmc_options = Options(
        basic_path,
        evaluation_parameters={"D": D},
        user_options=user_options,
    )
    return vp, gp, vbmc_options


def test_active_importance_sampling():
    vp, gp, vbmc_options = _scenario()
    D = vp.D
    active_is_viqr = active_importance_sampling(
        vp, gp, AcqFcnVIQR(), vbmc_options
    )
    active_is_imiqr = active_importance_sampling(
        vp, gp, AcqFcnIMIQR(), vbmc_options
    )

    assert (
        active_is_viqr["ln_weights"].shape
        == active_is_viqr["f_s2"].T.shape
        == (2, 10)
    )
    assert active_is_viqr["X"].shape == (10, D)
    assert active_is_imiqr["X"].shape == (2, 10, D)
    assert (
        active_is_imiqr["ln_weights"].shape
        == active_is_imiqr["f_s2"].T.shape
        == (2, 10)
    )


@pytest.mark.parametrize("acq_fcn", [AcqFcnVIQR(), AcqFcnIMIQR()])
def test_an_acquisition_asking_for_mcmc_importance_sampling_is_refused(
    acq_fcn,
):
    """The refinement of the samples of step 1 by an ensemble sampler,
    which ``acq_info["mcmc_importance_sampling"]`` asks for
    (``private/activeimportancesampling_vbmc.m``, lines 57 to 92), is not
    ported."""
    vp, gp, vbmc_options = _scenario()
    acq_fcn = copy.deepcopy(acq_fcn)
    acq_fcn.acq_info["mcmc_importance_sampling"] = True
    with pytest.raises(NotImplementedError, match="mcmc_importance_sampling"):
        active_importance_sampling(vp, gp, acq_fcn, vbmc_options)


@pytest.mark.parametrize("acq_fcn", [AcqFcnVIQR(), AcqFcnIMIQR()])
def test_draws_come_from_vp_rng_only(acq_fcn):
    """Every draw, the MCMC step's included, comes from ``vp.rng``: the
    same generator seed gives the same samples whatever the global state
    holds, and the global state is left as it was (until 2026-09-10 the
    slice sampler of the IMIQR branch drew from it)."""
    vp, gp, vbmc_options = _scenario()
    keys = ("X", "ln_weights", "f_s2", "K_Xa_X", "C_tmp")
    outs = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        before = np.random.get_state()
        vp.rng = np.random.default_rng(3)
        outs.append(active_importance_sampling(vp, gp, acq_fcn, vbmc_options))
        after = np.random.get_state()
        assert after[2] == before[2] and np.array_equal(after[1], before[1])
    for key in keys:
        assert np.array_equal(outs[0][key], outs[1][key])
    vp.rng = np.random.default_rng(4)
    other = active_importance_sampling(vp, gp, acq_fcn, vbmc_options)
    assert not np.array_equal(other["X"], outs[0]["X"])


def test_fess():
    D = 3
    K = 2
    vp = VariationalPosterior(D=D, K=K)
    vp.mu = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).T
    vp.w = np.array([[0.7, 0.3]])
    vp.lambd = np.ones(vp.lambd.shape)
    gp_means = np.arange(-5, 5).reshape((5, 2), order="F") * np.pi

    X = np.arange(-7, 8).reshape((5, 3), order="F")
    y = np.array(
        [
            sps.multivariate_normal.logpdf(
                x,
                mean=np.zeros(
                    D,
                ),
            )
            for x in X
        ]
    ).reshape((-1, 1))
    hyp = np.array(
        [
            # Covariance
            -2.0,
            -3.0,
            -4.0,  # log ell
            1.0,  # log sf2
            # Noise
            0.0,  # log std. dev. of noise
            # Mean
            -(D / 2) * np.log(2 * np.pi),  # MVN mode
            0.0,
            0.25,
            0.5,  # Mode location
            -0.5,
            0.0,
            0.5,  # log scale
        ]
    )
    hyp = np.vstack([hyp, 2 * hyp])
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
        ),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)

    Xa = 2 * np.arange(-4, 5).reshape((3, 3), order="F") / np.pi

    dirpath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dirpath, "compare_MATLAB", "fess.npz")

    fess_means = fess(vp, gp_means, X)
    fess_gp = fess(vp, gp, Xa)
    assert np.isscalar(fess_means)
    assert np.isscalar(fess_gp)
    with np.load(filepath, allow_pickle=False) as MATLAB:
        assert np.isclose(fess_means, MATLAB["fess_means"])
        assert np.isclose(fess_gp, MATLAB["fess_gp"])


def test_fess_draws_the_points_it_is_asked_for():
    """``X`` may be a number of samples to draw from the variational
    posterior, the documented default of 100 among them
    (``misc/fess_vbmc.m:4-12``). The value is the fractional effective
    sample size of the importance weights at the points drawn."""
    vp, gp, __ = _scenario()
    for N, call in (
        (100, lambda: fess(vp, gp)),
        (25, lambda: fess(vp, gp, 25)),
    ):
        vp.rng = np.random.default_rng(7)
        value = call()
        assert np.isscalar(value)

        # The same draws, and the definition of the quantity.
        vp.rng = np.random.default_rng(7)
        X, __ = vp.sample(N, orig_flag=False)
        f_bar, __ = gp.predict(X)
        ln_weights = (
            f_bar.ravel() - vp.pdf(X, orig_flag=False, log_flag=True).ravel()
        )
        weights = np.exp(ln_weights - np.amax(ln_weights))
        weights = weights / np.sum(weights)
        assert value == pytest.approx((1 / np.sum(weights**2)) / N)


def test_fess_checks_the_number_of_given_gp_means():
    """A matrix of GP means carries one row per sample point, whether the
    points were given or drawn."""
    vp, gp, __ = _scenario()
    vp.rng = np.random.default_rng(7)
    means = np.zeros((7, 2))
    with pytest.raises(ValueError, match="Mismatch"):
        fess(vp, means, 25)


def test_active_sample_proposal_pdf():
    D = 3
    K = 2
    vp = VariationalPosterior(D=D, K=K)
    vp.mu = np.array([[-1.0, -2.0, -3.0], [3.0, 2.0, 1.0]]).T
    vp.w = np.array([[0.7, 0.3]])
    vp.sigma = np.ones(vp.sigma.shape)
    vp.lambd = np.ones(vp.lambd.shape)
    X = np.arange(-7, 8).reshape((5, 3), order="F")
    y = np.array(
        [
            sps.multivariate_normal.logpdf(
                x,
                mean=np.zeros(
                    D,
                ),
            )
            for x in X
        ]
    ).reshape((-1, 1))
    hyp = np.array(
        [
            # Covariance
            -2.0,
            -3.0,
            -4.0,  # log ell
            1.0,  # log sf2
            # Noise
            0.0,  # log std. dev. of noise
            # Mean
            -(D / 2) * np.log(2 * np.pi),  # MVN mode
            0.0,
            0.25,
            0.5,  # Mode location
            -0.5,
            0.0,
            0.5,  # log scale
        ]
    )
    hyp = np.vstack([hyp, 2 * hyp])
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
        ),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)

    Xa = 2 * np.arange(-4, 5).reshape((3, 3), order="F") / np.pi
    w_vp = 0.5
    rect_delta = 2 * np.std(gp.X, ddof=1, axis=0)

    dirpath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(
        dirpath, "compare_MATLAB", "activesample_proposalpdf.npz"
    )

    ln_weights_viqr, f_s2_viqr = active_sample_proposal_pdf(
        Xa, gp, vp, w_vp, rect_delta, AcqFcnVIQR()
    )
    ln_weights_imiqr, f_s2_imiqr = active_sample_proposal_pdf(
        Xa, gp, vp, w_vp, rect_delta, AcqFcnIMIQR()
    )
    Ns_gp = hyp.shape[0]
    assert ln_weights_viqr.shape == ln_weights_imiqr.shape == (D, Ns_gp)
    assert f_s2_viqr.shape == f_s2_imiqr.shape == (D, Ns_gp)
    with np.load(filepath, allow_pickle=False) as MATLAB:
        assert np.allclose(ln_weights_viqr, MATLAB["ln_weights_viqr"])
        assert np.allclose(f_s2_viqr, MATLAB["f_s2_viqr"])
        assert np.allclose(ln_weights_imiqr, MATLAB["ln_weights_imiqr"])
        assert np.allclose(f_s2_imiqr, MATLAB["f_s2_imiqr"])


def test_proposal_pdf_gives_no_weight_where_the_proposal_is_zero():
    """A point that lies outside every component of the proposal has zero
    proposal density and so no importance weight. MATLAB's arithmetic
    gives NaN there and ``activeimportancesampling_vbmc.m:148`` turns it
    into ``-Inf``, which is what the caller does with every non-finite
    weight."""
    vp, gp, __ = _scenario()
    acq_fcn = AcqFcnIMIQR()
    rect_delta = 2 * np.std(gp.X, ddof=1, axis=0)
    w_vp = 0.5

    Xa = 2 * np.arange(-4, 5).reshape((3, 3), order="F") / np.pi
    far = np.full((1, vp.D), 1e5)
    assert np.all(vp.pdf(far, orig_flag=False, log_flag=True) == -np.inf)
    assert not np.any(np.all(np.abs(far - gp.X) < rect_delta, axis=1))

    ln_weights, f_s2 = active_sample_proposal_pdf(
        np.vstack([Xa, far]), gp, vp, w_vp, rect_delta, acq_fcn
    )
    assert np.all(ln_weights[-1] == -np.inf)

    # The other points are untouched.
    ln_weights_alone, f_s2_alone = active_sample_proposal_pdf(
        Xa, gp, vp, w_vp, rect_delta, acq_fcn
    )
    assert np.array_equal(ln_weights[:-1], ln_weights_alone)
    assert np.array_equal(f_s2[:-1], f_s2_alone)


def test_acq_log_f():
    D = 3
    K = 2
    vp = VariationalPosterior(D=D, K=K)
    vp.mu = np.array([[-1.5, -1.0, -0.5], [0.0, 1.0, 2.0]]).T
    vp.w = np.array([[0.7, 0.3]])
    vp.sigma = np.ones(vp.sigma.shape)
    vp.lambd = np.ones(vp.lambd.shape)
    X = np.arange(-7, 8).reshape((5, 3), order="F")
    y = np.array(
        [
            sps.multivariate_normal.logpdf(
                x,
                mean=np.zeros(
                    D,
                ),
            )
            for x in X
        ]
    ).reshape((-1, 1))
    hyp = np.array(
        [
            # Covariance
            -2.0,
            -3.0,
            -4.0,  # log ell
            1.0,  # log sf2
            # Noise
            0.0,  # log std. dev. of noise
            # Mean
            -(D / 2) * np.log(2 * np.pi),  # MVN mode
            0.0,
            0.25,
            0.5,  # Mode location
            -0.5,
            0.0,
            0.5,  # log scale
        ]
    )
    hyp = np.vstack([hyp, 2 * hyp])
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True,
        ),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)

    Xa = 2 * np.arange(-4, 5).reshape((3, 3), order="F") / np.pi

    dirpath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dirpath, "compare_MATLAB", "log_isbasefun.npz")
    print(filepath)

    viqr = AcqFcnVIQR()
    # Use vp weights for this test, since IMIQR uses them.
    viqr.acq_info["importance_sampling_vp"] = True
    y_viqr = viqr.is_log_full(Xa, gp=gp, vp=vp)
    y_imiqr = AcqFcnIMIQR().is_log_full(Xa, gp=gp, vp=vp)
    viqr = AcqFcnVIQR()
    y_viqr = viqr.is_log_full(Xa, gp=gp, vp=vp)
    # Add VP density to i.s. weights:
    y_viqr += np.maximum(
        vp.pdf(Xa, orig_flag=False, log_flag=True), np.log(float_info.min)
    )
    y_imiqr = AcqFcnIMIQR().is_log_full(Xa, gp=gp, vp=vp)

    assert y_viqr.shape == y_imiqr.shape == (D, 1)
    with np.load(filepath, allow_pickle=False) as MATLAB:
        assert np.allclose(y_viqr, MATLAB["y_viqr"], atol=1e-3)
        assert np.allclose(y_imiqr, MATLAB["y_imiqr"], atol=1e-3)
