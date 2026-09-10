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
        "active_importance_sampling_fess_thresh": 0,
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
