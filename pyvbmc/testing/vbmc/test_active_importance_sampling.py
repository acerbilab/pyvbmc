import os.path
from sys import float_info

import gpyreg as gpr
import numpy as np
import scipy.io
import scipy.stats as sps

from pyvbmc.acquisition_functions import AcqFcnIMIQR, AcqFcnVIQR
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.active_importance_sampling import (
    active_importance_sampling,
    active_sample_proposal_pdf,
    fess,
)
from pyvbmc.vbmc.options import Options


def test_active_importance_sampling():
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
    filepath = os.path.join(dirpath, "compare_MATLAB", "fess.mat")
    MATLAB = scipy.io.loadmat(filepath)

    fess_means = fess(vp, gp_means, X)
    fess_gp = fess(vp, gp, Xa)
    assert np.isscalar(fess_means)
    assert np.isscalar(fess_gp)
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
        dirpath, "compare_MATLAB", "activesample_proposalpdf.mat"
    )
    MATLAB = scipy.io.loadmat(filepath)

    ln_weights_viqr, f_s2_viqr = active_sample_proposal_pdf(
        Xa, gp, vp, w_vp, rect_delta, AcqFcnVIQR()
    )
    ln_weights_imiqr, f_s2_imiqr = active_sample_proposal_pdf(
        Xa, gp, vp, w_vp, rect_delta, AcqFcnIMIQR()
    )
    Ns_gp = hyp.shape[0]
    assert ln_weights_viqr.shape == ln_weights_imiqr.shape == (D, Ns_gp)
    assert f_s2_viqr.shape == f_s2_imiqr.shape == (D, Ns_gp)
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
    filepath = os.path.join(dirpath, "compare_MATLAB", "log_isbasefun.mat")
    print(filepath)
    MATLAB = scipy.io.loadmat(filepath)

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
    assert np.allclose(y_viqr, MATLAB["y_viqr"], atol=1e-3)
    assert np.allclose(y_imiqr, MATLAB["y_imiqr"], atol=1e-3)
