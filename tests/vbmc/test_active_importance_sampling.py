import numpy as np
import os.path
import scipy.io
import scipy.stats as sps

import gpyreg as gpr
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.active_importance_sampling import fess, activesample_proposalpdf
from pyvbmc.acquisition_functions import AcqFcnVIQR


def test_fess():
    D = 3
    K = 2
    vp = VariationalPosterior(
        D=D, K=K
    )
    vp.mu = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).T
    vp.w = np.array([[0.7, 0.3]])
    gp_means = np.arange(-5, 5).reshape((5, 2), order="F") * np.pi
    X = np.arange(-7, 8).reshape((5, 3), order="F")

    dirpath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dirpath, "compare_MATLAB", "fess.mat")
    fess_MATLAB = scipy.io.loadmat(filepath)["fess_MATLAB"]
    assert np.isclose(fess(vp, gp_means, X), fess_MATLAB)

def test_activesample_proposalpdf():
    D = 3
    K = 2
    vp = VariationalPosterior(
        D=D, K=K
    )
    vp.mu = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).T
    vp.w = np.array([[0.7, 0.3]])
    vp.sigma = np.ones(vp.sigma.shape)
    X = np.arange(-7, 8).reshape((5, 3), order="F")
    y = np.array([sps.multivariate_normal.logpdf(x, mean=np.zeros(D,)) for x in X])
    hyp = np.array(
        [
            [
                # Covariance
                -2.0, -3.0, -4.0, # log ell
                1.0,  # log sf2
                # Noise
                0.0,  # log std. dev. of noise
                # Mean
                -(D / 2) * np.log(2 * np.pi),  # MVN mode
                0.0, 0.0, 0.0,  # Mode location
                0.0, 0.0, 0.0  # log scale
            ]
        ]
    )
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)

    Xa = np.arange(-4, 5).reshape((3, 3), order="F")
    w_vp = 0.5
    rect_delta = 0.1
    acqviqr = AcqFcnVIQR()

    result = activesample_proposalpdf(Xa, gp, vp, w_vp, rect_delta, acqviqr, vp)
    print(result)
