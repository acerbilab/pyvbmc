import numpy as np

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.active_importance_sampling import fess


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

    fess_MATLAB = 0.218066216419553
    assert np.isclose(fess(vp, gp_means, X), fess_MATLAB)
