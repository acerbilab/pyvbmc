import pytest
import numpy as np
from scipy.stats import norm

import gpyreg as gpr

from pyvbmc.vbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior

def test_estimate_noise():
    # Back-up random number generator so as not to affect other tests
    # that might want to use different random numbers each run.
    state = np.random.get_state()
    np.random.seed(1234)
    
    N = 31
    D = 1
    X = -5 + np.random.rand(N, 1) * 10
    s2 = 0.05 * np.exp(0.5 * X)
    y = np.sin(X) + np.sqrt(s2) * norm.ppf(np.random.random_sample(X.shape))
    y[y < 0] = -np.abs(3 * y[y < 0]) ** 2

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(degree=3),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )
    
    hyp = np.array([[-2.5, 1.7, -7.5, 0.3, 2.6, 1.2]])
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)

    noise_estimate = VBMC._estimate_noise(gp)
    
    np.random.set_state(state)

    # Value taken from MATLAB which only applies for this exact setup.
    # Change any part of the test and it will not apply.
    assert np.isclose(noise_estimate, 0.106582207806606)
    
def test_get_hpd():
    order = np.random.permutation(range(0, 100))
    X = np.reshape(order.copy(), (-1, 1))
    y = X.copy()
    
    hpd_X, hpd_y, hpd_range = VBMC._get_hpd(X, y)
    
    assert np.all(hpd_X == hpd_y)
    assert np.all(hpd_X.flatten() == np.array(list(reversed(range(20, 100)))))
    assert hpd_range == np.array([79])
    
    hpd_X, hpd_y, hpd_range = VBMC._get_hpd(X, y, hpd_frac=0.5)
    
    assert np.all(hpd_X == hpd_y)
    assert np.all(hpd_X.flatten() == np.array(list(reversed(range(50, 100)))))
    assert hpd_range == np.array([49])
