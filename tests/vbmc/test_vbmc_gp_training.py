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
    
def test_get_training_data_no_noise():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1
    
    vbmc = VBMC(f, x0, None, None, plb, pub)
    
    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = vbmc._get_training_data()
    
    assert X_train.shape == (0, 3)
    assert y_train.shape == (0, 1)
    assert s2_train is None
    assert t_train.shape == (0, 1)
    
    # Create dummy data.
    sample_count = 10
    window = vbmc.optim_state["pub"] - vbmc.optim_state["plb"]
    rnd_tmp = np.random.rand(sample_count, window.shape[1])
    Xs = window * rnd_tmp + vbmc.optim_state["plb"]
    ys = f(Xs)
    
    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        vbmc.function_logger.X_flag[sample_idx] = True
        vbmc.function_logger.x[sample_idx] = Xs[sample_idx]
        vbmc.function_logger.y[sample_idx] = ys[sample_idx]
        vbmc.function_logger.fun_evaltime[sample_idx] = 1e-5
    
    # Then make sure we get that data back.    
    X_train, y_train, s2_train, t_train = vbmc._get_training_data()
    
    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys)
    assert s2_train is None
    assert np.all(t_train == 1e-5)
    
def test_get_training_data_noise():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1
    user_options = {"specifytargetnoise": True}
    
    vbmc = VBMC(f, x0, None, None, plb, pub, user_options)
    
    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = vbmc._get_training_data()
    
    assert X_train.shape == (0, 3)
    assert y_train.shape == (0, 1)
    assert s2_train.shape == (0, 1)
    assert t_train.shape == (0, 1)
    
    # Create dummy data.
    sample_count = 10
    window = vbmc.optim_state["pub"] - vbmc.optim_state["plb"]
    rnd_tmp = np.random.rand(sample_count, window.shape[1])
    Xs = window * rnd_tmp + vbmc.optim_state["plb"]
    ys = f(Xs)
    
    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        vbmc.function_logger.X_flag[sample_idx] = True
        vbmc.function_logger.x[sample_idx] = Xs[sample_idx]
        vbmc.function_logger.y[sample_idx] = ys[sample_idx]
        vbmc.function_logger.S[sample_idx] = 1
        vbmc.function_logger.fun_evaltime[sample_idx] = 1e-5
    
    # Then make sure we get that data back.    
    X_train, y_train, s2_train, t_train = vbmc._get_training_data()
    
    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys)
    assert np.all(s2_train == 1)
    assert np.all(t_train == 1e-5)
