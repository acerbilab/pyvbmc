import pytest
import numpy as np
import numpy.random as npr
import scipy.stats as sps

from pyvbmc.likelihood_free import pseudo_likelihood
from pyvbmc.vbmc import VBMC


def test_pseudo_likelihood_simple():
    ll, v_scale, h_scale = pseudo_likelihood(lambda t: t, lambda d: d, data=0.0, return_scale=True)
    x1 = np.linspace(0, 0.9)
    x2 = np.linspace(0.9, 10.0)
    y1 = np.array([ll(xi) for xi in x1])
    y2 = np.array([ll(xi) for xi in x2])
    assert np.all(y1 == y1[0])  # Constant up to a * epsilon
    assert np.all(y2[1:] < y2[:-1])  # Decreasing after a*epsilon
    assert np.allclose(y1, np.log(v_scale))


def test_pseudo_likelihood_complex():
    D = 2
    N = 1000

    def fake_sim(theta):
        return sps.multivariate_normal.rvs(cov=np.eye(D), size=N)

    def fake_summary(d_theta):
        return np.var(d_theta[:, 0], ddof=1)

    d_obs = 0.5 * sps.multivariate_normal.rvs(cov=np.eye(D), size=N)

    ll_data, v_scale, h_scale = pseudo_likelihood(
        fake_sim, fake_summary, data=d_obs, return_scale=True
    )
    ll_no_data = pseudo_likelihood(
        fake_sim,
        fake_summary,
    )

    # Call without default data produces missing argument error:
    with pytest.raises(TypeError) as execinfo:
        ll_no_data(0)
    assert (
        "log_likelihood() missing 1 required positional argument: 'd'"
        in execinfo.value.args[0]
    )

    M = 10
    thetas = np.arange(M)
    data_lls = np.zeros(M)
    no_data_lls = np.zeros(M)
    for (i, theta) in enumerate(thetas):
        data_lls[i] = ll_data(theta)
        no_data_lls[i] = ll_no_data(theta, d_obs)
    assert np.allclose(data_lls, no_data_lls, rtol=1e-4)
    assert np.allclose(data_lls, np.log(v_scale), rtol=1e-4)

    N = 10000
    d_obs = np.sqrt(2) * sps.multivariate_normal.rvs(cov=np.eye(D), size=N)
    ll, v_scale, h_scale = pseudo_likelihood(
        fake_sim, fake_summary, data=d_obs, a=0.0, return_scale=True
    )
    # Difference in variance should be about 1.
    # q(u) is truncated Student's t:
    val = np.log(2) + sps.t(df=7, scale=h_scale).logpdf(1.0)
    lls = np.array([ll(theta) for theta in range(10)])
    assert np.allclose(lls, val, rtol=1e-1)


def test_q_random():
    N = 10
    dfs = npr.randint(low=1, high=30, size=N)
    eps = npr.lognormal(sigma=8, size=N)
    aas = npr.uniform(0, 0.995, size=N)
    ps = npr.uniform(0, 1, size=N)
    for (df, ep, a, p) in zip(dfs, eps, aas, ps):
        # Should find a solution without error:
        ll = pseudo_likelihood(
            lambda t: t, lambda d: d, data=0.0, epsilon=ep, a=a, p=p, df=df
        )

        # When a is zero, should be a truncated Student's t:
        ll, v_scale, h_scale = pseudo_likelihood(
            lambda t: t,
            lambda d: d,
            data=0.0,
            epsilon=ep,
            a=0.0,
            p=p,
            df=df,
            return_scale=True,
        )
        x = np.linspace(0, 5 * ep)
        y1 = np.log(2) + sps.t(df=df, scale=h_scale).logpdf(x)
        y2 = np.array([ll(xi, 0.0) for xi in x])
        assert np.allclose(y1, y2)


def test_vbmc_optimize_pseudo_ll():
    D = 5
    N = 500

    sim_data = sps.multivariate_normal.rvs(cov=np.eye(D), size=N)
    d_obs = 2 * sps.multivariate_normal.rvs(cov=np.eye(D), size=N)

    def fake_sim(theta):  # Deterministic simulation
        return np.linalg.norm(theta) * sim_data

    # Variance close to 1 for d_obs, close to norm(theta) for d_theta:
    def fake_summary(d_theta):
        return np.mean(np.var(d_theta, axis=0))

    llfun = pseudo_likelihood(fake_sim, fake_summary, data=d_obs, epsilon=1.0)

    def ltarget(t):  # Pseudo-likelihood + wide prior
        return llfun(t) + sps.multivariate_normal.logpdf(t, cov=8 * np.eye(D))

    x0 = np.ones((1, D)) ** (1 / D)
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    plb = -2 * np.ones((1, D))
    pub = 2 * np.ones((1, D))
    options = {
        # "searchacqfcn": ["@acqviqr_vbmc"],
        "maxfunevals": 15  # Just run until end of warm-up.
    }

    vbmc = VBMC(ltarget, x0, lb, ub, plb, pub, user_options=options)
    vbmc.optimize()
