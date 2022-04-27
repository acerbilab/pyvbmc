import numpy as np
import scipy as sp
import scipy.stats as sps

def pseudo_likelihood(sim_fun, summary, data=None, epsilon=1.0, a=0.9, p=0.99, df=7, return_scale=False):
    r"""Construct a pseudo-likelihood for likelihood-free inference.

    Parameters
    ----------
    sim_fun : callable
        The simulation function, which takes an array of parameters :\math: `\theta` and returns a generated dataset, or (if `summary` is `None`) a statistic measuring the distance between the generated and observed data.
    summary : callable
        The summary statistic, which takes data and returns a scalar. The
        likelihood is then computed as :math: `q(|summary(d_{\theta}) - summary(d_{obs})|)`, where
        :math: `q` is described below.
    data
        The observed data. If `None`, then the returned callable will expect
        the data as a second argument.
    epsilon : float
        The window :math: `(0, \epsilon)` will contain a fraction `p` of the likelihood
        probability mass of :math: `q`. Default `1.0`.
    a : float, in [0, 0.995]
        :math: `q(u)` will be maximum and constant for :math: `u \in [0, a*\epsilon]`. Default `0.9`.
    p : float, in (0, 1)
        The probability mass to be contained in the window :math: `(0, \epsilon)`. Default `0.99`.
    df : int, >= 1
        The degrees of freedom for the Student's t-distribution determining the rate at which :math: `q(u)` vanishes for :math: `u > \epsilon`. Default `7`.
    return_scale : bool
        Whether to return the vertical and horizontal scaling factors of the Student's t-distribution. Default `False`.
    """
    if not (0 <= a) and (a <= 0.995):
        raise ValueError("Parameter should be 0 <= a <= 0.995.")
    if not (0 <= a) and (a <= 0.995):
        raise ValueError("Parameter p should be 0 < p < 1.")
    st = sps.t(df=df)  # Base Student's t-distribution
    v_scale = 1 / st.pdf(0)  # Continuity at a * epsilon

    def target(h_scale):
        int_0_eps = a*epsilon + (v_scale / h_scale) * (st.cdf((1-a) * epsilon * h_scale) - 0.5)
        int_0_inf = a*epsilon + 0.5 * (v_scale / h_scale)
        return int_0_eps / int_0_inf - p

    # Establish lower and upper bounds, then optimize:
    lower = 1/epsilon * a * p + 1e-16
    upper = 1/epsilon * 1/(1-a) * 1/(1-p)
    h_scale = sp.optimize.brentq(target, lower, upper)

    norm_factor = a*epsilon + 0.5 * (v_scale / h_scale)

    def ll(s):
        if s <= a * epsilon:
            return -np.log(norm_factor)
        else:
            return -st.logpdf(0.0) + st.logpdf((s - a * epsilon) * h_scale)\
                - np.log(norm_factor)

    if not np.isclose(
            p,
            sp.integrate.quad(lambda s: np.exp(ll(s)), 0, epsilon)[0]
    ):
        raise ValueError("Could not find a solution, please try with softer" +
                         "rolloff (lower a and/or p).")

    def log_likelihood(theta, d=data):
        d_theta = sim_fun(theta)
        delta = np.abs(summary(d_theta) - summary(d))
        return ll(delta)

    if return_scale:  # Primarily for testing
        return log_likelihood, v_scale, h_scale
    else:
        return log_likelihood
