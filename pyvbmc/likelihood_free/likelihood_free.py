import numpy as np
import scipy as sp
import scipy.stats as sps


def pseudo_likelihood(
    sim_fun,
    epsilon,
    summary=lambda d: d,
    data=None,
    a=0.9,
    p=0.99,
    df=7,
    return_plot_fun = False,
    return_scale=False
):
    r"""Construct a pseudo-likelihood for likelihood-free inference.

    Parameters
    ----------
    sim_fun : callable
        The simulation function, which takes a (1,D) array of parameters
        :\math: `\theta` and returns a dataset :math: `d_{\theta}` generated by
        those parameters.
    epsilon : float
        The window :math: `(0, \epsilon)` will contain a fraction `p` of the
        likelihood probability mass of :math: `q`.
    summary : callable
        The summary statistic, which takes data and returns a scalar. The
        likelihood is then computed as
        :math: `q(|summary(d_{\theta}) - summary(d_{obs})|)`, where :math: `q`
        is described below. Default is identity, in case the simulation itself
        already returns a scalar summary statistic.
    data
        The observed data :math: `d_{obs}`. If `None`, then the
        returned callable will require the data as a second argument, and will
        evaluate `summary(d_obs)` on each call in addition to
        `summary(d_theta)`.
    a : float, optional, in [0, 0.995]
        :math: `q(u)` will be maximum and constant for
        :math: `u \in [0, a*\epsilon]`. Default `0.9`.
    p : float, optional, in (0, 1)
        The probability mass to be contained in the window
        :math: `(0, \epsilon)`. Default `0.99`.
    df : int, optional, >= 1
        The degrees of freedom for the Student's t-distribution determining the
        rate at which :math: `q(u)` vanishes for :math: `u > \epsilon`.
        Default `7`.
    return_plot_fun : bool, optional
        Whether to return the likelihood as a function of :math: `s = S(D)`,
        e.g. for plotting/debugging. Default `False`
    return_scale : bool, optional
        Whether to return the vertical and horizontal scaling factors of the
        Student's t-distribution. Default `False`.

    Returns
    -------
    log_likelihood : callable
        If `data` is not `None`, the returned `log_likelihood` will require
        `theta` as a first argument, and `d=data` as an optional
        second argument with default `data`. Otherwise `log_likelihood` will
        require two arguments, `theta` and `d`.
    plot_fun : callable, optional
        If `return_plot_fun` is `True`, this will return the log-likelihood as
        a function of the distance of summary statistics
        :math: `\delta = |S(d_{\theta}) - S(d_{obs})|`, for plotting/debugging
        purposes.
    v_scale : float
        If `return_scale` is `True`, returns the maximum value of the
        likelihood.
    h_scale : float
        If `return_scale` is `True`, returns the horizontal rescaling factor
        chosen for the Student's t tail (equivalent to scipy's `scale`
        parameter).
    Raises
    ------
    ValueError
        If it is not the case that epsilon > 0,  0 <= a <= 0.995,  and
        0 < p < 1.
    """
    if not (epsilon >= 0):
        raise ValueError("Parameter epsilon be > 0.")
    if not (0 <= a) and (a <= 0.995):
        raise ValueError("Parameter should be 0 <= a <= 0.995.")
    if not (0 <= a) and (a <= 0.995):
        raise ValueError("Parameter p should be 0 < p < 1.")

    if df == np.inf:
        st = sps.norm()  # Base normal distribution
    else:
        st = sps.t(df=df)  # Base Student's t-distribution
    v_scale = 1 / st.pdf(0)  # Continuity at a * epsilon
    # Find horizontal scale such that `p` prob. mass is inside (0, a*epsilon)

    def target(h_scale):
        int_0_eps = a * epsilon + (v_scale / h_scale) * (
            st.cdf((1 - a) * epsilon * h_scale) - 0.5
        )
        int_0_inf = a * epsilon + 0.5 * (v_scale / h_scale)
        return int_0_eps / int_0_inf - p

    # Establish lower and upper bounds, then optimize:
    lower = 1 / epsilon * a * p + 1e-16
    upper = 1 / epsilon * 1 / (1 - a) * 1 / (1 - p)
    h_scale = sp.optimize.brentq(target, lower, upper)

    norm_factor = a * epsilon + 0.5 * (v_scale / h_scale)

    def ll(delta):
        if delta <= a * epsilon:
            return -np.log(norm_factor)
        else:
            return (
                -st.logpdf(0.0)
                + st.logpdf((delta - a * epsilon) * h_scale)
                - np.log(norm_factor)
            )

    if not np.isclose(
        p, sp.integrate.quad(lambda delta: np.exp(ll(delta)), 0, epsilon)[0]
    ):
        raise ValueError(
            "Could not find a solution, please try with softer"
            + "rolloff (lower a and/or p)."
        )

    if data is not None:
        summary_data = summary(data)

        def log_likelihood(theta):
            if np.ndim(theta) > 1:
                lls = []
                nrows, __ = theta.shape
                for i in range(nrows):
                    d_theta = sim_fun(theta[i, :])
                    delta = np.abs(summary(d_theta) - summary_data)
                    lls.append(ll(delta))
                return np.array(lls)
            else:
                d_theta = sim_fun(theta)
                delta = np.abs(summary(d_theta) - summary_data)
                return ll(delta)

    else:
        def log_likelihood(theta, d):
            d_theta = sim_fun(theta)
            delta = np.abs(summary(d_theta) - summary(d))
            return ll(delta)

    if return_plot_fun:
        def ll_plot(delta):
            if np.ndim(delta) == 0:
                return ll(delta)
            else:
                return np.array([ll(d) for d in delta])
        if return_scale:
            return log_likelihood, ll_plot, 1/norm_factor, 1/h_scale
        else:
            return log_likelihood, ll_plot
    else:
        if return_scale:
            return log_likelihood, 1/norm_factor, 1/h_scale
        else:
            return log_likelihood