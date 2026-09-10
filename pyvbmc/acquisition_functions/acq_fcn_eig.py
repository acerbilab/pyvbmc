import sys

import gpyreg as gpr
import numpy as np
from scipy.linalg import solve_triangular

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnEIG(AbstractAcqFcn):
    r"""
    Expected information gain (EIG) acquisition function.

    The mutual information between a new observation :math:`y_*` at the
    candidate :math:`\theta_*` and the expected log joint
    :math:`\mathcal{G} = \int q_\phi(\theta) f(\theta) d\theta`, Eq. 7 of
    [1]_ (a port of ``acqeig_vbmc.m``):

    .. math::

        a(\theta_*) = \frac{1}{2} \log \left(1 - \rho^2(\theta_*)\right),
        \quad \rho^2(\theta_*) = \frac{\mathbb{E}_\phi[C_\Xi(f(\cdot),
        f(\theta_*))]^2}{v_\Xi(\theta_*)\, \mathbb{V}[\mathcal{G}]},

    averaged over the GP hyperparameter samples, where :math:`C_\Xi` is the
    GP posterior covariance, :math:`v_\Xi` the predictive variance inclusive
    of observation noise and :math:`\mathbb{V}[\mathcal{G}]` the posterior
    variance of the expected log joint. The integrated covariance is
    analytic for the squared exponential kernel and the Gaussian mixture
    (Eq. S18 of [1]_). Every quantity is a closed form of the same
    integrals as the expected log joint, so the acquisition costs about one
    GP prediction per candidate. The wrapper minimizes, so the value is
    minus the information gain, non-positive.

    Parameters
    ----------
    components : bool, optional
        If True, the information gain about the vector of per-component
        expected log joints :math:`(\mathcal{G}_1, \ldots, \mathcal{G}_K)`,
        :math:`\mathcal{G}_k = \int \mathcal{N}(\theta; \mu_k, \sigma_k^2
        \Lambda) f(\theta) d\theta`, instead of about their weighted sum:
        :math:`\rho^2 = c^\top J^{-1} c / v_\Xi(\theta_*)` with :math:`c_k
        = \mathbb{E}_{q_k}[C_\Xi(f(\cdot), f(\theta_*))]` and :math:`J` the
        covariance of the components' integrals. It credits observations
        that move mass between components, to which the scalar EIG is blind.
        Default False.

    Notes
    -----
    Requires ``optim_state["var_log_joint_samples"]`` (and, with
    ``components``, ``optim_state["cov_log_joint_components"]``), which
    ``active_sample`` computes before each search when
    ``acq_info["compute_var_log_joint"]`` is set.

    References
    ----------
    .. [1] Acerbi, L. (2020). "Variational Bayesian Monte Carlo with Noisy
       Likelihoods." Advances in Neural Information Processing Systems 33.
    """

    def __init__(self, components=False):
        super().__init__()
        self.acq_info["log_flag"] = False
        self.acq_info["compute_var_log_joint"] = True
        self.acq_info["components"] = bool(components)
        self.components = bool(components)

    def _compute_acquisition_function(
        self,
        Xs: np.ndarray,
        vp: VariationalPosterior,
        gp: gpr.GP,
        function_logger: FunctionLogger,
        optim_state: dict,
        f_mu: np.ndarray,
        f_s2: np.ndarray,
        f_bar: np.ndarray,
        var_tot: np.ndarray,
    ):
        """
        Compute the value of the acquisition function.

        Parameters
        ----------
        Xs : np.ndarray
            The candidates, shape ``(N, D)``, in transformed coordinates.
        vp : VariationalPosterior
            The VP object.
        gp : gpyreg.GP
            The GP object.
        function_logger : FunctionLogger
            The object responsible for caching evaluations of the log-joint.
        optim_state : dict
            The dictionary describing PyVBMC's internal state.
        f_mu : np.ndarray
            The ``(N, Ns_gp)`` GP predictive means at the candidates.
        f_s2 : np.ndarray
            The ``(N, Ns_gp)`` GP predictive variances at the candidates.
        f_bar : np.ndarray
            Unused for this acquisition function.
        var_tot : np.ndarray
            Unused for this acquisition function.
        """
        realmin = sys.float_info.min
        Ns_gp = f_mu.shape[1]

        # Estimate observation noise at test points from nearest neighbor.
        sn2 = super()._estimate_observation_noise(Xs, gp, optim_state)
        y_s2 = f_s2 + sn2.reshape(-1, 1)  # (Nx, Ns_gp), with noise

        var_G = np.atleast_1d(
            np.asarray(optim_state["var_log_joint_samples"], dtype=float)
        ).reshape(
            -1
        )  # (Ns_gp,)

        c = integrated_posterior_covariance(Xs, vp, gp)  # (Nx, Ns_gp, K)
        if getattr(self, "components", False):
            J = np.asarray(optim_state["cov_log_joint_components"])
            J = J.reshape(Ns_gp, vp.K, vp.K)
            quad = np.empty(c.shape[:2])
            for s in range(Ns_gp):
                quad[:, s] = _quadratic_form(J[s], c[:, s, :])
            rho2 = quad / y_s2
        else:
            w = np.asarray(vp.w, dtype=float).reshape(-1)
            int_K = c @ w  # (Nx, Ns_gp)
            rho2 = int_K**2 / (var_G[None, :] * y_s2)

        rho2 = np.minimum(rho2, 1.0)
        acq = 0.5 * np.sum(np.log(np.maximum(realmin, 1 - rho2)), axis=1)
        return acq / Ns_gp


def _quadratic_form(J, c):
    """``c_x^T J^{-1} c_x`` for every row of ``c``, through a Cholesky
    factor of ``J`` regularized by a jitter relative to its trace (at least
    machine epsilon in absolute terms, so relatively large when the
    covariance is far below one; safe here because ``c`` lies in the range
    of ``J``, so a near-null direction carries no weight, and the caller
    clips ``rho^2`` at one)."""
    K = J.shape[0]
    Js = 0.5 * (J + J.T)
    jitter = np.spacing(1) * max(np.trace(Js) / K, 1.0)
    for _ in range(12):
        try:
            L = np.linalg.cholesky(Js + jitter * np.eye(K))
            break
        except np.linalg.LinAlgError:
            jitter *= 100.0
    else:  # pragma: no cover - a covariance this degenerate is not expected
        return np.einsum("xk,xk->x", c @ np.linalg.pinv(Js), c)
    y = solve_triangular(L, c.T, lower=True, check_finite=False)  # (K, Nx)
    return np.sum(y**2, axis=0)


def integrated_posterior_covariance(Xs, vp, gp):
    r"""Covariance of each component's integral with the GP at the candidates.

    For every hyperparameter sample :math:`s` and component :math:`k`,

    .. math::

        c_{sk}(\theta_*) = \int \mathcal{N}(\theta; \mu_k, \sigma_k^2
        \Lambda)\, C_{\Xi, s}(f(\theta), f(\theta_*))\, d\theta
        = \mathcal{K}_{sk}(\theta_*) - z_{sk}^\top
        [\kappa_s(\Theta, \Theta) + \Sigma_{\mathrm{obs}}]^{-1}
        \kappa_s(\Theta, \theta_*),

    with :math:`\mathcal{K}_{sk}` the squared exponential kernel integrated
    against the component and :math:`z_{sk}` that integral at every
    training input (the ``z`` of ``_gp_log_joint``), Eq. S18 of Acerbi
    (2020). The mean function does not enter a posterior covariance.

    Parameters
    ----------
    Xs : np.ndarray
        The candidates, shape ``(Nx, D)``.
    vp : VariationalPosterior
        The variational posterior.
    gp : gpyreg.GP
        The GP; only the ``SquaredExponential`` covariance is supported.

    Returns
    -------
    c : np.ndarray
        Shape ``(Nx, Ns, K)``.

    Raises
    ------
    ValueError
        For a covariance function other than ``SquaredExponential``.
    """
    if not isinstance(
        gp.covariance, gpr.covariance_functions.SquaredExponential
    ):
        raise ValueError(
            "Covariance functions besides SquaredExponential are not "
            "supported yet."
        )
    Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
    D, K = vp.D, vp.K
    Ns = len(gp.posteriors)
    Nx = Xs.shape[0]
    mu_T = np.array(vp.mu, dtype=float).T  # (K, D)
    sigma = np.array(vp.sigma, dtype=float).reshape(-1)  # (K,)
    lambd = np.array(vp.lambd, dtype=float).reshape(-1)  # (D,)
    cov_N = gp.covariance.hyperparameter_count(D)

    c = np.empty((Nx, Ns, K))
    for s, post in enumerate(gp.posteriors):
        hyp = np.ravel(post.hyp)
        ell = np.exp(hyp[0:D])  # (D,)
        ln_sf2 = 2 * hyp[D]
        # Kernel integrated against each component: a Gaussian of width
        # tau at the component's mean, with normalization lnnf.
        tau = np.sqrt(sigma[:, None] ** 2 * lambd[None, :] ** 2 + ell**2)
        lnnf = ln_sf2 + np.sum(hyp[0:D]) - np.sum(np.log(tau), axis=1)
        # ... at the training inputs (K, N) and at the candidates (K, Nx).
        delta = (mu_T[:, :, None] - gp.X.T[None, :, :]) / tau[:, :, None]
        z = np.exp(
            lnnf[:, None] - 0.5 * np.einsum("kdn,kdn->kn", delta, delta)
        )
        delta = (mu_T[:, :, None] - Xs.T[None, :, :]) / tau[:, :, None]
        k_int = np.exp(
            lnnf[:, None] - 0.5 * np.einsum("kdn,kdn->kn", delta, delta)
        )
        # A = [K + Sigma_obs]^{-1} z^T, one multi-RHS solve per sample.
        if post.L_chol:
            sn2_eff = 1 / post.sW[0] ** 2
            A = (
                solve_triangular(
                    post.L,
                    solve_triangular(post.L, z.T, trans=1, check_finite=False),
                    check_finite=False,
                )
                / sn2_eff
            )  # (N, K)
        else:
            # L = -[K + Sigma_obs]^{-1} in this parametrization.
            A = -(post.L @ z.T)
        K_Xs_X = gp.covariance.compute(hyp[0:cov_N], Xs, gp.X)  # (Nx, N)
        c[:, s, :] = k_int.T - K_Xs_X @ A
    return c
