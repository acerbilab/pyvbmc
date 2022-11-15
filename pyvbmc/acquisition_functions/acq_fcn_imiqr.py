import gpyreg as gpr
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import norm

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnIMIQR(AbstractAcqFcn):
    r"""
    Integrated Median Interquantile Range (IMIQR) acquisition function.

    Approximates the Integrated Median Interquantile Range (IMIQR) via
    importance samples from the GP surrogate.
    """

    def __init__(self, quantile=0.75):
        self.acq_info = {}
        self.acq_info["log_flag"] = True
        self.acq_info["importance_sampling"] = True
        self.acq_info["importance_sampling_vp"] = False
        self.acq_info["quantile"] = quantile

        self.u = norm.ppf(quantile)

    def _compute_acquisition_function(
        self,
        Xs: np.ndarray,
        vp: VariationalPosterior,
        gp: gpr.GP,
        function_logger: FunctionLogger,
        optim_state: dict,
        f_mu: np.ndarray,
        f_s2: np.ndarray,
        f_bar: None,
        var_tot: None,
    ):
        r"""
        Compute the value of the acquisition function.

        Parameters
        ----------
        Xs : np.ndarray
            The coordinates at which to evaluate the acquisition function. Of
            shape ``(N, D)`` where ``D`` is the problem dimension.
        vp : VariationalPosterior
            The VP object.
        gp : gpyreg.GP
            The GP object.
        function_logger : FunctionLogger
            The object responsible for caching evaluations of the log-joint.
        optim_state : dict
            The dictionary describing PyVBMC's internal state.
        f_mu : np.ndarray
            A ``(N, Ns_gp)`` array of GP predictive means at the importance
            sampling points, where ``Ns_gp`` is the number of GP posterior
            hyperparameter samples.
        f_s2 : np.ndarray
            A ``(N, Ns_gp)`` array of GP predictive variances at the importance
            sampling points, where ``Ns_gp`` is the number of GP posterior
            hyperparameter samples.
        f_bar : None
            Unused for this acquisition function.
        var_tot : None
            Unused for this acquisition function.

        Raises
        ------
        ValueError
            For choices of GP covariance function which are not implemented.
            Currently, only ``SquaredExponential`` covariance is implemented.
        """
        # Xs is in *transformed* coordinates
        [Nx, D] = Xs.shape
        Ns_gp = f_mu.shape[1]

        # Estimate observation noise at test points from nearest neighbor.
        sn2 = super()._estimate_observation_noise(Xs, gp, optim_state)
        y_s2 = f_s2 + sn2.reshape(-1, 1)  # Predictive variance at test points,
        # inclusive of observation noise.

        # Different importance sampling inputs for different GP
        # hyperparameters?
        multiple_inputs_flag = (
            optim_state["active_importance_sampling"]["X"].ndim == 3
        )
        if multiple_inputs_flag:
            Na = optim_state["active_importance_sampling"]["X"].shape[1]
            Xa = np.zeros((Na, D))
        else:
            Na = optim_state["active_importance_sampling"]["X"].shape[0]
            Xa = optim_state["active_importance_sampling"]["X"]

        # Compute acquisition function via importance sampling
        acq = np.zeros((Nx, Ns_gp))

        cov_N = gp.covariance.hyperparameter_count(gp.D)
        for s in range(Ns_gp):
            cov_hyp = gp.posteriors[s].hyp[0:cov_N]  # Covariance hyperparams
            L = gp.posteriors[s].L
            L_chol = gp.posteriors[s].L_chol
            sn2_eff = 1 / gp.posteriors[s].sW[0] ** 2

            if multiple_inputs_flag:
                Xa[:, :] = optim_state["active_importance_sampling"]["X"][
                    s, :, :
                ]

            # Compute cross-kernel matrices
            if isinstance(
                gp.covariance, gpr.covariance_functions.SquaredExponential
            ):
                K_X_Xs = gp.covariance.compute(cov_hyp, gp.X, Xs)
                K_Xa_Xs = gp.covariance.compute(cov_hyp, Xa, Xs)
                K_Xa_X = optim_state["active_importance_sampling"]["K_Xa_X"][
                    s, :, :
                ]
            else:
                raise ValueError(
                    "Covariance functions besides"
                    + "SquaredExponential are not supported yet."
                )

            if L_chol:
                C = (
                    K_Xa_Xs.T
                    - K_X_Xs.T
                    @ solve_triangular(
                        L,
                        solve_triangular(
                            L, K_Xa_X.T, trans=True, check_finite=False
                        ),
                        check_finite=False,
                    )
                    / sn2_eff
                )
            else:
                C = K_Xa_Xs.T + K_X_Xs.T @ (L @ K_Xa_X.T)

            tau2 = C**2 / y_s2[:, s].reshape(-1, 1)
            s_pred = np.sqrt(
                np.maximum(
                    optim_state["active_importance_sampling"]["f_s2"][:, s].T
                    - tau2,
                    0.0,
                )
            )

            ln_weights = optim_state["active_importance_sampling"][
                "ln_weights"
            ][s, :]

            # zz = ln(weights * sinh(u * s_pred)) + C
            zz = (
                ln_weights
                + self.u * s_pred
                + np.log1p(-np.exp(-2 * self.u * s_pred))
            )
            # logsumexp
            ln_max = np.amax(zz, axis=1)
            ln_max[ln_max == -np.inf] = 0.0  # Avoid -inf + inf
            acq[:, s] = (
                np.log(np.sum(np.exp(zz - ln_max.reshape(-1, 1)), axis=1))
                + ln_max
            )

        if Ns_gp > 1:
            M = np.amax(acq, axis=1)
            M[M == -np.inf] = 0.0  # Avoid -inf + inf
            acq = M + np.log(
                np.sum(np.exp(acq - M.reshape(-1, 1)), axis=1) / Ns_gp
            )

        return acq

    def is_log_base(self, x, **kwargs):
        r"""Importance sampling proposal log density, base part.

        The base density of the importance sampling proposal distribution, used
        for computing i.s. weights (in addition to the full proposal density).
        The full proposal log density is ``is_log_full = is_log_base +
        is_log_added``. IMIQR approximates an expectation w.r.t. to the
        (exponentiated) GP using importance sampling, so this base log density
        is the GP mean, ``f_mu``.

        Parameters
        ----------
        f_mu : np.ndarray
            The gp posterior mean at the points of interest.

        Returns
        -------
        f_mu : np.ndarray
            The log density of the target against which the i.s. expectation is
            taken. Here, just the log density as modeled by the GP.
        """
        return kwargs["f_mu"]

    def is_log_added(self, **kwargs):
        r"""Importance sampling proposal log density, added part.

        The added term in the importance sampling proposal log density: The
        full proposal log density is ``is_log_full = is_log_base +
        is_log_added``. Added part for VIQR/IMIQR is :math: `\\log [\\sinh(u *
        f_s)]``, where ``f_s`` is the GP predictive variance at the input
        points.

        Parameters
        ----------
        f_s2 : np.ndarray
            The predicted posterior variance at the points of interest.

        Returns
        -------
        y : np.ndarray
            The added log density term at the points of interest.
        """
        f_s = np.sqrt(kwargs["f_s2"])
        return self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))

    def is_log_full(self, x, **kwargs):
        r"""Importance sampling full proposal log density.

        The full proposal log density, used for MCMC sampling: ``is_log_full =
        is_log_base + is_log_added``.

        Parameters
        ----------
        f_mu : np.ndarray
            The gp posterior mean at the points of interest. Either [``f_s2``
            and ``f_mu``] or ``gp`` must be provided.
        f_s2 : np.ndarray, optional
            The predicted posterior variance at the points of interest. Either
            [``f_s2`` and ``f_mu``] or ``gp`` must be provided.
        gp : gpyreg.GaussianProcess, optional
            The GP modeling the log-density. Either [``f_s2`` and ``f_mu``] or
            ``gp`` must be provided.

        Returns
        -------
        y : np.ndarray
            The full log density of the importance sampling proposal
            distribution at the points of interest. Of the same shape as
            ``f_s2``.

        Raises
        ------
        ValueError
            If [``f_s2`` or ``f_mu``] are not provided, and ``gp`` is not
            provided.
        """
        f_mu = kwargs.pop("f_mu", None)
        f_s2 = kwargs.pop("f_s2", None)
        if f_mu is None or f_s2 is None:  # Try to get pre-computed f_mu/s2
            gp = kwargs.get("gp")
            if gp is None:  # Otherwise use GP to predict
                raise ValueError(
                    "Must provide gp as keyword argument if f_mu/f_s2 are not"
                    + "provided."
                )
            f_mu, f_s2 = gp.predict(np.atleast_2d(x), add_noise=True)
        return self.is_log_base(
            x, f_mu=f_mu, f_s2=f_s2, **kwargs
        ) + self.is_log_added(f_mu=f_mu, f_s2=f_s2, **kwargs)
