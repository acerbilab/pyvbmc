import gpyreg as gpr
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnVIQR(AbstractAcqFcn):
    r"""
    Variational Interquantile Range (VIQR) acquisition function.

    Approximates the Integrated Median Interquantile Range (IMIQR) by simple
    Monte Carlo using samples from the Variational Posterior.
    """

    def __init__(self, quantile=0.75):
        self.acq_info = {}
        self.acq_info["log_flag"] = True
        self.acq_info["importance_sampling"] = True
        self.acq_info["importance_sampling_vp"] = False
        self.acq_info["variational_importance_sampling"] = True
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
        # Missing port, integrated mean function, lines 49 to 57.

        # Xs is in *transformed* coordinates
        [Nx, D] = Xs.shape
        Ns_gp = f_mu.shape[1]

        # Estimate observation noise at test points from nearest neighbor.
        sn2 = super()._estimate_observation_noise(Xs, gp, optim_state)
        y_s2 = f_s2 + sn2.reshape(-1, 1)  # Predictive variance at test points,
        # inclusive of observation noise.

        Xa = optim_state["active_importance_sampling"]["X"]
        acq = np.zeros((Nx, Ns_gp))

        # Compute acquisition function via importance sampling

        cov_N = gp.covariance.hyperparameter_count(gp.D)
        for s in range(Ns_gp):
            hyp = gp.posteriors[s].hyp[0:cov_N]  # Covariance hyperparameters
            L_chol = gp.posteriors[s].L_chol

            # Compute cross-kernel matrices
            if isinstance(
                gp.covariance, gpr.covariance_functions.SquaredExponential
            ):  # Hard-coded SE-ard for speed (re-use Xs_ell)
                # Functionally equivalent to:
                # K_Xs_X = gp.covariance.compute(hyp, Xs, gp.X)
                # K_Xs_Xa = gp.covariance.compute(hyp, Xs, Xa)
                # K_Xa_X = optim_state["active_importance_sampling"]["K_Xa_X"][
                #     s, :, :
                # ]
                ell = np.exp(hyp[0:D])
                sf2 = np.exp(2 * hyp[D])
                Xs_ell = Xs / ell

                tmp = cdist(Xs_ell, gp.X / ell, "sqeuclidean")
                K_Xs_X = sf2 * np.exp(-tmp / 2)

                tmp = cdist(Xs_ell, Xa / ell, "sqeuclidean")
                K_Xs_Xa = sf2 * np.exp(-tmp / 2)

                C_tmp = optim_state["active_importance_sampling"]["C_tmp"][
                    s, :, :
                ]
            else:
                raise ValueError(
                    "Covariance functions besides"
                    + "SquaredExponential are not supported yet."
                )

            if L_chol:
                C = K_Xs_Xa - K_Xs_X @ C_tmp
            else:
                C = K_Xs_Xa + K_Xs_X @ C_tmp

            # Missing port, integrated meanfun

            tau2 = C**2 / y_s2[:, s].reshape(-1, 1)
            s_pred = np.sqrt(
                np.maximum(
                    optim_state["active_importance_sampling"]["f_s2"][:, s].T
                    - tau2,
                    0.0,
                )
            )

            # zz = ln(weights * sinh(u * s_pred)) + C
            # (VIQR uses simple Monte Carlo, so weights are constant).
            zz = self.u * s_pred + np.log1p(-np.exp(-2 * self.u * s_pred))
            # logsumexp
            ln_max = np.amax(zz, axis=1)
            ln_max[ln_max == -np.inf] = 0.0  # Avoid -inf + inf
            acq[:, s] = ln_max + np.log(
                np.sum(np.exp(zz - ln_max.reshape(-1, 1)), axis=1)
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
        is_log_added``. VIQR approximates an expectation w.r.t. to the VP
        using simple Monte Carlo, so this base density is constant (log=0).

        Parameters
        ----------
        x : np.ndarray
            The input points, shape ``(N, D)`` where ``N`` is the number of
            points and ``D`` is the dimension.
        f_s2 : np.ndarray
            The predicted posterior variance at the points of interest.

        Returns
        -------
        z : np.ndarray
            The log base part of the importance sampling weights (zeros), shape
            ``f_s2``.
        """
        f_s2 = kwargs["f_s2"]
        return np.zeros(f_s2.shape)

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
            The added log density term at the points of interest. Of the same
            shape as ``f_s2``.
        """
        f_s = np.sqrt(kwargs["f_s2"])
        return self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))

    def is_log_full(self, x, **kwargs):
        r"""Importance sampling full proposal log density.

        The full proposal log density, used for MCMC sampling: ``is_log_full =
        is_log_base + is_log_added``.

        Parameters
        ----------
        f_s2 : np.ndarray, optional
            The predicted posterior variance at the points of interest. Either
            ``f_s2`` or ``gp`` must be provided.
        gp : gpyreg.GaussianProcess, optional
            The GP modeling the log-density. Either ``f_s2`` or ``gp`` must be
            provided.

        Returns
        -------
        y : np.ndarray
            The full log density of the importance sampling proposal
            distribution at the points of interest. Of the same shape as
            ``f_s2``.

        Raises
        ------
        ValueError
            If neither ``f_s2`` nor ``gp`` are provided.
        """
        f_s2 = kwargs.pop("f_s2", None)  # Try to get pre-computed f_s2
        if f_s2 is None:  # Otherwise use GP to predict
            gp = kwargs.get("gp")
            if gp is None:
                raise ValueError(
                    "Must provide gp as keyword argument if f_s2 is not"
                    + "provided."
                )
            __, f_s2 = gp.predict(np.atleast_2d(x), add_noise=True)
        # base + added (base part is 0):
        return self.is_log_added(f_s2=f_s2, **kwargs)
