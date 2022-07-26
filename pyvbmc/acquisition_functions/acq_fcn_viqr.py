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
        self.acq_info = dict()
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
                # K_X_Xs = gp.covariance.compute(hyp, gp.X, Xs)
                # K_Xa_Xs = gp.covariance.compute(hyp, Xa, Xs)
                # K_Xa_X = optim_state["active_importance_sampling"]["K_Xa_X"][
                #     :, :, s
                # ]
                ell = np.exp(hyp[0:D])
                sf2 = np.exp(2 * hyp[D])
                Xs_ell = Xs / ell

                tmp = cdist(gp.X / ell, Xs_ell, "sqeuclidean")
                K_X_Xs = sf2 * np.exp(-tmp / 2)

                tmp = cdist(Xa / ell, Xs_ell, "sqeuclidean")
                K_Xa_Xs = sf2 * np.exp(-tmp / 2)

                # K_Xa_X = optim_state["active_importance_sampling"]["K_Xa_X"][
                #     :, :, s
                # ]
                C_tmp = optim_state["active_importance_sampling"]["C_tmp"][
                    :, :, s
                ].copy()
            else:
                raise ValueError(
                    "Covariance functions besides"
                    + "SquaredExponential are not supported yet."
                )

            if L_chol:
                # C = (
                #     K_Xa_Xs.T
                #     - K_X_Xs.T
                #     @ solve_triangular(
                #         L,
                #         solve_triangular(
                #             L, K_Xa_X.T, trans=True, check_finite=False
                #         ),
                #         check_finite=False,
                #     )
                #     / sn2_eff
                # )
                C = K_Xa_Xs.T - K_X_Xs.T @ C_tmp
                # print(C.shape)
            else:
                # C = K_Xa_Xs.T + K_X_Xs.T @ (L @ K_Xa_X.T)
                C = K_Xa_Xs.T + K_X_Xs.T @ C_tmp

            # Missing port, integrated meanfun

            tau2 = C**2 / y_s2[:, s].reshape(-1, 1)
            s_pred = np.sqrt(
                np.maximum(
                    optim_state["active_importance_sampling"]["f_s2"][:, s].T
                    - tau2,
                    0.0,
                )
            )

            # ln_weights is 0 here: VIQR uses simple Monte Carlo, not
            # importance sampling.
            ln_weights = optim_state["active_importance_sampling"][
                "ln_weights"
            ][s, :]

            # zz = ln(sinh(u * s_pred)) + C
            zz = (
                ln_weights
                + self.u * s_pred
                + np.log1p(-np.exp(-2 * self.u * s_pred))
            )
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

        assert np.all(~np.isnan(acq))
        return acq

    def is_log_f1(self, v_ln_pdf, f_mu, f_s2):
        """Importance sampling log base proposal (fixed part)."""
        return np.zeros(f_s2.shape)

    def is_log_f2(self, f_mu, f_s2):
        """% Importance sampling log base proposal (added part)
        (Full log base proposal is fixed + added)"""
        f_s = np.sqrt(f_s2)
        return self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))

    def is_log_f(self, v_ln_pdf, f_mu, f_s2):
        """Importance sampling log base proposal distribution."""
        f_s = np.sqrt(f_s2)
        return v_ln_pdf + self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))
