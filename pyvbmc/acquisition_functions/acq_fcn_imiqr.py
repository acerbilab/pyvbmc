import sys

import gpyreg as gpr
import numpy as np
from scipy.linalg import solve_triangular
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnIMIQR(AbstractAcqFcn):
    r"""
    Integrated Median Interquantile Range (IMIQR) acquisition function.

    Approximates the Integrated Median Interquantile Range (IMIQR) via
    importance samples from the GP surrogate.
    """

    def __init__(self):
        self.importance_sampling = True
        self.importance_sampling_vp = False
        self.log_flag = True
        self.u = 0.6745  # inverse normal cdf of 0.75

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
        r"""
        Compute the value of the acquisition function.
        """
        # Xs is in *transformed* coordinates
        [Nx, D] = Xs.shape
        Ns_gp = f_mu.shape[1]
        Na = optim_state["active_importance_sampling"]["X"].shape[0]

        # Estimate observation noise at test points from nearest neighbor.
        sn2 = super()._estimate_observation_noise(Xs, gp, optim_state)
        y_s2 = f_s2 + sn2.reshape(-1,1)  # Predictive variance at test points

        # Different importance sampling inputs for different GP
        # hyperparameters?
        multiple_inputs_flag = optim_state["active_importance_sampling"]["X"].ndim == 3
        if multiple_inputs_flag:
            Xa = np.zeros((Na, D))
        else:
            Xa = optim_state["active_importance_sampling"]["X"]
        acq = np.zeros((Nx, Ns_gp))

        # Compute acquisition function via importance sampling

        cov_N = gp.covariance.hyperparameter_count(gp.D)
        for s in range(Ns_gp):
            cov_hyp = gp.posteriors[s].hyp[0:cov_N]  # Covariance hyperparams
            L = gp.posteriors[s].L
            L_chol = gp.posteriors[s].L_chol
            sn2_eff = 1 / gp.posteriors[s].sW[1]**2

            if multiple_inputs_flag:
                Xa[:, :] = optim_state["active_importance_sampling"]["X"][:, :, s]

            # Compute cross-kernel matrices
            if isinstance(gp.covariance,
                          gpr.covariance_functions.SquaredExponential):
                K_X_Xs = gp.covariance.compute(cov_hyp, gp.X, Xs)
                K_Xa_Xs = gp.covariance.compute(cov_hyp, Xa, Xs)
                K_Xa_X = optim_state["active_importance_sampling"]["K_Xa_X"][:, :, s]
            else:
                raise ValueError("Covariance functions besides" ++
                                 "SquaredExponential are not supported yet.")

            if L_chol:
                C = K_Xa_Xs.T - K_X_Xs.T @ \
                    solve_triangular(L,
                                     solve_triangular(L,
                                                      K_Xa_X.T,
                                                      trans=True,
                                                      check_finite=False
                                                      ),
                                     check_finite=False
                                     ) / sn2_eff
            else:
                C = K_Xa_Xs.T + K_X_Xs.T @ (L @ K_Xa_X.T)

            tau2 = C**2 / y_s2[:, s].reshape(-1, 1)
            s_pred = np.sqrt(np.maximum(
                optim_state["active_importance_sampling"]["f_s2"][:, s].T
                - tau2,
                0.0
            ))

            ln_weights = optim_state["active_importance_sampling"]["ln_weights"][s, :]

            # zz = ln(weights * sinh(u * s_pred)) + C
            zz = ln_weights + self.u * s_pred\
                + np.log1p(-np.exp(-2 * self.u * s_pred))
            # logsumexp
            ln_max = np.amax(zz, axis=1)
            ln_max[ln_max == -np.inf] = 0.0  # Avoid -inf + inf
            __, n_samples = zz.shape
            acq[:, s] = np.log(
                np.sum(np.exp(zz - ln_max.reshape(-1, 1)), axis=1)
            ) + ln_max - np.log(n_samples)

        if Ns_gp > 1:
            M = np.amax(acq, axis=1)
            M[M == -np.inf] = 0.0  # Avoid -inf + inf
            acq = M + np.log(
                np.sum(np.exp(acq - M.reshape(-1, 1)), axis=1)
                / Ns_gp
            )

        return acq


    def is_log_f1(self, v_ln_pdf, f_mu, f_s2):
        # Importance sampling log base proposal (shared part)
        return f_mu

    def is_log_f2(self, f_mu, f_s2):
        # Importance sampling log base proposal (shared part)
        f_s = np.sqrt(f_s2)
        return self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))

    def is_log_f(self, v_ln_pdf, f_mu, f_s2):
        # Importance sampling log base proposal distribution
        f_s = np.sqrt(f_s2)
        return f_mu + self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))
