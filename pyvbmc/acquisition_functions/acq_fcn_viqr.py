import sys

import gpyreg as gpr
import numpy as np
from scipy.linalg import solve_triangular
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnVIQR(AbstractAcqFcn):
    r"""
    Variational Interquantile Range (VIQR) acquisition function.

    Approximates the Integrated Median Interquantile Range (IMIQR) by simple
    Monte Carlo using samples from the Variational Posterior.
    """

    def __init__(self):
        self.importance_sampling = True
        self.importance_sampling_vp = False
        self.variational_importance_sampling = True
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
        # Missing port, integrated mean function, lines 49 to 57.

        # Xs is in *transformed* coordinates
        [Nx, D] = Xs.shape
        Ns_gp = f_mu.shape[1]

        # Estimate observation noise at test points from nearest neighbor.
        sn2 = super()._estimate_observation_noise(Xs, gp, optim_state)
        y_s2 = f_s2 + sn2.reshape(-1,1)  # Predictive variance at test points

        Xa = optim_state["active_importance_sampling"]["Xa"]
        acq = np.zeros((Nx, Ns_gp))

        # Compute acquisition function via importance sampling

        cov_N = gp.covariance.hyperparameter_count(gp.D)
        for s in range(Ns_gp):
            hyp = gp.posteriors[s].hyp[0:cov_N]  # Covariance hyperparameters
            L = gp.posteriors[s].L
            L_chol = gp.posteriors[s].L_chol
            sn2_eff = 1 / gp.posteriors[s].sW[1]**2

            # Compute cross-kernel matrices
            if isinstance(gp.covariance,
                          gpr.covariance_functions.SquaredExponential):
                Ks_mat = gp.covariance.compute(hyp, gp.X, Xs)
                Ka_mat = gp.covariance.compute(hyp, Xa, Xs)
                Kax_mat = optim_state["active_importance_sampling"]["Kax_mat"][:, :, s]
            else:
                raise ValueError("Covariance functions besides" ++
                                 "SquaredExponential are not supported yet.")

            if L_chol:
                C = Ka_mat.T - Ks_mat.T @ \
                    solve_triangular(L,
                                     solve_triangular(L,
                                                      Kax_mat.T,
                                                      trans=True,
                                                      check_finite=False
                                                      ),
                                     check_finite=False
                                     ) / sn2_eff
            else:
                C = Ka_mat.T + Ks_mat.T @ (L @ Kax_mat.T)

            # Missing port, integrated meanfun

            tau2 = C**2 / y_s2[:, s].reshape(-1, 1)
            s_pred = np.sqrt(np.maximum(
                optim_state["active_importance_sampling"]["f_s2a"][:, s].T
                - tau2,
                0.0
            ))

            ln_w = optim_state["active_importance_sampling"]["ln_w"][s, :]
            # ln_w should be 0 here: since we are sampling Xa from the VP
            # no extra importance sampling weight is required.
            # It is included for compatibility.

            # zz = ln(weights * sinh(u * s_pred))
            zz = ln_w + self.u * s_pred\
                + np.log1p(-np.exp(-2 * self.u * s_pred))
            # logsumexp
            ln_max = np.amax(zz, axis=1)
            ln_max[ln_max == -np.inf] = 0.0  # Avoid -inf + inf
            acq[:, s] = np.log(
                np.sum(np.exp(zz - ln_max.reshape(-1, 1)), axis=1)
            ) + ln_max

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
        return np.zeros(f_s2.shape)

    def is_log_f2(self, f_mu, f_s2):
        # Importance sampling log base proposal (shared part)
        f_s = np.sqrt(f_s2)
        return self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))

    def is_log_f(self, v_ln_pdf, f_mu, f_s2):
        # Importance sampling log base proposal distribution
        f_s = np.sqrt(f_s2)
        return v_ln_pdf + self.u * f_s + np.log1p(-np.exp(-2 * self.u * f_s))
