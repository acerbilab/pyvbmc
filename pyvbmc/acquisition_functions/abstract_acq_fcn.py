import sys
from abc import ABC, abstractmethod

import gpyreg as gpr
import numpy as np

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior


class AbstractAcqFcn(ABC):
    """
    Abstract acquisition function for VBMC.
    """

    def __init__(self):
        self.acq_info = {}
        self.acq_info["compute_var_log_joint"] = False
        # Whether the function value is in log space
        self.acq_info["log_flag"] = False

    def get_info(self):
        """
        Return a dict with information about the acquisition function.

        Returns
        -------
        acq_info : dict
            A dict containing information about the acquisition function.
        """
        return self.acq_info

    def __call__(
        self,
        Xs: np.ndarray,
        gp: gpr.GP,
        vp: VariationalPosterior,
        function_logger: FunctionLogger,
        optim_state: dict,
    ):
        """
        Calculate the acquisition function for the given inputs.

        Parameters
        ----------
        Xs : np.ndarray
            Input points.
        gp : gpr.GP
            The GaussianProcess of the VBMC instance this function is
            called from.
        vp : VariationalPosterior
            The VariationalPosterior of the VBMC instance this function is
            called from.
        function_logger : FunctionLogger
            The FunctionLogger of the VBMC instance this function is
            called from.
        optim_state : dict
            The optim_state of the VBMC instance this function is
            called from.

        Returns
        -------
        acq : np.ndarray
            The output of the acquisition function, shape ``(N,)`` where ``N``
            is the number of input points.
        """
        if Xs.ndim == 1:
            Xs = Xs[None, :]

        # Map integer inputs
        Xs = self._real2int(
            Xs, vp.parameter_transformer, optim_state.get("integer_vars")
        )

        # Compute GP posterior predictive mean and variance

        # GP mean and variance for each hyperparameter sample
        f_mu, f_s2 = gp.predict(x_star=Xs, separate_samples=True)

        # Compute total variance
        Ns = f_mu.shape[1]
        f_bar = np.sum(f_mu, axis=1, keepdims=True) / Ns  # Mean across samples
        var_bar = (
            np.sum(f_s2, axis=1, keepdims=True) / Ns
        )  # Average variance across samples

        # Sample variance
        if Ns > 1:
            var_f = np.sum((f_mu - f_bar) ** 2, axis=1, keepdims=True) / (
                Ns - 1
            )
        else:
            var_f = 0

        f_bar = np.ravel(f_bar)
        var_tot = np.ravel(var_f + var_bar)  # Total variance

        # Compute acquisition function
        acq = self._compute_acquisition_function(
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
        )

        # Regularization: penalize points where GP uncertainty
        # is below threshold
        if optim_state.get("variance_regularized_acq_fcn"):
            # Try not to go below this variance
            tol_var = optim_state.get("tol_gp_var")
            idx_gp_uncertainty = var_tot < tol_var

            if np.any(idx_gp_uncertainty):
                if self.acq_info.get("log_flag"):
                    acq[idx_gp_uncertainty] += (
                        tol_var / var_tot[idx_gp_uncertainty] - 1
                    )
                else:
                    acq[idx_gp_uncertainty] *= np.exp(
                        -(tol_var / var_tot[idx_gp_uncertainty] - 1)
                    )

        realmax = sys.float_info.max
        acq = np.maximum(acq, -realmax)

        # Hard bound checking: discard points too close to bounds
        X_orig = vp.parameter_transformer.inverse(Xs)
        idx_bounds = np.logical_or(
            np.any(X_orig < optim_state.get("lb_eps_orig"), axis=1),
            np.any(X_orig > optim_state.get("ub_eps_orig"), axis=1),
        )
        acq[idx_bounds] = np.Inf

        # Re-shape to 1-D, if necessary (to avoid errors in cma.fmin)
        if acq.ndim > 1:
            if acq.shape[0] != acq.size and acq.shape[1] != acq.size:
                raise ValueError(
                    "Acquisition function should return a 1-D result (or a 2-D result which has size 1 along at least one axis)."
                )
            acq = acq.reshape(-1)
        return acq

    @abstractmethod
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
        Abstract method that must be implemented in each subclass. It computes
        the value of the acquisition function.
        """

    @staticmethod
    def _real2int(
        X: np.ndarray,
        parameter_transformer: ParameterTransformer,
        integer_vars: np.ndarray,
    ):
        """
        Convert to integer-valued representation.

        Parameters
        ----------
        X : np.ndarray
            The points to be converted.
        parameter_transformer : ParameterTransformer
            The appropriate ParameterTransformer to convert between the spaces.
        integer_vars : np.ndarray
            A mask to determine which dimensions are integer vars.
        """

        if np.any(integer_vars):
            X_temp = parameter_transformer.inverse(X)
            X_temp[:, integer_vars] = np.around(X_temp[:, integer_vars])
            X_temp = parameter_transformer(X_temp)
            X[:, integer_vars] = X_temp[:, integer_vars]

        return X

    @staticmethod
    def _sq_dist(a: np.array, b: np.array):
        """
        Compute matrix of all pairwise squared distances between two sets
        of vectors, stored in the columns of the two matrices `a` and `b`.

        Parameters
        ----------
        a : np.array, shape (n, D)
            First set of vectors.
        b : np.array, shape (m, D)
            Second set of vectors.

        Returns
        -------
        c: np.array, shape(n, m)
            The matrix of all pairwise squared distances.
        """
        n = a.shape[0]
        m = b.shape[0]
        mu = (m / (n + m)) * np.mean(b, axis=0) + (n / (n + m)) * np.mean(
            a, axis=0
        )
        a = a - mu
        b = b - mu
        c = np.sum(a * a, axis=1, keepdims=True) + (
            np.sum(b * b, axis=1, keepdims=True).T - (2 * a @ b.T)
        )
        return np.maximum(c, 0)

    def _estimate_observation_noise(
        self, Xs: np.ndarray, gp: gpr.GP, optim_state: dict
    ):
        """
        Estimate observation noise at test points from nearest neighbor.

        Parameters
        ----------
        Xs : np.ndarray
            The test points.
        gp : gpr.GP
            The GaussianProcess of the VBMC instance this function is
            called from.
        optim_state : dict
            The optim_state of the VBMC instance this function is
            called from.

        Returns
        -------
        sn2 : np.ndarray
            The estimated observation noise.
        """

        # unravel_index as the indicies are 1D otherwise
        pos = np.argmin(
            self._sq_dist(
                Xs / optim_state.get("gp_length_scale"),
                gp.temporary_data.get("X_rescaled"),
            ),
            axis=1,
        )
        sn2 = gp.temporary_data.get("sn2_new")[pos]

        return sn2
