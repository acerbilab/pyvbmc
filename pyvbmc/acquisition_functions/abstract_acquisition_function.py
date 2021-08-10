import sys
from abc import ABC, abstractmethod

import gpyreg as gpr
import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior


class AbstractAcquisitionFunction(ABC):
    def __init__(self):
        self.acq_info = dict()

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
        optim_state : dict
            The optim_steate of the VBMC instance this function is
            called from.

        Returns
        -------
        acq : np.ndarray
            The output of the acquisition function.
        """

        # Map integer inputs
        Xs = self._real2int(
            Xs, vp.parameter_transformer, optim_state.get("integervars")
        )

        # Compute GP posterior predictive mean and variance

        if (
            hasattr(vp, "delta")
            and vp.delta is not None
            and np.any(vp.delta > 0)
        ):
            # Quadrature mean and variance for each hyperparameter sample
            f_mu, f_s2 = gp.quad(
                mu=Xs,
                sigma=vp.delta.T,
                compute_var=True,
                separate_samples=True,
            )
        else:
            # GP mean and variance for each hyperparameter sample
            f_mu, f_s2 = gp.predict(x_star=Xs, separate_samples=True)

        # Compute total variance
        Ns = f_mu.shape[1]
        f_bar = np.sum(f_mu, axis=1, keepdims=True) / Ns  # Mean across samples
        var_bar = np.sum(f_s2, axis=1) / Ns  # Average variance across samples

        # Sample variance
        if Ns > 1:
            var_f = np.sum((f_mu - f_bar) ** 2, axis=1) / (Ns - 1)
        else:
            var_f = 0

        var_tot = var_f + var_bar  # Total variance

        # Compute acquisition function
        acq = self._compute_acquisition_function(
            Xs, vp, gp, function_logger, optim_state, f_mu, f_s2, f_bar, var_tot
        )

        # Regularization: penalize points where GP uncertainty
        # is below threshold
        if optim_state.get("variance_regularized_acq_fcn"):
            # Try not to go below this variance
            tol_var = optim_state.get("tol_gp_var")
            idx_gp_uncertainty = var_tot < tol_var

            if np.any(idx_gp_uncertainty):
                if "log_flag" in self.acq_info and self.acq_info.get(
                    "log_flag"
                ):
                    acq[idx_gp_uncertainty] += (
                        tol_var / var_tot[idx_gp_uncertainty] - 1
                    )
                else:
                    acq[idx_gp_uncertainty] *= np.exp(
                        -(tol_var / var_tot[idx_gp_uncertainty] - 1)
                    )

        realmin = sys.float_info.min
        acq = np.maximum(acq, realmin)

        # Hard bound checking: discard points too close to bounds
        X_orig = vp.parameter_transformer.inverse(Xs)
        idx_bounds = np.logical_or(
            np.any(X_orig < optim_state.get("lb_eps_orig"), axis=1),
            np.any(X_orig > optim_state.get("ub_eps_orig"), axis=1),
        )
        acq[idx_bounds] = np.Inf

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
        integervars: np.ndarray,
    ):
        """
        Convert to integer-valued representation.

        Parameters
        ----------
        X : np.ndarray
            The points to be converted.
        parameter_transformer : ParameterTransformer
            The appropriate ParameterTransformer to convert between the spaces.
        integervars : np.ndarray
            A mask to determine which dimensions are integer vars.
        """

        if np.any(integervars):
            X_temp = parameter_transformer.inverse(X)
            X_temp[:, integervars] = np.around(X_temp[:, integervars])
            X_temp = parameter_transformer(X_temp)
            X[:, integervars] = X_temp[:, integervars]

        return X
