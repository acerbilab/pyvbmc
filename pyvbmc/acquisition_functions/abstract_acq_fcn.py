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

    Notes
    -----
    A subclass that sets ``acq_info["compute_var_log_joint"] = True`` is
    given the posterior uncertainty of the expected log joint before each
    search: ``active_sample`` then writes its variance per GP
    hyperparameter sample to ``optim_state["var_log_joint_samples"]``,
    shape ``(Ns,)`` and a scalar when ``Ns == 1``, and the covariance of
    the mixture components' integrals to
    ``optim_state["cov_log_joint_components"]``, shape ``(Ns, K, K)``.
    The built-in acquisitions leave the flag disabled and neither entry
    exists for them.
    """

    def __init__(self):
        self.acq_info = {}
        # Custom-acquisition hook; built-in acquisitions leave this disabled.
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

        # Compute GP posterior predictive mean and variance, optionally with
        # call-local information used by the acquisition implementation.
        f_mu, f_s2, context = self._predict_with_context(Xs, gp)

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

        # Compute the acquisition function. The context is deliberately
        # local to this call; in particular, it is not retained on any of
        # the participating objects when evaluation succeeds or raises.
        try:
            acq = self._compute_acquisition_function_with_context(
                Xs,
                vp,
                gp,
                function_logger,
                optim_state,
                f_mu,
                f_s2,
                f_bar,
                var_tot,
                context,
            )
        finally:
            context = None

        # Normalize documented vector outputs before applying pointwise masks.
        acq = np.asarray(acq)
        M = Xs.shape[0]
        if acq.ndim == 1:
            valid_shape = acq.shape == (M,)
        elif acq.ndim == 2:
            valid_shape = acq.shape in ((1, M), (M, 1))
        else:
            valid_shape = False
        if not valid_shape:
            raise ValueError(
                "Acquisition function should return a vector with one value "
                "per input point, with shape (M,), (1, M), or (M, 1)."
            )
        acq = acq.reshape(-1)

        # Regularization: penalize points where GP uncertainty
        # is below threshold
        if "variance_regularized_acq_fcn" in optim_state:
            variance_regularized = optim_state["variance_regularized_acq_fcn"]
        else:
            variance_regularized = optim_state.get(
                "variance_regularized_acqfcn", False
            )
        if variance_regularized:
            # Try not to go below this variance
            tol_var = optim_state.get("tol_gp_var")
            idx_gp_uncertainty = var_tot < tol_var

            if np.any(idx_gp_uncertainty):
                idx_zero_variance = np.logical_and(
                    idx_gp_uncertainty, var_tot == 0
                )
                idx_positive_variance = np.logical_and(
                    idx_gp_uncertainty, var_tot > 0
                )
                if self.acq_info.get("log_flag"):
                    acq[idx_positive_variance] += (
                        tol_var / var_tot[idx_positive_variance] - 1
                    )
                    acq[idx_zero_variance] = np.inf
                else:
                    acq[idx_positive_variance] *= np.exp(
                        -(tol_var / var_tot[idx_positive_variance] - 1)
                    )
                    acq[idx_zero_variance] = 0.0

        realmax = sys.float_info.max
        acq = np.maximum(acq, -realmax)

        # Hard bound checking: discard points too close to bounds
        X_orig = vp.parameter_transformer.inverse(Xs)
        idx_bounds = np.logical_or(
            np.any(X_orig < optim_state.get("lb_eps_orig"), axis=1),
            np.any(X_orig > optim_state.get("ub_eps_orig"), axis=1),
        )
        acq[idx_bounds] = np.inf

        return acq

    def _predict_with_context(self, Xs: np.ndarray, gp: gpr.GP):
        """Predict at candidates and optionally return call-local context.

        Subclasses may override this hook to request information produced
        during GP prediction for use in acquisition evaluation. The default
        preserves the ordinary prediction call and supplies no context.

        Returns
        -------
        f_mu, f_s2 : np.ndarray
            Per-hyperparameter-sample GP predictive means and variances.
        context : object or None
            Information passed to
            :meth:`_compute_acquisition_function_with_context` for this call
            only.
        """
        f_mu, f_s2 = gp.predict(x_star=Xs, separate_samples=True)
        return f_mu, f_s2, None

    def _compute_acquisition_function_with_context(
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
        context,
    ):
        """Compute an acquisition with optional call-local prediction data.

        The default delegates to :meth:`_compute_acquisition_function` with
        its established signature. Existing subclasses therefore require no
        changes. A subclass that consumes context should override this hook
        while keeping its original acquisition method available as the
        context-free implementation.
        """
        return self._compute_acquisition_function(
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
    def _check_quantile(quantile):
        """
        The upper quantile of an interquantile range, as a float.

        Parameters
        ----------
        quantile : float
            A real number strictly between 0.5 and 1: a Python or NumPy
            scalar, or an array with one element.

        Returns
        -------
        quantile : float
            The quantile as a Python float.

        Raises
        ------
        ValueError
            If ``quantile`` is not one real number strictly between 0.5
            and 1.
        """
        value = np.asarray(quantile)
        if value.size == 1 and value.dtype.kind in "fiu":
            value = float(value.reshape(-1)[0])
            if 0.5 < value < 1:
                return value
        raise ValueError(
            "The quantile must be a number strictly between 0.5 and "
            f"1, not {quantile!r}."
        )

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
            The points to be converted, either a single point of shape
            ``(D,)`` or an array of points of shape ``(n, D)``. The
            integer-valued coordinates are snapped in place.
        parameter_transformer : ParameterTransformer
            The appropriate ParameterTransformer to convert between the spaces.
        integer_vars : np.ndarray
            A mask to determine which dimensions are integer vars.

        Returns
        -------
        X : np.ndarray
            The converted points, in the shape they were given in.
        """

        if np.any(integer_vars):
            # A single point is snapped through a two-dimensional view of
            # it, so that the caller keeps the shape it passed and the
            # coordinates are written back in either shape.
            X_2d = X[None, :] if X.ndim == 1 else X
            X_temp = parameter_transformer.inverse(X_2d)
            # MATLAB's `round` (misc/real2int_vbmc.m:7) sends a half away
            # from zero; `np.around` sends it to the nearer even integer.
            # The fractional part is exact, where `abs(x) + 0.5` rounds
            # the largest number below a half up to one.
            X_int = X_temp[:, integer_vars]
            X_whole = np.trunc(X_int)
            X_temp[:, integer_vars] = X_whole + np.sign(X_int) * (
                np.abs(X_int - X_whole) >= 0.5
            )
            X_temp = parameter_transformer(X_temp)
            X_2d[:, integer_vars] = X_temp[:, integer_vars]

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
