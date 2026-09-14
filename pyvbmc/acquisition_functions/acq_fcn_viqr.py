import gpyreg as gpr
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import norm

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn

_LOG_2 = np.log(2.0)
_LOG_FLOAT_MAX = np.log(np.finfo(np.float64).max)
_VIQR_KERNEL_CACHE_MAX_BYTES = 128 * 1024**2

# Eligibility is checked against these import-time implementations rather
# than being inferred from class membership. This also detects methods
# replaced on a class or an individual instance after import, without adding
# anything to serialized acquisition or GP state.
_ORIGINAL_GP_PREDICT = gpr.GP.predict
_ORIGINAL_SE_COMPUTE = gpr.covariance_functions.SquaredExponential.compute


def _log_viqr_sum(a):
    r"""Compute ``log(2 * sum(sinh(a), axis=1))`` safely in float64.

    The direct expression is faster for ordinary VIQR arguments and avoids
    cancellation in ``1 - exp(-2a)`` for extremely small positive values.
    Its upper bound leaves a factor-of-two margin below the largest float64
    value after accounting for every term in the sum. Rows outside the safe
    finite, nonnegative range retain the original log-space calculation.
    """
    a = np.asarray(a, dtype=np.float64)
    n_terms = a.shape[1]
    # For a >= 0, 2 * n_terms * sinh(a) <= n_terms * exp(a). Subtracting
    # log(2) leaves a factor-of-two reserve for that complete quantity.
    direct_limit = _LOG_FLOAT_MAX - np.log(2 * n_terms)

    def direct(values):
        with np.errstate(divide="ignore"):
            return np.log(np.sum(np.sinh(values), axis=1)) + _LOG_2

    # Scalar extrema avoid per-row allocations when every row is safe.
    if a.size and np.min(a) >= 0.0 and np.max(a) <= direct_limit:
        return direct(a)

    row_min = np.min(a, axis=1)
    row_max = np.max(a, axis=1)
    safe = (
        np.isfinite(row_min)
        & np.isfinite(row_max)
        & (row_min >= 0.0)
        & (row_max <= direct_limit)
    )

    if np.all(safe):
        return direct(a)

    result = np.empty(a.shape[0], dtype=np.float64)
    if np.any(safe):
        result[safe] = direct(a[safe])

    # Original log-sinh followed by log-sum-exp. Besides avoiding overflow,
    # this preserves the former handling of unsupported arguments.
    fallback = a[~safe]
    zz = fallback + np.log1p(-np.exp(-2 * fallback))
    ln_max = np.amax(zz, axis=1)
    ln_max[ln_max == -np.inf] = 0.0  # Avoid -inf + inf
    result[~safe] = ln_max + np.log(
        np.sum(np.exp(zz - ln_max.reshape(-1, 1)), axis=1)
    )
    return result


class AcqFcnVIQR(AbstractAcqFcn):
    r"""
    Variational Interquantile Range (VIQR) acquisition function.

    Approximates the Integrated Median Interquantile Range (IMIQR) by simple
    Monte Carlo using samples from the Variational Posterior.

    Parameters
    ----------
    quantile : float, optional
        The upper quantile :math:`p_u` of the interquantile range; the
        default 0.75 gives :math:`u = \Phi^{-1}(0.75)`.
    loss : {"iqr", "iqr_reduction"}, optional
        The loss whose expectation under the variational posterior the
        acquisition minimizes. Both losses share the same look-ahead
        predictive standard deviation :math:`s_{\Xi \cup \theta_*}(\theta_a)`
        at the importance points :math:`\theta_a`: the GP posterior after a
        hypothetical observation at the candidate :math:`\theta_*`, which
        does not depend on the value observed there.

        - ``"iqr"`` (default): the log integrated interquantile range after
          the observation, :math:`\log \sum_a 2 \sinh(u\, s_{\Xi \cup
          \theta_*}(\theta_a))`, the VIQR of [1]_.
        - ``"iqr_reduction"``: minus the log of the integrated *reduction*
          of the interquantile range, :math:`-\log \sum_a w_a [\sinh(u\,
          s_\Xi(\theta_a)) - \sinh(u\, s_{\Xi \cup \theta_*}(\theta_a))]`.
          It has the same minimizer as ``"iqr"`` over any fixed candidate
          set. The log of the reduction spans orders of magnitude where the
          log of the residual range is nearly flat, which matters to the
          tolerance-based termination of the search optimizer.

        :math:`w_a` are the importance weights, normalized over the
        hyperparameter samples and the importance points together (so the
        reduction acquisition carries a constant :math:`\log N_s` that does
        not move the minimizer) and uniform under the simple Monte Carlo of
        VIQR. The reduction is non-negative, so ``"iqr_reduction"`` is
        ``+inf`` where the candidate reduces the interquantile range at no
        importance point.

    References
    ----------
    .. [1] Acerbi, L. (2020). "Variational Bayesian Monte Carlo with Noisy
       Likelihoods." Advances in Neural Information Processing Systems 33.
    """

    LOSSES = ("iqr", "iqr_reduction")

    def __init__(self, quantile=0.75, loss="iqr"):
        if loss not in self.LOSSES:
            raise ValueError(
                f"Unknown loss {loss!r}; expected one of {self.LOSSES}."
            )
        self.acq_info = {}
        self.acq_info["log_flag"] = True
        self.acq_info["importance_sampling"] = True
        self.acq_info["importance_sampling_vp"] = False
        self.acq_info["variational_importance_sampling"] = True
        self.acq_info["quantile"] = quantile
        self.acq_info["loss"] = loss
        self.loss = loss

        self.u = norm.ppf(quantile)

    def _predict_with_context(self, Xs: np.ndarray, gp: gpr.GP):
        """Request prediction kernels for compatible standard VIQR calls."""
        if self._can_reuse_prediction_kernel(Xs, gp):
            f_mu, f_s2, cross_covariance = gp.predict(
                x_star=Xs,
                separate_samples=True,
                return_cross_covariance=True,
            )
            return f_mu, f_s2, cross_covariance
        return super()._predict_with_context(Xs, gp)

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
        """Compute VIQR using prediction kernels when context provides them."""
        if context is None:
            return super()._compute_acquisition_function_with_context(
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
        return self._compute_viqr(
            Xs,
            gp,
            optim_state,
            f_mu,
            f_s2,
            cross_covariance=context,
        )

    def _can_reuse_prediction_kernel(self, Xs: np.ndarray, gp: gpr.GP):
        """Whether this call can safely retain predictor cross-kernels."""
        if (
            type(self) is not AcqFcnVIQR
            or getattr(self, "loss", "iqr") != "iqr"
        ):
            return False
        if (
            type(gp.covariance)
            is not gpr.covariance_functions.SquaredExponential
        ):
            return False
        if not _is_original_bound_method(gp, "predict", _ORIGINAL_GP_PREDICT):
            return False
        if not _is_original_bound_method(
            gp.covariance, "compute", _ORIGINAL_SE_COMPUTE
        ):
            return False
        if not _is_original_bound_method(
            self,
            "_compute_acquisition_function",
            _ORIGINAL_VIQR_COMPUTE_ACQUISITION,
        ):
            return False

        # Untrained and empty states retain the established prediction path.
        if (
            getattr(gp, "y", None) is None
            or getattr(gp, "X", None) is None
            or gp.X.shape[0] == 0
            or Xs.shape[0] == 0
            or len(gp.posteriors) == 0
        ):
            return False

        # Start with Python integers so unusually large shapes cannot overflow
        # before comparison with the payload cap. Prediction and PyVBMC state
        # are float64 on this supported path.
        payload_bytes = (
            8 * int(gp.X.shape[0]) * int(Xs.shape[0]) * int(len(gp.posteriors))
        )
        return payload_bytes <= _VIQR_KERNEL_CACHE_MAX_BYTES

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
            A ``(N, Ns_gp)`` array of GP predictive means at the candidate
            points, where ``Ns_gp`` is the number of GP posterior
            hyperparameter samples.
        f_s2 : np.ndarray
            A ``(N, Ns_gp)`` array of GP predictive variances at the candidate
            points, where ``Ns_gp`` is the number of GP posterior
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
            Also when the instance carries a ``loss`` outside ``LOSSES``.
        """
        return self._compute_viqr(Xs, gp, optim_state, f_mu, f_s2)

    def _compute_viqr(
        self,
        Xs: np.ndarray,
        gp: gpr.GP,
        optim_state: dict,
        f_mu: np.ndarray,
        f_s2: np.ndarray,
        cross_covariance=None,
    ):
        """Shared VIQR arithmetic with optional predictor cross-kernels.

        ``cross_covariance`` is a per-hyperparameter-sample tuple of latent
        kernel matrices with shape ``(N_training, N_candidates)``. It is
        consumed through transpose views during this call only.
        """
        # Missing port, integrated mean function, lines 49 to 57.

        # Xs is in *transformed* coordinates
        [Nx, D] = Xs.shape
        Ns_gp = f_mu.shape[1]

        # Estimate observation noise at test points from nearest neighbor.
        sn2 = super()._estimate_observation_noise(Xs, gp, optim_state)
        y_s2 = f_s2 + sn2.reshape(-1, 1)  # Predictive variance at test points,
        # inclusive of observation noise.

        active_is = optim_state["active_importance_sampling"]
        Xa = active_is["X"]
        # Instances pickled before the ``loss`` argument existed are VIQR;
        # one carrying a loss this version does not implement is rejected
        # rather than computed as another.
        loss = getattr(self, "loss", "iqr")
        if loss not in self.LOSSES:
            raise ValueError(
                f"Unknown loss {loss!r}; expected one of {self.LOSSES}."
            )
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

                if cross_covariance is None:
                    tmp = cdist(Xs_ell, gp.X / ell, "sqeuclidean")
                    K_Xs_X = sf2 * np.exp(-tmp / 2)
                else:
                    K_Xs_X = cross_covariance[s].T

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

            if loss != "iqr":
                acq[:, s] = -self._log_iqr_reduction(
                    tau2,
                    active_is["f_s2"][:, s],
                    active_is["ln_weights"][s, :],
                )
                continue

            s_pred = np.sqrt(
                np.maximum(
                    optim_state["active_importance_sampling"]["f_s2"][:, s].T
                    - tau2,
                    0.0,
                )
            )

            # VIQR uses simple Monte Carlo, so the weights are constant.
            acq[:, s] = _log_viqr_sum(self.u * s_pred)

        if Ns_gp > 1:
            if loss != "iqr":
                # Average the reductions over the hyperparameter samples.
                acq = -(logsumexp(-acq, axis=1) - np.log(Ns_gp))
            else:
                M = np.amax(acq, axis=1)
                M[M == -np.inf] = 0.0  # Avoid -inf + inf
                acq = M + np.log(
                    np.sum(np.exp(acq - M.reshape(-1, 1)), axis=1) / Ns_gp
                )

        return acq

    def _log_iqr_reduction(self, tau2, f_s2_a, ln_w):
        r"""Log of the importance-weighted integrated reduction of the
        interquantile range.

        Parameters
        ----------
        tau2 : np.ndarray
            The look-ahead reduction of the GP posterior variance at the
            importance points, shape ``(Nx, Na)`` for ``Nx`` candidates.
        f_s2_a : np.ndarray
            The current GP posterior variance at the importance points,
            shape ``(Na,)``.
        ln_w : np.ndarray
            The normalized log importance weights, shape ``(Na,)``.

        Returns
        -------
        ln_r : np.ndarray
            The log of the weighted reduction per candidate, shape
            ``(Nx,)``; ``-inf`` where the candidate reduces the range at no
            importance point.
        """
        s_a = np.sqrt(f_s2_a)  # (Na,)
        s_pred = np.sqrt(np.maximum(f_s2_a - tau2, 0.0))  # (Nx, Na)
        # s_a - s_pred = tau2 / (s_a + s_pred): no cancellation when the
        # reduction is a small fraction of the variance, the common case
        # with observation noise.
        denom = s_a + s_pred
        d = np.divide(tau2, denom, out=np.zeros_like(tau2), where=denom > 0)
        # sinh(u s_a) - sinh(u s_pred) = 2 cosh(A) sinh(B) with
        # A = u (s_a + s_pred) / 2 and B = u (s_a - s_pred) / 2, in log
        # space (no overflow for large s):
        # log 2 + logcosh(A) + logsinh(B)
        #   = A + B + log1p(exp(-2A)) + log(-expm1(-2B)) - log 2.
        A = 0.5 * self.u * denom
        B = 0.5 * self.u * d
        with np.errstate(divide="ignore"):
            ln_term = (
                A
                + B
                + np.log1p(np.exp(-2 * A))
                + np.log(-np.expm1(-2 * B))
                - np.log(2)
            )
        return logsumexp(ln_term + ln_w, axis=1)

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


def _is_original_bound_method(instance, name, original):
    """Return whether ``instance.name`` resolves to ``original``."""
    method = getattr(instance, name)
    return (
        getattr(method, "__self__", None) is instance
        and getattr(method, "__func__", None) is original
    )


_ORIGINAL_VIQR_COMPUTE_ACQUISITION = AcqFcnVIQR._compute_acquisition_function
