import gpyreg as gpr
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import norm

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


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
    loss : {"iqr", "iqr_reduction", "var_reduction", "sd_reduction"}, optional
        The loss whose expectation under the variational posterior the
        acquisition minimizes. All members share the same look-ahead
        predictive standard deviation :math:`s_{\Xi \cup \theta_*}(\theta_a)`
        at the importance points :math:`\theta_a`: the GP posterior after a
        hypothetical observation at the candidate :math:`\theta_*`, which
        does not depend on the value observed there.

        - ``"iqr"`` (default): the log integrated interquantile range after
          the observation, :math:`\log \sum_a \sinh(u\, s_{\Xi \cup
          \theta_*}(\theta_a))`, the VIQR of [1]_.
        - ``"iqr_reduction"``: minus the log of the integrated *reduction*
          of the interquantile range, :math:`-\log \sum_a w_a [\sinh(u\,
          s_\Xi(\theta_a)) - \sinh(u\, s_{\Xi \cup \theta_*}(\theta_a))]`.
          It has the same minimizer as ``"iqr"`` over any fixed candidate
          set. The log of the reduction spans orders of magnitude where the
          log of the residual range is nearly flat, which matters to the
          tolerance-based termination of the search optimizer.
        - ``"var_reduction"``: minus the log of the integrated reduction of
          the GP posterior variance, :math:`-\log \sum_a w_a [s^2_\Xi
          (\theta_a) - s^2_{\Xi \cup \theta_*}(\theta_a)]`. The cheapest
          member: one matrix-vector product per hyperparameter sample, no
          transcendental sweep over the importance points.
        - ``"sd_reduction"``: minus the log of the integrated reduction of
          the GP posterior standard deviation, :math:`-\log \sum_a w_a
          [s_\Xi(\theta_a) - s_{\Xi \cup \theta_*}(\theta_a)]`.

        :math:`w_a` are the normalized importance weights, uniform under
        the simple Monte Carlo of VIQR. The reductions are non-negative, so
        the reduction acquisitions are ``+inf`` where the candidate reduces
        the loss at no importance point.

    References
    ----------
    .. [1] Acerbi, L. (2020). "Variational Bayesian Monte Carlo with Noisy
       Likelihoods." Advances in Neural Information Processing Systems 33.
    """

    LOSSES = ("iqr", "iqr_reduction", "var_reduction", "sd_reduction")

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

        active_is = optim_state["active_importance_sampling"]
        Xa = active_is["X"]
        # Instances pickled before the ``loss`` argument existed are VIQR.
        loss = getattr(self, "loss", "iqr")
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

            if loss != "iqr":
                acq[:, s] = -self._log_reduction(
                    loss,
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

    def _log_reduction(self, loss, tau2, f_s2_a, ln_w):
        r"""Log of the importance-weighted integrated reduction of a loss.

        Parameters
        ----------
        loss : str
            One of the reduction members of `LOSSES`.
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
            ``(Nx,)``; ``-inf`` where the candidate reduces the loss at no
            importance point.
        """
        if loss == "var_reduction":
            with np.errstate(divide="ignore"):
                return np.log(tau2 @ np.exp(ln_w))

        s_a = np.sqrt(f_s2_a)  # (Na,)
        s_pred = np.sqrt(np.maximum(f_s2_a - tau2, 0.0))  # (Nx, Na)
        # s_a - s_pred = tau2 / (s_a + s_pred): no cancellation when the
        # reduction is a small fraction of the variance, the common case
        # with observation noise.
        denom = s_a + s_pred
        d = np.divide(tau2, denom, out=np.zeros_like(tau2), where=denom > 0)
        if loss == "sd_reduction":
            with np.errstate(divide="ignore"):
                ln_term = np.log(d)
        else:  # "iqr_reduction"
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
