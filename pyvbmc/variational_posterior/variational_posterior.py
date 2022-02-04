# for annotating VP as input of itself in mtv
from __future__ import annotations

import sys

import corner
import matplotlib.pyplot as plt
import numpy as np
from pyvbmc.decorators import handle_0D_1D_input
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import kde1d, kldiv_mvn
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import gammaln
import gpyreg
from pyvbmc.rotoscaling.unscent_warp import unscent_warp


class VariationalPosterior:
    """
    The variational posterior class used in ``pyvbmc``.

    The variational posterior represents the approximate posterior as returned 
    by the Variational Bayesian Monte Carlo (VBMC) algorithm.
    
    In VBMC, the variational posterior is a weighted mixture of multivariate 
    normal distributions (see Notes below for details).

    Parameters
    ----------
    D : int
        The number of dimensions (i.e., parameters) of the posterior.
    K : int, optional
        The number of mixture components, default 2.
    x0 : np.ndarray, optional
        The starting vector for the mixture components means. It can be a
        single array or multiple rows (up to `K`); missing rows are
        duplicated by making copies of `x0`, default ``np.zeros``.
    parameter_transformer : ParameterTransformer, optional
        The ``ParameterTransformer`` object specifying the transformation of 
        the input space that leads to the current representation used by the
        variational posterior, by default uses an identity transform.
    
    Notes
    -----
    In VBMC, the variational posterior is defined as a mixture of multivariate 
    normal distributions as follows:

    .. math:: q({\\theta}) = \sum_{k = 1}^K w_k N(\\theta, \mu_k, \sigma^2_k \Lambda)

    where :math:`w_k` are the mixture weights, :math:`\mu_k` the component means, 
    :math:`\sigma_k` scaling factors, and :math:`\Lambda` is a diagonal matrix 
    common to all components with elements :math:`\lambda^2_d` on the diagonal,
    for :math:`1 \le d \le D`.

    Note that :math:`q({\\theta})` is defined in an unconstrained space. 
    Constrained variables in the posterior are mapped to a trasformed, 
    unconstrained space via a nonlinear mapping (represented by a 
    ``ParameterTransformer`` object). The transformation is handled 
    automatically.

    In practice, you would almost never create a new ``VariationalPosterior`` 
    object, simply use the variational posteriors returned by ``VBMC``.

    """

    def __init__(
        self, D: int, K: int = 2, x0=None, parameter_transformer=None
    ):
        self.D = D  # number of dimensions
        self.K = K  # number of components

        if x0 is None:
            x0 = np.zeros((D, K))
        elif x0.size == D:
            x0.reshape(-1, 1)  # reshape to vertical array
            x0 = np.tile(x0, (K, 1)).T  # copy vector
        else:
            x0 = x0.T
            x0 = np.tile(x0, int(np.ceil(self.K / x0.shape[1])))
            x0 = x0[:, 0 : self.K]

        self.w = np.ones((1, K)) / K
        self.eta = np.ones((1, K)) / K
        self.mu = x0 + 1e-6 * np.random.randn(self.D, self.K)
        self.sigma = 1e-3 * np.ones((1, K))
        self.lambd = np.ones((self.D, 1))

        # By default, optimize all variational parameters
        self.optimize_weights = True
        self.optimize_mu = True
        self.optimize_sigma = True
        self.optimize_lambd = True

        if parameter_transformer is None:
            self.parameter_transformer = ParameterTransformer(self.D)
        else:
            self.parameter_transformer = parameter_transformer

        self.delta = None
        self.bounds = None
        self.stats = None
        self._mode = None

    def get_bounds(self, X: np.ndarray, options, K: int = None):
        """
        Compute soft bounds for variational posterior parameters.

        These bounds are used during the variational optimization in ``VBMC``.

        Parameters
        ----------
        X : ndarray, shape (N, D)
            Training inputs.
        options : Options
            Program options.
        K : int, optional
            The number of mixture components. By default we use the
            number provided at class instantiation.

        Returns
        -------
        theta_bnd : dict
            A dictionary of soft bounds with the following elements:
                **lb** : np.ndarray
                    Lower bounds.
                **ub** : np.ndarray
                    Upper bounds.
                **tol_con** : float
                    Fractional tolerance for constraint violation of
                    variational parameters.
                **weight_threshold** : float, optional
                     Threshold below which weights are penalized.
                **weight_penalty** : float, optional
                     The penalty for weight below the threshold.
        """
        if K is None:
            K = self.K

        # Soft-bound loss is computed on MU and SCALE (which is SIGMA times
        # LAMBDA)

        # Start with reversed bounds (see below)
        if self.bounds is None:
            self.bounds = {
                "mu_lb": np.full((self.D,), np.inf),
                "mu_ub": np.full((self.D,), -np.inf),
                "lnscale_lb": np.full((self.D,), np.inf),
                "lnscale_ub": np.full((self.D,), -np.inf),
            }

        # Set bounds for mean parameters of variational components
        self.bounds["mu_lb"] = np.minimum(
            np.min(X, axis=0), self.bounds["mu_lb"]
        )
        self.bounds["mu_ub"] = np.maximum(
            np.max(X, axis=0), self.bounds["mu_ub"]
        )

        # Set bounds for log scale parameters of variational components.
        ln_range = np.log(np.max(X, axis=0) - np.min(X, axis=0))
        self.bounds["lnscale_lb"] = np.minimum(
            self.bounds["lnscale_lb"], ln_range + np.log(options["tollength"])
        )
        self.bounds["lnscale_ub"] = np.maximum(
            self.bounds["lnscale_ub"], ln_range
        )

        # Set bounds for log weight parameters of variation components.
        if self.optimize_weights:
            # prevent warning to be printed when doing final boost
            if options["tolweight"] == 0:
                self.bounds["eta_lb"] = -np.inf
            else:
                self.bounds["eta_lb"] = np.log(0.5 * options["tolweight"])
            self.bounds["eta_ub"] = 0

        lb_list = []
        ub_list = []
        if self.optimize_mu:
            lb_list.append(np.tile(self.bounds["mu_lb"], (K,)))
            ub_list.append(np.tile(self.bounds["mu_ub"], (K,)))
        if self.optimize_sigma or self.optimize_lambd:
            lb_list.append(np.tile(self.bounds["lnscale_lb"], (K,)))
            ub_list.append(np.tile(self.bounds["lnscale_ub"], (K,)))
        if self.optimize_weights:
            lb_list.append(np.tile(self.bounds["eta_lb"], (K,)))
            ub_list.append(np.tile(self.bounds["eta_ub"], (K,)))

        theta_bnd = {
            "lb": np.concatenate(lb_list),
            "ub": np.concatenate(ub_list),
        }

        theta_bnd["tol_con"] = options["tolconloss"]

        # Weight below a certain threshold are penalized.
        if self.optimize_weights:
            theta_bnd["weight_threshold"] = max(
                1 / (4 * K), options["tolweight"]
            )
            theta_bnd["weight_penalty"] = options["weightpenalty"]

        return theta_bnd

    def sample(
        self,
        N: int,
        origflag: bool = True,
        balanceflag: bool = False,
        df: float = np.inf,
    ):
        """
        Draw random samples from the variational posterior.

        Parameters
        ----------
        N : int
            Number of samples to draw.
        origflag : bool, optional
            If `origflag` is ``True``, the random vectors are returned 
            in the original parameter space. If ``False``, they are returned in 
            the transformed, unconstrained space used internally by VBMC. 
            By default ``True``.
        balanceflag : bool, optional
            If `balanceflag` is ``True``, the generating process is balanced 
            such that the random samples come from each mixture component 
            exactly proportionally (or as close as possible) to the variational 
            mixture weights. If ``False``, the generating mixture component for 
            each sample is determined randomly, according to the mixture weights.
            By default ``False``.
        df : float, optional
            Generate the samples from a heavy-tailed version of the variational 
            posterior, in which the multivariate normal components have been 
            replaced by multivariate `t`-distributions with `df` degrees of 
            freedom. The default is ``np.inf``, limit in which the 
            `t`-distribution becomes a multivariate normal.

        Returns
        -------
        X : np.ndarray
            `X` is an `N`-by-`D` matrix of random vectors drawn from the 
            variational posterior.
        I : np.ndarray
            `I` is an `N`-by-1 array such that the `i`-th element of `I` 
            indicates the index of the variational mixture component from which
            the `i`-th row of X has been generated.
        """
        # missing to sample from gp
        gp_sample = False
        if N < 1:
            x = np.zeros((0, self.D))
            i = np.zeros((0, 1))
            return x, i
        elif gp_sample:
            pass
        else:
            lambd_row = self.lambd.reshape(1, -1)

            if self.K > 1:
                if balanceflag:
                    # exact split of samples according to mixture weights
                    repeats = np.floor(self.w * N).astype("int")
                    i = np.repeat(range(self.K), repeats.flatten())

                    # compute remainder samples (with correct weights) if needed
                    if N > i.shape[0]:
                        w_extra = self.w * N - repeats
                        repeats_extra = np.ceil(np.sum(w_extra))
                        w_extra += self.w * (repeats_extra - sum(w_extra))
                        w_extra /= np.sum(w_extra)
                        i_extra = np.random.choice(
                            range(self.K),
                            size=repeats_extra.astype("int"),
                            p=w_extra.flatten(),
                        )
                        i = np.append(i, i_extra)

                    np.random.shuffle(i)
                    i = i[:N]
                else:
                    i = np.random.choice(
                        range(self.K), size=N, p=self.w.flatten()
                    )

                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.T[i]
                        + lambd_row
                        * np.random.randn(N, self.D)
                        * self.sigma[:, i].T
                    )
                else:
                    t = (
                        df
                        / 2
                        / np.sqrt(np.random.gamma(df / 2, df / 2, (N, 1)))
                    )
                    x = (
                        self.mu.T[i]
                        + lambd_row
                        * np.random.randn(N, self.D)
                        * t
                        * self.sigma[:, i].T
                    )
            else:
                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.T
                        + lambd_row * np.random.randn(N, self.D) * self.sigma
                    )
                else:
                    t = (
                        df
                        / 2
                        / np.sqrt(np.random.gamma(df / 2, df / 2, (N, 1)))
                    )
                    x = (
                        self.mu.T
                        + lambd_row
                        * t
                        * np.random.randn(N, self.D)
                        * self.sigma
                    )
                i = np.zeros(N)
            if origflag:
                x = self.parameter_transformer.inverse(x)
        return x, i

    @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
    def pdf(
        self,
        x: np.ndarray,
        origflag: bool = True,
        logflag: bool = False,
        transflag: bool = False,
        gradflag: bool = False,
        df: float = np.inf,
    ):
        """
        Probability density function of the variational posterior.

        Compute the probability density function (pdf) of the variational 
        posterior at one or multiple input points.

        Parameters
        ----------
        x : np.ndarray
            `x` is a matrix of inputs to evaluate the pdf at.
            The rows of the `N`-by-`D` matrix `x` correspond to observations or 
            points, and columns correspond to variables or coordinates.
        origflag : bool, optional
            Controls if the value of the posterior density should be evaluated
            in the original parameter space for `origflag` is ``True``, or in the 
            transformed space if `origflag` is ``False``, by default ``True``.
        logflag : bool, optional
            If `logflag` is ``True`` return the logarithm of the pdf, 
            by default ``False``.
        transflag : bool, optional
            Specifies if `x` is already specified in transformed space.
            `transflag` = ``True`` assumes that `x` is already specified in 
            tranformed space. Otherwise, `x` is specified in the original 
            parameter space. By default ``False``.
        gradflag : bool, optional
            If ``True`` the gradient of the pdf is returned as a second output, 
            by default ``False``.
        df : float, optional
            Compute the pdf of a heavy-tailed version of the variational 
            posterior, in which the multivariate normal components
            have been replaced by multivariate `t`-distributions with
            `df` degrees of freedom. The default is `df` = ``np.inf``, limit in 
            which the `t`-distribution becomes a multivariate normal.
        
        Returns
        -------
        pdf: np.ndarray
            The probability density of the variational posterior
            evaluated at each row of `x`.
        gradient: np.ndarray
            If `gradflag` is ``True``, the function returns the gradient as well.

        Raises
        ------
        NotImplementedError
            Raised if `df` is non-zero and finite and `gradflag` = ``True``
            (Gradient of heavy-tailed pdf not supported yet).
        NotImplementedError
            Raised if `origflag` = ``True`` and `logflag` = ``True`` and 
            `gradflag` = ``True`` (Gradient computation in original space not 
            supported yet).
        """
        N, D = x.shape

        # compute pdf only for points inside bounds in origspace
        if origflag:
            mask = np.logical_and(
                np.all(x > self.parameter_transformer.lb_orig, axis=1),
                np.all(x < self.parameter_transformer.ub_orig, axis=1),
            )
        else:
            mask = np.full(N, True)

        # Convert points to transformed space
        if origflag and not transflag:
            x[mask] = self.parameter_transformer(x[mask])
        lamd_row = self.lambd.reshape(1, -1)

        y = np.zeros((N, 1))
        if gradflag:
            dy = np.zeros((N, D))

        if not np.isfinite(df) or df == 0:
            # compute pdf of variational posterior

            # common normalization factor
            nf = 1 / (2 * np.pi) ** (D / 2) / np.prod(lamd_row)
            for k in range(self.K):
                d2 = np.sum(
                    ((x - self.mu.T[k]) / (self.sigma[:, k].dot(lamd_row)))
                    ** 2,
                    axis=1,
                )
                nn = (
                    nf
                    * self.w[:, k]
                    / self.sigma[:, k] ** D
                    * np.exp(-0.5 * d2)[:, np.newaxis]
                )
                y += nn
                if gradflag:
                    dy -= (
                        nn
                        * (x - self.mu.T[k])
                        / ((lamd_row ** 2) * self.sigma[:, k] ** 2)
                    )

        else:
            # Compute pdf of heavy-tailed variant of variational posterior

            if df > 0:
                # (This uses a multivariate t-distribution which is not the same
                # thing as the product of D univariate t-distributions)

                # common normalization factor
                nf = (
                    np.exp(gammaln((df + D) / 2) - gammaln(df / 2))
                    / (df * np.pi) ** (D / 2)
                    / np.prod(self.lambd)
                )

                for k in range(self.K):
                    d2 = np.sum(
                        ((x - self.mu.T[k]) / (self.sigma[:, k].dot(lamd_row)))
                        ** 2,
                        axis=1,
                    )
                    nn = (
                        nf
                        * self.w[:, k]
                        / self.sigma[:, k] ** D
                        * (1 + d2 / df) ** (-(df + D) / 2)
                    )[:, np.newaxis]
                    y += nn
                    if gradflag:
                        raise NotImplementedError(
                            "Gradient of heavy-tailed pdf not supported yet."
                        )
            else:
                # (This uses a product of D univariate t-distributions)

                df_abs = abs(df)

                # Common normalization factor
                nf = (
                    np.exp(gammaln((df_abs + 1) / 2) - gammaln(df_abs / 2))
                    / np.sqrt(df_abs * np.pi)
                ) ** D / np.prod(self.lambd)

                for k in range(self.K):
                    d2 = (
                        (x - self.mu.T[k]) / (self.sigma[:, k].dot(lamd_row))
                    ) ** 2
                    nn = (
                        nf
                        * self.w[:, k]
                        / self.sigma[:, k] ** D
                        * np.prod(
                            (1 + d2 / df_abs) ** (-(df_abs + 1) / 2), axis=1
                        )[:, np.newaxis]
                    )
                    y += nn
                    if gradflag:
                        raise NotImplementedError(
                            "Gradient of heavy-tailed pdf not supported yet."
                        )

        if logflag:
            if gradflag:
                dy = dy / y
            y = np.log(y)

        # apply jacobian correction
        if origflag:
            if logflag:
                y[mask] -= self.parameter_transformer.log_abs_det_jacobian(
                    x[mask]
                )[:, np.newaxis]
                if gradflag:
                    raise NotImplementedError(
                        """vbmc_pdf:NoOriginalGrad: Gradient computation
                         in original space not supported yet."""
                    )
            else:
                y[mask] /= np.exp(
                    self.parameter_transformer.log_abs_det_jacobian(x[mask])[
                        :, np.newaxis
                    ]
                )

        # pdf is 0 outside the bounds in origspace
        y[~mask] = 0

        if gradflag:
            return y, dy
        else:
            return y

    def get_parameters(self, rawflag=True):
        """
        Get variational posterior parameters as single array.

        Return all the active ``VariationalPosterior`` parameters
        flattened as a 1D (numpy) array, possibly transformed.

        Parameters
        ----------
        rawflag : bool, optional
            Specifies whether the sigma and lambda parameters are
            returned as raw (unconstrained) or not, by default ``True``.

        Returns
        -------
        theta : np.ndarray
            The variational posterior parameters flattenend as a 1D array.
        """

        nl = np.sqrt(np.sum(self.lambd ** 2) / self.D)

        self.lambd = self.lambd.reshape(-1, 1) / nl
        self.sigma = self.sigma.reshape(1, -1) * nl

        # Ensure that weights are normalized
        if self.optimize_weights:
            self.w = self.w.reshape(1, -1) / np.sum(self.w)

        # remove mode (at least this is done in Matlab)

        if self.optimize_mu:
            theta = self.mu.flatten(order="F")
        else:
            theta = np.array(list())

        constrained_parameters = np.array(list())

        if self.optimize_sigma:
            constrained_parameters = np.concatenate(
                (constrained_parameters, self.sigma.flatten())
            )

        if self.optimize_lambd:
            constrained_parameters = np.concatenate(
                (constrained_parameters, self.lambd.flatten())
            )

        if self.optimize_weights:
            constrained_parameters = np.concatenate(
                (constrained_parameters, self.w.flatten())
            )

        if rawflag:
            return np.concatenate((theta, np.log(constrained_parameters)))
        else:
            return np.concatenate((theta, constrained_parameters))

    def set_parameters(self, theta: np.ndarray, rawflag=True):
        """
        Set variational posterior parameters from a single array.

        Takes as input a ``numpy`` array and assigns it to the
        variational posterior parameters.

        Parameters
        ----------
        theta : np.ndarray
            The array with the parameters that should be assigned.
        rawflag : bool, optional
            Specifies whether the sigma and lambda parameters are
            passed as raw (unconstrained) or not, by default ``True``.

        Raises
        ------
        ValueError
            Raised if sigma, lambda and weights are not positive
            and rawflag = ``False``.
        """

        # Make sure we don't get issues with references.
        theta = theta.copy()

        # check if sigma, lambda and weights are positive when rawflag = False
        if not rawflag:
            check_idx = 0
            if self.optimize_weights:
                check_idx -= self.K
            if self.optimize_lambd:
                check_idx -= self.D
            if self.optimize_sigma:
                check_idx -= self.K
            if np.any(theta[-check_idx:] < 0.0):
                raise ValueError(
                    """sigma, lambda and weights must be positive
                    when rawflag = False"""
                )

        if self.optimize_mu:
            self.mu = np.reshape(
                theta[: self.D * self.K], (self.D, self.K), order="F"
            )
            start_idx = self.D * self.K
        else:
            start_idx = 0

        if self.optimize_sigma:
            if rawflag:
                self.sigma = np.exp(theta[start_idx : start_idx + self.K])
            else:
                self.sigma = theta[start_idx : start_idx + self.K]
            start_idx += self.K

        if self.optimize_lambd:
            if rawflag:
                self.lambd = np.exp(theta[start_idx : start_idx + self.D]).T
            else:
                self.lambd = theta[start_idx : start_idx + self.D].T

        if self.optimize_weights:
            eta = theta[-self.K :]
            if rawflag:
                eta = eta - np.amax(eta)
                self.w = np.exp(eta.T)[:, np.newaxis]
            else:
                self.w = eta.T[:, np.newaxis]

        nl = np.sqrt(np.sum(self.lambd ** 2) / self.D)

        self.lambd = self.lambd.reshape(-1, 1) / nl
        self.sigma = self.sigma.reshape(1, -1) * nl

        # Ensure that weights are normalized
        if self.optimize_weights:
            self.w = self.w.reshape(1, -1) / np.sum(self.w)

        # remove mode
        self._mode = None

    def moments(self, N: int = int(1e6), origflag=True, covflag=False):
        """
        Compute mean and covariance matrix of variational posterior.

        In the original space, the moments are estimated via Monte Carlo
        sampling. If requested in the transformed (unconstrained) space used
        internally by VBMC, the moments can be computed analytically.

        Parameters
        ----------
        N : int, optional
            Number of samples used to estimate the moments, by default ``int(1e6)``.
        origflag : bool, optional
            If ``True``, compute moments in the original parameter space, 
            otherwise in the transformed VBMC space. By default ``True``.
        covflag : bool, optional
            If ``True``, return the covariance matrix as a second return value,
            by default ``False``.

        Returns
        -------
        mean: np.ndarray
            The mean of the variational posterior.
        cov: np.ndarray
            If `covflag` is ``True``, returns the covariance matrix as well.
        """
        if origflag:
            x, _ = self.sample(int(N), origflag=True, balanceflag=True)
            mubar = np.mean(x, axis=0)
            if covflag:
                sigma = np.cov(x.T)
        else:
            mubar = np.sum(self.w * self.mu, axis=1)

            if covflag:
                sigma = (
                    np.sum(self.w * self.sigma ** 2)
                    * np.eye(len(self.lambd))
                    * self.lambd
                )
                for k in range(self.K):
                    sigma += self.w[:, k] * (
                        (self.mu[:, k] - mubar)[:, np.newaxis]
                    ).dot((self.mu[:, k] - mubar)[:, np.newaxis].T)
        if covflag:
            return mubar.reshape(1, -1), sigma
        else:
            return mubar.reshape(1, -1)

    def whiten(
            self, vbmc, vp_old
    ):
        """
        Calculate whitening and update self.parameter_transformer accordingly

        Parameters
        ----------
        vbmc: VBMC
            The VBMC object from which the whitening routine was called.
        vp_old: VariationalPosterior
            A (deep) copy of the current variational posterior.

        Returns
        -------
        None.
        """

        # vp_old = copy.deepcopy(vp_old)

        old_transformer = self.parameter_transformer
        old_lb = self.parameter_transformer.lb_orig
        old_ub = self.parameter_transformer.ub_orig
        oldMu = self.parameter_transformer.inverse(self.mu.T)

        # Calculate rescaling and rotation from moments:
        mean, cov = self.moments(origflag=True, covflag=True)
        R_mat = self.parameter_transformer.R_mat
        scale = self.parameter_transformer.scale
        delta = self.parameter_transformer.delta
        cov = R_mat @ np.diag(scale) @ cov @ np.diag(scale) @ R_mat.T
        cov = np.diag(delta) @ cov @ np.diag(delta)
        U, s, Vh = np.linalg.svd(cov)
        if np.linalg.det(U) < 0:
            U[:, 0] = -U[:, 0]
        scale = np.sqrt(s)
        # scale = np.ones(self.D)
        print(scale)
        print(U)

        # Update Plausible Bounds:
        Nrnd = 100000
        xx = np.random.rand(Nrnd, self.D) * \
            (vbmc.optim_state["pub_orig"]-vbmc.optim_state["plb_orig"])\
            + vbmc.optim_state["plb_orig"]
        self.parameter_transformer = ParameterTransformer(
            self.D,
            vbmc.lower_bounds,
            vbmc.upper_bounds,
            vbmc.plausible_lower_bounds,
            vbmc.plausible_upper_bounds,
            scale,
            U
        )
        self.parameter_transformer.mu = np.zeros(self.D)
        self.parameter_transformer.delta = np.ones(self.D)
        yy = self.parameter_transformer(xx)
        [plb, pub] = np.quantile(yy, [0.05, 0.95], axis=0)
        delta_temp = pub-plb
        plb = plb - delta_temp/9
        pub = pub + delta_temp/9
        plb = np.reshape(plb, (-1, len(plb)))
        pub = np.reshape(pub, (-1, len(pub)))
        self.parameter_transformer = ParameterTransformer(
            self.D,
            vbmc.lower_bounds,
            vbmc.upper_bounds,
            plb,
            pub,
            scale,
            U
        )
        self.parameter_transformer.mu = np.zeros(self.D)
        self.parameter_transformer.delta = np.ones(self.D)
        # FIXME: Method should return new optim_state, to be
        # explicitly updated in main routine?
        vbmc.plausible_lower_bounds = plb
        vbmc.plausible_upper_bounds = pub
        # vbmc.optim_state["plb"] = plb
        # vbmc.optim_state["pub"] = pub

        # Not needed, shared object:
        vbmc.parameter_transformer = self.parameter_transformer
        # self.parameter_transformer.R_mat = U
        # self.parameter_transformer.scale = scale

        # TODO: Add temperature scaling?
        T = 1
        # Adjust stored points after warping:
        X_flag = vbmc.function_logger.X_flag
        # x_orig = vbmc.optim_state["cache"]["x_orig"][X_flag, :]
        # y_orig = vbmc.optim_state["cache"]["y_orig"][X_flag]
        X_orig = vbmc.function_logger.X[X_flag, :]
        y_orig = vbmc.function_logger.y[X_flag].T
        X = self.parameter_transformer(X_orig)
        dy = self.parameter_transformer.log_abs_det_jacobian(X)
        y = y_orig + dy/T
        print(y.shape)
        vbmc.function_logger.X[X_flag, :] = X
        vbmc.function_logger.y[X_flag] = y.T

        # vbmc.x0 = self.parameter_transformer(x_orig)

        # Update search bounds:
        def warpfun(x):
            # Copy probably unneccesary:
            x = np.copy(x)
            return self.parameter_transformer(
                                    vp_old.parameter_transformer.inverse(x)
                                              )
        Nrnd = 1000
        xx = np.random.rand(Nrnd, self.D) * \
            (vbmc.optim_state["ub_search"] - vbmc.optim_state["lb_search"])\
            + vbmc.optim_state["lb_search"]
        yy = warpfun(xx)
        yyMin = np.min(yy, axis=0)
        yyMax = np.max(yy, axis=0)
        delta = yyMax - yyMin
        vbmc.optim_state["lb_search"] = np.reshape(yyMin - delta/Nrnd,
                                                   (-1, len(yyMin - delta/Nrnd)
                                                    ))
        vbmc.optim_state["ub_search"] = np.reshape(yyMax + delta/Nrnd,
                                                   (-1, len(yyMax + delta/Nrnd)
                                                    ))

        # If search cache is not empty, update it:
        if vbmc.optim_state["search_cache"]:
            vbmc.optim_state["search_cache"] = warpfun(
                vbmc.optim_state["search_cache"]
            )

        # Update other state fields:
        vbmc.optim_state["recompute_var_post"] = True
        vbmc.optim_state["skipactivesampling"] = True
        vbmc.optim_state["warping_count"] += 1
        vbmc.optim_state["last_warping"] = vbmc.optim_state["iter"]
        vbmc.optim_state["last_successful_warping"] = vbmc.optim_state["iter"]

        # Reset GP Hyperparameters:
        vbmc.optim_state["run_mean"] = []
        vbmc.optim_state["run_cov"] = []
        vbmc.optim_state["last_run_avg"] = np.nan

        # Warp VP components (warp_gpandvp_vbmc.m):
        Ncov = vp_old.gp.covariance.hyperparameter_count(self.D)
        Nnoise = vp_old.gp.noise.hyperparameter_count()
        Nmean = vp_old.gp.mean.hyperparameter_count(self.D)
        # MATLAB: if ~isempty(gp_old.outwarpfun); Noutwarp = gp_old.Noutwarp; else; Noutwarp = 0; end
        # (Not used, see gaussian_process.py)

        Ns_gp = len(vp_old.gp.posteriors)
        hyp_warped = np.zeros([Ncov + Nnoise + Nmean, Ns_gp])

        for s in range(Ns_gp):
            hyp = vp_old.gp.posteriors[s].hyp
            hyp_warped[:, s] = hyp.copy()

            # UpdateGP input length scales (Not needed for linear warping?)
            ell = np.exp(hyp[0:self.D]).T
            (__, ell_new, __) = unscent_warp(warpfun, vp_old.gp.X, ell)
            hyp_warped[0:self.D,s] = np.mean(np.log(ell_new), axis=0)

            # We assume relatively no change to GP output and noise scales
            if isinstance(vp_old.gp.mean, gpyreg.mean_functions.ConstantMean):
                # Warp constant mean
                m0 = hyp(Ncov+Nnoise+1);
                dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(vp_old.gp.X)
                dy = self.parameter_transformer.log_abs_det_jacobian(warpfun(vp_old.gp.X))
                m0w = m0 + (np.mean(dy, axis=0) - np.mean(dy_old, axis=0))/T

                hyp_warped[Ncov+Nnoise+1, s] = m0w

            elif isinstance(vp_old.gp.mean, gpyreg.mean_functions.NegativeQuadratic):
                # Warp quadratic mean
                m0 = hyp[Ncov + Nnoise + 1]
                xm = hyp[Ncov + Nnoise + 1 : Ncov + Nnoise + 1 + self.D]
                omega = np.exp(hyp[Ncov + Nnoise + 1 + self.D : Ncov + Nnoise + 1 + self.D + self.D])

                # Warp location and scale
                (xmw, omegaw, __) = unscent_warp(warpfun, xm, omega)
                [xmw, omegaw] = [xm, omega]

                # Warp maximum
                dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(xm)
                dy = self.parameter_transformer.log_abs_det_jacobian(xmw)
                m0w = m0 + (dy - dy_old)/T

                hyp_warped[Ncov + Nnoise + 1, s] = m0w
                hyp_warped[Ncov + Nnoise + 1 : Ncov + Nnoise + 1 + self.D, s] = xmw
                hyp_warped[Ncov + Nnoise + 1 + self.D : Ncov + Nnoise + 1 + self.D + self.D, s] = np.log(omegaw)
            else:
                raise ValueError("Unsupported GP mean function for input warping.")

        # Update GP:
        # TODO: Check for correctness
        self.gp.hyp = hyp_warped
        mu = vp_old.mu.T
        sigmalambda = (vp_old.lambd * vp_old.sigma).T

        (muw, sigmalambdaw, __) = unscent_warp(warpfun, mu, sigmalambda)

        self.mu = muw.T
        plt.scatter(*zip(*vp_old.mu.T))
        plt.scatter(*zip(*self.mu.T))
        plt.axis("equal")
        print(self.parameter_transformer.R_mat)
        print(vp_old.parameter_transformer.R_mat)
        lambdaw = np.sqrt(self.D*np.mean(
            sigmalambdaw**2 / (sigmalambdaw**2+2),
            axis=0))
        # lambdaw = np.reshape(lambdaw, (-1, len(lambdaw)))
        print(lambdaw.shape)
        print(self.lambd.shape)
        self.lambd[:, 0] = lambdaw

        sigmaw = np.exp(np.mean(np.log(sigmalambdaw / lambdaw), axis=1))
        self.sigma[0, :] = sigmaw

        # Probably unnecessary: should be shared with parent by reference:
        vbmc.function_logger.parameter_transformer = self.parameter_transformer
        # Approximate change in weight:
        dy_old = vp_old.parameter_transformer.log_abs_det_jacobian(mu)
        dy = self.parameter_transformer.log_abs_det_jacobian(mu)

        ww = vp_old.w * np.exp((dy - dy_old)/T)
        self.w = ww / np.sum(ww)

        # TODO: Implement logging of warp action, see warp_input_vbmc.m,
        # lines 171 to 178.

    def mode(
        self,
        nmax: int = 20,
        origflag=True,
    ):
        """
        Find the mode of the variational posterior.

        Parameters
        ----------
        nmax : int, optional
            Maximum number of optimization runs to find the mode.
            If `nmax` < `self.K`, the starting points for the optimization are 
            chosen as the centers of the components with the highest values of 
            the pdf at those points. By default `nmax` = 20.
        origflag : bool, optional
            If ``True`` find the mode of the variational posterior in the 
            original parameter space, otherwise in the transformed parameter 
            space. By default ``True``.

        Returns
        -------
        mode: np.ndarray
            The mode of the variational posterior.

        Notes
        -----
        The mode is not invariant to nonlinear reparameterizations of 
        the input space, so the mode in the original space and the mode in the
        transformed (unconstrained) space will generally be in different 
        locations (even after applying the appropriate transformations).
        """

        def nlnpdf(x0, origflag=origflag):
            if origflag:
                y = self.pdf(x0, origflag=True, logflag=True, gradflag=False)
                return -y
            else:
                y, dy = self.pdf(
                    x0, origflag=False, logflag=True, gradflag=True
                )
                return -y, -dy

        if origflag and self._mode is not None:
            return self._mode
        else:
            x0_mat = self.mu.T

            if nmax < self.K:
                # First, evaluate pdf at all modes
                y0_vec = -1 * self.pdf(x0_mat, origflag=True, logflag=True)
                # Start from first NMAX solutions
                y0_idx = np.argsort(y0_vec)[:-1]
                x0_mat = x0_mat[y0_idx]

            x_min = np.zeros((x0_mat.shape[0], self.D))
            ff = np.full((x0_mat.shape[0], 1), np.inf)

            for k in range(x0_mat.shape[1]):
                x0 = x0_mat[k]

                if origflag:
                    x0 = self.parameter_transformer.inverse(x0)

                if origflag:
                    bounds = np.asarray(
                        [
                            np.concatenate(
                                (
                                    self.parameter_transformer.lb_orig[:, k],
                                    self.parameter_transformer.ub_orig[:, k],
                                ),
                                axis=None,
                            )
                        ]
                        * x0.size
                    )
                    x0 = np.minimum(
                        self.parameter_transformer.ub_orig,
                        np.maximum(x0, self.parameter_transformer.lb_orig),
                    )
                    x_min[k], ff[k], _ = fmin_l_bfgs_b(
                        func=nlnpdf, x0=x0, bounds=bounds, approx_grad=True
                    )
                else:
                    x_min[k], ff[k], _ = fmin_l_bfgs_b(func=nlnpdf, x0=x0)

            # Get mode and store it
            idx_min = np.argmin(ff)
            x = x_min[idx_min]

            if origflag:
                self._mode = x

            return x

    def mtv(
        self,
        vp2: VariationalPosterior = None,
        samples: np.ndarray = None,
        N: int = int(1e5),
    ):
        """
        Marginal total variation distances between two variational posteriors.

        Compute the total variation distance between the variational 
        posterior and a second posterior, separately for each dimension (hence
        "marginal" total variation distance, MTV). The second posterior can be
        specified either as a ``VariationalPosterior`` or as a set of samples.

        Parameters
        ----------
        vp2 : VariationalPosterior, optional
            The other ``VariationalPosterior``, by default ``None``.
        samples : np.ndarray, optional
            An `N`-by-`D` matrix of samples from the other variational
            posterior, by default ``None``.
        N : int, optional
            The number of random draws to estimate the MTV, by default ``int(1e5)``.

        Returns
        -------
        mtv: np.ndarray
            A `D`-element vector whose elements are the total variation distance
            between the marginal distributions of `vp` and `vp1` or `samples`,
            for each coordinate dimension.

        Raises
        ------
        ValueError
            Raised if neither `vp2` nor `samples` are specified.

        Notes
        -----
        The total variation distance between two densities `p1` and `p2` is:

        .. math:: TV(p1, p2) = \\frac{1}{2} \int | p1(x) - p2(x) | dx.

        """
        if vp2 is None and samples is None:
            raise ValueError("Either vp2 or samples have to be not None")

        xx1, _ = self.sample(N, True, True)
        if vp2 is not None:
            xx2, _ = vp2.sample(N, True, True)
            lb2 = vp2.parameter_transformer.lb_orig
            ub2 = vp2.parameter_transformer.ub_orig
        else:
            xx2 = samples
            lb2 = np.full((1, xx2.shape[1]), -np.inf)
            ub2 = np.full((1, xx2.shape[1]), np.inf)

        nkde = 2 ** 13
        mtv = np.zeros((1, self.D))
        # Set bounds for kernel density estimate
        lb1_xx = np.amin(xx1, axis=0)
        ub1_xx = np.amax(xx1, axis=0)
        range1 = ub1_xx - lb1_xx
        lb1 = np.maximum(
            lb1_xx - range1 / 10, self.parameter_transformer.lb_orig
        )
        ub1 = np.minimum(
            ub1_xx + range1 / 10, self.parameter_transformer.ub_orig
        )

        lb2_xx = np.amin(xx2, axis=0)
        ub2_xx = np.amax(xx2, axis=0)
        range2 = ub2_xx - lb2_xx
        lb2 = np.maximum(lb2_xx - range2 / 10, lb2)
        ub2 = np.minimum(ub2_xx + range2 / 10, ub2)

        # Compute marginal total variation
        for d in range(self.D):

            yy1, x1mesh, _ = kde1d(xx1[:, d], nkde, lb1[:, d], ub1[:, d])
            # Ensure normalization
            yy1 = yy1 / (trapezoid(yy1) * (x1mesh[1] - x1mesh[0]))

            yy2, x2mesh, _ = kde1d(xx2[:, d], nkde, lb2[:, d], ub2[:, d])
            # Ensure normalization
            yy2 = yy2 / (trapezoid(yy2) * (x2mesh[1] - x2mesh[0]))

            f = lambda x: np.abs(
                interp1d(
                    x1mesh,
                    yy1,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)
                - interp1d(
                    x2mesh,
                    yy2,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)
            )
            bb = np.sort(
                np.array([x1mesh[0], x1mesh[-1], x2mesh[0], x2mesh[-1]])
            )
            for j in range(3):
                xx_range = np.linspace(bb[j], bb[j + 1], num=int(1e5))
                mtv[:, d] = mtv[:, d] + 0.5 * trapezoid(f(xx_range)) * (
                    xx_range[1] - xx_range[0]
                )
        return mtv

    def kldiv(
        self,
        vp2: VariationalPosterior = None,
        samples: np.ndarray = None,
        N: int = int(1e5),
        gaussflag: bool = False,
    ):
        """
        Kullback-Leibler divergence between two variational posteriors.

        Compute the forward and reverse Kullback-Leibler (KL) divergence between
        two posteriors. The other variational posterior can be specified as 
        `vp2` (an instance of the class ``VariationalPosterior``) or with 
        `samples`. One of the two must be specified.

        Parameters
        ----------
        vp2 : VariationalPosterior, optional
            The other ``VariationalPosterior``, by default None.
        samples : np.ndarray, optional
            An `N`-by-`D` matrix of samples from the other variational
            posterior, by default ``None``.
        N : int, optional
            The number of random samples to estimate the KL divergence,
            by default ``int(1e5)``.
        gaussflag : bool, optional
            If ``True``, returns a "Gaussianized" KL-divergence, that is the KL 
            divergence between two multivariate normal distributions with the 
            same moments as the variational posteriors given as inputs. 
            By default ``False``.

        Returns
        -------
        kldiv: np.ndarray
            A two-element vector containing the forward and reverse 
            Kullback-Leibler divergence between the two posteriors.

        Raises
        ------
        ValueError
            Raised if neither `vp2` nor `samples` are specified.
        ValueError
            Raised if `vp2` is not provided but `gaussflag` = ``False``.

        Notes
        -----
        Since the KL divergence is not symmetric, the method returns both the
        forward and the reverse KL divergence, that is KL(`vp1` || `vp2`) and 
        KL(`vp2` || `vp1`).

        """

        if samples is None and vp2 is None:
            raise ValueError("Either vp2 or samples have to be not None")

        if not gaussflag and vp2 is None:
            raise ValueError(
                "Unless the KL divergence is gaussianized, VP2 is required."
            )

        if gaussflag:
            if N == 0:
                raise ValueError(
                    """Analytical moments are available
                    only for the transformed space."""
                )
            else:
                q1mu, q1sigma = self.moments(N, True, True)
                if vp2 is not None:
                    q2mu, q2sigma = vp2.moments(N, True, True)
                else:
                    q2mu = np.mean(samples)
                    q2sigma = np.cov(samples.T)

            kls = kldiv_mvn(q1mu, q1sigma, q2mu, q2sigma)
        else:
            minp = sys.float_info.min

            xx1, _ = self.sample(N, True, True)
            q1 = self.pdf(xx1, True)
            q2 = vp2.pdf(xx1, True)
            q1[np.logical_or(q1 == 0, np.isinf(q1))] = 1
            q2[np.logical_or(q2 == 0, np.isinf(q2))] = minp
            kl1 = -np.mean(np.log(q2) - np.log(q1))

            xx2, _ = vp2.sample(N, True, True)
            q1 = self.pdf(xx2, True)
            q2 = vp2.pdf(xx2, True)
            q1[np.logical_or(q1 == 0, np.isinf(q1))] = minp
            q2[np.logical_or(q2 == 0, np.isinf(q2))] = 1
            kl2 = -np.mean(np.log(q1) - np.log(q2))
            kls = np.concatenate((kl1, kl2), axis=None)

        # Correct for numerical errors
        kls[kls < 0] = 0
        return kls

    def plot(
        self,
        n_samples: int = int(1e5),
        title: str = None,
        plot_data: bool = False,
        highlight_data: list = None,
        plot_vp_centres: bool = False,
        plot_style: dict = None,
    ):
        """
        Plot the variational posterior.

        `plot` displays the variational posterior as a cornerplot showing the 
        1D and 2D marginals, estimated from samples. It uses the  `corner
        <https://corner.readthedocs.io/en/latest/index.html>`_ package.
                    
        
        `plot` also optionally displays the centres of the variational mixture
        components and the datapoints of the underlying Gaussian process (GP) 
        used by ``VBMC``. The plot can be enhanced by custom styles and specific 
        datapoints of the GP can be highlighted.

        Parameters
        ----------
        n_samples : int, optional
            The number of posterior samples used to create the plot, by default 
            ``int(1e5)``.
        title : str, optional
            The title of the plot, by default ``None``.
        plot_data : bool, optional
            Whether to plot the datapoints of the GP, by default ``False``.
        highlight_data : list, optional
            Indices of the GP datapoints that should be plotted in a different
            way than the other datapoints, by default ``None``.
        plot_vp_centres : bool, optional
            Whether to plot the centres of the `vp` components, by default ``False``.
        plot_style : dict, optional
            A dictionary of plot styling options. The possible options are:
                **corner** : dict, optional
                    Styling options directly passed to the corner function.
                    By default: ``{"fig": plt.figure(figsize=(8, 8)),
                    "labels": labels}``. See the documentation of `corner
                    <https://corner.readthedocs.io/en/latest/index.html>`_.
                **data** : dict, optional
                    Styling options used to plot the GP data.
                    By default: ``{"s":15, "color":'blue', "facecolors": "none"}``.
                **highlight_data** : dict, optional
                    Styling options used to plot the highlighted GP data.
                    By default: ``{"s":15, "color":"orange"}``.
                **vp_centre** : dict, optional
                    Styling options used to plot the `vp` centres.
                    By default: ``{"marker":"x", "color":"red"}``.


        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting ``matplotlib`` figure of the plot.

        """
        # generate samples
        Xs, _ = self.sample(n_samples)

        # cornerplot with samples of vp
        fig = plt.figure(figsize=(6, 6))
        labels = ["$x_{}$".format(i) for i in range(self.D)]
        corner_style = dict({"fig": fig, "labels": labels})

        if plot_style is None:
            plot_style = dict()

        if "corner" in plot_style:
            corner_style.update(plot_style.get("corner"))

        # suppress warnings for small datasets with quiet=True
        fig = corner.corner(Xs, quiet=True, **corner_style)

        # style of the gp data
        data_style = dict({"s": 15, "color": "blue", "facecolors": "none"})

        if "data" in plot_style:
            data_style.update(plot_style.get("data"))

        highlighted_data_style = dict(
            {
                "s": 15,
                "color": "orange",
            }
        )

        if "highlight_data" in plot_style:
            highlighted_data_style.update(plot_style.get("highlight_data"))

        axes = np.array(fig.axes).reshape((self.D, self.D))

        # plot gp data
        if plot_data and hasattr(self, "gp"):

            # highlight nothing when argument is None
            if highlight_data is None or highlight_data.size == 0:
                highlight_data = np.array([False] * len(self.gp.X))
                normal_data = ~highlight_data
            else:
                normal_data = [
                    i for i in range(len(self.gp.X)) if i not in highlight_data
                ]

            orig_X_norm = self.parameter_transformer.inverse(
                self.gp.X[normal_data]
            )
            orig_X_highlight = self.parameter_transformer.inverse(
                self.gp.X[highlight_data]
            )

            for r in range(1, self.D):
                for c in range(self.D - 1):
                    if r > c:
                        axes[r, c].scatter(
                            orig_X_norm[:, c], orig_X_norm[:, r], **data_style
                        )
                        axes[r, c].scatter(
                            orig_X_highlight[:, c],
                            orig_X_highlight[:, r],
                            **highlighted_data_style
                        )

        # style of the vp centres
        vp_centre_style = dict(
            {
                "marker": "x",
                "color": "red",
            }
        )

        if "vp_centre" in plot_style:
            vp_centre_style.update(plot_style["vp_centre"])

        # plot centres of vp components
        if plot_vp_centres:
            for r in range(1, self.D):
                for c in range(self.D - 1):
                    if r > c:
                        for component in self.parameter_transformer.inverse(
                            self.mu.T
                        ):
                            axes[r, c].plot(
                                component[c], component[r], **vp_centre_style
                            )

        if title is not None:
            fig.suptitle(title)

        # adjust spacing between subplots
        fig.tight_layout(pad=0.5)

        return fig
