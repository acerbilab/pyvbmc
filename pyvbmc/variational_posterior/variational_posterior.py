# for annotating VP as input of itself in mtv
from __future__ import annotations

import sys

import numpy as np
from pyvbmc.decorators import handle_0D_1D_input
from pyvbmc.stats import kde1d
from pyvbmc.parameter_transformer import ParameterTransformer
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import gammaln

from pyvbmc.stats import kldiv_mvn


class VariationalPosterior:
    """
    The Variational Posterior class used in the context of VBMC.

    Parameters
    ----------
    D : int
        The number of dimensions of the Variational Posterior.
    K : int, optional
        The number of mixture components, default 2.
    x0 : np.ndarray, optional
        The starting vector for the mixture components means, it can be a
        single array or multiple rows (up to K); missing rows are
        duplicated by making copies of x0, default np.zeros.
    parameter_transformer : ParameterTransformer, optional
        The ParameterTransformer object specifying the transformation of the
        input space that leads to the current representation used by the
        variational posterior, by default uses an identity transform.
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

    def get_bounds(self, X: np.ndarray, options, K: int = None):
        """
        Compute soft bounds for variational posterior parameters.

        Parameters
        ==========
        X : ndarray, shape (N, D)
            Training inputs.
        options : Options
            Program options.
        K : int, optional
            The number of mixture components. By default we use the
            number provided at class instantiation.

        Returns
        =======
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
        Sample random samples from variational posterior-

        Parameters
        ----------
        N : int
            The number of samples to be sampled.
        origflag : bool, optional
            Controls if the the random vectors should be returned in the
            original parameter space if origflag=True (default),
            or in the transformed VBMC space
            if origflag=False, by default True.
        balanceflag : bool, optional
            The boolean balanceflag=True balances the generating process
            such that the random samples in X come from each
            mixture component exactly proportionally
            (or as close as possible) to the variational mixture weights.
            If balanceflag=False (default), the generating mixture
            for each sample is determined randomly according to
            the mixture weights, by default False.
        df : float, optional
            Generate the samples from a heavy-tailed version
            of the variational posterior, in which
            the multivariate normal components have been replaced by
            multivariate t-distributions with DF degrees of freedom.
            The default is df=Inf, limit in which the t-distribution
            becomes a multivariate normal, by default np.inf.

        Returns
        -------
        X : np.ndarray
            X is an N-by-D matrix of random vectors chosen
            from the variational posterior.
        I : np.ndarray
            I is an N-by-1 array such that the N-th
            element of I indicates the index of the
            variational mixture component from which
            the N-th row of X has been generated.
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
            lamd_row = self.lambd.reshape(1, -1)

            rng = np.random.default_rng()
            if self.K > 1:
                if balanceflag:
                    # exact split of samples according to mixture weigths
                    repeats = np.floor(self.w * N).astype("int")
                    i = np.repeat(range(self.K), repeats.flatten())

                    # compute remainder samples (with correct weights) if needed
                    if N > i.shape[0]:
                        w_extra = self.w * N - repeats
                        repeats_extra = np.ceil(np.sum(w_extra))
                        w_extra += self.w * (repeats_extra - sum(w_extra))
                        w_extra /= np.sum(w_extra)
                        i_extra = rng.choice(
                            range(self.K),
                            size=repeats_extra.astype("int"),
                            p=w_extra.flatten(),
                        )
                        i = np.append(i, i_extra)

                    rng.shuffle(i)
                    i = i[:N]
                else:
                    i = rng.choice(range(self.K), size=N, p=self.w.flatten())

                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.T[i]
                        + lamd_row
                        * np.random.randn(N, self.D)
                        * self.sigma[:, i].T
                    )
                else:
                    t = df / 2 / np.sqrt(rng.gamma(df / 2, df / 2, (N, 1)))
                    x = (
                        self.mu.T[i]
                        + lamd_row
                        * np.random.randn(N, self.D)
                        * t
                        * self.sigma[:, i].T
                    )
            else:
                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.T
                        + lamd_row * np.random.randn(N, self.D) * self.sigma
                    )
                else:
                    t = df / 2 / np.sqrt(rng.gamma(df / 2, df / 2, (N, 1)))
                    x = (
                        self.mu.T
                        + lamd_row
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
        Implements the probability density function of VP approximation.

        Parameters
        ----------
        x : np.ndarray
            X is a matrix of rows to evaluate the pdf at.
            The Rows of the N-by-D matrix x
            correspond to observations or points,
            and columns correspond to variables or coordinates.
        origflag : bool, optional
            Controls if the value of the posterior density should be evaluated
            in the original parameter space for origflag=True (default), or
            in the transformed space if origflag=False, by default True.
        logflag : bool, optional
            Controls if the the log pdf should be returned if logflag=True,
            this is by default False.
        transflag : bool, optional
            Specifies if X is already specified in transformed space.
            TRANSFLAG=True assumes that X is already specified in tranformed
            space. Otherwise, X is specified in the original parameter
            space, by default False.
        gradflag : bool, optional
            If gradflag=True the gradient should be returned, by default False.
        df : float, optional
            Compute the pdf of a heavy-tailed version of the
            variational posterior, in which the multivariate normal components
            have been replaced by  multivariate t-distributions with
            DF degrees of freedom.The default is df=Inf, limit in which the
            t-distribution becomes a multivariate normal., by default np.inf

        Returns
        -------
        pdf: np.ndarray
            The probability density of the variational posterior
            evaluated at each row of x.
        gradient: np.ndarray
            If the gradflag=True, the function returns
            the gradient as well.

        Raises
        ------
        NotImplementedError
            Raised if np.isfinite(df) and df > 0 and gradflag=True
            (Gradient of heavy-tailed pdf not supported yet).
        NotImplementedError
            Raised if np.isfinite(df) and df < 0 and gradflag=True
            (Gradient of heavy-tailed pdf not supported yet).
        NotImplementedError
            Raised if oriflag=True and logflag=True and gradflag=True
            (Gradient computation in original space not supported yet).
        """
        # Convert points to transformed space
        if origflag and not transflag:
            x = self.parameter_transformer(x)
        lamd_row = self.lambd.reshape(1, -1)
        N, D = x.shape
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
                y -= self.parameter_transformer.log_abs_det_jacobian(x)[
                    :, np.newaxis
                ]
                if gradflag:
                    raise NotImplementedError(
                        """vbmc_pdf:NoOriginalGrad: Gradient computation
                         in original space not supported yet."""
                    )
            else:
                y /= np.exp(
                    self.parameter_transformer.log_abs_det_jacobian(x)[
                        :, np.newaxis
                    ]
                )

        if gradflag:
            return y, dy
        else:
            return y

    def get_parameters(self, rawflag=True):
        """
        Return all the active VariationalPosterior parameters
        flattened as a 1D (numpy) array and properly transformed.

        Parameters
        ----------
        rawflag : bool, optional
            Specifies whether the sigma and lambda are
            returned as raw (unconstrained) or not, by default True.

        Returns
        -------
        theta : np.ndarray
            The VP parameters flattenend as an 1D array.
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
        Takes as input an np array and assigns it to the
        variational posterior parameters.

        Parameters
        ----------
        theta : np.ndarray
            The array with the parameters that should be assigned.
        rawflag : bool, optional
            Specifies whether the sigma and lambda are
            passed as raw (unconstrained) or not, by default True.

        Raises
        ------
        ValueError
            Raised if sigma, lambda and weights are not positive
            and rawflag = False
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
        if hasattr(self, "_mode"):
            delattr(self, "_mode")

    def moments(self, N: int = int(1e6), origflag=True, covflag=False):
        """
        Compute the mean MU and covariance matrix SIGMA
        of the variational posterior via Monte Carlo sampling.

        Parameters
        ----------
        N : int, optional
            The number of samples to compute moments from, by default int(1e6).
        origflag : bool, optional
            if origflag=True (default) sample in the original parameter space or
            if origflag=False sample in the transformed VBMC space,
            by default True.
        covflag : bool, optional
            If covflag=True returns covariance as second return value,
            by default False.

        Returns
        -------
        mean: np.ndarray
            The mean of the variational posterior.
        cov: np.ndarray
            If covflag=True returns covariance as second return value.
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

    def mode(self, nmax: int = 20, origflag=True):
        """
        Find the mode of Variational Posterior.

        Parameters
        ----------
        nmax : int, optional
            Maximum number of optimization runs to find the mode.
            If nmax < self.K the starting points for the optimization are chosen
            as the centers of the components with the highest values of the pdf
            at those points, by default 20.
        origflag : bool, optional
            if origflag=True (default) find the  mode of the
            variational posterior in the original parameter space or if
            origflag=False in the transformed parameter space, by default True.

        Returns
        -------
        mode: np.ndarray
            The mode of the variational posterior.
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

        if origflag and hasattr(self, "_mode") and self._mode is not None:
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
        Returns the marginal total variation (MTV) distances between the VP and
        another variational posterior.

        The other variational posterior can be specified as `vp2` (an instance
        of the class VariationalPosterior) or using with `samples` from the
        other VariationalPosterior. One of the two must be specified.

        Parameters
        ----------
        vp2 : VariationalPosterior, optional
            The other VariationalPosterior, by default None.
        samples : np.ndarray, optional
            An N-by-D matrix of samples from the other variational
            posterior, by default None.
        N : int, optional
            The number of random draws to estimate the MTV, by default int(1e5).

        Returns
        -------
        mtv: np.ndarray
            A D-element vector whose elements are the total variation distance
            between the marginal distributions of VP and VP2 or samples,
            for each coordinate dimension.

        Raises
        ------
        ValueError
            Raised if neither vp2 nor samples are specified.
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
        Compute the Kullback-Leibler divergence between two variational
        posteriors.

        The other variational posterior can be specified as `vp2` (an instance
        of the class VariationalPosterior) or using with `samples` from the
        other VariationalPosterior. One of the two must be specified.

        Parameters
        ----------
        vp2 : VariationalPosterior, optional
            The other VariationalPosterior, by default None.
        samples : np.ndarray, optional
            An N-by-D matrix of samples from the other variational
            posterior, by default None.
        N : int, optional
            The number of random draws to estimate the kldiv,
            by default int(1e5).
        gaussflag : bool, optional
            Returns a "Gaussianized" KL-divergence if gaussflag=True,
            that is the KL divergence between two
            multivariate normal distibutions with the same moments
            as the variational posteriors given as inputs, by default False.

        Returns
        -------
        kldiv: np.ndarray
            The Kullback-Leibler divergence between the two VPs.

        Raises
        ------
        ValueError
            Raised if neither vp2 nor samples are specified.
        ValueError
            Raised if vp2 is not provided but gaussflag=true.
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
