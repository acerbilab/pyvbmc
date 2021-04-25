# for annotating VP as input of itself in mtv
from __future__ import annotations

import sys

import numpy as np
from decorators import handle_1D_input
from parameter_transformer import ParameterTransformer
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import gammaln
from scipy.stats import gaussian_kde


class VariationalPosterior(object):
    """
    Variational Posterior class
    """

    def __init__(self):
        self.d = None  # number of dimensions
        self.k: int = None  # number of components
        self.w = None
        self.mu = None
        self.sigma = None
        self.lamb = None
        self.optimize_mu = None
        self.optimize_sigma = None
        self.optimize_lamb = None
        self.optimize_weights = None
        self.bounds = None
        self.parameter_transformer = ParameterTransformer(3)

    def sample(
        self,
        n: int,
        origflag: bool = True,
        balanceflag: bool = False,
        df: float = np.inf,
    ):
        """
        sample random samples from variational posterior

        Parameters
        ----------
        N : int
            number of samples
        origflag : bool, optional
            returns the random vectors in the original
            parameter space if origflag=True (default),
            or in the transformed VBMC space
            if origflag=False, by default True
        balanceflag : bool, optional
            balanceflag=True balances the generating process
            such that the random samples in X come from each
            mixture component exactly proportionally
            (or as close as possible) to the variational mixture weights.
            If balanceflag=False (default), the generating mixture
            for each sample is determined randomly according to
            the mixture weights, by default False
        df : float, optional
            samples generated from a heavy-tailed version
            of the variational posterior, in which
            the multivariate normal components have been replaced by
            multivariate t-distributions with DF degrees of freedom.
            The default is df=Inf, limit in which the t-distribution
            becomes a multivariate normal., by default np.inf

        Returns
        -------
        X : np.ndarray
            N-by-D matrix X of random vectors chosen
            from the variational posterior
        I : np.ndarray
            N-by-1 array such that the n-th
            element of I indicates the index of the
            variational mixture component from which
            the n-th row of X has been generated.
        """
        # missing to sample from gp
        gp_sample = False
        if n < 1:
            x = np.zeros((0, self.d))
            i = np.zeros((0, 1))
            return x, i
        elif gp_sample:
            pass
        else:
            rng = np.random.default_rng()
            if self.k > 1:
                if balanceflag:
                    # exact split of samples according to mixture weigths
                    repeats = np.floor(self.w * n).astype("int")
                    i = np.repeat(range(self.k), repeats.flatten())

                    # compute remainder samples (with correct weights) if needed
                    if n > i.shape[0]:
                        w_extra = self.w * n - repeats
                        repeats_extra = np.ceil(np.sum(w_extra))
                        w_extra += self.w * (repeats_extra - sum(w_extra))
                        w_extra /= np.sum(w_extra)
                        i_extra = rng.choice(
                            range(self.k),
                            size=repeats_extra.astype("int"),
                            p=w_extra.flatten(),
                        )
                        i = np.append(i, i_extra)

                    rng.shuffle(i)
                    i = i[:n]
                else:
                    i = rng.choice(range(self.k), size=n, p=self.w.flatten())

                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.conj().T[i]
                        + self.lamb.conj().T
                        * np.random.randn(n, self.d)
                        * self.sigma.conj().T[i]
                    )
                else:
                    t = df / 2 / np.sqrt(rng.gamma(df / 2, df / 2, (n, 1)))
                    x = (
                        self.mu.conj().T[i]
                        + self.lamb.conj().T
                        * np.random.randn(n, self.d)
                        * t
                        * self.sigma.conj().T[i]
                    )
            else:
                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.conj().T
                        + self.lamb.conj().T
                        * np.random.randn(n, self.d)
                        * self.sigma.conj().T
                    )
                else:
                    t = df / 2 / np.sqrt(rng.gamma(df / 2, df / 2, (n, 1)))
                    x = (
                        self.mu.conj().T
                        + self.lamb.conj().T
                        * t
                        * np.random.randn(n, self.d)
                        * self.sigma.conj().T
                    )
                i = np.zeros(n)
            if origflag:
                x = self.parameter_transformer.inverse(x)
        return x, i

    @handle_1D_input(kwarg="x", argpos=0)
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
        pdf probability density function of VBMC posterior approximation

        Parameters
        ----------
        x : np.ndarray
            matrix of rows to evaluate the pdf at
            Rows of the N-by-D matrix x
            correspond to observations or points,
            and columns correspond to variables or coordinates.
        origflag : bool, optional
            returns the random vectors in the original
            parameter space if origflag=True (default),
            or in the transformed VBMC space
            if origflag=False, by default True
        logflag : bool, optional
            returns the value of the log pdf
            if LOGFLAG=True, otherwise
            the posterior density, by default False
        transflag : bool, optional
            transflag=True assumes
            that X is already specified in transformed VBMC space.
            Otherwise, X is specified
            in the original parameter space, by default False
        gradflag : bool, optional
            gradflag = True returns gradient as well, by default False
        df : float, optional
            pdf of a heavy-tailed version
            of the variational posterior, in which
            the multivariate normal components have been replaced by
            multivariate t-distributions with DF degrees of freedom.
            The default is df=Inf, limit in which the t-distribution
            becomes a multivariate normal., by default np.inf

        Returns
        -------
        pdf: np.ndarray
            probability density of the variational posterior
            evaluated at each row of x.
        gradient: np.ndarray
            if gradflag is True, the function returns
            the gradient as well

        Raises
        ------
        NotImplementedError
            np.isfinite(df) and df > 0 and gradflag=True 
            (Gradient of heavy-tailed pdf not supported yet)
        NotImplementedError
            np.isfinite(df) and df < 0 and gradflag=True 
            (Gradient of heavy-tailed pdf not supported yet)
        NotImplementedError
            oriflag=True and logflag=True and gradflag=True 
            (Gradient computation in original space not supported yet)
        """
        # Convert points to transformed space
        if origflag and not transflag:
            x = self.parameter_transformer(x)

        n, d = x.shape
        y = np.zeros((n, 1))
        if gradflag:
            dy = np.zeros((n, d))

        if not np.isfinite(df) or df == 0:
            # compute pdf of variational posterior

            # common normalization factor
            nf = 1 / (2 * np.pi) ** (d / 2) / np.prod(self.lamb.conj().T)
            for k in range(self.k):
                d2 = np.sum(
                    (
                        (x - self.mu.conj().T[k])
                        / (self.sigma[:, k].dot(self.lamb.conj().T))
                    )
                    ** 2,
                    axis=1,
                )
                nn = (
                    nf
                    * self.w[:, k]
                    / self.sigma[:, k] ** d
                    * np.exp(-0.5 * d2)[:, np.newaxis]
                )
                y += nn
                if gradflag:
                    dy -= (
                        nn
                        * (x - self.mu.conj().T[k])
                        / ((self.lamb.conj().T ** 2) * self.sigma[:, k] ** 2)
                    )

        else:
            # Compute pdf of heavy-tailed variant of variational posterior

            if df > 0:
                # (This uses a multivariate t-distribution which is not the same thing
                # as the product of D univariate t-distributions)

                # common normalization factor
                nf = (
                    np.exp(gammaln((df + d) / 2) - gammaln(df / 2))
                    / (df * np.pi) ** (d / 2)
                    / np.prod(self.lamb)
                )

                for k in range(self.k):
                    d2 = np.sum(
                        (
                            (x - self.mu.conj().T[k])
                            / (self.sigma[:, k].dot(self.lamb.conj().T))
                        )
                        ** 2,
                        axis=1,
                    )
                    nn = (
                        nf
                        * self.w[:, k]
                        / self.sigma[:, k] ** d
                        * (1 + d2 / df) ** (-(df + d) / 2)
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
                ) ** d / np.prod(self.lamb)

                for k in range(self.k):
                    d2 = (
                        (x - self.mu.conj().T[k])
                        / (self.sigma[:, k].dot(self.lamb.conj().T))
                    ) ** 2
                    nn = (
                        nf
                        * self.w[:, k]
                        / self.sigma[:, k] ** d
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
                        "vbmc_pdf:NoOriginalGrad: Gradient computation in original space not supported yet."
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
        get_parameters return all the active VariationalPosterior parameters
        flattened as a 1D (numpy) array and properly transformed

        Parameters
        ----------
        rawflag : bool, optional
            specifies whether the sigma and lambda are
            returned as raw (unconstrained) or not, by default True

        Returns
        -------
        theta : np.ndarray
            parameters flattenend as a 1D array
        """

        nl = np.sqrt(np.sum(self.lamb ** 2) / self.d)

        self.lamb = self.lamb / nl
        self.sigma = self.sigma.conj().T * nl

        # Ensure that weights are normalized
        if self.optimize_weights:
            self.w = self.w.conj().T / np.sum(self.w)

        # remove mode (at least this is done in Matlab)

        if self.optimize_mu:
            theta = self.mu.flatten()
        else:
            theta = np.array(list())

        constrained_parameters = np.array(list())

        if self.optimize_sigma:
            constrained_parameters = np.concatenate(
                (constrained_parameters, self.sigma.flatten())
            )

        if self.optimize_lamb:
            constrained_parameters = np.concatenate(
                (constrained_parameters, self.lamb.flatten())
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
        set_parameters takes as input an np array and assigns it to the
        variational posterior parameters

        Parameters
        ----------
        theta : np.ndarray
            array with the parameters
        rawflag : bool, optional
            specifies whether the sigma and lambda are
            passed as raw (unconstrained) or not, by default True

        Raises
        ------
        ValueError
            sigma, lambda and weights must be positive when rawflag = False
        """ 

        # check if sigma, lambda and weights are positive when rawflag = False
        if not rawflag:
            check_idx = 0
            if self.optimize_weights:
                check_idx -= self.k
            if self.optimize_lamb:
                check_idx -= self.d
            if self.optimize_sigma:
                check_idx -= self.k
            if np.any(theta[-check_idx:] < 0.0):
                raise ValueError(
                    "sigma, lambda and weights must be positive when rawflag = False"
                )

        if self.optimize_mu:
            self.mu = np.reshape(theta[: self.d * self.k], (self.d, self.k))
            start_idx = self.d * self.k
        else:
            start_idx = 0

        if self.optimize_sigma:
            if rawflag:
                self.sigma = np.exp(theta[start_idx : start_idx + self.k])
            else:
                self.sigma = theta[start_idx : start_idx + self.k]
            start_idx += self.k

        if self.optimize_lamb:
            if rawflag:
                self.lamb = np.exp(theta[start_idx : start_idx + self.d]).T
            else:
                self.lamb = theta[start_idx : start_idx + self.d].T

        if self.optimize_weights:
            eta = theta[-self.k :]
            if rawflag:
                eta = eta - np.amax(eta)
                self.w = np.exp(eta.T)[:, np.newaxis]
            else:
                self.w = eta.T[:, np.newaxis]

        nl = np.sqrt(np.sum(self.lamb ** 2) / self.d)

        self.lamb = self.lamb / nl
        self.sigma = self.sigma.conj().T * nl

        # Ensure that weights are normalized
        if self.optimize_weights:
            self.w = self.w.conj().T / np.sum(self.w)

        # remove mode
        if(hasattr(self, "_mode")):
            delattr(self, '_mode')

    def moments(self, n: int = int(1e6), origflag=True, covflag=False):
        """
        moments computes the mean MU and covariance matrix SIGMA
        of the variational posterior via Monte Carlo sampling.

        Parameters
        ----------
        n : int, optional
            number of samples to compute
            moments from, by default int(1e6)
        origflag : bool, optional
            samples in the original parameter space
            if origflag=True (default),
            or in the transformed VBMC space
            if origflag=False, by default True,
        covflag : bool, optional
            returns covariance as second return value
            if covflag = True, by default False

        Returns
        -------
        mean: np.ndarray
            mean of the variational posterior
        cov: np.ndarray
            if covflag is True, the function returns
            a tuple of (mean, covariance) of the
            variational posterior

        """
        if origflag:
            x, _ = self.sample(int(n), origflag=True, balanceflag=True)
            mubar = np.mean(x, axis=0)
            if covflag:
                sigma = np.cov(x.T)
        else:
            mubar = np.sum(self.w * self.mu, axis=1)

            if covflag:
                sigma = (
                    np.sum(self.w * self.sigma ** 2)
                    * np.eye(len(self.lamb))
                    * self.lamb
                )
                for k in range(self.k):
                    sigma += self.w[:, k] * (
                        (self.mu[:, k] - mubar)[:, np.newaxis]
                    ).dot((self.mu[:, k] - mubar)[:, np.newaxis].conj().T)
        if covflag:
            return mubar.conj().T, sigma
        else:
            return mubar.conj().T

    def mode(self, nmax: int = 20, origflag=True):
        """
        get_mode Find mode of VBMC posterior approximation

        Parameters
        ----------
        nmax : int, optional
            [description], by default 20
        origflag : bool, optional
            mode of the variational posterior
            in the original parameter space
            if origflag=True (default),
            or in the transformed VBMC space
            if origflag=False, by default True

        Returns
        -------
        mode: np.ndarray
            the mode of the variational posterior
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
            x0_mat = self.mu.conj().T

            if nmax < self.k:
                # First, evaluate pdf at all modes
                y0_vec = -1 * self.pdf(x0_mat, origflag=True, logflag=True)
                # Start from first NMAX solutions
                y0_idx = np.argsort(y0_vec)[:-1]
                x0_mat = x0_mat[y0_idx]

            x_min = np.zeros((x0_mat.shape[0], self.d))
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
        mtv Marginal Total Variation distances between two variational posteriors.

        Parameters
        ----------
        vp2 : VariationalPosterior, optional
            other VariationalPosterior, by default None
        samples : np.ndarray, optional
            N-by-D matrices of samples from variational
            posteriors, by default None
        N : int, optional
            number of random draws
            to estimate the MTV, by default int(1e5)

        Returns
        -------
        mtv: np.ndarray
            D-element vector whose elements are the total variation
            distance between the marginal distributions of VP and VP2 or samples,
            for each coordinate dimension.

        Raises
        ------
        ValueError
            If neither vp2 nor samples are specified
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
        mtv = np.zeros((1, self.d))
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

        def compute_density(data, n, min, max):
            # set up the grid over which the density estimate is computed
            xmesh = (np.linspace(0, max - min, num=n) + min).T
            kernel = gaussian_kde(dataset=data)
            return kernel(xmesh), xmesh[0]

        # Compute marginal total variation
        for d in range(self.d):

            yy1, x1mesh = compute_density(
                xx1[:, d], nkde, lb1[:, d], ub1[:, d]
            )
            # Ensure normalization
            yy1 = yy1 / (trapezoid(yy1) * (x1mesh[1] - x1mesh[0]))

            yy2, x2mesh = compute_density(
                xx2[:, d], nkde, lb2[:, d], ub2[:, d]
            )
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
        kldiv Kullback-Leibler divergence between two variational posteriors

        Parameters
        ----------
        vp2 : VariationalPosterior, optional
            other VariationalPosterior, by default None
        samples : np.ndarray, optional
            N-by-D matrices of samples from variational
            posteriors, by default None
        N : int, optional
            number of random draws
            to estimate the MTV, by default int(1e5)
        gaussflag : bool, optional
            returns "Gaussianized" KL-divergence if GAUSSFLAG=True,
            that is the KL divergence between two
            multivariate normal distibutions with the same moments
            as the variational posteriors given as inputs, by default False

        Returns
        -------
        kldiv: np.ndarray
             Kullback-Leibler divergence

        Raises
        ------
        ValueError
            If neither vp2 nor samples are specified
        ValueError
            if vp2 is not provided but gaussflag true
        """

        if samples is None and vp2 is None:
            raise ValueError("Either vp2 or samples have to be not None")

        if not gaussflag and vp2 is None:
            raise ValueError(
                "Unless the KL divergence is gaussianized, VP2 is required."
            )

        def mvnkl(mu1, sigma1, mu2, sigma2):
            # Kullback-Leibler divergence between two multivariate normal pdfs
            if np.ndim(sigma1) == 0:
                sigma1 = np.array([np.array([sigma1])])
            if np.ndim(sigma2) == 0:
                sigma2 = np.array([np.array([sigma2])])
            if np.ndim(mu1) == 1:
                mu1 = np.array([mu1])
            if np.ndim(mu2) == 1:
                mu2 = np.array([mu2])

            d = mu1.shape[1]
            dmu = (mu1 - mu2).T
            detq1 = np.linalg.det(sigma1)
            detq2 = np.linalg.det(sigma2)
            lndet = np.log(detq2 / detq1)
            a, _, _, _ = np.linalg.lstsq(sigma2, sigma1, rcond=None)
            b, _, _, _ = np.linalg.lstsq(sigma2, dmu, rcond=None)
            kl1 = 0.5 * (np.trace(a) + dmu.T @ b - d + lndet)
            a, _, _, _ = np.linalg.lstsq(sigma1, sigma2, rcond=None)
            b, _, _, _ = np.linalg.lstsq(sigma1, dmu, rcond=None)
            kl2 = 0.5 * (np.trace(a) + dmu.T @ b - d - lndet)
            return np.concatenate((kl1, kl2), axis=None)

        if gaussflag:
            if N == 0:
                pass
            else:
                q1mu, q1sigma = self.moments(N, True, True)
                if vp2 is not None:
                    q2mu, q2sigma = vp2.moments(N, True, True)
                else:
                    q2mu = np.mean(samples)
                    q2sigma = np.cov(samples.T)

            kls = mvnkl(q1mu, q1sigma, q2mu, q2sigma)
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
